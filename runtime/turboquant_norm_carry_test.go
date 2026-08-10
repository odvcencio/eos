package eosruntime

import (
	"context"
	"testing"

	"m31labs.dev/turboquant"
)

// axisNormCarryVector returns a dim-length vector with value on the given
// axis and 0 elsewhere. Two vectors built on the same axis with different
// values share an identical quantized direction (MSE indices and QJL sign
// bits), so only turboquant.IPQuantized.Norm can separate their inner-product
// scores. That makes the pair a precise probe for norm-carry regressions: if
// Norm is dropped, both vectors score identically instead of proportionally
// to value.
func axisNormCarryVector(dim, axis int, value float32) []float32 {
	vec := make([]float32, dim)
	vec[axis] = value
	return vec
}

// TestTurboQuantRetrievalDocConstructionPreservesNonUnitNorm round-trips a
// turboquant.IPQuantized from q.Quantize(vec) for a non-unit (2x scaled)
// vector through the exact struct construction used at the retrieval call
// site (runtime/retrieval_turboquant.go, "turboQuantRetrievalDoc{ID: ...,
// Vector: qx}") and asserts Norm survives unchanged.
func TestTurboQuantRetrievalDocConstructionPreservesNonUnitNorm(t *testing.T) {
	dim := 8
	q := turboquant.NewIPWithSeed(dim, 4, 1)
	vec := axisNormCarryVector(dim, 0, 2) // non-unit: 2x a unit axis vector.
	qx := q.Quantize(vec)
	if qx.Norm == 0 {
		t.Fatalf("Quantize(vec) reported Norm 0 for a non-unit, non-zero input")
	}

	doc := turboQuantRetrievalDoc{ID: "d1", Vector: qx}
	if doc.Vector.Norm != qx.Norm {
		t.Fatalf("turboQuantRetrievalDoc dropped Norm: got %v, want %v", doc.Vector.Norm, qx.Norm)
	}
}

// TestTurboQuantMultiVectorChildConstructionPreservesNonUnitNorm is the
// multivector analog of TestTurboQuantRetrievalDocConstructionPreservesNonUnitNorm,
// covering runtime/retrieval_multivector_turboquant.go's
// "turboQuantMultiVectorChild{ParentID: ..., ChildID: ..., Vector: qx}".
func TestTurboQuantMultiVectorChildConstructionPreservesNonUnitNorm(t *testing.T) {
	dim := 8
	q := turboquant.NewIPWithSeed(dim, 4, 1)
	vec := axisNormCarryVector(dim, 0, 2) // non-unit: 2x a unit axis vector.
	qx := q.Quantize(vec)
	if qx.Norm == 0 {
		t.Fatalf("Quantize(vec) reported Norm 0 for a non-unit, non-zero input")
	}

	child := turboQuantMultiVectorChild{ParentID: "p1", ChildID: "c1", Vector: qx}
	if child.Vector.Norm != qx.Norm {
		t.Fatalf("turboQuantMultiVectorChild dropped Norm: got %v, want %v", child.Vector.Norm, qx.Norm)
	}
}

// TestEvaluateTurboQuantRetrievalBitsRanksByCarriedNorm locks the live
// runtime/retrieval_turboquant.go evaluator (evaluateTurboQuantRetrievalBits,
// which builds qx via q.Quantize around line 421-423) against a future
// refactor that rebuilds turboquant.IPQuantized{...} literals and drops Norm.
// Both documents share one quantized direction (identical MSE/signs) and
// differ only by norm, so a correct ranking is possible only if Norm reaches
// scoring; a doc with Norm dropped to 0 would tie every other zero-Norm doc
// and the alphabetically-first ID would incorrectly win ties
// (retrievalScoreBetter breaks ties by ID), sinking NDCG@10 below 1.
func TestEvaluateTurboQuantRetrievalBitsRanksByCarriedNorm(t *testing.T) {
	dim := 8
	docs := []retrievalVectorRecord{
		{ID: "doc_a_small_norm", Vector: axisNormCarryVector(dim, 0, 0.01)},
		{ID: "doc_z_large_norm", Vector: axisNormCarryVector(dim, 0, 50)},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: axisNormCarryVector(dim, 0, 1)},
	}
	qrels := retrievalQrels{"q1": {"doc_z_large_norm": 1}}

	rows, err := evaluateTurboQuantRetrievalBits(context.Background(), dim, 8, 100, 0, 1, nil, "", docs, queries, qrels, RetrievalEvalQualityMetrics{}, 0, 0, "test", "")
	if err != nil {
		t.Fatalf("evaluateTurboQuantRetrievalBits: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].Quality.NDCGAt10 != 1 {
		t.Fatalf("ndcg@10 = %v, want 1: doc_z_large_norm's carried Norm must outrank doc_a_small_norm despite an identical quantized direction", rows[0].Quality.NDCGAt10)
	}
}

// TestEvaluateTurboQuantMultiVectorBitsRanksByCarriedNorm is the multivector
// analog of TestEvaluateTurboQuantRetrievalBitsRanksByCarriedNorm, locking
// runtime/retrieval_multivector_turboquant.go's evaluateTurboQuantMultiVectorBits
// (which builds qx via q.Quantize around line 558-560) the same way.
func TestEvaluateTurboQuantMultiVectorBitsRanksByCarriedNorm(t *testing.T) {
	dim := 8
	children := []retrievalChildVectorRecord{
		{ParentID: "parent_a_small_norm", ChildID: "c1", Vector: axisNormCarryVector(dim, 0, 0.01)},
		{ParentID: "parent_z_large_norm", ChildID: "c1", Vector: axisNormCarryVector(dim, 0, 50)},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: axisNormCarryVector(dim, 0, 1)},
	}
	qrels := retrievalQrels{"q1": {"parent_z_large_norm": 1}}

	metrics, err := evaluateTurboQuantMultiVectorBits(context.Background(), dim, dim, 8, 100, 0, 1,
		TurboQuantMultiVectorAggregationMax, 0, children, queries, qrels, RetrievalEvalQualityMetrics{},
		2, 1, 0, 0, 0, 0, nil, "test")
	if err != nil {
		t.Fatalf("evaluateTurboQuantMultiVectorBits: %v", err)
	}
	if metrics.Quality.NDCGAt10 != 1 {
		t.Fatalf("ndcg@10 = %v, want 1: parent_z_large_norm's carried Norm must outrank parent_a_small_norm despite an identical quantized direction", metrics.Quality.NDCGAt10)
	}
}
