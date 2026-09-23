package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"testing"

	"m31labs.dev/turboquant"
)

func TestTopTurboQuantRetrievalScoresPrunesQ2Q3AndQ5WithExactParity(t *testing.T) {
	for _, bitWidth := range []int{2, 3, 5} {
		t.Run(fmt.Sprintf("bits%d", bitWidth), func(t *testing.T) {
			q, prepared, docs := turboQuantPruningFixture(t, bitWidth, 2, 12)

			got, stats := topTurboQuantRetrievalScoresWithStats(q, prepared, docs, 2)
			want := topTurboQuantRetrievalScoresExhaustive(q, prepared, docs, 2)

			assertRetrievalScoresEqual(t, got, want)
			if !stats.PruningSupported || !stats.PruningUsed {
				t.Fatalf("stats pruning support/use = %v/%v, want true/true", stats.PruningSupported, stats.PruningUsed)
			}
			if stats.CandidatesPruned == 0 {
				t.Fatalf("candidates pruned = 0, want nonzero")
			}
			if stats.CandidateCount != int64(len(docs)) {
				t.Fatalf("candidate count = %d, want %d", stats.CandidateCount, len(docs))
			}
			if stats.CandidatesScored+stats.CandidatesPruned != stats.CandidateCount {
				t.Fatalf("scored + pruned = %d + %d, want %d", stats.CandidatesScored, stats.CandidatesPruned, stats.CandidateCount)
			}
		})
	}
}

func TestTopTurboQuantRetrievalScoresQ4UnsupportedScoresAllCandidates(t *testing.T) {
	q, prepared, docs := turboQuantPruningFixture(t, 4, 2, 12)

	got, stats := topTurboQuantRetrievalScoresWithStats(q, prepared, docs, 2)
	want := topTurboQuantRetrievalScoresExhaustive(q, prepared, docs, 2)

	assertRetrievalScoresEqual(t, got, want)
	if stats.PruningSupported || stats.PruningUsed {
		t.Fatalf("stats pruning support/use = %v/%v, want false/false", stats.PruningSupported, stats.PruningUsed)
	}
	if stats.CandidatesPruned != 0 {
		t.Fatalf("candidates pruned = %d, want 0", stats.CandidatesPruned)
	}
	if stats.CandidatesScored != int64(len(docs)) {
		t.Fatalf("candidates scored = %d, want %d", stats.CandidatesScored, len(docs))
	}
}

func TestTopTurboQuantRetrievalScoresDoesNotPruneBeforeHeapIsFull(t *testing.T) {
	q, prepared, docs := turboQuantPruningFixture(t, 3, 2, 8)

	got, stats := topTurboQuantRetrievalScoresWithStats(q, prepared, docs, len(docs)+10)
	want := topTurboQuantRetrievalScoresExhaustive(q, prepared, docs, len(docs)+10)

	assertRetrievalScoresEqual(t, got, want)
	if !stats.PruningSupported {
		t.Fatalf("pruning supported = false, want true")
	}
	if stats.PruningUsed {
		t.Fatalf("pruning used = true, want false while heap never overflows")
	}
	if stats.CandidatesPruned != 0 || stats.CandidatesScored != int64(len(docs)) {
		t.Fatalf("stats scored/pruned = %d/%d, want %d/0", stats.CandidatesScored, stats.CandidatesPruned, len(docs))
	}
}

func TestTopTurboQuantRetrievalScoresScoresEqualityBoundsForTieSafety(t *testing.T) {
	const dim = 8
	q := turboquant.NewIPWithSeed(dim, 3, 123)
	query := normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	prepared := q.PrepareQuery(query)
	zero := make([]float32, dim)
	docs := []turboQuantRetrievalDoc{
		{ID: "m", Vector: q.Quantize(zero)},
		{ID: "a", Vector: q.Quantize(zero)},
	}

	got, stats := topTurboQuantRetrievalScoresWithStats(q, prepared, docs, 1)
	want := topTurboQuantRetrievalScoresExhaustive(q, prepared, docs, 1)

	assertRetrievalScoresEqual(t, got, want)
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("top doc = %+v, want tie winner a", got)
	}
	if stats.CandidatesPruned != 0 || stats.CandidatesScored != int64(len(docs)) {
		t.Fatalf("stats scored/pruned = %d/%d, want %d/0", stats.CandidatesScored, stats.CandidatesPruned, len(docs))
	}
}

func TestTopTurboQuantRetrievalScoresRandomParityAgainstExhaustive(t *testing.T) {
	// Exercise the actual prepared-IP path across dimensions, scales, widths,
	// and heap cutoffs.  ScoreUpperBound is allowed to prune only when its
	// float32 bound is still above the exact prepared score; any under-bound
	// here would change the returned ranking relative to the exhaustive path.
	rng := rand.New(rand.NewSource(0xA0C7))
	for _, dim := range []int{8, 32, 128, 384} {
		for _, bitWidth := range []int{2, 3, 5} {
			q := turboquant.NewIPWithSeed(dim, bitWidth, int64(9000+dim+bitWidth))
			docs := make([]turboQuantRetrievalDoc, 256)
			for i := range docs {
				vector := make([]float32, dim)
				scale := float32(0.001 + 12*rng.Float64())
				for j := range vector {
					vector[j] = float32(rng.NormFloat64()) * scale
				}
				docs[i] = turboQuantRetrievalDoc{ID: fmt.Sprintf("d-%d-%d", dim, i), Vector: q.Quantize(vector)}
			}
			for queryIndex := 0; queryIndex < 32; queryIndex++ {
				query := make([]float32, dim)
				for i := range query {
					query[i] = float32(rng.NormFloat64())
				}
				prepared := q.PrepareQuery(query)
				for _, topK := range []int{1, 7, 100, 255} {
					got, _ := topTurboQuantRetrievalScoresWithStats(q, prepared, docs, topK)
					want := topTurboQuantRetrievalScoresExhaustive(q, prepared, docs, topK)
					assertRetrievalScoresEqual(t, got, want)
				}
			}
		}
	}
}

func TestEvaluateTurboQuantRetrievalBitsSerializesPruningCounters(t *testing.T) {
	const dim = 8
	docs := make([]retrievalVectorRecord, 0, 112)
	for i := 0; i < 102; i++ {
		docs = append(docs, retrievalVectorRecord{
			ID:     fmt.Sprintf("d%03d", i),
			Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0}),
		})
	}
	for i := 102; i < 112; i++ {
		docs = append(docs, retrievalVectorRecord{ID: fmt.Sprintf("d%03d", i), Vector: make([]float32, dim)})
	}
	queries := []retrievalVectorRecord{{
		ID:     "q1",
		Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0}),
	}}
	qrels := retrievalQrels{"q1": {"d000": 1}}

	rows, err := evaluateTurboQuantRetrievalBits(context.Background(), dim, 3, 1, 0, 123, []int{101}, TurboQuantRerankStorageDense, 0, docs, queries, qrels, RetrievalEvalQualityMetrics{}, int64(len(docs)*dim*4), int64(len(docs)), "tiny", "")
	if err != nil {
		t.Fatalf("evaluateTurboQuantRetrievalBits: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want direct and rerank rows", len(rows))
	}
	row := rows[0]
	if !row.PruningSupported || !row.PruningUsed {
		t.Fatalf("row pruning support/use = %v/%v, want true/true", row.PruningSupported, row.PruningUsed)
	}
	if row.CandidateCount != int64(len(docs)) || row.CandidatesPruned == 0 {
		t.Fatalf("row counters = candidates %d pruned %d, want %d and nonzero", row.CandidateCount, row.CandidatesPruned, len(docs))
	}
	if row.ScoresPerSecond <= 0 || row.CandidateDecisionsPerSecond <= row.ScoresPerSecond {
		t.Fatalf("direct score rates = scores/s %.2f candidate decisions/s %.2f, want pruned full-score rate below candidate decision rate", row.ScoresPerSecond, row.CandidateDecisionsPerSecond)
	}
	rerankRow := rows[1]
	if rerankRow.RerankScores != 101 {
		t.Fatalf("rerank_scores = %d, want 101", rerankRow.RerankScores)
	}
	if rerankRow.ScoresPerSecond <= rerankRow.CandidateDecisionsPerSecond {
		t.Fatalf("rerank score rates = scores/s %.2f candidate decisions/s %.2f, want scores/s to include compact full scores plus rerank scores", rerankRow.ScoresPerSecond, rerankRow.CandidateDecisionsPerSecond)
	}
	data, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal row: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal row: %v", err)
	}
	for _, key := range []string{"candidate_count", "candidates_scored", "candidates_pruned", "pruning_supported", "pruning_used"} {
		if _, ok := decoded[key]; !ok {
			t.Fatalf("serialized row missing %q: %s", key, data)
		}
	}
}

func turboQuantPruningFixture(t *testing.T, bitWidth, positiveDocs, totalDocs int) (*turboquant.IPQuantizer, turboquant.PreparedQuery, []turboQuantRetrievalDoc) {
	t.Helper()
	const dim = 8
	if positiveDocs > totalDocs {
		t.Fatalf("positiveDocs %d > totalDocs %d", positiveDocs, totalDocs)
	}
	q := turboquant.NewIPWithSeed(dim, bitWidth, 123)
	query := normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})
	prepared := q.PrepareQuery(query)
	docs := make([]turboQuantRetrievalDoc, 0, totalDocs)
	for i := 0; i < positiveDocs; i++ {
		docs = append(docs, turboQuantRetrievalDoc{
			ID:     fmt.Sprintf("positive-%02d", i),
			Vector: q.Quantize(query),
		})
	}
	zero := make([]float32, dim)
	for i := positiveDocs; i < totalDocs; i++ {
		docs = append(docs, turboQuantRetrievalDoc{
			ID:     fmt.Sprintf("zero-%02d", i),
			Vector: q.Quantize(zero),
		})
	}
	return q, prepared, docs
}

func assertRetrievalScoresEqual(t *testing.T, got, want []retrievalScoredDoc) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("score count = %d, want %d\ngot=%+v\nwant=%+v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i].ID != want[i].ID || got[i].Score != want[i].Score {
			t.Fatalf("score %d = %+v, want %+v\ngot=%+v\nwant=%+v", i, got[i], want[i], got, want)
		}
	}
}
