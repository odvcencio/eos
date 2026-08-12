package eosruntime

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/turboquant"
)

func TestComputeRetrievalQualityPerfectRanking(t *testing.T) {
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1})},
	}
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0.2, 0.1})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 2 || relevantPairs != 2 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.NDCGAt10 != 1 || quality.MRRAt10 != 1 || quality.RecallAt10 != 1 || quality.RecallAt100 != 1 {
		t.Fatalf("quality = %+v, want perfect ranking", quality)
	}
	if quality.NDCGAt100 != 1 || quality.PrecisionAt1 != 1 || quality.PrecisionAt5 != 0.2 || quality.PrecisionAt10 != 0.1 {
		t.Fatalf("precision/ndcg@100 = %+v, want perfect top hit with fixed-k precision", quality)
	}
	if quality.HitAt1 != 1 || quality.HitAt5 != 1 || quality.HitAt10 != 1 || quality.MAPAt10 != 1 || quality.MAPAt100 != 1 {
		t.Fatalf("hit/map quality = %+v, want perfect ranking", quality)
	}
}

func TestComputeHybridRetrievalQualityMinmaxAlphaWeightsBM25(t *testing.T) {
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0})},
	}
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{0, 1})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{1, 0})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0.5, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
	}
	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "alpha exact target"},
		{ID: "d2", Text: "beta dense distractor"},
		{ID: "d3", Text: "gamma fallback"},
	}
	index, err := buildBM25Index(context.Background(), corpus)
	if err != nil {
		t.Fatalf("build BM25 index: %v", err)
	}
	denseQuality, _, _, _, _ := computeRetrievalQuality(queries, docs, qrels, 100)

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries, err := computeHybridRetrievalQuality(
		context.Background(),
		queries,
		docs,
		map[string][]string{"q1": tokenizeBM25Text("alpha")},
		index,
		qrels,
		100,
		"tiny",
		"",
		RetrievalEvalHybridConfig{Method: "minmax_blend", Alpha: 0.75, RRFK: 60, RRFLambda: 1},
	)
	if err != nil {
		t.Fatalf("compute hybrid quality: %v", err)
	}
	if queriesCount != 1 || relevantPairs != 1 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if denseQuality.NDCGAt10 >= 1 {
		t.Fatalf("dense quality = %+v, want imperfect dense baseline", denseQuality)
	}
	if quality.NDCGAt10 != 1 || quality.MRRAt10 != 1 {
		t.Fatalf("hybrid quality = %+v, want BM25-weighted top hit", quality)
	}
}

func TestEvaluateSparseLexicalHashHeadVectorHybridRecoversDenseMiss(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"beta dense distractor"}`+"\n"+
			`{"_id":"d3","text":"gamma fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1]}`+"\n"+
			`{"_id":"d2","embedding":[1,0]}`+"\n"+
			`{"_id":"d3","embedding":[0.5,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"_id":"q1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d2","nonzeros":1,"terms":[{"term":"beta","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"test","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "head.json")
	if _, err := FitSparseLexicalHashHead(SparseLexicalHashHeadFitConfig{
		DatasetName: "tiny",
		Split:       "test",
		LabelsPath:  labelsPath,
		HeadPath:    headPath,
		HashBins:    65536,
	}); err != nil {
		t.Fatalf("fit hash head: %v", err)
	}
	denseMetrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
		TopK:            100,
	})
	if err != nil {
		t.Fatalf("evaluate dense vectors: %v", err)
	}
	if denseMetrics.Quality.NDCGAt10 >= 1 || denseMetrics.Quality.MRRAt10 >= 1 {
		t.Fatalf("dense metrics = %+v, want imperfect dense ranking", denseMetrics.Quality)
	}

	for _, tc := range []struct {
		name   string
		topK   int
		hybrid RetrievalEvalHybridConfig
	}{
		{name: "minmax", topK: 99, hybrid: RetrievalEvalHybridConfig{Method: "minmax_blend", Alpha: 0.75, AlphaSet: true}},
		{name: "rrf", topK: 100, hybrid: RetrievalEvalHybridConfig{Method: "rrf", RRFK: 60, RRFLambda: 10}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			perQueryPath := filepath.Join(dir, tc.name+".per-query.jsonl")
			metrics, err := EvaluateSparseLexicalHashHeadVectorHybrid(context.Background(), SparseLexicalHashHeadEvalConfig{
				DatasetName:       "tiny",
				Split:             "test",
				CorpusPath:        corpusPath,
				QueriesPath:       queriesPath,
				QrelsPath:         qrelsPath,
				LabelsPath:        labelsPath,
				HeadPath:          headPath,
				DocVectorPath:     docVectorsPath,
				QueryVectorPath:   queryVectorsPath,
				TopK:              tc.topK,
				PerQueryJSONLPath: perQueryPath,
				Hybrid:            tc.hybrid,
			})
			if err != nil {
				t.Fatalf("evaluate hybrid: %v", err)
			}
			if metrics.Backend != "sparse_lexical_hash_head_vectors_hybrid" || metrics.Inputs.LabelPath != labelsPath || metrics.Inputs.HeadPath != headPath {
				t.Fatalf("metrics identity/inputs = %+v", metrics)
			}
			if metrics.Config.Hybrid == nil || metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 {
				t.Fatalf("hybrid/sparse stats missing: hybrid=%+v sparse=%+v", metrics.Config.Hybrid, metrics.SparseLexical)
			}
			if metrics.Config.TopK != 100 {
				t.Fatalf("metrics top_k = %d, want normalized 100", metrics.Config.TopK)
			}
			if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
				t.Fatalf("hybrid quality = %+v, want recovered top hit", metrics.Quality)
			}
			data, err := os.ReadFile(perQueryPath)
			if err != nil {
				t.Fatalf("read per-query: %v", err)
			}
			var row RetrievalEvalPerQueryRow
			if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &row); err != nil {
				t.Fatalf("decode per-query row: %v", err)
			}
			if row.FirstRelevantRank != 1 || len(row.TopK) == 0 || row.TopK[0].DocID != "d1" {
				t.Fatalf("per-query row = %+v", row)
			}
			if row.TopK[0].DenseRank == nil || *row.TopK[0].DenseRank != 3 || row.TopK[0].BM25Rank == nil || *row.TopK[0].BM25Rank != 1 {
				t.Fatalf("component ranks = dense:%v sparse:%v, want 3/1", row.TopK[0].DenseRank, row.TopK[0].BM25Rank)
			}
		})
	}
}

func TestEvaluateSparseLexicalProjectionHeadVectorHybridPredictsSparseRecovery(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"dense distractor"}`+"\n"+
			`{"_id":"d3","text":"fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	fitQueryVectorsPath := filepath.Join(dir, "fit-query-vectors.jsonl")
	evalQueryVectorsPath := filepath.Join(dir, "eval-query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1,0]}`+"\n"+
			`{"_id":"d2","embedding":[1,0.1,0]}`+"\n"+
			`{"_id":"d3","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(fitQueryVectorsPath, []byte(`{"_id":"q1","embedding":[0,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write fit query vectors: %v", err)
	}
	if err := os.WriteFile(evalQueryVectorsPath, []byte(`{"_id":"q1","embedding":[1,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write eval query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "projection-head.json")
	head, err := FitSparseLexicalProjectionHead(SparseLexicalProjectionHeadFitConfig{
		DatasetName:       "tiny",
		Split:             "train",
		LabelsPath:        labelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   fitQueryVectorsPath,
		HeadPath:          headPath,
		HashBins:          65536,
		MaxPrototypes:     8,
		MaxPredictedTerms: 4,
	})
	if err != nil {
		t.Fatalf("fit projection head: %v", err)
	}
	if head.Schema != SparseLexicalProjectionHeadSchema || !head.Experimental || head.Split != "train" || head.Config.Dimension != 3 || head.Hashing.Bins != 65536 || len(head.Prototypes) != 1 {
		t.Fatalf("projection head = %+v", head)
	}
	if _, err := ReadSparseLexicalProjectionHead(headPath); err != nil {
		t.Fatalf("read projection head: %v", err)
	}

	denseMetrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: evalQueryVectorsPath,
		TopK:            100,
	})
	if err != nil {
		t.Fatalf("evaluate dense vectors: %v", err)
	}
	if denseMetrics.Quality.NDCGAt10 >= 1 || denseMetrics.Quality.MRRAt10 >= 1 {
		t.Fatalf("dense metrics = %+v, want imperfect dense ranking", denseMetrics.Quality)
	}

	perQueryPath := filepath.Join(dir, "projection.per-query.jsonl")
	metrics, err := EvaluateSparseLexicalProjectionHeadVectorHybrid(context.Background(), SparseLexicalProjectionHeadEvalConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		HeadPath:          headPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   evalQueryVectorsPath,
		TopK:              100,
		PerQueryJSONLPath: perQueryPath,
		Hybrid:            RetrievalEvalHybridConfig{Method: "minmax", Alpha: 0.75, AlphaSet: true},
	})
	if err != nil {
		t.Fatalf("evaluate projection hybrid: %v", err)
	}
	if metrics.Backend != "sparse_lexical_projection_head_vectors_hybrid" || metrics.Inputs.LabelPath != "" || metrics.Inputs.HeadPath != headPath {
		t.Fatalf("metrics identity/inputs = %+v", metrics)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 {
		t.Fatalf("hybrid/sparse stats missing: hybrid=%+v sparse=%+v", metrics.Config.Hybrid, metrics.SparseLexical)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("projection hybrid quality = %+v, want recovered top hit", metrics.Quality)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query: %v", err)
	}
	var row RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.FirstRelevantRank != 1 || len(row.TopK) == 0 || row.TopK[0].DocID != "d1" {
		t.Fatalf("per-query row = %+v", row)
	}
	if row.TopK[0].DenseRank == nil || *row.TopK[0].DenseRank != 2 || row.TopK[0].BM25Rank == nil || *row.TopK[0].BM25Rank != 1 {
		t.Fatalf("component ranks = dense:%v sparse:%v, want 2/1", row.TopK[0].DenseRank, row.TopK[0].BM25Rank)
	}
}

func TestEvaluateSparseLexicalLinearHeadVectorHybridPredictsSparseRecoveryWithoutEvalLabels(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"dense distractor"}`+"\n"+
			`{"_id":"d3","text":"fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	fitQueryVectorsPath := filepath.Join(dir, "fit-query-vectors.jsonl")
	evalQueryVectorsPath := filepath.Join(dir, "eval-query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1,0]}`+"\n"+
			`{"_id":"d2","embedding":[1,0.1,0]}`+"\n"+
			`{"_id":"d3","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(fitQueryVectorsPath, []byte(`{"_id":"q1","embedding":[0,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write fit query vectors: %v", err)
	}
	if err := os.WriteFile(evalQueryVectorsPath, []byte(`{"_id":"q1","embedding":[1,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write eval query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "linear-head.json")
	head, err := FitSparseLexicalLinearHead(SparseLexicalLinearHeadFitConfig{
		DatasetName:       "tiny",
		Split:             "train",
		LabelsPath:        labelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   fitQueryVectorsPath,
		HeadPath:          headPath,
		HashBins:          65536,
		MaxBins:           8,
		MaxPredictedTerms: 4,
		Epochs:            12,
		LearningRate:      0.1,
		NegativeRatio:     1,
	})
	if err != nil {
		t.Fatalf("fit linear head: %v", err)
	}
	if head.Schema != SparseLexicalLinearHeadSchema || !head.Experimental || head.Split != "train" || head.Config.Dimension != 3 || head.Hashing.Bins != 65536 || len(head.Bins) != 1 {
		t.Fatalf("linear head = %+v", head)
	}
	if _, err := ReadSparseLexicalLinearHead(headPath); err != nil {
		t.Fatalf("read linear head: %v", err)
	}
	if err := os.Remove(labelsPath); err != nil {
		t.Fatalf("remove fit labels before eval: %v", err)
	}

	metrics, err := EvaluateSparseLexicalLinearHeadVectorHybrid(context.Background(), SparseLexicalLinearHeadEvalConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		HeadPath:          headPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   evalQueryVectorsPath,
		TopK:              100,
		PerQueryJSONLPath: filepath.Join(dir, "linear.per-query.jsonl"),
		Hybrid:            RetrievalEvalHybridConfig{Method: "minmax", Alpha: 0.75, AlphaSet: true},
	})
	if err != nil {
		t.Fatalf("evaluate linear hybrid: %v", err)
	}
	if metrics.Backend != "sparse_lexical_linear_head_vectors_hybrid" || metrics.Inputs.LabelPath != "" || metrics.Inputs.HeadPath != headPath {
		t.Fatalf("metrics identity/inputs = %+v", metrics)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || metrics.SparseLexical == nil || metrics.SparseLexical.Representation != "experimental_sparse_lexical_linear_head" || metrics.SparseLexical.HashBins != 65536 {
		t.Fatalf("hybrid/sparse stats missing: hybrid=%+v sparse=%+v", metrics.Config.Hybrid, metrics.SparseLexical)
	}
	if metrics.SparseLexical.DocumentMaxHashNNZ != 4 || metrics.SparseLexical.QueryMaxHashNNZ != 4 || metrics.SparseLexical.ScoreThreshold != 0 {
		t.Fatalf("default sparse calibration stats = %+v", metrics.SparseLexical)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("linear hybrid quality = %+v, want recovered top hit", metrics.Quality)
	}

	calibratedMetrics, err := EvaluateSparseLexicalLinearHeadVectorHybrid(context.Background(), SparseLexicalLinearHeadEvalConfig{
		DatasetName:     "tiny",
		Split:           "test",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		HeadPath:        headPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: evalQueryVectorsPath,
		TopK:            100,
		DocMaxTerms:     2,
		QueryMaxTerms:   1,
		ScoreThreshold:  0.000001,
		Hybrid:          RetrievalEvalHybridConfig{Method: "minmax", Alpha: 0.75, AlphaSet: true},
	})
	if err != nil {
		t.Fatalf("evaluate calibrated linear hybrid: %v", err)
	}
	if calibratedMetrics.SparseLexical == nil || calibratedMetrics.SparseLexical.DocumentMaxHashNNZ != 2 || calibratedMetrics.SparseLexical.QueryMaxHashNNZ != 1 || calibratedMetrics.SparseLexical.ScoreThreshold != 0.000001 {
		t.Fatalf("calibrated sparse stats = %+v", calibratedMetrics.SparseLexical)
	}

	badHead := head
	badHead.Config.Dimension = 2
	badData, err := json.Marshal(badHead)
	if err != nil {
		t.Fatalf("marshal bad head: %v", err)
	}
	badHeadPath := filepath.Join(dir, "bad-linear-head.json")
	if err := os.WriteFile(badHeadPath, badData, 0o644); err != nil {
		t.Fatalf("write bad head: %v", err)
	}
	_, err = EvaluateSparseLexicalLinearHeadVectorHybrid(context.Background(), SparseLexicalLinearHeadEvalConfig{
		DatasetName:     "tiny",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		HeadPath:        badHeadPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: evalQueryVectorsPath,
		TopK:            100,
		Hybrid:          RetrievalEvalHybridConfig{Method: "sparse_only"},
	})
	if err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("bad dimension err = %v", err)
	}
}

func TestFuseHybridScoresDefaultLeavesFusedOrder(t *testing.T) {
	denseScores := []retrievalScoredDoc{
		{ID: "dense-winner", Score: 1},
		{ID: "tail", Score: 0.5},
		{ID: "lexical-winner", Score: 0},
	}
	bm25Scores := []retrievalScoredDoc{
		{ID: "lexical-winner", Score: 10},
		{ID: "tail", Score: 5},
		{ID: "dense-winner", Score: 0},
	}

	got := fuseHybridScores(denseScores, bm25Scores, 100, RetrievalEvalHybridConfig{
		Method: "minmax_blend",
		Alpha:  0.75,
	})
	if len(got) < 3 {
		t.Fatalf("fused length = %d, want at least 3", len(got))
	}
	want := []string{"lexical-winner", "tail", "dense-winner"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("fused[%d] = %q, want %q; got=%+v", i, got[i].ID, id, got[:3])
		}
	}
	if got[0].DenseRank != 3 || got[0].BM25Rank != 1 {
		t.Fatalf("lexical winner component ranks = dense:%d bm25:%d, want 3/1", got[0].DenseRank, got[0].BM25Rank)
	}
	if got[0].DenseScore == nil || *got[0].DenseScore != 0 || got[0].BM25Score == nil || *got[0].BM25Score != 10 {
		t.Fatalf("lexical winner component raw scores = dense:%v bm25:%v, want 0/10", got[0].DenseScore, got[0].BM25Score)
	}
	if got[0].DenseNormalizedScore == nil || *got[0].DenseNormalizedScore != 0 || got[0].BM25NormalizedScore == nil || *got[0].BM25NormalizedScore != 1 {
		t.Fatalf("lexical winner component normalized scores = dense:%v bm25:%v, want 0/1", got[0].DenseNormalizedScore, got[0].BM25NormalizedScore)
	}
}

func TestFuseHybridScoresDenseProtectTopKPreservesDensePrefix(t *testing.T) {
	denseScores := []retrievalScoredDoc{
		{ID: "dense-winner", Score: 1},
		{ID: "dense-second", Score: 0.9},
		{ID: "lexical-winner", Score: 0},
	}
	bm25Scores := []retrievalScoredDoc{
		{ID: "lexical-winner", Score: 10},
		{ID: "dense-second", Score: 5},
		{ID: "dense-winner", Score: 0},
	}

	got := fuseHybridScores(denseScores, bm25Scores, 100, RetrievalEvalHybridConfig{
		Method:           "minmax_blend",
		Alpha:            0.75,
		DenseProtectTopK: 2,
	})
	if len(got) < 3 {
		t.Fatalf("fused length = %d, want at least 3", len(got))
	}
	want := []string{"dense-winner", "dense-second", "lexical-winner"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("protected fused[%d] = %q, want %q; got=%+v", i, got[i].ID, id, got[:3])
		}
	}
	if got[0].DenseRank != 1 || got[0].BM25Rank != 3 {
		t.Fatalf("protected dense winner component ranks = dense:%d bm25:%d, want 1/3", got[0].DenseRank, got[0].BM25Rank)
	}
	if got[0].DenseScore == nil || *got[0].DenseScore != 1 || got[0].BM25Score == nil || *got[0].BM25Score != 0 {
		t.Fatalf("protected dense winner component raw scores = dense:%v bm25:%v, want 1/0", got[0].DenseScore, got[0].BM25Score)
	}
	if got[0].DenseNormalizedScore == nil || *got[0].DenseNormalizedScore != 1 || got[0].BM25NormalizedScore == nil || *got[0].BM25NormalizedScore != 0 {
		t.Fatalf("protected dense winner component normalized scores = dense:%v bm25:%v, want 1/0", got[0].DenseNormalizedScore, got[0].BM25NormalizedScore)
	}
}

func TestFuseHybridScoresDenseCandidatesOnlyReranksDenseSet(t *testing.T) {
	denseScores := []retrievalScoredDoc{
		{ID: "dense-winner", Score: 1},
		{ID: "dense-tail", Score: 0.9},
	}
	bm25Scores := []retrievalScoredDoc{
		{ID: "sparse-only", Score: 100},
		{ID: "dense-tail", Score: 50},
		{ID: "dense-winner", Score: 0},
	}

	got := fuseHybridScores(denseScores, bm25Scores, 2, RetrievalEvalHybridConfig{
		Method:              "minmax_blend",
		Alpha:               0.75,
		DenseCandidatesOnly: true,
	})
	if len(got) != 2 {
		t.Fatalf("fused length = %d, want dense top-k length 2: %+v", len(got), got)
	}
	want := []string{"dense-tail", "dense-winner"}
	for i, id := range want {
		if got[i].ID != id {
			t.Fatalf("dense-candidates-only fused[%d] = %q, want %q; got=%+v", i, got[i].ID, id, got)
		}
	}
	for _, score := range got {
		if score.ID == "sparse-only" {
			t.Fatalf("sparse-only candidate entered dense-candidates-only result: %+v", got)
		}
	}
	if got[0].DenseRank != 2 || got[0].BM25Rank != 2 {
		t.Fatalf("reranked dense tail component ranks = dense:%d bm25:%d, want 2/2", got[0].DenseRank, got[0].BM25Rank)
	}
	if got[0].DenseScore == nil || math.Abs(*got[0].DenseScore-0.9) > 1e-6 || got[0].BM25Score == nil || *got[0].BM25Score != 50 {
		t.Fatalf("reranked dense tail component scores = dense:%v bm25:%v, want 0.9/50", got[0].DenseScore, got[0].BM25Score)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsQualityAndCost(t *testing.T) {
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ID: "d3", Vector: normalizeRetrievalVector([]float32{0, 0, 1, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq",
		TopK:        100,
	}, []int{8}, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-tq" {
		t.Fatalf("metrics identity = schema:%q dataset:%q", metrics.Schema, metrics.Dataset)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want perfect", metrics.Dense.Quality)
	}
	if metrics.Dense.VectorBytes != int64(len(docs)*len(docs[0].Vector)*4) {
		t.Fatalf("dense vector bytes = %d", metrics.Dense.VectorBytes)
	}
	if metrics.Dense.QueryLatency.Count != len(queries) || metrics.Dense.QueryLatency.P95MS < 0 {
		t.Fatalf("dense query latency = %+v, want populated latency metrics", metrics.Dense.QueryLatency)
	}
	if len(metrics.Rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(metrics.Rows))
	}
	row := metrics.Rows[0]
	if row.Bits != 8 || row.Method != "turboquant_ip_b8" {
		t.Fatalf("row identity = bits:%d method:%q", row.Bits, row.Method)
	}
	if row.VectorBytes <= 0 || row.VectorBytes >= row.DenseVectorBytes {
		t.Fatalf("quantized bytes = %d dense bytes = %d", row.VectorBytes, row.DenseVectorBytes)
	}
	if row.CompressionRatio <= 1 {
		t.Fatalf("compression ratio = %v, want > 1", row.CompressionRatio)
	}
	if row.QueryLatency.Count != len(queries) || row.QueryLatency.P95MS < 0 {
		t.Fatalf("quantized query latency = %+v, want populated latency metrics", row.QueryLatency)
	}
	if row.Quality.NDCGAt10 < 0.99 || row.Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", row.Quality)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsDenseRerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerank(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with rerank: %v", err)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_dense_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	if rerank.RerankStorage != TurboQuantRerankStorageDense || rerank.RerankSidecarBytes != rerank.DenseVectorBytes {
		t.Fatalf("dense rerank storage = storage:%q sidecar:%d dense:%d", rerank.RerankStorage, rerank.RerankSidecarBytes, rerank.DenseVectorBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes+rerank.RerankSidecarBytes || rerank.TotalCompression <= 0 {
		t.Fatalf("dense rerank total accounting = %+v", rerank)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
	if rerank.VectorBytes != metrics.Rows[0].VectorBytes || rerank.CompressionRatio != metrics.Rows[0].CompressionRatio {
		t.Fatalf("rerank storage = %+v direct = %+v", rerank, metrics.Rows[0])
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsCompactReconstructRerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-compact-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with compact rerank: %v", err)
	}
	if metrics.Config.RerankStorage != TurboQuantRerankStorageCompactReconstruct {
		t.Fatalf("config rerank storage = %q", metrics.Config.RerankStorage)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and compact rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_reconstruct_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("compact rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	if rerank.RerankStorage != TurboQuantRerankStorageCompactReconstruct || rerank.RerankSidecarBytes != 0 {
		t.Fatalf("compact rerank storage = storage:%q sidecar:%d", rerank.RerankStorage, rerank.RerankSidecarBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes || rerank.TotalCompression != rerank.CompressionRatio {
		t.Fatalf("compact rerank total accounting = %+v", rerank)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("compact rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
}

// mixedWidthRerankFixtureDim is the dimension shared by every mixed-width
// (--rerank-bits) test fixture below.
const mixedWidthRerankFixtureDim = 8

// mixedWidthRerankQuery, mixedWidthRerankDocA, and mixedWidthRerankDocB are a
// seed=1, dim=8 query/document pair found by direct search against
// turboquant.NewIPWithSeed(8, 4, 1) and turboquant.NewIPWithSeed(8, 8, 1):
// docA's true (dense) dot product with the query exceeds docB's, but
// dequantizing BOTH from bit=4 codes flips that order (docB scores higher),
// while dequantizing both from an independent bit=8 sidecar restores the
// correct order. Both quantizers are deterministic given the seed, so these
// literals reproduce the same ordering on every run.
var (
	mixedWidthRerankQuery = []float32{1, 0, 0, 0, 0, 0, 0, 0}
	mixedWidthRerankDocA  = []float32{1, -0.295449, -0.7959283, 2.3557012, -2.5782795, 2.6204405, -2.0905173, 1.4592514}
	mixedWidthRerankDocB  = []float32{0.995, -1.7334143, 1.330273, 2.9839318, 2.8408797, 2.1353064, 0.76346475, 1.5357535}
)

// TestTopTurboQuantMixedWidthRerankScoresChangesOrderVsSameBitsRerank is the
// most direct proof that an independent-width sidecar changes rerank
// ordering: the same two candidates, from the same primary overfetch set,
// rerank to opposite top-1 IDs depending on whether the rerank surface reuses
// the primary bit=4 codes (topTurboQuantReconstructRerankScores) or reads an
// independent bit=8 sidecar (topTurboQuantMixedWidthRerankScores).
func TestTopTurboQuantMixedWidthRerankScoresChangesOrderVsSameBitsRerank(t *testing.T) {
	seed := int64(1)
	q4 := turboquant.NewIPWithSeed(mixedWidthRerankFixtureDim, 4, seed)
	q8 := turboquant.NewIPWithSeed(mixedWidthRerankFixtureDim, 8, seed)
	candidates := []retrievalScoredDoc{
		{ID: "doc_a", Score: 1},
		{ID: "doc_b", Score: 0.995},
	}

	sameBitsDocs := map[string]turboquant.IPQuantized{
		"doc_a": q4.Quantize(mixedWidthRerankDocA),
		"doc_b": q4.Quantize(mixedWidthRerankDocB),
	}
	sameBitsRanked := topTurboQuantReconstructRerankScores(q4, mixedWidthRerankQuery, candidates, sameBitsDocs, 2)
	if len(sameBitsRanked) != 2 || sameBitsRanked[0].ID != "doc_b" {
		t.Fatalf("same-bits (b4) rerank = %+v, want doc_b misordered ahead of doc_a", sameBitsRanked)
	}

	mixedWidthDocs := map[string]turboquant.IPQuantized{
		"doc_a": q8.Quantize(mixedWidthRerankDocA),
		"doc_b": q8.Quantize(mixedWidthRerankDocB),
	}
	mixedWidthRanked := topTurboQuantMixedWidthRerankScores(q8, mixedWidthRerankQuery, candidates, mixedWidthDocs, 2)
	if len(mixedWidthRanked) != 2 || mixedWidthRanked[0].ID != "doc_a" {
		t.Fatalf("mixed-width (b8 sidecar) rerank = %+v, want doc_a corrected to rank 1", mixedWidthRanked)
	}
}

// mixedWidthRerankCorpus builds the full-pipeline fixture for the
// evaluateTurboQuantVectorRetrievalWithRerankStorage-level mixed-width rerank
// tests: mixedWidthRerankDocA and mixedWidthRerankDocB, plus enough filler
// documents (a large negative dot product with the query, so they never
// contend for the top ranks) to exercise a realistic overfetch depth.
func mixedWidthRerankCorpus(total int) []retrievalVectorRecord {
	docs := make([]retrievalVectorRecord, total)
	for i := range docs {
		vec := []float32{-1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07}
		vec[1+(i%7)] += float32(i) * 0.0001
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("filler%d", i), Vector: vec}
	}
	docs[0] = retrievalVectorRecord{ID: "doc_a", Vector: mixedWidthRerankDocA}
	docs[1] = retrievalVectorRecord{ID: "doc_b", Vector: mixedWidthRerankDocB}
	return docs
}

// TestEvaluateTurboQuantVectorRetrievalMixedWidthRerankCorrectsSameBitsMisorder
// proves requirement (a) through the full pipeline entry point: with
// --rerank-storage=compact-reconstruct and --rerank-bits unset, the b4
// same-bits rerank still misranks doc_b (not relevant) ahead of doc_a (the
// only relevant document), so NDCG@10 falls below a perfect score. Setting
// RerankBits=8 reranks from an independent sidecar instead and recovers a
// perfect NDCG@10, changing the candidate ordering exactly as bullet (a)
// requires.
func TestEvaluateTurboQuantVectorRetrievalMixedWidthRerankCorrectsSameBitsMisorder(t *testing.T) {
	docs := mixedWidthRerankCorpus(250)
	queries := []retrievalVectorRecord{{ID: "q1", Vector: mixedWidthRerankQuery}}
	qrels := retrievalQrels{"q1": {"doc_a": 1}}

	sameBits, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "mixed-width-rerank-corrects",
		TopK:        100,
	}, []int{4}, []int{200}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("same-bits evaluate: %v", err)
	}
	if len(sameBits.Rows) != 2 {
		t.Fatalf("same-bits rows = %d, want direct and rerank rows", len(sameBits.Rows))
	}
	sameBitsRerank := sameBits.Rows[1]
	if sameBitsRerank.Method != "turboquant_ip_b4_overfetch200_reconstruct_rerank" || sameBitsRerank.RerankBits != 0 {
		t.Fatalf("same-bits rerank row identity = method:%q rerank_bits:%d, want unchanged same-width method and rerank_bits 0", sameBitsRerank.Method, sameBitsRerank.RerankBits)
	}
	if sameBitsRerank.Quality.NDCGAt10 >= 0.9 {
		t.Fatalf("same-bits (b4) rerank ndcg@10 = %v, want a clear misorder (< 0.9): doc_b should still outrank doc_a", sameBitsRerank.Quality.NDCGAt10)
	}

	mixedWidth, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "mixed-width-rerank-corrects",
		TopK:        100,
		RerankBits:  8,
	}, []int{4}, []int{200}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("mixed-width evaluate: %v", err)
	}
	if len(mixedWidth.Rows) != 2 {
		t.Fatalf("mixed-width rows = %d, want direct and rerank rows", len(mixedWidth.Rows))
	}
	mixedWidthRerank := mixedWidth.Rows[1]
	if mixedWidthRerank.Method != "turboquant_ip_b4_overfetch200_reconstruct_rerank_b8" || mixedWidthRerank.RerankBits != 8 {
		t.Fatalf("mixed-width rerank row identity = method:%q rerank_bits:%d, want a b8-suffixed method and rerank_bits 8", mixedWidthRerank.Method, mixedWidthRerank.RerankBits)
	}
	if mixedWidthRerank.Quality.NDCGAt10 != 1 {
		t.Fatalf("mixed-width (b4 retrieve + b8 sidecar rerank) ndcg@10 = %v, want a perfect score once doc_a is corrected to rank 1", mixedWidthRerank.Quality.NDCGAt10)
	}
	if mixedWidthRerank.Quality.NDCGAt10 <= sameBitsRerank.Quality.NDCGAt10 {
		t.Fatalf("mixed-width rerank ndcg@10 = %v did not improve on same-bits rerank ndcg@10 = %v", mixedWidthRerank.Quality.NDCGAt10, sameBitsRerank.Quality.NDCGAt10)
	}
}

// TestEvaluateTurboQuantVectorRetrievalMixedWidthRerankAccountsSidecarBytes
// proves requirement (b): the sidecar byte cost is the corpus size times
// turboquantVectorBytes(dim, rerankBits) (the same helper every other
// TurboQuant storage accounting site uses), and TotalVectorBytes is exactly
// the direct (primary-width) bytes plus that sidecar cost.
func TestEvaluateTurboQuantVectorRetrievalMixedWidthRerankAccountsSidecarBytes(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}
	const rerankBits = 4

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-mixed-width-bytes",
		TopK:        100,
		RerankBits:  rerankBits,
	}, []int{8}, []int{110}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with mixed-width rerank: %v", err)
	}
	if metrics.Config.RerankBits != rerankBits {
		t.Fatalf("config rerank_bits = %d, want %d", metrics.Config.RerankBits, rerankBits)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and mixed-width rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_reconstruct_rerank_b4" || rerank.RerankBits != rerankBits {
		t.Fatalf("mixed-width rerank row identity = method:%q rerank_bits:%d", rerank.Method, rerank.RerankBits)
	}
	dim := len(docs[0].Vector)
	wantSidecarBytes := int64(len(docs)) * turboquantVectorBytes(dim, rerankBits)
	if rerank.RerankSidecarBytes != wantSidecarBytes {
		t.Fatalf("mixed-width rerank sidecar bytes = %d, want %d (docs * turboquantVectorBytes(dim, rerankBits))", rerank.RerankSidecarBytes, wantSidecarBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes+wantSidecarBytes {
		t.Fatalf("mixed-width rerank total bytes = %d, want direct (%d) + sidecar (%d) = %d", rerank.TotalVectorBytes, rerank.VectorBytes, wantSidecarBytes, rerank.VectorBytes+wantSidecarBytes)
	}
	if rerank.TotalCompression != ratioFloat64(float64(rerank.DenseVectorBytes), float64(rerank.TotalVectorBytes)) {
		t.Fatalf("mixed-width rerank total compression = %v, want dense/total", rerank.TotalCompression)
	}
}

// TestEvaluateTurboQuantVectorRetrievalRerankBitsEmptyPreservesCurrentOutputs
// proves requirement (c): leaving RerankBits at its zero value (the
// --rerank-bits default) reproduces the pre-existing compact-reconstruct
// rerank byte-for-byte, on the SAME fixture used above to prove the b4/b8
// misorder-then-correct behavior in (a) — an empty --rerank-bits must still
// exhibit the OLD same-bits misorder, not silently pick up any sidecar path.
func TestEvaluateTurboQuantVectorRetrievalRerankBitsEmptyPreservesCurrentOutputs(t *testing.T) {
	docs := mixedWidthRerankCorpus(250)
	queries := []retrievalVectorRecord{{ID: "q1", Vector: mixedWidthRerankQuery}}
	qrels := retrievalQrels{"q1": {"doc_a": 1}}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "mixed-width-rerank-bits-empty",
		TopK:        100,
	}, []int{4}, []int{200}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if metrics.Config.RerankBits != 0 {
		t.Fatalf("config rerank_bits = %d, want 0 with --rerank-bits unset", metrics.Config.RerankBits)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and rerank rows", len(metrics.Rows))
	}
	direct, rerank := metrics.Rows[0], metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b4_overfetch200_reconstruct_rerank" {
		t.Fatalf("rerank method = %q, want the unsuffixed same-width method name unchanged by this feature", rerank.Method)
	}
	if rerank.RerankBits != 0 {
		t.Fatalf("rerank_bits = %d, want 0 (no field leak) when --rerank-bits is empty", rerank.RerankBits)
	}
	if rerank.RerankSidecarBytes != 0 || rerank.TotalVectorBytes != rerank.VectorBytes || rerank.TotalCompression != rerank.CompressionRatio {
		t.Fatalf("rerank byte accounting = %+v, want no sidecar bytes at all when --rerank-bits is empty", rerank)
	}
	// The same-width reconstruct rerank still dequantizes from the PRIMARY
	// b4 codes (unchanged), so it reproduces the exact same misorder proven
	// in TestEvaluateTurboQuantVectorRetrievalMixedWidthRerankCorrectsSameBitsMisorder:
	// the direct row and the compact-reconstruct rerank row score identically.
	if rerank.Quality.NDCGAt10 != direct.Quality.NDCGAt10 {
		t.Fatalf("rerank ndcg@10 = %v, want it to match the direct row's ndcg@10 = %v (bit-identical rerank at the same width)", rerank.Quality.NDCGAt10, direct.Quality.NDCGAt10)
	}
}

// TestEvaluateTurboQuantVectorRetrievalRerankBitsRequiresCompactReconstructStorage
// guards the validation that --rerank-bits only means something for
// --rerank-storage=compact-reconstruct: dense and fp16 rerank already
// dequantize at a fixed, non-TurboQuant width, so a mixed TurboQuant bit
// width has no surface to apply to and must fail loudly instead of being
// silently ignored.
func TestEvaluateTurboQuantVectorRetrievalRerankBitsRequiresCompactReconstructStorage(t *testing.T) {
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})}}
	qrels := retrievalQrels{"q1": {"d1": 1}}

	_, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "rerank-bits-wrong-storage",
		TopK:        100,
		RerankBits:  8,
	}, []int{4}, []int{1}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err == nil {
		t.Fatal("evaluation succeeded with rerank-bits set under fp16 rerank storage")
	}
	if !strings.Contains(err.Error(), "rerank-bits") || !strings.Contains(err.Error(), TurboQuantRerankStorageCompactReconstruct) {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestEvaluateTurboQuantVectorRetrievalRerankBitsRejectsOutOfRangeWidth
// guards that --rerank-bits is validated with the same 2..8 bound as
// --bits, instead of reaching turboquant.NewIPWithSeed with an invalid
// width.
func TestEvaluateTurboQuantVectorRetrievalRerankBitsRejectsOutOfRangeWidth(t *testing.T) {
	docs := []retrievalVectorRecord{
		{ID: "d1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "d2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})}}
	qrels := retrievalQrels{"q1": {"d1": 1}}

	_, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "rerank-bits-out-of-range",
		TopK:        100,
		RerankBits:  9,
	}, []int{4}, []int{1}, TurboQuantRerankStorageCompactReconstruct, docs, queries, qrels)
	if err == nil {
		t.Fatal("evaluation succeeded with rerank-bits = 9 (out of the 2..8 range)")
	}
	if !strings.Contains(err.Error(), "rerank-bits") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEvaluateTurboQuantVectorRetrievalReportsFP16RerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-tq-fp16-rerank",
		TopK:        100,
	}, []int{8}, []int{110}, "half", docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with fp16 rerank: %v", err)
	}
	if metrics.Config.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("config rerank storage = %q", metrics.Config.RerankStorage)
	}
	if len(metrics.Rows) != 2 {
		t.Fatalf("rows = %d, want direct and fp16 rerank rows", len(metrics.Rows))
	}
	rerank := metrics.Rows[1]
	if rerank.Method != "turboquant_ip_b8_overfetch110_fp16_rerank" || rerank.RerankOverfetch != 110 {
		t.Fatalf("fp16 rerank row identity = method:%q overfetch:%d", rerank.Method, rerank.RerankOverfetch)
	}
	wantSidecarBytes := int64(len(docs) * len(docs[0].Vector) * 2)
	if rerank.RerankStorage != TurboQuantRerankStorageFP16 || rerank.RerankSidecarBytes != wantSidecarBytes {
		t.Fatalf("fp16 rerank storage = storage:%q sidecar:%d want:%d", rerank.RerankStorage, rerank.RerankSidecarBytes, wantSidecarBytes)
	}
	if rerank.TotalVectorBytes != rerank.VectorBytes+wantSidecarBytes {
		t.Fatalf("fp16 rerank total bytes = %d, want %d", rerank.TotalVectorBytes, rerank.VectorBytes+wantSidecarBytes)
	}
	if rerank.RerankSidecarBytes == rerank.DenseVectorBytes {
		t.Fatalf("fp16 rerank sidecar unexpectedly matches dense f32 sidecar bytes: %+v", rerank)
	}
	if rerank.TotalCompression >= rerank.CompressionRatio || rerank.TotalCompression >= 2 {
		t.Fatalf("fp16 total compression = %.6f quant compression = %.6f", rerank.TotalCompression, rerank.CompressionRatio)
	}
	if rerank.RerankScores != int64(len(queries)*110) || rerank.RerankScoreSeconds <= 0 {
		t.Fatalf("fp16 rerank accounting = scores:%d seconds:%f", rerank.RerankScores, rerank.RerankScoreSeconds)
	}
	if rerank.QueryLatency.Count != len(queries) || rerank.QueryLatency.P95MS < 0 {
		t.Fatalf("fp16 rerank query latency = %+v, want populated latency metrics", rerank.QueryLatency)
	}
	if rerank.Quality.NDCGAt10 < 0.99 || rerank.Quality.RecallAt100 != 1 {
		t.Fatalf("fp16 rerank quality = %+v, want near-perfect", rerank.Quality)
	}
}

func TestEvaluateTurboQuantVectorRetrievalFP16RerankOverfetch200MatchesSingleAndList(t *testing.T) {
	docs := make([]retrievalVectorRecord, 230)
	for i := range docs {
		vec := make([]float32, 8)
		vec[i%len(vec)] = 1
		vec[(i*3+1)%len(vec)] += float32(i%17) * 0.001
		switch i {
		case 0:
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		case 1:
			vec = []float32{0, 1, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0.1, 0, 0, 0, 0, 0, 0})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0.1, 1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"d1": 1},
		"q2": {"d2": 1},
	}

	dir := t.TempDir()
	singlePerQueryPath := filepath.Join(dir, "single.per-query.jsonl")
	listPerQueryPath := filepath.Join(dir, "list.per-query.jsonl")
	cfg := RetrievalEvalConfig{
		DatasetName:   "tiny-tq-fp16-command-shape",
		TopK:          100,
		QuantizerSeed: 5581486560434873699,
	}
	singleCfg := cfg
	singleCfg.PerQueryJSONLPath = singlePerQueryPath
	single, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), singleCfg, []int{4}, []int{200}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate single overfetch: %v", err)
	}
	listCfg := cfg
	listCfg.PerQueryJSONLPath = listPerQueryPath
	list, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), listCfg, []int{4}, []int{125, 150, 200, 225}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate overfetch list: %v", err)
	}

	singleRow := findTurboQuantMetricRow(t, single, "turboquant_ip_b4_overfetch200_fp16_rerank")
	listRow := findTurboQuantMetricRow(t, list, "turboquant_ip_b4_overfetch200_fp16_rerank")
	assertStableTurboQuantMetricRowsEqual(t, singleRow, listRow)

	singlePerQueryRows := readTurboQuantPerQueryRowsByMethod(t, singlePerQueryPath, "turboquant_ip_b4_overfetch200_fp16_rerank")
	listPerQueryRows := readTurboQuantPerQueryRowsByMethod(t, listPerQueryPath, "turboquant_ip_b4_overfetch200_fp16_rerank")
	if len(singlePerQueryRows) != len(listPerQueryRows) {
		t.Fatalf("per-query overfetch200 rows: single=%d list=%d", len(singlePerQueryRows), len(listPerQueryRows))
	}
	for i := range singlePerQueryRows {
		singleData, err := json.Marshal(singlePerQueryRows[i])
		if err != nil {
			t.Fatalf("marshal single per-query row: %v", err)
		}
		listData, err := json.Marshal(listPerQueryRows[i])
		if err != nil {
			t.Fatalf("marshal list per-query row: %v", err)
		}
		if string(singleData) != string(listData) {
			t.Fatalf("per-query row %d differs\nsingle=%s\nlist=%s", i, singleData, listData)
		}
	}
}

func TestEvaluateTurboQuantVectorRetrievalPerQueryTopKExtendsFormalFP16RerankRows(t *testing.T) {
	docs := make([]retrievalVectorRecord, 230)
	for i := range docs {
		vec := make([]float32, 8)
		vec[i%len(vec)] = 1
		vec[(i*3+1)%len(vec)] += float32(i%17) * 0.001
		if i == 0 {
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0.1, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"d1": 1}}

	perQueryPath := filepath.Join(t.TempDir(), "formal-window.per-query.jsonl")
	metrics, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-tq-fp16-formal-window",
		TopK:              100,
		PerQueryTopK:      120,
		PerQueryJSONLPath: perQueryPath,
		QuantizerSeed:     5581486560434873699,
	}, []int{4}, []int{200}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate formal fp16 rerank window: %v", err)
	}
	if metrics.Config.TopK != 100 {
		t.Fatalf("metrics top_k = %d, want 100", metrics.Config.TopK)
	}
	rerankMetric := findTurboQuantMetricRow(t, metrics, "turboquant_ip_b4_overfetch200_fp16_rerank")
	if rerankMetric.RerankOverfetch != 200 || rerankMetric.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("rerank metric = %+v", rerankMetric)
	}

	rows := readTurboQuantPerQueryRowsByMethod(t, perQueryPath, "turboquant_ip_b4_overfetch200_fp16_rerank")
	if len(rows) != 1 {
		t.Fatalf("rerank per-query rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Method != "turboquant_ip_b4_overfetch200_fp16_rerank" || row.Bits != 4 || row.RerankOverfetch != 200 || row.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("rerank per-query row identity = %+v", row)
	}
	if len(row.TopK) != 120 {
		t.Fatalf("rerank per-query top_k len = %d, want 120", len(row.TopK))
	}
	if row.TopK[119].Rank != 120 {
		t.Fatalf("rerank per-query final rank = %+v, want rank 120", row.TopK[119])
	}
}

func findTurboQuantMetricRow(t *testing.T, metrics TurboQuantRetrievalEvalMetrics, method string) TurboQuantRetrievalBitMetrics {
	t.Helper()
	for _, row := range metrics.Rows {
		if row.Method == method {
			return row
		}
	}
	t.Fatalf("method %q not found in %+v", method, metrics.Rows)
	return TurboQuantRetrievalBitMetrics{}
}

func assertStableTurboQuantMetricRowsEqual(t *testing.T, left, right TurboQuantRetrievalBitMetrics) {
	t.Helper()
	left.QuantizeSeconds = 0
	left.ScoreSeconds = 0
	left.RerankScoreSeconds = 0
	left.ScoresPerSecond = 0
	left.DocsPerSecond = 0
	left.QueryLatency = RetrievalEvalLatencyMetrics{}
	right.QuantizeSeconds = 0
	right.ScoreSeconds = 0
	right.RerankScoreSeconds = 0
	right.ScoresPerSecond = 0
	right.DocsPerSecond = 0
	right.QueryLatency = RetrievalEvalLatencyMetrics{}
	leftData, err := json.Marshal(left)
	if err != nil {
		t.Fatalf("marshal left metric row: %v", err)
	}
	rightData, err := json.Marshal(right)
	if err != nil {
		t.Fatalf("marshal right metric row: %v", err)
	}
	if string(leftData) != string(rightData) {
		t.Fatalf("stable metric rows differ\nsingle=%s\nlist=%s", leftData, rightData)
	}
}

func readTurboQuantPerQueryRowsByMethod(t *testing.T, path, method string) []TurboQuantRetrievalPerQueryRow {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	rows := []TurboQuantRetrievalPerQueryRow{}
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var row TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode per-query row: %v", err)
		}
		if row.Method == method {
			rows = append(rows, row)
		}
	}
	return rows
}

func TestTurboQuantFP16RerankRescueLeavesIncompleteWindowUnchanged(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(119)
	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	if len(got) != len(fp16Ranked) {
		t.Fatalf("rescued length = %d, want %d", len(got), len(fp16Ranked))
	}
	for i := range fp16Ranked {
		if got[i].ID != fp16Ranked[i].ID {
			t.Fatalf("rescued rank %d = %q, want unchanged %q", i+1, got[i].ID, fp16Ranked[i].ID)
		}
	}
}

func TestTurboQuantFP16RerankRescuePromotesStrongCompactBoundaryCandidate(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(121)
	compactScores["d120"] = 10_000
	compactRanks["d120"] = 1

	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	top100 := map[string]bool{}
	for _, doc := range got[:100] {
		top100[doc.ID] = true
	}
	if !top100["d120"] {
		t.Fatalf("d120 was not rescued into top100")
	}
	if top100["d100"] {
		t.Fatalf("d100 remained in top100 after higher-priority d120 rescue")
	}
	if got[99].ID != "d120" {
		t.Fatalf("rescued rank 100 = %q, want d120", got[99].ID)
	}
}

func TestTurboQuantFP16RerankRescueCompactTieBreakMatchesSimulation(t *testing.T) {
	fp16Ranked, compactScores, compactRanks := turboQuantRescueFixture(121)
	compactScores["d110"] = 10_000
	compactScores["d111"] = 10_000
	compactRanks["d110"] = 7
	compactRanks["d111"] = 7

	got := applyTurboQuantFP16RerankRescue(fp16Ranked, compactScores, compactRanks)
	if got[99].ID != "d111" {
		t.Fatalf("rescued compact/doc tie rank 100 = %q, want d111", got[99].ID)
	}
}

func TestEvaluateTurboQuantVectorRetrievalWritesCompactPerQueryJSONL(t *testing.T) {
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		vec := []float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}
		vec[i%len(vec)] += float32(i) * 0.0001
		if i == 0 {
			vec = []float32{1, 0, 0, 0, 0, 0, 0, 0}
		}
		docs[i] = retrievalVectorRecord{ID: fmt.Sprintf("d%d", i+1), Vector: normalizeRetrievalVector(vec)}
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})}}
	qrels := retrievalQrels{"q1": {"d1": 1}}
	perQueryPath := filepath.Join(t.TempDir(), "compact.per-query.jsonl")

	_, err := evaluateTurboQuantVectorRetrievalWithRerankStorage(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-compact",
		TopK:              100,
		PerQueryJSONLPath: perQueryPath,
	}, []int{8}, []int{110}, TurboQuantRerankStorageFP16, docs, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate turboquant retrieval with per-query: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want direct and rerank rows\n%s", len(lines), data)
	}
	var direct, rerank TurboQuantRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &direct); err != nil {
		t.Fatalf("decode direct row: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &rerank); err != nil {
		t.Fatalf("decode rerank row: %v", err)
	}
	if direct.Schema != TurboQuantRetrievalPerQuerySchema || direct.Method != "turboquant_ip_b8" || direct.Bits != 8 {
		t.Fatalf("direct row = %+v", direct)
	}
	if direct.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed || rerank.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("per-query seeds = direct:%d rerank:%d", direct.QuantizerSeed, rerank.QuantizerSeed)
	}
	if rerank.Method != "turboquant_ip_b8_overfetch110_fp16_rerank" || rerank.RerankOverfetch != 110 || rerank.RerankStorage != TurboQuantRerankStorageFP16 {
		t.Fatalf("rerank row = %+v", rerank)
	}
	if len(rerank.TopK) == 0 || rerank.TopK[0].DenseRank == nil || rerank.TopK[0].CompactRank == nil || rerank.TopK[0].DenseScore == nil || rerank.TopK[0].CompactScore == nil {
		t.Fatalf("rerank top doc missing dense/compact evidence: %+v", rerank.TopK)
	}
}

func turboQuantRescueFixture(n int) ([]retrievalScoredDoc, map[string]float32, map[string]int) {
	fp16Ranked := make([]retrievalScoredDoc, n)
	compactScores := make(map[string]float32, n)
	compactRanks := make(map[string]int, n)
	for i := range fp16Ranked {
		id := fmt.Sprintf("d%03d", i+1)
		fp16Ranked[i] = retrievalScoredDoc{ID: id, Score: float32(n - i)}
		compactScores[id] = float32(n - i)
		compactRanks[id] = i + 1
	}
	return fp16Ranked, compactScores, compactRanks
}

func TestMineCompactTextHardNegativesWritesManifestAndGuardsTestSplit(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "tiny")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","title":"positive","text":"alpha positive document"}`+"\n"+
			`{"_id":"d2","title":"negative","text":"alpha hard negative"}`+"\n"+
			`{"_id":"d3","title":"negative","text":"alpha boundary negative"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	perQueryPath := filepath.Join(dir, "compact.per-query.jsonl")
	row := TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
			{Rank: 2, DocID: "d1", Score: 0.8, Relevance: 1},
			{Rank: 3, DocID: "d3", Score: 0.7, Relevance: 0},
		},
	}
	data, _ := json.Marshal(row)
	if err := os.WriteFile(perQueryPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write per-query: %v", err)
	}
	blockedOutput := filepath.Join(dir, "blocked.jsonl")
	_, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        blockedOutput,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to mine train-selection rows from test split") {
		t.Fatalf("expected test split guard error, got %v", err)
	}
	_, err = MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        blockedOutput,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    true,
		AllowTestSmoke:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to mine train-selection rows from test split") {
		t.Fatalf("allow-test-smoke must not bypass train-selection guard, got %v", err)
	}
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifestPath := filepath.Join(dir, "manifest.json")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "test",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		ManifestPath:      manifestPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		TrainSelection:    false,
		NegativesPerRow:   2,
	})
	if err != nil {
		t.Fatalf("mine compact hard negatives: %v", err)
	}
	if manifest.TrainAllowed || manifest.LeakGuardStatus != "validation_smoke_no_train_test_split" || manifest.RowsEmitted != 1 || manifest.Negatives != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
	if manifest.PerQuerySHA256 == "" || manifest.HardNegativesSHA256 == "" || manifest.ReasonCounts["top10_competitor"] != 1 {
		t.Fatalf("manifest hashes/reasons = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Source == "" || len(examples[0].Negatives) != 2 {
		t.Fatalf("examples = %+v", examples)
	}
}

func TestMineCompactTextHardNegativesFiltersQrelsPositiveNegatives(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive one","text":"alpha first positive"}`,
			`{"_id":"d2","title":"positive two","text":"alpha stale row positive"}`,
			`{"_id":"d3","title":"negative","text":"alpha true negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.95, Relevance: 0},
			{Rank: 2, DocID: "d3", Score: 0.90, Relevance: 0},
		},
	})
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		NegativesPerRow:   2,
		MaxExamples:       1,
	})
	if err != nil {
		t.Fatalf("mine compact hard negatives: %v", err)
	}
	if manifest.QrelsRelevanceMismatches != 1 || manifest.Negatives != 1 {
		t.Fatalf("manifest mismatch/negative counts = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || len(examples[0].Negatives) != 1 || !strings.Contains(examples[0].Negatives[0], "true negative") {
		t.Fatalf("qrels-positive doc leaked or expected negative missing: %+v", examples)
	}
	if strings.Contains(strings.Join(examples[0].Negatives, "\n"), "stale row positive") {
		t.Fatalf("qrels-positive document emitted as negative: %+v", examples[0].Negatives)
	}
}

func TestMineCompactTextHardNegativesFiltersDuplicatePositiveTextNegatives(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","text":"alpha duplicate positive text"}`,
			`{"_id":"d2","text":"alpha   duplicate\npositive text"}`,
			`{"_id":"d3","title":"negative","text":"alpha true negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.95, Relevance: 0},
			{Rank: 2, DocID: "d3", Score: 0.90, Relevance: 0},
		},
	})
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		NegativesPerRow:   2,
	})
	if err != nil {
		t.Fatalf("mine compact hard negatives: %v", err)
	}
	if manifest.DuplicatePositiveTextNegativesSkipped != 1 || manifest.Negatives != 1 {
		t.Fatalf("manifest duplicate skip/negative counts = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || len(examples[0].Negatives) != 1 || !strings.Contains(examples[0].Negatives[0], "true negative") {
		t.Fatalf("duplicate positive text leaked or expected negative missing: %+v", examples)
	}
	var queryID string
	if err := json.Unmarshal(examples[0].ExtraFields["query_id"], &queryID); err != nil || queryID != "q1" {
		t.Fatalf("query_id provenance = %q err=%v fields=%+v", queryID, err, examples[0].ExtraFields)
	}
	var positiveDocID string
	if err := json.Unmarshal(examples[0].ExtraFields["positive_doc_id"], &positiveDocID); err != nil || positiveDocID != "d1" {
		t.Fatalf("positive_doc_id provenance = %q err=%v fields=%+v", positiveDocID, err, examples[0].ExtraFields)
	}
	var negativeDocIDs []string
	if err := json.Unmarshal(examples[0].ExtraFields["negative_doc_ids"], &negativeDocIDs); err != nil || len(negativeDocIDs) != 1 || negativeDocIDs[0] != "d3" {
		t.Fatalf("negative_doc_ids provenance = %+v err=%v fields=%+v", negativeDocIDs, err, examples[0].ExtraFields)
	}
}

func TestMineCompactTextHardNegativesQuantizerSeedDefaultAndMismatch(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive","text":"alpha positive"}`,
			`{"_id":"d2","title":"negative","text":"alpha negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
	)
	row := TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed + 1,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
		},
	}
	mismatchPath := writeCompactMiningRows(t, dir, row)
	_, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: mismatchPath,
		OutputPath:        filepath.Join(dir, "mismatch.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
	})
	if err == nil || !strings.Contains(err.Error(), "quantizer seed mismatch") {
		t.Fatalf("expected quantizer seed mismatch, got %v", err)
	}

	row.QuantizerSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	matchPath := writeCompactMiningRows(t, dir, row)
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: matchPath,
		OutputPath:        filepath.Join(dir, "matched.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
	})
	if err != nil {
		t.Fatalf("mine with default seed: %v", err)
	}
	if manifest.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("manifest quantizer seed = %d, want default %d", manifest.QuantizerSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
}

func TestMineCompactTextHardNegativesDerivesCompactReconstructMethod(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"positive","text":"alpha positive"}`,
			`{"_id":"d2","title":"negative","text":"alpha negative"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td1\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_reconstruct_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageCompactReconstruct,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
		},
	})
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        filepath.Join(dir, "hard-negatives.jsonl"),
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageCompactReconstruct,
	})
	if err != nil {
		t.Fatalf("mine compact reconstruct defaults: %v", err)
	}
	if manifest.Method != "turboquant_ip_b4_overfetch200_reconstruct_rerank" || manifest.RowsMatched != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestMineCompactTextHardNegativesMaxDocsPreservesQrelsPositive(t *testing.T) {
	dir, corpusPath, queriesPath, qrelsPath := writeCompactMiningDataset(t,
		[]string{
			`{"_id":"d1","title":"negative","text":"alpha early negative"}`,
			`{"_id":"d2","title":"filler","text":"alpha early filler"}`,
			`{"_id":"d3","title":"filler","text":"alpha later filler"}`,
			`{"_id":"d4","title":"positive","text":"alpha late positive"}`,
		},
		"query-id\tcorpus-id\tscore\nq1\td4\t1\n",
	)
	perQueryPath := writeCompactMiningRows(t, dir, TurboQuantRetrievalPerQueryRow{
		Schema:          TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   TurboQuantRerankStorageFP16,
		QuantizerSeed:   DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d1", Score: 0.9, Relevance: 0},
		},
	})
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifest, err := MineCompactTextHardNegatives(context.Background(), CompactHardNegativeMiningConfig{
		DatasetName:       "tiny",
		Split:             "train",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		PerQueryJSONLPath: perQueryPath,
		OutputPath:        outputPath,
		BitWidth:          4,
		Overfetch:         200,
		RerankStorage:     TurboQuantRerankStorageFP16,
		MaxDocs:           2,
	})
	if err != nil {
		t.Fatalf("mine with capped corpus: %v", err)
	}
	if manifest.RowsEmitted != 1 || manifest.SkippedNoPositive != 0 {
		t.Fatalf("manifest = %+v", manifest)
	}
	examples, err := ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read mined hard negatives: %v", err)
	}
	if len(examples) != 1 || !strings.Contains(examples[0].Positive, "late positive") {
		t.Fatalf("positive beyond cap was not preserved: %+v", examples)
	}
}

func writeCompactMiningDataset(t *testing.T, corpusLines []string, qrels string) (dir, corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	dir = t.TempDir()
	datasetDir := filepath.Join(dir, "tiny")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath = filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath = filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath = filepath.Join(datasetDir, "qrels", "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(strings.Join(corpusLines, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte(qrels), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir, corpusPath, queriesPath, qrelsPath
}

func writeCompactMiningRows(t *testing.T, dir string, rows ...TurboQuantRetrievalPerQueryRow) string {
	t.Helper()
	path := filepath.Join(dir, fmt.Sprintf("compact-%d.per-query.jsonl", time.Now().UnixNano()))
	var b strings.Builder
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal per-query row: %v", err)
		}
		b.Write(data)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write per-query: %v", err)
	}
	return path
}

func TestPercentileDurationUsesConservativeNearestRankForSmallSamples(t *testing.T) {
	durations := []time.Duration{
		1 * time.Millisecond,
		2 * time.Millisecond,
		3 * time.Millisecond,
		4 * time.Millisecond,
		5 * time.Millisecond,
	}
	tests := []struct {
		name string
		p    float64
		want time.Duration
	}{
		{name: "min", p: 0, want: 1 * time.Millisecond},
		{name: "median", p: 0.50, want: 3 * time.Millisecond},
		{name: "p95", p: 0.95, want: 5 * time.Millisecond},
		{name: "p99", p: 0.99, want: 5 * time.Millisecond},
		{name: "max", p: 1, want: 5 * time.Millisecond},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := percentileDuration(durations, tt.p); got != tt.want {
				t.Fatalf("percentileDuration(p=%v) = %v, want %v", tt.p, got, tt.want)
			}
		})
	}
}

func TestComputeRetrievalQualityUsesBoundedTopK(t *testing.T) {
	queries := []retrievalVectorRecord{{ID: "q", Vector: []float32{1}}}
	docs := make([]retrievalVectorRecord, 120)
	for i := range docs {
		docs[i] = retrievalVectorRecord{
			ID:     fmt.Sprintf("d%03d", i),
			Vector: []float32{float32(200 - i)},
		}
	}
	qrels := retrievalQrels{
		"q": {
			"d000": 1,
			"d009": 1,
			"d099": 1,
			"d100": 1,
		},
	}

	quality, queriesCount, relevantPairs, skippedDocs, skippedQueries := computeRetrievalQuality(queries, docs, qrels, 100)
	if queriesCount != 1 || relevantPairs != 4 || skippedDocs != 0 || skippedQueries != 0 {
		t.Fatalf("counts = queries:%d relevant:%d skippedDocs:%d skippedQueries:%d", queriesCount, relevantPairs, skippedDocs, skippedQueries)
	}
	if quality.MRRAt10 != 1 {
		t.Fatalf("mrr@10 = %v, want 1", quality.MRRAt10)
	}
	if quality.PrecisionAt1 != 1 {
		t.Fatalf("precision@1 = %v, want 1", quality.PrecisionAt1)
	}
	if quality.PrecisionAt5 != 0.2 {
		t.Fatalf("precision@5 = %v, want 0.2", quality.PrecisionAt5)
	}
	if quality.PrecisionAt10 != 0.2 {
		t.Fatalf("precision@10 = %v, want 0.2", quality.PrecisionAt10)
	}
	if quality.HitAt1 != 1 || quality.HitAt5 != 1 || quality.HitAt10 != 1 {
		t.Fatalf("hit quality = %+v, want hits at 1/5/10", quality)
	}
	if quality.RecallAt10 != 0.5 {
		t.Fatalf("recall@10 = %v, want 0.5", quality.RecallAt10)
	}
	if quality.RecallAt100 != 0.75 {
		t.Fatalf("recall@100 = %v, want 0.75", quality.RecallAt100)
	}
	if math.Abs(quality.MAPAt10-0.3) > 1e-12 {
		t.Fatalf("map@10 = %.12f, want 0.300000000000", quality.MAPAt10)
	}
	if math.Abs(quality.MAPAt100-0.3075) > 1e-12 {
		t.Fatalf("map@100 = %.12f, want 0.307500000000", quality.MAPAt100)
	}
}

func TestEmbedRetrievalTextsGroupsByTokenLengthAndPreservesOrder(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyMaskedEmbeddingManifest(), tinyEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}
	if err := model.attachTokenizer(tinyEmbeddingTokenizerFile()); err != nil {
		t.Fatalf("attach tokenizer: %v", err)
	}
	records := []retrievalTextRecord{
		{ID: "long-1", Text: "aa"},
		{ID: "short", Text: "a"},
		{ID: "long-2", Text: "aa"},
	}

	got, err := embedRetrievalTexts(context.Background(), model, records, 2, EmbeddingRoleRaw)
	if err != nil {
		t.Fatalf("embed retrieval texts: %v", err)
	}
	if len(got) != len(records) {
		t.Fatalf("embedded rows = %d, want %d", len(got), len(records))
	}
	for i, record := range records {
		if got[i].ID != record.ID {
			t.Fatalf("row %d id = %q, want %q", i, got[i].ID, record.ID)
		}
		want, err := model.EmbedText(context.Background(), record.Text)
		if err != nil {
			t.Fatalf("embed text %q: %v", record.Text, err)
		}
		wantRows, err := embeddingRows(want.Embeddings, 1)
		if err != nil {
			t.Fatalf("embedding rows: %v", err)
		}
		wantVector := normalizeRetrievalVector(wantRows[0])
		for j, wantValue := range wantVector {
			if diff := math.Abs(float64(got[i].Vector[j] - wantValue)); diff > 1e-5 {
				t.Fatalf("row %d vector[%d] = %v, want %v", i, j, got[i].Vector[j], wantValue)
			}
		}
	}
}

func TestRetrievalEvalRoPEBatchSizeInvariant(t *testing.T) {
	model := loadTinyRoPERetrievalEvalModel(t)
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels", "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"a"}`+"\n"+
			`{"_id":"d2","text":"bb"}`+"\n"+
			`{"_id":"d3","text":"ccc"}`+"\n"+
			`{"_id":"d4","text":"dddd"}`+"\n"+
			`{"_id":"d5","text":"eeeee"}`+"\n"+
			`{"_id":"d6","text":"ffffff"}`+"\n"+
			`{"_id":"d7","text":"ggggggg"}`+"\n"+
			`{"_id":"d8","text":"aaaaaaaa"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"ccc"}`+"\n"+
			`{"_id":"q2","text":"aaaaaaaa"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td3\t1\nq2\td8\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	corpus := []retrievalTextRecord{
		{ID: "d1", Text: "a"},
		{ID: "d2", Text: "bb"},
		{ID: "d3", Text: "ccc"},
		{ID: "d4", Text: "dddd"},
		{ID: "d5", Text: "eeeee"},
		{ID: "d6", Text: "ffffff"},
		{ID: "d7", Text: "ggggggg"},
		{ID: "d8", Text: "aaaaaaaa"},
	}
	baseVectors, err := embedRetrievalTexts(context.Background(), model, corpus, 1, EmbeddingRoleRaw)
	if err != nil {
		t.Fatalf("embed batch size 1: %v", err)
	}
	for _, batchSize := range []int{2, 4, 8} {
		got, err := embedRetrievalTexts(context.Background(), model, corpus, batchSize, EmbeddingRoleRaw)
		if err != nil {
			t.Fatalf("embed batch size %d: %v", batchSize, err)
		}
		assertRetrievalVectorsClose(t, fmt.Sprintf("batch-size-%d", batchSize), got, baseVectors, 5e-3)
	}

	var baseline RetrievalEvalMetrics
	for _, batchSize := range []int{1, 2, 4, 8} {
		metrics, err := EvaluateEmbeddingRetrieval(context.Background(), model, RetrievalEvalConfig{
			DatasetName: "tiny-rope",
			CorpusPath:  corpusPath,
			QueriesPath: queriesPath,
			QrelsPath:   qrelsPath,
			BatchSize:   batchSize,
			TopK:        100,
			RoleMode:    EmbeddingRoleModeRaw,
		})
		if err != nil {
			t.Fatalf("eval batch size %d: %v", batchSize, err)
		}
		if batchSize == 1 {
			baseline = metrics
			continue
		}
		assertRetrievalQualityClose(t, fmt.Sprintf("batch-size-%d", batchSize), metrics.Quality, baseline.Quality, 1e-12)
	}
}

func loadTinyRoPERetrievalEvalModel(t *testing.T) *EmbeddingModel {
	t.Helper()
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param attn_q: f16[D, D] @weight("weights/attn_q")
param attn_k: f16[D, D] @weight("weights/attn_k")
param attn_v: f16[D, D] @weight("weights/attn_v")
param attn_o: f16[D, D] @weight("weights/attn_o")
param projection: f16[D, E] @weight("weights/projection")

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_raw = gather(token_embedding, tokens)
    let hidden = rope(hidden_raw)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_raw = gather(token_embedding, tokens)
    let hidden = rope(hidden_raw)
    let q = @matmul(hidden, attn_q)
    let k = @matmul(hidden, attn_k)
    let v = @matmul(hidden, attn_v)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, attn_o)
    let projected = @matmul(attended, projection)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_rope_retrieval_eval_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, tinyRoPEAttentionEmbeddingManifest(), tinyRoPEAttentionEmbedWeights()...)
	if err != nil {
		t.Fatalf("load embedding: %v", err)
	}
	tokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b", "c", "d", "e", "f", "g"},
		UnknownToken: "[UNK]",
	}
	if err := model.attachTokenizer(tokenizer); err != nil {
		t.Fatalf("attach tokenizer: %v", err)
	}
	return model
}

func assertRetrievalVectorsClose(t *testing.T, label string, got, want []retrievalVectorRecord, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: vectors = %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("%s: row %d id = %q, want %q", label, i, got[i].ID, want[i].ID)
		}
		if len(got[i].Vector) != len(want[i].Vector) {
			t.Fatalf("%s: row %d vector dim = %d, want %d", label, i, len(got[i].Vector), len(want[i].Vector))
		}
		for j, wantValue := range want[i].Vector {
			if diff := math.Abs(float64(got[i].Vector[j] - wantValue)); diff > tol {
				t.Fatalf("%s: row %d vector[%d] = %v, want %v (diff %v)", label, i, j, got[i].Vector[j], wantValue, diff)
			}
		}
	}
}

func assertRetrievalQualityClose(t *testing.T, label string, got, want RetrievalEvalQualityMetrics, tol float64) {
	t.Helper()
	assertClose := func(name string, got, want float64) {
		t.Helper()
		if diff := math.Abs(got - want); diff > tol {
			t.Fatalf("%s: %s = %.15g, want %.15g (diff %.15g)", label, name, got, want, diff)
		}
	}
	assertClose("ndcg_at_10", got.NDCGAt10, want.NDCGAt10)
	assertClose("ndcg_at_100", got.NDCGAt100, want.NDCGAt100)
	assertClose("mrr_at_10", got.MRRAt10, want.MRRAt10)
	assertClose("precision_at_1", got.PrecisionAt1, want.PrecisionAt1)
	assertClose("precision_at_5", got.PrecisionAt5, want.PrecisionAt5)
	assertClose("precision_at_10", got.PrecisionAt10, want.PrecisionAt10)
	assertClose("hit_at_1", got.HitAt1, want.HitAt1)
	assertClose("hit_at_5", got.HitAt5, want.HitAt5)
	assertClose("hit_at_10", got.HitAt10, want.HitAt10)
	assertClose("map_at_10", got.MAPAt10, want.MAPAt10)
	assertClose("map_at_100", got.MAPAt100, want.MAPAt100)
	assertClose("recall_at_10", got.RecallAt10, want.RecallAt10)
	assertClose("recall_at_100", got.RecallAt100, want.RecallAt100)
}

func TestReadBEIRRetrievalFiles(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","title":"Title","text":"Document body"}`+"\n"+
			`{"_id":"d2","text":"Other document"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"document query"}`+"\n"+
			`{"_id":"q2","text":"unused query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	corpusPath, queriesPath, gotQrelsPath := BEIRRetrievalPaths(dir, "test")
	if gotQrelsPath != qrelsPath {
		t.Fatalf("qrels path = %q, want %q", gotQrelsPath, qrelsPath)
	}
	qrels, err := readBEIRQrels(gotQrelsPath)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	corpus, err := readBEIRCorpus(corpusPath, 0)
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	queries, skipped, err := readBEIRQueries(queriesPath, qrels, 0)
	if err != nil {
		t.Fatalf("read queries: %v", err)
	}
	if len(corpus) != 2 || corpus[0].Text != "Title\nDocument body" {
		t.Fatalf("corpus = %+v", corpus)
	}
	if len(queries) != 1 || queries[0].ID != "q1" || skipped != 0 {
		t.Fatalf("queries = %+v skipped=%d", queries, skipped)
	}
}

func TestReadBEIRQrelsAcceptsTRECFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.tsv")
	if err := os.WriteFile(path, []byte("q1\tQ0\td1\t2\nq1\tQ0\td2\t0\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	qrels, err := readBEIRQrels(path)
	if err != nil {
		t.Fatalf("read qrels: %v", err)
	}
	if got := qrels["q1"]["d1"]; got != 2 {
		t.Fatalf("qrels[q1][d1] = %v, want 2", got)
	}
	if _, ok := qrels["q1"]["d2"]; ok {
		t.Fatalf("non-positive qrel was retained: %+v", qrels)
	}
}

func TestEvaluateBM25RetrievalRanksLexicalMatch(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha finance"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	metrics, err := EvaluateBM25Retrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny",
		CorpusPath:  corpusPath,
		QueriesPath: queriesPath,
		QrelsPath:   qrelsPath,
		TopK:        100,
	})
	if err != nil {
		t.Fatalf("evaluate bm25: %v", err)
	}
	if metrics.Backend != "bm25" || metrics.Dataset != "tiny" {
		t.Fatalf("metrics identity = backend:%q dataset:%q", metrics.Backend, metrics.Dataset)
	}
	if metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 || metrics.Inputs.ScoredPairs != 2 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 || metrics.Quality.RecallAt10 != 1 || metrics.Quality.RecallAt100 != 1 || metrics.Quality.MAPAt10 != 1 {
		t.Fatalf("quality = %+v, want perfect lexical ranking", metrics.Quality)
	}
}

func TestEvaluateVectorCacheRetrievalUsesBEIRQualityMetrics(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"id":"d1","vector":[1,0]}`+"\n"+
			`{"id":"d2","vector":[0,1]}`+"\n"+
			`{"id":"d3","vector":[0.8,0.6]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"id":"q1","vector":[0.7,0.7]}`+"\n"+
			`{"id":"q2","vector":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-vectors",
		ArtifactPath:    "external-model",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
		BackendName:     "external",
		TopK:            100,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache retrieval: %v", err)
	}
	wantNDCG := (1/math.Log2(3) + 1) / 2
	if math.Abs(metrics.Quality.NDCGAt10-wantNDCG) > 1e-12 {
		t.Fatalf("ndcg@10 = %.12f, want %.12f", metrics.Quality.NDCGAt10, wantNDCG)
	}
	if metrics.Quality.MRRAt10 != 0.75 || metrics.Quality.RecallAt10 != 1 || metrics.Quality.RecallAt100 != 1 || metrics.Quality.HitAt1 != 0.5 || metrics.Quality.HitAt5 != 1 {
		t.Fatalf("quality = %+v", metrics.Quality)
	}
	if metrics.Schema != RetrievalEvalMetricsSchema || metrics.Dataset != "tiny-vectors" || metrics.Artifact != "external-model" || metrics.Backend != "external" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath {
		t.Fatalf("vector paths = %+v", metrics.Inputs)
	}
	if metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 2 || metrics.Inputs.RelevantPairs != 2 || metrics.Inputs.ScoredPairs != 6 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
}

func TestEvaluateVectorCacheRetrievalWritesPerQueryJSONL(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	perQueryPath := filepath.Join(dir, "per-query.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t2\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"id":"d1","vector":[1,0]}`+"\n"+
			`{"id":"d2","vector":[0,1]}`+"\n"+
			`{"id":"d3","vector":[0.8,0.6]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[0.7,0.7]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-vectors",
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         qrelsPath,
		DocVectorPath:     docVectorsPath,
		QueryVectorPath:   queryVectorsPath,
		TopK:              100,
		PerQueryJSONLPath: perQueryPath,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache retrieval: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("per-query lines = %d, want 1\n%s", len(lines), data)
	}
	var row RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.Schema != RetrievalEvalPerQuerySchema || row.Dataset != "tiny-vectors" || row.QueryID != "q1" {
		t.Fatalf("row identity = %+v", row)
	}
	if row.RelevantCount != 1 || row.FirstRelevantRank != 2 {
		t.Fatalf("row relevant summary = count:%d first:%d, want count 1 first rank 2", row.RelevantCount, row.FirstRelevantRank)
	}
	if len(row.TopK) != 3 {
		t.Fatalf("top_k len = %d, want 3", len(row.TopK))
	}
	if row.TopK[0].Rank != 1 || row.TopK[0].DocID != "d3" || row.TopK[0].Relevance != 0 {
		t.Fatalf("top_k[0] = %+v, want d3 non-relevant", row.TopK[0])
	}
	if row.TopK[1].Rank != 2 || row.TopK[1].DocID != "d1" || row.TopK[1].Relevance != 2 {
		t.Fatalf("top_k[1] = %+v, want d1 relevance 2", row.TopK[1])
	}
	if math.Abs(row.Quality.NDCGAt10-metrics.Quality.NDCGAt10) > 1e-12 || row.Quality.MRRAt10 != 0.5 {
		t.Fatalf("row quality = %+v metrics quality = %+v", row.Quality, metrics.Quality)
	}
}

func TestEvaluateTurboQuantVectorCacheRetrievalUsesExternalCaches(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d3","embedding":[0,0,1,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"_id":"q1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"q2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	metrics, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:     "tiny-cache-tq",
		ArtifactPath:    "bge-cache",
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
		BackendName:     "bge",
		TopK:            100,
	}, []int{8})
	if err != nil {
		t.Fatalf("evaluate turboquant vector cache retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-cache-tq" || metrics.Artifact != "bge-cache" || metrics.Backend != "bge" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath {
		t.Fatalf("vector paths = %+v", metrics.Inputs)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want perfect", metrics.Dense.Quality)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Bits != 8 || metrics.Rows[0].CompressionRatio <= 1 {
		t.Fatalf("rows = %+v", metrics.Rows)
	}
	if metrics.Rows[0].Quality.NDCGAt10 < 0.99 || metrics.Rows[0].Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", metrics.Rows[0].Quality)
	}
}

func TestReadRetrievalChildVectorCachePreservesMultipleChildrenPerParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child-vectors.jsonl")
	if err := os.WriteFile(path, []byte(
		`{"parent_id":"p2","child_id":"p2-c1","vector":[0,1]}`+"\n"+
			`{"parent_id":"p1","child_id":"p1-c1","embedding":[1,0]}`+"\n"+
			`{"parent_id":"p1","child_id":"p1-c2","values":[0.8,0.6]}`+"\n"+
			`{"id":"p3","vector":[0,0.5]}`+"\n"+
			`{"parent_id":"skip","child_id":"skip-c1","vector":[1,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write child vectors: %v", err)
	}

	children, missing, dim, err := readRetrievalChildVectorCache(path, []string{"p1", "p2", "p3", "missing"})
	if err != nil {
		t.Fatalf("read child vectors: %v", err)
	}
	if missing != 1 || dim != 2 || len(children) != 4 {
		t.Fatalf("missing=%d dim=%d children=%d", missing, dim, len(children))
	}
	got := []string{}
	for _, child := range children {
		got = append(got, child.ParentID+"/"+child.ChildID)
	}
	want := []string{"p1/p1-c1", "p1/p1-c2", "p2/p2-c1", "p3/p3"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("child order = %v, want %v", got, want)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalAggregatesChildrenByParent(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1}}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "tiny-multivector",
		TopK:        100,
		BaselineDim: 32,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate multivector turboquant retrieval: %v", err)
	}
	if metrics.Schema != TurboQuantMultiVectorRetrievalEvalMetricsSchema || metrics.Dataset != "tiny-multivector" {
		t.Fatalf("metrics identity = schema:%q dataset:%q", metrics.Schema, metrics.Dataset)
	}
	if metrics.Inputs.Parents != 2 || metrics.Inputs.ParentCount != 2 || metrics.Inputs.ChildVectors != 3 || metrics.Inputs.ChildCount != 3 || metrics.Inputs.AverageChildrenPerParent != 1.5 || metrics.Inputs.AvgChildrenPerParent != 1.5 || metrics.Inputs.MaxChildrenPerParent != 2 || metrics.Inputs.ScoredChildPairs != 3 {
		t.Fatalf("input accounting = %+v", metrics.Inputs)
	}
	if metrics.Config.BaselineDim != 32 {
		t.Fatalf("baseline dim = %d", metrics.Config.BaselineDim)
	}
	if metrics.Dense.BaselineDim != 32 || metrics.Dense.DenseBaselineBytes != int64(32*4) || metrics.Dense.DenseParentBytes != int64(2*32*4) || metrics.Dense.DenseChildBytes != int64(3*8*4) {
		t.Fatalf("dense bytes = %+v", metrics.Dense)
	}
	if metrics.Dense.VectorsThatFitInOneDenseBaseline != 4 || metrics.Dense.StorageMultipleOfDenseBaseline != 0.375 {
		t.Fatalf("dense baseline accounting = %+v", metrics.Dense)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || metrics.Dense.Quality.MRRAt10 != 1 || metrics.Dense.Quality.RecallAt100 != 1 {
		t.Fatalf("dense quality = %+v, want parent p1 top-ranked by second child", metrics.Dense.Quality)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Bits != 8 {
		t.Fatalf("rows = %+v", metrics.Rows)
	}
	row := metrics.Rows[0]
	if row.Method != "turboquant_ip_b8_child_max" || row.QuantizedChildBytes <= 0 || row.DenseChildCompression <= 1 {
		t.Fatalf("row accounting = %+v", row)
	}
	if metrics.Config.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed || row.QuantizerSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("quantizer seeds = config:%d row:%d", metrics.Config.QuantizerSeed, row.QuantizerSeed)
	}
	if row.ParentBudgetStorageMultiple <= 0 || row.StorageMultipleOfDenseBaseline != row.ParentBudgetStorageMultiple || row.DenseParentBytes != metrics.Dense.DenseParentBytes || row.DenseBaselineBytes != metrics.Dense.DenseBaselineBytes || row.DenseChildBytes != metrics.Dense.DenseChildBytes {
		t.Fatalf("row storage = %+v", row)
	}
	if row.BaselineDim != 32 || row.ParentCount != 2 || row.ChildCount != 3 || row.AvgChildrenPerParent != 1.5 || row.MaxChildrenPerParent != 2 || row.QuantizedVectorBytes <= 0 || row.VectorsThatFitInOneDenseBaseline <= 0 {
		t.Fatalf("row baseline accounting = %+v", row)
	}
	if row.Quality.NDCGAt10 < 0.99 || row.Quality.RecallAt100 != 1 {
		t.Fatalf("quantized quality = %+v, want near-perfect", row.Quality)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalFailsMissingRelevantParentsByDefault(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1, "missing-parent": 1}}

	_, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName: "missing-relevant",
		TopK:        100,
	}, []int{8}, children, queries, qrels)
	if err == nil || !strings.Contains(err.Error(), "missing 1 qrels-relevant parent documents") || !strings.Contains(err.Error(), "--allow-missing-relevant") {
		t.Fatalf("strict coverage error = %v", err)
	}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:          "missing-relevant",
		TopK:                 100,
		AllowMissingRelevant: true,
		QuantizerSeed:        17,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("allow missing relevant: %v", err)
	}
	if !metrics.Config.AllowMissingRelevant || metrics.Config.QuantizerSeed != 17 {
		t.Fatalf("config = %+v", metrics.Config)
	}
	if metrics.Rows[0].SkippedRelevantDocs != 1 || metrics.Rows[0].SkippedQueries != 0 {
		t.Fatalf("skipped counts = row:%+v skipped:%+v", metrics.Rows[0], metrics.SkippedCounts)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalSeededRowsRepeat(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0.9, 0.1, 0.2, 0.3, 0.05, 0.01, 0.4, 0.2})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{0.1, 0.8, 0.1, 0.2, 0.3, 0.2, 0.1, 0.4})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0.2, 0.1, 0.9, 0.1, 0.4, 0.3, 0.2, 0.1})},
		{ParentID: "p3", ChildID: "p3-a", Vector: normalizeRetrievalVector([]float32{0.1, 0.2, 0.2, 0.9, 0.1, 0.4, 0.3, 0.2})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0.1, 0.2, 0.3, 0.1, 0, 0.4, 0.2})},
		{ID: "q2", Vector: normalizeRetrievalVector([]float32{0.1, 0.2, 1, 0.1, 0.4, 0.2, 0.1, 0})},
	}
	qrels := retrievalQrels{
		"q1": {"p1": 1},
		"q2": {"p2": 1},
	}
	cfg := RetrievalEvalConfig{DatasetName: "seeded", TopK: 100, QuantizerSeed: 12345}

	first, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), cfg, []int{2, 4}, children, queries, qrels)
	if err != nil {
		t.Fatalf("first evaluation: %v", err)
	}
	second, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), cfg, []int{2, 4}, children, queries, qrels)
	if err != nil {
		t.Fatalf("second evaluation: %v", err)
	}
	if first.Config.QuantizerSeed != 12345 || second.Config.QuantizerSeed != 12345 {
		t.Fatalf("config seeds = %d/%d", first.Config.QuantizerSeed, second.Config.QuantizerSeed)
	}
	for i := range first.Rows {
		a, b := first.Rows[i], second.Rows[i]
		if a.Bits != b.Bits || a.Method != b.Method || a.QuantizerSeed != b.QuantizerSeed || a.QuantizedChildBytes != b.QuantizedChildBytes {
			t.Fatalf("row identity mismatch %d: %+v vs %+v", i, a, b)
		}
		if a.Quality != b.Quality || a.NDCGAt10Delta != b.NDCGAt10Delta || a.RecallAt100Delta != b.RecallAt100Delta {
			t.Fatalf("quality mismatch %d: %+v vs %+v", i, a, b)
		}
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalWritesPerQueryJSONL(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p1", ChildID: "p1-b", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
		{ParentID: "p2", ChildID: "p2-a", Vector: normalizeRetrievalVector([]float32{0, 1, 0, 0, 0, 0, 0, 0})},
	}
	queries := []retrievalVectorRecord{
		{ID: "q1", Vector: normalizeRetrievalVector([]float32{1, 0, 0, 0, 0, 0, 0, 0})},
	}
	qrels := retrievalQrels{"q1": {"p1": 1}}
	perQueryPath := filepath.Join(t.TempDir(), "multivector.per-query.jsonl")

	_, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:       "tiny-multivector",
		TopK:              1,
		QuantizerSeed:     99,
		PerQueryJSONLPath: perQueryPath,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate multivector turboquant retrieval: %v", err)
	}
	data, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want dense and q8 rows\n%s", len(lines), data)
	}
	var dense, compact TurboQuantMultiVectorRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &dense); err != nil {
		t.Fatalf("decode dense row: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &compact); err != nil {
		t.Fatalf("decode compact row: %v", err)
	}
	if dense.Schema != TurboQuantMultiVectorRetrievalPerQuerySchema || dense.Dataset != "tiny-multivector" || dense.Method != "float32_child_max" || dense.QueryID != "q1" {
		t.Fatalf("dense row identity = %+v", dense)
	}
	if dense.FirstRelevantRank != 1 || dense.RelevantCount != 1 || len(dense.TopK) == 0 || dense.TopK[0].DocID != "p1" || dense.TopK[0].ChildID != "p1-b" || dense.TopK[0].ChildScore == nil {
		t.Fatalf("dense row evidence = %+v", dense)
	}
	if compact.Method != "turboquant_ip_b8_child_max" || compact.Bits != 8 || compact.QuantizerSeed != 99 || compact.ScoringSurface != "turboquant_ip_prepared_child_max" {
		t.Fatalf("compact row identity = %+v", compact)
	}
	if len(compact.TopK) == 0 || compact.TopK[0].CompactRank == nil || *compact.TopK[0].CompactRank != 1 || compact.TopK[0].CompactChildID == "" || compact.TopK[0].DenseRank == nil || compact.TopK[0].DenseChildID != "p1-b" {
		t.Fatalf("compact row evidence = %+v", compact)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalDefaultMaxLabelsRemain(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "p1", ChildID: "p1-a", Vector: []float32{0.9, 0, 0, 0, 0, 0, 0, 0}},
		{ParentID: "p2", ChildID: "p2-a", Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}},
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}}}
	qrels := retrievalQrels{"q1": {"p1": 1}}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:   "tiny-default-max",
		TopK:          2,
		QuantizerSeed: 99,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate multivector turboquant retrieval: %v", err)
	}
	if metrics.Config.Aggregation != TurboQuantMultiVectorAggregationMax || metrics.Config.ChildCountPenalty != 0 {
		t.Fatalf("aggregation config = %+v", metrics.Config)
	}
	if metrics.Dense.Aggregation != TurboQuantMultiVectorAggregationMax || metrics.Dense.ChildCountPenalty != 0 {
		t.Fatalf("dense aggregation = %+v", metrics.Dense)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Method != "turboquant_ip_b8_child_max" || metrics.Rows[0].Aggregation != TurboQuantMultiVectorAggregationMax || metrics.Rows[0].ChildCountPenalty != 0 {
		t.Fatalf("row default identity = %+v", metrics.Rows)
	}
}

func TestTopMeanMultiVectorAggregationCanChangeParentOrdering(t *testing.T) {
	children := []retrievalChildVectorRecord{
		{ParentID: "steady", ChildID: "steady-a", Vector: []float32{0.9, 0, 0, 0, 0, 0, 0, 0}},
		{ParentID: "steady", ChildID: "steady-b", Vector: []float32{0.9, 0, 0, 0, 0, 0, 0, 0}},
		{ParentID: "spiky", ChildID: "spiky-a", Vector: []float32{1.0, 0, 0, 0, 0, 0, 0, 0}},
		{ParentID: "spiky", ChildID: "spiky-b", Vector: []float32{-1.0, 0, 0, 0, 0, 0, 0, 0}},
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}}}
	qrels := retrievalQrels{"q1": {"steady": 1}}

	maxMetrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:   "tiny-max",
		TopK:          2,
		QuantizerSeed: 99,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate max multivector retrieval: %v", err)
	}
	top2Metrics, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:                  "tiny-top2",
		TopK:                         2,
		QuantizerSeed:                99,
		MultiVectorAggregation:       "top2-mean",
		MultiVectorChildCountPenalty: 0,
	}, []int{8}, children, queries, qrels)
	if err != nil {
		t.Fatalf("evaluate top2 multivector retrieval: %v", err)
	}
	if maxMetrics.Dense.Quality.NDCGAt10 >= top2Metrics.Dense.Quality.NDCGAt10 {
		t.Fatalf("dense quality max=%+v top2=%+v, want top2 ordering improvement", maxMetrics.Dense.Quality, top2Metrics.Dense.Quality)
	}
	if top2Metrics.Config.Aggregation != "top2-mean" || top2Metrics.Rows[0].Aggregation != "top2-mean" {
		t.Fatalf("top2 aggregation metrics = config:%+v row:%+v", top2Metrics.Config, top2Metrics.Rows[0])
	}
	scores := topDenseMultiVectorParentScores(queries[0].Vector, children, 2, "top2-mean", 0)
	if len(scores) < 2 || scores[0].ID != "steady" || scores[0].ChildID != "steady-a" || scores[0].ChildScore == nil || scores[1].ID != "spiky" {
		t.Fatalf("top2 scores = %+v", scores)
	}
}

func TestEvaluateTurboQuantMultiVectorRetrievalRejectsNegativeChildCountPenalty(t *testing.T) {
	children := []retrievalChildVectorRecord{{ParentID: "p1", ChildID: "p1-a", Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}}}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: []float32{1, 0, 0, 0, 0, 0, 0, 0}}}
	qrels := retrievalQrels{"q1": {"p1": 1}}

	_, err := evaluateTurboQuantMultiVectorRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:                  "tiny-negative-penalty",
		TopK:                         1,
		MultiVectorChildCountPenalty: -0.1,
	}, []int{8}, children, queries, qrels)
	if err == nil {
		t.Fatal("evaluation succeeded with negative child-count penalty")
	}
	if !strings.Contains(err.Error(), "child-count-penalty must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEvaluateVectorCacheRetrievalRejectsDimensionMismatch(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(corpusPath, []byte(`{"_id":"d1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(`{"id":"d1","vector":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	_, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		CorpusPath:      corpusPath,
		QueriesPath:     queriesPath,
		QrelsPath:       qrelsPath,
		DocVectorPath:   docVectorsPath,
		QueryVectorPath: queryVectorsPath,
	})
	if err == nil || !strings.Contains(err.Error(), "dimension") {
		t.Fatalf("error = %v, want dimension mismatch", err)
	}
}

func TestMineBM25TextHardNegativesUsesTopLexicalNonPositive(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	examples, summary, err := MineBM25TextHardNegatives(context.Background(), RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        2,
		MaxExamples:          1,
	})
	if err != nil {
		t.Fatalf("mine hard negatives: %v", err)
	}
	if summary.DatasetName != "tiny" || summary.Examples != 1 || summary.PositivePairs != 1 || summary.Negatives != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] != "alpha distractor" {
		t.Fatalf("examples = %+v", examples)
	}
}

func TestMineBM25TextHardNegativesSkipsDuplicatePositiveText(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"alpha duplicate positive"}`+"\n"+
			`{"_id":"d2","text":"alpha\tduplicate   positive"}`+"\n"+
			`{"_id":"d3","text":"alpha true negative"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}

	examples, summary, err := MineBM25TextHardNegatives(context.Background(), RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        3,
	})
	if err != nil {
		t.Fatalf("mine hard negatives: %v", err)
	}
	if summary.DuplicatePositiveTextNegativesSkipped != 1 || summary.Negatives != 1 {
		t.Fatalf("summary duplicate skip/negative counts = %+v", summary)
	}
	if len(examples) != 1 || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] != "alpha true negative" {
		t.Fatalf("examples = %+v", examples)
	}
	var negativeDocIDs []string
	if err := json.Unmarshal(examples[0].ExtraFields["negative_doc_ids"], &negativeDocIDs); err != nil || len(negativeDocIDs) != 1 || negativeDocIDs[0] != "d3" {
		t.Fatalf("negative_doc_ids provenance = %+v err=%v fields=%+v", negativeDocIDs, err, examples[0].ExtraFields)
	}
}

func TestModelMiningNegativeTextsUsesTopModelNonPositive(t *testing.T) {
	scores := []retrievalScoredDoc{
		{ID: "positive", Score: 0.99},
		{ID: "hard", Score: 0.98},
		{ID: "duplicate", Score: 0.97},
		{ID: "blank", Score: 0.96},
		{ID: "easy", Score: 0.10},
	}
	positives := map[string]bool{"positive": true}
	docText := map[string]string{
		"positive":  "target",
		"hard":      "hard negative",
		"duplicate": "hard negative",
		"blank":     " ",
		"easy":      "easy negative",
	}

	negatives := modelMiningNegativeTexts(scores, positives, docText, RetrievalHardNegativeMiningConfig{
		NegativesPerPositive: 2,
		CandidateTopK:        4,
	})

	if len(negatives) != 2 || negatives[0] != "hard negative" || negatives[1] != "easy negative" {
		t.Fatalf("negatives = %+v", negatives)
	}
}

func TestModelMiningNegativeCandidatesSkipsDuplicatePositiveText(t *testing.T) {
	scores := []retrievalScoredDoc{
		{ID: "duplicate-positive-text", Score: 0.99},
		{ID: "hard", Score: 0.98},
		{ID: "easy", Score: 0.10},
	}
	positives := map[string]bool{"positive": true}
	docText := map[string]string{
		"positive":                "target positive text",
		"duplicate-positive-text": "target   positive\ntext",
		"hard":                    "hard negative",
		"easy":                    "easy negative",
	}
	result := modelMiningNegativeCandidates(scores, positives, retrievalPositiveTextFingerprints(map[string]float64{"positive": 1}, docText), docText, RetrievalHardNegativeMiningConfig{
		NegativesPerPositive: 2,
		CandidateTopK:        3,
	})
	if result.DuplicatePositiveTextNegativesSkipped != 1 {
		t.Fatalf("duplicate skip count = %d, want 1", result.DuplicatePositiveTextNegativesSkipped)
	}
	if got := scoredTextValues(result.Candidates); len(got) != 2 || got[0] != "hard negative" || got[1] != "easy negative" {
		t.Fatalf("negatives = %+v", got)
	}
}

func TestMineModelTextHardNegativesUsesResolvedRoleMode(t *testing.T) {
	model := loadTinyRoleRetrievalMiningModel(t)
	dir := t.TempDir()
	corpusPath, queriesPath, qrelsPath := writeRoleSensitiveMiningDataset(t, dir)

	autoExamples, autoSummary, err := MineModelTextHardNegatives(context.Background(), model, RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny-role",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        2,
		BatchSize:            2,
	})
	if err != nil {
		t.Fatalf("mine auto role negatives: %v", err)
	}
	if autoSummary.RoleMode != EmbeddingRoleModeQueryDocument {
		t.Fatalf("auto summary role mode = %q, want query-document", autoSummary.RoleMode)
	}
	if len(autoExamples) != 1 || len(autoExamples[0].Negatives) != 1 || autoExamples[0].Negatives[0] != "b" {
		t.Fatalf("auto examples = %+v, want role-conditioned negative b", autoExamples)
	}

	rawExamples, rawSummary, err := MineModelTextHardNegatives(context.Background(), model, RetrievalHardNegativeMiningConfig{
		DatasetName:          "tiny-role",
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        2,
		BatchSize:            2,
		RoleMode:             EmbeddingRoleModeRaw,
	})
	if err != nil {
		t.Fatalf("mine raw role negatives: %v", err)
	}
	if rawSummary.RoleMode != EmbeddingRoleModeRaw {
		t.Fatalf("raw summary role mode = %q, want raw", rawSummary.RoleMode)
	}
	if len(rawExamples) != 1 || len(rawExamples[0].Negatives) != 1 || rawExamples[0].Negatives[0] != "a" {
		t.Fatalf("raw examples = %+v, want raw negative a", rawExamples)
	}
}

func TestMineModelTextHardNegativesRejectsQueryDocumentRoleModeForLegacyModel(t *testing.T) {
	model := loadTinyRetrievalExportModel(t)
	dir := t.TempDir()
	datasetDir := writeTinyRetrievalExportDataset(t, dir)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")
	_, _, err := MineModelTextHardNegatives(context.Background(), model, RetrievalHardNegativeMiningConfig{
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            qrelsPath,
		NegativesPerPositive: 1,
		CandidateTopK:        2,
		RoleMode:             EmbeddingRoleModeQueryDocument,
	})
	if err == nil || !strings.Contains(err.Error(), "role-mode query-document requires a role-conditioned embedding model") {
		t.Fatalf("err = %v, want query-document role-mode rejection", err)
	}
}

func loadTinyRoleRetrievalMiningModel(t *testing.T) *EmbeddingModel {
	t.Helper()
	model, _ := loadTinyRoleRetrievalMiningModelWithArtifact(t)
	return model
}

func loadTinyRoleRetrievalMiningModelWithArtifact(t *testing.T) (*EmbeddingModel, string) {
	t.Helper()
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "role_embed"})
	if err != nil {
		t.Fatalf("build role source: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "role_embed.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyRoleEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{4, 2}, []float32{
			0, 0,
			1, 0,
			0, 1,
			1, 1,
		}),
		"role_embedding": backend.NewTensorF16([]int{3, 2}, []float32{
			0, 0,
			-1, 1,
			0, 0,
		}),
		"projection": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	tokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "a", "b", "c"},
		UnknownToken: "[PAD]",
	}
	if err := tokenizer.WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	packageManifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": DefaultEmbeddingManifestPath(artifactPath),
		"weights":            DefaultWeightFilePath(artifactPath),
		"tokenizer":          DefaultTokenizerPath(artifactPath),
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	if err := packageManifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load role package: %v", err)
	}
	return model, artifactPath
}

func writeRoleSensitiveMiningDataset(t *testing.T, dir string) (corpusPath, queriesPath, qrelsPath string) {
	t.Helper()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.MkdirAll(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath = filepath.Join(dir, "corpus.jsonl")
	queriesPath = filepath.Join(dir, "queries.jsonl")
	qrelsPath = filepath.Join(qrelsDir, "train.tsv")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"d1","text":"c"}`+"\n"+
			`{"_id":"d2","text":"a"}`+"\n"+
			`{"_id":"d3","text":"b"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return corpusPath, queriesPath, qrelsPath
}
