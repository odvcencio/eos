package eosruntime

import (
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func writeTinyRetrievalGateFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"a"}`+"\n"+
			`{"_id":"d2","text":"a a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir
}

// Pairwise eval data must drive per-epoch evals, best selection, and restore;
// before cfg.EvalPairs existed the contrastive-train+pairwise-eval combination
// silently ran zero in-training evals (best_epoch always equalled the final
// epoch and early stopping never fired).
func TestFitContrastiveEvalPairsDrivePerEpochSelection(t *testing.T) {
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	cfg := EmbeddingTrainRunConfig{
		Epochs:      3,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        7,
		RestoreBest: true,
		EvalPairs:   tinyEmbeddingPairDataset(),
	}
	summary, err := trainer.FitContrastive(tinyEmbeddingContrastiveDataset(), nil, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	// initial (restore-best) + one per epoch + final = epochs + 2.
	if want := cfg.Epochs + 2; summary.Workload.ActualEvalPasses != want {
		t.Fatalf("eval passes = %d, want %d", summary.Workload.ActualEvalPasses, want)
	}
	if summary.FinalEval == nil || summary.BestEval == nil || summary.LastEval == nil {
		t.Fatalf("expected final/best/last eval metrics, got %+v", summary)
	}
	if len(summary.History) != cfg.Epochs {
		t.Fatalf("history = %d entries, want %d", len(summary.History), cfg.Epochs)
	}
	for _, record := range summary.History {
		if record.Eval == nil {
			t.Fatalf("epoch %d missing eval", record.Epoch)
		}
	}
	wantPairs := int64(len(cfg.EvalPairs) * (cfg.Epochs + 2))
	if summary.Workload.ActualEvalPairs != wantPairs {
		t.Fatalf("eval pairs = %d, want %d (pairwise accounting)", summary.Workload.ActualEvalPairs, wantPairs)
	}
}

// The retrieval gate must survive best-checkpoint restore: restoring rebuilds
// the trainer, and before the fix that wiped the configured gate so the
// post-restore final eval silently reported zero retrieval metrics.
func TestFitContrastiveRetrievalGateSurvivesRestore(t *testing.T) {
	dataset := writeTinyRetrievalGateFixture(t)
	corpusPath, queriesPath, qrelsPath := BEIRRetrievalPaths(dataset, "test")
	tok := tinyEmbeddingTokenizerFile()
	trainer := newTinyTrainableEmbeddingTrainer(t, 0.05)
	cfg := EmbeddingTrainRunConfig{
		Epochs:               2,
		BatchSize:            2,
		Shuffle:              true,
		Seed:                 7,
		RestoreBest:          true,
		SelectMetric:         "retrieval_ndcg",
		EvalPairs:            tinyEmbeddingPairDataset(),
		RetrievalEvalRuntime: New(cuda.New(), metal.New()),
		RetrievalEval: RetrievalEvalConfig{
			DatasetName: "tiny",
			CorpusPath:  corpusPath,
			QueriesPath: queriesPath,
			QrelsPath:   qrelsPath,
			BatchSize:   2,
		},
		RetrievalEvalTokenizer: &tok,
	}
	summary, err := trainer.FitContrastive(tinyEmbeddingContrastiveDataset(), nil, cfg)
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if !summary.RestoredBest {
		t.Fatal("expected best-checkpoint restore to run")
	}
	// Two docs and one relevant qrel: as long as the gate executes, the
	// relevant doc is in the top-10 and all retrieval metrics are strictly
	// positive. Zero means the gate was skipped.
	if summary.FinalEval == nil || summary.FinalEval.RetrievalNDCGAt10 <= 0 {
		t.Fatalf("final eval retrieval nDCG = %+v, want > 0 after restore", summary.FinalEval)
	}
	if summary.FinalEval.RetrievalMAPAt100 <= 0 || summary.FinalEval.RetrievalRecallAt100 <= 0 {
		t.Fatalf("final eval retrieval MAP/recall = %+v, want > 0 after restore", summary.FinalEval)
	}
	if summary.FinalEval.RetrievalEval == nil {
		t.Fatalf("final eval retrieval detail missing: %+v", summary.FinalEval)
	}
	if got := summary.FinalEval.RetrievalEval; got.Backend == "" || got.Inputs.Documents != 2 || got.Inputs.Queries != 1 || got.Inputs.ScoredPairs != 2 {
		t.Fatalf("final eval retrieval detail = %+v, want backend and counts", got)
	}
	if got := summary.FinalEval.RetrievalEval.Throughput; got.ElapsedSeconds <= 0 || got.DocumentEmbedSeconds <= 0 || got.QueryEmbedSeconds <= 0 || got.ScoreSeconds <= 0 || got.DocumentsPerSecond <= 0 || got.QueriesPerSecond <= 0 || got.ScoresPerSecond <= 0 {
		t.Fatalf("final eval retrieval throughput = %+v, want positive timings/rates", got)
	}
	for _, record := range summary.History {
		if record.Eval == nil || record.Eval.RetrievalNDCGAt10 <= 0 {
			t.Fatalf("epoch %d retrieval nDCG missing: %+v", record.Epoch, record.Eval)
		}
		if record.Eval.RetrievalMAPAt100 <= 0 || record.Eval.RetrievalRecallAt100 <= 0 {
			t.Fatalf("epoch %d retrieval MAP/recall missing: %+v", record.Epoch, record.Eval)
		}
		if record.Eval.RetrievalEval == nil {
			t.Fatalf("epoch %d retrieval detail missing: %+v", record.Epoch, record.Eval)
		}
	}
}
