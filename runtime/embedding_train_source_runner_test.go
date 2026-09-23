package eosruntime

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestScoreSpectrumSourceBatchCapZeroMatchesRowBatching(t *testing.T) {
	counts := []int{2, 3, 1, 4, 2}
	order := []int{4, 0, 3, 1, 2}
	withoutCap, withoutCapPairs := scoreSpectrumBatchWorkForOrder(order, func(index int) int {
		return counts[index]
	}, 2, 0)
	withExplicitZero, withExplicitZeroPairs := scoreSpectrumBatchWorkForOrder(order, func(index int) int {
		return counts[index]
	}, 2, 0)
	if withoutCap != withExplicitZero || withoutCapPairs != withExplicitZeroPairs {
		t.Fatalf("cap=0 work = %d/%d, explicit zero = %d/%d", withoutCap, withoutCapPairs, withExplicitZero, withExplicitZeroPairs)
	}
	if withoutCap != 3 || withoutCapPairs != 12 {
		t.Fatalf("cap=0 work = %d/%d, want 3 batches/12 candidates", withoutCap, withoutCapPairs)
	}
}

func TestScoreSpectrumSourceBatchCapSplitsRowsWithoutTruncation(t *testing.T) {
	counts := []int{2, 3, 4, 2}
	order := []int{0, 1, 2, 3}
	spans, pairs := scoreSpectrumBatchSpans(order, func(index int) int { return counts[index] }, 4, 5)
	want := []scoreSpectrumBatchSpan{
		{start: 0, end: 2, pairs: 5, candidates: 5},
		{start: 2, end: 3, pairs: 4, candidates: 4},
		{start: 3, end: 4, pairs: 2, candidates: 2},
	}
	if !reflect.DeepEqual(spans, want) {
		t.Fatalf("spans = %+v, want %+v", spans, want)
	}
	if pairs != 11 {
		t.Fatalf("pairs = %d, want 11", pairs)
	}
	for i, span := range spans {
		var got int
		for _, index := range order[span.start:span.end] {
			got += counts[index]
		}
		if got != int(span.candidates) {
			t.Fatalf("span %d candidate count = %d, metadata = %d", i, got, span.candidates)
		}
	}
}

func TestScoreSpectrumSourceBatchCapKeepsOversizedRowAtomic(t *testing.T) {
	counts := []int{7, 2, 5, 1}
	spans, pairs := scoreSpectrumBatchSpans([]int{0, 1, 2, 3}, func(index int) int { return counts[index] }, 4, 5)
	want := []scoreSpectrumBatchSpan{
		{start: 0, end: 1, pairs: 7, candidates: 7},
		{start: 1, end: 2, pairs: 2, candidates: 2},
		{start: 2, end: 3, pairs: 5, candidates: 5},
		{start: 3, end: 4, pairs: 1, candidates: 1},
	}
	if !reflect.DeepEqual(spans, want) {
		t.Fatalf("oversized-row spans = %+v, want %+v", spans, want)
	}
	if pairs != 15 {
		t.Fatalf("oversized-row pairs = %d, want 15", pairs)
	}
}

func TestScoreSpectrumSourceBatchShufflePlanIsDeterministic(t *testing.T) {
	counts := []int{2, 7, 1, 4, 3, 6, 2}
	makePlan := func() ([]int, []scoreSpectrumBatchSpan, int64) {
		order := make([]int, len(counts))
		for i := range order {
			order[i] = i
		}
		rng := rand.New(rand.NewSource(41))
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		spans, pairs := scoreSpectrumBatchSpans(order, func(index int) int { return counts[index] }, 3, 8)
		return order, spans, pairs
	}
	order1, spans1, pairs1 := makePlan()
	order2, spans2, pairs2 := makePlan()
	if !reflect.DeepEqual(order1, order2) || !reflect.DeepEqual(spans1, spans2) || pairs1 != pairs2 {
		t.Fatalf("shuffle plan was not deterministic: first=%v/%+v/%d second=%v/%+v/%d", order1, spans1, pairs1, order2, spans2, pairs2)
	}
}

func TestFitScoreSpectrumSourceExactCapAccounting(t *testing.T) {
	source, err := NewEmbeddingScoreSpectrumSliceSource(tinyEmbeddingScoreSpectrumDataset())
	if err != nil {
		t.Fatalf("new score-spectrum source: %v", err)
	}
	defer source.Close()
	trainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	var progress []EmbeddingTrainProgress
	summary, err := trainer.FitScoreSpectrumSource(source, nil, EmbeddingTrainRunConfig{
		Epochs:                          1,
		BatchSize:                       2,
		ScoreSpectrumMaxBatchCandidates: 2,
		Shuffle:                         false,
		ProgressEverySteps:              1,
		Progress:                        func(update EmbeddingTrainProgress) { progress = append(progress, update) },
	})
	if err != nil {
		t.Fatalf("fit score-spectrum source: %v", err)
	}
	if summary.StepsRun != 2 || summary.Workload.TrainBatchesPerEpoch != 2 {
		t.Fatalf("steps/batches = %d/%d, want 2/2", summary.StepsRun, summary.Workload.TrainBatchesPerEpoch)
	}
	if summary.Workload.PlannedTrainPairs != 4 || summary.Workload.ActualTrainPairs != 4 {
		t.Fatalf("planned/actual train pairs = %d/%d, want 4/4", summary.Workload.PlannedTrainPairs, summary.Workload.ActualTrainPairs)
	}
	if summary.Workload.ActualTrainExamples != 2 || summary.Workload.ActualTotalExamples != 2 {
		t.Fatalf("actual train/total examples = %d/%d, want 2/2", summary.Workload.ActualTrainExamples, summary.Workload.ActualTotalExamples)
	}
	if len(progress) != 3 {
		t.Fatalf("progress updates = %d, want train_start plus two completed batches", len(progress))
	}
	if progress[0].Phase != "train_start" || progress[0].Architecture != EmbeddingArchitectureLegacyV1 ||
		progress[0].Route != "legacy_host" || progress[0].Branch != "legacy_host" ||
		progress[0].CompactBackend != "host" || progress[0].ResidentRequested || progress[0].CandidateCap != 2 ||
		progress[0].FirstSpanRows != 1 || progress[0].FirstSpanCandidates != 2 {
		t.Fatalf("pre-step route diagnostics = %+v, want legacy host route and first span metadata", progress[0])
	}
	if progress[1].Batches != 2 || progress[2].Batches != 2 || progress[1].BatchPairs != 2 || progress[2].BatchPairs != 2 {
		t.Fatalf("progress batches/pairs = %+v, want two batches of two candidates", progress)
	}
}

func TestFitScoreSpectrumSourceCapZeroMatchesSliceCompatibility(t *testing.T) {
	trainSet := tinyEmbeddingScoreSpectrumDataset()
	cfg := EmbeddingTrainRunConfig{Epochs: 1, BatchSize: 2, Shuffle: false, Seed: 9}
	wantTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	want, err := wantTrainer.FitScoreSpectrum(trainSet, nil, cfg)
	if err != nil {
		t.Fatalf("slice fit: %v", err)
	}
	source, err := NewEmbeddingScoreSpectrumSliceSource(trainSet)
	if err != nil {
		t.Fatalf("new source: %v", err)
	}
	defer source.Close()
	gotTrainer := newTinyTrainable3DEmbeddingTrainer(t, 0.05)
	got, err := gotTrainer.FitScoreSpectrumSource(source, nil, cfg)
	if err != nil {
		t.Fatalf("source fit: %v", err)
	}
	if got.StepsRun != want.StepsRun || got.Workload.TrainBatchesPerEpoch != want.Workload.TrainBatchesPerEpoch || got.Workload.PlannedTrainPairs != want.Workload.PlannedTrainPairs || got.Workload.ActualTrainPairs != want.Workload.ActualTrainPairs || got.Workload.ActualTrainExamples != want.Workload.ActualTrainExamples {
		t.Fatalf("source compatibility workload = %+v, slice = %+v", got.Workload, want.Workload)
	}
	if got.FinalTrain != want.FinalTrain {
		t.Fatalf("source final metrics = %+v, slice = %+v", got.FinalTrain, want.FinalTrain)
	}
}
