package eosruntime

import (
	"math"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCompactForwardPackedStateReconstructsNativeState(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	forward := trainer.prepareCompactForwardWeights()
	tokens := [][]int32{
		{1, 2, 3},
		{3, 2, 4},
	}
	rawMasks := [][]int32{
		{1, 1, 0},
		{1, 0, 1},
	}
	roles := []int32{trainer.queryRoleIndex(), trainer.documentRoleIndex()}
	seqs, masks := encodeCompactPackedTestSequences(t, trainer, forward, tokens, rawMasks, roles)
	shape := compactPackedTestShape(trainer, len(seqs), len(tokens[0]), forward)
	finalRows := compactPackedTestFinalRows(t, trainer, forward, seqs, shape)

	result, err := packCompactForwardEncodedSequences(shape, seqs, finalRows)
	if err != nil {
		t.Fatalf("pack compact forward state: %v", err)
	}
	reconstructed, err := reconstructCompactForwardPackedState(shape, result, tokens, masks, roles)
	if err != nil {
		t.Fatalf("reconstruct compact forward state: %v", err)
	}
	assertCompactPackedSequencesEqual(t, reconstructed, seqs)
	if got, want := reconstructed[0].tokens[0], tokens[0][0]; got != want {
		t.Fatalf("reconstructed token = %d, want %d", got, want)
	}
	tokens[0][0] = 99
	masks[0][0] = 0
	if got := reconstructed[0].tokens[0]; got != 1 {
		t.Fatalf("reconstructed tokens aliased caller tokens, got %d", got)
	}
	if got := reconstructed[0].layers[0].mask[0]; got != 1 {
		t.Fatalf("reconstructed masks aliased caller masks, got %d", got)
	}

	span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, 0, compactForwardPackedFieldAttnQ))
	result.Data[span.Offset] = 1234
	if got := reconstructed[0].layers[0].attnQ[0]; got == 1234 {
		t.Fatalf("reconstructed attnQ aliased packed backing after result mutation")
	}
	finalSpan := compactForwardPackedSpanByName(result.Layout, compactForwardPackedSequenceSpanName(1, compactForwardPackedFinalPooled))
	result.Data[finalSpan.Offset] = -777
	if got := reconstructed[1].pooled[0]; got == -777 {
		t.Fatalf("reconstructed pooled aliased packed backing after result mutation")
	}
}

func TestCompactForwardPackedStateReconstructionSurvivesReusedBacking(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	forward := trainer.prepareCompactForwardWeights()
	tokens := [][]int32{{1, 2, 3}}
	rawMasks := [][]int32{{1, 1, 0}}
	roles := []int32{trainer.rawRoleIndex()}
	seqs, masks := encodeCompactPackedTestSequences(t, trainer, forward, tokens, rawMasks, roles)
	shape := compactPackedTestShape(trainer, len(seqs), len(tokens[0]), forward)
	finalRows := compactPackedTestFinalRows(t, trainer, forward, seqs, shape)

	result, err := packCompactForwardEncodedSequences(shape, seqs, finalRows)
	if err != nil {
		t.Fatalf("pack compact forward state: %v", err)
	}
	reconstructed, err := reconstructCompactForwardPackedState(shape, result, tokens, masks, roles)
	if err != nil {
		t.Fatalf("reconstruct compact forward state: %v", err)
	}
	before := append([]float32(nil), reconstructed[0].layers[0].hidden...)
	for i := range result.Data {
		result.Data[i] = float32(i + 1000)
	}
	assertCompactPackedFloat32SliceEqual(t, "detached hidden", reconstructed[0].layers[0].hidden, before)
}

func TestCompactForwardPackedStatePacksMaskActiveCountsForEveryLayer(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	forward := trainer.prepareCompactForwardWeights()
	tokens := [][]int32{{1, 2, 3}}
	rawMasks := [][]int32{{1, 0, 1}}
	roles := []int32{trainer.rawRoleIndex()}
	seqs, _ := encodeCompactPackedTestSequences(t, trainer, forward, tokens, rawMasks, roles)
	shape := compactPackedTestShape(trainer, len(seqs), len(tokens[0]), forward)
	finalRows := compactPackedTestFinalRows(t, trainer, forward, seqs, shape)

	result, err := packCompactForwardEncodedSequences(shape, seqs, finalRows)
	if err != nil {
		t.Fatalf("pack compact forward state: %v", err)
	}
	for layer := 0; layer < shape.Layers; layer++ {
		span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, layer, compactForwardPackedFieldActiveCount))
		if got, want := result.Data[span.Offset], float32(2); got != want {
			t.Fatalf("layer %d activeCount = %v, want %v", layer, got, want)
		}
	}
}

func TestCompactForwardPackedStateRejectsMalformedData(t *testing.T) {
	shape := backend.CompactForwardShape{
		Batch:     1,
		Tokens:    2,
		ModelDim:  4,
		FFNDim:    6,
		Heads:     2,
		HeadDim:   2,
		Layers:    1,
		OutputDim: 4,
	}
	layout, err := buildCompactForwardPackedStateLayout(shape)
	if err != nil {
		t.Fatalf("build layout: %v", err)
	}
	total := layout.Spans[len(layout.Spans)-1].Offset + layout.Spans[len(layout.Spans)-1].Len
	data := make([]float32, total)
	data[0] = 2
	tokens := [][]int32{{1, 2}}
	masks := [][]int32{{1, 1}}
	roles := []int32{0}

	cases := []struct {
		name   string
		mutate func(*backend.CompactForwardResult)
	}{
		{
			name: "unsupported version",
			mutate: func(result *backend.CompactForwardResult) {
				result.Layout.Version = backend.CompactForwardPackedStateVersion + 1
			},
		},
		{
			name: "wrong span offset",
			mutate: func(result *backend.CompactForwardResult) {
				result.Layout.Spans[1].Offset++
			},
		},
		{
			name: "wrong data length",
			mutate: func(result *backend.CompactForwardResult) {
				result.Data = result.Data[:len(result.Data)-1]
			},
		},
		{
			name: "nan active count",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, 0, compactForwardPackedFieldActiveCount))
				result.Data[span.Offset] = float32(math.NaN())
			},
		},
		{
			name: "non-integer active count",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, 0, compactForwardPackedFieldActiveCount))
				result.Data[span.Offset] = 1.5
			},
		},
		{
			name: "wrong token length",
			mutate: func(result *backend.CompactForwardResult) {
				result.Layout.Shape.Tokens = 3
			},
		},
		{
			name: "active count less than mask",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, 0, compactForwardPackedFieldActiveCount))
				result.Data[span.Offset] = 1
			},
		},
		{
			name: "wrong final normalized",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedSequenceSpanName(0, compactForwardPackedFinalNormalized))
				result.Data[span.Offset] = 1
			},
		},
		{
			name: "wrong final output rows",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedSequenceSpanName(0, compactForwardPackedFinalOutputRows))
				result.Data[span.Offset] = 1
			},
		},
		{
			name: "wrong final pooled",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedSequenceSpanName(0, compactForwardPackedFinalPooled))
				result.Data[span.Offset] = 1
			},
		},
		{
			name: "wrong final layer pooled duplicate",
			mutate: func(result *backend.CompactForwardResult) {
				span := compactForwardPackedSpanByName(result.Layout, compactForwardPackedLayerSpanName(0, 0, compactForwardPackedFieldPooled))
				result.Data[span.Offset] = 1
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := backend.CompactForwardResult{
				Layout: cloneCompactForwardPackedLayout(layout),
				Data:   append([]float32(nil), data...),
			}
			tc.mutate(&result)
			if _, err := reconstructCompactForwardPackedState(shape, result, tokens, masks, roles); err == nil {
				t.Fatalf("reconstruct compact forward packed state succeeded, want error")
			}
		})
	}
	if _, err := reconstructCompactForwardPackedState(shape, backend.CompactForwardResult{
		Layout: cloneCompactForwardPackedLayout(layout),
		Data:   append([]float32(nil), data...),
	}, tokens, [][]int32{{0, 0}}, roles); err == nil {
		t.Fatalf("reconstruct compact forward packed state succeeded for zero active mask")
	}

	if _, err := buildCompactForwardPackedStateLayout(backend.CompactForwardShape{
		Batch:     1,
		Tokens:    1,
		ModelDim:  3,
		FFNDim:    4,
		Heads:     2,
		HeadDim:   2,
		Layers:    1,
		OutputDim: 3,
	}); err == nil {
		t.Fatalf("build layout succeeded for malformed head layout")
	}
	if _, err := buildCompactForwardPackedStateLayout(backend.CompactForwardShape{
		Batch:     int(^uint(0) >> 1),
		Tokens:    int(^uint(0) >> 1),
		ModelDim:  int(^uint(0) >> 1),
		FFNDim:    1,
		Heads:     1,
		HeadDim:   int(^uint(0) >> 1),
		Layers:    1,
		OutputDim: 1,
	}); err == nil {
		t.Fatalf("build layout succeeded for overflowing shape")
	}
	if _, err := buildCompactForwardPackedStateLayout(backend.CompactForwardShape{
		Batch:               1,
		Tokens:              1,
		ModelDim:            3,
		FFNDim:              4,
		Heads:               1,
		HeadDim:             3,
		Layers:              1,
		OutputDim:           2,
		HasOutputProjection: false,
	}); err == nil {
		t.Fatalf("build layout succeeded for non-projected output width mismatch")
	}
}

func TestCompactForwardPackedStateRequiresProjectionOutputRows(t *testing.T) {
	trainer := newCompactEmbeddingTrainerForTest(t, 3)
	forward := trainer.prepareCompactForwardWeights()
	tokens := [][]int32{{1, 2, 3}}
	masks := [][]int32{{1, 1, 1}}
	roles := []int32{trainer.rawRoleIndex()}
	seqs, _ := encodeCompactPackedTestSequences(t, trainer, forward, tokens, masks, roles)
	shape := compactPackedTestShape(trainer, 1, 3, forward)

	if _, err := packCompactForwardEncodedSequences(shape, seqs, nil); err == nil {
		t.Fatalf("pack compact forward state succeeded without projected final rows")
	}
	finalRows := compactPackedTestFinalRows(t, trainer, forward, seqs, shape)
	if _, err := packCompactForwardEncodedSequences(shape, seqs, finalRows); err != nil {
		t.Fatalf("pack compact forward state with projected final rows: %v", err)
	}
}

func encodeCompactPackedTestSequences(t *testing.T, trainer *EmbeddingTrainer, forward *compactEmbeddingForwardWeights, tokens, masks [][]int32, roles []int32) ([]*embeddingEncodedSequence, [][]int32) {
	t.Helper()
	seqs := make([]*embeddingEncodedSequence, len(tokens))
	preparedMasks := make([][]int32, len(tokens))
	for i := range tokens {
		mask, err := trainer.prepareMask(tokens[i], masks[i])
		if err != nil {
			t.Fatalf("prepare mask %d: %v", i, err)
		}
		preparedMasks[i] = mask
		seq, err := trainer.encodeCompactSequence(tokens[i], mask, roles[i], forward)
		if err != nil {
			t.Fatalf("encode compact sequence %d: %v", i, err)
		}
		seqs[i] = seq
	}
	return seqs, preparedMasks
}

func compactPackedTestShape(trainer *EmbeddingTrainer, batch, tokens int, forward *compactEmbeddingForwardWeights) backend.CompactForwardShape {
	return backend.CompactForwardShape{
		Batch:               batch,
		Tokens:              tokens,
		ModelDim:            trainer.manifest.ModelDim,
		FFNDim:              trainer.manifest.FFNDim,
		Heads:               trainer.manifest.AttentionHeads,
		HeadDim:             trainer.manifest.HeadDim,
		Layers:              len(forward.layers),
		OutputDim:           compactOutputWidth(trainer.manifest.ModelDim, forward.outputProjection),
		HasOutputProjection: forward.outputProjection != nil,
	}
}

func compactPackedTestFinalRows(t *testing.T, trainer *EmbeddingTrainer, forward *compactEmbeddingForwardWeights, seqs []*embeddingEncodedSequence, shape backend.CompactForwardShape) [][]float32 {
	t.Helper()
	rows := make([][]float32, len(seqs))
	for i, seq := range seqs {
		out, err := trainer.compactFinalOutputRows(seq.finalLayer().projected, shape.Tokens, shape.ModelDim, forward.outputProjection)
		if err != nil {
			t.Fatalf("final output rows %d: %v", i, err)
		}
		rows[i] = out
	}
	return rows
}

func assertCompactPackedSequencesEqual(t *testing.T, got, want []*embeddingEncodedSequence) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("sequence count = %d, want %d", len(got), len(want))
	}
	for seqIndex := range want {
		if got[seqIndex].role != want[seqIndex].role {
			t.Fatalf("sequence %d role = %d, want %d", seqIndex, got[seqIndex].role, want[seqIndex].role)
		}
		assertCompactPackedInt32SliceEqual(t, "tokens", got[seqIndex].tokens, want[seqIndex].tokens)
		assertCompactPackedFloat32SliceEqual(t, "pooled", got[seqIndex].pooled, want[seqIndex].pooled)
		if len(got[seqIndex].layers) != len(want[seqIndex].layers) {
			t.Fatalf("sequence %d layers = %d, want %d", seqIndex, len(got[seqIndex].layers), len(want[seqIndex].layers))
		}
		for layerIndex := range want[seqIndex].layers {
			assertCompactPackedLayerEqual(t, seqIndex, layerIndex, got[seqIndex].layers[layerIndex], want[seqIndex].layers[layerIndex])
		}
	}
}

func assertCompactPackedLayerEqual(t *testing.T, seqIndex, layerIndex int, got, want *embeddingSequenceState) {
	t.Helper()
	wantActive := want.activeCount
	if wantActive == 0 {
		wantActive = compactForwardMaskActiveCount(want.mask)
	}
	if got.activeCount != wantActive {
		t.Fatalf("sequence %d layer %d activeCount = %d, want %d", seqIndex, layerIndex, got.activeCount, wantActive)
	}
	prefix := compactForwardPackedLayerSpanName(seqIndex, layerIndex, "")
	assertCompactPackedInt32SliceEqual(t, prefix+"tokens", got.tokens, want.tokens)
	assertCompactPackedInt32SliceEqual(t, prefix+"mask", got.mask, want.mask)
	assertCompactPackedFloat32SliceEqual(t, prefix+"input", got.input, want.input)
	assertCompactPackedFloat32SliceEqual(t, prefix+"hidden", got.hidden, want.hidden)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnQ", got.attnQ, want.attnQ)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnK", got.attnK, want.attnK)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnV", got.attnV, want.attnV)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnScores", got.attnScores, want.attnScores)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnMixed", got.attnMixed, want.attnMixed)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnOutput", got.attnOutput, want.attnOutput)
	assertCompactPackedFloat32SliceEqual(t, prefix+"attnResidual", got.attnResidual, want.attnResidual)
	assertCompactPackedFloat32SliceEqual(t, prefix+"ffnHidden", got.ffnHidden, want.ffnHidden)
	assertCompactPackedFloat32SliceEqual(t, prefix+"activated", got.activated, want.activated)
	assertCompactPackedFloat32SliceEqual(t, prefix+"ffnOutput", got.ffnOutput, want.ffnOutput)
	assertCompactPackedFloat32SliceEqual(t, prefix+"ffnResidual", got.ffnResidual, want.ffnResidual)
	assertCompactPackedFloat32SliceEqual(t, prefix+"projected", got.projected, want.projected)
	assertCompactPackedFloat32SliceEqual(t, prefix+"normalized", got.normalized, want.normalized)
	assertCompactPackedFloat32SliceEqual(t, prefix+"pooled", got.pooled, want.pooled)
}

func assertCompactPackedFloat32SliceEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func assertCompactPackedInt32SliceEqual(t *testing.T, name string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %d, want %d", name, i, got[i], want[i])
		}
	}
}

func cloneCompactForwardPackedLayout(layout backend.CompactForwardPackedStateLayout) backend.CompactForwardPackedStateLayout {
	layout.Spans = append([]backend.CompactForwardStateSpan(nil), layout.Spans...)
	return layout
}
