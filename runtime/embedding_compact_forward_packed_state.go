package eosruntime

import (
	"fmt"
	"math"

	"m31labs.dev/eos/runtime/backend"
)

const (
	compactForwardPackedFieldInput        = "input"
	compactForwardPackedFieldHidden       = "hidden"
	compactForwardPackedFieldAttnQ        = "attnQ"
	compactForwardPackedFieldAttnK        = "attnK"
	compactForwardPackedFieldAttnV        = "attnV"
	compactForwardPackedFieldAttnScores   = "attnScores"
	compactForwardPackedFieldAttnMixed    = "attnMixed"
	compactForwardPackedFieldAttnOutput   = "attnOutput"
	compactForwardPackedFieldAttnResidual = "attnResidual"
	compactForwardPackedFieldFFNHidden    = "ffnHidden"
	compactForwardPackedFieldActivated    = "activated"
	compactForwardPackedFieldFFNOutput    = "ffnOutput"
	compactForwardPackedFieldFFNResidual  = "ffnResidual"
	compactForwardPackedFieldProjected    = "projected"
	compactForwardPackedFieldNormalized   = "normalized"
	compactForwardPackedFieldPooled       = "pooled"
	compactForwardPackedFieldActiveCount  = "activeCount"
	compactForwardPackedFinalNormalized   = "final.normalized"
	compactForwardPackedFinalOutputRows   = "final.outputRows"
	compactForwardPackedFinalPooled       = "final.pooled"
)

var compactForwardPackedLayerFields = []struct {
	name string
	kind compactForwardPackedFieldKind
}{
	{compactForwardPackedFieldActiveCount, compactForwardPackedFieldScalar},
	{compactForwardPackedFieldInput, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldHidden, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnQ, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnK, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnV, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnScores, compactForwardPackedFieldAttentionScores},
	{compactForwardPackedFieldAttnMixed, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnOutput, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldAttnResidual, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldFFNHidden, compactForwardPackedFieldFFNRows},
	{compactForwardPackedFieldActivated, compactForwardPackedFieldFFNRows},
	{compactForwardPackedFieldFFNOutput, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldFFNResidual, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldProjected, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldNormalized, compactForwardPackedFieldModelRows},
	{compactForwardPackedFieldPooled, compactForwardPackedFieldLayerPooled},
}

type compactForwardPackedFieldKind int

const (
	compactForwardPackedFieldScalar compactForwardPackedFieldKind = iota
	compactForwardPackedFieldModelRows
	compactForwardPackedFieldFFNRows
	compactForwardPackedFieldAttentionScores
	compactForwardPackedFieldLayerPooled
)

func buildCompactForwardPackedStateLayout(shape backend.CompactForwardShape) (backend.CompactForwardPackedStateLayout, error) {
	if err := validateCompactForwardPackedShape(shape); err != nil {
		return backend.CompactForwardPackedStateLayout{}, err
	}
	layout := backend.CompactForwardPackedStateLayout{
		Version: backend.CompactForwardPackedStateVersion,
		Shape:   shape,
	}
	offset := 0
	appendSpan := func(name string, length int) error {
		if length < 0 {
			return fmt.Errorf("compact forward packed span %q has negative length %d", name, length)
		}
		next, ok := checkedAddInt(offset, length)
		if !ok {
			return fmt.Errorf("compact forward packed span %q overflows offset calculation", name)
		}
		layout.Spans = append(layout.Spans, backend.CompactForwardStateSpan{Name: name, Offset: offset, Len: length})
		offset = next
		return nil
	}
	for seq := 0; seq < shape.Batch; seq++ {
		for layer := 0; layer < shape.Layers; layer++ {
			for _, field := range compactForwardPackedLayerFields {
				if err := appendSpan(compactForwardPackedLayerSpanName(seq, layer, field.name), compactForwardPackedLayerFieldLen(shape, layer, field.kind)); err != nil {
					return backend.CompactForwardPackedStateLayout{}, err
				}
			}
		}
		if err := appendSpan(compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalNormalized), compactForwardPackedModelRowsLen(shape)); err != nil {
			return backend.CompactForwardPackedStateLayout{}, err
		}
		if err := appendSpan(compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalOutputRows), compactForwardPackedOutputRowsLen(shape)); err != nil {
			return backend.CompactForwardPackedStateLayout{}, err
		}
		if err := appendSpan(compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalPooled), shape.OutputDim); err != nil {
			return backend.CompactForwardPackedStateLayout{}, err
		}
	}
	return layout, nil
}

func validateCompactForwardPackedState(layout backend.CompactForwardPackedStateLayout, data []float32) error {
	expected, err := buildCompactForwardPackedStateLayout(layout.Shape)
	if err != nil {
		return err
	}
	if layout.Version != backend.CompactForwardPackedStateVersion {
		return fmt.Errorf("compact forward packed state version %d is unsupported", layout.Version)
	}
	if len(layout.Spans) != len(expected.Spans) {
		return fmt.Errorf("compact forward packed span count %d, want %d", len(layout.Spans), len(expected.Spans))
	}
	for i := range expected.Spans {
		got := layout.Spans[i]
		want := expected.Spans[i]
		if got != want {
			return fmt.Errorf("compact forward packed span %d = %+v, want %+v", i, got, want)
		}
		if got.Offset < 0 || got.Len < 0 {
			return fmt.Errorf("compact forward packed span %q has negative offset/length", got.Name)
		}
		end, ok := checkedAddInt(got.Offset, got.Len)
		if !ok || end > len(data) {
			return fmt.Errorf("compact forward packed span %q [%d:%d] exceeds data length %d", got.Name, got.Offset, end, len(data))
		}
	}
	total := 0
	if len(expected.Spans) > 0 {
		last := expected.Spans[len(expected.Spans)-1]
		var ok bool
		total, ok = checkedAddInt(last.Offset, last.Len)
		if !ok {
			return fmt.Errorf("compact forward packed total length overflows")
		}
	}
	if len(data) != total {
		return fmt.Errorf("compact forward packed data length %d, want %d", len(data), total)
	}
	for seq := 0; seq < layout.Shape.Batch; seq++ {
		for layer := 0; layer < layout.Shape.Layers; layer++ {
			name := compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldActiveCount)
			value := data[compactForwardPackedSpanByName(expected, name).Offset]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("compact forward packed activeCount for sequence %d layer %d is not finite", seq, layer)
			}
			active := int(value)
			if float32(active) != value || active < 0 || active > layout.Shape.Tokens {
				return fmt.Errorf("compact forward packed activeCount for sequence %d layer %d = %v, want integer in [0,%d]", seq, layer, value, layout.Shape.Tokens)
			}
		}
	}
	return nil
}

func reconstructCompactForwardPackedState(expectedShape backend.CompactForwardShape, result backend.CompactForwardResult, tokens, masks [][]int32, roles []int32) ([]*embeddingEncodedSequence, error) {
	if err := validateCompactForwardPackedShape(expectedShape); err != nil {
		return nil, err
	}
	if result.Layout.Shape != expectedShape {
		return nil, fmt.Errorf("compact forward packed result shape = %+v, want %+v", result.Layout.Shape, expectedShape)
	}
	if err := validateCompactForwardPackedState(result.Layout, result.Data); err != nil {
		return nil, err
	}
	shape := expectedShape
	if err := validateCompactForwardPackedInputs(shape, tokens, masks, roles); err != nil {
		return nil, err
	}
	if err := validateCompactForwardPackedActiveCounts(result.Layout, result.Data, masks); err != nil {
		return nil, err
	}
	if err := validateCompactForwardPackedFinalSpans(result.Layout, result.Data, masks); err != nil {
		return nil, err
	}
	// One O(len(Data)) copy detaches host backward state from accelerator result
	// buffers. The ABI still requires producers to transfer ownership without
	// reuse; the copy is a defensive host-lifetime boundary.
	data := append([]float32(nil), result.Data...)
	spans := compactForwardPackedSpanMap(result.Layout)
	view := func(name string) []float32 {
		span := spans[name]
		return data[span.Offset : span.Offset+span.Len]
	}
	out := make([]*embeddingEncodedSequence, shape.Batch)
	for seq := 0; seq < shape.Batch; seq++ {
		active := compactForwardMaskActiveCount(masks[seq])
		encoded := &embeddingEncodedSequence{
			layers: make([]*embeddingSequenceState, shape.Layers),
			tokens: append([]int32(nil), tokens[seq]...),
			role:   roles[seq],
		}
		for layer := 0; layer < shape.Layers; layer++ {
			state := &embeddingSequenceState{
				tokens:       append([]int32(nil), tokens[seq]...),
				mask:         append([]int32(nil), masks[seq]...),
				activeCount:  active,
				input:        view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldInput)),
				hidden:       view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldHidden)),
				attnQ:        view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnQ)),
				attnK:        view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnK)),
				attnV:        view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnV)),
				attnScores:   view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnScores)),
				attnMixed:    view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnMixed)),
				attnOutput:   view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnOutput)),
				attnResidual: view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldAttnResidual)),
				ffnHidden:    view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldFFNHidden)),
				activated:    view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldActivated)),
				ffnOutput:    view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldFFNOutput)),
				ffnResidual:  view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldFFNResidual)),
				projected:    view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldProjected)),
				normalized:   view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldNormalized)),
				pooled:       view(compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldPooled)),
			}
			encoded.layers[layer] = state
		}
		encoded.pooled = view(compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalPooled))
		out[seq] = encoded
	}
	return out, nil
}

func packCompactForwardEncodedSequences(shape backend.CompactForwardShape, seqs []*embeddingEncodedSequence, finalOutputRows [][]float32) (backend.CompactForwardResult, error) {
	layout, err := buildCompactForwardPackedStateLayout(shape)
	if err != nil {
		return backend.CompactForwardResult{}, err
	}
	if len(seqs) != shape.Batch {
		return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence count %d, want %d", len(seqs), shape.Batch)
	}
	if finalOutputRows != nil && len(finalOutputRows) != shape.Batch {
		return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed final output row count %d, want %d", len(finalOutputRows), shape.Batch)
	}
	total := 0
	if len(layout.Spans) > 0 {
		last := layout.Spans[len(layout.Spans)-1]
		total = last.Offset + last.Len
	}
	data := make([]float32, total)
	spans := compactForwardPackedSpanMap(layout)
	put := func(name string, values []float32) error {
		span, ok := spans[name]
		if !ok {
			return fmt.Errorf("compact forward packed span %q is not present", name)
		}
		if len(values) != span.Len {
			return fmt.Errorf("compact forward packed span %q length %d, want %d", name, len(values), span.Len)
		}
		copy(data[span.Offset:span.Offset+span.Len], values)
		return nil
	}
	putScalar := func(name string, value int) error {
		span, ok := spans[name]
		if !ok {
			return fmt.Errorf("compact forward packed span %q is not present", name)
		}
		if span.Len != 1 {
			return fmt.Errorf("compact forward packed span %q length %d, want 1", name, span.Len)
		}
		data[span.Offset] = float32(value)
		return nil
	}
	for seqIndex, seq := range seqs {
		if seq == nil || len(seq.tokens) != shape.Tokens || len(seq.layers) != shape.Layers {
			return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d shape mismatch", seqIndex)
		}
		sequenceActive := -1
		for layerIndex, state := range seq.layers {
			if state == nil {
				return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d layer %d is nil", seqIndex, layerIndex)
			}
			if len(state.mask) != shape.Tokens {
				return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d layer %d mask length %d, want %d", seqIndex, layerIndex, len(state.mask), shape.Tokens)
			}
			active := compactForwardMaskActiveCount(state.mask)
			if active == 0 {
				return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d layer %d mask selects zero tokens", seqIndex, layerIndex)
			}
			if sequenceActive < 0 {
				sequenceActive = active
			} else if active != sequenceActive {
				return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d layer %d active mask count %d, want %d", seqIndex, layerIndex, active, sequenceActive)
			}
			if state.activeCount != 0 && state.activeCount != active {
				return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d layer %d activeCount %d, want mask active count %d", seqIndex, layerIndex, state.activeCount, active)
			}
			if err := putScalar(compactForwardPackedLayerSpanName(seqIndex, layerIndex, compactForwardPackedFieldActiveCount), active); err != nil {
				return backend.CompactForwardResult{}, err
			}
			for _, field := range []struct {
				name   string
				values []float32
			}{
				{compactForwardPackedFieldInput, state.input},
				{compactForwardPackedFieldHidden, state.hidden},
				{compactForwardPackedFieldAttnQ, state.attnQ},
				{compactForwardPackedFieldAttnK, state.attnK},
				{compactForwardPackedFieldAttnV, state.attnV},
				{compactForwardPackedFieldAttnScores, state.attnScores},
				{compactForwardPackedFieldAttnMixed, state.attnMixed},
				{compactForwardPackedFieldAttnOutput, state.attnOutput},
				{compactForwardPackedFieldAttnResidual, state.attnResidual},
				{compactForwardPackedFieldFFNHidden, state.ffnHidden},
				{compactForwardPackedFieldActivated, state.activated},
				{compactForwardPackedFieldFFNOutput, state.ffnOutput},
				{compactForwardPackedFieldFFNResidual, state.ffnResidual},
				{compactForwardPackedFieldProjected, state.projected},
				{compactForwardPackedFieldNormalized, state.normalized},
				{compactForwardPackedFieldPooled, state.pooled},
			} {
				if err := put(compactForwardPackedLayerSpanName(seqIndex, layerIndex, field.name), field.values); err != nil {
					return backend.CompactForwardResult{}, err
				}
			}
		}
		final := seq.finalLayer()
		finalNormalized, err := compactForwardNormalizeRows(final.projected, shape.Tokens, shape.ModelDim)
		if err != nil {
			return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d final normalized: %w", seqIndex, err)
		}
		if err := put(compactForwardPackedSequenceSpanName(seqIndex, compactForwardPackedFinalNormalized), finalNormalized); err != nil {
			return backend.CompactForwardResult{}, err
		}
		outputRows := finalOutputRowsForPackedSequence(shape, finalNormalized, finalOutputRows, seqIndex)
		if outputRows == nil {
			return backend.CompactForwardResult{}, fmt.Errorf("compact forward packed sequence %d final output rows are required when output projection is present", seqIndex)
		}
		if err := put(compactForwardPackedSequenceSpanName(seqIndex, compactForwardPackedFinalOutputRows), outputRows); err != nil {
			return backend.CompactForwardResult{}, err
		}
		if err := put(compactForwardPackedSequenceSpanName(seqIndex, compactForwardPackedFinalPooled), seq.pooled); err != nil {
			return backend.CompactForwardResult{}, err
		}
	}
	result := backend.CompactForwardResult{Layout: layout, Data: data}
	if err := validateCompactForwardPackedState(result.Layout, result.Data); err != nil {
		return backend.CompactForwardResult{}, err
	}
	packedMasks := make([][]int32, shape.Batch)
	for seqIndex, seq := range seqs {
		packedMasks[seqIndex] = seq.layers[0].mask
	}
	if err := validateCompactForwardPackedActiveCounts(result.Layout, result.Data, packedMasks); err != nil {
		return backend.CompactForwardResult{}, err
	}
	if err := validateCompactForwardPackedFinalSpans(result.Layout, result.Data, packedMasks); err != nil {
		return backend.CompactForwardResult{}, err
	}
	return result, nil
}

func finalOutputRowsForPackedSequence(shape backend.CompactForwardShape, finalNormalized []float32, finalOutputRows [][]float32, seqIndex int) []float32 {
	if finalOutputRows != nil {
		return finalOutputRows[seqIndex]
	}
	if !shape.HasOutputProjection && shape.OutputDim == shape.ModelDim {
		return finalNormalized
	}
	return nil
}

func compactForwardNormalizeRows(rows []float32, rowCount, width int) ([]float32, error) {
	if len(rows) != rowCount*width {
		return nil, fmt.Errorf("row data length %d, want %d", len(rows), rowCount*width)
	}
	out := make([]float32, len(rows))
	for row := 0; row < rowCount; row++ {
		base := row * width
		norm := vectorNorm(rows[base : base+width])
		if norm == 0 {
			copy(out[base:base+width], rows[base:base+width])
			continue
		}
		for col := 0; col < width; col++ {
			out[base+col] = rows[base+col] / norm
		}
	}
	return out, nil
}

func validateCompactForwardPackedShape(shape backend.CompactForwardShape) error {
	switch {
	case shape.Batch <= 0:
		return fmt.Errorf("compact forward packed batch must be positive")
	case shape.Tokens <= 0:
		return fmt.Errorf("compact forward packed tokens must be positive")
	case shape.ModelDim <= 0:
		return fmt.Errorf("compact forward packed model_dim must be positive")
	case shape.FFNDim <= 0:
		return fmt.Errorf("compact forward packed ffn_dim must be positive")
	case shape.Heads <= 0:
		return fmt.Errorf("compact forward packed heads must be positive")
	case shape.HeadDim <= 0:
		return fmt.Errorf("compact forward packed head_dim must be positive")
	case shape.Layers <= 0:
		return fmt.Errorf("compact forward packed layers must be positive")
	case shape.OutputDim <= 0:
		return fmt.Errorf("compact forward packed output_dim must be positive")
	}
	if !shape.HasOutputProjection && shape.OutputDim != shape.ModelDim {
		return fmt.Errorf("compact forward packed output_dim=%d must equal model_dim=%d without output projection", shape.OutputDim, shape.ModelDim)
	}
	headWidth := checkedMulInts(shape.Heads, shape.HeadDim)
	if headWidth < 0 || headWidth != shape.ModelDim {
		return fmt.Errorf("compact forward packed heads=%d head_dim=%d do not match model_dim=%d", shape.Heads, shape.HeadDim, shape.ModelDim)
	}
	modelRowsLen := compactForwardPackedModelRowsLen(shape)
	ffnRowsLen := compactForwardPackedFFNRowsLen(shape)
	attnScoresLen := compactForwardPackedAttentionScoresLen(shape)
	outputRowsLen := compactForwardPackedOutputRowsLen(shape)
	for _, length := range []int{modelRowsLen, ffnRowsLen, attnScoresLen, outputRowsLen} {
		if length < 0 {
			return fmt.Errorf("compact forward packed shape overflows length calculation")
		}
	}
	pooledLen := shape.ModelDim
	if shape.OutputDim > pooledLen {
		pooledLen = shape.OutputDim
	}
	perLayerLen := 0
	for _, length := range []int{
		1,
		modelRowsLen, modelRowsLen, modelRowsLen, modelRowsLen, modelRowsLen,
		attnScoresLen,
		modelRowsLen, modelRowsLen, modelRowsLen,
		ffnRowsLen, ffnRowsLen,
		modelRowsLen, modelRowsLen, modelRowsLen, modelRowsLen,
		pooledLen,
	} {
		var ok bool
		perLayerLen, ok = checkedAddInt(perLayerLen, length)
		if !ok {
			return fmt.Errorf("compact forward packed shape overflows per-layer length calculation")
		}
	}
	perSequenceLen, ok := checkedMulInt(perLayerLen, shape.Layers)
	if !ok {
		return fmt.Errorf("compact forward packed shape overflows per-sequence length calculation")
	}
	for _, length := range []int{modelRowsLen, outputRowsLen, shape.OutputDim} {
		perSequenceLen, ok = checkedAddInt(perSequenceLen, length)
		if !ok {
			return fmt.Errorf("compact forward packed shape overflows per-sequence length calculation")
		}
	}
	if _, ok := checkedMulInt(perSequenceLen, shape.Batch); !ok {
		return fmt.Errorf("compact forward packed shape overflows total length calculation")
	}
	return nil
}

func validateCompactForwardPackedInputs(shape backend.CompactForwardShape, tokens, masks [][]int32, roles []int32) error {
	if len(tokens) != shape.Batch {
		return fmt.Errorf("compact forward packed token batch %d, want %d", len(tokens), shape.Batch)
	}
	if len(masks) != shape.Batch {
		return fmt.Errorf("compact forward packed mask batch %d, want %d", len(masks), shape.Batch)
	}
	if len(roles) != shape.Batch {
		return fmt.Errorf("compact forward packed role batch %d, want %d", len(roles), shape.Batch)
	}
	for i := 0; i < shape.Batch; i++ {
		if len(tokens[i]) != shape.Tokens {
			return fmt.Errorf("compact forward packed tokens[%d] length %d, want %d", i, len(tokens[i]), shape.Tokens)
		}
		if len(masks[i]) != shape.Tokens {
			return fmt.Errorf("compact forward packed masks[%d] length %d, want %d", i, len(masks[i]), shape.Tokens)
		}
		if compactForwardMaskActiveCount(masks[i]) == 0 {
			return fmt.Errorf("compact forward packed masks[%d] select zero tokens", i)
		}
	}
	return nil
}

func validateCompactForwardPackedActiveCounts(layout backend.CompactForwardPackedStateLayout, data []float32, masks [][]int32) error {
	for seq := 0; seq < layout.Shape.Batch; seq++ {
		want := compactForwardMaskActiveCount(masks[seq])
		if want == 0 {
			return fmt.Errorf("compact forward packed masks[%d] select zero tokens", seq)
		}
		for layer := 0; layer < layout.Shape.Layers; layer++ {
			name := compactForwardPackedLayerSpanName(seq, layer, compactForwardPackedFieldActiveCount)
			span := compactForwardPackedSpanByName(layout, name)
			value := data[span.Offset]
			got := int(value)
			if got != want {
				return fmt.Errorf("compact forward packed activeCount for sequence %d layer %d = %d, want mask active count %d", seq, layer, got, want)
			}
		}
	}
	return nil
}

func validateCompactForwardPackedFinalSpans(layout backend.CompactForwardPackedStateLayout, data []float32, masks [][]int32) error {
	shape := layout.Shape
	for seq := 0; seq < shape.Batch; seq++ {
		finalProjected := compactForwardPackedSlice(layout, data, compactForwardPackedLayerSpanName(seq, shape.Layers-1, compactForwardPackedFieldProjected))
		finalNormalized := compactForwardPackedSlice(layout, data, compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalNormalized))
		wantNormalized, err := compactForwardNormalizeRows(finalProjected, shape.Tokens, shape.ModelDim)
		if err != nil {
			return fmt.Errorf("compact forward packed sequence %d final normalized: %w", seq, err)
		}
		if !compactForwardFloat32SlicesClose(finalNormalized, wantNormalized, 1e-5) {
			return fmt.Errorf("compact forward packed sequence %d final.normalized does not match normalized final projected rows", seq)
		}
		outputRows := compactForwardPackedSlice(layout, data, compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalOutputRows))
		if !shape.HasOutputProjection && !compactForwardFloat32SlicesClose(outputRows, finalNormalized, 1e-5) {
			return fmt.Errorf("compact forward packed sequence %d final.outputRows must equal final.normalized without output projection", seq)
		}
		pooled, active, err := meanPoolRows(outputRows, shape.Tokens, shape.OutputDim, masks[seq])
		if err != nil {
			return fmt.Errorf("compact forward packed sequence %d final pooled rows: %w", seq, err)
		}
		if active != compactForwardMaskActiveCount(masks[seq]) {
			return fmt.Errorf("compact forward packed sequence %d final pooled active count %d, want %d", seq, active, compactForwardMaskActiveCount(masks[seq]))
		}
		finalPooled := compactForwardPackedSlice(layout, data, compactForwardPackedSequenceSpanName(seq, compactForwardPackedFinalPooled))
		layerPooled := compactForwardPackedSlice(layout, data, compactForwardPackedLayerSpanName(seq, shape.Layers-1, compactForwardPackedFieldPooled))
		if !compactForwardFloat32SlicesClose(finalPooled, pooled, 1e-5) {
			return fmt.Errorf("compact forward packed sequence %d final.pooled does not match pooled final.outputRows", seq)
		}
		if !compactForwardFloat32SlicesClose(layerPooled, pooled, 1e-5) {
			return fmt.Errorf("compact forward packed sequence %d final layer pooled does not match pooled final.outputRows", seq)
		}
	}
	return nil
}

func compactForwardPackedSlice(layout backend.CompactForwardPackedStateLayout, data []float32, name string) []float32 {
	span := compactForwardPackedSpanByName(layout, name)
	return data[span.Offset : span.Offset+span.Len]
}

func compactForwardMaskActiveCount(mask []int32) int {
	active := 0
	for _, value := range mask {
		if value != 0 {
			active++
		}
	}
	return active
}

func compactForwardFloat32SlicesClose(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > tolerance {
			return false
		}
	}
	return true
}

func compactForwardPackedLayerFieldLen(shape backend.CompactForwardShape, layer int, kind compactForwardPackedFieldKind) int {
	switch kind {
	case compactForwardPackedFieldScalar:
		return 1
	case compactForwardPackedFieldModelRows:
		return compactForwardPackedModelRowsLen(shape)
	case compactForwardPackedFieldFFNRows:
		return compactForwardPackedFFNRowsLen(shape)
	case compactForwardPackedFieldAttentionScores:
		return compactForwardPackedAttentionScoresLen(shape)
	case compactForwardPackedFieldLayerPooled:
		if layer == shape.Layers-1 {
			return shape.OutputDim
		}
		return shape.ModelDim
	default:
		return -1
	}
}

func compactForwardPackedModelRowsLen(shape backend.CompactForwardShape) int {
	return checkedMulInts(shape.Tokens, shape.ModelDim)
}

func compactForwardPackedFFNRowsLen(shape backend.CompactForwardShape) int {
	return checkedMulInts(shape.Tokens, shape.FFNDim)
}

func compactForwardPackedAttentionScoresLen(shape backend.CompactForwardShape) int {
	return checkedMulInts(shape.Heads, shape.Tokens, shape.Tokens)
}

func compactForwardPackedOutputRowsLen(shape backend.CompactForwardShape) int {
	return checkedMulInts(shape.Tokens, shape.OutputDim)
}

func compactForwardPackedLayerSpanName(seq, layer int, field string) string {
	return fmt.Sprintf("seq%d.layer%d.%s", seq, layer, field)
}

func compactForwardPackedSequenceSpanName(seq int, field string) string {
	return fmt.Sprintf("seq%d.%s", seq, field)
}

func compactForwardPackedSpanMap(layout backend.CompactForwardPackedStateLayout) map[string]backend.CompactForwardStateSpan {
	out := make(map[string]backend.CompactForwardStateSpan, len(layout.Spans))
	for _, span := range layout.Spans {
		out[span.Name] = span
	}
	return out
}

func compactForwardPackedSpanByName(layout backend.CompactForwardPackedStateLayout, name string) backend.CompactForwardStateSpan {
	for _, span := range layout.Spans {
		if span.Name == name {
			return span
		}
	}
	return backend.CompactForwardStateSpan{}
}

func checkedMulInts(values ...int) int {
	out := 1
	for _, value := range values {
		if value < 0 {
			return -1
		}
		next, ok := checkedMulInt(out, value)
		if !ok {
			return -1
		}
		out = next
	}
	return out
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a == 0 || b == 0 {
		return 0, true
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}
