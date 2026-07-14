//go:build linux && cgo

package cuda

import (
	"math"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCompactForwardAcceleratorPackedParity(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()

	req := backend.CompactForwardRequest{
		Shape: backend.CompactForwardShape{
			Batch:               2,
			Tokens:              3,
			ModelDim:            4,
			FFNDim:              5,
			Heads:               2,
			HeadDim:             2,
			Layers:              2,
			OutputDim:           3,
			HasOutputProjection: true,
		},
		Tokens: [][]int32{{1, 2, 1}, {3, 0, 2}},
		Masks:  [][]int32{{1, 1, 0}, {1, 0, 1}},
		Roles:  []int32{1, 2},
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	got, err := accel.RunCompactForward(req)
	if err != nil {
		t.Fatalf("run compact forward: %v", err)
	}
	want := hostCompactForwardForCUDATest(req, false, true)
	assertCompactForwardResultClose(t, got, want, 1e-6)
	stats := accel.Stats()
	if stats.PackedDownloads != 1 || stats.IntermediateD2H != 0 {
		t.Fatalf("transfer stats = %+v, want one packed download and zero intermediate D2H", stats)
	}
	if stats.IntermediateDownloadedBytes != 0 || stats.StatusDownloadedBytes != 4 || stats.LastStatusDownloadedBytes != 4 {
		t.Fatalf("status/intermediate transfer stats = %+v, want status in final packed download and no intermediate D2H", stats)
	}
	if stats.LastPackedFloats != len(got.Data) || stats.LastPackedBytes != int64(len(got.Data)*4) || stats.LastDownloadBytes != int64((len(got.Data)+1)*4) {
		t.Fatalf("packed stats = %+v data=%d", stats, len(got.Data))
	}
	if wantLaunches := expectedCompactForwardLaunches(req.Shape); stats.LastKernelLaunches != wantLaunches || stats.LastKernelSynchronizations != wantLaunches {
		t.Fatalf("launch stats = %+v, want launches/syncs %d", stats, wantLaunches)
	}
	got.Data[0] = 777
	again, err := accel.RunCompactForward(req)
	if err != nil {
		t.Fatalf("run compact forward again: %v", err)
	}
	if again.Data[0] == 777 {
		t.Fatalf("compact result reused caller-mutated data backing")
	}
}

func TestCompactForwardAcceleratorNoProjectionMultiTokenRoPEParity(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, true, false)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape: backend.CompactForwardShape{
			Batch:     1,
			Tokens:    3,
			ModelDim:  4,
			FFNDim:    5,
			Heads:     2,
			HeadDim:   2,
			Layers:    2,
			OutputDim: 4,
		},
		Tokens: [][]int32{{2, 1, 3}},
		Masks:  [][]int32{{1, 1, 1}},
		Roles:  []int32{0},
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	got, err := accel.RunCompactForward(req)
	if err != nil {
		t.Fatalf("run compact forward: %v", err)
	}
	want := hostCompactForwardForCUDATest(req, true, false)
	assertCompactForwardResultClose(t, got, want, 1e-6)
	assertRoPEInputUsesNonZeroAdjacentProductionPairs(t, got)
}

func TestCompactForwardAcceleratorFastGELUParity(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape: backend.CompactForwardShape{
			Batch:               1,
			Tokens:              3,
			ModelDim:            4,
			FFNDim:              5,
			Heads:               2,
			HeadDim:             2,
			Layers:              2,
			OutputDim:           3,
			HasOutputProjection: true,
		},
		Tokens:   [][]int32{{4, 2, 1}},
		Masks:    [][]int32{{1, 1, 1}},
		Roles:    []int32{2},
		GELUMode: backend.CompactForwardGELUFast,
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	got, err := accel.RunCompactForward(req)
	if err != nil {
		t.Fatalf("run compact forward fast gelu: %v", err)
	}
	want := hostCompactForwardForCUDATest(req, false, true)
	assertCompactForwardResultClose(t, got, want, 1e-6)
	exactReq := req
	exactReq.GELUMode = backend.CompactForwardGELUExact
	exact := hostCompactForwardForCUDATest(exactReq, false, true)
	if compactForwardMaxAbs(got.Data, exact.Data) <= 1e-6 {
		t.Fatalf("fast GELU packed output unexpectedly matched exact GELU within tolerance")
	}
}

func TestCompactForwardAcceleratorRejectsUnsupportedGELUModeBeforeLaunch(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape:    backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true},
		Tokens:   [][]int32{{1, 2}},
		Masks:    [][]int32{{1, 1}},
		Roles:    []int32{0},
		GELUMode: "erf",
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	if _, err := accel.RunCompactForward(req); err == nil || !strings.Contains(err.Error(), "unsupported GELU mode") {
		t.Fatalf("unsupported GELU mode error = %v", err)
	}
	if stats := accel.Stats(); stats.KernelLaunches != 0 || stats.PackedDownloads != 0 {
		t.Fatalf("unsupported GELU mode launched work: %+v", stats)
	}
}

func TestCompactForwardAcceleratorPreflightFailures(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape:  backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true},
		Tokens: [][]int32{{1, 2}},
		Masks:  [][]int32{{0, 0}},
		Roles:  []int32{0},
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	if _, err := accel.RunCompactForward(req); err == nil || !strings.Contains(err.Error(), "zero tokens") {
		t.Fatalf("all-mask error = %v, want zero-token preflight failure", err)
	}
	req.Masks = [][]int32{{1, 1}}
	req.Shape.ModelDim = 6
	if _, err := accel.RunCompactForward(req); err == nil || !strings.Contains(err.Error(), "head_dim") {
		t.Fatalf("shape error = %v, want head layout failure", err)
	}
}

func TestCompactForwardAcceleratorRejectsStaleResidentRef(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape:  backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true},
		Tokens: [][]int32{{1, 2}},
		Masks:  [][]int32{{1, 1}},
		Roles:  []int32{0},
	}
	req.ResidentRefs = compactForwardResidentRefsForTest(t, accel, req.Shape)
	for i := range req.ResidentRefs {
		if req.ResidentRefs[i].Name == "layer0_attn_q" {
			token := req.ResidentRefs[i].Token.(*optimizerResidentParameterToken)
			req.ResidentRefs[i].Token = &optimizerResidentParameterToken{owner: token.owner, name: token.name, generation: 0}
			break
		}
	}
	if _, err := accel.RunCompactForward(req); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("stale ref error = %v, want stale resident failure", err)
	}
}

func TestCompactForwardAcceleratorRejectsMissingResidentRef(t *testing.T) {
	accel, cleanup := newBoundCompactForwardTestAccelerator(t, false, true)
	defer cleanup()
	req := backend.CompactForwardRequest{
		Shape:        backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true},
		Tokens:       [][]int32{{1, 2}},
		Masks:        [][]int32{{1, 1}},
		Roles:        []int32{0},
		ResidentRefs: compactForwardResidentRefsForTest(t, accel, backend.CompactForwardShape{Batch: 1, Tokens: 2, ModelDim: 4, FFNDim: 5, Heads: 2, HeadDim: 2, Layers: 2, OutputDim: 3, HasOutputProjection: true})[1:],
	}
	if _, err := accel.RunCompactForward(req); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("missing ref error = %v, want missing resident ref failure", err)
	}
}

func TestCompactForwardAcceleratorActualShapeSizingPreflight(t *testing.T) {
	shape := backend.CompactForwardShape{
		Batch:               8,
		Tokens:              512,
		ModelDim:            384,
		FFNDim:              1536,
		Heads:               12,
		HeadDim:             32,
		Layers:              2,
		OutputDim:           128,
		HasOutputProjection: true,
	}
	bytes, err := compactForwardMemoryEstimateBytes(shape, 32000, 3)
	if err != nil {
		t.Fatalf("actual-shape compact memory estimate: %v", err)
	}
	const rtx5070TiConservativeBytes = int64(16) * 1024 * 1024 * 1024
	if bytes <= 0 || bytes >= rtx5070TiConservativeBytes {
		t.Fatalf("actual-shape compact memory estimate = %d bytes, want positive and < 16GiB", bytes)
	}
	if _, _, err := compactForwardPackedLayout(shape); err != nil {
		t.Fatalf("actual-shape compact packed layout: %v", err)
	}

	overflow := shape
	overflow.Tokens = 1 << 20
	if _, err := compactForwardMemoryEstimateBytes(overflow, 32000, 3); err == nil {
		t.Fatalf("overflow compact memory estimate succeeded")
	}
}

func newBoundCompactForwardTestAccelerator(t *testing.T, rope, projection bool) (*CompactForwardAccelerator, func()) {
	t.Helper()
	optAny, err := NewOptimizerAccelerator()
	if err != nil {
		t.Fatalf("new optimizer accelerator: %v", err)
	}
	opt := optAny.(*optimizerAccelerator)
	accel, err := NewCompactForwardAccelerator()
	if err != nil {
		opt.Close()
		t.Fatalf("new compact forward accelerator: %v", err)
	}
	layers := []CompactForwardLayerNames{
		{AttentionQ: "layer0_attn_q", AttentionK: "layer0_attn_k", AttentionV: "layer0_attn_v", AttentionO: "layer0_attn_o", FFNUp: "layer0_ffn_up", FFNDown: "layer0_ffn_down"},
		{AttentionQ: "layer1_attn_q", AttentionK: "layer1_attn_k", AttentionV: "layer1_attn_v", AttentionO: "layer1_attn_o", FFNUp: "layer1_ffn_up", FFNDown: "layer1_ffn_down"},
	}
	outName := ""
	if projection {
		outName = "output_projection"
	}
	accel.Configure(layers, "token_embedding", "role_embedding", outName, rope)
	for name, tensor := range compactForwardTestWeights(projection) {
		zero := backend.NewTensorF32(tensor.Shape, make([]float32, len(tensor.F32)))
		cfg := backend.OptimizerUpdateConfig{Optimizer: "sgd", LearningRate: 0, Scale: 1, DeferSync: true}
		if err := opt.ApplyUpdate(name, cfg, tensor.Clone(), nil, nil, zero); err != nil {
			accel.Close()
			opt.Close()
			t.Fatalf("make resident %s: %v", name, err)
		}
		ref, ok := opt.ResidentParameter(name)
		if !ok {
			accel.Close()
			opt.Close()
			t.Fatalf("resident %s missing", name)
		}
		if err := accel.BindResident(name, tensor, ref); err != nil {
			accel.Close()
			opt.Close()
			t.Fatalf("bind resident %s: %v", name, err)
		}
	}
	return accel, func() {
		accel.Close()
		opt.Close()
	}
}

func compactForwardResidentRefsForTest(t *testing.T, accel *CompactForwardAccelerator, shape backend.CompactForwardShape) []backend.CompactForwardResidentRef {
	t.Helper()
	names := accel.requiredResidentNames(shape)
	refs := make([]backend.CompactForwardResidentRef, 0, len(names))
	for _, name := range names {
		token := accel.bridged[name]
		if token == nil {
			t.Fatalf("missing test resident token %q", name)
		}
		binding, ok := accel.bindingForName(name)
		if !ok {
			t.Fatalf("missing test resident binding %q", name)
		}
		refs = append(refs, backend.CompactForwardResidentRef{
			Name:     name,
			Backend:  accel.Backend(),
			Token:    token,
			Elements: binding.elements,
		})
	}
	return refs
}

func compactForwardTestWeights(projection bool) map[string]*backend.Tensor {
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{5, 4}, seqData(20, 0.07, -0.4)),
		"role_embedding":  backend.NewTensorF32([]int{3, 4}, seqData(12, 0.03, -0.15)),
	}
	for layer := 0; layer < 2; layer++ {
		prefix := "layer" + string(rune('0'+layer)) + "_"
		weights[prefix+"attn_q"] = backend.NewTensorF32([]int{4, 4}, seqData(16, 0.021+float64(layer)*0.003, -0.12))
		weights[prefix+"attn_k"] = backend.NewTensorF32([]int{4, 4}, seqData(16, -0.017-float64(layer)*0.002, 0.11))
		weights[prefix+"attn_v"] = backend.NewTensorF32([]int{4, 4}, seqData(16, 0.013+float64(layer)*0.002, -0.08))
		weights[prefix+"attn_o"] = backend.NewTensorF32([]int{4, 4}, seqData(16, -0.019, 0.09+float64(layer)*0.01))
		weights[prefix+"ffn_up"] = backend.NewTensorF32([]int{4, 5}, seqData(20, 0.015+float64(layer)*0.001, -0.05))
		weights[prefix+"ffn_down"] = backend.NewTensorF32([]int{5, 4}, seqData(20, -0.014, 0.06+float64(layer)*0.005))
	}
	if projection {
		weights["output_projection"] = backend.NewTensorF32([]int{4, 3}, seqData(12, 0.025, -0.1))
	}
	return weights
}

func seqData(n int, step, offset float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(offset + step*float64((i%7)-3))
	}
	return out
}

func hostCompactForwardForCUDATest(req backend.CompactForwardRequest, rope, projection bool) backend.CompactForwardResult {
	shape := req.Shape
	weights := compactForwardTestWeights(projection)
	layout, total, _ := compactForwardPackedLayout(shape)
	data := make([]float32, total)
	put := func(name string, values []float32) {
		span := compactForwardSpanByName(layout, name)
		copy(data[span.Offset:span.Offset+span.Len], values)
	}
	for b := 0; b < shape.Batch; b++ {
		current := gatherHost(weights["token_embedding"], weights["role_embedding"], req.Tokens[b], req.Roles[b], rope)
		for layer := 0; layer < shape.Layers; layer++ {
			prefix := "layer" + string(rune('0'+layer)) + "_"
			input := append([]float32(nil), current...)
			q := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_q"].F32, shape.ModelDim)
			k := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_k"].F32, shape.ModelDim)
			v := matmulHost(input, shape.Tokens, shape.ModelDim, weights[prefix+"attn_v"].F32, shape.ModelDim)
			scores, mixed := attentionHost(q, k, v, req.Masks[b], shape)
			attnOut := matmulHost(mixed, shape.Tokens, shape.ModelDim, weights[prefix+"attn_o"].F32, shape.ModelDim)
			attnResidual := addHost(attnOut, input)
			hidden := layerNormHost(attnResidual, shape.Tokens, shape.ModelDim)
			ffnHidden := matmulHost(hidden, shape.Tokens, shape.ModelDim, weights[prefix+"ffn_up"].F32, shape.FFNDim)
			activated := geluHost(ffnHidden, req.GELUMode)
			ffnOut := matmulHost(activated, shape.Tokens, shape.FFNDim, weights[prefix+"ffn_down"].F32, shape.ModelDim)
			ffnResidual := addHost(ffnOut, hidden)
			projected := layerNormHost(ffnResidual, shape.Tokens, shape.ModelDim)
			normalized := normalizeRowsHost(projected, shape.Tokens, shape.ModelDim)
			pooled := poolHost(normalized, req.Masks[b], shape.Tokens, shape.ModelDim)
			put(compactForwardLayerSpanName(b, layer, "activeCount"), []float32{float32(activeHost(req.Masks[b]))})
			for _, item := range []struct {
				name string
				vals []float32
			}{{"input", input}, {"hidden", hidden}, {"attnQ", q}, {"attnK", k}, {"attnV", v}, {"attnScores", scores}, {"attnMixed", mixed}, {"attnOutput", attnOut}, {"attnResidual", attnResidual}, {"ffnHidden", ffnHidden}, {"activated", activated}, {"ffnOutput", ffnOut}, {"ffnResidual", ffnResidual}, {"projected", projected}, {"normalized", normalized}, {"pooled", pooled}} {
				if item.name == "pooled" && layer == shape.Layers-1 && shape.OutputDim != shape.ModelDim {
					continue
				}
				put(compactForwardLayerSpanName(b, layer, item.name), item.vals)
			}
			current = projected
		}
		finalNorm := normalizeRowsHost(current, shape.Tokens, shape.ModelDim)
		outputRows := finalNorm
		if projection {
			outputRows = matmulHost(finalNorm, shape.Tokens, shape.ModelDim, weights["output_projection"].F32, shape.OutputDim)
		}
		finalPooled := poolHost(outputRows, req.Masks[b], shape.Tokens, shape.OutputDim)
		if shape.OutputDim != shape.ModelDim {
			put(compactForwardLayerSpanName(b, shape.Layers-1, "pooled"), finalPooled)
		}
		put(compactForwardSequenceSpanName(b, "final.normalized"), finalNorm)
		put(compactForwardSequenceSpanName(b, "final.outputRows"), outputRows)
		put(compactForwardSequenceSpanName(b, "final.pooled"), finalPooled)
	}
	return backend.CompactForwardResult{Layout: layout, Data: data}
}

func assertCompactForwardResultClose(t *testing.T, got, want backend.CompactForwardResult, tol float32) {
	t.Helper()
	if len(got.Data) != len(want.Data) || len(got.Layout.Spans) != len(want.Layout.Spans) {
		t.Fatalf("packed shape got data=%d spans=%d want data=%d spans=%d", len(got.Data), len(got.Layout.Spans), len(want.Data), len(want.Layout.Spans))
	}
	for i, span := range want.Layout.Spans {
		if got.Layout.Spans[i] != span {
			t.Fatalf("span %d = %+v, want %+v", i, got.Layout.Spans[i], span)
		}
	}
	maxAbs := float32(0)
	maxIndex := 0
	for i := range want.Data {
		if !finite32(got.Data[i]) {
			t.Fatalf("nonfinite data[%d]=%v", i, got.Data[i])
		}
		d := abs32(got.Data[i] - want.Data[i])
		if d > maxAbs {
			maxAbs, maxIndex = d, i
		}
	}
	if maxAbs > tol {
		spanName := ""
		for _, span := range got.Layout.Spans {
			if maxIndex >= span.Offset && maxIndex < span.Offset+span.Len {
				spanName = span.Name
				break
			}
		}
		t.Fatalf("max abs = %g at %d span=%s got=%g want=%g tol=%g", maxAbs, maxIndex, spanName, got.Data[maxIndex], want.Data[maxIndex], tol)
	}
}

func assertRoPEInputUsesNonZeroAdjacentProductionPairs(t *testing.T, got backend.CompactForwardResult) {
	t.Helper()
	weights := compactForwardTestWeights(false)
	token := weights["token_embedding"]
	role := weights["role_embedding"]
	tokens := []int32{2, 1, 3}
	roleID := int32(0)
	want := gatherHost(token, role, tokens, roleID, true)
	splitHalf := gatherHostSplitHalfRoPE(token, role, tokens, roleID)
	span := compactForwardSpanByName(got.Layout, compactForwardLayerSpanName(0, 0, "input"))
	gotInput := got.Data[span.Offset : span.Offset+span.Len]
	if compactForwardMaxAbs(want[4:8], splitHalf[4:8]) <= 1e-6 || compactForwardMaxAbs(want[8:12], splitHalf[8:12]) <= 1e-6 {
		t.Fatalf("test fixture does not separate adjacent and split-half RoPE on nonzero positions")
	}
	if maxAbs := compactForwardMaxAbs(gotInput, want); maxAbs > 1e-6 {
		t.Fatalf("RoPE input max abs = %g, want adjacent production pairing", maxAbs)
	}
}

func expectedCompactForwardLaunches(shape backend.CompactForwardShape) int64 {
	launches := int64(1) // gather
	launches += int64(shape.Layers) * 11
	for layer := 0; layer < shape.Layers; layer++ {
		perSeq := int64(17)
		if layer == shape.Layers-1 && shape.OutputDim != shape.ModelDim {
			perSeq = 16
		}
		launches += int64(shape.Batch) * perSeq
	}
	launches++ // final normalization/pool
	if shape.HasOutputProjection {
		launches += 2 // projection matmul and final output pool
	}
	perSeqFinal := int64(3)
	if shape.OutputDim != shape.ModelDim {
		perSeqFinal++
	}
	launches += int64(shape.Batch) * perSeqFinal
	launches += 2 // status word plus final packed copy into the downloaded buffer
	return launches
}

func gatherHost(token, role *backend.Tensor, tokens []int32, roleID int32, rope bool) []float32 {
	d := token.Shape[1]
	out := make([]float32, len(tokens)*d)
	for r, tok := range tokens {
		copy(out[r*d:(r+1)*d], token.F32[int(tok)*d:int(tok)*d+d])
		for c := 0; c < d; c++ {
			out[r*d+c] += role.F32[int(roleID)*d+c]
		}
	}
	if rope {
		applyProductionRoPEHost(out, len(tokens), d)
	}
	return out
}

func gatherHostSplitHalfRoPE(token, role *backend.Tensor, tokens []int32, roleID int32) []float32 {
	d := token.Shape[1]
	out := make([]float32, len(tokens)*d)
	for r, tok := range tokens {
		copy(out[r*d:(r+1)*d], token.F32[int(tok)*d:int(tok)*d+d])
		for c := 0; c < d; c++ {
			out[r*d+c] += role.F32[int(roleID)*d+c]
		}
	}
	half := d / 2
	for row := 0; row < len(tokens); row++ {
		base := row * d
		for i := 0; i < half; i++ {
			theta := float64(row) / math.Pow(10000, float64(2*i)/float64(d))
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			x0, x1 := out[base+i], out[base+i+half]
			out[base+i] = x0*c - x1*s
			out[base+i+half] = x0*s + x1*c
		}
	}
	return out
}

func applyProductionRoPEHost(data []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		base := row * cols
		for col := 0; col+1 < cols; col += 2 {
			theta := float64(row) / math.Pow(10000, float64(col)/float64(cols))
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			x0, x1 := data[base+col], data[base+col+1]
			data[base+col] = x0*c - x1*s
			data[base+col+1] = x0*s + x1*c
		}
	}
}

func matmulHost(lhs []float32, rows, inner int, rhs []float32, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			sum := float32(0)
			for k := 0; k < inner; k++ {
				sum += lhs[r*inner+k] * rhs[k*cols+c]
			}
			out[r*cols+c] = sum
		}
	}
	return out
}

func attentionHost(q, k, v []float32, mask []int32, shape backend.CompactForwardShape) ([]float32, []float32) {
	T, D := shape.Tokens, shape.ModelDim
	scores := make([]float32, shape.Heads*T*T)
	mixed := make([]float32, T*D)
	scale := float32(1 / math.Sqrt(float64(shape.HeadDim)))
	for h := 0; h < shape.Heads; h++ {
		for query := 0; query < T; query++ {
			row := scores[(h*T+query)*T : (h*T+query+1)*T]
			for key := 0; key < T; key++ {
				sum := float32(0)
				for c := 0; c < shape.HeadDim; c++ {
					sum += q[query*D+h*shape.HeadDim+c] * k[key*D+h*shape.HeadDim+c]
				}
				row[key] = sum * scale
			}
			softmaxMaskedHost(row, mask)
			for c := 0; c < shape.HeadDim; c++ {
				sum := float32(0)
				for key := 0; key < T; key++ {
					sum += row[key] * v[key*D+h*shape.HeadDim+c]
				}
				mixed[query*D+h*shape.HeadDim+c] = sum
			}
		}
	}
	return scores, mixed
}

func softmaxMaskedHost(row []float32, mask []int32) {
	maxVal := float32(math.Inf(-1))
	active := false
	for i, v := range row {
		if mask[i] == 0 {
			continue
		}
		if !active || v > maxVal {
			maxVal = v
		}
		active = true
	}
	sum := float32(0)
	for i, v := range row {
		if mask[i] == 0 {
			row[i] = 0
			continue
		}
		e := float32(math.Exp(float64(v - maxVal)))
		row[i] = e
		sum += e
	}
	if sum != 0 {
		for i := range row {
			if mask[i] != 0 {
				row[i] /= sum
			}
		}
	}
}

func addHost(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func layerNormHost(src []float32, rows, cols int) []float32 {
	out := make([]float32, len(src))
	for r := 0; r < rows; r++ {
		base := r * cols
		mean := float32(0)
		for c := 0; c < cols; c++ {
			mean += src[base+c]
		}
		mean /= float32(cols)
		variance := float32(0)
		for c := 0; c < cols; c++ {
			d := src[base+c] - mean
			variance += d * d
		}
		variance /= float32(cols)
		inv := float32(1 / math.Sqrt(float64(variance)+1e-5))
		for c := 0; c < cols; c++ {
			out[base+c] = (src[base+c] - mean) * inv
		}
	}
	return out
}

func geluHost(src []float32, mode string) []float32 {
	out := make([]float32, len(src))
	for i, x := range src {
		inner := float32(0.7978845608) * (x + float32(0.044715)*x*x*x)
		tanh := float32(math.Tanh(float64(inner)))
		if mode == backend.CompactForwardGELUFast {
			tanh = fastTanhHost(inner)
		}
		out[i] = 0.5 * x * (1 + tanh)
	}
	return out
}

func fastTanhHost(x float32) float32 {
	if x >= 3 {
		return 1
	}
	if x <= -3 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}

func normalizeRowsHost(src []float32, rows, cols int) []float32 {
	out := make([]float32, len(src))
	for r := 0; r < rows; r++ {
		base := r * cols
		norm := float32(0)
		for c := 0; c < cols; c++ {
			norm += src[base+c] * src[base+c]
		}
		norm = float32(math.Sqrt(float64(norm)))
		for c := 0; c < cols; c++ {
			if norm == 0 {
				out[base+c] = src[base+c]
			} else {
				out[base+c] = src[base+c] / norm
			}
		}
	}
	return out
}

func poolHost(rows []float32, mask []int32, rowCount, width int) []float32 {
	out := make([]float32, width)
	active := activeHost(mask)
	for r := 0; r < rowCount; r++ {
		if mask[r] == 0 {
			continue
		}
		for c := 0; c < width; c++ {
			out[c] += rows[r*width+c]
		}
	}
	for c := 0; c < width; c++ {
		out[c] /= float32(active)
	}
	return out
}

func activeHost(mask []int32) int {
	active := 0
	for _, v := range mask {
		if v != 0 {
			active++
		}
	}
	return active
}

func finite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func compactForwardMaxAbs(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	maxAbs := float32(0)
	for i := range a {
		if d := abs32(a[i] - b[i]); d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}
