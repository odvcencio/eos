package compiler

import (
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

func TestBuildTinyEmbedSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyEmbed))

	bundle, err := Build(src, Options{ModuleName: "tiny_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := len(bundle.Artifact.Params); got != 2 {
		t.Fatalf("param count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Name; got != "embed" {
		t.Fatalf("entrypoint name = %q, want embed", got)
	}
	if got := len(bundle.Artifact.Steps); got != 4 {
		t.Fatalf("step count = %d, want 4", got)
	}
	if bundle.Artifact.Steps[0].Kind != "gather" {
		t.Fatalf("first step kind = %q, want gather", bundle.Artifact.Steps[0].Kind)
	}
	for _, step := range bundle.Artifact.Steps {
		if step.Entry != "embed" {
			t.Fatalf("step entry = %q, want embed: %+v", step.Entry, step)
		}
	}
	if got := len(bundle.Artifact.Kernels); got != 1 {
		t.Fatalf("kernel count = %d, want 1", got)
	}
	kernel := bundle.Artifact.Kernels[0]
	if kernel.Name != "l2_normalize" {
		t.Fatalf("kernel name = %q, want l2_normalize", kernel.Name)
	}
	if got := len(kernel.Body); got != 2 {
		t.Fatalf("kernel body len = %d, want 2", got)
	}
	if kernel.Body[0].Op != "normalize" || kernel.Body[1].Op != "return" {
		t.Fatalf("unexpected kernel body: %+v", kernel.Body)
	}
	if kernel.Hints.Memory != "workgroup_local" || !kernel.Hints.Subgroup {
		t.Fatalf("unexpected kernel hints: %+v", kernel.Hints)
	}
	if got := len(kernel.Variants); got != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", got, len(kernelBackendEmitters))
	}
	assertAllKernelVariantSources(t, kernel)
}

func TestKernelVariantsFollowBackendEmitterTable(t *testing.T) {
	bundle, err := Build([]byte(sourceForPreset(PresetTinyEmbed)), Options{ModuleName: "tiny_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	kernel := bundle.Artifact.Kernels[0]
	if got := len(kernel.Variants); got != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", got, len(kernelBackendEmitters))
	}
	byBackend := map[eosartifact.BackendKind]eosartifact.KernelVariant{}
	for _, variant := range kernel.Variants {
		byBackend[variant.Backend] = variant
	}
	for _, emitter := range kernelBackendEmitters {
		variant, ok := byBackend[emitter.backend]
		if !ok {
			t.Fatalf("missing variant for backend %q", emitter.backend)
		}
		if want := kernel.Name + "_" + emitter.suffix; variant.Entry != want {
			t.Fatalf("entry for backend %q = %q, want %q", emitter.backend, variant.Entry, want)
		}
		if variant.Source == "" {
			t.Fatalf("source for backend %q is empty", emitter.backend)
		}
	}
}

func TestBuildBatchEmbedSource(t *testing.T) {
	src := []byte(`
param token_embedding: f16[V, D] @weight("weights/token_embedding")
param projection: f16[D, E] @weight("weights/projection")

kernel l2_normalize(x: f16[B, T, E]) -> f16[B, T, E] {
    return normalize(x)
}

pipeline embed_batch(tokens: i32[B, T]) -> f16[B, T, E] {
    let hidden = gather(token_embedding, tokens)
    let projected = @matmul(hidden, projection)
    return l2_normalize(projected)
}
`)

	bundle, err := Build(src, Options{ModuleName: "batch_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "embed_batch" {
		t.Fatalf("entrypoint name = %q, want embed_batch", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "B,T,E" {
		t.Fatalf("output shape = %q, want B,T,E", got)
	}
	if got := len(bundle.Artifact.Steps); got != 4 {
		t.Fatalf("step count = %d, want 4", got)
	}
	if bundle.Artifact.Steps[0].Kind != "gather" {
		t.Fatalf("first step kind = %q, want gather", bundle.Artifact.Steps[0].Kind)
	}
	if bundle.Artifact.Steps[1].Kind != "matmul" {
		t.Fatalf("second step kind = %q, want matmul", bundle.Artifact.Steps[1].Kind)
	}
	foundHidden := false
	for _, buf := range bundle.Artifact.Buffers {
		if buf.Name != "hidden" {
			continue
		}
		foundHidden = true
		if got := strings.Join(buf.Shape, ","); got != "B,T,D" {
			t.Fatalf("hidden buffer shape = %q, want B,T,D", got)
		}
	}
	if !foundHidden {
		t.Fatal("missing hidden buffer")
	}
}

func TestBuildTinyEmbedPooledSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyEmbedPooled))

	bundle, err := Build(src, Options{ModuleName: "tiny_embed_pooled"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := len(bundle.Artifact.EntryPoints); got != 2 {
		t.Fatalf("entrypoint count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Name; got != "embed_pooled" {
		t.Fatalf("entrypoint[0] = %q, want embed_pooled", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "E" {
		t.Fatalf("embed_pooled output shape = %q, want E", got)
	}
	if got := bundle.Artifact.EntryPoints[1].Name; got != "embed_pooled_batch" {
		t.Fatalf("entrypoint[1] = %q, want embed_pooled_batch", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[1].Outputs[0].Type.Tensor.Shape, ","); got != "B,E" {
		t.Fatalf("embed_pooled_batch output shape = %q, want B,E", got)
	}
	foundMeanPool := false
	for _, kernel := range bundle.Artifact.Kernels {
		if kernel.Name != "mean_pool" {
			continue
		}
		foundMeanPool = true
		if len(kernel.Body) != 2 || kernel.Body[0].Op != "mean_pool" {
			t.Fatalf("unexpected mean_pool body: %+v", kernel.Body)
		}
	}
	if !foundMeanPool {
		t.Fatal("missing mean_pool kernel")
	}
}

func TestBuildTrainableQuantizedEmbedSource(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := Build(src, Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.Params[0].Type.Tensor.DType; got != "q8" {
		t.Fatalf("token_embedding dtype = %q, want q8", got)
	}
	if !bundle.Artifact.Params[0].Trainable || !bundle.Artifact.Params[1].Trainable {
		t.Fatalf("expected trainable params, got %+v", bundle.Artifact.Params)
	}
	foundDequant := false
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == "dequant" {
			foundDequant = true
			break
		}
	}
	if !foundDequant {
		t.Fatal("expected dequant step in quantized embed artifact")
	}
}

func TestBuildTinyEmbedMaskedPooledSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyEmbedMaskedPooled))

	bundle, err := Build(src, Options{ModuleName: "tiny_embed_masked_pooled"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := len(bundle.Artifact.EntryPoints); got != 2 {
		t.Fatalf("entrypoint count = %d, want 2", got)
	}
	if got := len(bundle.Artifact.EntryPoints[0].Inputs); got != 2 {
		t.Fatalf("embed_pooled input count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Inputs[1].Name; got != "attention_mask" {
		t.Fatalf("embed_pooled input[1] = %q, want attention_mask", got)
	}
	foundMeanPool := false
	for _, kernel := range bundle.Artifact.Kernels {
		if kernel.Name != "mean_pool" {
			continue
		}
		foundMeanPool = true
		if len(kernel.Inputs) != 2 {
			t.Fatalf("mean_pool input count = %d, want 2", len(kernel.Inputs))
		}
		if len(kernel.Body) != 2 || len(kernel.Body[0].Inputs) != 2 {
			t.Fatalf("unexpected masked mean_pool body: %+v", kernel.Body)
		}
	}
	if !foundMeanPool {
		t.Fatal("missing mean_pool kernel")
	}
}

func TestBuildEncoderTrainableQ8x2Preset(t *testing.T) {
	bundle, err := Build(nil, Options{ModuleName: "encoder_trainable_q8x2", Preset: PresetEncoderTrainableQ8x2})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	assertEncoderTrainablePresetParams(t, bundle.Artifact.Params, "q8")
	if got := len(bundle.Artifact.EntryPoints); got != 2 {
		t.Fatalf("entrypoint count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Name; got != "embed_pooled" {
		t.Fatalf("entrypoint[0] = %q, want embed_pooled", got)
	}
	assertValueBinding(t, bundle.Artifact.EntryPoints[0].Inputs, "role_ids", "i32", "T")
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "D" {
		t.Fatalf("embed_pooled output shape = %q, want D", got)
	}
	assertValueBinding(t, bundle.Artifact.EntryPoints[1].Inputs, "role_ids", "i32", "B,T")
	if got := strings.Join(bundle.Artifact.EntryPoints[1].Outputs[0].Type.Tensor.Shape, ","); got != "B,D" {
		t.Fatalf("embed_pooled_batch output shape = %q, want B,D", got)
	}
	foundGELU := false
	foundRoPE := false
	foundMaskedSoftmax := false
	foundMaskedMeanPool := false
	scaledScoreMatMuls := 0
	for _, kernel := range bundle.Artifact.Kernels {
		for _, op := range kernel.Body {
			if op.Op == "gelu" {
				foundGELU = true
			}
			if op.Op == "rope" {
				foundRoPE = true
			}
			if op.Op == "masked_softmax" && len(op.Inputs) == 2 {
				foundMaskedSoftmax = true
			}
			if op.Op == "mean_pool" && len(op.Inputs) == 2 {
				foundMaskedMeanPool = true
			}
		}
	}
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == eosartifact.StepMatMul && len(step.Outputs) == 1 && strings.HasPrefix(step.Outputs[0], "scores") && step.Attributes["scale"] == "rsqrt_rhs_rows" {
			scaledScoreMatMuls++
		}
	}
	if !foundGELU {
		t.Fatal("expected GELU op in default encoder preset")
	}
	if !foundRoPE {
		t.Fatal("expected RoPE op in default encoder preset")
	}
	if !hasStep(bundle.Artifact.Steps, eosartifact.StepGather, "role_embedding,role_ids") {
		t.Fatal("expected role_embedding gather in default encoder preset")
	}
	if !foundMaskedSoftmax {
		t.Fatal("expected masked_softmax op in default encoder preset")
	}
	if !foundMaskedMeanPool {
		t.Fatal("expected masked mean_pool op in default encoder preset")
	}
	if scaledScoreMatMuls != 4 {
		t.Fatalf("scaled score matmul count = %d, want 4", scaledScoreMatMuls)
	}
}

func TestBuildCompactMultiheadAttentionBuiltin(t *testing.T) {
	src := []byte(`
param q: f16[T, D] @weight("weights/q")
param k: f16[T, D] @weight("weights/k")
param v: f16[T, D] @weight("weights/v")

pipeline compact_attention(attention_mask: i32[T]) -> f16[T, D] {
    return compact_multihead_attention_h2(q, k, v, attention_mask)
}
`)
	bundle, err := Build(src, Options{ModuleName: "compact_attention"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	for _, kernel := range bundle.Artifact.Kernels {
		for _, op := range kernel.Body {
			if op.Op == "compact_multihead_attention" {
				if got := op.Attributes["num_attention_heads"]; got != "2" {
					t.Fatalf("num_attention_heads = %q, want 2", got)
				}
				return
			}
		}
	}
	t.Fatal("compiled artifact missing compact_multihead_attention op")
}

func TestBuildEncoderTrainableQ4x2Preset(t *testing.T) {
	bundle, err := Build(nil, Options{ModuleName: "encoder_trainable_q4x2", Preset: PresetEncoderTrainableQ4x2})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	assertEncoderTrainablePresetParams(t, bundle.Artifact.Params, "q4")
	if got := len(bundle.Artifact.EntryPoints); got != 2 {
		t.Fatalf("entrypoint count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Name; got != "embed_pooled" {
		t.Fatalf("entrypoint[0] = %q, want embed_pooled", got)
	}
	assertValueBinding(t, bundle.Artifact.EntryPoints[0].Inputs, "role_ids", "i32", "T")
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "D" {
		t.Fatalf("embed_pooled output shape = %q, want D", got)
	}
	assertValueBinding(t, bundle.Artifact.EntryPoints[1].Inputs, "role_ids", "i32", "B,T")
	if got := strings.Join(bundle.Artifact.EntryPoints[1].Outputs[0].Type.Tensor.Shape, ","); got != "B,D" {
		t.Fatalf("embed_pooled_batch output shape = %q, want B,D", got)
	}
}

func assertEncoderTrainablePresetParams(t *testing.T, params []eosartifact.Param, dtype string) {
	t.Helper()
	want := []struct {
		name    string
		shape   string
		binding string
	}{
		{"token_embedding", "V,D", "weights/token_embedding"},
		{"role_embedding", "3,D", "weights/role_embedding"},
		{"attn_q", "D,D", "weights/attn_q"},
		{"attn_k", "D,D", "weights/attn_k"},
		{"attn_v", "D,D", "weights/attn_v"},
		{"attn_o", "D,D", "weights/attn_o"},
		{"ffn_up", "D,H", "weights/ffn_up"},
		{"projection", "H,D", "weights/projection"},
	}
	if got := len(params); got != len(want) {
		t.Fatalf("param count = %d, want %d", got, len(want))
	}
	for i, wantParam := range want {
		param := params[i]
		if param.Name != wantParam.name {
			t.Fatalf("param[%d] name = %q, want %q", i, param.Name, wantParam.name)
		}
		if !param.Trainable {
			t.Fatalf("param %q is not trainable", param.Name)
		}
		if param.Binding != wantParam.binding {
			t.Fatalf("param %q binding = %q, want %q", param.Name, param.Binding, wantParam.binding)
		}
		if param.Type.Tensor == nil {
			t.Fatalf("param %q type = %+v, want tensor", param.Name, param.Type)
		}
		if param.Type.Tensor.DType != dtype {
			t.Fatalf("param %q dtype = %q, want %s", param.Name, param.Type.Tensor.DType, dtype)
		}
		if got := strings.Join(param.Type.Tensor.Shape, ","); got != wantParam.shape {
			t.Fatalf("param %q shape = %q, want %q", param.Name, got, wantParam.shape)
		}
	}
}

func assertValueBinding(t *testing.T, bindings []eosartifact.ValueBinding, name, dtype, shape string) {
	t.Helper()
	for _, binding := range bindings {
		if binding.Name != name {
			continue
		}
		if binding.Type.Tensor == nil {
			t.Fatalf("binding %q type = %+v, want tensor", name, binding.Type)
		}
		if binding.Type.Tensor.DType != dtype {
			t.Fatalf("binding %q dtype = %q, want %q", name, binding.Type.Tensor.DType, dtype)
		}
		if got := strings.Join(binding.Type.Tensor.Shape, ","); got != shape {
			t.Fatalf("binding %q shape = %q, want %q", name, got, shape)
		}
		return
	}
	t.Fatalf("missing binding %q in %+v", name, bindings)
}

func hasStep(steps []eosartifact.Step, kind eosartifact.StepKind, inputs string) bool {
	for _, step := range steps {
		if step.Kind == kind && strings.Join(step.Inputs, ",") == inputs {
			return true
		}
	}
	return false
}

func TestBuildTinyAttentionEmbedSource(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let wq_f = dequant(attn_q)
    let wk_f = dequant(attn_k)
    let wv_f = dequant(attn_v)
    let wo_f = dequant(attn_o)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq_f)
    let k = @matmul(hidden, wk_f)
    let v = @matmul(hidden, wv_f)
    let kt = transpose(k)
    let scores = @matmul(q, kt)
    let probs = softmax(scores)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo_f)
    let projected = @matmul(attended, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := Build(src, Options{ModuleName: "tiny_attention_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if got := len(bundle.Artifact.Params); got != 6 {
		t.Fatalf("param count = %d, want 6", got)
	}
	foundTranspose := false
	foundBatchedScores := false
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == "transpose" {
			foundTranspose = true
		}
		if step.Name == "matmul" && len(step.Outputs) == 1 && step.Outputs[0] == "scores" {
			foundBatchedScores = true
		}
	}
	if !foundTranspose {
		t.Fatal("expected transpose step in attention artifact")
	}
	if !foundBatchedScores {
		t.Fatal("expected scores matmul step in attention artifact")
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[1].Outputs[0].Type.Tensor.Shape, ","); got != "B,E" {
		t.Fatalf("embed_pooled_batch output shape = %q, want B,E", got)
	}
}

func TestBuildSparseAttentionStep(t *testing.T) {
	src := []byte(`
pipeline attend(q: f16[Q, D], k: f16[T, D], v: f16[T, V]) -> f16[Q, V] {
    return sparse_attention(q, k, v, 2)
}
`)

	bundle, err := Build(src, Options{ModuleName: "sparse_attention"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	found := false
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == eosartifact.StepSparseAttention {
			found = true
			if got := step.Attributes["top_k"]; got != "2" {
				t.Fatalf("sparse top_k = %q, want 2", got)
			}
		}
	}
	if !found {
		t.Fatal("expected sparse_attention step")
	}
	if !containsString(bundle.Artifact.Requirements.Capabilities, eosartifact.CapabilitySparseAttention) {
		t.Fatalf("capabilities = %v, want sparse_attention", bundle.Artifact.Requirements.Capabilities)
	}
}

func TestBuildTurboSparseAttentionStep(t *testing.T) {
	src := []byte(`
pipeline attend(q: f16[B, Q, D], kc: q4[B, D, T, 1], kn: q_norm[B, T, 1], vc: q4[B, V, T, 1], vn: q_norm[B, T, 1]) -> f16[B, Q, V] {
    return turbo_sparse_attention(q, kc, kn, vc, vn, 3)
}
`)

	bundle, err := Build(src, Options{ModuleName: "turbo_sparse_attention"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	found := false
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == eosartifact.StepTurboSparseAttention {
			found = true
			if got := step.Attributes["top_k"]; got != "3" {
				t.Fatalf("turbo sparse top_k = %q, want 3", got)
			}
		}
	}
	if !found {
		t.Fatal("expected turbo_sparse_attention step")
	}
	if !containsString(bundle.Artifact.Requirements.Capabilities, eosartifact.CapabilitySparseAttention) {
		t.Fatalf("capabilities = %v, want sparse_attention", bundle.Artifact.Requirements.Capabilities)
	}
	if !containsString(bundle.Artifact.Requirements.Capabilities, eosartifact.CapabilityTurboQuant) {
		t.Fatalf("capabilities = %v, want turboquant", bundle.Artifact.Requirements.Capabilities)
	}
}

func TestBuildRoutedTurboSparseAttentionStep(t *testing.T) {
	src := []byte(`
pipeline attend(q: f16[B, Q, D], kc: q4[B, D, T, 1], kn: q_norm[B, T, 1], vc: q4[B, V, T, 1], vn: q_norm[B, T, 1]) -> f16[B, Q, V] {
    return turbo_sparse_attention(q, kc, kn, vc, vn, 3, 16, 4)
}
`)

	bundle, err := Build(src, Options{ModuleName: "routed_turbo_sparse_attention"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	for _, step := range bundle.Artifact.Steps {
		if step.Kind == eosartifact.StepTurboSparseAttention {
			if got := step.Attributes["top_k"]; got != "3" {
				t.Fatalf("turbo sparse top_k = %q, want 3", got)
			}
			if got := step.Attributes["route_block_size"]; got != "16" {
				t.Fatalf("route_block_size = %q, want 16", got)
			}
			if got := step.Attributes["route_top_blocks"]; got != "4" {
				t.Fatalf("route_top_blocks = %q, want 4", got)
			}
			return
		}
	}
	t.Fatal("expected turbo_sparse_attention step")
}

func TestBuildTinyDecodeSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyDecode))

	bundle, err := Build(src, Options{ModuleName: "tiny_decode"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "decode_step" {
		t.Fatalf("entrypoint name = %q, want decode_step", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Inputs[1].Type.Kind; got != "kv_cache" {
		t.Fatalf("cache input kind = %q, want kv_cache", got)
	}
	if got := len(bundle.Artifact.Kernels); got != 3 {
		t.Fatalf("kernel count = %d, want 3", got)
	}
	foundKVRead := false
	foundKVWrite := false
	foundBinaryAdd := false
	foundRoPE := false
	foundSoftmax := false
	for _, step := range bundle.Artifact.Steps {
		if step.Entry != "decode_step" {
			t.Fatalf("step entry = %q, want decode_step: %+v", step.Entry, step)
		}
		if step.Kind == "kv_read" {
			foundKVRead = true
		}
		if step.Kind == "kv_write" {
			foundKVWrite = true
		}
	}
	for _, kernel := range bundle.Artifact.Kernels {
		switch kernel.Name {
		case "binary_add":
			foundBinaryAdd = true
		case "rope":
			foundRoPE = true
		case "softmax":
			foundSoftmax = true
			if len(kernel.Body) != 2 || kernel.Body[0].Op != "softmax" {
				t.Fatalf("unexpected softmax kernel body: %+v", kernel.Body)
			}
			if kernel.Hints.Memory != "workgroup_local" || !kernel.Hints.Subgroup {
				t.Fatalf("unexpected softmax hints: %+v", kernel.Hints)
			}
			assertAllKernelVariantSources(t, kernel)
		}
	}
	if !foundKVRead || !foundKVWrite {
		t.Fatalf("expected kv steps, got %+v", bundle.Artifact.Steps)
	}
	if !foundBinaryAdd || !foundRoPE || !foundSoftmax {
		t.Fatalf("expected binary_add/rope/softmax kernels, got %+v", bundle.Artifact.Kernels)
	}
}

// TestRoPEKernelCUDAVariantCarriesABIMarkerAndSeqLenParam guards two halves
// of the CUDA rope stale-artifact fix together: (1) emitKernelVariants tags
// the CUDA "rope" kernel variant's Meta with "rope_abi": "v2" (see
// ropeKernelABIVersion) so tooling can see at a glance which ABI generation
// a kernel was emitted with, and (2) the actual emitted CUDA source (what
// runtime/backends/cuda's validateRoPEKernelABI re-derives ABI compatibility
// from, since that's what nvrtc compiles and the GPU executes) declares the
// `seq_len` parameter the 5-argument runRoPEKernel/launchRoPE launcher
// passes. It also checks a sibling non-rope kernel (softmax) is NOT tagged
// with rope_abi, so the marker doesn't leak onto unrelated kernels.
func TestRoPEKernelCUDAVariantCarriesABIMarkerAndSeqLenParam(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyDecode))

	bundle, err := Build(src, Options{ModuleName: "tiny_decode"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	foundRoPE := false
	foundSoftmax := false
	for _, kernel := range bundle.Artifact.Kernels {
		var cudaVariant *eosartifact.KernelVariant
		for i := range kernel.Variants {
			if kernel.Variants[i].Backend == eosartifact.BackendCUDA {
				cudaVariant = &kernel.Variants[i]
				break
			}
		}
		if cudaVariant == nil {
			t.Fatalf("kernel %q missing CUDA variant", kernel.Name)
		}
		switch kernel.Name {
		case "rope":
			foundRoPE = true
			if got := cudaVariant.Meta["rope_abi"]; got != ropeKernelABIVersion {
				t.Fatalf("rope CUDA variant Meta[rope_abi] = %q, want %q", got, ropeKernelABIVersion)
			}
			if !strings.Contains(cudaVariant.Entry, "rope") {
				t.Fatalf("rope CUDA variant entry = %q, want it to mention rope", cudaVariant.Entry)
			}
			if !strings.Contains(cudaVariant.Source, "int seq_len") {
				t.Fatalf("rope CUDA source missing seq_len parameter:\n%s", cudaVariant.Source)
			}
		case "softmax":
			foundSoftmax = true
			if _, ok := cudaVariant.Meta["rope_abi"]; ok {
				t.Fatalf("softmax CUDA variant should not carry a rope_abi tag, got Meta=%+v", cudaVariant.Meta)
			}
		}
	}
	if !foundRoPE || !foundSoftmax {
		t.Fatalf("expected rope and softmax kernels in tiny_decode preset, got %+v", bundle.Artifact.Kernels)
	}
}

func TestBuildTinyScoreSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyScore))

	bundle, err := Build(src, Options{ModuleName: "tiny_score"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "score" {
		t.Fatalf("entrypoint name = %q, want score", got)
	}
	if got := len(bundle.Artifact.Params); got != 1 {
		t.Fatalf("param count = %d, want 1", got)
	}
	if got := len(bundle.Artifact.Steps); got != 2 {
		t.Fatalf("step count = %d, want 2", got)
	}
	if bundle.Artifact.Steps[0].Kind != "launch_kernel" || bundle.Artifact.Steps[0].Kernel != "cosine" {
		t.Fatalf("unexpected score launch step: %+v", bundle.Artifact.Steps[0])
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.DType; got != "f32" {
		t.Fatalf("output dtype = %q, want f32", got)
	}
	if got := len(bundle.Artifact.Kernels); got != 1 {
		t.Fatalf("kernel count = %d, want 1", got)
	}
	kernel := bundle.Artifact.Kernels[0]
	if kernel.Name != "cosine" {
		t.Fatalf("kernel name = %q, want cosine", kernel.Name)
	}
	if len(kernel.Body) != 2 || kernel.Body[0].Op != "cosine" {
		t.Fatalf("unexpected cosine kernel body: %+v", kernel.Body)
	}
	if kernel.Hints.Memory != "workgroup_local" || !kernel.Hints.Subgroup {
		t.Fatalf("unexpected cosine score hints: %+v", kernel.Hints)
	}
	if got := len(kernel.Variants); got != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", got, len(kernelBackendEmitters))
	}
	cudaVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendCUDA)
	metalVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendMetal)
	if !strings.Contains(cudaVariant.Source, "const float* query") || !strings.Contains(metalVariant.Source, "const device float* query") {
		t.Fatalf("expected emitted score kernel sources, got %+v", kernel.Variants)
	}
}

func TestBuildTinyGELUSource(t *testing.T) {
	src := []byte(`
pipeline ffn(x: f16[T, D], w: f16[D, E]) -> f16[T, E] {
    let projected = @matmul(x, w)
    return gelu(projected)
}
`)

	bundle, err := Build(src, Options{ModuleName: "tiny_gelu"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "ffn" {
		t.Fatalf("entrypoint name = %q, want ffn", got)
	}
	if got := len(bundle.Artifact.Steps); got != 3 {
		t.Fatalf("step count = %d, want 3", got)
	}
	if got := bundle.Artifact.Steps[0].Kind; got != "matmul" {
		t.Fatalf("step[0] kind = %q, want matmul", got)
	}
	if got := bundle.Artifact.Steps[1].Kind; got != "launch_kernel" {
		t.Fatalf("step[1] kind = %q, want launch_kernel", got)
	}
	if got := bundle.Artifact.Steps[1].Kernel; got != "gelu" {
		t.Fatalf("step[1] kernel = %q, want gelu", got)
	}
	if got := bundle.Artifact.Steps[2].Kind; got != "return" {
		t.Fatalf("step[2] kind = %q, want return", got)
	}
	if got := len(bundle.Artifact.Kernels); got != 1 {
		t.Fatalf("kernel count = %d, want 1", got)
	}
	kernel := bundle.Artifact.Kernels[0]
	if kernel.Name != "gelu" {
		t.Fatalf("kernel name = %q, want gelu", kernel.Name)
	}
	if len(kernel.Body) != 2 || kernel.Body[0].Op != "gelu" || kernel.Body[1].Op != "return" {
		t.Fatalf("unexpected gelu kernel body: %+v", kernel.Body)
	}
	if kernel.Hints.Memory != "device_local" || kernel.Hints.VectorWidth != 4 {
		t.Fatalf("unexpected gelu hints: %+v", kernel.Hints)
	}
	if got := len(kernel.Variants); got != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", got, len(kernelBackendEmitters))
	}
	cudaVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendCUDA)
	metalVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendMetal)
	if !strings.Contains(cudaVariant.Source, "tanhf") || !strings.Contains(metalVariant.Source, "tanh(") {
		t.Fatalf("expected emitted gelu backend sources, got %+v", kernel.Variants)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape[1]; got != "E" {
		t.Fatalf("output dim[1] = %q, want E", got)
	}
}

func TestBuildTinyLayerNormSource(t *testing.T) {
	src := []byte(`
pipeline encoder_block(x: f16[T, D], w: f16[D, D]) -> f16[T, D] {
    let projected = @matmul(x, w)
    let residual = projected + x
    return layernorm(residual)
}
`)

	bundle, err := Build(src, Options{ModuleName: "tiny_layernorm"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	foundLayerNorm := false
	for _, kernel := range bundle.Artifact.Kernels {
		if kernel.Name != "layernorm" {
			continue
		}
		foundLayerNorm = true
		if len(kernel.Body) != 2 || kernel.Body[0].Op != "layernorm" {
			t.Fatalf("unexpected layernorm kernel body: %+v", kernel.Body)
		}
		if kernel.Hints.Memory != "workgroup_local" || !kernel.Hints.Subgroup {
			t.Fatalf("unexpected layernorm hints: %+v", kernel.Hints)
		}
		assertAllKernelVariantSources(t, kernel)
	}
	if !foundLayerNorm {
		t.Fatal("missing layernorm kernel")
	}
}

func TestBuildTinyRerankSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyRerank))

	bundle, err := Build(src, Options{ModuleName: "tiny_rerank"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "rerank" {
		t.Fatalf("entrypoint name = %q, want rerank", got)
	}
	if got := len(bundle.Artifact.Steps); got != 4 {
		t.Fatalf("step count = %d, want 4", got)
	}
	if bundle.Artifact.Steps[0].Kind != "launch_kernel" || bundle.Artifact.Steps[0].Kernel != "cosine" {
		t.Fatalf("unexpected rerank score step: %+v", bundle.Artifact.Steps[0])
	}
	if bundle.Artifact.Steps[1].Kind != "topk" || bundle.Artifact.Steps[1].Attributes["k"] != "2" {
		t.Fatalf("unexpected rerank topk step: %+v", bundle.Artifact.Steps[1])
	}
	if bundle.Artifact.Steps[2].Kind != "gather" {
		t.Fatalf("unexpected rerank gather step: %+v", bundle.Artifact.Steps[2])
	}
	if got := len(bundle.Artifact.EntryPoints[0].Outputs); got != 2 {
		t.Fatalf("output count = %d, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Name; got != "top_ids" {
		t.Fatalf("output[0].name = %q, want top_ids", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[1].Name; got != "top_scores" {
		t.Fatalf("output[1].name = %q, want top_scores", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.DType; got != "i32" {
		t.Fatalf("output[0] dtype = %q, want i32", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "2" {
		t.Fatalf("output[0] shape = %q, want 2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[1].Type.Tensor.DType; got != "f32" {
		t.Fatalf("output[1] dtype = %q, want f32", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[1].Type.Tensor.Shape, ","); got != "2" {
		t.Fatalf("output[1] shape = %q, want 2", got)
	}
	returnStep := bundle.Artifact.Steps[3]
	if returnStep.Kind != "return" {
		t.Fatalf("return step kind = %q, want return", returnStep.Kind)
	}
	if got := strings.Join(returnStep.Outputs, ","); got != "top_ids,top_scores" {
		t.Fatalf("return outputs = %q, want top_ids,top_scores", got)
	}
}

func TestBuildRejectsReturnValueCountMismatch(t *testing.T) {
	src := []byte(`
pipeline rerank(query: f16[D], docs: q4[N, D]) -> (top_ids: i32[2], top_scores: f32[2]) {
    let scores = cosine(query, docs)
    return topk(scores, 2)
}
`)

	_, err := Build(src, Options{ModuleName: "bad_return_count"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "return value count mismatch: got 1, want 2") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildTinySelectSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinySelect))

	bundle, err := Build(src, Options{ModuleName: "tiny_select"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "select_scores" {
		t.Fatalf("entrypoint name = %q, want select_scores", got)
	}
	if got := len(bundle.Artifact.Steps); got != 4 {
		t.Fatalf("step count = %d, want 4", got)
	}
	if bundle.Artifact.Steps[0].Kind != "launch_kernel" || bundle.Artifact.Steps[0].Kernel != "cosine" {
		t.Fatalf("unexpected select score step: %+v", bundle.Artifact.Steps[0])
	}
	if bundle.Artifact.Steps[1].Kind != "topk" || bundle.Artifact.Steps[1].Attributes["k"] != "2" {
		t.Fatalf("unexpected select topk step: %+v", bundle.Artifact.Steps[1])
	}
	if bundle.Artifact.Steps[2].Kind != "gather" {
		t.Fatalf("unexpected select gather step: %+v", bundle.Artifact.Steps[2])
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.DType; got != "f32" {
		t.Fatalf("output dtype = %q, want f32", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "2" {
		t.Fatalf("output shape = %q, want 2", got)
	}
}

func TestBuildTinyRetrieveSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyRetrieve))

	bundle, err := Build(src, Options{ModuleName: "tiny_retrieve"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "retrieve" {
		t.Fatalf("entrypoint name = %q, want retrieve", got)
	}
	if got := len(bundle.Artifact.Steps); got != 5 {
		t.Fatalf("step count = %d, want 5", got)
	}
	if bundle.Artifact.Steps[0].Kind != "launch_kernel" || bundle.Artifact.Steps[0].Kernel != "cosine" {
		t.Fatalf("unexpected retrieve score step: %+v", bundle.Artifact.Steps[0])
	}
	if bundle.Artifact.Steps[1].Kind != "topk" || bundle.Artifact.Steps[1].Attributes["k"] != "2" {
		t.Fatalf("unexpected retrieve topk step: %+v", bundle.Artifact.Steps[1])
	}
	if bundle.Artifact.Steps[2].Kind != "gather" || bundle.Artifact.Steps[3].Kind != "gather" {
		t.Fatalf("unexpected retrieve gather steps: %+v %+v", bundle.Artifact.Steps[2], bundle.Artifact.Steps[3])
	}
	if got := len(bundle.Artifact.EntryPoints[0].Outputs); got != 3 {
		t.Fatalf("output count = %d, want 3", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[2].Name; got != "top_docs" {
		t.Fatalf("output[2].name = %q, want top_docs", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[2].Type.Tensor.DType; got != "q4" {
		t.Fatalf("output[2] dtype = %q, want q4", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[2].Type.Tensor.Shape, ","); got != "2,D" {
		t.Fatalf("output[2] shape = %q, want 2,D", got)
	}
	if got := strings.Join(bundle.Artifact.Steps[4].Outputs, ","); got != "top_ids,top_scores,top_docs" {
		t.Fatalf("return outputs = %q, want top_ids,top_scores,top_docs", got)
	}
}

func TestBuildTinyCandidatesSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyCandidates))

	bundle, err := Build(src, Options{ModuleName: "tiny_candidates"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "rerank_candidates" {
		t.Fatalf("entrypoint name = %q, want rerank_candidates", got)
	}
	if got := len(bundle.Artifact.Params); got != 0 {
		t.Fatalf("param count = %d, want 0", got)
	}
	if got := len(bundle.Artifact.EntryPoints[0].Inputs); got != 3 {
		t.Fatalf("input count = %d, want 3", got)
	}
	if got := len(bundle.Artifact.EntryPoints[0].Outputs); got != 3 {
		t.Fatalf("output count = %d, want 3", got)
	}
	if got := len(bundle.Artifact.Steps); got != 6 {
		t.Fatalf("step count = %d, want 6", got)
	}
	if bundle.Artifact.Steps[0].Kind != "launch_kernel" || bundle.Artifact.Steps[0].Kernel != "cosine" {
		t.Fatalf("unexpected candidate score step: %+v", bundle.Artifact.Steps[0])
	}
	if bundle.Artifact.Steps[1].Kind != "topk" || bundle.Artifact.Steps[1].Attributes["k"] != "2" {
		t.Fatalf("unexpected candidate topk step: %+v", bundle.Artifact.Steps[1])
	}
	if bundle.Artifact.Steps[2].Kind != "gather" || bundle.Artifact.Steps[3].Kind != "gather" || bundle.Artifact.Steps[4].Kind != "gather" {
		t.Fatalf("unexpected candidate gather steps: %+v", bundle.Artifact.Steps)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Name; got != "top_candidate_ids" {
		t.Fatalf("output[0].name = %q, want top_candidate_ids", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.DType; got != "i64" {
		t.Fatalf("output[0] dtype = %q, want i64", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[2].Type.Tensor.DType; got != "q4" {
		t.Fatalf("output[2] dtype = %q, want q4", got)
	}
	if got := strings.Join(bundle.Artifact.Steps[len(bundle.Artifact.Steps)-1].Outputs, ","); got != "top_candidate_ids,top_scores,top_docs" {
		t.Fatalf("return outputs = %q, want top_candidate_ids,top_scores,top_docs", got)
	}
}

func TestBuildTinyBatchCandidatesSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyBatchCandidates))

	bundle, err := Build(src, Options{ModuleName: "tiny_batch_candidates"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "rerank_candidates_batch" {
		t.Fatalf("entrypoint name = %q, want rerank_candidates_batch", got)
	}
	if got := len(bundle.Artifact.EntryPoints[0].Inputs); got != 3 {
		t.Fatalf("input count = %d, want 3", got)
	}
	if got := len(bundle.Artifact.EntryPoints[0].Outputs); got != 3 {
		t.Fatalf("output count = %d, want 3", got)
	}
	if got := len(bundle.Artifact.Steps); got != 6 {
		t.Fatalf("step count = %d, want 6", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.DType; got != "i64" {
		t.Fatalf("output[0] dtype = %q, want i64", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.Tensor.Shape, ","); got != "Q,2" {
		t.Fatalf("output[0] shape = %q, want Q,2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[1].Type.Tensor.DType; got != "f32" {
		t.Fatalf("output[1] dtype = %q, want f32", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[1].Type.Tensor.Shape, ","); got != "Q,2" {
		t.Fatalf("output[1] shape = %q, want Q,2", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[2].Type.Tensor.DType; got != "q4" {
		t.Fatalf("output[2] dtype = %q, want q4", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[2].Type.Tensor.Shape, ","); got != "Q,2,D" {
		t.Fatalf("output[2] shape = %q, want Q,2,D", got)
	}
}

func TestBuildTinyPackedCandidatesSource(t *testing.T) {
	src := []byte(sourceForPreset(PresetTinyPackedCandidates))

	bundle, err := Build(src, Options{ModuleName: "tiny_packed_candidates"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "rerank_candidates_packed" {
		t.Fatalf("entrypoint name = %q, want rerank_candidates_packed", got)
	}
	if got := len(bundle.Artifact.Steps); got != 7 {
		t.Fatalf("step count = %d, want 7", got)
	}
	if got := bundle.Artifact.Steps[5].Kind; got != "pack_candidates" {
		t.Fatalf("pack step kind = %q, want pack_candidates", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Kind; got != "candidate_pack" {
		t.Fatalf("output kind = %q, want candidate_pack", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.CandidatePack.Shape, ","); got != "2,D" {
		t.Fatalf("output shape = %q, want 2,D", got)
	}
}

func TestBuildBatchedPackedCandidatesSource(t *testing.T) {
	src := []byte(`
pipeline rerank_candidates_packed_batch(queries: f16[Q, D], docs: q4[Q, N, D], candidate_ids: i64[Q, N]) -> candidate_pack[Q, 2, D] {
    let scores = cosine(queries, docs)
    let top_indices = topk(scores, 2)
    let top_candidate_ids = gather(candidate_ids, top_indices)
    let top_scores = gather(scores, top_indices)
    let top_docs = gather(docs, top_indices)
    return pack_candidates(top_candidate_ids, top_scores, top_docs)
}
`)

	bundle, err := Build(src, Options{ModuleName: "packed_batch"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	if got := bundle.Artifact.EntryPoints[0].Name; got != "rerank_candidates_packed_batch" {
		t.Fatalf("entrypoint name = %q, want rerank_candidates_packed_batch", got)
	}
	if got := bundle.Artifact.EntryPoints[0].Outputs[0].Type.Kind; got != "candidate_pack" {
		t.Fatalf("output kind = %q, want candidate_pack", got)
	}
	if got := strings.Join(bundle.Artifact.EntryPoints[0].Outputs[0].Type.CandidatePack.Shape, ","); got != "Q,2,D" {
		t.Fatalf("output shape = %q, want Q,2,D", got)
	}
}

func TestBuildDirectQuantizedDotSource(t *testing.T) {
	src := []byte(`
pipeline score(query: f16[D], docs: q4[N, D]) -> f32[N] {
    return dot(query, docs)
}
`)

	bundle, err := Build(src, Options{ModuleName: "dot_score"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if got := len(bundle.Artifact.Kernels); got != 1 {
		t.Fatalf("kernel count = %d, want 1", got)
	}
	kernel := bundle.Artifact.Kernels[0]
	if kernel.Name != "dot" {
		t.Fatalf("kernel name = %q, want dot", kernel.Name)
	}
	if kernel.Inputs[1].Type.Tensor == nil || kernel.Inputs[1].Type.Tensor.DType != "q4" {
		t.Fatalf("docs input dtype = %+v, want q4", kernel.Inputs[1].Type)
	}
}

func TestBuildRejectsUnknownCallableWithSpan(t *testing.T) {
	src := []byte(`
pipeline bad(x: f16[T, D]) -> f16[T, D] {
    return nope(x)
}
`)

	_, err := Build(src, Options{ModuleName: "bad"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), `error:3:12: unknown callable "nope"`) {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildRejectsPipelineCalls(t *testing.T) {
	src := []byte(`
pipeline inner(x: f16[T, D]) -> f16[T, D] {
    return x
}

pipeline outer(x: f16[T, D]) -> f16[T, D] {
    return inner(x)
}
`)

	_, err := Build(src, Options{ModuleName: "bad"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "pipeline calls are not supported yet: inner") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildRejectsMatmulShapeMismatch(t *testing.T) {
	src := []byte(`
param w: f16[E, E] @weight("weights/w")

pipeline bad(x: f16[T, D]) -> f16[T, E] {
    return @matmul(x, w)
}
`)

	_, err := Build(src, Options{ModuleName: "bad"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "@matmul inner dimensions mismatch: D vs E") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildRejectsNumericLiteralExpressions(t *testing.T) {
	src := []byte(`
pipeline bad(x: f16[T, D]) -> f16[T, D] {
    return x + 1
}
`)

	_, err := Build(src, Options{ModuleName: "bad"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "numeric literals in expressions are not lowered yet") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildEmitsPromotedDequantKernelVariants(t *testing.T) {
	src := []byte(`
kernel qdequant(x: q4[T, D]) -> f16[T, D] {
    return dequant(x)
}

pipeline deq(x: q4[T, D]) -> f16[T, D] {
    return qdequant(x)
}
`)

	bundle, err := Build(src, Options{ModuleName: "deq"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if got := len(bundle.Artifact.Kernels); got != 1 {
		t.Fatalf("kernel count = %d, want 1", got)
	}
	kernel := bundle.Artifact.Kernels[0]
	if kernel.Name != "qdequant" {
		t.Fatalf("kernel name = %q, want qdequant", kernel.Name)
	}
	if got := len(kernel.Variants); got != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", got, len(kernelBackendEmitters))
	}
	cudaVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendCUDA)
	metalVariant := kernelVariantByBackend(t, kernel, eosartifact.BackendMetal)
	if !strings.Contains(cudaVariant.Source, "_cuda(") || !strings.Contains(metalVariant.Source, "_metal(") {
		t.Fatalf("expected emitted CUDA and Metal dequant sources, got %+v", kernel.Variants)
	}
}

func kernelVariantByBackend(t *testing.T, kernel eosartifact.Kernel, kind eosartifact.BackendKind) eosartifact.KernelVariant {
	t.Helper()
	for _, variant := range kernel.Variants {
		if variant.Backend == kind {
			return variant
		}
	}
	t.Fatalf("missing variant for backend %q in %+v", kind, kernel.Variants)
	return eosartifact.KernelVariant{}
}

func assertAllKernelVariantSources(t *testing.T, kernel eosartifact.Kernel) {
	t.Helper()
	if len(kernel.Variants) != len(kernelBackendEmitters) {
		t.Fatalf("variant count = %d, want backend emitter count %d", len(kernel.Variants), len(kernelBackendEmitters))
	}
	for _, variant := range kernel.Variants {
		if variant.Source == "" {
			t.Fatalf("source for backend %q is empty in %+v", variant.Backend, kernel.Variants)
		}
	}
}

func containsString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func TestBuildRejectsScoreShapeMismatch(t *testing.T) {
	src := []byte(`
pipeline bad(query: f16[D], docs: f16[N, E]) -> f32[N] {
    return cosine(query, docs)
}
`)

	_, err := Build(src, Options{ModuleName: "bad_score"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "cosine dimension mismatch: query D vs docs E") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}

func TestBuildRejectsTopKNonLiteralLimit(t *testing.T) {
	src := []byte(`
pipeline rerank(scores: f32[N], k: i32[1]) -> i32[2] {
    return topk(scores, k)
}
`)

	_, err := Build(src, Options{ModuleName: "bad_topk"})
	if err == nil {
		t.Fatal("expected diagnostic error")
	}
	if !strings.Contains(err.Error(), "topk requires a positive integer literal limit") {
		t.Fatalf("unexpected diagnostic:\n%v", err)
	}
}
