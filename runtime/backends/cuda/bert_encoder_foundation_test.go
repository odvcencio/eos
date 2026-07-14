package cuda

import (
	"fmt"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestBERTCUDAFoundationPlansValidBGE200InputFixture(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	if len(inputs) != 200 {
		t.Fatalf("fixture inputs=%d, want 200", len(inputs))
	}

	plan, status, err := planBGEPretrainedBERTResidentWeights(step, inputs)
	if err != nil {
		t.Fatalf("plan resident weights: %v", err)
	}
	if !status.ShapeValidated || !status.WeightsPlanned {
		t.Fatalf("status = %+v, want shape and weight planning", status)
	}
	if status.FullDeviceReady {
		t.Fatal("foundation planning must remain fail-closed for full-device readiness")
	}
	if len(plan.Weights) != 197 {
		t.Fatalf("planned weights=%d, want 197", len(plan.Weights))
	}

	counts := map[bertCUDAWeightRole]int{}
	for _, weight := range plan.Weights {
		counts[weight.Role]++
		if weight.Name == "" || weight.InputIndex < 3 || weight.Slot == "" {
			t.Fatalf("weight metadata is incomplete: %+v", weight)
		}
	}
	if counts[bertCUDAWeightEmbedding] != 3 {
		t.Fatalf("embedding role count=%d, want 3", counts[bertCUDAWeightEmbedding])
	}
	if counts[bertCUDAWeightVector] != 50 {
		t.Fatalf("vector role count=%d, want 50", counts[bertCUDAWeightVector])
	}
	if counts[bertCUDAWeightDenseMatrix] != 72 {
		t.Fatalf("dense matrix role count=%d, want 72", counts[bertCUDAWeightDenseMatrix])
	}
	if counts[bertCUDAWeightDenseBias] != 72 {
		t.Fatalf("dense bias role count=%d, want 72", counts[bertCUDAWeightDenseBias])
	}

	first := plan.Weights[0]
	if first.InputIndex != 3 || first.Layer != -1 || first.Slot != "token_embeddings" || first.Name != "token_embeddings" {
		t.Fatalf("first weight metadata = %+v", first)
	}
	last := plan.Weights[len(plan.Weights)-1]
	if last.InputIndex != 199 || last.Layer != 11 || last.Slot != "output_layernorm_bias" || last.Name != "encoder_layer_11_output_layernorm_bias" {
		t.Fatalf("last weight metadata = %+v", last)
	}
}

func TestBERTCUDAFoundationRejectsDuplicateResidentWeightNames(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	step.Inputs[4] = step.Inputs[3]

	_, _, err := planBGEPretrainedBERTResidentWeights(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("error = %v, want duplicate-name rejection", err)
	}
}

func TestBERTCUDAFoundationRejectsMissingResidentWeightNames(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	step.Inputs[3] = ""

	_, _, err := planBGEPretrainedBERTResidentWeights(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "missing a stable artifact name") {
		t.Fatalf("error = %v, want missing-name rejection", err)
	}
}

func TestBERTCUDAFoundationRejectsBadDType(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	inputs[3].DType = "q4"

	_, err := validateBGEPretrainedBERTEmbedderStep(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "dtype") {
		t.Fatalf("error = %v, want dtype rejection", err)
	}
}

func TestBERTCUDAFoundationRejectsBadShape(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	inputs[6] = newCUDAFoundationTensor("f32", []int{bgeSmallHiddenSize - 1})

	_, err := validateBGEPretrainedBERTEmbedderStep(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "shape") {
		t.Fatalf("error = %v, want shape rejection", err)
	}
}

func TestBERTCUDAFoundationRejectsInvalidIDValues(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(inputs []*backend.Tensor)
		want   string
	}{
		{name: "token", mutate: func(inputs []*backend.Tensor) { inputs[0].I32[1] = bgeSmallVocabSize }, want: "input_ids[1]"},
		{name: "token type", mutate: func(inputs []*backend.Tensor) { inputs[2].I32[1] = bgeSmallTypeVocabSize }, want: "token_type_ids[1]"},
		{name: "mask", mutate: func(inputs []*backend.Tensor) { inputs[1].I32[1] = 2 }, want: "attention_mask[1]"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			step, inputs := validBGEFoundationFixture(t)
			tt.mutate(inputs)
			_, err := validateBGEPretrainedBERTEmbedderStep(step, inputs)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want invalid-ID rejection mentioning %q", err, tt.want)
			}
		})
	}
}

func TestBERTCUDASelectedContractRejectsShapeCompatibleImpostorMetadata(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	mod := validBGEContractModule()
	mod.Metadata["model_name"] = "shape-compatible/impostor"

	_, err := validateSelectedBGECUDAContract(mod, step, inputs)
	if err == nil || !strings.Contains(err.Error(), "model_name") {
		t.Fatalf("error = %v, want exact metadata rejection", err)
	}
}

func TestBERTCUDASelectedContractRejectsShapeCompatibleImpostorAttributes(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	mod := validBGEContractModule()
	step.Attributes["pooling"] = "masked_mean"

	_, err := validateSelectedBGECUDAContract(mod, step, inputs)
	if err == nil || !strings.Contains(err.Error(), "pooling") {
		t.Fatalf("error = %v, want exact attr rejection", err)
	}
}

func TestBERTCUDASelectedContractRejectsShapeCompatibleImpostorPlan(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	mod := validBGEContractModule()
	step.Inputs[8], step.Inputs[9] = step.Inputs[9], step.Inputs[8]

	_, err := validateSelectedBGECUDAContract(mod, step, inputs)
	if err == nil || !strings.Contains(err.Error(), "step input[8]") {
		t.Fatalf("error = %v, want exact input-plan rejection", err)
	}
}

func TestBERTCUDASelectedContractCacheRejectsSameBackingMutation(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	mod := validBGEContractModule()
	provenance, err := validateSelectedBGEPackageProvenance(mod)
	if err != nil {
		t.Fatalf("provenance: %v", err)
	}
	firstWeight, perWeight, err := bgeSelectedWeightFingerprint(inputs)
	if err != nil {
		t.Fatalf("weight fingerprint: %v", err)
	}
	firstContract := bgeSelectedContractFingerprint(mod, step, provenance, firstWeight)
	firstCacheKey, err := bgeSelectedContractCacheKey(mod, step, inputs, provenance, firstWeight)
	if err != nil {
		t.Fatalf("first cache key: %v", err)
	}
	contract := bertCUDASelectedContract{
		ContractFingerprint: firstContract,
		WeightFingerprint:   firstWeight,
		WeightFingerprints:  perWeight,
		Provenance:          provenance,
		CacheKey:            firstCacheKey,
	}
	device := &deviceRuntime{bertSelectedContractCache: map[string]bertCUDASelectedContract{firstCacheKey: contract}}

	inputs[8].F32[0] = 1
	secondWeight, _, err := bgeSelectedWeightFingerprint(inputs)
	if err != nil {
		t.Fatalf("second weight fingerprint: %v", err)
	}
	secondCacheKey, err := bgeSelectedContractCacheKey(mod, step, inputs, provenance, secondWeight)
	if err != nil {
		t.Fatalf("second cache key: %v", err)
	}
	if secondCacheKey == firstCacheKey {
		t.Fatalf("same-backing mutation did not change selected cache key: %s", firstCacheKey)
	}
	_, err = validateSelectedBGECUDAContractCached(device, mod, step, inputs)
	if err == nil || !strings.Contains(err.Error(), "contract fingerprint mismatch") {
		t.Fatalf("cached validation err=%v, want fail-closed mutation rejection", err)
	}
}

func TestBERTCUDASelectedContractFingerprintChangesForShapeCompatibleWeights(t *testing.T) {
	step, inputs := validBGEFoundationFixture(t)
	mod := validBGEContractModule()
	provenance, err := validateSelectedBGEPackageProvenance(mod)
	if err != nil {
		t.Fatalf("provenance: %v", err)
	}
	firstWeight, _, err := bgeSelectedWeightFingerprint(inputs)
	if err != nil {
		t.Fatalf("first weight fingerprint: %v", err)
	}
	firstContract := bgeSelectedContractFingerprint(mod, step, provenance, firstWeight)
	inputs[8].F32[0] = 1
	secondWeight, _, err := bgeSelectedWeightFingerprint(inputs)
	if err != nil {
		t.Fatalf("second weight fingerprint: %v", err)
	}
	secondContract := bgeSelectedContractFingerprint(mod, step, provenance, secondWeight)
	if firstWeight == secondWeight || firstContract == secondContract {
		t.Fatalf("shape-compatible weight mutation did not change fingerprints: first=%s/%s second=%s/%s", firstWeight, firstContract, secondWeight, secondContract)
	}
}

func TestBERTCUDASelectedPackageProvenanceRejectsSpoofedComponents(t *testing.T) {
	tests := []struct {
		name string
		key  string
		bad  any
		want string
	}{
		{name: "verified", key: "pretrained_bert_package_provenance_verified", bad: false, want: "not verified"},
		{name: "version", key: "pretrained_bert_package_provenance_version", bad: "wrong.version", want: "provenance_version"},
		{name: "package sha", key: "pretrained_bert_package_sha256", bad: strings.Repeat("0", 64), want: "package_sha256"},
		{name: "identity sha", key: "pretrained_bert_package_identity_sha256", bad: strings.Repeat("1", 64), want: "identity_sha256"},
		{name: "module sha", key: "pretrained_bert_package_module_sha256", bad: strings.Repeat("2", 64), want: "module_sha256"},
		{name: "weights sha", key: "pretrained_bert_package_weights_sha256", bad: strings.Repeat("3", 64), want: "weights_sha256"},
		{name: "generation missing", key: "pretrained_bert_package_weight_set_generation", bad: "", want: "weight_set_generation"},
		{name: "role schema", key: "pretrained_bert_retrieval_role_schema", bad: "wrong.schema", want: "role_schema"},
		{name: "query role", key: "pretrained_bert_retrieval_query_role", bad: "search", want: "query_role"},
		{name: "document role", key: "pretrained_bert_retrieval_document_role", bad: "passage", want: "document_role"},
		{name: "query prefix", key: "pretrained_bert_retrieval_query_prefix", bad: "query: ", want: "query_prefix"},
		{name: "document prefix", key: "pretrained_bert_retrieval_document_prefix", bad: "passage: ", want: "document_prefix"},
		{name: "pooling", key: "pretrained_bert_retrieval_pooling", bad: "masked_mean", want: "pooling"},
		{name: "normalization", key: "pretrained_bert_retrieval_normalization", bad: "none", want: "normalization"},
		{name: "max length", key: "pretrained_bert_retrieval_max_length", bad: 256, want: "max_length"},
		{name: "native dim", key: "pretrained_bert_retrieval_native_dim", bad: 768, want: "native_dim"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mod := validBGEContractModule()
			mod.Metadata[tt.key] = tt.bad
			_, err := validateSelectedBGEPackageProvenance(mod)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v, want rejection mentioning %q", err, tt.want)
			}
		})
	}
}

func validBGEFoundationFixture(t *testing.T) (eosartifact.Step, []*backend.Tensor) {
	t.Helper()
	specs := bgeCUDAWeightSpecs()
	inputs := make([]*backend.Tensor, 3+len(specs))
	inputs[0] = backend.NewTensorI32([]int{1, 2}, []int32{101, 102})
	inputs[1] = backend.NewTensorI32([]int{1, 2}, []int32{1, 1})
	inputs[2] = backend.NewTensorI32([]int{1, 2}, []int32{0, 0})

	names := make([]string, len(inputs))
	names[0] = "input_ids"
	names[1] = "attention_mask"
	names[2] = "token_type_ids"
	for _, spec := range specs {
		inputs[spec.inputIndex] = newCUDAFoundationTensor("f32", spec.shape)
		if spec.layer >= 0 {
			names[spec.inputIndex] = fmt.Sprintf("encoder_layer_%d_%s", spec.layer, spec.slot)
		} else {
			names[spec.inputIndex] = spec.slot
		}
	}

	return eosartifact.Step{
		Entry:   "bert_embed",
		Kind:    eosartifact.StepBERTEmbedder,
		Name:    "bert_embedder_reference",
		Inputs:  names,
		Outputs: []string{"embeddings"},
		Attributes: map[string]string{
			"num_hidden_layers":   "12",
			"num_attention_heads": "12",
			"hidden_act":          "gelu",
			"pooling":             "cls",
			"normalization":       "l2",
			"epsilon":             "1e-12",
		},
	}, inputs
}

func validBGEContractModule() *eosartifact.Module {
	return &eosartifact.Module{
		Name: "pretrained_bert_embedder",
		Metadata: map[string]any{
			"model_name":    selectedBGEModelName,
			"architecture":  selectedBGEArchitecture,
			"pooling":       "cls",
			"normalization": "l2",
			"pretrained_bert_package_provenance_version":    "manta/pretrained-bert-package-runtime-provenance/v1",
			"pretrained_bert_package_provenance_verified":   true,
			"pretrained_bert_package_sha256":                selectedBGEPackageSHA256,
			"pretrained_bert_package_identity_sha256":       selectedBGEPackageIdentitySHA256,
			"pretrained_bert_package_module_sha256":         selectedBGEModuleSHA256,
			"pretrained_bert_package_weights_sha256":        selectedBGEWeightsSHA256,
			"pretrained_bert_package_weight_set_generation": "fixture-generation",
			"pretrained_bert_retrieval_role_schema":         "manta.pretrained_bert_retrieval_role_contract.v1",
			"pretrained_bert_retrieval_query_role":          "query",
			"pretrained_bert_retrieval_document_role":       "document",
			"pretrained_bert_retrieval_query_prefix":        selectedBGEQueryPrefix,
			"pretrained_bert_retrieval_document_prefix":     selectedBGEDocumentPrefix,
			"pretrained_bert_retrieval_pooling":             "cls",
			"pretrained_bert_retrieval_normalization":       "l2",
			"pretrained_bert_retrieval_max_length":          bgeSmallMaxPositions,
			"pretrained_bert_retrieval_native_dim":          bgeSmallHiddenSize,
		},
	}
}

func newCUDAFoundationTensor(dtype string, shape []int) *backend.Tensor {
	elements := 1
	for _, dim := range shape {
		elements *= dim
	}
	return &backend.Tensor{DType: dtype, Shape: append([]int(nil), shape...), F32: make([]float32, elements)}
}
