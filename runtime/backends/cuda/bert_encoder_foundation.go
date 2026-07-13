package cuda

import (
	"fmt"
	"strconv"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const (
	bgeSmallVocabSize        = 30522
	bgeSmallHiddenSize       = 384
	bgeSmallLayers           = 12
	bgeSmallHeads            = 12
	bgeSmallIntermediateSize = 1536
	bgeSmallMaxPositions     = 512
	bgeSmallTypeVocabSize    = 2
	bgeSmallLayerNormEpsilon = 1e-12

	pretrainedBERTCUDAFoundationContract = "full_device_pretrained_bert_encoder.v1"
)

type bertCUDAEncoderFoundationStatus struct {
	Contract          string
	ShapeValidated    bool
	WeightsPlanned    bool
	FullDeviceReady   bool
	MissingComponents []string
}

func newBERTCUDAEncoderFoundationStatus() bertCUDAEncoderFoundationStatus {
	return bertCUDAEncoderFoundationStatus{
		Contract:          pretrainedBERTCUDAFoundationContract,
		FullDeviceReady:   false,
		MissingComponents: []string{"dense self-attention context", "no-download resident GEMM composition", "full StepBERTEmbedder CUDA dispatch"},
	}
}

type bertCUDAWeightRole int

const (
	bertCUDAWeightEmbedding bertCUDAWeightRole = iota
	bertCUDAWeightVector
	bertCUDAWeightDenseMatrix
	bertCUDAWeightDenseBias
)

type bertCUDAResidentWeight struct {
	Name       string
	Role       bertCUDAWeightRole
	Shape      []int
	InputIndex int
	Layer      int
	Slot       string
}

type bertCUDAResidentWeightPlan struct {
	Weights []bertCUDAResidentWeight
}

func validateBGEPretrainedBERTEmbedderStep(step eosartifact.Step, inputs []*backend.Tensor) (bertCUDAEncoderFoundationStatus, error) {
	status := newBERTCUDAEncoderFoundationStatus()
	if step.Kind != eosartifact.StepBERTEmbedder {
		return status, fmt.Errorf("cuda bert foundation supports only StepBERTEmbedder, got %s", step.Kind)
	}
	layers, err := bertStepIntAttr(step.Attributes, "num_hidden_layers", 1)
	if err != nil {
		return status, err
	}
	heads, err := bertStepIntAttr(step.Attributes, "num_attention_heads", 1)
	if err != nil {
		return status, err
	}
	epsilon, err := bertStepFloatAttr(step.Attributes, "epsilon", bgeSmallLayerNormEpsilon)
	if err != nil {
		return status, err
	}
	hiddenAct := step.Attributes["hidden_act"]
	if hiddenAct == "" {
		hiddenAct = "gelu"
	}
	pooling := step.Attributes["pooling"]
	if pooling == "" {
		pooling = "masked_mean"
	}
	normalization := step.Attributes["normalization"]
	if normalization == "" {
		normalization = "l2"
	}
	if layers != bgeSmallLayers || heads != bgeSmallHeads || hiddenAct != "gelu" || pooling != "cls" || normalization != "l2" || epsilon != bgeSmallLayerNormEpsilon {
		return status, fmt.Errorf("unsupported BGE CUDA encoder attrs: layers=%d heads=%d hidden_act=%q pooling=%q normalization=%q epsilon=%g", layers, heads, hiddenAct, pooling, normalization, epsilon)
	}
	expected := 8 + bgeSmallLayers*16
	if len(inputs) != expected {
		return status, fmt.Errorf("BGE CUDA encoder expects %d tensors, got %d", expected, len(inputs))
	}
	if err := validateBGEInputIDs(inputs[0], "input_ids"); err != nil {
		return status, err
	}
	if err := validateBGEInputIDs(inputs[1], "attention_mask"); err != nil {
		return status, err
	}
	if err := validateBGEInputIDs(inputs[2], "token_type_ids"); err != nil {
		return status, err
	}
	if !inputs[0].EqualShape(inputs[1]) || !inputs[0].EqualShape(inputs[2]) {
		return status, fmt.Errorf("BGE CUDA encoder input_id, attention_mask, token_type_id shapes must match")
	}
	if inputs[0].Shape[1] > bgeSmallMaxPositions {
		return status, fmt.Errorf("BGE CUDA encoder token length %d exceeds max positions %d", inputs[0].Shape[1], bgeSmallMaxPositions)
	}
	for _, check := range bgeCUDAWeightSpecs()[:5] {
		if err := validateBGEFloatTensor(inputs[check.inputIndex], check.slot, check.shape); err != nil {
			return status, err
		}
	}
	for _, check := range bgeCUDAWeightSpecs()[5:] {
		if err := validateBGEFloatTensor(inputs[check.inputIndex], fmt.Sprintf("encoder_layer_%d_%s", check.layer, check.slot), check.shape); err != nil {
			return status, err
		}
	}
	status.ShapeValidated = true
	return status, nil
}

func planBGEPretrainedBERTResidentWeights(step eosartifact.Step, inputs []*backend.Tensor) (bertCUDAResidentWeightPlan, bertCUDAEncoderFoundationStatus, error) {
	status, err := validateBGEPretrainedBERTEmbedderStep(step, inputs)
	if err != nil {
		return bertCUDAResidentWeightPlan{}, status, err
	}
	if len(step.Inputs) != len(inputs) {
		return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight planning requires %d stable input names, got %d", len(inputs), len(step.Inputs))
	}
	plan := bertCUDAResidentWeightPlan{Weights: make([]bertCUDAResidentWeight, 0, len(inputs)-3)}
	seen := map[string]int{}
	for _, spec := range bgeCUDAWeightSpecs() {
		name := step.Inputs[spec.inputIndex]
		if name == "" {
			return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight input %d (%s) is missing a stable artifact name", spec.inputIndex, spec.slot)
		}
		if previous, ok := seen[name]; ok {
			return bertCUDAResidentWeightPlan{}, status, fmt.Errorf("BGE CUDA encoder resident weight name %q is duplicated at inputs %d and %d", name, previous, spec.inputIndex)
		}
		seen[name] = spec.inputIndex
		plan.Weights = append(plan.Weights, bertCUDAResidentWeight{
			Name:       name,
			Role:       spec.role,
			Shape:      append([]int(nil), inputs[spec.inputIndex].Shape...),
			InputIndex: spec.inputIndex,
			Layer:      spec.layer,
			Slot:       spec.slot,
		})
	}
	status.WeightsPlanned = true
	return plan, status, nil
}

type bertCUDAWeightSpec struct {
	inputIndex int
	layer      int
	slot       string
	shape      []int
	role       bertCUDAWeightRole
}

func bgeCUDAWeightSpecs() []bertCUDAWeightSpec {
	specs := []bertCUDAWeightSpec{
		{3, -1, "token_embeddings", []int{bgeSmallVocabSize, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{4, -1, "position_embeddings", []int{bgeSmallMaxPositions, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{5, -1, "token_type_embeddings", []int{bgeSmallTypeVocabSize, bgeSmallHiddenSize}, bertCUDAWeightEmbedding},
		{6, -1, "embedding_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{7, -1, "embedding_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
	}
	layerSlots := []struct {
		slot  string
		shape []int
		role  bertCUDAWeightRole
	}{
		{"attention_query_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_query_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_key_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_key_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_value_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_value_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_output_weight", []int{bgeSmallHiddenSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"attention_output_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"attention_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"attention_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"intermediate_weight", []int{bgeSmallIntermediateSize, bgeSmallHiddenSize}, bertCUDAWeightDenseMatrix},
		{"intermediate_bias", []int{bgeSmallIntermediateSize}, bertCUDAWeightDenseBias},
		{"output_weight", []int{bgeSmallHiddenSize, bgeSmallIntermediateSize}, bertCUDAWeightDenseMatrix},
		{"output_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightDenseBias},
		{"output_layernorm_weight", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
		{"output_layernorm_bias", []int{bgeSmallHiddenSize}, bertCUDAWeightVector},
	}
	for layer := 0; layer < bgeSmallLayers; layer++ {
		for offset, slot := range layerSlots {
			specs = append(specs, bertCUDAWeightSpec{8 + layer*16 + offset, layer, slot.slot, slot.shape, slot.role})
		}
	}
	return specs
}

func bertStepIntAttr(attrs map[string]string, name string, defaultValue int) (int, error) {
	if attrs == nil || attrs[name] == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(attrs[name])
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("BGE CUDA encoder attr %s=%q is invalid", name, attrs[name])
	}
	return value, nil
}

func bertStepFloatAttr(attrs map[string]string, name string, defaultValue float64) (float64, error) {
	if attrs == nil || attrs[name] == "" {
		return defaultValue, nil
	}
	value, err := strconv.ParseFloat(attrs[name], 64)
	if err != nil || value < 0 {
		return 0, fmt.Errorf("BGE CUDA encoder attr %s=%q is invalid", name, attrs[name])
	}
	return value, nil
}

func validateBGEInputIDs(t *backend.Tensor, name string) error {
	if t == nil || t.DType != "i32" || len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || t.Elements() != len(t.I32) {
		return fmt.Errorf("BGE CUDA encoder %s must be i32 [B,T] with matching backing data", name)
	}
	return nil
}

func validateBGEFloatTensor(t *backend.Tensor, name string, shape []int) error {
	if t == nil {
		return fmt.Errorf("BGE CUDA encoder %s must be f32-compatible shape %v", name, shape)
	}
	if t.DType != "f32" && t.DType != "f16" {
		return fmt.Errorf("BGE CUDA encoder %s dtype %q is not f32-compatible", name, t.DType)
	}
	if len(t.Shape) != len(shape) || t.Elements() != len(t.F32) {
		return fmt.Errorf("BGE CUDA encoder %s must be f32-compatible shape %v", name, shape)
	}
	for i := range shape {
		if t.Shape[i] != shape[i] {
			return fmt.Errorf("BGE CUDA encoder %s shape %v, want %v", name, t.Shape, shape)
		}
	}
	return nil
}
