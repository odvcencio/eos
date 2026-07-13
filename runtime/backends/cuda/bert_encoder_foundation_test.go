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
		Kind:   eosartifact.StepBERTEmbedder,
		Inputs: names,
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

func newCUDAFoundationTensor(dtype string, shape []int) *backend.Tensor {
	elements := 1
	for _, dim := range shape {
		elements *= dim
	}
	return &backend.Tensor{DType: dtype, Shape: append([]int(nil), shape...), F32: make([]float32, elements)}
}
