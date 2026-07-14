package cuda

import (
	"context"
	"strings"
	"sync"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

func TestBERTCUDA12LayerHiddenGateEnv(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "")
	if bertCUDA12LayerHiddenGateEnabled() {
		t.Fatal("hidden gate should default off")
	}
	for _, value := range []string{"1", "true", "TRUE", "yes", "YES", "on", "ON"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", value)
			if !bertCUDA12LayerHiddenGateEnabled() {
				t.Fatalf("hidden gate value %q should enable", value)
			}
		})
	}
	t.Setenv("EOS_BERT_CUDA_MAX_BATCH_TOKENS", "bad")
	if _, err := bertCUDAMaxBatchTokensFromEnv(17); err == nil || !strings.Contains(err.Error(), "EOS_BERT_CUDA_MAX_BATCH_TOKENS") {
		t.Fatalf("invalid max batch tokens error=%v, want env-specific failure", err)
	}
	t.Setenv("EOS_BERT_CUDA_MAX_BATCH_TOKENS", "19")
	got, err := bertCUDAMaxBatchTokensFromEnv(17)
	if err != nil || got != 19 {
		t.Fatalf("max batch tokens=%d err=%v, want 19 nil", got, err)
	}
}

func TestBERTCUDA12LayerHiddenGateOffDoesNotHandleStepBERTEmbedder(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "")
	exec := &executor{}
	result, handled, err := exec.dispatchStep(context.Background(), eosartifact.Step{Kind: eosartifact.StepBERTEmbedder}, eosartifact.ValueType{}, nil)
	if err != nil || handled {
		t.Fatalf("dispatch result=%+v handled=%t err=%v, want unhandled nil", result, handled, err)
	}
}

func TestBERTCUDA12LayerHiddenGateOnFailsClosedWithoutDevice(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "1")
	exec := &executor{}
	_, handled, err := exec.dispatchStep(context.Background(), eosartifact.Step{Kind: eosartifact.StepBERTEmbedder}, eosartifact.ValueType{}, []*backend.Tensor{})
	if !handled || err == nil || !strings.Contains(err.Error(), "device runtime is unavailable") {
		t.Fatalf("handled=%t err=%v, want fail-closed missing-device error", handled, err)
	}
}

func TestBERTCUDA12LayerHiddenDispatchConcurrentFailClosedRaceOriented(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "1")
	step, inputs := validBGEFoundationFixture(t)
	exec := &executor{
		module: validBGEContractModule(),
		device: &deviceRuntime{
			residentMatrices:    map[string]residentMatrix{},
			bertResidentTensors: map[string]residentTensor{},
			bertResidentCache:   map[string]bertCUDAResidentBindingCache{},
		},
	}
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for i := 0; i < cap(errs); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, err := exec.dispatchStep(context.Background(), step, eosartifact.ValueType{}, inputs)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err == nil {
			t.Fatal("concurrent hidden dispatch unexpectedly succeeded for non-selected fixture")
		}
		if !strings.Contains(err.Error(), "contract fingerprint mismatch") && !strings.Contains(err.Error(), "cuda runtime is not initialized") {
			t.Fatalf("concurrent hidden dispatch err=%v, want fail-closed contract/runtime error", err)
		}
	}
}
