//go:build linux && cgo

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backend"
)

func TestBERTCUDA12LayerPreflightRejectsBatchTokenGuardBeforeResidentChecks(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_MAX_BATCH_TOKENS", "3")
	step, inputs := validBGEFoundationFixture(t)
	inputs[0].Shape = []int{2, 2}
	inputs[0].I32 = []int32{101, 102, 101, 102}
	inputs[1].Shape = []int{2, 2}
	inputs[1].I32 = []int32{1, 1, 1, 1}
	inputs[2].Shape = []int{2, 2}
	inputs[2].I32 = []int32{0, 0, 0, 0}

	rt := &deviceRuntime{residentMatrices: map[string]residentMatrix{}, bertResidentTensors: map[string]residentTensor{}}
	_, stats, err := rt.preflightBGEFullEncoder(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "batch tokens 4 exceed") {
		t.Fatalf("err=%v, want batch-token guard failure", err)
	}
	if stats.MaxBatchTokens != 3 {
		t.Fatalf("max batch tokens=%d, want 3", stats.MaxBatchTokens)
	}
	if stats.UploadedBytes != 0 || stats.DownloadedBytes != 0 {
		t.Fatalf("preflight failure must not transfer bytes: %+v", stats)
	}
}

func TestBERTCUDA12LayerPreflightFailsClosedOnMissingResidentWeights(t *testing.T) {
	t.Setenv("EOS_BERT_CUDA_MAX_BATCH_TOKENS", "512")
	step, inputs := validBGEFoundationFixture(t)
	rt := &deviceRuntime{residentMatrices: map[string]residentMatrix{}, bertResidentTensors: map[string]residentTensor{}}
	_, stats, err := rt.preflightBGEFullEncoder(step, inputs)
	if err == nil || !strings.Contains(err.Error(), "is not bound") {
		t.Fatalf("err=%v, want missing resident weight failure", err)
	}
	if stats.UploadedBytes != 0 || stats.DownloadedBytes != 0 || stats.IntermediateDownloadedBytes != 0 {
		t.Fatalf("missing-resident preflight must not transfer bytes: %+v", stats)
	}
}

func TestBERTCUDASelectedPackageContractFingerprint(t *testing.T) {
	if os.Getenv("EOS_BERT_CUDA_SELECTED_PACKAGE_CONTRACT") == "" {
		t.Skip("set EOS_BERT_CUDA_SELECTED_PACKAGE_CONTRACT=1 to verify the selected BGE CUDA contract fingerprint")
	}
	mod, weights := loadSelectedBGEPackageModuleAndWeights(t)
	step := selectedBGEStep(t, mod)
	inputs := selectedBGEContractInputs(t, step, weights)
	contract, err := validateSelectedBGECUDAContract(mod, step, inputs)
	if err != nil {
		t.Fatalf("selected BGE CUDA contract: %v", err)
	}
	if selectedBGEContractFingerprintSHA256 == "TODO" {
		t.Fatalf("selectedBGEContractFingerprintSHA256 is not pinned; computed contract=%s weight=%s", contract.ContractFingerprint, contract.WeightFingerprint)
	}
	if contract.ContractFingerprint != selectedBGEContractFingerprintSHA256 {
		t.Fatalf("contract fingerprint=%s want %s weight=%s", contract.ContractFingerprint, selectedBGEContractFingerprintSHA256, contract.WeightFingerprint)
	}
	t.Logf("selected BGE contract fingerprint=%s weight_fingerprint=%s", contract.ContractFingerprint, contract.WeightFingerprint)
}

func TestBERTCUDASelectedPackageHidden12LayerParityAndAccounting(t *testing.T) {
	if os.Getenv("EOS_BERT_CUDA_SELECTED_PACKAGE_PARITY") == "" {
		t.Skip("set EOS_BERT_CUDA_SELECTED_PACKAGE_PARITY=1 to run selected-package hidden CUDA parity")
	}
	packagePath := selectedBGEPackagePath(t)
	ctx := context.Background()
	texts := []string{"what is cuda?", "small embedding parity check"}

	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "")
	hostEmbedder, err := eosruntime.LoadImportedBERTEmbedderCandidate(ctx, packagePath, eosruntime.New(New()))
	if err != nil {
		t.Fatalf("load host selected package: %v", err)
	}
	hostStart := time.Now()
	hostVectors, hostPrefix, err := hostEmbedder.EmbedTextBatchWithRole(ctx, texts, "query")
	if err != nil {
		t.Fatalf("host embed selected package: %v", err)
	}
	hostNanos := time.Since(hostStart).Nanoseconds()
	if hostPrefix != selectedBGEQueryPrefix {
		t.Fatalf("host query prefix=%q want selected prefix", hostPrefix)
	}

	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "1")
	cudaEmbedder, err := eosruntime.LoadImportedBERTEmbedderCandidate(ctx, packagePath, eosruntime.New(New()))
	if err != nil {
		t.Fatalf("load hidden CUDA selected package: %v", err)
	}
	coldStart := time.Now()
	cudaVectors, cudaPrefix, err := cudaEmbedder.EmbedTextBatchWithRole(ctx, texts, "query")
	if err != nil {
		t.Fatalf("hidden CUDA embed selected package: %v", err)
	}
	coldNanos := time.Since(coldStart).Nanoseconds()
	if cudaPrefix != selectedBGEQueryPrefix {
		t.Fatalf("cuda query prefix=%q want selected prefix", cudaPrefix)
	}
	warmStart := time.Now()
	warmVectors, _, err := cudaEmbedder.EmbedTextBatchWithRole(ctx, texts, "query")
	if err != nil {
		t.Fatalf("warm hidden CUDA embed selected package: %v", err)
	}
	warmNanos := time.Since(warmStart).Nanoseconds()
	assertEmbeddingRowsClose(t, cudaVectors, hostVectors, "cold hidden vs host")
	assertEmbeddingRowsClose(t, warmVectors, hostVectors, "warm hidden vs host")

	mod, weights := loadSelectedBGEPackageModuleAndWeights(t)
	step := selectedBGEStep(t, mod)
	inputs := selectedBGEContractInputs(t, step, weights)
	expectedProvenance, err := validateSelectedBGEPackageProvenance(mod)
	if err != nil {
		t.Fatalf("selected package provenance: %v", err)
	}
	loadOptions := append(weights.LoadOptions(), eosruntime.WithRequireBackend(eosartifact.BackendCUDA))
	program, err := eosruntime.New(New()).Load(ctx, mod, loadOptions...)
	if err != nil {
		t.Fatalf("load direct selected package program: %v", err)
	}
	result, err := program.Run(ctx, backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      inputs[0],
			"attention_mask": inputs[1],
			"token_type_ids": inputs[2],
		},
	})
	if err != nil {
		t.Fatalf("direct hidden CUDA selected run: %v", err)
	}
	meta := result.Outputs["embeddings"].Metadata
	if meta["execution_mode"] != "pretrained_bert_cuda_hidden_resident_12layer" {
		t.Fatalf("execution_mode=%v metadata=%+v", meta["execution_mode"], meta)
	}
	if meta["full_device_execution"] != false || meta["device_encoder_contract_satisfied"] != false {
		t.Fatalf("public hidden contract flags changed: %+v", meta)
	}
	requirePositiveInt64Metadata(t, meta, "input_uploaded_bytes")
	requirePositiveInt64Metadata(t, meta, "resident_uploaded_bytes")
	requirePositiveInt64Metadata(t, meta, "final_downloaded_bytes")
	requirePositiveInt64Metadata(t, meta, "status_downloaded_bytes")
	if got := metadataInt64ForTest(meta, "intermediate_downloaded_bytes"); got != 0 {
		t.Fatalf("intermediate_downloaded_bytes=%d want 0", got)
	}
	if meta["contract_fingerprint_sha256"] != selectedBGEContractFingerprintSHA256 {
		t.Fatalf("contract fingerprint metadata=%v want %s", meta["contract_fingerprint_sha256"], selectedBGEContractFingerprintSHA256)
	}
	assertSelectedBGEProvenanceMetadata(t, meta, expectedProvenance)

	result, err = program.Run(ctx, backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      inputs[0],
			"attention_mask": inputs[1],
			"token_type_ids": inputs[2],
		},
	})
	if err != nil {
		t.Fatalf("direct warm hidden CUDA selected run: %v", err)
	}
	warmMeta := result.Outputs["embeddings"].Metadata
	if got := metadataInt64ForTest(warmMeta, "resident_uploaded_bytes"); got != 0 {
		t.Fatalf("warm resident_uploaded_bytes=%d want 0 metadata=%+v", got, warmMeta)
	}
	if got := metadataInt64ForTest(warmMeta, "resident_cache_hits"); got <= 0 {
		t.Fatalf("warm resident_cache_hits=%d want positive metadata=%+v", got, warmMeta)
	}
	assertSelectedBGEProvenanceMetadata(t, warmMeta, expectedProvenance)
	t.Logf("selected-package hidden CUDA parity max_abs<=5e-4 min_cos>=0.99999 host_ns=%d cold_ns=%d warm_ns=%d direct_cold=%+v direct_warm=%+v", hostNanos, coldNanos, warmNanos, meta, warmMeta)
}

func TestBERTCUDASelectedPackageHidden12LayerConcurrentInference(t *testing.T) {
	if os.Getenv("EOS_BERT_CUDA_SELECTED_PACKAGE_CONCURRENCY") == "" {
		t.Skip("set EOS_BERT_CUDA_SELECTED_PACKAGE_CONCURRENCY=1 to run selected-package hidden CUDA concurrency")
	}
	t.Setenv("EOS_BERT_CUDA_12LAYER_HIDDEN", "1")
	ctx := context.Background()
	mod, weights := loadSelectedBGEPackageModuleAndWeights(t)
	step := selectedBGEStep(t, mod)
	inputs := selectedBGEContractInputs(t, step, weights)
	loadOptions := append(weights.LoadOptions(), eosruntime.WithRequireBackend(eosartifact.BackendCUDA))
	program, err := eosruntime.New(New()).Load(ctx, mod, loadOptions...)
	if err != nil {
		t.Fatalf("load direct selected package program: %v", err)
	}
	runOnce := func() (*backend.Tensor, map[string]any, error) {
		result, err := program.Run(ctx, backend.Request{
			Entry: "bert_embed",
			Inputs: map[string]any{
				"input_ids":      inputs[0],
				"attention_mask": inputs[1],
				"token_type_ids": inputs[2],
			},
		})
		if err != nil {
			return nil, nil, err
		}
		value := result.Outputs["embeddings"]
		tensor, ok := value.Data.(*backend.Tensor)
		if !ok || tensor == nil {
			return nil, nil, fmt.Errorf("output embeddings type %T, want *backend.Tensor", value.Data)
		}
		return tensor, value.Metadata, nil
	}
	want, coldMeta, err := runOnce()
	if err != nil {
		t.Fatalf("cold selected package hidden run: %v", err)
	}
	if metadataInt64ForTest(coldMeta, "resident_uploaded_bytes") <= 0 {
		t.Fatalf("cold resident upload metadata=%+v, want positive", coldMeta)
	}

	const workers = 4
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			got, meta, err := runOnce()
			if err != nil {
				errs <- fmt.Errorf("worker %d: %w", worker, err)
				return
			}
			if metadataInt64ForTest(meta, "resident_uploaded_bytes") != 0 {
				errs <- fmt.Errorf("worker %d resident_uploaded_bytes=%d, want 0 metadata=%+v", worker, metadataInt64ForTest(meta, "resident_uploaded_bytes"), meta)
				return
			}
			if metadataInt64ForTest(meta, "resident_cache_hits") <= 0 || meta["contract_cache_hit"] != true {
				errs <- fmt.Errorf("worker %d warm cache metadata=%+v, want resident hits and contract_cache_hit", worker, meta)
				return
			}
			rowMax, rowCos := bertVectorMaxAbsAndCos(got.F32, want.F32)
			if rowMax > 5e-4 || rowCos < 0.99999 {
				errs <- fmt.Errorf("worker %d parity max_abs=%g min_cos=%g", worker, rowMax, rowCos)
				return
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func selectedBGEPackagePath(t *testing.T) string {
	t.Helper()
	candidates := []string{
		os.Getenv("EOS_BERT_CUDA_SELECTED_PACKAGE_PATH"),
		filepath.Join("runs", "pretrained-bert-current-hf-parity-v1-20260629T090818Z", "bge", "bge-small-en-v1.5.imported.mll"),
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	t.Skip("selected BGE package unavailable; set EOS_BERT_CUDA_SELECTED_PACKAGE_PATH")
	return ""
}

func loadSelectedBGEPackageModuleAndWeights(t *testing.T) (*eosartifact.Module, eosruntime.WeightFile) {
	t.Helper()
	packagePath := selectedBGEPackagePath(t)
	artifacts, err := eosruntime.LoadPretrainedBERTPackageRuntimeArtifacts(packagePath)
	if err != nil {
		t.Fatalf("read selected package: %v", err)
	}
	pkg := artifacts.Package
	if pkg.IdentitySHA256 != selectedBGEPackageIdentitySHA256 || pkg.ModelName != selectedBGEModelName || pkg.Pooling != "cls" || pkg.Normalization != "l2" || pkg.MaxLength != bgeSmallMaxPositions {
		t.Fatalf("selected package identity/contract mismatch: identity=%s model=%q pooling=%q normalization=%q max_length=%d", pkg.IdentitySHA256, pkg.ModelName, pkg.Pooling, pkg.Normalization, pkg.MaxLength)
	}
	if artifacts.PackageSHA256 != selectedBGEPackageSHA256 {
		t.Fatalf("selected package sha=%s want %s", artifacts.PackageSHA256, selectedBGEPackageSHA256)
	}
	if artifacts.WeightSetGeneration == "" {
		t.Fatal("selected package runtime weight generation is empty")
	}
	return artifacts.Module, artifacts.Weights
}

func selectedBGEStep(t *testing.T, mod *eosartifact.Module) eosartifact.Step {
	t.Helper()
	for _, step := range mod.Steps {
		if step.Kind == eosartifact.StepBERTEmbedder {
			return step
		}
	}
	t.Fatal("selected package module has no StepBERTEmbedder")
	return eosartifact.Step{}
}

func selectedBGEContractInputs(t *testing.T, step eosartifact.Step, weights eosruntime.WeightFile) []*backend.Tensor {
	t.Helper()
	inputs := make([]*backend.Tensor, len(step.Inputs))
	inputs[0] = backend.NewTensorI32([]int{1, 2}, []int32{101, 102})
	inputs[1] = backend.NewTensorI32([]int{1, 2}, []int32{1, 1})
	inputs[2] = backend.NewTensorI32([]int{1, 2}, []int32{0, 0})
	for i := 3; i < len(step.Inputs); i++ {
		tensor := weights.Weights[step.Inputs[i]]
		if tensor == nil {
			t.Fatalf("selected package missing weight %q", step.Inputs[i])
		}
		inputs[i] = tensor
	}
	return inputs
}

func assertEmbeddingRowsClose(t *testing.T, got, want [][]float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows=%d want %d", label, len(got), len(want))
	}
	maxAbs := float32(0)
	minCos := float32(1)
	for row := range got {
		if len(got[row]) != len(want[row]) {
			t.Fatalf("%s row %d dims=%d want %d", label, row, len(got[row]), len(want[row]))
		}
		rowMax, rowCos := bertVectorMaxAbsAndCos(got[row], want[row])
		if rowMax > maxAbs {
			maxAbs = rowMax
		}
		if rowCos < minCos {
			minCos = rowCos
		}
	}
	if maxAbs > 5e-4 || minCos < 0.99999 {
		t.Fatalf("%s parity max_abs=%g min_cos=%g want <=5e-4/>=0.99999", label, maxAbs, minCos)
	}
}

func bertVectorMaxAbsAndCos(a, b []float32) (float32, float32) {
	var maxAbs, dot, na, nb float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	cos := float64(1)
	if na > 0 && nb > 0 {
		cos = dot / math.Sqrt(na*nb)
	}
	return float32(maxAbs), float32(cos)
}

func requirePositiveInt64Metadata(t *testing.T, meta map[string]any, key string) {
	t.Helper()
	if got := metadataInt64ForTest(meta, key); got <= 0 {
		t.Fatalf("%s=%d want positive metadata=%+v", key, got, meta)
	}
}

func assertSelectedBGEProvenanceMetadata(t *testing.T, meta map[string]any, expected bertCUDASelectedPackageProvenance) {
	t.Helper()
	for _, item := range []struct {
		key  string
		want string
	}{
		{"package_sha256", expected.PackageSHA256},
		{"package_identity_sha256", expected.PackageIdentity},
		{"module_sha256", expected.ModuleSHA256},
		{"weights_sha256", expected.WeightsSHA256},
		{"weight_set_generation", expected.WeightSetGeneration},
		{"retrieval_role_schema", expected.RoleSchema},
		{"retrieval_query_role", expected.QueryRole},
		{"retrieval_document_role", expected.DocumentRole},
		{"retrieval_query_prefix", expected.QueryPrefix},
		{"retrieval_document_prefix", expected.DocumentPrefix},
		{"pooling", expected.Pooling},
		{"normalization", expected.Normalization},
	} {
		if got, _ := meta[item.key].(string); got != item.want {
			t.Fatalf("%s metadata=%q want %q metadata=%+v", item.key, got, item.want, meta)
		}
	}
	if got := metadataInt64ForTest(meta, "max_length"); got != int64(expected.MaxLength) {
		t.Fatalf("max_length metadata=%d want %d metadata=%+v", got, expected.MaxLength, meta)
	}
	if got := metadataInt64ForTest(meta, "native_dim"); got != int64(expected.NativeDim) {
		t.Fatalf("native_dim metadata=%d want %d metadata=%+v", got, expected.NativeDim, meta)
	}
}

func metadataInt64ForTest(meta map[string]any, key string) int64 {
	switch v := meta[key].(type) {
	case int:
		return int64(v)
	case int64:
		return v
	case float64:
		return int64(v)
	case string:
		var parsed int64
		_, _ = fmt.Sscanf(v, "%d", &parsed)
		return parsed
	default:
		return 0
	}
}
