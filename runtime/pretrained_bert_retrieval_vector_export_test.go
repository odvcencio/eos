package eosruntime

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	mll "m31labs.dev/mll"
)

func TestPretrainedBERTRetrievalVectorExportWritesEvaluatorCompatibleCaches(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	rt := New(cuda.New())
	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors: %v", err)
	}
	if summary.Documents != 1 || summary.Queries != 1 || summary.NativeDim != 2 || summary.OutputDim != 2 {
		t.Fatalf("summary counts/dims = %+v", summary)
	}
	if summary.ExecutionMode != "pretrained_bert_host_reference" || summary.QualityClaim {
		t.Fatalf("summary mode/quality = %+v", summary)
	}
	if summary.EncoderExecution.ExecutionMode != "pretrained_bert_host_reference" ||
		summary.EncoderExecution.FullDeviceExecution ||
		summary.EncoderExecution.ValidatedDeviceEncoder ||
		!summary.EncoderExecution.HostReference ||
		!summary.EncoderExecution.OpportunisticDeviceOpsIgnored {
		t.Fatalf("summary encoder execution provenance = %+v", summary.EncoderExecution)
	}
	if summary.QueryPrefix != "query " || summary.DocumentPrefix != "doc " || summary.LegacyDocPrefix != "doc " || summary.MaxLength != 4 || summary.BatchSize != 1 {
		t.Fatalf("summary config = %+v", summary)
	}
	if !summary.DocumentRoleApplied || !summary.QueryRoleApplied {
		t.Fatalf("summary role flags = doc:%v query:%v, want true/true", summary.DocumentRoleApplied, summary.QueryRoleApplied)
	}
	if summary.DocVectorPath != filepath.Join(outputDir, "doc-vectors.jsonl") ||
		summary.QueryVectorPath != filepath.Join(outputDir, "query-vectors.jsonl") {
		t.Fatalf("summary paths = %+v", summary)
	}

	docRows := readTinyVectorRows(t, summary.DocVectorPath)
	queryRows := readTinyVectorRows(t, summary.QueryVectorPath)
	if got := rowIDs(docRows); !slices.Equal(got, []string{"d1"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(queryRows); !slices.Equal(got, []string{"q1"}) {
		t.Fatalf("query ids = %v", got)
	}
	assertFiniteUnitishVector(t, docRows[0].Embedding, 2)
	assertFiniteUnitishVector(t, queryRows[0].Embedding, 2)

	manifestData, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.Schema != PretrainedBERTRetrievalVectorExportManifestSchema || manifest.SourceDir != sourceDir ||
		manifest.ModulePath != modulePath || manifest.WeightsPath != weightsPath {
		t.Fatalf("manifest = %+v", manifest)
	}
	if manifest.DocumentPrefix != "doc " || manifest.LegacyDocPrefix != "doc " || manifest.QueryPrefix != "query " {
		t.Fatalf("manifest prefixes = canonical:%q legacy:%q query:%q", manifest.DocumentPrefix, manifest.LegacyDocPrefix, manifest.QueryPrefix)
	}
	if !manifest.DocumentRoleApplied || !manifest.QueryRoleApplied {
		t.Fatalf("manifest role flags = doc:%v query:%v, want true/true", manifest.DocumentRoleApplied, manifest.QueryRoleApplied)
	}
	if !isSHA256Hex(manifest.EmbeddingSpaceID) || !isSHA256Hex(manifest.ModuleSHA256) || !isSHA256Hex(manifest.WeightsSHA256) ||
		!isSHA256Hex(manifest.ConfigSHA256) || !isSHA256Hex(manifest.VocabSHA256) || !isSHA256Hex(manifest.TokenizerConfigSHA256) {
		t.Fatalf("manifest provenance hashes = embedding:%q module:%q weights:%q config:%q vocab:%q tokenizer_config:%q",
			manifest.EmbeddingSpaceID, manifest.ModuleSHA256, manifest.WeightsSHA256, manifest.ConfigSHA256, manifest.VocabSHA256, manifest.TokenizerConfigSHA256)
	}
	if manifest.Normalization != "l2" {
		t.Fatalf("manifest normalization = %q, want l2", manifest.Normalization)
	}
	if manifest.EncoderExecution.ExecutionMode != "pretrained_bert_host_reference" ||
		manifest.EncoderExecution.SelectedBackend == "" ||
		manifest.EncoderExecution.DeviceEncoderContract != "full_device_pretrained_bert_encoder" ||
		manifest.EncoderExecution.DeviceEncoderContractSatisfied {
		t.Fatalf("manifest encoder execution provenance = %+v", manifest.EncoderExecution)
	}
	var manifestJSON map[string]any
	if err := json.Unmarshal(manifestData, &manifestJSON); err != nil {
		t.Fatalf("parse manifest json object: %v", err)
	}
	if manifestJSON["document_prefix"] != "doc " || manifestJSON["doc_prefix"] != "doc " {
		t.Fatalf("manifest json prefixes = document_prefix:%v doc_prefix:%v", manifestJSON["document_prefix"], manifestJSON["doc_prefix"])
	}
	if manifestJSON["document_role_applied"] != true || manifestJSON["query_role_applied"] != true {
		t.Fatalf("manifest json role flags = doc:%v query:%v", manifestJSON["document_role_applied"], manifestJSON["query_role_applied"])
	}
	encoderExecutionJSON, ok := manifestJSON["encoder_execution"].(map[string]any)
	if !ok || encoderExecutionJSON["execution_mode"] != "pretrained_bert_host_reference" ||
		encoderExecutionJSON["full_device_execution"] != false ||
		encoderExecutionJSON["opportunistic_device_ops_ignored"] != true {
		t.Fatalf("manifest json encoder_execution = %#v", manifestJSON["encoder_execution"])
	}
	if _, ok := encoderExecutionJSON["cuda_dense_acceleration_available"]; ok {
		t.Fatalf("manifest json encoder_execution must not infer cuda availability from policy/backend: %#v", encoderExecutionJSON)
	}

	_, _, qrelsPath := BEIRRetrievalPaths(datasetDir, "test")
	metrics, err := EvaluateVectorCacheRetrieval(context.Background(), RetrievalEvalConfig{
		DatasetName:          "tiny-bert",
		CorpusPath:           filepath.Join(datasetDir, "corpus.jsonl"),
		QueriesPath:          filepath.Join(datasetDir, "queries.jsonl"),
		QrelsPath:            qrelsPath,
		DocVectorPath:        summary.DocVectorPath,
		QueryVectorPath:      summary.QueryVectorPath,
		BackendName:          summary.ExecutionMode,
		TopK:                 1,
		AllowMissingRelevant: false,
	})
	if err != nil {
		t.Fatalf("evaluate vector cache: %v", err)
	}
	if metrics.Inputs.Documents != 1 || metrics.Inputs.Queries != 1 || metrics.Quality.HitAt1 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestPretrainedBERTEncoderExecutionProvenanceDoesNotInferCUDAAvailability(t *testing.T) {
	t.Setenv("EOS_BERT_DENSE_ACCEL", "")
	embedder := &PretrainedBERTTextEmbedder{
		program: &Program{executor: stubExecutor{kind: eosartifact.BackendCUDA}},
	}

	provenance := embedder.EncoderExecutionProvenance()
	if provenance.SelectedBackend != "cuda" || provenance.DenseAccelerationPolicy != "enabled" {
		t.Fatalf("backend/policy provenance = %+v, want cuda/enabled", provenance)
	}
	if provenance.CUDADenseAccelerationObserved || provenance.OpportunisticDeviceOpsUsed ||
		provenance.CUDADenseMatMulRuns != 0 || provenance.CUDADenseMatMulUploadedBytes != 0 {
		t.Fatalf("host-only provenance inferred CUDA acceleration from selected backend: %+v", provenance)
	}
	if provenance.FullDeviceExecution || provenance.ValidatedDeviceEncoder || provenance.DeviceEncoderContractSatisfied {
		t.Fatalf("host reference provenance satisfied device contract: %+v", provenance)
	}
	data, err := json.Marshal(provenance)
	if err != nil {
		t.Fatalf("marshal provenance: %v", err)
	}
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatalf("parse provenance json: %v", err)
	}
	if fields["selected_backend"] != "cuda" || fields["dense_acceleration_policy"] != "enabled" {
		t.Fatalf("provenance json backend/policy = %#v", fields)
	}
	if _, ok := fields["cuda_dense_acceleration_available"]; ok {
		t.Fatalf("provenance json must not include inferred availability field: %#v", fields)
	}
	if fields["cuda_dense_acceleration_observed"] != false || fields["opportunistic_device_ops_used"] != false {
		t.Fatalf("provenance json inferred observed acceleration: %#v", fields)
	}
}

func TestPretrainedBERTEncoderExecutionProvenanceRecordsObservedCUDACounters(t *testing.T) {
	t.Setenv("EOS_BERT_DENSE_ACCEL", "disabled")
	embedder := &PretrainedBERTTextEmbedder{
		program: &Program{executor: stubExecutor{kind: eosartifact.BackendCUDA}},
	}
	embedder.recordEncoderExecutionResult(backend.Result{
		Metadata: map[string]string{
			"cuda_matmul_bound_right_runs":   "2",
			"cuda_matmul_run_uploaded_bytes": "128",
		},
		Outputs: map[string]backend.Value{
			"embeddings": {
				Metadata: map[string]any{"execution_status": "host_reference_with_cuda_dense_matmul"},
			},
		},
	})
	embedder.recordEncoderExecutionResult(backend.Result{
		Metadata: map[string]string{
			"cuda_matmul_bound_right_runs":   "-7",
			"cuda_matmul_run_uploaded_bytes": "not-an-int",
		},
	})

	provenance := embedder.EncoderExecutionProvenance()
	if provenance.DenseAccelerationPolicy != "disabled" || provenance.SelectedBackend != "cuda" {
		t.Fatalf("backend/policy provenance = %+v, want cuda/disabled", provenance)
	}
	if !provenance.CUDADenseAccelerationObserved || !provenance.OpportunisticDeviceOpsUsed {
		t.Fatalf("observed counters did not mark opportunistic CUDA use: %+v", provenance)
	}
	if provenance.CUDADenseMatMulRuns != 2 || provenance.CUDADenseMatMulUploadedBytes != 128 {
		t.Fatalf("observed counters = runs:%d bytes:%d, want 2/128", provenance.CUDADenseMatMulRuns, provenance.CUDADenseMatMulUploadedBytes)
	}
	if provenance.HostReferenceExecutionStatus != "host_reference_with_cuda_dense_matmul" {
		t.Fatalf("host reference execution status = %q", provenance.HostReferenceExecutionStatus)
	}
	if provenance.FullDeviceExecution || provenance.ValidatedDeviceEncoder || provenance.DeviceEncoderContractSatisfied {
		t.Fatalf("opportunistic CUDA counters satisfied full device contract: %+v", provenance)
	}
}

func TestPretrainedBERTRetrievalVectorExportRequireDeviceEncoderFailsClosedForHostReference(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:          "tiny-bert",
		DatasetDir:           datasetDir,
		OutputDir:            outputDir,
		SourceDir:            sourceDir,
		ModulePath:           modulePath,
		WeightsPath:          weightsPath,
		BatchSize:            1,
		MaxLength:            4,
		Runtime:              New(cuda.New()),
		RequireDeviceEncoder: true,
	})
	if err == nil {
		t.Fatal("export succeeded with RequireDeviceEncoder=true, want fail-closed error")
	}
	for _, want := range []string{"--require-device-encoder requested", "pretrained_bert_host_reference", "full_device_execution=false", "opportunistic CUDA matmul"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v, want substring %q", err, want)
		}
	}
	if _, statErr := os.Stat(filepath.Join(outputDir, "doc-vectors.jsonl")); !os.IsNotExist(statErr) {
		t.Fatalf("doc vectors stat err = %v, want not exist", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(outputDir, "manifest.json")); !os.IsNotExist(statErr) {
		t.Fatalf("manifest stat err = %v, want not exist", statErr)
	}
}

func TestPretrainedBERTRetrievalVectorExportLoadsSourceFreePackage(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		PackagePath:    packagePath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export package pretrained BERT retrieval vectors: %v", err)
	}
	if summary.SourceDir != "" || summary.ModulePath != "" || summary.WeightsPath != "" {
		t.Fatalf("package summary should not require source/module/weights paths: %+v", summary)
	}
	if summary.PackagePath != packagePath || !isSHA256Hex(summary.PackageSHA256) || !isSHA256Hex(summary.PackageIdentitySHA256) {
		t.Fatalf("package identity fields = %+v", summary)
	}
	if summary.QualityClaim {
		t.Fatalf("package summary quality claim = true")
	}
	if !isSHA256Hex(summary.EmbeddingSpaceID) || !isSHA256Hex(summary.ModuleSHA256) || !isSHA256Hex(summary.WeightsSHA256) ||
		!isSHA256Hex(summary.ConfigSHA256) || !isSHA256Hex(summary.VocabSHA256) || !isSHA256Hex(summary.TokenizerConfigSHA256) {
		t.Fatalf("package summary hashes = %+v", summary)
	}
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.DocVectorPath)[0].Embedding, 2)
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.QueryVectorPath)[0].Embedding, 2)

	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.PackagePath != packagePath || manifest.PackageIdentitySHA256 != summary.PackageIdentitySHA256 || manifest.QualityClaim {
		t.Fatalf("manifest package fields = %+v", manifest)
	}
}

func TestPretrainedBERTRetrievalVectorExportUsesPackageRoleContract(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	packagePath = writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.RetrievalRoleContract = &PretrainedBERTRetrievalRoleContract{
			Schema:         PretrainedBERTRetrievalRoleContractSchema,
			QueryRole:      "query",
			DocumentRole:   "document",
			QueryPrefix:    "query: ",
			DocumentPrefix: "passage: ",
			Pooling:        pkg.Pooling,
			MaxLength:      pkg.MaxLength,
		}
		pkg.IdentitySHA256 = pkg.IdentityHash()
	})
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:            "tiny-bert",
		DatasetDir:             datasetDir,
		OutputDir:              outputDir,
		PackagePath:            packagePath,
		UsePackageRoleContract: true,
		BatchSize:              1,
		MaxLength:              4,
		Runtime:                New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export package pretrained BERT retrieval vectors: %v", err)
	}
	if summary.QueryPrefix != "query: " || summary.DocumentPrefix != "passage: " {
		t.Fatalf("summary prefixes = %q/%q", summary.QueryPrefix, summary.DocumentPrefix)
	}
	if !summary.QueryRoleApplied || !summary.DocumentRoleApplied {
		t.Fatalf("summary role flags = %+v", summary)
	}
}

func TestPretrainedBERTTextEmbedderPrefixForRole(t *testing.T) {
	embedder := &PretrainedBERTTextEmbedder{
		packagePath: "fixture.imported.mll",
		retrievalRoleContract: &PretrainedBERTRetrievalRoleContract{
			Schema:         PretrainedBERTRetrievalRoleContractSchema,
			QueryRole:      EmbeddingRoleQuery,
			DocumentRole:   EmbeddingRoleDocument,
			QueryPrefix:    "q: ",
			DocumentPrefix: "d: ",
		},
	}
	for _, tt := range []struct {
		role string
		want string
	}{
		{role: EmbeddingRoleRaw, want: ""},
		{role: "", want: ""},
		{role: EmbeddingRoleQuery, want: "q: "},
		{role: EmbeddingRoleDocument, want: "d: "},
		{role: " QUERY ", want: "q: "},
	} {
		got, err := embedder.PrefixForRole(tt.role)
		if err != nil {
			t.Fatalf("PrefixForRole(%q): %v", tt.role, err)
		}
		if got != tt.want {
			t.Fatalf("PrefixForRole(%q) = %q, want %q", tt.role, got, tt.want)
		}
	}
	if _, err := embedder.PrefixForRole("other"); err == nil || !strings.Contains(err.Error(), "raw, query, or document") {
		t.Fatalf("unsupported role err = %v", err)
	}
}

func TestPretrainedBERTTextEmbedderPrefixForRoleRejectsLegacyPackage(t *testing.T) {
	embedder := &PretrainedBERTTextEmbedder{packagePath: "legacy.imported.mll"}
	prefix, err := embedder.PrefixForRole(EmbeddingRoleRaw)
	if err != nil || prefix != "" {
		t.Fatalf("raw prefix = %q err=%v", prefix, err)
	}
	_, err = embedder.PrefixForRole(EmbeddingRoleQuery)
	if err == nil || !strings.Contains(err.Error(), "does not declare retrieval_role_contract") {
		t.Fatalf("query err = %v, want retrieval_role_contract error", err)
	}
}

func TestPretrainedBERTTextEmbedderEmbedTextBatchWithRoleUsesPackageContract(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	packagePath = writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.RetrievalRoleContract = &PretrainedBERTRetrievalRoleContract{
			Schema:         PretrainedBERTRetrievalRoleContractSchema,
			QueryRole:      EmbeddingRoleQuery,
			DocumentRole:   EmbeddingRoleDocument,
			QueryPrefix:    "q: ",
			DocumentPrefix: "",
			Pooling:        pkg.Pooling,
			MaxLength:      pkg.MaxLength,
		}
		pkg.IdentitySHA256 = pkg.IdentityHash()
	})
	embedder, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		PackagePath: packagePath,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("load package embedder: %v", err)
	}
	withRole, prefix, err := embedder.EmbedTextBatchWithRole(context.Background(), []string{"hello"}, EmbeddingRoleQuery)
	if err != nil {
		t.Fatalf("EmbedTextBatchWithRole: %v", err)
	}
	if prefix != "q: " {
		t.Fatalf("prefix = %q", prefix)
	}
	explicit, err := embedder.EmbedTextBatch(context.Background(), []string{"hello"}, "q: ")
	if err != nil {
		t.Fatalf("EmbedTextBatch explicit: %v", err)
	}
	if !slices.Equal(withRole[0], explicit[0]) {
		t.Fatalf("role embedding differs from explicit prefix embedding: role=%v explicit=%v", withRole[0], explicit[0])
	}
}

func TestPretrainedBERTRetrievalVectorExportUsePackageRoleContractRejectsConflictingPrefixes(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	packagePath = writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.RetrievalRoleContract = &PretrainedBERTRetrievalRoleContract{
			Schema:         PretrainedBERTRetrievalRoleContractSchema,
			QueryRole:      "query",
			DocumentRole:   "document",
			QueryPrefix:    "query: ",
			DocumentPrefix: "passage: ",
			Pooling:        pkg.Pooling,
			MaxLength:      pkg.MaxLength,
		}
		pkg.IdentitySHA256 = pkg.IdentityHash()
	})
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:            "tiny-bert",
		DatasetDir:             datasetDir,
		OutputDir:              outputDir,
		PackagePath:            packagePath,
		QueryPrefix:            "different: ",
		QueryPrefixSet:         true,
		UsePackageRoleContract: true,
		BatchSize:              1,
		MaxLength:              4,
		Runtime:                New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "conflicts with package retrieval role contract") {
		t.Fatalf("err = %v, want package contract conflict", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportUsePackageRoleContractRejectsLegacyPackage(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:            "tiny-bert",
		DatasetDir:             datasetDir,
		OutputDir:              outputDir,
		PackagePath:            packagePath,
		UsePackageRoleContract: true,
		BatchSize:              1,
		MaxLength:              4,
		Runtime:                New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "does not declare retrieval_role_contract") {
		t.Fatalf("err = %v, want missing package role contract", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportPackageResumeRequiresManifestIdentity(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		PackagePath: packagePath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("seed package export: %v", err)
	}
	beforeDocs := fileSize(t, seed.DocVectorPath)

	manifestPath := filepath.Join(outputDir, "manifest.json")
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("remove manifest: %v", err)
	}
	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		PackagePath: packagePath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "requires manifest") {
		t.Fatalf("err = %v, want missing manifest error", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected missing-manifest package resume: got %d want %d", got, beforeDocs)
	}

	legacy := PretrainedBERTRetrievalVectorExportSummary{
		Schema:    PretrainedBERTRetrievalVectorExportManifestSchema,
		OutputDim: seed.OutputDim,
	}
	data, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy manifest: %v", err)
	}
	if err := os.WriteFile(manifestPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write legacy manifest: %v", err)
	}
	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		PackagePath: packagePath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "missing embedding_space_id") {
		t.Fatalf("err = %v, want missing embedding_space_id error", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected legacy-manifest package resume: got %d want %d", got, beforeDocs)
	}
}

func TestPretrainedBERTPackageReadRejectsTamperedEmbeddedBytes(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)

	tests := []struct {
		name   string
		tamper func(*PretrainedBERTPackage)
	}{
		{
			name: "vocab",
			tamper: func(pkg *PretrainedBERTPackage) {
				pkg.Vocab = append([]byte(nil), pkg.Vocab...)
				pkg.Vocab = append(pkg.Vocab, []byte("tampered_token\n")...)
			},
		},
		{
			name: "config",
			tamper: func(pkg *PretrainedBERTPackage) {
				pkg.ConfigJSON = append([]byte(nil), pkg.ConfigJSON...)
				pkg.ConfigJSON = append(pkg.ConfigJSON, '\n')
			},
		},
		{
			name: "tokenizer_config",
			tamper: func(pkg *PretrainedBERTPackage) {
				pkg.TokenizerConfigJSON = []byte(`{"do_lower_case":false}` + "\n")
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tampered := writeTamperedPretrainedBERTPackage(t, packagePath, tt.tamper)
			_, err := ReadPretrainedBERTPackageFile(tampered)
			if err == nil || !strings.Contains(err.Error(), "file table") {
				t.Fatalf("read tampered package err = %v, want file table mismatch", err)
			}
		})
	}
}

func TestPretrainedBERTPackageReadRejectsTamperedParsedConfig(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)

	tampered := writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.Config.HiddenAct = "relu"
	})
	_, err := ReadPretrainedBERTPackageFile(tampered)
	if err == nil || !strings.Contains(err.Error(), "config does not match embedded config.json") {
		t.Fatalf("read tampered parsed config err = %v, want embedded config mismatch", err)
	}
}

func TestPretrainedBERTPackageReadRejectsTamperedParsedSTMetadata(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "cls", 3)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)

	tampered := writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		if pkg.STMetadata == nil {
			t.Fatalf("test package missing ST metadata")
		}
		pkg.STMetadata.Pooling = "masked_mean"
		pkg.STMetadata.MaxSeqLength = 4
	})
	_, err := ReadPretrainedBERTPackageFile(tampered)
	if err == nil || !strings.Contains(err.Error(), "sentence-transformers metadata does not match embedded metadata files") {
		t.Fatalf("read tampered parsed ST metadata err = %v, want embedded ST metadata mismatch", err)
	}
}

func TestPretrainedBERTPackageReadRejectsHiddenSTMetadataWithoutEmbeddedFiles(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)

	tampered := writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.STMetadata = &PretrainedBERTSTMetadata{
			Pooling:         "cls",
			PoolingSource:   filepath.Join("1_Pooling", "config.json"),
			MaxSeqLength:    3,
			MaxLengthSource: "sentence_bert_config.json",
		}
	})
	_, err := ReadPretrainedBERTPackageFile(tampered)
	if err == nil || !strings.Contains(err.Error(), "sentence-transformers metadata does not match embedded metadata files") {
		t.Fatalf("read hidden ST metadata err = %v, want embedded ST metadata mismatch", err)
	}
}

func TestPretrainedBERTPackageReadRejectsTamperedResolvedMaxLength(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "cls", 3)
	packagePath := writeTinyPretrainedBERTPackageFromFixture(t, sourceDir, modulePath, weightsPath)

	tampered := writeTamperedPretrainedBERTPackage(t, packagePath, func(pkg *PretrainedBERTPackage) {
		pkg.MaxLength = 4
		pkg.IdentitySHA256 = pkg.IdentityHash()
	})
	_, err := ReadPretrainedBERTPackageFile(tampered)
	if err == nil || !strings.Contains(err.Error(), "max_length 4 does not match embedded config/sentence-transformers max length 3") {
		t.Fatalf("read tampered max length err = %v, want resolved max length mismatch", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportWritesCompactOutputDim(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		OutputDim:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors: %v", err)
	}
	if summary.NativeDim != 2 || summary.OutputDim != 1 {
		t.Fatalf("summary dims = native:%d output:%d, want 2/1", summary.NativeDim, summary.OutputDim)
	}
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.DocVectorPath)[0].Embedding, 1)
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.QueryVectorPath)[0].Embedding, 1)

	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.NativeDim != 2 || manifest.OutputDim != 1 {
		t.Fatalf("manifest dims = native:%d output:%d, want 2/1", manifest.NativeDim, manifest.OutputDim)
	}
}

func TestPretrainedBERTRetrievalVectorExportAppliesProjectionHeadAndRecordsIdentity(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	headPath := writeTinyPretrainedBERTProjectionHead(t, 2, 1, []float32{1, 0})

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headPath,
		MaxLength:          4,
		Runtime:            New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors with projection head: %v", err)
	}
	if summary.NativeDim != 2 || summary.OutputDim != 1 || summary.ProjectionHeadPath != headPath {
		t.Fatalf("summary dims/head = %+v", summary)
	}
	if !isSHA256Hex(summary.ProjectionHeadSHA256) || !isSHA256Hex(summary.ProjectionHeadIdentity) || summary.ProjectionHeadSchema != PretrainedBERTProjectionHeadSchema {
		t.Fatalf("summary projection identity = %+v", summary)
	}
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.DocVectorPath)[0].Embedding, 1)
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.QueryVectorPath)[0].Embedding, 1)

	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if manifest.ProjectionHeadPath != headPath || manifest.ProjectionHeadIdentity != summary.ProjectionHeadIdentity || manifest.OutputDim != 1 {
		t.Fatalf("manifest projection fields = %+v", manifest)
	}

	noHeadDir := filepath.Join(t.TempDir(), "no-head")
	noHead, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   noHeadDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		OutputDim:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors without projection head: %v", err)
	}
	if noHead.EmbeddingSpaceID == summary.EmbeddingSpaceID {
		t.Fatalf("projection head embedding space matched prefix truncation id %q", summary.EmbeddingSpaceID)
	}
}

func TestPretrainedBERTRetrievalVectorExportProjectionHeadIdentityIgnoresPath(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	headA := writeTinyPretrainedBERTProjectionHead(t, 2, 1, []float32{1, 0})
	headB := filepath.Join(t.TempDir(), "repackaged", "head-copy.mll")
	if err := os.MkdirAll(filepath.Dir(headB), 0o755); err != nil {
		t.Fatalf("mkdir repackaged head dir: %v", err)
	}
	repackaged, err := NewPretrainedBERTProjectionHead(2, 1, []float32{1, 0})
	if err != nil {
		t.Fatalf("new repackaged head: %v", err)
	}
	repackaged.SourceModel = "same-functional-head-at-new-location"
	repackaged.Loss = "alternate-audit-loss-label"
	if err := WritePretrainedBERTProjectionHeadFile(headB, repackaged); err != nil {
		t.Fatalf("write head B: %v", err)
	}

	export := func(outputDir, headPath string) PretrainedBERTRetrievalVectorExportSummary {
		t.Helper()
		summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
			DatasetName:        "tiny-bert",
			DatasetDir:         datasetDir,
			OutputDir:          outputDir,
			SourceDir:          sourceDir,
			ModulePath:         modulePath,
			WeightsPath:        weightsPath,
			BatchSize:          1,
			ProjectionHeadPath: headPath,
			MaxLength:          4,
			Runtime:            New(cuda.New()),
		})
		if err != nil {
			t.Fatalf("export with projection head %s: %v", headPath, err)
		}
		return summary
	}

	first := export(filepath.Join(t.TempDir(), "vectors-a"), headA)
	second := export(filepath.Join(t.TempDir(), "vectors-b"), headB)
	if first.ProjectionHeadPath == second.ProjectionHeadPath {
		t.Fatalf("test did not use different head paths")
	}
	if first.ProjectionHeadSHA256 == second.ProjectionHeadSHA256 {
		t.Fatalf("test did not produce different projection head file hashes: %q", first.ProjectionHeadSHA256)
	}
	if first.ProjectionHeadIdentity == "" || first.ProjectionHeadIdentity != second.ProjectionHeadIdentity {
		t.Fatalf("projection identities = %q and %q, want match", first.ProjectionHeadIdentity, second.ProjectionHeadIdentity)
	}
	if first.EmbeddingSpaceID == "" || first.EmbeddingSpaceID != second.EmbeddingSpaceID {
		t.Fatalf("embedding space ids = %q and %q, want match for same functional head", first.EmbeddingSpaceID, second.EmbeddingSpaceID)
	}
}

func TestPretrainedBERTRetrievalVectorExportRejectsProjectionHeadShapeMismatch(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	headPath := writeTinyPretrainedBERTProjectionHead(t, 3, 1, []float32{1, 0, 0})

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          filepath.Join(t.TempDir(), "vectors"),
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headPath,
		MaxLength:          4,
		Runtime:            New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "projection head input_dim 3 does not match") {
		t.Fatalf("err = %v, want projection input_dim mismatch", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportRejectsTooLargeOutputDim(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   filepath.Join(t.TempDir(), "vectors"),
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		OutputDim:   3,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err == nil || !strings.Contains(err.Error(), "output-dim 3 exceeds pretrained BERT native dimension 2") {
		t.Fatalf("err = %v, want output-dim too large error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportUsesSentenceTransformersMetadataDefaults(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "cls", 3)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	outputDir := filepath.Join(t.TempDir(), "vectors")

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors: %v", err)
	}
	if summary.MaxLength != 3 || !strings.Contains(summary.MaxLengthSource, "sentence_bert_config.json") {
		t.Fatalf("summary max length = %d source=%q", summary.MaxLength, summary.MaxLengthSource)
	}
	if summary.Pooling != "cls" {
		t.Fatalf("summary pooling = %q, want cls", summary.Pooling)
	}
	assertFiniteUnitishVector(t, readTinyVectorRows(t, summary.DocVectorPath)[0].Embedding, 2)

	overrideDir := filepath.Join(t.TempDir(), "override")
	override, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   overrideDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export pretrained BERT retrieval vectors with override: %v", err)
	}
	if override.MaxLength != 4 || override.MaxLengthSource != "explicit" {
		t.Fatalf("override max length = %d source=%q", override.MaxLength, override.MaxLengthSource)
	}
	if override.Pooling != "cls" {
		t.Fatalf("override pooling = %q, want cls", override.Pooling)
	}
}

func TestPretrainedBERTRetrievalVectorExportEmbeddingSpaceIDIsStable(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "cls", 3)
	datasetDir := writeTinyPretrainedBERTBEIRFixture(t)
	rt := New(cuda.New())

	export := func(outputDir string, outputDim int) PretrainedBERTRetrievalVectorExportSummary {
		t.Helper()
		summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
			DatasetName:    "tiny-bert",
			DatasetDir:     datasetDir,
			OutputDir:      outputDir,
			SourceDir:      sourceDir,
			ModulePath:     modulePath,
			WeightsPath:    weightsPath,
			QueryPrefix:    "query ",
			DocumentPrefix: "doc ",
			BatchSize:      1,
			OutputDim:      outputDim,
			Runtime:        rt,
		})
		if err != nil {
			t.Fatalf("export pretrained BERT retrieval vectors: %v", err)
		}
		return summary
	}

	first := export(filepath.Join(t.TempDir(), "vectors-a"), 0)
	second := export(filepath.Join(t.TempDir(), "vectors-b"), 0)
	if first.EmbeddingSpaceID == "" || first.EmbeddingSpaceID != second.EmbeddingSpaceID {
		t.Fatalf("embedding space ids = %q and %q, want stable non-empty match", first.EmbeddingSpaceID, second.EmbeddingSpaceID)
	}
	compact := export(filepath.Join(t.TempDir(), "vectors-compact"), 1)
	if compact.EmbeddingSpaceID == first.EmbeddingSpaceID {
		t.Fatalf("compact embedding space id = %q, want different from full-dim id", compact.EmbeddingSpaceID)
	}
	if !isSHA256Hex(first.SentenceTransformersPoolingSHA256) || !isSHA256Hex(first.SentenceTransformersConfigSHA256) {
		t.Fatalf("sentence-transformers hashes = pooling:%q config:%q", first.SentenceTransformersPoolingSHA256, first.SentenceTransformersConfigSHA256)
	}
}

func TestPretrainedBERTTextEmbedderRejectsTooLargeMaxLength(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	_, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		MaxLength:   5,
		Runtime:     New(cuda.New()),
	})
	if err == nil {
		t.Fatalf("expected max length error")
	}
}

func TestPretrainedBERTTextEmbedderDynamicBatchLengthMatchesFixedCLS(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "cls", 4)
	embedder, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("load embedder: %v", err)
	}
	texts := []string{"alpha", "alpha"}
	dynamic, err := embedder.EmbedTextBatch(context.Background(), texts, "")
	if err != nil {
		t.Fatalf("dynamic embed: %v", err)
	}
	fixed, err := embedTextBatchFixedPretrainedBERTMaxLength(context.Background(), embedder, texts, "")
	if err != nil {
		t.Fatalf("fixed embed: %v", err)
	}
	assertEmbeddingBatchesClose(t, dynamic, fixed, 1e-6)
}

func TestPretrainedBERTTextEmbedderDynamicBatchLengthMatchesFixedMaskedMean(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(t, "masked_mean", 4)
	embedder, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("load embedder: %v", err)
	}
	texts := []string{"alpha", "alpha"}
	dynamic, err := embedder.EmbedTextBatch(context.Background(), texts, "")
	if err != nil {
		t.Fatalf("dynamic embed: %v", err)
	}
	fixed, err := embedTextBatchFixedPretrainedBERTMaxLength(context.Background(), embedder, texts, "")
	if err != nil {
		t.Fatalf("fixed embed: %v", err)
	}
	assertEmbeddingBatchesClose(t, dynamic, fixed, 1e-6)
}

func TestPretrainedBERTRetrievalVectorExportResumeAppendsPartialCaches(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 3)
	rt := New(cuda.New())
	seedOutput := filepath.Join(t.TempDir(), "seed")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      seedOutput,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
	})
	if err != nil {
		t.Fatalf("seed export: %v", err)
	}
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	writeVectorRows(t, filepath.Join(outputDir, "doc-vectors.jsonl"), readTinyVectorRows(t, seed.DocVectorPath)[:1])
	writeVectorRows(t, filepath.Join(outputDir, "query-vectors.jsonl"), readTinyVectorRows(t, seed.QueryVectorPath)[:1])

	var progress []PretrainedBERTRetrievalVectorExportProgress
	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
		Resume:         true,
		ProgressEvery:  1,
		Progress: func(p PretrainedBERTRetrievalVectorExportProgress) {
			progress = append(progress, p)
		},
	})
	if err != nil {
		t.Fatalf("resume export: %v", err)
	}
	if !summary.Resume || summary.ReusedDocuments != 1 || summary.ReusedQueries != 1 || summary.WrittenDocuments != 2 || summary.WrittenQueries != 2 {
		t.Fatalf("summary resume counters = %+v", summary)
	}
	if len(progress) == 0 {
		t.Fatalf("expected progress callbacks")
	}
	if got := rowIDs(readTinyVectorRows(t, summary.DocVectorPath)); !slices.Equal(got, []string{"d1", "d2", "d3"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.QueryVectorPath)); !slices.Equal(got, []string{"q1", "q2", "q3"}) {
		t.Fatalf("query ids = %v", got)
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	data, err := os.ReadFile(filepath.Join(outputDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	if !manifest.Resume || manifest.ReusedDocuments != 1 || manifest.ReusedQueries != 1 || manifest.WrittenDocuments != 2 || manifest.WrittenQueries != 2 {
		t.Fatalf("manifest resume counters = %+v", manifest)
	}
}

func TestPretrainedBERTRetrievalVectorExportProjectionHeadResumeRequiresManifestIdentity(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	rt := New(cuda.New())
	headPath := writeTinyPretrainedBERTProjectionHead(t, 2, 1, []float32{1, 0})
	outputDir := filepath.Join(t.TempDir(), "vectors")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headPath,
		MaxLength:          4,
		Runtime:            rt,
	})
	if err != nil {
		t.Fatalf("seed export: %v", err)
	}
	beforeDocs := fileSize(t, seed.DocVectorPath)

	manifestPath := filepath.Join(outputDir, "manifest.json")
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("remove manifest: %v", err)
	}
	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headPath,
		MaxLength:          4,
		Runtime:            rt,
		Resume:             true,
	})
	if err == nil || !strings.Contains(err.Error(), "requires manifest") {
		t.Fatalf("err = %v, want missing manifest error", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected missing-manifest resume: got %d want %d", got, beforeDocs)
	}

	legacy := PretrainedBERTRetrievalVectorExportSummary{
		Schema:    PretrainedBERTRetrievalVectorExportManifestSchema,
		OutputDim: 1,
	}
	data, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy manifest: %v", err)
	}
	if err := os.WriteFile(manifestPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write legacy manifest: %v", err)
	}
	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headPath,
		MaxLength:          4,
		Runtime:            rt,
		Resume:             true,
	})
	if err == nil || !strings.Contains(err.Error(), "missing embedding_space_id") {
		t.Fatalf("err = %v, want missing embedding_space_id error", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected legacy-manifest resume: got %d want %d", got, beforeDocs)
	}
}

func TestPretrainedBERTRetrievalVectorExportProjectionHeadResumeRejectsDifferentIdentitySameDim(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	rt := New(cuda.New())
	headA := writeTinyPretrainedBERTProjectionHead(t, 2, 1, []float32{1, 0})
	headB := writeTinyPretrainedBERTProjectionHead(t, 2, 1, []float32{0, 1})
	outputDir := filepath.Join(t.TempDir(), "vectors")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headA,
		MaxLength:          4,
		Runtime:            rt,
	})
	if err != nil {
		t.Fatalf("seed export: %v", err)
	}
	beforeDocs := fileSize(t, seed.DocVectorPath)

	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:        "tiny-bert",
		DatasetDir:         datasetDir,
		OutputDir:          outputDir,
		SourceDir:          sourceDir,
		ModulePath:         modulePath,
		WeightsPath:        weightsPath,
		BatchSize:          1,
		ProjectionHeadPath: headB,
		MaxLength:          4,
		Runtime:            rt,
		Resume:             true,
	})
	if err == nil || !strings.Contains(err.Error(), "embedding_space_id") {
		t.Fatalf("err = %v, want embedding_space_id mismatch", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected different-head resume: got %d want %d", got, beforeDocs)
	}
}

func BenchmarkPretrainedBERTTextEmbedderShortBatchDynamicLength(b *testing.B) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(b, "cls", 4)
	benchmarkPretrainedBERTTextEmbedderShortBatch(b, sourceDir, modulePath, weightsPath, false)
}

func BenchmarkPretrainedBERTTextEmbedderShortBatchFixedMaxLength(b *testing.B) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixtureWithST(b, "cls", 4)
	benchmarkPretrainedBERTTextEmbedderShortBatch(b, sourceDir, modulePath, weightsPath, true)
}

func benchmarkPretrainedBERTTextEmbedderShortBatch(b *testing.B, sourceDir, modulePath, weightsPath string, fixed bool) {
	embedder, err := LoadPretrainedBERTTextEmbedder(context.Background(), PretrainedBERTTextEmbedderConfig{
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		b.Fatalf("load embedder: %v", err)
	}
	texts := []string{"alpha", "alpha", "alpha", "alpha", "alpha", "alpha", "alpha", "alpha"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if fixed {
			_, err = embedTextBatchFixedPretrainedBERTMaxLength(context.Background(), embedder, texts, "")
		} else {
			_, err = embedder.EmbedTextBatch(context.Background(), texts, "")
		}
		if err != nil {
			b.Fatalf("embed: %v", err)
		}
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsEmbeddingSpaceMismatchBeforeAppend(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	rt := New(cuda.New())
	outputDir := filepath.Join(t.TempDir(), "vectors")
	seed, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
	})
	if err != nil {
		t.Fatalf("seed export: %v", err)
	}
	beforeDocs := fileSize(t, seed.DocVectorPath)
	beforeQueries := fileSize(t, seed.QueryVectorPath)

	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "changed query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		MaxLength:      4,
		Runtime:        rt,
		Resume:         true,
	})
	if err == nil || !strings.Contains(err.Error(), "embedding_space_id") {
		t.Fatalf("err = %v, want embedding_space_id mismatch", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected resume: got %d want %d", got, beforeDocs)
	}
	if got := fileSize(t, seed.QueryVectorPath); got != beforeQueries {
		t.Fatalf("query vector size changed after rejected resume: got %d want %d", got, beforeQueries)
	}

	_, err = ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:    "tiny-bert",
		DatasetDir:     datasetDir,
		OutputDir:      outputDir,
		SourceDir:      sourceDir,
		ModulePath:     modulePath,
		WeightsPath:    weightsPath,
		QueryPrefix:    "query ",
		DocumentPrefix: "doc ",
		BatchSize:      1,
		OutputDim:      1,
		MaxLength:      4,
		Runtime:        rt,
		Resume:         true,
	})
	if err == nil || !strings.Contains(err.Error(), "embedding_space_id") {
		t.Fatalf("err = %v, want embedding_space_id mismatch for output-dim change", err)
	}
	if got := fileSize(t, seed.DocVectorPath); got != beforeDocs {
		t.Fatalf("doc vector size changed after rejected output-dim resume: got %d want %d", got, beforeDocs)
	}
	if got := fileSize(t, seed.QueryVectorPath); got != beforeQueries {
		t.Fatalf("query vector size changed after rejected output-dim resume: got %d want %d", got, beforeQueries)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsMismatchedID(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(`{"id":"not-d1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "want prefix id") {
		t.Fatalf("err = %v, want mismatched id error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsDimensionMismatch(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	stale := `{"id":"d1","embedding":[1,0]}` + "\n" + `{"id":"d2","embedding":[1,0,0]}` + "\n"
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(stale), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "has dimension 3, want 2") {
		t.Fatalf("err = %v, want dimension mismatch error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportResumeRejectsCompleteCacheWithWrongModelDimension(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	wrongDimDocs := []retrievalVectorExportRow{
		{ID: "d1", Embedding: []float32{1, 0, 0}},
		{ID: "d2", Embedding: []float32{0, 1, 0}},
	}
	wrongDimQueries := []retrievalVectorExportRow{
		{ID: "q1", Embedding: []float32{1, 0, 0}},
		{ID: "q2", Embedding: []float32{0, 1, 0}},
	}
	writeVectorRows(t, filepath.Join(outputDir, "doc-vectors.jsonl"), wrongDimDocs)
	writeVectorRows(t, filepath.Join(outputDir, "query-vectors.jsonl"), wrongDimQueries)

	_, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
		Resume:      true,
	})
	if err == nil || !strings.Contains(err.Error(), "want current model output dimension 2") {
		t.Fatalf("err = %v, want current model dimension error", err)
	}
}

func TestPretrainedBERTRetrievalVectorExportWithoutResumeOverwritesStaleFiles(t *testing.T) {
	sourceDir, modulePath, weightsPath := writeTinyPretrainedBERTExportFixture(t)
	datasetDir := writeTinyPretrainedBERTBEIRFixtureN(t, 2)
	outputDir := filepath.Join(t.TempDir(), "vectors")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "doc-vectors.jsonl"), []byte(`{"id":"stale","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale doc vectors: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "query-vectors.jsonl"), []byte(`{"id":"stale","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write stale query vectors: %v", err)
	}

	summary, err := ExportPretrainedBERTRetrievalVectors(context.Background(), PretrainedBERTRetrievalVectorExportConfig{
		DatasetName: "tiny-bert",
		DatasetDir:  datasetDir,
		OutputDir:   outputDir,
		SourceDir:   sourceDir,
		ModulePath:  modulePath,
		WeightsPath: weightsPath,
		BatchSize:   1,
		MaxLength:   4,
		Runtime:     New(cuda.New()),
	})
	if err != nil {
		t.Fatalf("export without resume: %v", err)
	}
	if summary.Resume || summary.ReusedDocuments != 0 || summary.ReusedQueries != 0 || summary.WrittenDocuments != 2 || summary.WrittenQueries != 2 {
		t.Fatalf("summary counters = %+v", summary)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.DocVectorPath)); !slices.Equal(got, []string{"d1", "d2"}) {
		t.Fatalf("doc ids = %v", got)
	}
	if got := rowIDs(readTinyVectorRows(t, summary.QueryVectorPath)); !slices.Equal(got, []string{"q1", "q2"}) {
		t.Fatalf("query ids = %v", got)
	}
}

func writeTinyPretrainedBERTExportFixture(t testing.TB) (sourceDir, modulePath, weightsPath string) {
	t.Helper()
	dir := t.TempDir()
	sourceDir = filepath.Join(dir, "source")
	if err := os.MkdirAll(sourceDir, 0o755); err != nil {
		t.Fatalf("mkdir source: %v", err)
	}
	cfg := PretrainedBERTConfig{
		ModelType:             "bert",
		VocabSize:             8,
		HiddenSize:            2,
		NumHiddenLayers:       2,
		NumAttentionHeads:     1,
		IntermediateSize:      3,
		HiddenAct:             "gelu",
		MaxPositionEmbeddings: 4,
		TypeVocabSize:         2,
		LayerNormEps:          0.25,
	}
	configData, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "config.json"), append(configData, '\n'), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "vocab.txt"), []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nalpha\nquery\ndoc\n"), 0o644); err != nil {
		t.Fatalf("write vocab: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sourceDir, "tokenizer_config.json"), []byte(`{"do_lower_case":true}`+"\n"), 0o644); err != nil {
		t.Fatalf("write tokenizer config: %v", err)
	}

	plan, err := PlanPretrainedBERTImportFromDir(sourceDir, "fixture")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	mod, err := BuildPretrainedBERTEmbedderModule(plan)
	if err != nil {
		t.Fatalf("build module: %v", err)
	}
	modulePath = filepath.Join(dir, "bert_embed.mll")
	if err := eosartifact.WriteFile(modulePath, mod); err != nil {
		t.Fatalf("write module: %v", err)
	}
	weights, _, err := BuildPretrainedBERTWeightFileFromDecoded(PretrainedBERTDecodedWeightSet{Tensors: tinyPretrainedBERTExportDecodedWeights()})
	if err != nil {
		t.Fatalf("build weights: %v", err)
	}
	weightsPath = filepath.Join(dir, "bert_weights.mll")
	if err := weights.WriteFile(weightsPath); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	return sourceDir, modulePath, weightsPath
}

func writeTinyPretrainedBERTProjectionHead(t testing.TB, inputDim, outputDim int, weights []float32) string {
	t.Helper()
	head, err := NewPretrainedBERTProjectionHead(inputDim, outputDim, weights)
	if err != nil {
		t.Fatalf("new projection head: %v", err)
	}
	head.SourceModel = "tiny-bert"
	head.Initialization = "unit-test"
	head.Loss = "listwise_kl_softmax_dot"
	head.DataProvenance = "unit-test train split only"
	path := filepath.Join(t.TempDir(), "projection-head.mll")
	if err := WritePretrainedBERTProjectionHeadFile(path, head); err != nil {
		t.Fatalf("write projection head: %v", err)
	}
	return path
}

func writeTinyPretrainedBERTExportFixtureWithST(t testing.TB, pooling string, maxSeqLength int) (sourceDir, modulePath, weightsPath string) {
	t.Helper()
	sourceDir, modulePath, weightsPath = writeTinyPretrainedBERTExportFixture(t)
	if err := os.MkdirAll(filepath.Join(sourceDir, "1_Pooling"), 0o755); err != nil {
		t.Fatalf("mkdir pooling: %v", err)
	}
	cls := pooling == "cls"
	mean := pooling == "masked_mean"
	poolingConfig := fmt.Sprintf(`{
		"pooling_mode_cls_token": %t,
		"pooling_mode_mean_tokens": %t,
		"pooling_mode_max_tokens": false,
		"pooling_mode_mean_sqrt_len_tokens": false,
		"pooling_mode_weightedmean_tokens": false,
		"pooling_mode_lasttoken": false
	}`+"\n", cls, mean)
	if err := os.WriteFile(filepath.Join(sourceDir, "1_Pooling", "config.json"), []byte(poolingConfig), 0o644); err != nil {
		t.Fatalf("write pooling config: %v", err)
	}
	if maxSeqLength > 0 {
		if err := os.WriteFile(filepath.Join(sourceDir, "sentence_bert_config.json"), []byte(fmt.Sprintf(`{"max_seq_length":%d}`+"\n", maxSeqLength)), 0o644); err != nil {
			t.Fatalf("write sentence config: %v", err)
		}
	}
	plan, err := PlanPretrainedBERTImportFromDir(sourceDir, "fixture")
	if err != nil {
		t.Fatalf("plan with st metadata: %v", err)
	}
	mod, err := BuildPretrainedBERTEmbedderModule(plan)
	if err != nil {
		t.Fatalf("build st module: %v", err)
	}
	if err := eosartifact.WriteFile(modulePath, mod); err != nil {
		t.Fatalf("rewrite st module: %v", err)
	}
	return sourceDir, modulePath, weightsPath
}

func writeTinyPretrainedBERTPackageFromFixture(t testing.TB, sourceDir, modulePath, weightsPath string) string {
	t.Helper()
	cfg, err := LoadPretrainedBERTConfig(filepath.Join(sourceDir, "config.json"))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	plan, err := PlanPretrainedBERTImportFromDir(sourceDir, "fixture")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	moduleBytes, err := os.ReadFile(modulePath)
	if err != nil {
		t.Fatalf("read module: %v", err)
	}
	weightsBytes, err := os.ReadFile(weightsPath)
	if err != nil {
		t.Fatalf("read weights: %v", err)
	}
	configJSON, err := os.ReadFile(filepath.Join(sourceDir, "config.json"))
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	vocab, err := os.ReadFile(filepath.Join(sourceDir, "vocab.txt"))
	if err != nil {
		t.Fatalf("read vocab: %v", err)
	}
	tokenizerConfig, err := os.ReadFile(filepath.Join(sourceDir, "tokenizer_config.json"))
	if err != nil {
		t.Fatalf("read tokenizer config: %v", err)
	}
	stPooling, err := readOptionalPretrainedBERTPackageFile(sourceDir, filepath.Join("1_Pooling", "config.json"))
	if err != nil {
		t.Fatalf("read ST pooling config: %v", err)
	}
	stConfig, err := readOptionalPretrainedBERTPackageFile(sourceDir, "sentence_bert_config.json")
	if err != nil {
		t.Fatalf("read ST config: %v", err)
	}
	stMetadata, err := parsePretrainedBERTPackageSTMetadata(stPooling, stConfig)
	if err != nil {
		t.Fatalf("parse ST metadata: %v", err)
	}
	maxLength, _ := resolvePretrainedBERTMaxLength(0, cfg, stMetadata)
	pkg := PretrainedBERTPackage{
		Version:                PretrainedBERTPackageVersion,
		ModelName:              plan.ModelName,
		Architecture:           plan.Architecture,
		Config:                 cfg,
		STMetadata:             stMetadata,
		Pooling:                plan.PoolingPolicy,
		Normalization:          "l2",
		MaxLength:              maxLength,
		NativeDim:              cfg.HiddenSize,
		ModuleSHA256:           sha256BytesHex(moduleBytes),
		WeightsSHA256:          sha256BytesHex(weightsBytes),
		ModuleBytes:            moduleBytes,
		WeightsBytes:           weightsBytes,
		ConfigJSON:             configJSON,
		Vocab:                  vocab,
		TokenizerConfigJSON:    tokenizerConfig,
		STPoolingConfigJSON:    stPooling,
		SentenceBERTConfigJSON: stConfig,
	}
	pkg.Files = pretrainedBERTPackageFiles(pkg)
	pkg.IdentitySHA256 = pkg.IdentityHash()
	data, err := encodePretrainedBERTPackageMLL(pkg)
	if err != nil {
		t.Fatalf("encode package: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny-pretrained-bert.imported.mll")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write package: %v", err)
	}
	return path
}

func writeTamperedPretrainedBERTPackage(t testing.TB, sourcePath string, tamper func(*PretrainedBERTPackage)) string {
	t.Helper()
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read source package: %v", err)
	}
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		t.Fatalf("parse source package: %v", err)
	}
	body, ok := reader.Section(tagXPBT)
	if !ok {
		t.Fatalf("source package missing XPBT section")
	}
	var pkg PretrainedBERTPackage
	if err := json.Unmarshal(body, &pkg); err != nil {
		t.Fatalf("parse source package payload: %v", err)
	}
	tamper(&pkg)
	tamperedBody, err := json.Marshal(pkg)
	if err != nil {
		t.Fatalf("marshal tampered package payload: %v", err)
	}
	sections := make([]mll.SectionInput, 0, reader.SectionCount())
	for _, entry := range reader.DirectoryEntries() {
		sectionBody, ok := reader.Section(entry.Tag)
		if !ok {
			t.Fatalf("source package missing section %q", string(entry.Tag[:]))
		}
		input := mll.SectionInput{
			Tag:           entry.Tag,
			Body:          append([]byte(nil), sectionBody...),
			Flags:         entry.Flags,
			SchemaVersion: entry.SchemaVersion,
		}
		if entry.Tag == mll.TagHEAD {
			head, err := mll.ReadHeadSection(input.Body)
			if err != nil {
				t.Fatalf("parse HEAD section: %v", err)
			}
			input.DigestBody = head.DigestBody(reader.Profile())
		}
		if entry.Tag == tagXPBT {
			input.Body = tamperedBody
		}
		sections = append(sections, input)
	}
	tamperedData, err := mll.WriteToBytes(reader.Profile(), reader.Version(), sections)
	if err != nil {
		t.Fatalf("encode tampered package: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tampered-"+filepath.Base(sourcePath))
	if err := os.WriteFile(path, tamperedData, 0o644); err != nil {
		t.Fatalf("write tampered package: %v", err)
	}
	return path
}

func tinyPretrainedBERTExportDecodedWeights() []PretrainedBERTDecodedWeightTensor {
	decoded := []PretrainedBERTDecodedWeightTensor{
		{Name: "embeddings.word_embeddings.weight", Role: "token_embeddings", SourceDType: "F32", Shape: []int64{8, 2}, Values: []float32{
			0, 0,
			0.1, -0.1,
			1, 0,
			0, 1,
			-1, 0,
			1, 2,
			2, 1,
			-0.5, 1.5,
		}},
		{Name: "embeddings.position_embeddings.weight", Role: "position_embeddings", SourceDType: "F32", Shape: []int64{4, 2}, Values: []float32{0, 1, 1, 0, -1, 0.5, 0.25, -0.25}},
		{Name: "embeddings.token_type_embeddings.weight", Role: "token_type_embeddings", SourceDType: "F32", Shape: []int64{2, 2}, Values: []float32{0, 0, 2, -2}},
		{Name: "embeddings.LayerNorm.weight", Role: "embedding_layernorm_weight", SourceDType: "F32", Shape: []int64{2}, Values: []float32{2, 3}},
		{Name: "embeddings.LayerNorm.bias", Role: "embedding_layernorm_bias", SourceDType: "F32", Shape: []int64{2}, Values: []float32{0.5, -0.5}},
	}
	layer0 := pretrainedBERTSingleLayerDecodedWeights()
	decoded = append(decoded, layer0...)
	for _, tensor := range layer0 {
		tensor.Name = strings.Replace(tensor.Name, ".0.", ".1.", 1)
		tensor.Role = strings.Replace(tensor.Role, "_0_", "_1_", 1)
		decoded = append(decoded, tensor)
	}
	return decoded
}

func writeTinyPretrainedBERTBEIRFixture(t *testing.T) string {
	return writeTinyPretrainedBERTBEIRFixtureN(t, 1)
}

func writeTinyPretrainedBERTBEIRFixtureN(t *testing.T, n int) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	var corpus, queries, qrels strings.Builder
	qrels.WriteString("query-id\tcorpus-id\tscore\n")
	for i := 1; i <= n; i++ {
		fmt.Fprintf(&corpus, `{"_id":"d%d","title":"", "text":"alpha"}`+"\n", i)
		fmt.Fprintf(&queries, `{"_id":"q%d","text":"alpha"}`+"\n", i)
		fmt.Fprintf(&qrels, "q%d\td%d\t1\n", i, i)
	}
	if err := os.WriteFile(filepath.Join(dir, "corpus.jsonl"), []byte(corpus.String()), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "queries.jsonl"), []byte(queries.String()), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "qrels", "test.tsv"), []byte(qrels.String()), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	return dir
}

func writeVectorRows(t *testing.T, path string, rows []retrievalVectorExportRow) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			t.Fatalf("write row: %v", err)
		}
	}
	if err := writer.Flush(); err != nil {
		t.Fatalf("flush rows: %v", err)
	}
}

func readTinyVectorRows(t *testing.T, path string) []retrievalVectorExportRow {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer file.Close()
	var rows []retrievalVectorExportRow
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var row retrievalVectorExportRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			t.Fatalf("parse row: %v", err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan rows: %v", err)
	}
	return rows
}

func embedTextBatchFixedPretrainedBERTMaxLength(ctx context.Context, embedder *PretrainedBERTTextEmbedder, texts []string, prefix string) ([][]float32, error) {
	inputIDs := make([]int32, 0, len(texts)*embedder.maxLength)
	attentionMask := make([]int32, 0, len(texts)*embedder.maxLength)
	tokenTypeIDs := make([]int32, 0, len(texts)*embedder.maxLength)
	for _, text := range texts {
		encoded, err := embedder.tokenizer.Encode(prefix+text, HFWordPieceEncodeOptions{
			MaxLength:      embedder.maxLength,
			PadToMaxLength: true,
		})
		if err != nil {
			return nil, err
		}
		inputIDs = append(inputIDs, encoded.IDs...)
		attentionMask = append(attentionMask, encoded.AttentionMask...)
		tokenTypeIDs = append(tokenTypeIDs, encoded.TokenTypeIDs...)
	}
	result, err := embedder.program.Run(ctx, backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      backend.NewTensorI32([]int{len(texts), embedder.maxLength}, inputIDs),
			"attention_mask": backend.NewTensorI32([]int{len(texts), embedder.maxLength}, attentionMask),
			"token_type_ids": backend.NewTensorI32([]int{len(texts), embedder.maxLength}, tokenTypeIDs),
		},
	})
	if err != nil {
		return nil, err
	}
	value, ok := result.Outputs["embeddings"]
	if !ok {
		return nil, fmt.Errorf("bert_embed output missing embeddings")
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok {
		return nil, fmt.Errorf("bert_embed embeddings output has data type %T, want *backend.Tensor", value.Data)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != len(texts) {
		return nil, fmt.Errorf("bert_embed embeddings shape = %v, want [%d,D]", tensor.Shape, len(texts))
	}
	dim := tensor.Shape[1]
	out := make([][]float32, len(texts))
	for row := range texts {
		start := row * dim
		out[row] = append([]float32(nil), tensor.F32[start:start+dim]...)
	}
	return out, nil
}

func assertEmbeddingBatchesClose(t *testing.T, got, want [][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("embedding batch rows = %d, want %d", len(got), len(want))
	}
	for row := range got {
		if len(got[row]) != len(want[row]) {
			t.Fatalf("embedding row %d dim = %d, want %d", row, len(got[row]), len(want[row]))
		}
		for col := range got[row] {
			diff := math.Abs(float64(got[row][col] - want[row][col]))
			if diff > tol {
				t.Fatalf("embedding[%d][%d] = %.9g, want %.9g (diff %.3g > %.3g)", row, col, got[row][col], want[row][col], diff, tol)
			}
		}
	}
}

func rowIDs(rows []retrievalVectorExportRow) []string {
	out := make([]string, len(rows))
	for i, row := range rows {
		out[i] = row.ID
	}
	return out
}

func isSHA256Hex(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, ch := range value {
		if (ch < '0' || ch > '9') && (ch < 'a' || ch > 'f') {
			return false
		}
	}
	return true
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return info.Size()
}

func assertFiniteUnitishVector(t *testing.T, vector []float32, dim int) {
	t.Helper()
	if len(vector) != dim {
		t.Fatalf("vector dim = %d, want %d", len(vector), dim)
	}
	var norm float64
	for i, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("vector[%d] = %v, want finite", i, value)
		}
		norm += float64(value * value)
	}
	if math.Sqrt(norm) < 0.9 || math.Sqrt(norm) > 1.1 {
		t.Fatalf("vector norm = %g, want near 1", math.Sqrt(norm))
	}
}
