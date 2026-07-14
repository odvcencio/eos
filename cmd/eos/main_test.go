package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/models"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backend"
	mll "m31labs.dev/mll"
)

func TestRunGraphPrintsSourceJSON(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	output := captureRunOutput(t, []string{"graph", "--format", "json", srcPath})
	var payload struct {
		GraphVersion int    `json:"graph_version"`
		InputKind    string `json:"input_kind"`
		Module       string `json:"module"`
		Counts       struct {
			SourceDecls      int `json:"source_decls"`
			ArtifactKernels  int `json:"artifact_kernels"`
			KernelSourceVars int `json:"kernel_source_variants"`
		} `json:"counts"`
		Artifact struct {
			Name    string `json:"name"`
			Kernels []struct {
				Name     string `json:"name"`
				Variants []struct {
					Backend     string `json:"backend"`
					SourceBytes int    `json:"source_bytes"`
				} `json:"variants"`
			} `json:"kernels"`
		} `json:"artifact"`
	}
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("unmarshal graph output: %v\n%s", err, output)
	}
	if payload.GraphVersion != 1 || payload.InputKind != "source" || payload.Module != "embed" {
		t.Fatalf("unexpected graph identity: %+v", payload)
	}
	if payload.Counts.SourceDecls == 0 || payload.Counts.ArtifactKernels == 0 || payload.Counts.KernelSourceVars == 0 {
		t.Fatalf("graph counts missing compiler structure: %+v", payload.Counts)
	}
	if len(payload.Artifact.Kernels) == 0 || len(payload.Artifact.Kernels[0].Variants) == 0 {
		t.Fatalf("graph output missing kernel variant summary: %+v", payload.Artifact)
	}
	if payload.Artifact.Kernels[0].Variants[0].SourceBytes == 0 {
		t.Fatalf("variant source byte count was not recorded: %+v", payload.Artifact.Kernels[0].Variants[0])
	}
}

func TestRunKernelsExtractsBackendSources(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outDir := filepath.Join(dir, "kernels")
	output := captureRunOutput(t, []string{"kernels", "--backend", "webgpu", "--out", outDir, srcPath})
	if !strings.Contains(output, "wrote ") || !strings.Contains(output, outDir) {
		t.Fatalf("unexpected kernels output:\n%s", output)
	}
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read kernel manifest: %v", err)
	}
	var manifest struct {
		Module            string `json:"module"`
		KernelSourceCount int    `json:"kernel_source_count"`
		Kernels           []struct {
			Backend     string `json:"backend"`
			SourceFile  string `json:"source_file"`
			SourceBytes int    `json:"source_bytes"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal kernel manifest: %v\n%s", err, data)
	}
	if manifest.Module != "embed" || manifest.KernelSourceCount == 0 || len(manifest.Kernels) == 0 {
		t.Fatalf("unexpected kernel manifest: %+v", manifest)
	}
	for _, kernel := range manifest.Kernels {
		if kernel.Backend != "webgpu" {
			t.Fatalf("backend filter leaked variant: %+v", kernel)
		}
		sourcePath := filepath.Join(outDir, kernel.SourceFile)
		source, err := os.ReadFile(sourcePath)
		if err != nil {
			t.Fatalf("read extracted source %q: %v", sourcePath, err)
		}
		if !strings.Contains(string(source), "@compute") || kernel.SourceBytes != len(source) {
			t.Fatalf("unexpected extracted WGSL source %q", sourcePath)
		}
	}
}

func TestRunKernelsValidateRecordsPrismChecks(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outDir := filepath.Join(dir, "kernels")
	captureRunOutput(t, []string{"kernels", "--backend", "webgpu", "--validate", "--out", outDir, srcPath})
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read kernel manifest: %v", err)
	}
	var manifest struct {
		Kernels []struct {
			Validation *struct {
				EntryChecked bool   `json:"entry_checked"`
				ToolSkipped  bool   `json:"tool_skipped"`
				ToolError    string `json:"tool_error"`
			} `json:"validation"`
		} `json:"kernels"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal kernel manifest: %v\n%s", err, data)
	}
	if len(manifest.Kernels) == 0 || manifest.Kernels[0].Validation == nil {
		t.Fatalf("validation metadata missing: %+v", manifest)
	}
	if !manifest.Kernels[0].Validation.EntryChecked {
		t.Fatalf("Prism entry check was not recorded: %+v", manifest.Kernels[0].Validation)
	}
}

func TestRunCompileBundleWritesInspectionArtifacts(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "embed.eos")
	outPath := filepath.Join(dir, "embed.mll")
	bundleDir := filepath.Join(dir, "bundle")
	output := captureRunOutput(t, []string{"compile", "--bundle", bundleDir, srcPath, outPath})
	for _, want := range []string{"bundle: " + bundleDir, "compiled "} {
		if !strings.Contains(output, want) {
			t.Fatalf("compile bundle output missing %q\noutput:\n%s", want, output)
		}
	}
	for _, path := range []string{
		filepath.Join(bundleDir, "manifest.json"),
		filepath.Join(bundleDir, "source.eos"),
		filepath.Join(bundleDir, "artifact.mll"),
		filepath.Join(bundleDir, "graph.json"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected bundle file %q: %v", path, err)
		}
	}
	if _, err := os.Stat(filepath.Join(bundleDir, "kernels")); err != nil {
		t.Fatalf("expected kernels dir: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(bundleDir, "manifest.json"))
	if err != nil {
		t.Fatalf("read bundle manifest: %v", err)
	}
	var manifest struct {
		BundleVersion     int    `json:"bundle_version"`
		Module            string `json:"module"`
		ArtifactPath      string `json:"artifact_path"`
		KernelSourceCount int    `json:"kernel_source_count"`
		KernelSources     []struct {
			SourceFile string `json:"source_file"`
		} `json:"kernel_sources"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("unmarshal bundle manifest: %v\n%s", err, data)
	}
	if manifest.BundleVersion != 1 || manifest.Module != "embed" || manifest.ArtifactPath != outPath || manifest.KernelSourceCount == 0 {
		t.Fatalf("unexpected bundle manifest: %+v", manifest)
	}
	if len(manifest.KernelSources) == 0 || !strings.HasPrefix(manifest.KernelSources[0].SourceFile, "kernels/") {
		t.Fatalf("bundle kernel sources should be manifest-relative: %+v", manifest.KernelSources)
	}
}

func TestCompileRendersSourceDiagnostics(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "bad.eos")
	src := `kernel broken(x: f16[T, E]) -> f16[T, E] {
    return missing
}
`
	if err := os.WriteFile(srcPath, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	runErr := run([]string{"compile", srcPath, filepath.Join(dir, "bad.mll")})
	if runErr == nil {
		t.Fatal("compile succeeded, want diagnostic error")
	}
	var out strings.Builder
	printCommandError(&out, runErr)
	for _, want := range []string{"EOS1001 error", "--> " + srcPath + ":2:", "return missing", "^", "hint:"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("diagnostic output missing %q\n--- output ---\n%s", want, out.String())
		}
	}
}

func TestRunDoctorReportsRuntimeFacts(t *testing.T) {
	output := captureRunOutput(t, []string{"doctor"})
	for _, want := range []string{
		"artifact schema:",
		"go: ",
		"backends:",
		"cuda",
		"webgpu",
		"tools:",
		"env:",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("doctor output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestExportSparseTokenPoolVectorsAcceptsResumeProgressFlags(t *testing.T) {
	err := runExportSparseTokenPoolVectors([]string{"--resume", "--progress-every", "7", "--tokenizer-max-seq", "4096"})
	if err == nil {
		t.Fatal("runExportSparseTokenPoolVectors succeeded without positional args, want usage error")
	}
	if strings.Contains(err.Error(), "flag provided but not defined") {
		t.Fatalf("sparse token-pool flags were not registered: %v", err)
	}
	if !strings.Contains(err.Error(), "usage: eos export-sparse-token-pool-vectors") {
		t.Fatalf("error = %v, want usage error after parsing flags", err)
	}
}

func TestExportSparseEncoderVectorsAcceptsTokenizerMaxSequenceFlag(t *testing.T) {
	err := runExportSparseEncoderVectors([]string{"--tokenizer-max-seq", "4096"})
	if err == nil {
		t.Fatal("runExportSparseEncoderVectors succeeded without positional args, want usage error")
	}
	if strings.Contains(err.Error(), "flag provided but not defined") {
		t.Fatalf("sparse encoder tokenizer max sequence flag was not registered: %v", err)
	}
	if !strings.Contains(err.Error(), "usage: eos export-sparse-encoder-vectors") {
		t.Fatalf("error = %v, want usage error after parsing flags", err)
	}
}

func TestRunMineRetrievalCompactHardNegativesRequiresNoTrainTestSmoke(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "tiny")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","title":"positive","text":"alpha positive"}`+"\n"+
			`{"_id":"d2","title":"negative","text":"alpha negative"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	row := eosruntime.TurboQuantRetrievalPerQueryRow{
		Schema:          eosruntime.TurboQuantRetrievalPerQuerySchema,
		Dataset:         "tiny",
		QueryID:         "q1",
		Method:          "turboquant_ip_b4_overfetch200_fp16_rerank",
		Bits:            4,
		RerankOverfetch: 200,
		RerankStorage:   eosruntime.TurboQuantRerankStorageFP16,
		QuantizerSeed:   eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
		TopK: []eosruntime.RetrievalEvalPerQueryTopDoc{
			{Rank: 1, DocID: "d2", Score: 0.9, Relevance: 0},
		},
	}
	data, err := json.Marshal(row)
	if err != nil {
		t.Fatalf("marshal per-query row: %v", err)
	}
	perQueryPath := filepath.Join(dir, "compact.per-query.jsonl")
	if err := os.WriteFile(perQueryPath, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write per-query row: %v", err)
	}

	blockedOutput := filepath.Join(dir, "blocked.jsonl")
	err = run([]string{
		"mine-retrieval-compact-hard-negatives",
		"--split", "test",
		"--allow-test-smoke",
		"--per-query-jsonl", perQueryPath,
		"--overfetch", "200",
		datasetDir,
		blockedOutput,
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to mine train-selection rows from test split") {
		t.Fatalf("allow-test-smoke bypassed test train-selection guard, err=%v", err)
	}

	outputPath := filepath.Join(dir, "hard-negatives.jsonl")
	manifestPath := filepath.Join(dir, "manifest.json")
	captureRunOutput(t, []string{
		"mine-retrieval-compact-hard-negatives",
		"--split", "test",
		"--allow-test-smoke",
		"--train-selection=false",
		"--per-query-jsonl", perQueryPath,
		"--manifest-json", manifestPath,
		"--overfetch", "200",
		datasetDir,
		outputPath,
	})
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest eosruntime.CompactHardNegativeMiningManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("unmarshal manifest: %v\n%s", err, manifestData)
	}
	if manifest.TrainSelection || manifest.TrainAllowed || manifest.LeakGuardStatus != "validation_smoke_no_train_test_split" {
		t.Fatalf("manifest guard state = %+v", manifest)
	}
}

func TestRunDefaultEmbedderPathOnly(t *testing.T) {
	output := strings.TrimSpace(captureRunOutput(t, []string{"default-embedder", "--path-only"}))
	if filepath.Base(output) != models.DefaultEmbedderArtifactFilename {
		t.Fatalf("default embedder path = %q", output)
	}
	if !strings.Contains(filepath.ToSlash(output), models.DefaultEmbedderAssetRelativeDir+"/"+models.DefaultEmbedderArtifactFilename) {
		t.Fatalf("default embedder path does not point at asset dir: %q", output)
	}
}

func TestRunDefaultEmbedderVerifyJSON(t *testing.T) {
	output := captureRunOutput(t, []string{"default-embedder", "--verify", "--json"})
	var payload struct {
		Asset struct {
			AssetID        string `json:"asset_id"`
			ModelName      string `json:"model_name"`
			ArtifactPath   string `json:"artifact_path"`
			ArtifactSHA256 string `json:"artifact_sha256"`
		} `json:"asset"`
		Verification struct {
			OK    bool `json:"ok"`
			Files []struct {
				Role   string `json:"role"`
				SHA256 string `json:"sha256"`
				OK     bool   `json:"ok"`
			} `json:"files"`
		} `json:"verification"`
	}
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("unmarshal default embedder JSON: %v\n%s", err, output)
	}
	if payload.Asset.AssetID != models.DefaultEmbedderAssetID || payload.Asset.ModelName != models.DefaultEmbeddingModelName {
		t.Fatalf("unexpected default embedder identity: %+v", payload.Asset)
	}
	if payload.Asset.ArtifactSHA256 != models.DefaultEmbedderArtifactSHA256 {
		t.Fatalf("artifact sha = %q", payload.Asset.ArtifactSHA256)
	}
	if !payload.Verification.OK || len(payload.Verification.Files) != 2 {
		t.Fatalf("unexpected verification: %+v", payload.Verification)
	}
	for _, check := range payload.Verification.Files {
		if !check.OK || check.SHA256 == "" {
			t.Fatalf("bad verification check: %+v", check)
		}
	}
}

func TestRunImportedEmbedderCandidatePathOnlyUsesPackageOverride(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "BAAI/bge-small-en-v1.5", true)
	output := strings.TrimSpace(captureRunOutput(t, []string{
		"imported-embedder-candidate",
		"--package", packagePath,
		"--path-only",
	}))
	if output != packagePath {
		t.Fatalf("candidate package path = %q, want %q", output, packagePath)
	}
	if strings.Contains(output, models.DefaultEmbedderAssetID) || strings.Contains(output, models.DefaultEmbedderAssetRelativeDir) {
		t.Fatalf("path-only output should not claim default asset: %q", output)
	}
}

func TestRunImportedEmbedderCandidateJSONIsNonDefaultReference(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "BAAI/bge-small-en-v1.5", true)
	output := captureRunOutput(t, []string{
		"imported-embedder-candidate",
		"--package", packagePath,
		"--json",
	})
	var payload struct {
		Asset struct {
			CandidateID              string `json:"candidate_id"`
			PublicID                 string `json:"public_id"`
			ModelName                string `json:"model_name"`
			DisplayName              string `json:"display_name"`
			LegacyModelName          string `json:"legacy_model_name"`
			SourceModel              string `json:"source_model"`
			Status                   string `json:"status"`
			PublicIdentityNote       string `json:"public_identity_note"`
			PackagePath              string `json:"package_path"`
			PackageSHA256            string `json:"package_sha256"`
			PackageIdentity          string `json:"package_identity"`
			SourceSnapshotCommit     string `json:"source_snapshot_commit"`
			UpstreamModelURL         string `json:"upstream_model_url"`
			LicenseID                string `json:"license_id"`
			LicenseNoticeRequired    bool   `json:"license_notice_required"`
			ProvenanceNoticeRequired bool   `json:"provenance_notice_required"`
			Attribution              string `json:"attribution"`
			RoleContractSchema       string `json:"role_contract_schema"`
			QueryRole                string `json:"query_role"`
			QueryPrefix              string `json:"query_prefix"`
			DocumentRole             string `json:"document_role"`
			DocumentPrefix           string `json:"document_prefix"`
			Pooling                  string `json:"pooling"`
			Normalization            string `json:"normalization"`
			MaxLength                int    `json:"max_length"`
			NativeDim                int    `json:"native_dim"`
			QualityClaim             bool   `json:"quality_claim"`
			DefaultAliasChanged      bool   `json:"default_alias_changed"`
			LoadPath                 string `json:"load_path"`
		} `json:"asset"`
		Verification *models.ImportedEmbedderCandidateVerification `json:"verification,omitempty"`
	}
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("unmarshal imported candidate JSON: %v\n%s", err, output)
	}
	if payload.Asset.CandidateID != models.ImportedEmbedderCandidateID {
		t.Fatalf("candidate id = %q", payload.Asset.CandidateID)
	}
	if payload.Asset.PublicID != models.ImportedEmbedderCandidatePublicID {
		t.Fatalf("public id = %q", payload.Asset.PublicID)
	}
	if payload.Asset.ModelName != models.ImportedEmbedderCandidateModelName {
		t.Fatalf("model name = %q", payload.Asset.ModelName)
	}
	if payload.Asset.ModelName != models.ImportedEmbedderCandidatePublicID || payload.Asset.LegacyModelName != models.ImportedEmbedderCandidateLegacyModelName {
		t.Fatalf("unexpected public/legacy model fields: %+v", payload.Asset)
	}
	if payload.Asset.DisplayName != models.ImportedEmbedderCandidatePublicName {
		t.Fatalf("display name = %q", payload.Asset.DisplayName)
	}
	if payload.Asset.SourceModel != models.ImportedEmbedderCandidateSourceModel {
		t.Fatalf("source model = %q", payload.Asset.SourceModel)
	}
	if payload.Asset.Status != models.ImportedEmbedderCandidateStatus {
		t.Fatalf("status = %q", payload.Asset.Status)
	}
	if !strings.Contains(payload.Asset.PublicIdentityNote, "public_id/model_name/display_name") ||
		!strings.Contains(payload.Asset.PublicIdentityNote, "legacy_model_name is compatibility") {
		t.Fatalf("public identity note = %q", payload.Asset.PublicIdentityNote)
	}
	if payload.Asset.PackagePath != packagePath {
		t.Fatalf("unexpected package path fields: %+v", payload.Asset)
	}
	if payload.Asset.PackageSHA256 != models.ImportedEmbedderCandidatePackageSHA256 || payload.Asset.PackageIdentity != models.ImportedEmbedderCandidatePackageIdentity {
		t.Fatalf("unexpected package identity fields: %+v", payload.Asset)
	}
	if payload.Asset.SourceSnapshotCommit != "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a" || payload.Asset.UpstreamModelURL != "https://huggingface.co/BAAI/bge-small-en-v1.5" {
		t.Fatalf("unexpected provenance fields: %+v", payload.Asset)
	}
	if payload.Asset.LicenseID != "MIT" || !payload.Asset.LicenseNoticeRequired || !payload.Asset.ProvenanceNoticeRequired || payload.Asset.Attribution != "FlagEmbedding/BAAI" {
		t.Fatalf("unexpected license fields: %+v", payload.Asset)
	}
	if payload.Asset.RoleContractSchema != eosruntime.PretrainedBERTRetrievalRoleContractSchema ||
		payload.Asset.QueryRole != "query" ||
		payload.Asset.QueryPrefix != "Represent this sentence for searching relevant passages: " ||
		payload.Asset.DocumentRole != "document" ||
		payload.Asset.DocumentPrefix != "" ||
		payload.Asset.Pooling != "cls" ||
		payload.Asset.Normalization != "l2" ||
		payload.Asset.MaxLength != 512 ||
		payload.Asset.NativeDim != 384 {
		t.Fatalf("unexpected role contract fields: %+v", payload.Asset)
	}
	if payload.Asset.QualityClaim || payload.Asset.DefaultAliasChanged {
		t.Fatalf("candidate unexpectedly claims quality/default: %+v", payload.Asset)
	}
	if payload.Asset.LoadPath != "runtime.LoadImportedBERTEmbedderCandidate" {
		t.Fatalf("load path = %q", payload.Asset.LoadPath)
	}
	if payload.Verification != nil {
		t.Fatalf("verification should be omitted without --verify: %+v", payload.Verification)
	}
}

func TestRunImportedEmbedderCandidateTextPrintsAuditFields(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "BAAI/bge-small-en-v1.5", true)
	output := captureRunOutput(t, []string{
		"imported-embedder-candidate",
		"--package", packagePath,
	})
	for _, want := range []string{
		"package_sha256: " + models.ImportedEmbedderCandidatePackageSHA256,
		"package_identity: " + models.ImportedEmbedderCandidatePackageIdentity,
		"source_snapshot_commit: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
		"upstream_model_url: https://huggingface.co/BAAI/bge-small-en-v1.5",
		"license: MIT attribution=\"FlagEmbedding/BAAI\" notice_required=true provenance_notice_required=true",
		"role_contract: schema=" + eosruntime.PretrainedBERTRetrievalRoleContractSchema,
		"query_prefix=\"Represent this sentence for searching relevant passages: \"",
		"document_prefix=\"\" pooling=cls normalization=l2 max_length=512 native_dim=384",
		"quality_claim: false",
		"default_alias_changed: false",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output missing %q:\n%s", want, output)
		}
	}
}

func TestRunImportedEmbedderCandidateRequiresExplicitPackageOrRoot(t *testing.T) {
	err := runImportedEmbedderCandidate([]string{"--json"})
	if err == nil || !strings.Contains(err.Error(), "package path is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunImportedEmbedderCandidateVerifyJSONMismatch(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "BAAI/bge-small-en-v1.5", true)
	output, err := captureRunOutputAndError(t, []string{
		"imported-embedder-candidate",
		"--package", packagePath,
		"--verify",
		"--json",
	})
	if err == nil {
		t.Fatal("expected verification mismatch")
	}
	var payload struct {
		Asset struct {
			CandidateID     string `json:"candidate_id"`
			PublicID        string `json:"public_id"`
			ModelName       string `json:"model_name"`
			LegacyModelName string `json:"legacy_model_name"`
		} `json:"asset"`
		Verification struct {
			OK   bool `json:"ok"`
			File struct {
				OK             bool   `json:"ok"`
				SHA256         string `json:"sha256"`
				ExpectedSHA256 string `json:"expected_sha256"`
			} `json:"file"`
		} `json:"verification"`
	}
	if err := json.Unmarshal([]byte(output), &payload); err != nil {
		t.Fatalf("unmarshal verification JSON: %v\n%s", err, output)
	}
	if payload.Asset.CandidateID != models.ImportedEmbedderCandidateID ||
		payload.Asset.PublicID != models.ImportedEmbedderCandidatePublicID ||
		payload.Asset.ModelName != models.ImportedEmbedderCandidateModelName ||
		payload.Asset.LegacyModelName != models.ImportedEmbedderCandidateLegacyModelName {
		t.Fatalf("unexpected asset identity: %+v", payload.Asset)
	}
	if payload.Verification.OK || payload.Verification.File.OK {
		t.Fatalf("expected failed file verification: %+v", payload.Verification)
	}
	if payload.Verification.File.SHA256 == "" || payload.Verification.File.ExpectedSHA256 != models.ImportedEmbedderCandidatePackageSHA256 {
		t.Fatalf("unexpected file check: %+v", payload.Verification.File)
	}
}

func TestVerifyImportedEmbedderCandidateWithConfigFixtureSuccessAndIdentityMismatch(t *testing.T) {
	packagePath := writeCommandPretrainedBERTPackageFixture(t, "BAAI/bge-small-en-v1.5", true)
	pkg, err := eosruntime.ReadPretrainedBERTPackageFile(packagePath)
	if err != nil {
		t.Fatalf("read package: %v", err)
	}
	packageSHA := commandTestSHA256File(t, packagePath)
	report, err := models.VerifyImportedEmbedderCandidateWithConfig(models.ImportedEmbedderCandidateVerifyConfig{
		PackagePath:            packagePath,
		ExpectedSHA256:         packageSHA,
		ExpectedIdentitySHA256: pkg.IdentityHash(),
	})
	if err != nil {
		t.Fatalf("verify fixture candidate: %v", err)
	}
	if !report.OK || !report.File.OK || !report.Identity.OK {
		t.Fatalf("unexpected success report: %+v", report)
	}
	report, err = models.VerifyImportedEmbedderCandidateWithConfig(models.ImportedEmbedderCandidateVerifyConfig{
		PackagePath:            packagePath,
		ExpectedSHA256:         packageSHA,
		ExpectedIdentitySHA256: strings.Repeat("0", 64),
	})
	if err == nil || !strings.Contains(err.Error(), "verification failed") {
		t.Fatalf("expected identity mismatch, got report=%+v err=%v", report, err)
	}
	if report.OK || report.Identity.OK || !report.File.OK {
		t.Fatalf("unexpected mismatch report: %+v", report)
	}
}

func TestParseNonNegativeFloatMapAllowsZeroTeacherSourceWeight(t *testing.T) {
	got, err := parseNonNegativeFloatMap("scifact=1,nfcorpus=0,fiqa=0.25")
	if err != nil {
		t.Fatalf("parse non-negative float map: %v", err)
	}
	if got["scifact"] != 1 {
		t.Fatalf("scifact weight = %f, want 1", got["scifact"])
	}
	if got["nfcorpus"] != 0 {
		t.Fatalf("nfcorpus weight = %f, want 0", got["nfcorpus"])
	}
	if got["fiqa"] != 0.25 {
		t.Fatalf("fiqa weight = %f, want 0.25", got["fiqa"])
	}
}

func TestParseNonNegativeFloatMapRejectsNegativeTeacherSourceWeight(t *testing.T) {
	if _, err := parseNonNegativeFloatMap("fiqa=-0.1"); err == nil {
		t.Fatal("parse non-negative float map succeeded for negative weight")
	}
}

func TestFormatTrainThroughputIncludesExamplePairAndStepRates(t *testing.T) {
	summary := eosruntime.EmbeddingTrainRunSummary{
		Workload: eosruntime.EmbeddingTrainWorkload{
			ActualTotalExamples: 100,
			ActualTotalPairs:    10000,
			ActualTrainExamples: 80,
			ActualTrainPairs:    8000,
			ActualEvalExamples:  20,
			ActualEvalPairs:     2000,
		},
		Elapsed:       10 * time.Second,
		TrainDuration: 4 * time.Second,
		EvalDuration:  2 * time.Second,
		StepsRun:      8,
	}

	output := formatTrainThroughput(summary)
	for _, want := range []string{
		"elapsed=10s",
		"examples/s=10.00",
		"pairs/s=1000.00",
		"train_examples/s=20.00",
		"train_pairs/s=2000.00",
		"eval_examples/s=10.00",
		"eval_pairs/s=1000.00",
		"optimizer_steps/s=2.00",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("throughput output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestWriteTurboQuantRetrievalMetricsTSVIncludesLatencyColumns(t *testing.T) {
	path := filepath.Join(t.TempDir(), "turboquant.tsv")
	metrics := eosruntime.TurboQuantRetrievalEvalMetrics{
		Dataset: "tiny",
		Dense: eosruntime.TurboQuantDenseRetrievalMetrics{
			Quality: eosruntime.RetrievalEvalQualityMetrics{
				NDCGAt10:    1,
				NDCGAt100:   1,
				MRRAt10:     1,
				RecallAt10:  1,
				RecallAt100: 1,
			},
			VectorBytes:     64,
			ScoresPerSecond: 1000,
			QueryLatency: eosruntime.RetrievalEvalLatencyMetrics{
				Count: 2,
				P50MS: 0.1,
				P95MS: 0.2,
				P99MS: 0.3,
				MaxMS: 0.4,
			},
		},
		Rows: []eosruntime.TurboQuantRetrievalBitMetrics{{
			Bits:             4,
			Method:           "turboquant_ip_b4_overfetch250_fp16_rerank",
			RerankOverfetch:  250,
			RerankStorage:    eosruntime.TurboQuantRerankStorageFP16,
			Quality:          eosruntime.RetrievalEvalQualityMetrics{NDCGAt10: 1, NDCGAt100: 1, MRRAt10: 1, RecallAt10: 1, RecallAt100: 1},
			VectorBytes:      8,
			DenseVectorBytes: 64,
			CompressionRatio: 8,
			TotalVectorBytes: 40,
			TotalCompression: 1.6,
			ScoresPerSecond:  900,
			QueryLatency:     eosruntime.RetrievalEvalLatencyMetrics{Count: 2, P50MS: 0.5, P95MS: 0.6, P99MS: 0.7, MaxMS: 0.8},
			RerankScores:     500,
		}},
	}
	if err := writeTurboQuantRetrievalMetricsTSV(path, metrics); err != nil {
		t.Fatalf("write tsv: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read tsv: %v", err)
	}
	text := string(data)
	for _, want := range []string{
		"query_latency_p50_ms\tquery_latency_p95_ms\tquery_latency_p99_ms\tquery_latency_max_ms",
		"\t0.100000\t0.200000\t0.300000\t0.400000\t",
		"\t0.500000\t0.600000\t0.700000\t0.800000\t",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("tsv missing %q\n%s", want, text)
		}
	}
}

func TestRunPlanMultiVectorStorageWritesTSVAndJSON(t *testing.T) {
	jsonPath := filepath.Join(t.TempDir(), "multivector-storage.json")
	output := captureRunOutput(t, []string{
		"plan-multivector-storage",
		"--dim", "128",
		"--baseline-dim", "3072",
		"--bits", "2,4",
		"--vectors-per-object", "1,16",
		"--objects", "1000",
		"--json", jsonPath,
	})
	for _, want := range []string{
		"dim\tbaseline_dim\tbits\tobjects\tvectors_per_object\tdense_parent_bytes",
		"quantized_vector_bytes\tvector_overhead_bytes\tdense_vector_storage_bytes\tquantized_vector_storage_bytes\ttotal_quantized_bytes\tpacked_object_overhead_bytes\tpacked_quantized_storage_bytes\tpacked_total_quantized_bytes",
		"packed_vectors_that_fit_in_one_dense_vector\tpacked_fits_in_one_dense_vector_storage\tpacked_storage_multiple_of_dense_parent_cost",
		"128\t3072\t2\t1000\t1\t12288\t12288000\t12288\t12288000\t36\tnone\t0\t36\t0\t12288\t36\t36000\t0\t36\t36000\t341.333333\t341.333333\t341.333333\t341\ttrue\t0.002930\t341\ttrue\t0.002930",
		"128\t3072\t4\t1000\t16\t12288\t12288000\t12288\t12288000\t68\tnone\t0\t68\t0\t12288\t68\t1088000\t0\t1088\t1088000\t180.705882\t11.294118\t11.294118\t180\ttrue\t0.088542\t180\ttrue\t0.088542",
		"json: " + jsonPath,
		"summary: rows=4 dim=128 baseline_dim=3072 objects=1000 sidecar_storage=none vector_overhead_bytes=0 packed_object_overhead_bytes=0",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-multivector-storage output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json: %v", err)
	}
	var plan eosruntime.MultiVectorStoragePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("decode json: %v\n%s", err, data)
	}
	if plan.Schema != eosruntime.MultiVectorStoragePlanSchema || len(plan.Rows) != 4 {
		t.Fatalf("plan identity = schema:%q rows:%d", plan.Schema, len(plan.Rows))
	}
	if plan.Config.BaselineDim != 3072 || plan.Rows[0].BaselineDim != 3072 {
		t.Fatalf("baseline dim = config:%d row:%d", plan.Config.BaselineDim, plan.Rows[0].BaselineDim)
	}
	if plan.Config.VectorOverheadBytes != 0 || plan.Rows[0].VectorOverheadBytes != 0 {
		t.Fatalf("vector overhead = config:%d row:%d", plan.Config.VectorOverheadBytes, plan.Rows[0].VectorOverheadBytes)
	}
	if plan.Config.PackedObjectOverheadBytes != 0 || plan.Rows[0].PackedObjectOverheadBytes != 0 {
		t.Fatalf("packed object overhead = config:%d row:%d", plan.Config.PackedObjectOverheadBytes, plan.Rows[0].PackedObjectOverheadBytes)
	}
	if plan.Rows[0].VectorsThatFitInOneDenseVector != 341 {
		t.Fatalf("vectors_that_fit = %d", plan.Rows[0].VectorsThatFitInOneDenseVector)
	}
	if plan.Rows[0].PackedVectorsThatFitInOneDenseVector != 341 {
		t.Fatalf("packed_vectors_that_fit = %d", plan.Rows[0].PackedVectorsThatFitInOneDenseVector)
	}
}

func TestRunPlanMultiVectorStorageAccountsForVectorOverhead(t *testing.T) {
	jsonPath := filepath.Join(t.TempDir(), "multivector-storage-overhead.json")
	output := captureRunOutput(t, []string{
		"plan-multivector-storage",
		"--dim", "128",
		"--baseline-dim", "3072",
		"--bits", "2",
		"--vectors-per-object", "64,128,256",
		"--objects", "1000",
		"--vector-overhead-bytes", "32",
		"--packed-object-overhead-bytes", "32",
		"--json", jsonPath,
	})
	for _, want := range []string{
		"128\t3072\t2\t1000\t64\t12288\t12320000\t12288\t12320000\t36\tnone\t0\t36\t32\t12320\t68\t4352000\t32\t2336\t2336000\t181.176471\t2.830882\t5.273973\t181\ttrue\t0.353247\t341\ttrue\t0.189610",
		"128\t3072\t2\t1000\t128\t12288\t12320000\t12288\t12320000\t36\tnone\t0\t36\t32\t12320\t68\t8704000\t32\t4640\t4640000\t181.176471\t1.415441\t2.655172\t181\ttrue\t0.706494\t341\ttrue\t0.376623",
		"128\t3072\t2\t1000\t256\t12288\t12320000\t12288\t12320000\t36\tnone\t0\t36\t32\t12320\t68\t17408000\t32\t9248\t9248000\t181.176471\t0.707721\t1.332180\t181\tfalse\t1.412987\t341\ttrue\t0.750649",
		"json: " + jsonPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-multivector-storage output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json: %v", err)
	}
	var plan eosruntime.MultiVectorStoragePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("decode json: %v\n%s", err, data)
	}
	row := plan.Rows[0]
	if plan.Config.VectorOverheadBytes != 32 || row.VectorOverheadBytes != 32 || plan.Config.PackedObjectOverheadBytes != 32 || row.PackedObjectOverheadBytes != 32 {
		t.Fatalf("overhead = vector config:%d row:%d packed config:%d row:%d", plan.Config.VectorOverheadBytes, row.VectorOverheadBytes, plan.Config.PackedObjectOverheadBytes, row.PackedObjectOverheadBytes)
	}
	if row.DenseVectorStorageBytes != 12320 || row.QuantizedVectorStorageBytes != 68 {
		t.Fatalf("storage bytes = dense:%d quantized:%d", row.DenseVectorStorageBytes, row.QuantizedVectorStorageBytes)
	}
	if row.PackedQuantizedStorageBytes != 2336 || row.PackedTotalQuantizedBytes != 2336000 || row.PackedVectorsThatFitInOneDenseVector != 341 {
		t.Fatalf("packed storage = per_parent:%d total:%d fit:%d", row.PackedQuantizedStorageBytes, row.PackedTotalQuantizedBytes, row.PackedVectorsThatFitInOneDenseVector)
	}
}

func TestRunPlanMultiVectorStorageAccountsForPackedParentOverheadByBitWidth(t *testing.T) {
	jsonPath := filepath.Join(t.TempDir(), "multivector-storage-packed.json")
	_ = captureRunOutput(t, []string{
		"plan-multivector-storage",
		"--dim", "128",
		"--baseline-dim", "3072",
		"--bits", "2,4,8",
		"--vectors-per-object", "100",
		"--objects", "1000",
		"--vector-overhead-bytes", "32",
		"--packed-object-overhead-bytes", "32",
		"--json", jsonPath,
	})
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json: %v", err)
	}
	var plan eosruntime.MultiVectorStoragePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("decode json: %v\n%s", err, data)
	}
	tests := []struct {
		bits          int
		currentFit    int64
		packedFit     int64
		packedStorage int64
		packedFits    bool
	}{
		{bits: 2, currentFit: 181, packedFit: 341, packedStorage: 3632, packedFits: true},
		{bits: 4, currentFit: 123, packedFit: 180, packedStorage: 6832, packedFits: true},
		{bits: 8, currentFit: 75, packedFit: 93, packedStorage: 13232, packedFits: false},
	}
	for i, tt := range tests {
		row := plan.Rows[i]
		if row.Bits != tt.bits || row.VectorsThatFitInOneDenseVector != tt.currentFit || row.PackedVectorsThatFitInOneDenseVector != tt.packedFit {
			t.Fatalf("row %d fit = bits:%d current:%d packed:%d", i, row.Bits, row.VectorsThatFitInOneDenseVector, row.PackedVectorsThatFitInOneDenseVector)
		}
		if row.PackedQuantizedStorageBytes != tt.packedStorage || row.PackedFitsInOneDenseVectorStorage != tt.packedFits {
			t.Fatalf("q%d packed storage = bytes:%d fits:%t", tt.bits, row.PackedQuantizedStorageBytes, row.PackedFitsInOneDenseVectorStorage)
		}
	}
}

func TestRunPlanMultiVectorStorageDerivesTimeSeriesWindows(t *testing.T) {
	jsonPath := filepath.Join(t.TempDir(), "multivector-storage-series.json")
	output := captureRunOutput(t, []string{
		"plan-multivector-storage",
		"--dim", "128",
		"--baseline-dim", "3072",
		"--bits", "2",
		"--series-lengths", "256,1024",
		"--window-size", "64",
		"--window-stride", "16",
		"--objects", "1000",
		"--vector-overhead-bytes", "32",
		"--packed-object-overhead-bytes", "32",
		"--json", jsonPath,
	})
	for _, want := range []string{
		"packed_storage_multiple_of_dense_parent_cost\tseries_length\twindow_size\twindow_stride\tderived_window_count",
		"128\t3072\t2\t1000\t13\t12288\t12320000\t12288\t12320000\t36\tnone\t0\t36\t32\t12320\t68\t884000\t32\t500\t500000\t181.176471\t13.936652\t24.640000\t181\ttrue\t0.071753\t341\ttrue\t0.040584\t256\t64\t16\t13",
		"128\t3072\t2\t1000\t61\t12288\t12320000\t12288\t12320000\t36\tnone\t0\t36\t32\t12320\t68\t4148000\t32\t2228\t2228000\t181.176471\t2.970106\t5.529623\t181\ttrue\t0.336688\t341\ttrue\t0.180844\t1024\t64\t16\t61",
		"json: " + jsonPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-multivector-storage output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json: %v", err)
	}
	var plan eosruntime.MultiVectorStoragePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		t.Fatalf("decode json: %v\n%s", err, data)
	}
	if got, want := plan.Config.VectorsPerObject, []int{13, 61}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("vectors_per_object = %v, want %v", got, want)
	}
	if plan.Config.WindowSize != 64 || plan.Config.WindowStride != 16 || len(plan.Config.SeriesLengths) != 2 {
		t.Fatalf("config time series = lengths:%v window:%d stride:%d", plan.Config.SeriesLengths, plan.Config.WindowSize, plan.Config.WindowStride)
	}
	if plan.Rows[1].SeriesLength != 1024 || plan.Rows[1].DerivedWindowCount != 61 {
		t.Fatalf("series row = %+v", plan.Rows[1])
	}
}

func TestRunPlanMultiVectorStorageRejectsExplicitVectorsWithSeriesLengths(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"plan-multivector-storage",
		"--series-lengths", "256",
		"--window-size", "64",
		"--vectors-per-object", "13",
	})
	if err == nil {
		t.Fatal("plan-multivector-storage succeeded with explicit vectors-per-object and series-lengths")
	}
	if !strings.Contains(err.Error(), "use either --series-lengths") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunPlanMultiVectorStorageRejectsNegativeVectorOverhead(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"plan-multivector-storage",
		"--vector-overhead-bytes", "-1",
	})
	if err == nil {
		t.Fatal("plan-multivector-storage succeeded with negative overhead")
	}
	if !strings.Contains(err.Error(), "vector-overhead-bytes must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunPlanMultiVectorStorageRejectsNegativePackedObjectOverhead(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"plan-multivector-storage",
		"--packed-object-overhead-bytes", "-1",
	})
	if err == nil {
		t.Fatal("plan-multivector-storage succeeded with negative packed object overhead")
	}
	if !strings.Contains(err.Error(), "packed-object-overhead-bytes must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunEvalRetrievalMultiVectorTurboQuantWritesMetrics(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "child-doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	metricsPath := filepath.Join(dir, "metrics.json")
	tsvPath := filepath.Join(dir, "metrics.tsv")
	perQueryPath := filepath.Join(dir, "per-query.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"p1","text":"alpha parent"}`+"\n"+
			`{"_id":"p2","text":"beta parent"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"alpha query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\tp1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"parent_id":"p1","child_id":"p1-a","vector":[0,1,0,0,0,0,0,0]}`+"\n"+
			`{"parent_id":"p1","child_id":"p1-b","vector":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"parent_id":"p2","child_id":"p2-a","vector":[0,1,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[1,0,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	output := captureRunOutput(t, []string{
		"eval-retrieval-multivector-turboquant",
		"--dataset", "tiny-multivector",
		"--backend", "unit",
		"--artifact", "unit-cache",
		"--bits", "8",
		"--quantizer-seed", "99",
		"--baseline-dim", "32",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--metrics-json", metricsPath,
		"--metrics-tsv", tsvPath,
		"--per-query-jsonl", perQueryPath,
		dir,
	})
	for _, want := range []string{
		"retrieval multivector turboquant: dataset=tiny-multivector backend=unit parents=2 child_vectors=3 avg_children=1.50",
		"dense-child: ndcg@10=1.000000",
		"baseline_dim=32 dense_baseline_bytes=128 dense_baseline_total_bytes=256 dense_child_bytes=96 storage_multiple=0.38x",
		"q8: ndcg@10=",
		"metrics: " + metricsPath,
		"metrics_tsv: " + tsvPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output missing %q\n%s", want, output)
		}
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics eosruntime.TurboQuantMultiVectorRetrievalEvalMetrics
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v\n%s", err, data)
	}
	if metrics.Schema != eosruntime.TurboQuantMultiVectorRetrievalEvalMetricsSchema || metrics.Artifact != "unit-cache" || metrics.Backend != "unit" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.Parents != 2 || metrics.Inputs.ChildVectors != 3 || metrics.Inputs.ScoredChildPairs != 3 || metrics.Dense.Quality.NDCGAt10 != 1 {
		t.Fatalf("metrics accounting/quality = %+v dense=%+v", metrics.Inputs, metrics.Dense.Quality)
	}
	if metrics.Config.BaselineDim != 32 || metrics.Inputs.ParentCount != 2 || metrics.Inputs.ChildCount != 3 || metrics.Inputs.MaxChildrenPerParent != 2 {
		t.Fatalf("baseline/count accounting = config:%+v inputs:%+v", metrics.Config, metrics.Inputs)
	}
	if metrics.Dense.DenseBaselineBytes != 128 || metrics.Dense.DenseBaselineTotalBytes != 256 || metrics.Dense.StorageMultipleOfDenseBaseline != 0.375 {
		t.Fatalf("dense storage accounting = %+v", metrics.Dense)
	}
	if metrics.Config.AllowMissingRelevant || metrics.Config.QuantizerSeed != 99 || len(metrics.Rows) != 1 || metrics.Rows[0].QuantizerSeed != 99 {
		t.Fatalf("seed/strict config = config:%+v rows:%+v", metrics.Config, metrics.Rows)
	}
	if metrics.Rows[0].BaselineDim != 32 || metrics.Rows[0].QuantizedVectorBytes <= 0 || metrics.Rows[0].VectorsThatFitInOneDenseBaseline <= 0 || metrics.Rows[0].StorageMultipleOfDenseBaseline <= 0 {
		t.Fatalf("row storage accounting = %+v", metrics.Rows[0])
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query: %v", err)
	}
	perQueryLines := strings.Split(strings.TrimSpace(string(perQueryData)), "\n")
	if len(perQueryLines) != 2 {
		t.Fatalf("per-query lines = %d, want dense and q8 rows\n%s", len(perQueryLines), perQueryData)
	}
	var perQueryRow eosruntime.TurboQuantMultiVectorRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(perQueryLines[1]), &perQueryRow); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if perQueryRow.Schema != eosruntime.TurboQuantMultiVectorRetrievalPerQuerySchema || perQueryRow.Method != "turboquant_ip_b8_child_max" || perQueryRow.QuantizerSeed != 99 || len(perQueryRow.TopK) == 0 || perQueryRow.TopK[0].ChildID == "" {
		t.Fatalf("per-query row = %+v", perQueryRow)
	}
	tsv, err := os.ReadFile(tsvPath)
	if err != nil {
		t.Fatalf("read tsv: %v", err)
	}
	for _, want := range []string{
		"quantizer_seed\tallow_missing_relevant\tbaseline_dim\tparent_count\tchild_count",
		"vectors_that_fit_in_one_dense_baseline\tstorage_multiple_of_dense_baseline\tparent_budget_storage_multiple",
		"tiny-multivector\tdense-child",
		"tiny-multivector\tquantized-child\t8\tturboquant_ip_b8_child_max\t99\tfalse\t32\t2\t3",
	} {
		if !strings.Contains(string(tsv), want) {
			t.Fatalf("tsv missing %q\n%s", want, tsv)
		}
	}
}

func TestRunEvalRetrievalMultiVectorTurboQuantAggregationFlagsRecordMetrics(t *testing.T) {
	dir := t.TempDir()
	qrelsDir := filepath.Join(dir, "qrels")
	if err := os.Mkdir(qrelsDir, 0o755); err != nil {
		t.Fatalf("mkdir qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	docVectorsPath := filepath.Join(dir, "child-doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	metricsPath := filepath.Join(dir, "metrics.json")
	perQueryPath := filepath.Join(dir, "per-query.jsonl")
	if err := os.WriteFile(corpusPath, []byte(
		`{"_id":"steady","text":"steady parent"}`+"\n"+
			`{"_id":"spiky","text":"spiky parent"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"_id":"q1","text":"steady query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\tsteady\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"parent_id":"steady","child_id":"steady-a","vector":[0.9,0,0,0,0,0,0,0]}`+"\n"+
			`{"parent_id":"steady","child_id":"steady-b","vector":[0.9,0,0,0,0,0,0,0]}`+"\n"+
			`{"parent_id":"spiky","child_id":"spiky-a","vector":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"parent_id":"spiky","child_id":"spiky-b","vector":[-1,0,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"id":"q1","vector":[1,0,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}

	captureRunOutput(t, []string{
		"eval-retrieval-multivector-turboquant",
		"--dataset", "tiny-aggregation",
		"--bits", "8",
		"--quantizer-seed", "99",
		"--aggregation", "top2-mean",
		"--child-count-penalty", "0.005",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		dir,
	})
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics eosruntime.TurboQuantMultiVectorRetrievalEvalMetrics
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v\n%s", err, data)
	}
	if metrics.Config.Aggregation != "top2-mean" || metrics.Config.ChildCountPenalty != 0.005 || metrics.Dense.Aggregation != "top2-mean" {
		t.Fatalf("aggregation metrics = config:%+v dense:%+v", metrics.Config, metrics.Dense)
	}
	if len(metrics.Rows) != 1 || metrics.Rows[0].Aggregation != "top2-mean" || metrics.Rows[0].ChildCountPenalty != 0.005 || !strings.Contains(metrics.Rows[0].Method, "top2-mean") {
		t.Fatalf("row aggregation metrics = %+v", metrics.Rows)
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(perQueryData)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want dense and q8 rows\n%s", len(lines), perQueryData)
	}
	var row eosruntime.TurboQuantMultiVectorRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.Aggregation != "top2-mean" || row.ChildCountPenalty != 0.005 || row.TopK[0].ChildID == "" || row.TopK[0].ChildScore == nil {
		t.Fatalf("per-query aggregation row = %+v", row)
	}
}

func TestRunEvalRetrievalMultiVectorTurboQuantRejectsNegativePenalty(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"eval-retrieval-multivector-turboquant",
		"--aggregation", "max",
		"--child-count-penalty", "-0.001",
		"--doc-vectors", "missing-docs.jsonl",
		"--query-vectors", "missing-queries.jsonl",
		"datasets/longembed/repo-docs",
	})
	if err == nil {
		t.Fatal("eval-retrieval-multivector-turboquant succeeded with negative penalty")
	}
	if !strings.Contains(err.Error(), "child-count-penalty must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunInitTrainCreatesTrainingPackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	for _, candidate := range []string{
		eosruntime.DefaultWeightFilePath(path),
		eosruntime.DefaultMemoryPlanPath(path),
		eosruntime.DefaultEmbeddingTrainManifestPath(path),
		eosruntime.DefaultEmbeddingCheckpointPath(path),
		eosruntime.DefaultEmbeddingTrainProfilePath(path),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
}

func TestRunInitTrainAppliesTrainingConfigWithDefaultManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", "--lr", "0.0125", "--weight-decay", "0.001", "--contrastive-loss", "infonce", "--temperature", "0.05", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate != 0.0125 {
		t.Fatalf("learning rate = %f, want 0.0125", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.WeightDecay != 0.001 {
		t.Fatalf("weight decay = %f, want 0.001", checkpoint.Config.WeightDecay)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
}

func TestRunInitModelCreatesDefaultEmbeddingTrainingPackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Name != "eos-embed-v1" {
		t.Fatalf("model name = %q, want eos-embed-v1", manifest.Name)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload initialized model package: %v", err)
	}
}

func TestRunInitModelHonorsEncoderRepeats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--encoder-repeats", "3",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 3 {
		t.Fatalf("encoder repeats = %d, want 3", manifest.EncoderRepeats)
	}
}

func TestRunInitModelHonorsModelDimAliasAndDefaultsOutputDim(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--model-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.ModelDim != 4 || manifest.OutputDim != 4 {
		t.Fatalf("manifest dims = model:%d output:%d, want 4/4", manifest.ModelDim, manifest.OutputDim)
	}
}

func TestRunInitModelCreatesCompactBootstrapPackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.mll")
	if err := run([]string{
		"init-model",
		"--architecture", eosruntime.EmbeddingArchitectureCompactTransformerV1,
		"--model-dim", "8",
		"--output-dim", "4",
		"--hidden-dim", "16",
		"--attention-heads", "1",
		"--encoder-repeats", "4",
		path,
	}); err != nil {
		t.Fatalf("run init-model compact: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.ArchitectureVersion != eosruntime.EmbeddingArchitectureCompactTransformerV1 ||
		manifest.ParameterTying != eosruntime.EmbeddingParameterTyingUntied ||
		manifest.ModelDim != 8 ||
		manifest.OutputDim != 4 ||
		manifest.FFNDim != 16 ||
		manifest.AttentionHeads != 1 ||
		manifest.HeadDim != 8 ||
		manifest.EncoderRepeats != 4 ||
		manifest.OutputProjectionParam != "output_projection" {
		t.Fatalf("unexpected compact manifest: %+v", manifest)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Tensors["layer3_attn_q"] == nil || checkpoint.MomentTensors["layer3_attn_q_moment_1"] == nil {
		t.Fatalf("checkpoint missing layer3 generic tensors: tensors=%v moments=%v", checkpoint.Tensors, checkpoint.MomentTensors)
	}
	trainer, err := eosruntime.LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load compact trainer package: %v", err)
	}
	metrics, err := trainer.EvaluatePairs([]eosruntime.EmbeddingPairExample{
		{LeftTokens: []int32{1, 4, 5}, RightTokens: []int32{1, 4, 5}, Target: 1},
		{LeftTokens: []int32{1, 4, 5}, RightTokens: []int32{5, 4, 1}, Target: -1},
	})
	if err != nil {
		t.Fatalf("evaluate compact trainer package: %v", err)
	}
	if math.IsNaN(float64(metrics.Loss)) || math.IsInf(float64(metrics.Loss), 0) || metrics.PairCount != 2 {
		t.Fatalf("compact trainer metrics = %+v, want finite 2-pair evaluation", metrics)
	}
}

func TestRunInitModelCreatesCompactMultiHeadServingGraph(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-multihead.mll")
	if err := run([]string{
		"init-model",
		"--architecture", eosruntime.EmbeddingArchitectureCompactTransformerV1,
		"--model-dim", "8",
		"--output-dim", "4",
		"--hidden-dim", "16",
		"--attention-heads", "2",
		path,
	}); err != nil {
		t.Fatalf("run init-model compact multi-head: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read compact multi-head manifest: %v", err)
	}
	if manifest.AttentionHeads != 2 || manifest.HeadDim != 4 {
		t.Fatalf("compact multi-head manifest heads/head_dim = %d/%d, want 2/4", manifest.AttentionHeads, manifest.HeadDim)
	}
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read compact multi-head artifact: %v", err)
	}
	for _, kernel := range mod.Kernels {
		for _, op := range kernel.Body {
			if op.Op == "compact_multihead_attention" && op.Attributes["num_attention_heads"] == "2" {
				return
			}
		}
	}
	t.Fatal("compact multi-head artifact missing compact_multihead_attention num_attention_heads=2")
}

func TestRunInitModelRejectsInvalidHeadConfig(t *testing.T) {
	err := run([]string{
		"init-model",
		"--architecture", eosruntime.EmbeddingArchitectureCompactTransformerV1,
		"--model-dim", "7",
		"--output-dim", "7",
		"--hidden-dim", "14",
		"--attention-heads", "2",
		filepath.Join(t.TempDir(), "bad-heads.mll"),
	})
	if err == nil || !strings.Contains(err.Error(), "model_dim 7 must be divisible by attention_heads 2") {
		t.Fatalf("run init-model error = %v, want head divisibility error", err)
	}
}

func TestRunInitModelBootstrapFromCopiesOverlap(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "source.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "2",
		"--hidden-dim", "4",
		"--seed", "7",
		sourcePath,
	}); err != nil {
		t.Fatalf("run source init-model: %v", err)
	}
	sourceCheckpointPath := eosruntime.DefaultEmbeddingCheckpointPath(sourcePath)
	sourceCheckpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(sourceCheckpointPath)
	if err != nil {
		t.Fatalf("read source checkpoint: %v", err)
	}
	sourceCheckpoint.TokenEmbedding = backend.NewTensorF32([]int{8, 2}, []float32{
		31, 32,
		33, 34,
		35, 36,
		37, 38,
		39, 40,
		41, 42,
		43, 44,
		45, 46,
	})
	if err := sourceCheckpoint.WriteFile(sourceCheckpointPath); err != nil {
		t.Fatalf("rewrite source checkpoint: %v", err)
	}

	targetPath := filepath.Join(dir, "target.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "3",
		"--hidden-dim", "5",
		"--seed", "11",
		"--bootstrap-from", sourcePath,
		targetPath,
	}); err != nil {
		t.Fatalf("run target init-model: %v", err)
	}
	targetCheckpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(targetPath))
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	if got := targetCheckpoint.TokenEmbedding.Shape; len(got) != 2 || got[0] != 8 || got[1] != 3 {
		t.Fatalf("target token shape = %v, want [8 3]", got)
	}
	for row := 0; row < 8; row++ {
		for col := 0; col < 2; col++ {
			got := targetCheckpoint.TokenEmbedding.F32[row*3+col]
			want := sourceCheckpoint.TokenEmbedding.F32[row*2+col]
			if got != want {
				t.Fatalf("token overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(targetPath); err != nil {
		t.Fatalf("reload bootstrapped package: %v", err)
	}
}

func TestRunInitModelHonorsWeightDType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--weight-dtype", "q4",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil || param.Type.Tensor.DType != "q4" {
			t.Fatalf("param %q dtype = %+v, want q4 tensor", param.Name, param.Type)
		}
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.WeightBits != 4 {
		t.Fatalf("weight bits = %d, want 4", checkpoint.Config.WeightBits)
	}
	if err := run([]string{
		"init-model",
		"--weight-dtype", "int4",
		filepath.Join(t.TempDir(), "bad.mll"),
	}); err == nil {
		t.Fatal("expected weight dtype error")
	}
}

func TestRunInitMirageCreatesArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "mirage-v1.mll")
	output := captureRunOutput(t, []string{
		"init-mirage",
		"--height", "16",
		"--width", "16",
		"--latent-channels", "8",
		"--bits", "2",
		path,
	})
	for _, want := range []string{
		"initialized Mirage Image v1 module",
		"capabilities: image_ops, turboquant, training_losses, host_fallback",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("init-mirage output missing %q\noutput:\n%s", want, output)
		}
	}
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if mod.Name != "mirage_image_v1" || len(mod.EntryPoints) != 4 {
		t.Fatalf("unexpected Mirage artifact: %+v", mod)
	}
}

func TestRunInitModelTrainCorpusExportFlow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		"--seed", "7",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "16", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "2", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained default package: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	sealedPath := eosruntime.DefaultMLLPath(path)
	if sealedPath == path {
		t.Fatalf("sealed export path reused artifact path %q", path)
	}
	if _, err := mll.ReadFile(sealedPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read sealed default model MLL: %v", err)
	}
	sealedInspect := captureRunOutput(t, []string{"inspect", sealedPath})
	for _, want := range []string{
		"embedding manifest: embedded",
		"package: embedded sealed MLL",
		"package verify: OK",
		"embedding model: eos-embed-v1",
	} {
		if !strings.Contains(sealedInspect, want) {
			t.Fatalf("sealed inspect output missing %q\noutput:\n%s", want, sealedInspect)
		}
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("inspect trained default package after export: %v", err)
	}
}

func TestRunEmbedTextLoadsSealedMLLTokenizer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "eos-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}

	output := captureRunOutput(t, []string{"embed-text", sealedPath, "a"})
	for _, want := range []string{
		"loaded embedding \"eos-embed-v1\"",
		"tokens: 1",
		"output: result",
		"embedding: f16[4]",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("embed-text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunEvalRetrievalWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "eos-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"a"}`+"\n"+
			`{"_id":"d2","text":"b"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"a"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	metricsPath := filepath.Join(dir, "retrieval.metrics.json")

	output := captureRunOutput(t, []string{"eval-retrieval", "--dataset", "tiny", "--batch-size", "2", "--metrics-json", metricsPath, sealedPath, datasetDir})
	for _, want := range []string{
		"retrieval eval: dataset=tiny",
		"quality: ndcg@10=",
		"recall@100=",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics struct {
		Schema  string `json:"schema"`
		Dataset string `json:"dataset"`
		Inputs  struct {
			Documents int `json:"documents"`
			Queries   int `json:"queries"`
		} `json:"inputs"`
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Inputs.Documents != 2 || metrics.Inputs.Queries != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRunExportRetrievalVectorsWritesChildCaches(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "16",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "one", "two", "three", "four", "five", "six", "seven", "eight", "query"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "eos-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"one two three four five six seven eight"}`+"\n"+
			`{"_id":"d2","text":"one two"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"one query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputDir := filepath.Join(dir, "vector-cache")
	manifestPath := filepath.Join(dir, "vector-cache.manifest.json")

	output := captureRunOutput(t, []string{
		"export-retrieval-vectors",
		"--dataset", "tiny-export",
		"--batch-size", "1",
		"--output-dim", "2",
		"--document-chunk-words", "4",
		"--document-chunk-overlap", "1",
		"--document-chunk-min-words", "2",
		"--manifest-json", manifestPath,
		sealedPath,
		datasetDir,
		outputDir,
	})
	childPath := filepath.Join(outputDir, "child-doc-vectors.jsonl")
	queryPath := filepath.Join(outputDir, "query-vectors.jsonl")
	for _, want := range []string{
		"exported retrieval vectors: dataset=tiny-export",
		"child_vectors=4",
		"dim=2",
		"model_dim: 4",
		"child_doc_vectors: " + childPath,
		"query_vectors: " + queryPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("export output missing %q\noutput:\n%s", want, output)
		}
	}
	childData, err := os.ReadFile(childPath)
	if err != nil {
		t.Fatalf("read child vectors: %v", err)
	}
	if !strings.Contains(string(childData), `"parent_id":"d1"`) || !strings.Contains(string(childData), `"child_id":"d1#chunk-0001"`) || !strings.Contains(string(childData), `"embedding"`) {
		t.Fatalf("unexpected child vector rows:\n%s", string(childData))
	}
	queryData, err := os.ReadFile(queryPath)
	if err != nil {
		t.Fatalf("read query vectors: %v", err)
	}
	if !strings.Contains(string(queryData), `"id":"q1"`) || !strings.Contains(string(queryData), `"embedding"`) {
		t.Fatalf("unexpected query vector rows:\n%s", string(queryData))
	}
	var manifest eosruntime.RetrievalVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != eosruntime.RetrievalVectorExportManifestSchema || manifest.Dataset != "tiny-export" || manifest.ChildVectors != 4 || manifest.QueryVectorPath != queryPath || manifest.Dimension != 2 || manifest.ModelDimension != 4 || manifest.OutputDimension != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestRunExportTimeSeriesVectorsWritesWindowCaches(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "24",
		"--max-seq", "16",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "series", "window", "sensor", "rising", "load", "values", "stats", "query", "temperature", "short", "s1", "s2", "q1"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "eos-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}

	seriesPath := filepath.Join(dir, "series.jsonl")
	queriesPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(seriesPath, []byte(
		`{"id":"s1","label":"load","values":[1,2,3,4,5]}`+"\n"+
			`{"_id":"s2","text":"short sensor","values":[8,9]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write series: %v", err)
	}
	if err := os.WriteFile(queriesPath, []byte(`{"id":"q1","text":"rising load query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	outputDir := filepath.Join(dir, "timeseries-cache")
	manifestPath := filepath.Join(dir, "timeseries-cache.manifest.json")

	output := captureRunOutput(t, []string{
		"export-timeseries-vectors",
		"--dataset", "tiny-series",
		"--batch-size", "1",
		"--output-dim", "2",
		"--window-size", "3",
		"--window-stride", "2",
		"--series-prefix", "series window: ",
		"--query-prefix", "query: ",
		"--manifest-json", manifestPath,
		sealedPath,
		seriesPath,
		queriesPath,
		outputDir,
	})
	childPath := filepath.Join(outputDir, "child-doc-vectors.jsonl")
	queryPath := filepath.Join(outputDir, "query-vectors.jsonl")
	corpusPath := filepath.Join(outputDir, "corpus.jsonl")
	beirQueriesPath := filepath.Join(outputDir, "queries.jsonl")
	for _, want := range []string{
		"exported time-series vectors: dataset=tiny-series",
		"series=2 queries=1 child_window_vectors=3 dim=2",
		"model_dim: 4",
		"windows: size=3 stride=2",
		"dataset_dir: " + outputDir,
		"corpus: " + corpusPath,
		"queries: " + beirQueriesPath,
		"child_doc_vectors: " + childPath,
		"query_vectors: " + queryPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("timeseries export output missing %q\noutput:\n%s", want, output)
		}
	}
	corpusData, err := os.ReadFile(corpusPath)
	if err != nil {
		t.Fatalf("read corpus helper: %v", err)
	}
	if !strings.Contains(string(corpusData), `"_id":"s1"`) || !strings.Contains(string(corpusData), `"text":"label: load\nseries_id: s1`) || !strings.Contains(string(corpusData), `points: count=5`) {
		t.Fatalf("unexpected corpus helper rows:\n%s", string(corpusData))
	}
	beirQueryData, err := os.ReadFile(beirQueriesPath)
	if err != nil {
		t.Fatalf("read BEIR query helper: %v", err)
	}
	if !strings.Contains(string(beirQueryData), `"_id":"q1"`) || !strings.Contains(string(beirQueryData), `"text":"rising load query"`) {
		t.Fatalf("unexpected BEIR query helper rows:\n%s", string(beirQueryData))
	}
	childData, err := os.ReadFile(childPath)
	if err != nil {
		t.Fatalf("read child vectors: %v", err)
	}
	if !strings.Contains(string(childData), `"parent_id":"s1"`) || !strings.Contains(string(childData), `"child_id":"s1#window-0001"`) || !strings.Contains(string(childData), `"embedding"`) {
		t.Fatalf("unexpected child vector rows:\n%s", string(childData))
	}
	queryData, err := os.ReadFile(queryPath)
	if err != nil {
		t.Fatalf("read query vectors: %v", err)
	}
	if !strings.Contains(string(queryData), `"id":"q1"`) || !strings.Contains(string(queryData), `"embedding"`) {
		t.Fatalf("unexpected query vector rows:\n%s", string(queryData))
	}
	var manifest eosruntime.TimeSeriesVectorExportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != eosruntime.TimeSeriesVectorExportManifestSchema || manifest.Dataset != "tiny-series" || manifest.ChildVectors != 3 || manifest.CorpusPath != corpusPath || manifest.BEIRQueriesPath != beirQueriesPath || manifest.QueryVectorPath != queryPath || manifest.Dimension != 2 || manifest.ModelDimension != 4 || manifest.OutputDimension != 2 || manifest.WindowSize != 3 || manifest.WindowStride != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}

	qrelsPath := filepath.Join(dir, "qrels.tsv")
	metricsPath := filepath.Join(dir, "timeseries-multivector.metrics.json")
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\ts1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	evalOutput := captureRunOutput(t, []string{
		"eval-retrieval-multivector-turboquant",
		"--dataset", "tiny-series",
		"--backend", "text-rendered-timeseries-windows",
		"--artifact", "tiny-sealed",
		"--bits", "8",
		"--doc-vectors", childPath,
		"--query-vectors", queryPath,
		"--qrels", qrelsPath,
		"--metrics-json", metricsPath,
		outputDir,
	})
	for _, want := range []string{
		"retrieval multivector turboquant: dataset=tiny-series backend=text-rendered-timeseries-windows parents=2 child_vectors=3",
		"dense-child: ndcg@10=",
		"q8: ndcg@10=",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(evalOutput, want) {
			t.Fatalf("timeseries eval output missing %q\noutput:\n%s", want, evalOutput)
		}
	}
}

func TestRunEvalRetrievalBM25WritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	metricsPath := filepath.Join(dir, "bm25.retrieval.metrics.json")

	output := captureRunOutput(t, []string{"eval-retrieval-bm25", "--dataset", "tiny", "--metrics-json", metricsPath, datasetDir})
	for _, want := range []string{
		"retrieval bm25: dataset=tiny backend=bm25",
		"quality: ndcg@10=1.000000",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval-bm25 output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "bm25" || metrics.Quality.NDCGAt10 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
}

func TestRunExportSparseLexicalLabelsWritesLabelsAndManifest(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha alpha finance"}`+"\n"+
			`{"_id":"d2","text":"beta medicine"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha finance"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	manifestPath := filepath.Join(dir, "manifest.json")

	output := captureRunOutput(t, []string{
		"export-sparse-lexical-labels",
		"--dataset", "tiny",
		"--split", "train",
		"--top-terms", "2",
		"--hash-bins", "16",
		"--manifest-json", manifestPath,
		datasetDir,
		labelsPath,
	})
	for _, want := range []string{
		"exported sparse lexical labels: dataset=tiny split=train docs=2 queries=1 top_terms=2 hash_bins=16",
		"density: doc_avg_nonzeros=",
		"truncation: document_records=0 document_terms_omitted=0 query_records=0 query_terms_omitted=0 exported_terms_exact=true",
		"oracle: queries=1 max_abs_score_delta=0 exact=true reconstruction_terms=unbounded_internal ndcg@10=1.000000 recall@100=1.000000",
		"labels: " + labelsPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("export-sparse-lexical-labels output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(labelsPath)
	if err != nil {
		t.Fatalf("read labels: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 {
		t.Fatalf("label lines = %d, want 3\n%s", len(lines), data)
	}
	var first struct {
		Schema     string `json:"schema"`
		RecordType string `json:"record_type"`
		ID         string `json:"id"`
		NonZeros   int    `json:"nonzeros"`
		Terms      []struct {
			Term    string  `json:"term"`
			Weight  float64 `json:"weight"`
			HashBin *uint32 `json:"hash_bin"`
		} `json:"terms"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("decode first label: %v", err)
	}
	if first.Schema != eosruntime.SparseLexicalLabelsSchema || first.RecordType != "document" || first.ID != "d1" || first.NonZeros > 2 {
		t.Fatalf("first label = %+v", first)
	}
	if len(first.Terms) == 0 || first.Terms[0].HashBin == nil || first.Terms[0].Weight <= 0 {
		t.Fatalf("first label terms = %+v", first.Terms)
	}
	var manifest eosruntime.SparseLexicalLabelExportSummary
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != eosruntime.SparseLexicalLabelsSchema || manifest.Dataset != "tiny" || manifest.Config.TopTerms != 2 || manifest.Hashing.Bins != 16 {
		t.Fatalf("manifest = %+v", manifest)
	}
	if !manifest.Oracle.ExactScoreReconstruction || manifest.Stats.DocumentMaxNNZ > 2 || manifest.Stats.QueryMaxNNZ > 2 {
		t.Fatalf("manifest oracle/stats = %+v %+v", manifest.Oracle, manifest.Stats)
	}
	if manifest.Oracle.ReconstructionTerms != "unbounded_internal" || !manifest.Oracle.ExportedTermsExact {
		t.Fatalf("manifest oracle scope = %+v", manifest.Oracle)
	}
}

func TestRunExportSparseLexicalLabelsRejectsInvalidArgs(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(`{"_id":"d1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "oracle top k below recall depth",
			args: []string{"export-sparse-lexical-labels", "--oracle-top-k", "10", datasetDir, labelsPath},
			want: "oracle top-k must be at least 100",
		},
		{
			name: "same labels and manifest path",
			args: []string{"export-sparse-lexical-labels", "--manifest-json", filepath.Join(dir, ".", "labels.jsonl"), datasetDir, labelsPath},
			want: "labels output path and manifest path must differ",
		},
	}
	if math.MaxInt > math.MaxUint32 {
		tests = append(tests, struct {
			name string
			args []string
			want string
		}{
			name: "oversized hash bins",
			args: []string{"export-sparse-lexical-labels", "--hash-bins", "4294967296", datasetDir, labelsPath},
			want: "hash bins must be <=",
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := captureRunOutputAndError(t, tt.args)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, output = %q, want err containing %q", err, output, tt.want)
			}
		})
	}
}

func TestRunEvalSparseLexicalLabelsWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"alpha"}`+"\n"+
			`{"_id":"q2","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":2}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"alpha","weight":0.1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q2","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	metricsPath := filepath.Join(dir, "sparse-labels.retrieval.metrics.json")

	output := captureRunOutput(t, []string{
		"eval-sparse-lexical-labels",
		"--dataset", "tiny",
		"--split", "train",
		"--labels", labelsPath,
		"--metrics-json", metricsPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval sparse lexical labels: dataset=tiny backend=sparse_lexical_labels_capped docs=3 queries=2",
		"labels: documents=3 queries=2",
		"representation=capped_exported_sparse_lexical_labels",
		"quality: ndcg@10=1.000000",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-sparse-lexical-labels output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "sparse_lexical_labels_capped" || metrics.Inputs.LabelPath != labelsPath || metrics.Quality.NDCGAt10 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
	if metrics.SparseLexical == nil || metrics.SparseLexical.DocumentLabels != 3 || metrics.SparseLexical.QueryLabels != 2 {
		t.Fatalf("sparse lexical stats = %+v", metrics.SparseLexical)
	}
}

func TestRunEvalSparseLexicalLabelsRejectsMissingLabels(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}

	err := run([]string{"eval-sparse-lexical-labels", "--dataset", "tiny", "--split", "train", "--labels", labelsPath, datasetDir})
	if err == nil || !strings.Contains(err.Error(), "missing required qrels coverage") {
		t.Fatalf("err = %v, want missing required qrels coverage", err)
	}
}

func TestRunEvalSparseLexicalLabelsRejectsSmallTopK(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	labelsPath := filepath.Join(dir, "labels.jsonl")

	output, err := captureRunOutputAndError(t, []string{
		"eval-sparse-lexical-labels",
		"--labels", labelsPath,
		"--top-k", "99",
		datasetDir,
	})
	if err == nil || !strings.Contains(err.Error(), "top-k must be at least 100") {
		t.Fatalf("err = %v, output = %q, want top-k minimum error", err, output)
	}
}

func TestRunSparseLexicalHashHeadWritesHeadAndMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"alpha"}`+"\n"+
			`{"_id":"q2","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":2,"terms":[{"term":"alpha","weight":2},{"term":"finance","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":1,"terms":[{"term":"alpha","weight":0.1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q2","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "head.json")
	fitOutput := captureRunOutput(t, []string{
		"fit-sparse-lexical-head",
		"--dataset", "tiny",
		"--split", "train",
		"--labels", labelsPath,
		"--hash-bins", "65536",
		"--head-json", headPath,
	})
	for _, want := range []string{
		"fit sparse lexical hash head: schema=manta.sparse_lexical_hash_head.v1 experimental=true dataset=tiny split=train hash_bins=65536",
		"labels: documents=3 queries=2",
		"head: " + headPath,
	} {
		if !strings.Contains(fitOutput, want) {
			t.Fatalf("fit-sparse-lexical-head output missing %q\noutput:\n%s", want, fitOutput)
		}
	}
	var head eosruntime.SparseLexicalHashHead
	headData, err := os.ReadFile(headPath)
	if err != nil {
		t.Fatalf("read head: %v", err)
	}
	if err := json.Unmarshal(headData, &head); err != nil {
		t.Fatalf("decode head: %v", err)
	}
	if head.Schema != eosruntime.SparseLexicalHashHeadSchema || !head.Experimental || head.Hashing.Bins != 65536 {
		t.Fatalf("head = %+v", head)
	}
	metricsPath := filepath.Join(dir, "hash-head.metrics.json")
	evalOutput := captureRunOutput(t, []string{
		"eval-sparse-lexical-head",
		"--dataset", "tiny",
		"--split", "train",
		"--labels", labelsPath,
		"--head-json", headPath,
		"--metrics-json", metricsPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval sparse lexical hash head: dataset=tiny backend=sparse_lexical_hash_head docs=3 queries=2",
		"head: hash_bins=65536 documents=3 queries=2",
		"representation=experimental_hashed_sparse_lexical_head",
		"quality: ndcg@10=1.000000",
		"metrics: " + metricsPath,
	} {
		if !strings.Contains(evalOutput, want) {
			t.Fatalf("eval-sparse-lexical-head output missing %q\noutput:\n%s", want, evalOutput)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Backend != "sparse_lexical_hash_head" || metrics.Inputs.HeadPath != headPath || metrics.Quality.NDCGAt10 != 1 {
		t.Fatalf("metrics = %+v", metrics)
	}
	if metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 || metrics.SparseLexical.DocumentLabels != 3 {
		t.Fatalf("sparse lexical stats = %+v", metrics.SparseLexical)
	}
}

func TestRunSparseLexicalHashHeadRejectsInvalidArgs(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	labelsPath := filepath.Join(dir, "labels.jsonl")
	headPath := filepath.Join(dir, "head.json")
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "fit missing head",
			args: []string{"fit-sparse-lexical-head", "--labels", labelsPath},
			want: "head-json is required",
		},
		{
			name: "fit zero hash bins",
			args: []string{"fit-sparse-lexical-head", "--labels", labelsPath, "--head-json", headPath, "--hash-bins", "0"},
			want: "hash bins must be positive",
		},
		{
			name: "eval small top k",
			args: []string{"eval-sparse-lexical-head", "--labels", labelsPath, "--head-json", headPath, "--top-k", "99", datasetDir},
			want: "top-k must be at least 100",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := captureRunOutputAndError(t, tt.args)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, output = %q, want err containing %q", err, output, tt.want)
			}
		})
	}
}

func TestRunEvalRetrievalVectorsWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0]}`+"\n"+
			`{"_id":"d2","embedding":[0,1]}`+"\n"+
			`{"_id":"d3","embedding":[0.8,0.6]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"_id":"q1","embedding":[0.7,0.7]}`+"\n"+
			`{"_id":"q2","embedding":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	metricsPath := filepath.Join(dir, "vectors.retrieval.metrics.json")
	perQueryPath := filepath.Join(dir, "vectors.retrieval.per-query.jsonl")

	output := captureRunOutput(t, []string{
		"eval-retrieval-vectors",
		"--dataset", "tiny",
		"--backend", "qwen-cache",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval vectors: dataset=tiny backend=qwen-cache",
		"quality: ndcg@10=",
		"metrics: " + metricsPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval-vectors output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	wantNDCG := (1/math.Log2(3) + 1) / 2
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "qwen-cache" || metrics.Artifact != "qwen3-embedding" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if math.Abs(metrics.Quality.NDCGAt10-wantNDCG) > 1e-12 || metrics.Quality.MRRAt10 != 0.75 {
		t.Fatalf("quality = %+v, want ndcg %.12f mrr 0.75", metrics.Quality, wantNDCG)
	}
	if metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 2 || metrics.Inputs.ScoredPairs != 6 {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(perQueryData)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query lines = %d, want 2\n%s", len(lines), perQueryData)
	}
	var first eosruntime.RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("decode first per-query row: %v", err)
	}
	if first.Schema != eosruntime.RetrievalEvalPerQuerySchema || first.Dataset != "tiny" || first.QueryID != "q1" || first.FirstRelevantRank != 2 {
		t.Fatalf("first per-query row = %+v", first)
	}
}

func TestRunEvalRetrievalVectorsHybridWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"beta dense distractor"}`+"\n"+
			`{"_id":"d3","text":"gamma fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1]}`+"\n"+
			`{"_id":"d2","embedding":[1,0]}`+"\n"+
			`{"_id":"d3","embedding":[0.5,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"_id":"q1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	metricsPath := filepath.Join(dir, "vectors.hybrid.metrics.json")
	perQueryPath := filepath.Join(dir, "vectors.hybrid.per-query.jsonl")

	output := captureRunOutput(t, []string{
		"eval-retrieval-vectors-hybrid",
		"--dataset", "tiny",
		"--backend", "qwen-cache-hybrid",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--method", "minmax",
		"--alpha", "0.75",
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval vectors hybrid: dataset=tiny backend=qwen-cache-hybrid",
		"hybrid: method=minmax_blend alpha=0.75",
		"quality: ndcg@10=1.000000",
		"metrics: " + metricsPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval-vectors-hybrid output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "qwen-cache-hybrid" || metrics.Artifact != "qwen3-embedding" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || metrics.Config.Hybrid.Alpha != 0.75 {
		t.Fatalf("hybrid config = %+v", metrics.Config.Hybrid)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("quality = %+v, want perfect hybrid top hit", metrics.Quality)
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	var row eosruntime.RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(perQueryData))), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.FirstRelevantRank != 1 || len(row.TopK) == 0 || row.TopK[0].DocID != "d1" {
		t.Fatalf("per-query row = %+v", row)
	}
	if row.TopK[0].DenseRank == nil || *row.TopK[0].DenseRank != 3 || row.TopK[0].BM25Rank == nil || *row.TopK[0].BM25Rank != 1 {
		t.Fatalf("hybrid component ranks = dense:%v bm25:%v, want 3/1", row.TopK[0].DenseRank, row.TopK[0].BM25Rank)
	}
	if row.TopK[0].DenseScore == nil || row.TopK[0].BM25Score == nil || row.TopK[0].DenseNormalizedScore == nil || row.TopK[0].BM25NormalizedScore == nil {
		t.Fatalf("hybrid component scores missing from per-query top doc: %+v", row.TopK[0])
	}

	protectedMetricsPath := filepath.Join(dir, "vectors.hybrid.protected.metrics.json")
	protectedPerQueryPath := filepath.Join(dir, "vectors.hybrid.protected.per-query.jsonl")
	protectedOutput := captureRunOutput(t, []string{
		"eval-retrieval-vectors-hybrid",
		"--dataset", "tiny",
		"--backend", "qwen-cache-hybrid",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--method", "minmax",
		"--alpha", "0.75",
		"--dense-protect-top-k", "1",
		"--metrics-json", protectedMetricsPath,
		"--per-query-jsonl", protectedPerQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval vectors hybrid: dataset=tiny backend=qwen-cache-hybrid",
		"dense_protect_top_k=1",
		"metrics: " + protectedMetricsPath,
		"per_query: " + protectedPerQueryPath,
	} {
		if !strings.Contains(protectedOutput, want) {
			t.Fatalf("protected eval-retrieval-vectors-hybrid output missing %q\noutput:\n%s", want, protectedOutput)
		}
	}
	data, err = os.ReadFile(protectedMetricsPath)
	if err != nil {
		t.Fatalf("read protected metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode protected metrics: %v", err)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.DenseProtectTopK != 1 {
		t.Fatalf("protected hybrid config = %+v", metrics.Config.Hybrid)
	}
	perQueryData, err = os.ReadFile(protectedPerQueryPath)
	if err != nil {
		t.Fatalf("read protected per-query JSONL: %v", err)
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(perQueryData))), &row); err != nil {
		t.Fatalf("decode protected per-query row: %v", err)
	}
	if row.FirstRelevantRank != 2 || len(row.TopK) < 2 || row.TopK[0].DocID != "d2" || row.TopK[1].DocID != "d1" {
		t.Fatalf("protected per-query row = %+v", row)
	}
	if row.TopK[0].DenseRank == nil || *row.TopK[0].DenseRank != 1 || row.TopK[0].BM25Rank == nil || row.TopK[0].DenseScore == nil || row.TopK[0].BM25Score == nil {
		t.Fatalf("protected hybrid component evidence missing from per-query top doc: %+v", row.TopK[0])
	}
}

func TestRunEvalSparseLexicalHeadVectorsHybridWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"beta dense distractor"}`+"\n"+
			`{"_id":"d3","text":"gamma fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1]}`+"\n"+
			`{"_id":"d2","embedding":[1,0]}`+"\n"+
			`{"_id":"d3","embedding":[0.5,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(`{"_id":"q1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d2","nonzeros":1,"terms":[{"term":"beta","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"test","id":"d3","nonzeros":1,"terms":[{"term":"gamma","weight":1}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"test","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "head.json")
	captureRunOutput(t, []string{
		"fit-sparse-lexical-head",
		"--dataset", "tiny",
		"--split", "test",
		"--labels", labelsPath,
		"--hash-bins", "65536",
		"--head-json", headPath,
	})
	metricsPath := filepath.Join(dir, "sparse-head.vectors.hybrid.metrics.json")
	perQueryPath := filepath.Join(dir, "sparse-head.vectors.hybrid.per-query.jsonl")
	output := captureRunOutput(t, []string{
		"eval-sparse-lexical-head-vectors-hybrid",
		"--dataset", "tiny",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--labels", labelsPath,
		"--head-json", headPath,
		"--method", "minmax",
		"--alpha", "0.75",
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval sparse lexical hash head vectors hybrid: dataset=tiny backend=sparse_lexical_hash_head_vectors_hybrid",
		"hybrid: method=minmax_blend alpha=0.75",
		"sparse_hash_head: hash_bins=65536",
		"quality: ndcg@10=1.000000",
		"labels: " + labelsPath,
		"head: " + headPath,
		"metrics: " + metricsPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-sparse-lexical-head-vectors-hybrid output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Backend != "sparse_lexical_hash_head_vectors_hybrid" || metrics.Artifact != "qwen3-embedding" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath || metrics.Inputs.LabelPath != labelsPath || metrics.Inputs.HeadPath != headPath {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 {
		t.Fatalf("hybrid/sparse config = hybrid:%+v sparse:%+v", metrics.Config.Hybrid, metrics.SparseLexical)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("quality = %+v, want recovered top hit", metrics.Quality)
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	var row eosruntime.RetrievalEvalPerQueryRow
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(perQueryData))), &row); err != nil {
		t.Fatalf("decode per-query row: %v", err)
	}
	if row.FirstRelevantRank != 1 || len(row.TopK) == 0 || row.TopK[0].DocID != "d1" {
		t.Fatalf("per-query row = %+v", row)
	}
	if row.TopK[0].DenseRank == nil || *row.TopK[0].DenseRank != 3 || row.TopK[0].BM25Rank == nil || *row.TopK[0].BM25Rank != 1 {
		t.Fatalf("component ranks = dense:%v sparse:%v, want 3/1", row.TopK[0].DenseRank, row.TopK[0].BM25Rank)
	}
}

func TestRunSparseLexicalProjectionHeadWritesMetricsJSON(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"dense distractor"}`+"\n"+
			`{"_id":"d3","text":"fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	fitQueryVectorsPath := filepath.Join(dir, "fit-query-vectors.jsonl")
	evalQueryVectorsPath := filepath.Join(dir, "eval-query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1,0]}`+"\n"+
			`{"_id":"d2","embedding":[1,0.1,0]}`+"\n"+
			`{"_id":"d3","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(fitQueryVectorsPath, []byte(`{"_id":"q1","embedding":[0,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write fit query vectors: %v", err)
	}
	if err := os.WriteFile(evalQueryVectorsPath, []byte(`{"_id":"q1","embedding":[1,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write eval query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "projection-head.json")
	fitOutput := captureRunOutput(t, []string{
		"fit-sparse-lexical-projection-head",
		"--dataset", "tiny",
		"--split", "train",
		"--labels", labelsPath,
		"--doc-vectors", docVectorsPath,
		"--query-vectors", fitQueryVectorsPath,
		"--hash-bins", "65536",
		"--max-prototypes", "8",
		"--max-terms", "4",
		"--head-json", headPath,
	})
	for _, want := range []string{
		"fit sparse lexical projection head: schema=manta.sparse_lexical_projection_head.v1 experimental=true dataset=tiny split=train",
		"dim=3 hash_bins=65536 prototypes=1 max_terms=4",
		"prototype_rank=support",
		"labels: " + labelsPath,
		"doc_vectors: " + docVectorsPath,
		"query_vectors: " + fitQueryVectorsPath,
		"head: " + headPath,
	} {
		if !strings.Contains(fitOutput, want) {
			t.Fatalf("fit-sparse-lexical-projection-head output missing %q\noutput:\n%s", want, fitOutput)
		}
	}
	var head eosruntime.SparseLexicalProjectionHead
	headData, err := os.ReadFile(headPath)
	if err != nil {
		t.Fatalf("read projection head: %v", err)
	}
	if err := json.Unmarshal(headData, &head); err != nil {
		t.Fatalf("decode projection head: %v", err)
	}
	if head.Config.PrototypeRank != eosruntime.SparseLexicalProjectionPrototypeRankSupport {
		t.Fatalf("prototype_rank = %q, want support", head.Config.PrototypeRank)
	}
	if _, err := captureRunOutputAndError(t, []string{
		"fit-sparse-lexical-projection-head",
		"--labels", labelsPath,
		"--doc-vectors", docVectorsPath,
		"--query-vectors", fitQueryVectorsPath,
		"--head-json", filepath.Join(dir, "bad-projection-head.json"),
		"--prototype-rank", "score_magic",
	}); err == nil || !strings.Contains(err.Error(), "prototype rank must be one of support, total_weight, avg_weight") {
		t.Fatalf("invalid prototype rank err = %v", err)
	}

	metricsPath := filepath.Join(dir, "projection.metrics.json")
	perQueryPath := filepath.Join(dir, "projection.per-query.jsonl")
	evalOutput := captureRunOutput(t, []string{
		"eval-sparse-lexical-projection-head-vectors-hybrid",
		"--dataset", "tiny",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", evalQueryVectorsPath,
		"--head-json", headPath,
		"--split", "test",
		"--method", "minmax",
		"--alpha", "0.75",
		"--dense-candidates-only",
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval sparse lexical projection head vectors hybrid: dataset=tiny backend=sparse_lexical_projection_head_vectors_hybrid",
		"hybrid: method=minmax_blend alpha=0.75",
		"dense_candidates_only=true",
		"sparse_projection_head: hash_bins=65536",
		"quality: ndcg@10=1.000000",
		"head: " + headPath,
		"metrics: " + metricsPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(evalOutput, want) {
			t.Fatalf("eval-sparse-lexical-projection-head-vectors-hybrid output missing %q\noutput:\n%s", want, evalOutput)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Backend != "sparse_lexical_projection_head_vectors_hybrid" || metrics.Artifact != "qwen3-embedding" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.LabelPath != "" || metrics.Inputs.HeadPath != headPath || metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != evalQueryVectorsPath {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || !metrics.Config.Hybrid.DenseCandidatesOnly || metrics.SparseLexical == nil || metrics.SparseLexical.HashBins != 65536 {
		t.Fatalf("hybrid/sparse config = hybrid:%+v sparse:%+v", metrics.Config.Hybrid, metrics.SparseLexical)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("quality = %+v, want recovered top hit", metrics.Quality)
	}
}

func TestRunSparseLexicalLinearHeadWritesMetricsJSONWithoutEvalLabels(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha exact target"}`+"\n"+
			`{"_id":"d2","text":"dense distractor"}`+"\n"+
			`{"_id":"d3","text":"fallback"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	fitQueryVectorsPath := filepath.Join(dir, "fit-query-vectors.jsonl")
	evalQueryVectorsPath := filepath.Join(dir, "eval-query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[0,1,0]}`+"\n"+
			`{"_id":"d2","embedding":[1,0.1,0]}`+"\n"+
			`{"_id":"d3","embedding":[1,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(fitQueryVectorsPath, []byte(`{"_id":"q1","embedding":[0,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write fit query vectors: %v", err)
	}
	if err := os.WriteFile(evalQueryVectorsPath, []byte(`{"_id":"q1","embedding":[1,1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write eval query vectors: %v", err)
	}
	labelsPath := filepath.Join(dir, "labels.jsonl")
	if err := os.WriteFile(labelsPath, []byte(
		`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d1","nonzeros":1,"terms":[{"term":"alpha","weight":3}]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d2","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"document","dataset":"tiny","split":"train","id":"d3","nonzeros":0,"terms":[]}`+"\n"+
			`{"schema":"manta.sparse_lexical_labels.v1","record_type":"query","dataset":"tiny","split":"train","id":"q1","nonzeros":1,"terms":[{"term":"alpha","weight":1}]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write labels: %v", err)
	}
	headPath := filepath.Join(dir, "linear-head.json")
	fitOutput := captureRunOutput(t, []string{
		"fit-sparse-lexical-linear-head",
		"--dataset", "tiny",
		"--split", "train",
		"--labels", labelsPath,
		"--doc-vectors", docVectorsPath,
		"--query-vectors", fitQueryVectorsPath,
		"--hash-bins", "65536",
		"--max-bins", "8",
		"--max-terms", "4",
		"--epochs", "12",
		"--learning-rate", "0.1",
		"--negative-ratio", "1",
		"--target-transform", "log1p",
		"--head-json", headPath,
	})
	for _, want := range []string{
		"fit sparse lexical linear head: schema=manta.sparse_lexical_linear_head.v1 experimental=true dataset=tiny split=train",
		"dim=3 hash_bins=65536 bins=1 max_terms=4",
		"bin_rank=support",
		"target_transform=log1p",
		"labels: " + labelsPath,
		"doc_vectors: " + docVectorsPath,
		"query_vectors: " + fitQueryVectorsPath,
		"head: " + headPath,
	} {
		if !strings.Contains(fitOutput, want) {
			t.Fatalf("fit-sparse-lexical-linear-head output missing %q\noutput:\n%s", want, fitOutput)
		}
	}
	var head eosruntime.SparseLexicalLinearHead
	headData, err := os.ReadFile(headPath)
	if err != nil {
		t.Fatalf("read linear head: %v", err)
	}
	if err := json.Unmarshal(headData, &head); err != nil {
		t.Fatalf("decode linear head: %v", err)
	}
	if head.Config.BinRank != eosruntime.SparseLexicalProjectionPrototypeRankSupport || head.Config.TargetTransform != eosruntime.SparseLexicalLinearHeadTargetTransformLog1p || len(head.Bins) != 1 || len(head.Bins[0].Weights) != 3 {
		t.Fatalf("linear head = %+v", head)
	}
	if _, err := captureRunOutputAndError(t, []string{
		"fit-sparse-lexical-linear-head",
		"--labels", labelsPath,
		"--doc-vectors", docVectorsPath,
		"--query-vectors", fitQueryVectorsPath,
		"--head-json", filepath.Join(dir, "bad-linear-head.json"),
		"--bin-rank", "score_magic",
	}); err == nil || !strings.Contains(err.Error(), "bin rank must be one of support, total_weight, avg_weight") {
		t.Fatalf("invalid bin rank err = %v", err)
	}
	if _, err := captureRunOutputAndError(t, []string{
		"fit-sparse-lexical-linear-head",
		"--labels", labelsPath,
		"--doc-vectors", docVectorsPath,
		"--query-vectors", fitQueryVectorsPath,
		"--head-json", filepath.Join(dir, "bad-transform-linear-head.json"),
		"--target-transform", "sqrt",
	}); err == nil || !strings.Contains(err.Error(), "target transform must be one of identity, log1p") {
		t.Fatalf("invalid target transform err = %v", err)
	}
	if err := os.Remove(labelsPath); err != nil {
		t.Fatalf("remove fit labels before eval: %v", err)
	}

	metricsPath := filepath.Join(dir, "linear.metrics.json")
	perQueryPath := filepath.Join(dir, "linear.per-query.jsonl")
	evalOutput := captureRunOutput(t, []string{
		"eval-sparse-lexical-linear-head-vectors-hybrid",
		"--dataset", "tiny",
		"--artifact", "qwen3-embedding",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", evalQueryVectorsPath,
		"--head-json", headPath,
		"--split", "test",
		"--method", "minmax",
		"--alpha", "0.75",
		"--doc-max-terms", "2",
		"--query-max-terms", "1",
		"--score-threshold", "0.000001",
		"--dense-candidates-only",
		"--metrics-json", metricsPath,
		"--per-query-jsonl", perQueryPath,
		datasetDir,
	})
	for _, want := range []string{
		"retrieval sparse lexical linear head vectors hybrid: dataset=tiny backend=sparse_lexical_linear_head_vectors_hybrid",
		"hybrid: method=minmax_blend alpha=0.75",
		"dense_candidates_only=true",
		"sparse_linear_head: hash_bins=65536",
		"predicted_doc_terms_max=2 predicted_query_terms_max=1 score_threshold=1e-06",
		"representation=experimental_sparse_lexical_linear_head",
		"quality: ndcg@10=1.000000",
		"head: " + headPath,
		"metrics: " + metricsPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(evalOutput, want) {
			t.Fatalf("eval-sparse-lexical-linear-head-vectors-hybrid output missing %q\noutput:\n%s", want, evalOutput)
		}
	}
	var metrics eosruntime.RetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema || metrics.Backend != "sparse_lexical_linear_head_vectors_hybrid" || metrics.Artifact != "qwen3-embedding" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.LabelPath != "" || metrics.Inputs.HeadPath != headPath || metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != evalQueryVectorsPath {
		t.Fatalf("input metrics = %+v", metrics.Inputs)
	}
	if metrics.Config.Hybrid == nil || metrics.Config.Hybrid.Method != "minmax_blend" || !metrics.Config.Hybrid.DenseCandidatesOnly || metrics.SparseLexical == nil || metrics.SparseLexical.Representation != "experimental_sparse_lexical_linear_head" {
		t.Fatalf("hybrid/sparse config = hybrid:%+v sparse:%+v", metrics.Config.Hybrid, metrics.SparseLexical)
	}
	if metrics.SparseLexical.DocumentMaxHashNNZ != 2 || metrics.SparseLexical.QueryMaxHashNNZ != 1 || metrics.SparseLexical.ScoreThreshold != 0.000001 {
		t.Fatalf("sparse calibration stats = %+v", metrics.SparseLexical)
	}
	if metrics.Quality.NDCGAt10 != 1 || metrics.Quality.MRRAt10 != 1 {
		t.Fatalf("quality = %+v, want recovered top hit", metrics.Quality)
	}
}

func TestRunEvalRetrievalVectorsTurboQuantWritesMetricsJSONAndTSV(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha"}`+"\n"+
			`{"_id":"d2","text":"beta"}`+"\n"+
			`{"_id":"d3","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"alpha query"}`+"\n"+
			`{"_id":"q2","text":"beta query"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "test.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	docVectorsPath := filepath.Join(dir, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(dir, "query-vectors.jsonl")
	if err := os.WriteFile(docVectorsPath, []byte(
		`{"_id":"d1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"d3","embedding":[0,0,1,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(
		`{"_id":"q1","embedding":[1,0,0,0,0,0,0,0]}`+"\n"+
			`{"_id":"q2","embedding":[0,1,0,0,0,0,0,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	metricsPath := filepath.Join(dir, "vectors.turboquant.metrics.json")
	metricsTSVPath := filepath.Join(dir, "vectors.turboquant.metrics.tsv")
	perQueryPath := filepath.Join(dir, "vectors.turboquant.per-query.jsonl")

	output := captureRunOutput(t, []string{
		"eval-retrieval-vectors-turboquant",
		"--dataset", "tiny",
		"--backend", "bge-cache",
		"--artifact", "bge-m3",
		"--doc-vectors", docVectorsPath,
		"--query-vectors", queryVectorsPath,
		"--bits", "8",
		"--quantizer-seed", "123",
		"--metrics-json", metricsPath,
		"--metrics-tsv", metricsTSVPath,
		"--per-query-jsonl", perQueryPath,
		"--per-query-top-k", "2",
		datasetDir,
	})
	for _, want := range []string{
		"retrieval vectors turboquant: dataset=tiny backend=bge-cache",
		"dense: ndcg@10=1.000000 ndcg@100=1.000000 map@10=1.000000 recall@100=1.000000",
		"q8: ndcg@10=",
		"metrics: " + metricsPath,
		"metrics_tsv: " + metricsTSVPath,
		"per_query: " + perQueryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-retrieval-vectors-turboquant output missing %q\noutput:\n%s", want, output)
		}
	}
	var metrics eosruntime.TurboQuantRetrievalEvalMetrics
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if metrics.Schema != eosruntime.TurboQuantRetrievalEvalMetricsSchema || metrics.Dataset != "tiny" || metrics.Backend != "bge-cache" || metrics.Artifact != "bge-m3" {
		t.Fatalf("metrics identity = %+v", metrics)
	}
	if metrics.Inputs.DocVectorPath != docVectorsPath || metrics.Inputs.QueryVectorPath != queryVectorsPath {
		t.Fatalf("vector paths = %+v", metrics.Inputs)
	}
	if metrics.Config.QuantizerSeed != 123 {
		t.Fatalf("quantizer seed metadata = config:%+v rows:%+v", metrics.Config, metrics.Rows)
	}
	if metrics.Dense.Quality.NDCGAt10 != 1 || len(metrics.Rows) != 1 || metrics.Rows[0].Bits != 8 {
		t.Fatalf("metrics = %+v", metrics)
	}
	tsv, err := os.ReadFile(metricsTSVPath)
	if err != nil {
		t.Fatalf("read metrics TSV: %v", err)
	}
	if !strings.Contains(string(tsv), "tiny\tquantized\t8\tturboquant_ip_b8") {
		t.Fatalf("metrics TSV missing q8 row:\n%s", string(tsv))
	}
	perQueryData, err := os.ReadFile(perQueryPath)
	if err != nil {
		t.Fatalf("read per-query JSONL: %v", err)
	}
	perQueryLines := strings.Split(strings.TrimSpace(string(perQueryData)), "\n")
	if len(perQueryLines) != 2 {
		t.Fatalf("per-query lines = %d, want one q8 row per query\n%s", len(perQueryLines), perQueryData)
	}
	for _, line := range perQueryLines {
		var row eosruntime.TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode per-query row: %v\n%s", err, line)
		}
		if row.Schema != eosruntime.TurboQuantRetrievalPerQuerySchema || row.Dataset != "tiny" || row.Method != "turboquant_ip_b8" || row.Bits != 8 || row.QuantizerSeed != 123 || len(row.TopK) == 0 {
			t.Fatalf("per-query row = %+v", row)
		}
	}
}

func TestRunMineRetrievalHardNegativesWritesTextJSONL(t *testing.T) {
	dir := t.TempDir()
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath := filepath.Join(dir, "hard-negatives.jsonl")

	output := captureRunOutput(t, []string{"mine-retrieval-hard-negatives", "--dataset", "tiny", "--negatives", "1", datasetDir, outputPath})
	for _, want := range []string{
		"mined retrieval hard negatives: dataset=tiny examples=1 positives=1 negatives=1 queries=1",
		"output: " + outputPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("mine-retrieval-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] != "alpha distractor" {
		t.Fatalf("examples = %+v", examples)
	}
	if len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v, want positive plus one negative", examples[0].TeacherScores)
	}
}

func writeMineRetrievalHardNegativesFixture(t *testing.T, dir string) (datasetDir, outputPath string) {
	t.Helper()
	datasetDir = filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath = filepath.Join(dir, "hard-negatives.jsonl")
	return datasetDir, outputPath
}

func TestRunMineRetrievalHardNegativesDefaultDFPruneThresholdAndMiningWorkersAreEchoed(t *testing.T) {
	dir := t.TempDir()
	datasetDir, outputPath := writeMineRetrievalHardNegativesFixture(t, dir)

	output := captureRunOutput(t, []string{"mine-retrieval-hard-negatives", "--dataset", "tiny", "--negatives", "1", datasetDir, outputPath})
	wantPrefix := fmt.Sprintf("config: df_prune_threshold=%.4f mining_workers=", eosruntime.DefaultBM25MiningDFPruneThreshold)
	if !strings.Contains(output, wantPrefix) {
		t.Fatalf("mine-retrieval-hard-negatives output missing default config echo %q\noutput:\n%s", wantPrefix, output)
	}
}

// writeMineRetrievalHardNegativesMultiQueryFixture is like
// writeMineRetrievalHardNegativesFixture but with three independent
// query/positive/negative topics instead of one, so that a
// "--mining-workers 3" request genuinely runs with 3 workers instead of
// being clamped down to len(queries): MineBM25TextHardNegatives clamps its
// effective worker count to the number of queries being mined (a worker
// with no query to process would be wasted), so a single-query fixture
// cannot distinguish "the --mining-workers flag reached the runtime" from
// "the flag was silently dropped and fell back to the (also-clamped-to-1)
// auto default." query1/d1/d2/d3 intentionally match
// writeMineRetrievalHardNegativesFixture exactly so tests asserting on that
// first query's mined example keep the same expectations.
func writeMineRetrievalHardNegativesMultiQueryFixture(t *testing.T, dir string) (datasetDir, outputPath string) {
	t.Helper()
	datasetDir = filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega unrelated"}`+"\n"+
			`{"_id":"d4","text":"beta target"}`+"\n"+
			`{"_id":"d5","text":"beta distractor"}`+"\n"+
			`{"_id":"d6","text":"gamma target"}`+"\n"+
			`{"_id":"d7","text":"gamma distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(
		`{"_id":"q1","text":"alpha"}`+"\n"+
			`{"_id":"q2","text":"beta"}`+"\n"+
			`{"_id":"q3","text":"gamma"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte(
		"query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td4\t1\nq3\td6\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath = filepath.Join(dir, "hard-negatives.jsonl")
	return datasetDir, outputPath
}

func TestRunMineRetrievalHardNegativesDFPruneThresholdAndMiningWorkersFlagsPlumb(t *testing.T) {
	dir := t.TempDir()
	datasetDir, outputPath := writeMineRetrievalHardNegativesMultiQueryFixture(t, dir)

	output := captureRunOutput(t, []string{
		"mine-retrieval-hard-negatives", "--dataset", "tiny", "--negatives", "1",
		"--df-prune-threshold", "0.5", "--mining-workers", "3",
		datasetDir, outputPath,
	})
	if !strings.Contains(output, "config: df_prune_threshold=0.5000 mining_workers=3") {
		t.Fatalf("mine-retrieval-hard-negatives output missing explicit df-prune-threshold/mining-workers config echo\noutput:\n%s", output)
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read hard negatives: %v", err)
	}
	if len(examples) != 3 || examples[0].Negatives[0] != "alpha distractor" {
		t.Fatalf("examples = %+v", examples)
	}
}

func TestRunMineRetrievalModelHardNegativesWritesTextJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "eos-embed-v1.mll")
	if err := run([]string{
		"init-model",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		path,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "alpha", "target", "distractor", "omega"},
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(path); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	sealedPath := filepath.Join(dir, "eos-embed-v1.sealed.mll")
	if err := run([]string{"export-mll", path, sealedPath}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	datasetDir := filepath.Join(dir, "dataset")
	if err := os.MkdirAll(filepath.Join(datasetDir, "qrels"), 0o755); err != nil {
		t.Fatalf("mkdir dataset: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "corpus.jsonl"), []byte(
		`{"_id":"d1","text":"alpha target"}`+"\n"+
			`{"_id":"d2","text":"alpha distractor"}`+"\n"+
			`{"_id":"d3","text":"omega distractor"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "queries.jsonl"), []byte(`{"_id":"q1","text":"alpha"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	if err := os.WriteFile(filepath.Join(datasetDir, "qrels", "train.tsv"), []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	outputPath := filepath.Join(dir, "model-hard-negatives.jsonl")

	output := captureRunOutput(t, []string{"mine-retrieval-model-hard-negatives", "--dataset", "tiny", "--negatives", "1", "--candidate-top-k", "2", "--batch-size", "2", "--role-mode", "raw", sealedPath, datasetDir, outputPath})
	for _, want := range []string{
		"mined model retrieval hard negatives: dataset=tiny",
		"role_mode=raw",
		"examples=1 positives=1 negatives=1 queries=1",
		"output: " + outputPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("mine-retrieval-model-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read model hard negatives: %v", err)
	}
	if len(examples) != 1 || examples[0].Query != "alpha" || examples[0].Positive != "alpha target" || len(examples[0].Negatives) != 1 || examples[0].Negatives[0] == "alpha target" {
		t.Fatalf("examples = %+v", examples)
	}
	if len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v, want positive plus one negative", examples[0].TeacherScores)
	}
}

func TestRunImportTeacherScoresWritesVectorsAndManifest(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := os.WriteFile(inputPath, []byte(`{"source":"scifact","query":"alpha","positive":"target","negatives":["distractor"],"request_only":true,"train_allowed":false}`+"\n"), 0o644); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	scoresPath := filepath.Join(dir, "scores.jsonl")
	if err := os.WriteFile(scoresPath, []byte(`{"source":"scifact","query":"alpha","scores":[0.9,0.1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")
	manifestPath := filepath.Join(dir, "teacher.manifest.json")

	output := captureRunOutput(t, []string{
		"import-teacher-scores",
		"--teacher-model-id", "teacher-a",
		"--teacher-revision", "rev1",
		"--score-scale", "cosine",
		"--manifest", manifestPath,
		inputPath,
		scoresPath,
		outputPath,
	})
	for _, want := range []string{
		"imported teacher scores: examples=1 updated=1",
		"teacher: model_id=teacher-a revision=rev1 score_scale=cosine",
		"output: " + outputPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("import-teacher-scores output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 2 || examples[0].TeacherScores[0] != 0.9 || examples[0].TeacherScores[1] != 0.1 {
		t.Fatalf("teacher scores = %+v", examples)
	}
	var outputRow map[string]any
	outputData, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output jsonl: %v", err)
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(outputData))), &outputRow); err != nil {
		t.Fatalf("decode output jsonl: %v\n%s", err, outputData)
	}
	if outputRow["request_only"] != true || outputRow["train_allowed"] != false {
		t.Fatalf("output metadata = %+v, want request_only=true train_allowed=false", outputRow)
	}
	var manifest teacherScoreImportSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_score_import.v1" || manifest.TeacherModelID != "teacher-a" || manifest.Updated != 1 || manifest.ExampleRows != 1 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestRunExportTeacherScoreRequestsRoundTripsThroughImport(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus", Query: "vitamin c", Positive: "ascorbic acid", Negatives: []string{"calcium", "zinc"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	requestPath := filepath.Join(dir, "teacher-requests.jsonl")
	manifestPath := filepath.Join(dir, "requests.manifest.json")

	output := captureRunOutput(t, []string{
		"export-teacher-score-requests",
		"--manifest", manifestPath,
		inputPath,
		requestPath,
	})
	for _, want := range []string{
		"exported teacher score requests: examples=1 exported=1 skipped_existing=0 rows=3 positive_rows=1 negative_rows=2",
		"output: " + requestPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("export-teacher-score-requests output missing %q\noutput:\n%s", want, output)
		}
	}
	data, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read requests: %v", err)
	}
	var requests []teacherScoreRequestRecord
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var record teacherScoreRequestRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			t.Fatalf("decode request %q: %v", line, err)
		}
		requests = append(requests, record)
	}
	if len(requests) != 3 || requests[0].Role != "positive" || requests[0].CandidateIndex != 0 || requests[1].Role != "negative" || requests[1].Candidate != "calcium" {
		t.Fatalf("requests = %+v", requests)
	}
	var manifest teacherScoreRequestSummary
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_score_requests.v1" || manifest.Rows != 3 || manifest.PositiveRows != 1 || manifest.NegativeRows != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}

	scorePath := filepath.Join(dir, "scores.jsonl")
	var scoreRows []string
	wantScores := []float32{0.8, 0.2, 0.1}
	for i, request := range requests {
		score := float64(wantScores[i])
		row, err := json.Marshal(teacherScoreImportRecord{
			Source:    request.Source,
			Query:     request.Query,
			Candidate: request.Candidate,
			Score:     &score,
		})
		if err != nil {
			t.Fatalf("encode score row: %v", err)
		}
		scoreRows = append(scoreRows, string(row))
	}
	if err := os.WriteFile(scorePath, []byte(strings.Join(scoreRows, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")
	_ = captureRunOutput(t, []string{"import-teacher-scores", inputPath, scorePath, outputPath})
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != len(wantScores) {
		t.Fatalf("examples = %+v", examples)
	}
	for i, want := range wantScores {
		if examples[0].TeacherScores[i] != want {
			t.Fatalf("teacher score %d = %f, want %f", i, examples[0].TeacherScores[i], want)
		}
	}
}

func TestRunExportTeacherScoreRequestsMissingOnly(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "q1", Positive: "p1", Negatives: []string{"n1"}, TeacherScores: []float32{0.8, 0.1}},
		{Source: "fiqa", Query: "q2", Positive: "p2", Negatives: []string{"n2"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	requestPath := filepath.Join(dir, "missing-requests.jsonl")

	output := captureRunOutput(t, []string{
		"export-teacher-score-requests",
		"--missing-only",
		inputPath,
		requestPath,
	})
	if !strings.Contains(output, "exported teacher score requests: examples=2 exported=1 skipped_existing=1 rows=2") {
		t.Fatalf("unexpected output:\n%s", output)
	}
	data, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read requests: %v", err)
	}
	if got := strings.Count(strings.TrimSpace(string(data)), "\n") + 1; got != 2 {
		t.Fatalf("request rows = %d, want 2\n%s", got, data)
	}
}

func TestRunImportTeacherScoresMatchesCandidateRows(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus", Query: "vitamin c", Positive: "ascorbic acid", Negatives: []string{"calcium", "zinc"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	scoresPath := filepath.Join(dir, "scores.jsonl")
	if err := os.WriteFile(scoresPath, []byte(
		`{"query":"vitamin c","candidate":"ascorbic acid","score":0.8}`+"\n"+
			`{"query":"vitamin c","candidate":"calcium","score":0.2}`+"\n"+
			`{"query":"vitamin c","candidate":"zinc","score":0.1}`+"\n"), 0o644); err != nil {
		t.Fatalf("write scores: %v", err)
	}
	outputPath := filepath.Join(dir, "with-teacher.jsonl")

	_ = captureRunOutput(t, []string{"import-teacher-scores", inputPath, scoresPath, outputPath})
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 3 {
		t.Fatalf("examples = %+v", examples)
	}
	want := []float32{0.8, 0.2, 0.1}
	for i, score := range want {
		if examples[0].TeacherScores[i] != score {
			t.Fatalf("teacher score %d = %f, want %f", i, examples[0].TeacherScores[i], score)
		}
	}
}

func TestRunScoreTeacherHardNegativesWritesScoresAndManifest(t *testing.T) {
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "teacher.mll")
	if err := run([]string{
		"init-model",
		"--name", "tiny-teacher",
		"--vocab-size", "8",
		"--max-seq", "8",
		"--embedding-dim", "4",
		"--hidden-dim", "8",
		artifactPath,
	}); err != nil {
		t.Fatalf("run init-model: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[UNK]", "a", "b", "c", "d", "e", "f"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if _, _, err := eosruntime.RebuildSiblingPackageManifest(artifactPath); err != nil {
		t.Fatalf("rebuild package manifest: %v", err)
	}
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "abc", Positive: "abc", Negatives: []string{"def"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	outputPath := filepath.Join(dir, "scored.jsonl")
	manifestPath := filepath.Join(dir, "teacher-score.manifest.json")

	output := captureRunOutput(t, []string{
		"score-teacher-hard-negatives",
		"--batch-size", "2",
		"--manifest", manifestPath,
		"--teacher-revision", "local",
		artifactPath,
		inputPath,
		outputPath,
	})
	for _, want := range []string{
		"scored teacher hard negatives: examples=1 updated=1",
		"teacher: model_id=tiny-teacher revision=local score_scale=cosine",
		"output: " + outputPath,
		"manifest: " + manifestPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("score-teacher-hard-negatives output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 2 {
		t.Fatalf("teacher scores = %+v", examples)
	}
	for i, score := range examples[0].TeacherScores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			t.Fatalf("teacher score %d is not finite: %f", i, score)
		}
	}
	var manifest teacherHardNegativeScoreSummary
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.Schema != "manta.teacher_hard_negative_score.v1" || manifest.TeacherModelID != "tiny-teacher" || manifest.Updated != 1 || manifest.BatchSize != 2 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestRunAuditTeacherScoresWritesSummary(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "scifact", Query: "q1", Positive: "p1", Negatives: []string{"n1", "n2"}, TeacherScores: []float32{0.9, 0.1, 0.2}},
		{Source: "fiqa", Query: "q2", Positive: "p2", Negatives: []string{"n3"}, TeacherScores: []float32{0.1, 0.8}},
		{Source: "fiqa", Query: "q3", Positive: "p3", Negatives: []string{"n4"}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	summaryPath := filepath.Join(dir, "teacher-audit.json")

	output := captureRunOutput(t, []string{
		"audit-teacher-scores",
		"--temperature", "1.5",
		inputPath,
		summaryPath,
	})
	for _, want := range []string{
		"audited teacher scores: examples=3 scored=2 missing=1",
		"positive_top1_rate=0.500000",
		"summary: " + summaryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("audit-teacher-scores output missing %q\noutput:\n%s", want, output)
		}
	}
	var summary teacherScoreAuditSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.Schema != "manta.teacher_score_audit.v1" || summary.Mode != "text" || summary.Examples != 3 || summary.ScoredExamples != 2 || summary.MissingExamples != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.Candidates != 7 || summary.ScoredCandidates != 5 || summary.PositiveTop1 != 1 {
		t.Fatalf("summary counts = %+v", summary)
	}
	if math.Abs(summary.PositiveTop1Rate-0.5) > 0.000001 || math.Abs(summary.PositiveMeanRank-1.5) > 0.000001 {
		t.Fatalf("summary ranks = %+v", summary)
	}
	if summary.MeanNormalizedEntropy <= 0 || summary.MeanNormalizedEntropy > 1 {
		t.Fatalf("summary normalized entropy = %f", summary.MeanNormalizedEntropy)
	}
	var rawSummary map[string]json.RawMessage
	if err := json.Unmarshal(data, &rawSummary); err != nil {
		t.Fatalf("decode raw summary: %v", err)
	}
	for _, key := range []string{"label_policy", "any_positive_top1", "duplicate_positive_negative_candidates"} {
		if _, ok := rawSummary[key]; ok {
			t.Fatalf("default audit summary unexpectedly included %q: %s", key, data)
		}
	}
	fiqa := summary.Sources["fiqa"]
	if fiqa.Examples != 2 || fiqa.ScoredExamples != 1 || fiqa.MissingExamples != 1 || fiqa.PositiveTop1 != 0 {
		t.Fatalf("fiqa source summary = %+v", fiqa)
	}
}

func TestRunAuditTeacherScoresQrelsCorpusCountsDuplicatePositiveTop1(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	inputRows := strings.Join([]string{
		`{"source":"nfcorpus","query":"query one","positive":"selected positive text","negatives":["duplicate relevant text","ordinary negative"],"teacher_scores":[0.2,0.9,0.1],"query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d2","d4"]}`,
	}, "\n") + "\n"
	if err := os.WriteFile(inputPath, []byte(inputRows), 0o644); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	qrelsPath := filepath.Join(dir, "qrels.tsv")
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	corpusRows := strings.Join([]string{
		`{"_id":"d1","title":"","text":"selected positive text"}`,
		`{"_id":"d2","title":"","text":"duplicate relevant text"}`,
		`{"_id":"d3","title":"","text":"duplicate relevant text"}`,
		`{"_id":"d4","title":"","text":"ordinary negative"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(corpusPath, []byte(corpusRows), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	summaryPath := filepath.Join(dir, "teacher-audit.json")

	captureRunOutput(t, []string{
		"audit-teacher-scores",
		"--qrels", qrelsPath,
		"--corpus", corpusPath,
		inputPath,
		summaryPath,
	})

	var summary teacherScoreAuditSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.LabelPolicy != "selected_positive_and_any_qrels_positive" {
		t.Fatalf("label policy = %q", summary.LabelPolicy)
	}
	if summary.PositiveTop1 != 0 || summary.AnyPositiveTop1 == nil || *summary.AnyPositiveTop1 != 1 {
		t.Fatalf("top1 counts selected=%d any=%v summary=%+v", summary.PositiveTop1, summary.AnyPositiveTop1, summary)
	}
	if summary.AnyPositiveTop1Rate == nil || math.Abs(*summary.AnyPositiveTop1Rate-1) > 0.000001 {
		t.Fatalf("any positive rate = %v", summary.AnyPositiveTop1Rate)
	}
	if summary.DuplicatePositiveNegativeCandidates == nil || *summary.DuplicatePositiveNegativeCandidates != 1 {
		t.Fatalf("duplicate positive negatives = %v", summary.DuplicatePositiveNegativeCandidates)
	}
	if summary.ExamplesWithDuplicatePositiveNegatives == nil || *summary.ExamplesWithDuplicatePositiveNegatives != 1 {
		t.Fatalf("examples with duplicate positive negatives = %v", summary.ExamplesWithDuplicatePositiveNegatives)
	}
	source := summary.Sources["nfcorpus"]
	if source.AnyPositiveTop1 == nil || *source.AnyPositiveTop1 != 1 || source.DuplicatePositiveNegativeCandidates == nil || *source.DuplicatePositiveNegativeCandidates != 1 {
		t.Fatalf("source summary = %+v", source)
	}
}

func TestRunAuditTeacherScoresQrelsCorpusCleanCaseMatchesSelectedPositive(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	inputRows := strings.Join([]string{
		`{"source":"nfcorpus","query":"query one","positive":"selected positive text","negatives":["ordinary negative","other negative"],"teacher_scores":[0.9,0.2,0.1],"query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d2","d4"]}`,
	}, "\n") + "\n"
	if err := os.WriteFile(inputPath, []byte(inputRows), 0o644); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	qrelsPath := filepath.Join(dir, "qrels.tsv")
	if err := os.WriteFile(qrelsPath, []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\n"), 0o644); err != nil {
		t.Fatalf("write qrels: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.jsonl")
	corpusRows := strings.Join([]string{
		`{"_id":"d1","title":"","text":"selected positive text"}`,
		`{"_id":"d2","title":"","text":"ordinary negative"}`,
		`{"_id":"d4","title":"","text":"other negative"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(corpusPath, []byte(corpusRows), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	summaryPath := filepath.Join(dir, "teacher-audit.json")

	captureRunOutput(t, []string{
		"audit-teacher-scores",
		"--qrels", qrelsPath,
		"--corpus", corpusPath,
		inputPath,
		summaryPath,
	})

	var summary teacherScoreAuditSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.PositiveTop1 != 1 || summary.AnyPositiveTop1 == nil || *summary.AnyPositiveTop1 != summary.PositiveTop1 {
		t.Fatalf("top1 counts selected=%d any=%v summary=%+v", summary.PositiveTop1, summary.AnyPositiveTop1, summary)
	}
	if summary.DuplicatePositiveNegativeCandidates == nil || *summary.DuplicatePositiveNegativeCandidates != 0 {
		t.Fatalf("duplicate positive negatives = %v", summary.DuplicatePositiveNegativeCandidates)
	}
	if summary.ExamplesWithDuplicatePositiveNegatives == nil || *summary.ExamplesWithDuplicatePositiveNegatives != 0 {
		t.Fatalf("examples with duplicate positive negatives = %v", summary.ExamplesWithDuplicatePositiveNegatives)
	}
	if summary.AnyPositiveMeanRank == nil || math.Abs(*summary.AnyPositiveMeanRank-summary.PositiveMeanRank) > 0.000001 {
		t.Fatalf("mean ranks selected=%f any=%v", summary.PositiveMeanRank, summary.AnyPositiveMeanRank)
	}
	if summary.AnyPositiveMeanMargin == nil || math.Abs(*summary.AnyPositiveMeanMargin-summary.PositiveMeanMargin) > 0.000001 {
		t.Fatalf("mean margins selected=%f any=%v", summary.PositiveMeanMargin, summary.AnyPositiveMeanMargin)
	}
}

func TestRunFilterTeacherScoresClearsUnsafeScores(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	inputRows := strings.Join([]string{
		`{"source":"scifact","query":"q1","positive":"p1","negatives":["n1","n2"],"teacher_scores":[0.9,0.1,0.2],"request_only":true,"train_allowed":false}`,
		`{"source":"fiqa","query":"q2","positive":"p2","negatives":["n3"],"teacher_scores":[0.1,0.8],"request_only":true,"train_allowed":false}`,
		`{"source":"fiqa","query":"q3","positive":"p3","negatives":["n4"],"request_only":true,"train_allowed":false}`,
	}, "\n") + "\n"
	if err := os.WriteFile(inputPath, []byte(inputRows), 0o644); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	outputPath := filepath.Join(dir, "filtered.jsonl")
	summaryPath := filepath.Join(dir, "filter-summary.json")

	output := captureRunOutput(t, []string{
		"filter-teacher-scores",
		inputPath,
		outputPath,
		summaryPath,
	})
	for _, want := range []string{
		"filtered teacher scores: examples=3 scored=2 missing=1 kept=1 cleared=1 dropped=0",
		"positive_top1_rate_before=0.500000",
		"kept_rate=0.500000",
		"output: " + outputPath,
		"summary: " + summaryPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("filter-teacher-scores output missing %q\noutput:\n%s", want, output)
		}
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 3 {
		t.Fatalf("examples = %d, want 3", len(examples))
	}
	if len(examples[0].TeacherScores) != 3 {
		t.Fatalf("passing teacher scores = %+v, want preserved", examples[0].TeacherScores)
	}
	if len(examples[1].TeacherScores) != 0 {
		t.Fatalf("failing teacher scores = %+v, want cleared", examples[1].TeacherScores)
	}
	if examples[1].Query != "q2" || examples[1].Positive != "p2" || len(examples[1].Negatives) != 1 {
		t.Fatalf("failing example fields were not preserved: %+v", examples[1])
	}
	if len(examples[2].TeacherScores) != 0 {
		t.Fatalf("missing teacher scores = %+v, want unchanged missing", examples[2].TeacherScores)
	}
	outputData, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output jsonl: %v", err)
	}
	for i, line := range strings.Split(strings.TrimSpace(string(outputData)), "\n") {
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode output row %d: %v\n%s", i, err, line)
		}
		if row["request_only"] != true || row["train_allowed"] != false {
			t.Fatalf("output row %d metadata = %+v, want request_only=true train_allowed=false", i, row)
		}
	}
	var summary teacherScoreFilterSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.Schema != "manta.teacher_score_filter.v1" || summary.Mode != "text" || summary.Examples != 3 || summary.Scored != 2 || summary.Missing != 1 || summary.KeptTeacherScores != 1 || summary.ClearedTeacherScores != 1 {
		t.Fatalf("summary = %+v", summary)
	}
	if summary.Candidates != 7 || math.Abs(summary.PositiveTop1RateBefore-0.5) > 0.000001 || math.Abs(summary.TeacherScoreKeptRate-0.5) > 0.000001 {
		t.Fatalf("summary rates/counts = %+v", summary)
	}
	fiqa := summary.Sources["fiqa"]
	if fiqa.Examples != 2 || fiqa.Scored != 1 || fiqa.Missing != 1 || fiqa.ClearedTeacherScores != 1 {
		t.Fatalf("fiqa source summary = %+v", fiqa)
	}
}

func TestRunFilterTeacherScoresMinMarginClearsMarginalTop1(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Source: "nfcorpus", Query: "q1", Positive: "p1", Negatives: []string{"n1"}, TeacherScores: []float32{0.51, 0.50}},
	}); err != nil {
		t.Fatalf("write hard negatives: %v", err)
	}
	outputPath := filepath.Join(dir, "filtered.jsonl")
	summaryPath := filepath.Join(dir, "filter-summary.json")

	_ = captureRunOutput(t, []string{
		"filter-teacher-scores",
		"--min-margin", "0.02",
		inputPath,
		outputPath,
		summaryPath,
	})
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 0 {
		t.Fatalf("examples = %+v, want preserved row with cleared teacher scores", examples)
	}
	var summary teacherScoreFilterSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.KeptTeacherScores != 0 || summary.ClearedTeacherScores != 1 || summary.PositiveTop1RateBefore != 1 || summary.PositiveTop1RateAfter != 0 {
		t.Fatalf("summary = %+v", summary)
	}
	if math.Abs(summary.MeanMarginBefore-0.01) > 0.000001 {
		t.Fatalf("mean margin before = %f, want 0.01", summary.MeanMarginBefore)
	}
}

func TestRunFilterTeacherScoresTokenizedMode(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "hard-negatives.tokens.jsonl")
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(inputPath, []eosruntime.EmbeddingHardNegativeExample{
		{
			Source:         "scifact",
			QueryTokens:    []int32{1},
			PositiveTokens: []int32{2},
			NegativeTokens: [][]int32{{3}},
			TeacherScores:  []float32{0.2, 0.8},
		},
	}); err != nil {
		t.Fatalf("write tokenized hard negatives: %v", err)
	}
	outputPath := filepath.Join(dir, "filtered.tokens.jsonl")
	summaryPath := filepath.Join(dir, "filter-summary.json")

	_ = captureRunOutput(t, []string{
		"filter-teacher-scores",
		"--mode", "tokenized",
		inputPath,
		outputPath,
		summaryPath,
	})
	examples, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read tokenized output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].TeacherScores) != 0 || len(examples[0].NegativeTokens) != 1 {
		t.Fatalf("tokenized examples = %+v, want preserved row with cleared teacher scores", examples)
	}
	var summary teacherScoreFilterSummary
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatalf("read summary: %v", err)
	}
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.Mode != "tokenized" || summary.ClearedTeacherScores != 1 || summary.Sources["scifact"].ClearedTeacherScores != 1 {
		t.Fatalf("summary = %+v", summary)
	}
}

func TestRunPlanSparseAttentionWritesBudgetReport(t *testing.T) {
	dir := t.TempDir()
	reportPath := filepath.Join(dir, "sparse-plan.json")
	output := captureRunOutput(t, []string{
		"plan-sparse-attention",
		"--key-lens", "64,256",
		"--query-dim", "16",
		"--value-dim", "32",
		"--top-k", "4",
		"--route-top-blocks", "2",
		"--bits", "4",
		"--require-subquadratic",
		"--max-score-fraction", "0.5",
		"--json", reportPath,
	})
	for _, want := range []string{
		"key_len\trouting",
		"64\tblock_anchor\t8\t2\t4\t16\t24\t0.375000",
		"256\tblock_anchor\t16\t2\t4\t32\t48\t0.187500",
		"gate=pass",
		"json: " + reportPath,
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-sparse-attention output missing %q\noutput:\n%s", want, output)
		}
	}
	var report sparseAttentionPlanReport
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if report.Schema != "manta.sparse_attention_plan.v1" || !report.Gate.Passed || report.Gate.SubquadraticRows != 2 {
		t.Fatalf("report gate = %+v schema=%q", report.Gate, report.Schema)
	}
	if len(report.Rows) != 2 {
		t.Fatalf("rows = %d", len(report.Rows))
	}
	first := report.Rows[0]
	if first.KeyLen != 64 || first.CandidateKeyBudget != 16 || first.EstimatedScoreCountPerQuery != 24 {
		t.Fatalf("first row = %+v", first)
	}
	if first.TurboQuantKVBytes != 2048 || math.Abs(first.TurboQuantCompressionRatio-3) > 0.000001 {
		t.Fatalf("first row TurboQuant memory = %+v", first)
	}
	if report.Gate.ScoreAlpha <= 0 || report.Gate.ScoreAlpha >= 1 {
		t.Fatalf("score alpha = %f, want sublinear", report.Gate.ScoreAlpha)
	}
}

func TestRunPlanSparseAttentionCanFailGate(t *testing.T) {
	output, err := captureRunOutputAndError(t, []string{
		"plan-sparse-attention",
		"--key-lens", "64",
		"--exact",
		"--require-subquadratic",
	})
	if err == nil {
		t.Fatalf("expected gate failure\noutput:\n%s", output)
	}
	if !strings.Contains(err.Error(), "not subquadratic") || !strings.Contains(output, "gate=fail") {
		t.Fatalf("unexpected failure err=%v output:\n%s", err, output)
	}
}

func TestRunCompareRetrievalMetricsCanRequireBaselineWin(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.retrieval.metrics.json")
	baselinePath := filepath.Join(dir, "baseline.retrieval.metrics.json")
	current := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "tiny",
		Backend: "cuda",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.30,
		},
	}
	baseline := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "tiny",
		Backend: "bm25",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.25,
		},
	}
	currentData, err := json.Marshal(current)
	if err != nil {
		t.Fatalf("marshal current: %v", err)
	}
	baselineData, err := json.Marshal(baseline)
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}
	if err := os.WriteFile(currentPath, currentData, 0o644); err != nil {
		t.Fatalf("write current: %v", err)
	}
	if err := os.WriteFile(baselinePath, baselineData, 0o644); err != nil {
		t.Fatalf("write baseline: %v", err)
	}

	output := captureRunOutput(t, []string{"compare-retrieval-metrics", "--require-win", currentPath, baselinePath})
	for _, want := range []string{
		"current: " + currentPath + " backend=cuda dataset=tiny",
		"baseline: " + baselinePath + " backend=bm25 dataset=tiny",
		"target: ndcg_at_10=0.3 baseline=0.25 required=0.25 ratio=1.2",
		"retrieval baseline gate: PASS",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("compare-retrieval-metrics output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardPassesAllSelectedDatasetMetrics(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "eos", NDCGAt10: 0.51, RecallAt100: 0.80},
		{Category: "short_retrieval", Dataset: "fiqa", Baseline: "eos", NDCGAt10: 0.12, RecallAt100: 0.35},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "eos", NDCGAt10: 0.50, RecallAt100: 0.79},
		{Category: "short_retrieval", Dataset: "fiqa", Baseline: "eos", NDCGAt10: 0.12, RecallAt100: 0.35},
	})

	output := captureRunOutput(t, []string{
		"gate-scoreboard",
		"--datasets", "scifact,fiqa",
		currentPath,
		anchorPath,
	})
	for _, want := range []string{
		"PASS dataset=scifact metric=ndcg_at_10 current=0.510000 anchor=0.500000 delta=+0.010000",
		"PASS dataset=fiqa metric=recall_at_100 current=0.350000 anchor=0.350000 delta=+0.000000",
		"macro metric=ndcg_at_10 current=0.315000 anchor=0.310000 delta=+0.005000",
		"scoreboard gate: PASS checks=4",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardEosSelectionFallsBackToLegacyMantaRows(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.51, RecallAt100: 0.80},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.50, RecallAt100: 0.79},
	})

	output := captureRunOutput(t, []string{
		"gate-scoreboard",
		"--datasets", "scifact",
		currentPath,
		anchorPath,
	})
	for _, want := range []string{
		"selection: category=short_retrieval baseline=eos",
		"PASS dataset=scifact metric=ndcg_at_10 current=0.510000 anchor=0.500000 delta=+0.010000",
		"scoreboard gate: PASS checks=2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard alias output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardEosHybridSelectionFallsBackToLegacyMantaHybridRows(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta-hybrid", Method: "hybrid_minmax_alpha0.75", NDCGAt10: 0.53, RecallAt100: 0.82},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta-hybrid", Method: "hybrid_minmax_alpha0.75", NDCGAt10: 0.52, RecallAt100: 0.81},
	})

	output := captureRunOutput(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-hybrid",
		"--method", "hybrid_minmax_alpha0.75",
		"--datasets", "scifact",
		currentPath,
		anchorPath,
	})
	for _, want := range []string{
		"selection: category=short_retrieval baseline=eos-hybrid method=hybrid_minmax_alpha0.75",
		"PASS dataset=scifact metric=recall_at_100 current=0.820000 anchor=0.810000 delta=+0.010000",
		"scoreboard gate: PASS checks=2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard hybrid alias output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardPassesTurboQuantStorageMetrics(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	method := "turboquant_ip_b8_overfetch200_fp16_rerank"
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{
			Category:              "short_retrieval",
			Dataset:               "scifact",
			Baseline:              "eos-turboquant-rerank",
			Method:                method,
			Bits:                  8,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.486955,
			RecallAt100:           0.775778,
			VectorBytes:           1347580,
			DenseVectorBytes:      5307392,
			RerankSidecarBytes:    2653696,
			TotalVectorBytes:      4001276,
			CompressionRatio:      3.938462,
			TotalCompressionRatio: 1.326425,
		},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{
			Category:              "short_retrieval",
			Dataset:               "scifact",
			Baseline:              "eos-turboquant-rerank",
			Method:                method,
			Bits:                  8,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.486955,
			RecallAt100:           0.775778,
			CompressionRatio:      3.938462,
			TotalCompressionRatio: 1.326425,
		},
	})

	output := captureRunOutput(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-turboquant-rerank",
		"--method", method,
		"--bits", "8",
		"--datasets", "scifact",
		"--metrics", "ndcg_at_10,recall_at_100,total_compression_ratio",
		currentPath,
		anchorPath,
	})
	for _, want := range []string{
		"selection: category=short_retrieval baseline=eos-turboquant-rerank method=turboquant_ip_b8_overfetch200_fp16_rerank bits=8",
		"PASS dataset=scifact metric=total_compression_ratio current=1.326425 anchor=1.326425 delta=+0.000000",
		"scoreboard gate: PASS checks=3",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard storage metric output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardFailsTurboQuantMissingQuantizerSeed(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	method := "turboquant_ip_b4_overfetch200_fp16_rerank"
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{
			Category:      "short_retrieval",
			Dataset:       "fiqa",
			Baseline:      "eos-turboquant-rerank",
			Method:        method,
			Bits:          4,
			QuantizerSeed: eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			NDCGAt10:      0.121,
			RecallAt100:   0.351,
		},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{
			Category:    "short_retrieval",
			Dataset:     "fiqa",
			Baseline:    "eos-turboquant-rerank",
			Method:      method,
			Bits:        4,
			NDCGAt10:    0.121,
			RecallAt100: 0.351,
		},
	})

	output, err := captureRunOutputAndError(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-turboquant-rerank",
		"--method", method,
		"--bits", "4",
		"--datasets", "fiqa",
		currentPath,
		anchorPath,
	})
	if err == nil {
		t.Fatalf("expected missing quantizer seed failure\noutput:\n%s", output)
	}
	for _, want := range []string{
		"compact scoreboard provenance failed",
		"dataset=fiqa",
		"anchor row",
		"missing quantizer_seed",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing quantizer seed error missing %q: %v\noutput:\n%s", want, err, output)
		}
	}
}

func TestRunGateScoreboardFailsTurboQuantMismatchedQuantizerSeed(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	method := "turboquant_ip_b4_overfetch200_fp16_rerank"
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{
			Category:      "short_retrieval",
			Dataset:       "fiqa",
			Baseline:      "eos-turboquant-rerank",
			Method:        method,
			Bits:          4,
			QuantizerSeed: eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			NDCGAt10:      0.121,
			RecallAt100:   0.351,
		},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{
			Category:      "short_retrieval",
			Dataset:       "fiqa",
			Baseline:      "eos-turboquant-rerank",
			Method:        method,
			Bits:          4,
			QuantizerSeed: eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed + 1,
			NDCGAt10:      0.121,
			RecallAt100:   0.351,
		},
	})

	output, err := captureRunOutputAndError(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-turboquant-rerank",
		"--method", method,
		"--bits", "4",
		"--datasets", "fiqa",
		currentPath,
		anchorPath,
	})
	if err == nil {
		t.Fatalf("expected mismatched quantizer seed failure\noutput:\n%s", output)
	}
	for _, want := range []string{
		"compact scoreboard provenance failed",
		"dataset=fiqa",
		"quantizer_seed mismatch",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("mismatched quantizer seed error missing %q: %v\noutput:\n%s", want, err, output)
		}
	}
}

func TestRunGateScoreboardPassesTurboQuantCrossMethodAnchorSelection(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	currentMethod := "turboquant_ip_b4_overfetch200_fp16_rerank"
	anchorMethod := "turboquant_ip_b4_overfetch250_fp16_rerank"
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{
			Category:              "short_retrieval",
			Dataset:               "scifact",
			Baseline:              "eos-turboquant-rerank",
			Method:                currentMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.487,
			RecallAt100:           0.776,
			TotalCompressionRatio: 2.652,
		},
		{
			Category:              "short_retrieval",
			Dataset:               "nfcorpus",
			Baseline:              "eos-turboquant-rerank",
			Method:                currentMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.357,
			RecallAt100:           0.282,
			TotalCompressionRatio: 2.638,
		},
		{
			Category:              "short_retrieval",
			Dataset:               "fiqa",
			Baseline:              "eos-turboquant-rerank",
			Method:                currentMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.308,
			RecallAt100:           0.553,
			TotalCompressionRatio: 2.649,
		},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{
			Category:              "short_retrieval",
			Dataset:               "scifact",
			Baseline:              "eos-turboquant-rerank",
			Method:                anchorMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.486,
			RecallAt100:           0.775,
			TotalCompressionRatio: 2.652,
		},
		{
			Category:              "short_retrieval",
			Dataset:               "nfcorpus",
			Baseline:              "eos-turboquant-rerank",
			Method:                anchorMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.357,
			RecallAt100:           0.282,
			TotalCompressionRatio: 2.638,
		},
		{
			Category:              "short_retrieval",
			Dataset:               "fiqa",
			Baseline:              "eos-turboquant-rerank",
			Method:                anchorMethod,
			Bits:                  4,
			QuantizerSeed:         eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed,
			RerankStorage:         "fp16",
			NDCGAt10:              0.307,
			RecallAt100:           0.552,
			TotalCompressionRatio: 2.649,
		},
	})

	output := captureRunOutput(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-turboquant-rerank",
		"--method", currentMethod,
		"--anchor-method", anchorMethod,
		"--bits", "4",
		"--datasets", "scifact,nfcorpus,fiqa",
		"--metrics", "ndcg_at_10,recall_at_100,total_compression_ratio",
		currentPath,
		anchorPath,
	})
	for _, want := range []string{
		"current selection: category=short_retrieval baseline=eos-turboquant-rerank method=turboquant_ip_b4_overfetch200_fp16_rerank bits=4",
		"anchor selection: category=short_retrieval baseline=eos-turboquant-rerank method=turboquant_ip_b4_overfetch250_fp16_rerank bits=4",
		"PASS dataset=fiqa metric=recall_at_100 current=0.553000 anchor=0.552000 delta=+0.001000",
		"scoreboard gate: PASS checks=9",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard cross-method output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardReportsMissingCrossMethodAnchorSelection(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	currentMethod := "turboquant_ip_b4_overfetch200_fp16_rerank"
	anchorMethod := "turboquant_ip_b4_overfetch250_fp16_rerank"
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{
			Category:    "short_retrieval",
			Dataset:     "scifact",
			Baseline:    "eos-turboquant-rerank",
			Method:      currentMethod,
			Bits:        4,
			NDCGAt10:    0.487,
			RecallAt100: 0.776,
		},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{
			Category:    "short_retrieval",
			Dataset:     "scifact",
			Baseline:    "eos-turboquant-rerank",
			Method:      anchorMethod,
			Bits:        4,
			NDCGAt10:    0.486,
			RecallAt100: 0.775,
		},
	})

	output, err := captureRunOutputAndError(t, []string{
		"gate-scoreboard",
		"--baseline", "eos-turboquant-rerank",
		"--method", currentMethod,
		"--bits", "4",
		"--datasets", "scifact",
		currentPath,
		anchorPath,
	})
	if err == nil {
		t.Fatalf("expected missing anchor row failure\noutput:\n%s", output)
	}
	for _, want := range []string{
		"anchor scoreboard row selection failed",
		"scoreboard row missing",
		"method=turboquant_ip_b4_overfetch200_fp16_rerank",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("missing anchor error does not contain %q: %v\noutput:\n%s", want, err, output)
		}
	}
}

func TestRunGateScoreboardFailsPerDatasetMissEvenWhenMacroWins(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.70, RecallAt100: 0.90},
		{Category: "short_retrieval", Dataset: "fiqa", Baseline: "manta", NDCGAt10: 0.11, RecallAt100: 0.34},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.50, RecallAt100: 0.79},
		{Category: "short_retrieval", Dataset: "fiqa", Baseline: "manta", NDCGAt10: 0.12, RecallAt100: 0.35},
	})

	output, err := captureRunOutputAndError(t, []string{
		"gate-retrieval-scoreboard",
		"--datasets", "scifact,fiqa",
		currentPath,
		anchorPath,
	})
	if err == nil {
		t.Fatalf("expected scoreboard gate failure\noutput:\n%s", output)
	}
	for _, want := range []string{
		"FAIL dataset=fiqa metric=ndcg_at_10 current=0.110000 anchor=0.120000 delta=-0.010000",
		"FAIL dataset=fiqa metric=recall_at_100 current=0.340000 anchor=0.350000 delta=-0.010000",
		"macro metric=ndcg_at_10 current=0.405000 anchor=0.310000 delta=+0.095000",
		"scoreboard gate: FAIL checks=4 failed=2",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("scoreboard gate failure output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateScoreboardFailsMissingDatasetRow(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.scoreboard.json")
	anchorPath := filepath.Join(dir, "anchor.scoreboard.json")
	writeScoreboardForTest(t, currentPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.51, RecallAt100: 0.80},
	})
	writeScoreboardForTest(t, anchorPath, []retrievalScoreboardRow{
		{Category: "short_retrieval", Dataset: "scifact", Baseline: "manta", NDCGAt10: 0.50, RecallAt100: 0.79},
		{Category: "short_retrieval", Dataset: "fiqa", Baseline: "manta", NDCGAt10: 0.12, RecallAt100: 0.35},
	})

	output, err := captureRunOutputAndError(t, []string{
		"gate-scoreboard",
		"--datasets", "scifact,fiqa",
		currentPath,
		anchorPath,
	})
	if err == nil {
		t.Fatalf("expected missing row failure\noutput:\n%s", output)
	}
	if !strings.Contains(err.Error(), "scoreboard row missing") || !strings.Contains(err.Error(), "dataset=fiqa") {
		t.Fatalf("unexpected missing row error: %v\noutput:\n%s", err, output)
	}
}

func TestRunTrainEmbedFitsContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", "--lr", "0.003", "--contrastive-loss", "infonce", "--temperature", "0.07", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.LearningRate < 0.00299 || checkpoint.Config.LearningRate > 0.00301 {
		t.Fatalf("learning rate = %f, want 0.003", checkpoint.Config.LearningRate)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature < 0.06999 || checkpoint.Config.Temperature > 0.07001 {
		t.Fatalf("temperature = %f, want 0.07", checkpoint.Config.Temperature)
	}
}

func TestValidateTrainEmbedListwiseGeometryWorkloadGuard(t *testing.T) {
	workload := eosruntime.EmbeddingTrainWorkload{
		TrainMode:          "listwise_geometry",
		TrainPairsPerEpoch: 101,
		EvalPairsPerPass:   99,
	}
	err := validateTrainEmbedListwiseGeometryWorkload(workload, eosruntime.EmbeddingTrainRunConfig{
		MaxListwiseGeometryTrainPairs: 100,
		MaxListwiseGeometryEvalPairs:  100,
	})
	if err == nil || !strings.Contains(err.Error(), "--max-listwise-train-pairs 100") {
		t.Fatalf("train guard error = %v, want max train pairs rejection", err)
	}

	workload.TrainPairsPerEpoch = 100
	workload.EvalPairsPerPass = 101
	err = validateTrainEmbedListwiseGeometryWorkload(workload, eosruntime.EmbeddingTrainRunConfig{
		MaxListwiseGeometryTrainPairs: 100,
		MaxListwiseGeometryEvalPairs:  100,
	})
	if err == nil || !strings.Contains(err.Error(), "--max-listwise-eval-pairs 100") {
		t.Fatalf("eval guard error = %v, want max eval pairs rejection", err)
	}

	if err := validateTrainEmbedListwiseGeometryWorkload(workload, eosruntime.EmbeddingTrainRunConfig{}); err != nil {
		t.Fatalf("disabled guard error = %v, want nil", err)
	}
}

func TestRunTrainEmbedExplicitZeroTeacherLossOverridesCheckpoint(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", "--teacher-loss-weight", "0.1", "--teacher-temperature", "2", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	checkpointPath := eosruntime.DefaultEmbeddingCheckpointPath(path)
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(checkpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	checkpoint.Config.TeacherSourceWeights = map[string]float32{"fiqa": 0.25, "nfcorpus": 0.05, "scifact": 1}
	if err := checkpoint.WriteFile(checkpointPath); err != nil {
		t.Fatalf("write checkpoint: %v", err)
	}
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	packageManifest, err := eosruntime.BuildPackageManifest(eosruntime.PackageTraining, mod, map[string]string{
		"artifact":           path,
		"embedding_manifest": eosruntime.DefaultEmbeddingManifestPath(path),
		"weights":            eosruntime.DefaultWeightFilePath(path),
		"memory_plan":        eosruntime.DefaultMemoryPlanPath(path),
		"train_manifest":     eosruntime.DefaultEmbeddingTrainManifestPath(path),
		"checkpoint":         checkpointPath,
		"train_profile":      eosruntime.DefaultEmbeddingTrainProfilePath(path),
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	if err := packageManifest.WriteFile(eosruntime.DefaultPackageManifestPath(path)); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}

	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{2}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "fiqa"},
		{QueryTokens: []int32{2}, PositiveTokens: []int32{2}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "scifact"},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write hard-negative train dataset: %v", err)
	}

	if err := run([]string{"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1", "--epochs", "1", "--batch-size", "2", "--teacher-loss-weight", "0", "--metrics-json", metricsPath, path, trainPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if metrics.Config.TeacherLossWeight != 0 {
		t.Fatalf("metrics teacher loss weight = %f, want explicit zero", metrics.Config.TeacherLossWeight)
	}
	if len(metrics.Config.TeacherSourceWeights) != 0 {
		t.Fatalf("metrics teacher source weights = %v, want cleared when teacher loss is disabled", metrics.Config.TeacherSourceWeights)
	}
	if metrics.Workload.TrainPairsPerEpoch != 8 {
		t.Fatalf("metrics train pairs/epoch = %d, want 8 without inherited teacher pairs", metrics.Workload.TrainPairsPerEpoch)
	}
	checkpoint, err = eosruntime.ReadEmbeddingTrainCheckpointFile(checkpointPath)
	if err != nil {
		t.Fatalf("read trained checkpoint: %v", err)
	}
	if checkpoint.Config.TeacherLossWeight != 0 {
		t.Fatalf("checkpoint teacher loss weight = %f, want explicit zero", checkpoint.Config.TeacherLossWeight)
	}
	if len(checkpoint.Config.TeacherSourceWeights) != 0 {
		t.Fatalf("checkpoint teacher source weights = %v, want cleared when teacher loss is disabled", checkpoint.Config.TeacherSourceWeights)
	}
}

func TestRunTrainEmbedAcceptsPreparedTurboQuantPrefixScoreMode(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "2", "--turboquant-prefix-score-mode", "prepared-ip", path, trainPath}); err != nil {
		t.Fatalf("run train-embed prepared turboquant prefix: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.TurboQuantPrefixScoreMode != eosruntime.TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("turboquant prefix score mode = %q, want %q", checkpoint.Config.TurboQuantPrefixScoreMode, eosruntime.TurboQuantPrefixScoreModePreparedIP)
	}
}

func TestRunTrainEmbedAcceptsTurboQuantPrefixObjectives(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:2=0.25", path, trainPath}); err != nil {
		t.Fatalf("run train-embed turboquant prefix objectives: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if got := checkpoint.Config.TurboQuantPrefixObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("turboquant prefix objectives = %+v", got)
	}
}

func TestRunTrainEmbedAcceptsTurboQuantRankMarginObjectives(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "grouped_infonce", "--matryoshka-dims", "2", "--turboquant-rank-margin-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", "--metrics-json", metricsPath, path, trainPath}); err != nil {
		t.Fatalf("run train-embed turboquant rank-margin objectives: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if got := checkpoint.Config.TurboQuantRankMarginObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("turboquant rank-margin objectives = %+v", got)
	}
	if checkpoint.Config.TurboQuantRankMargin != 0.02 {
		t.Fatalf("turboquant rank margin = %f, want default 0.02", checkpoint.Config.TurboQuantRankMargin)
	}
	if checkpoint.Config.TurboQuantPrefixScoreMode != eosruntime.TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("score mode = %q, want prepared_ip", checkpoint.Config.TurboQuantPrefixScoreMode)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if got := metrics.Config.TurboQuantRankMarginObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("metrics turboquant rank-margin objectives = %+v", got)
	}
	if metrics.Workload.TrainPairsPerEpoch != 10 {
		t.Fatalf("metrics train pairs/epoch = %d, want base grouped+matryoshka+rank-margin pairs", metrics.Workload.TrainPairsPerEpoch)
	}
}

func TestRunTrainEmbedAcceptsTurboQuantCompactObjectives(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "grouped_infonce", "--matryoshka-dims", "2", "--turboquant-compact-objectives", "2:2=0.25", "--metrics-json", metricsPath, path, trainPath}); err != nil {
		t.Fatalf("run train-embed turboquant compact objectives: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if got := checkpoint.Config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("turboquant compact objectives = %+v", got)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if got := metrics.Config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("metrics turboquant compact objectives = %+v", got)
	}
	if metrics.Workload.TrainPairsPerEpoch != 12 {
		t.Fatalf("metrics train pairs/epoch = %d, want base grouped+matryoshka+compact pairs", metrics.Workload.TrainPairsPerEpoch)
	}
}

func TestRunTrainEmbedEvalOnlyAllowsInheritedTurboQuantRankMarginCheckpoint(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	evalPath := filepath.Join(dir, "eval.jsonl")
	trainMetricsPath := filepath.Join(dir, "train.metrics.json")
	evalMetricsPath := filepath.Join(dir, "eval.metrics.json")
	trainExamples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, trainExamples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	evalExamples := []eosruntime.EmbeddingPairExample{
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{1, 2}, Target: 1},
		{LeftTokens: []int32{1, 2}, RightTokens: []int32{2, 3}, Target: -1},
	}
	if err := eosruntime.WriteEmbeddingPairExamplesFile(evalPath, evalExamples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "grouped_infonce", "--matryoshka-dims", "2", "--turboquant-rank-margin-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", "--metrics-json", trainMetricsPath, path, trainPath}); err != nil {
		t.Fatalf("run train-embed turboquant rank-margin objectives: %v", err)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read trained checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantRankMarginObjectives) == 0 {
		t.Fatal("trained checkpoint missing rank-margin objectives")
	}

	if err := run([]string{"train-embed", "--eval-only", "--metrics-json", evalMetricsPath, path, evalPath}); err != nil {
		t.Fatalf("run eval-only with inherited rank-margin checkpoint: %v", err)
	}
	data, err := os.ReadFile(evalMetricsPath)
	if err != nil {
		t.Fatalf("read eval metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal eval metrics: %v", err)
	}
	if len(metrics.Config.TurboQuantRankMarginObjectives) != 0 {
		t.Fatalf("eval-only metrics inherited rank-margin objectives: %+v", metrics.Config.TurboQuantRankMarginObjectives)
	}
	if metrics.Config.TurboQuantRankMargin != 0 {
		t.Fatalf("eval-only rank margin = %f, want 0", metrics.Config.TurboQuantRankMargin)
	}
	checkpoint, err = eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint after eval-only: %v", err)
	}
	if got := checkpoint.Config.TurboQuantRankMarginObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("checkpoint rank-margin objectives after eval-only = %+v", got)
	}
}

func TestRunTrainEmbedRejectsTurboQuantRankMarginWithoutHardNegativeTrain(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--turboquant-rank-margin-objectives", "2:2=0.25", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "requires --hard-negative-train") {
		t.Fatalf("rank-margin without hard-negative error = %v", err)
	}
}

func TestRunTrainEmbedClearTurboQuantPrefixClearsInheritedMetricsAndCheckpoint(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", path, trainPath}); err != nil {
		t.Fatalf("run initial train-embed turboquant prefix objectives: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--clear-turboquant-prefix", "--metrics-json", metricsPath, path, trainPath}); err != nil {
		t.Fatalf("run train-embed clear turboquant prefix: %v", err)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if len(metrics.Config.TurboQuantPrefixBits) != 0 {
		t.Fatalf("metrics turboquant prefix bits = %v, want cleared", metrics.Config.TurboQuantPrefixBits)
	}
	if len(metrics.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("metrics turboquant prefix objectives = %+v, want cleared", metrics.Config.TurboQuantPrefixObjectives)
	}
	if metrics.Config.TurboQuantPrefixWeight != 0 || metrics.Config.TurboQuantPrefixSeed != 0 || metrics.Config.TurboQuantPrefixScoreMode != "" {
		t.Fatalf("metrics turboquant associated config = weight:%f seed:%d mode:%q, want cleared", metrics.Config.TurboQuantPrefixWeight, metrics.Config.TurboQuantPrefixSeed, metrics.Config.TurboQuantPrefixScoreMode)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantPrefixBits) != 0 {
		t.Fatalf("checkpoint turboquant prefix bits = %v, want cleared", checkpoint.Config.TurboQuantPrefixBits)
	}
	if len(checkpoint.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("checkpoint turboquant prefix objectives = %+v, want cleared", checkpoint.Config.TurboQuantPrefixObjectives)
	}
	if checkpoint.Config.TurboQuantPrefixWeight != 0 || checkpoint.Config.TurboQuantPrefixSeed != 0 || checkpoint.Config.TurboQuantPrefixScoreMode != "" {
		t.Fatalf("checkpoint turboquant associated config = weight:%f seed:%d mode:%q, want cleared", checkpoint.Config.TurboQuantPrefixWeight, checkpoint.Config.TurboQuantPrefixSeed, checkpoint.Config.TurboQuantPrefixScoreMode)
	}
}

func TestRunTrainEmbedClearTurboQuantPrefixAllowsRankMarginOnly(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	contrastivePath := filepath.Join(dir, "contrastive.jsonl")
	hardTrainPath := filepath.Join(dir, "hard-train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	contrastiveExamples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(contrastivePath, contrastiveExamples); err != nil {
		t.Fatalf("write contrastive dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", path, contrastivePath}); err != nil {
		t.Fatalf("run initial train-embed turboquant prefix objectives: %v", err)
	}
	hardExamples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(hardTrainPath, hardExamples); err != nil {
		t.Fatalf("write hard-negative dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "grouped_infonce", "--clear-turboquant-prefix", "--turboquant-rank-margin-objectives", "2:2=0.25", "--metrics-json", metricsPath, path, hardTrainPath}); err != nil {
		t.Fatalf("run train-embed clear prefix with rank-margin objectives: %v", err)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if len(metrics.Config.TurboQuantPrefixBits) != 0 || len(metrics.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("metrics compact prefix config = bits:%v objectives:%+v, want cleared", metrics.Config.TurboQuantPrefixBits, metrics.Config.TurboQuantPrefixObjectives)
	}
	if metrics.Config.TurboQuantPrefixWeight != 0 {
		t.Fatalf("metrics prefix weight = %f, want 0 for rank-margin-only", metrics.Config.TurboQuantPrefixWeight)
	}
	if got := metrics.Config.TurboQuantRankMarginObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("metrics rank-margin objectives = %+v, want 2:2=0.25", got)
	}
	if metrics.Config.TurboQuantPrefixSeed != eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("metrics prefix seed = %d, want default %d", metrics.Config.TurboQuantPrefixSeed, eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed)
	}
	if metrics.Config.TurboQuantPrefixScoreMode != eosruntime.TurboQuantPrefixScoreModeReconstructCosine {
		t.Fatalf("metrics score mode = %q, want %q", metrics.Config.TurboQuantPrefixScoreMode, eosruntime.TurboQuantPrefixScoreModeReconstructCosine)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantPrefixBits) != 0 || len(checkpoint.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("checkpoint compact prefix config = bits:%v objectives:%+v, want cleared", checkpoint.Config.TurboQuantPrefixBits, checkpoint.Config.TurboQuantPrefixObjectives)
	}
	if got := checkpoint.Config.TurboQuantRankMarginObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("checkpoint rank-margin objectives = %+v, want 2:2=0.25", got)
	}
}

func TestRunTrainEmbedClearTurboQuantPrefixAllowsCompactObjectiveSeed(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	contrastivePath := filepath.Join(dir, "contrastive.jsonl")
	hardTrainPath := filepath.Join(dir, "hard-train.jsonl")
	metricsPath := filepath.Join(dir, "train.metrics.json")
	contrastiveExamples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(contrastivePath, contrastiveExamples); err != nil {
		t.Fatalf("write contrastive dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", path, contrastivePath}); err != nil {
		t.Fatalf("run initial train-embed turboquant prefix objectives: %v", err)
	}
	hardExamples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(hardTrainPath, hardExamples); err != nil {
		t.Fatalf("write hard-negative dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "grouped_infonce", "--clear-turboquant-prefix", "--turboquant-compact-objectives", "2:2=0.25", "--turboquant-prefix-seed", "777", "--metrics-json", metricsPath, path, hardTrainPath}); err != nil {
		t.Fatalf("run train-embed clear prefix with compact objective seed: %v", err)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		t.Fatalf("unmarshal metrics: %v", err)
	}
	if len(metrics.Config.TurboQuantPrefixBits) != 0 || len(metrics.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("metrics compact prefix config = bits:%v objectives:%+v, want cleared", metrics.Config.TurboQuantPrefixBits, metrics.Config.TurboQuantPrefixObjectives)
	}
	if metrics.Config.TurboQuantPrefixWeight != 0 {
		t.Fatalf("metrics prefix weight = %f, want 0 for compact-only", metrics.Config.TurboQuantPrefixWeight)
	}
	if got := metrics.Config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("metrics compact objectives = %+v, want 2:2=0.25", got)
	}
	if metrics.Config.TurboQuantPrefixSeed != 777 {
		t.Fatalf("metrics prefix seed = %d, want 777", metrics.Config.TurboQuantPrefixSeed)
	}
	if metrics.Config.TurboQuantPrefixScoreMode != "" {
		t.Fatalf("metrics score mode = %q, want empty for compact-only", metrics.Config.TurboQuantPrefixScoreMode)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(path))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if len(checkpoint.Config.TurboQuantPrefixBits) != 0 || len(checkpoint.Config.TurboQuantPrefixObjectives) != 0 {
		t.Fatalf("checkpoint compact prefix config = bits:%v objectives:%+v, want cleared", checkpoint.Config.TurboQuantPrefixBits, checkpoint.Config.TurboQuantPrefixObjectives)
	}
	if got := checkpoint.Config.TurboQuantCompactObjectives; len(got) != 1 || got[0].Dim != 2 || got[0].BitWidth != 2 || got[0].Weight != 0.25 {
		t.Fatalf("checkpoint compact objectives = %+v, want 2:2=0.25", got)
	}
	if checkpoint.Config.TurboQuantPrefixSeed != 777 {
		t.Fatalf("checkpoint prefix seed = %d, want 777", checkpoint.Config.TurboQuantPrefixSeed)
	}
	if checkpoint.Config.TurboQuantPrefixScoreMode != "" {
		t.Fatalf("checkpoint score mode = %q, want empty for compact-only", checkpoint.Config.TurboQuantPrefixScoreMode)
	}
}

func TestRunTrainEmbedRejectsClearTurboQuantPrefixWithNewPrefixConfig(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-bits", "2", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix bits error = %v, want mutually exclusive", err)
	}
	err = run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-objectives", "2:2=0.25", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix objectives error = %v, want mutually exclusive", err)
	}
	err = run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-seed", "777", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix seed error = %v, want mutually exclusive", err)
	}
	err = run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-score-mode", "prepared-ip", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix score mode error = %v, want mutually exclusive", err)
	}
	hardTrainPath := filepath.Join(t.TempDir(), "hard-train.jsonl")
	hardExamples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}, NegativeTokens: [][]int32{{2, 3}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}, NegativeTokens: [][]int32{{1, 2}}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}, NegativeMasks: [][]int32{{1, 1}}, TeacherScores: []float32{1, 0}},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(hardTrainPath, hardExamples); err != nil {
		t.Fatalf("write hard-negative dataset: %v", err)
	}
	err = run([]string{"train-embed", "--hard-negative-train", "--epochs", "1", "--batch-size", "2", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-compact-objectives", "2:2=0.25", "--turboquant-prefix-score-mode", "prepared-ip", path, hardTrainPath})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with compact-only score mode error = %v, want mutually exclusive", err)
	}
}

func TestRunTrainEmbedRejectsInvalidTurboQuantPrefixBits(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "1", path, trainPath})
	if err == nil {
		t.Fatal("expected invalid turboquant prefix bits error")
	}
	if !strings.Contains(err.Error(), "turboquant-prefix-bits") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainEmbedRejectsInvalidTurboQuantPrefixObjectives(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:1=0.25", path, trainPath})
	if err == nil {
		t.Fatal("expected invalid turboquant prefix objectives error")
	}
	if !strings.Contains(err.Error(), "turboquant-prefix-objectives") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainEmbedRejectsMixedTurboQuantPrefixObjectivesAndBits(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "2", "--turboquant-prefix-objectives", "2:2=0.25", path, trainPath})
	if err == nil {
		t.Fatal("expected mixed turboquant prefix objectives/bits error")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainEmbedRejectsInvalidTurboQuantPrefixScoreMode(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	err := run([]string{"train-embed", "--epochs", "1", "--batch-size", "2", "--contrastive-loss", "infonce", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "2", "--turboquant-prefix-score-mode", "bogus", path, trainPath})
	if err == nil {
		t.Fatal("expected invalid turboquant prefix score mode error")
	}
	if !strings.Contains(err.Error(), "turboquant_prefix_score_mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainCorpusRejectsInvalidTurboQuantPrefixScoreMode(t *testing.T) {
	err := run([]string{"train-corpus", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "2", "--turboquant-prefix-score-mode", "bogus", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil {
		t.Fatal("expected invalid turboquant prefix score mode error")
	}
	if !strings.Contains(err.Error(), "turboquant_prefix_score_mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainCorpusRejectsInvalidTurboQuantPrefixObjectives(t *testing.T) {
	err := run([]string{"train-corpus", "--matryoshka-dims", "2", "--turboquant-prefix-objectives", "2:1=0.25", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil {
		t.Fatal("expected invalid turboquant prefix objectives error")
	}
	if !strings.Contains(err.Error(), "turboquant-prefix-objectives") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainCorpusRejectsTurboQuantCompactObjectives(t *testing.T) {
	err := run([]string{"train-corpus", "--matryoshka-dims", "2", "--turboquant-compact-objectives", "2:2=0.25", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil {
		t.Fatal("expected compact objectives hard-negative requirement")
	}
	if !strings.Contains(err.Error(), "hard-negative-train") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainCorpusRejectsMixedTurboQuantPrefixObjectivesAndBits(t *testing.T) {
	err := run([]string{"train-corpus", "--matryoshka-dims", "2", "--turboquant-prefix-bits", "2", "--turboquant-prefix-objectives", "2:2=0.25", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil {
		t.Fatal("expected mixed turboquant prefix objectives/bits error")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunTrainCorpusRejectsClearTurboQuantPrefixWithNewPrefixConfig(t *testing.T) {
	err := run([]string{"train-corpus", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-bits", "2", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix bits error = %v, want mutually exclusive", err)
	}
	err = run([]string{"train-corpus", "--matryoshka-dims", "2", "--clear-turboquant-prefix", "--turboquant-prefix-objectives", "2:2=0.25", filepath.Join(t.TempDir(), "artifact.mll"), filepath.Join(t.TempDir(), "corpus.txt")})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("clear with prefix objectives error = %v, want mutually exclusive", err)
	}
}

func TestRunRenameEmbedRewritesPackageIdentity(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	renamedPath := filepath.Join(t.TempDir(), "manta-embed-v1.mll")

	if err := run([]string{"rename-embed", "--name", "manta-embed-v1", path, renamedPath}); err != nil {
		t.Fatalf("run rename-embed: %v", err)
	}

	mod, err := eosartifact.ReadFile(renamedPath)
	if err != nil {
		t.Fatalf("read renamed artifact: %v", err)
	}
	if mod.Name != "manta-embed-v1" {
		t.Fatalf("module name = %q, want manta-embed-v1", mod.Name)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed manifest: %v", err)
	}
	if manifest.Name != "manta-embed-v1" {
		t.Fatalf("manifest name = %q, want manta-embed-v1", manifest.Name)
	}
	trainManifest, err := eosruntime.ReadEmbeddingTrainManifestFile(eosruntime.DefaultEmbeddingTrainManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed train manifest: %v", err)
	}
	if trainManifest.Name != "manta-embed-v1" || trainManifest.Embedding.Name != "manta-embed-v1" {
		t.Fatalf("train manifest names = %q/%q, want manta-embed-v1", trainManifest.Name, trainManifest.Embedding.Name)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed checkpoint: %v", err)
	}
	if checkpoint.Manifest.Name != "manta-embed-v1" {
		t.Fatalf("checkpoint manifest name = %q, want manta-embed-v1", checkpoint.Manifest.Name)
	}
	packageManifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(renamedPath))
	if err != nil {
		t.Fatalf("read renamed package manifest: %v", err)
	}
	if err := packageManifest.VerifyFiles(map[string]string{
		"artifact":           renamedPath,
		"embedding_manifest": eosruntime.DefaultEmbeddingManifestPath(renamedPath),
		"tokenizer":          eosruntime.DefaultTokenizerPath(renamedPath),
		"weights":            eosruntime.DefaultWeightFilePath(renamedPath),
		"memory_plan":        eosruntime.DefaultMemoryPlanPath(renamedPath),
		"train_manifest":     eosruntime.DefaultEmbeddingTrainManifestPath(renamedPath),
		"checkpoint":         eosruntime.DefaultEmbeddingCheckpointPath(renamedPath),
		"train_profile":      eosruntime.DefaultEmbeddingTrainProfilePath(renamedPath),
	}); err != nil {
		t.Fatalf("verify renamed package manifest: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultTokenizerPath(renamedPath)); err != nil {
		t.Fatalf("renamed tokenizer sidecar missing: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(renamedPath); err != nil {
		t.Fatalf("reload renamed package: %v", err)
	}
}

func TestRunRenameEmbedRewritesTokenizerMaxSequence(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	retargetedPath := filepath.Join(t.TempDir(), "retargeted.mll")

	output := captureRunOutput(t, []string{"rename-embed", "--max-seq", "1024", path, retargetedPath})
	if !strings.Contains(output, "tokenizer max_sequence: 1024") {
		t.Fatalf("rename output missing max sequence\noutput:\n%s", output)
	}

	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(retargetedPath))
	if err != nil {
		t.Fatalf("read retargeted manifest: %v", err)
	}
	if manifest.Tokenizer.MaxSequence != 1024 {
		t.Fatalf("embedding manifest max_sequence = %d, want 1024", manifest.Tokenizer.MaxSequence)
	}
	trainManifest, err := eosruntime.ReadEmbeddingTrainManifestFile(eosruntime.DefaultEmbeddingTrainManifestPath(retargetedPath))
	if err != nil {
		t.Fatalf("read retargeted train manifest: %v", err)
	}
	if trainManifest.Embedding.Tokenizer.MaxSequence != 1024 {
		t.Fatalf("train manifest max_sequence = %d, want 1024", trainManifest.Embedding.Tokenizer.MaxSequence)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(eosruntime.DefaultEmbeddingCheckpointPath(retargetedPath))
	if err != nil {
		t.Fatalf("read retargeted checkpoint: %v", err)
	}
	if checkpoint.Manifest.Tokenizer.MaxSequence != 1024 {
		t.Fatalf("checkpoint max_sequence = %d, want 1024", checkpoint.Manifest.Tokenizer.MaxSequence)
	}
	packageManifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(retargetedPath))
	if err != nil {
		t.Fatalf("read retargeted package manifest: %v", err)
	}
	if err := packageManifest.VerifyFiles(map[string]string{
		"artifact":           retargetedPath,
		"embedding_manifest": eosruntime.DefaultEmbeddingManifestPath(retargetedPath),
		"tokenizer":          eosruntime.DefaultTokenizerPath(retargetedPath),
		"weights":            eosruntime.DefaultWeightFilePath(retargetedPath),
		"memory_plan":        eosruntime.DefaultMemoryPlanPath(retargetedPath),
		"train_manifest":     eosruntime.DefaultEmbeddingTrainManifestPath(retargetedPath),
		"checkpoint":         eosruntime.DefaultEmbeddingCheckpointPath(retargetedPath),
		"train_profile":      eosruntime.DefaultEmbeddingTrainProfilePath(retargetedPath),
	}); err != nil {
		t.Fatalf("verify retargeted package manifest: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultTokenizerPath(retargetedPath)); err != nil {
		t.Fatalf("retargeted tokenizer sidecar missing: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(retargetedPath); err != nil {
		t.Fatalf("reload retargeted package: %v", err)
	}
}

func TestRunRenameEmbedRejectsNoopAndNegativeMaxSequence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.mll")
	outputPath := filepath.Join(t.TempDir(), "output.mll")
	if err := run([]string{"rename-embed", path, outputPath}); err == nil || !strings.Contains(err.Error(), "requires --name or a positive --max-seq") {
		t.Fatalf("noop rename error = %v, want requires --name or positive --max-seq", err)
	}
	if err := run([]string{"rename-embed", "--max-seq=-1", path, outputPath}); err == nil || !strings.Contains(err.Error(), "--max-seq must be non-negative") {
		t.Fatalf("negative max-seq error = %v, want non-negative", err)
	}
}

func TestRunTrainEmbedPlanOnlyShowsWorkload(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"planned workload:",
		"train=4 contrastive examples",
		"steps/epoch=2",
		"train_pairs/epoch=8",
		"eval=4 contrastive examples",
		"eval_pairs/pass=16",
		"pairs(planned=80 actual=0)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("plan-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedNoTokenizerAllowsRetrievalEvalTokenizer(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.tokens.jsonl")
	evalPath := filepath.Join(dir, "eval.tokens.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	tokenizerPath := filepath.Join(dir, "retrieval.tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	output := captureRunOutput(t, []string{
		"train-embed",
		"--plan-only",
		"--no-tokenizer",
		"--retrieval-eval-dir", filepath.Join(dir, "missing-scifact"),
		"--retrieval-eval-tokenizer", tokenizerPath,
		"--retrieval-eval-role-mode", "raw",
		"--select-metric", "retrieval_ndcg",
		path,
		trainPath,
		evalPath,
	})
	if !strings.Contains(output, "planned workload:") {
		t.Fatalf("plan-only output missing workload\noutput:\n%s", output)
	}
}

// These tests drive train-embed's retrieval-gated selection defaults: with
// --restore-best on, runs that didn't explicitly pass --select-metric
// retrieval_ndcg and --retrieval-eval-dir have previously silently restored
// the untrained step-0 checkpoint (see commit 2861ac9's diagnosis).

func TestRunTrainEmbedAutoSelectsRetrievalNDCGWhenRetrievalEvalDirSetWithoutExplicitMetric(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.tokens.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	tokenizerPath := filepath.Join(dir, "retrieval.tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed",
		"--plan-only",
		"--no-tokenizer",
		"--retrieval-eval-dir", filepath.Join(dir, "missing-scifact"),
		"--retrieval-eval-tokenizer", tokenizerPath,
		"--retrieval-eval-role-mode", "raw",
		path,
		trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if !strings.Contains(stderr, `select-metric: auto-selected "retrieval_ndcg"`) {
		t.Fatalf("stderr missing retrieval-gated auto-selection log line\nstderr:\n%s", stderr)
	}
	if !strings.Contains(stderr, "--retrieval-eval-dir") {
		t.Fatalf("stderr auto-selection line must name --retrieval-eval-dir\nstderr:\n%s", stderr)
	}
	if strings.Contains(stderr, "restore-best with pairwise selection") {
		t.Fatalf("no restore-best warning expected once retrieval-eval-dir gates selection\nstderr:\n%s", stderr)
	}
}

func TestRunTrainEmbedExplicitSelectMetricWinsOverRetrievalEvalDirAutoSelect(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train.tokens.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version:      eosruntime.TokenizerFileVersion,
		Tokens:       []string{"<pad>", "<unk>", "alpha", "beta"},
		PadToken:     "<pad>",
		UnknownToken: "<unk>",
	}
	tokenizerPath := filepath.Join(dir, "retrieval.tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed",
		"--plan-only",
		"--no-tokenizer",
		"--retrieval-eval-dir", filepath.Join(dir, "missing-scifact"),
		"--retrieval-eval-tokenizer", tokenizerPath,
		"--retrieval-eval-role-mode", "raw",
		"--select-metric", "top1_accuracy",
		path,
		trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if strings.Contains(stderr, "auto-selected") {
		t.Fatalf("explicit --select-metric must not be overridden\nstderr:\n%s", stderr)
	}
}

func TestRunTrainEmbedWarnsOnRestoreBestPairwiseSelectionForHardNegativeTrain(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{2}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "fiqa"},
		{QueryTokens: []int32{2}, PositiveTokens: []int32{2}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "scifact"},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write hard-negative train dataset: %v", err)
	}
	// restore-best defaults to true and select-metric defaults to
	// top1_accuracy; no --retrieval-eval-dir is passed, so this must warn.
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1",
		"--epochs", "1", "--batch-size", "2",
		path, trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	for _, want := range []string{
		"WARNING",
		"--hard-negative-train",
		"--restore-best",
		"restore-best with pairwise selection has previously restored untrained",
		"step-0 checkpoints",
		"--retrieval-eval-dir",
	} {
		if !strings.Contains(stderr, want) {
			t.Fatalf("stderr missing %q in restore-best pairwise selection warning\nstderr:\n%s", want, stderr)
		}
	}
}

// TestRunTrainEmbedNoWarningForEvalOnly verifies the restore-best
// pairwise-selection warning does not fire under --eval-only: no selection
// (and hence no restore-best checkpoint choice) ever happens in the
// eval-only path, so the warning would be spurious there.
func TestRunTrainEmbedNoWarningForEvalOnly(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{2}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "fiqa"},
		{QueryTokens: []int32{2}, PositiveTokens: []int32{2}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "scifact"},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write hard-negative train dataset: %v", err)
	}
	// Same restore-best=true (default) and pairwise select-metric default that
	// TestRunTrainEmbedWarnsOnRestoreBestPairwiseSelectionForHardNegativeTrain
	// exercises, but with --eval-only set: no selection happens in this path,
	// so the warning must not fire.
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed", "--eval-only", "--hard-negative-train", "--hard-negatives-per-query", "1",
		path, trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed --eval-only: %v", err)
	}
	if strings.Contains(stderr, "restore-best with pairwise selection") {
		t.Fatalf("no warning expected under --eval-only (no selection occurs)\nstderr:\n%s", stderr)
	}
}

func TestRunTrainEmbedNoWarningWhenRestoreBestDisabled(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "hard-train.jsonl")
	examples := []eosruntime.EmbeddingHardNegativeExample{
		{QueryTokens: []int32{1}, PositiveTokens: []int32{1}, NegativeTokens: [][]int32{{2}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "fiqa"},
		{QueryTokens: []int32{2}, PositiveTokens: []int32{2}, NegativeTokens: [][]int32{{1}}, QueryMask: []int32{1}, PositiveMask: []int32{1}, NegativeMasks: [][]int32{{1}}, Source: "scifact"},
	}
	if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write hard-negative train dataset: %v", err)
	}
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1",
		"--epochs", "1", "--batch-size", "2", "--restore-best=false",
		path, trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if strings.Contains(stderr, "restore-best with pairwise selection") {
		t.Fatalf("no warning expected once --restore-best is disabled\nstderr:\n%s", stderr)
	}
}

func TestRunTrainEmbedNoWarningForPlainContrastiveTrain(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	// Plain contrastive training (no hard-negative/score-spectrum/listwise-
	// geometry/vector-distill mode) is not one of the retrieval-oriented
	// modes the warning targets, so existing plain-contrastive scripts must
	// stay silent even with restore-best's true default and no
	// --retrieval-eval-dir.
	_, stderr, err := captureRunStderrAndOutput(t, []string{
		"train-embed", "--epochs", "1", "--batch-size", "2", path, trainPath,
	})
	if err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if strings.Contains(stderr, "restore-best with pairwise selection") {
		t.Fatalf("no warning expected for plain contrastive training\nstderr:\n%s", stderr)
	}
}

func TestRunTrainEmbedRejectsInvalidRetrievalEvalRoleMode(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"train-embed",
		"--plan-only",
		"--retrieval-eval-role-mode", "prefixed",
		"artifact.mll",
		"train.jsonl",
	})
	if err == nil {
		t.Fatal("train-embed accepted invalid retrieval eval role mode")
	}
	if !strings.Contains(err.Error(), `unsupported retrieval-eval-role-mode "prefixed"`) {
		t.Fatalf("invalid role-mode error = %v", err)
	}
}

func TestRunTrainEmbedRejectsInvalidVectorDistillOptimizerSyncMode(t *testing.T) {
	_, err := captureRunOutputAndError(t, []string{
		"train-embed",
		"--plan-only",
		"--vector-distill-train",
		"--vector-distill-optimizer-sync", "eventually",
		"artifact.mll",
		"train.jsonl",
	})
	if err == nil {
		t.Fatal("train-embed accepted invalid vector-distill optimizer sync mode")
	}
	if !strings.Contains(err.Error(), `unsupported vector_distill_optimizer_sync "eventually"`) {
		t.Fatalf("invalid vector-distill optimizer sync error = %v", err)
	}
}

func TestRunTrainEmbedProgressEveryPrintsInnerProgress(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--epochs", "1", "--batch-size", "2", "--progress-every", "1", path, trainPath})
	for _, want := range []string{
		"progress: phase=fit_start mode=train",
		"progress: phase=train epoch=1 batch=1/2",
		"batch_examples=2",
		"epoch_pairs=4/",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("progress output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyUsesSingleContrastiveDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"train=0 contrastive examples",
		"eval=2 contrastive examples",
		"pairs(planned=4 actual=4)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyUsesSingleTextPairDataset(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", path, evalPath})
	for _, want := range []string{
		"evaluated package",
		"tokenizer:",
		"epochs: 0",
		"run_steps: 0",
		"final eval:",
		"pairs=2",
		"train=0 pairwise examples",
		"eval=2 pairwise examples",
		"pairs(planned=2 actual=2)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only text output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedEvalOnlyWritesMetricsJSON(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	metricsPath := filepath.Join(t.TempDir(), "metrics.json")

	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--metrics-json", metricsPath, path, evalPath})
	if !strings.Contains(output, "metrics: "+metricsPath) {
		t.Fatalf("eval-only output missing metrics path %q\noutput:\n%s", metricsPath, output)
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics json: %v", err)
	}
	var got struct {
		Schema  string `json:"schema"`
		Command string `json:"command"`
		Mode    string `json:"mode"`
		Summary struct {
			StepsRun int `json:"steps_run"`
		} `json:"summary"`
		FinalEval *struct {
			PairCount       int     `json:"pair_count"`
			Top1            float32 `json:"top1_accuracy"`
			MRR             float32 `json:"mean_reciprocal_rank"`
			RetrievalNDCG   float32 `json:"retrieval_ndcg_at_10"`
			RetrievalMAP    float32 `json:"retrieval_map_at_100"`
			RetrievalRecall float32 `json:"retrieval_recall_at_100"`
		} `json:"final_eval"`
		Workload struct {
			ActualEvalPairs int64 `json:"actual_eval_pairs"`
		} `json:"workload"`
		Throughput struct {
			EvalPairsPerSecond float64 `json:"eval_pairs_per_second"`
		} `json:"throughput"`
		ProfileDelta struct {
			OptimizerUpdates int64 `json:"optimizer_updates"`
		} `json:"profile_delta"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode metrics json: %v\n%s", err, string(data))
	}
	if got.Schema != "manta.embedding_train_metrics.v1" || got.Command != "train-embed" || got.Mode != "eval" {
		t.Fatalf("unexpected metrics identity: %+v", got)
	}
	if got.Summary.StepsRun != 0 {
		t.Fatalf("steps_run = %d, want 0", got.Summary.StepsRun)
	}
	if got.FinalEval == nil || got.FinalEval.PairCount != 4 {
		t.Fatalf("final_eval = %+v, want pair_count=4", got.FinalEval)
	}
	if got.FinalEval.Top1 <= 0 || got.FinalEval.MRR <= 0 {
		t.Fatalf("expected ranking metrics in JSON, got final_eval %+v", *got.FinalEval)
	}
	if got.FinalEval.RetrievalNDCG != 0 || got.FinalEval.RetrievalMAP != 0 || got.FinalEval.RetrievalRecall != 0 {
		t.Fatalf("expected zero retrieval metrics without retrieval eval, got final_eval %+v", *got.FinalEval)
	}
	if got.Workload.ActualEvalPairs != 4 {
		t.Fatalf("actual_eval_pairs = %d, want 4", got.Workload.ActualEvalPairs)
	}
	if got.Throughput.EvalPairsPerSecond <= 0 {
		t.Fatalf("eval_pairs_per_second = %f, want positive", got.Throughput.EvalPairsPerSecond)
	}
	if got.ProfileDelta.OptimizerUpdates != 0 {
		t.Fatalf("optimizer_updates = %d, want 0", got.ProfileDelta.OptimizerUpdates)
	}
}

func TestTrainMetricsPayloadIncludesEffectiveLearningRateAndMovement(t *testing.T) {
	payload := trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			Config:                eosruntime.EmbeddingTrainRunConfig{LearningRate: 0, MovementDiagnostics: true},
			EffectiveLearningRate: 2.5e-8,
			FinalTrain: eosruntime.EmbeddingTrainMetrics{
				Loss:         1.25,
				AverageScore: 0.5,
				BatchSize:    4,
				Movement: &eosruntime.EmbeddingTrainMovementMetrics{
					Gradient: eosruntime.EmbeddingTrainStatMetrics{
						L2Norm:       0.75,
						MaxAbs:       0.25,
						NonzeroCount: 3,
						TotalCount:   5,
					},
					ParameterDelta: eosruntime.EmbeddingTrainStatMetrics{
						L2Norm:       0.125,
						MaxAbs:       0.05,
						NonzeroCount: 2,
						TotalCount:   5,
					},
				},
			},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	if payload.Config.LearningRate != 0 || payload.Config.EffectiveLearningRate != 2.5e-8 {
		t.Fatalf("learning rates = configured:%g effective:%g, want 0 and 2.5e-8", payload.Config.LearningRate, payload.Config.EffectiveLearningRate)
	}
	if !payload.Config.MovementDiagnostics {
		t.Fatalf("movement diagnostics config = false, want true")
	}
	if payload.FinalTrain.Movement == nil {
		t.Fatalf("final train movement missing")
	}
	if payload.FinalTrain.Movement.Gradient.NonzeroCount != 3 || payload.FinalTrain.Movement.ParameterDelta.NonzeroCount != 2 {
		t.Fatalf("movement payload = %+v", payload.FinalTrain.Movement)
	}
}

func TestTrainMetricsPayloadCompactForwardAcceleratorJSON(t *testing.T) {
	payload := trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			EndProfile: eosruntime.EmbeddingTrainProfile{},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal metrics payload: %v", err)
	}
	if strings.Contains(string(data), `"compact_forward"`) {
		t.Fatalf("compact_forward present for absent compact backend: %s", string(data))
	}

	payload = trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			EndProfile: eosruntime.EmbeddingTrainProfile{
				CompactForwardBackend: eosartifact.BackendCUDA,
			},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	data, err = json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal cuda metrics payload: %v", err)
	}
	var got struct {
		Accelerators struct {
			CompactForward string `json:"compact_forward"`
		} `json:"accelerators"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal metrics payload: %v\n%s", err, string(data))
	}
	if got.Accelerators.CompactForward != "cuda" {
		t.Fatalf("compact_forward = %q, want cuda\n%s", got.Accelerators.CompactForward, string(data))
	}
}

func TestTrainMetricsPayloadCompactTrainAcceleratorJSON(t *testing.T) {
	payload := trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			EndProfile: eosruntime.EmbeddingTrainProfile{},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal metrics payload: %v", err)
	}
	var absentRaw map[string]any
	if err := json.Unmarshal(data, &absentRaw); err != nil {
		t.Fatalf("unmarshal absent metrics payload: %v\n%s", err, string(data))
	}
	absentDelta := absentRaw["profile_delta"].(map[string]any)
	for _, key := range compactTrainProfileDeltaCounterKeysForTest() {
		if _, ok := absentDelta[key]; ok {
			t.Fatalf("%s present for absent compact train stats: %s", key, string(data))
		}
	}
	if _, ok := absentDelta["compact_train_stats_available"]; ok {
		t.Fatalf("compact_train_stats_available present for absent compact train stats: %s", string(data))
	}
	if strings.Contains(string(data), `"compact_train"`) {
		t.Fatalf("compact_train present for absent compact train backend/stats: %s", string(data))
	}

	payload = trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			EndProfile: eosruntime.EmbeddingTrainProfile{
				CompactTrainBackend: eosartifact.BackendCUDA,
			},
			DeltaProfile: eosruntime.EmbeddingTrainProfile{
				CompactTrain: &backend.CompactTrainAcceleratorStats{
					ForwardCalls:          2,
					PooledDownloadedBytes: 1024,
					PackedBytesAvoided:    4096,
					LiveHandles:           0,
				},
			},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	data, err = json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal cuda metrics payload: %v", err)
	}
	var got struct {
		Accelerators struct {
			CompactTrain string `json:"compact_train"`
		} `json:"accelerators"`
		ProfileDelta struct {
			StatsAvailable       bool  `json:"compact_train_stats_available"`
			ForwardCalls         int64 `json:"compact_train_forward_calls"`
			PooledDownloadedByte int64 `json:"compact_train_pooled_downloaded_bytes"`
			PackedBytesAvoided   int64 `json:"compact_train_packed_bytes_avoided"`
		} `json:"profile_delta"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal metrics payload: %v\n%s", err, string(data))
	}
	if got.Accelerators.CompactTrain != "cuda" || !got.ProfileDelta.StatsAvailable || got.ProfileDelta.ForwardCalls != 2 || got.ProfileDelta.PooledDownloadedByte != 1024 || got.ProfileDelta.PackedBytesAvoided != 4096 {
		t.Fatalf("compact_train metrics mismatch: %+v\n%s", got, string(data))
	}
	var presentRaw map[string]any
	if err := json.Unmarshal(data, &presentRaw); err != nil {
		t.Fatalf("unmarshal raw compact train metrics payload: %v\n%s", err, string(data))
	}
	presentDelta := presentRaw["profile_delta"].(map[string]any)
	for _, key := range compactTrainProfileDeltaCounterKeysForTest() {
		value, ok := presentDelta[key]
		if !ok {
			t.Fatalf("%s missing from compact train stats payload: %s", key, string(data))
		}
		if _, ok := value.(float64); !ok {
			t.Fatalf("%s = %T(%v), want numeric JSON value", key, value, value)
		}
	}
	for _, key := range []string{
		"compact_train_backward_calls",
		"compact_train_grad_pooled_uploaded_bytes",
		"compact_train_host_grad_upload_bytes_avoided",
		"compact_train_graph_captures",
		"compact_train_graph_replays",
		"compact_train_workspace_arena_bytes",
		"compact_train_resident_grad_bytes",
		"compact_train_live_handles",
	} {
		if presentDelta[key] != float64(0) {
			t.Fatalf("%s = %v, want numeric zero\n%s", key, presentDelta[key], string(data))
		}
	}
}

func compactTrainProfileDeltaCounterKeysForTest() []string {
	return []string{
		"compact_train_forward_calls",
		"compact_train_backward_calls",
		"compact_train_pooled_downloaded_bytes",
		"compact_train_grad_pooled_uploaded_bytes",
		"compact_train_packed_bytes_avoided",
		"compact_train_host_grad_upload_bytes_avoided",
		"compact_train_kernel_launches",
		"compact_train_kernel_synchronizations",
		"compact_train_graph_captures",
		"compact_train_graph_replays",
		"compact_train_activation_arena_bytes",
		"compact_train_workspace_arena_bytes",
		"compact_train_resident_grad_bytes",
		"compact_train_live_handles",
		"compact_train_fallback_or_unhandled",
	}
}

func TestEvalMetricsPayloadIncludesRetrievalMetrics(t *testing.T) {
	payload := evalMetricsPayload(&eosruntime.EmbeddingEvalMetrics{
		RetrievalNDCGAt10:    0.14,
		RetrievalMAPAt100:    0.22,
		RetrievalRecallAt100: 0.31,
		RetrievalEval: &eosruntime.RetrievalEvalMetrics{
			Schema:  eosruntime.RetrievalEvalMetricsSchema,
			Dataset: "tiny",
			Backend: "cpu",
			Inputs: eosruntime.RetrievalEvalInputMetrics{
				Documents:     3,
				Queries:       2,
				RelevantPairs: 2,
				ScoredPairs:   6,
			},
			Throughput: eosruntime.RetrievalEvalThroughput{
				ElapsedSeconds:       1.25,
				DocumentEmbedSeconds: 0.5,
				QueryEmbedSeconds:    0.25,
				ScoreSeconds:         0.1,
				DocumentsPerSecond:   6,
				QueriesPerSecond:     8,
				ScoresPerSecond:      60,
			},
		},
	})
	if payload == nil {
		t.Fatal("eval metrics payload missing")
	}
	if payload.RetrievalNDCGAt10 != 0.14 || payload.RetrievalMAPAt100 != 0.22 || payload.RetrievalRecallAt100 != 0.31 {
		t.Fatalf("retrieval payload = %+v", payload)
	}
	if payload.RetrievalEval == nil || payload.RetrievalEval.Backend != "cpu" || payload.RetrievalEval.Inputs.Documents != 3 || payload.RetrievalEval.Throughput.ScoresPerSecond != 60 {
		t.Fatalf("retrieval eval payload = %+v", payload.RetrievalEval)
	}
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	var got struct {
		RetrievalEval *struct {
			Backend string `json:"backend"`
			Inputs  struct {
				Documents   int   `json:"documents"`
				Queries     int   `json:"queries"`
				ScoredPairs int64 `json:"scored_pairs"`
			} `json:"inputs"`
			Throughput struct {
				ElapsedSeconds  float64 `json:"elapsed_seconds"`
				ScoresPerSecond float64 `json:"scores_per_second"`
			} `json:"throughput"`
		} `json:"retrieval_eval"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal payload: %v\n%s", err, string(data))
	}
	if got.RetrievalEval == nil || got.RetrievalEval.Backend != "cpu" || got.RetrievalEval.Inputs.ScoredPairs != 6 || got.RetrievalEval.Throughput.ElapsedSeconds != 1.25 {
		t.Fatalf("json retrieval eval = %+v", got.RetrievalEval)
	}
}

func TestTrainMetricsPayloadIncludesEvalHistoryRetrievalMetrics(t *testing.T) {
	payload := trainMetricsPayload(
		"train-embed",
		"train",
		"model.mll",
		"",
		eosruntime.EmbeddingTrainRunSummary{
			EvalHistory: []eosruntime.EmbeddingTrainEvalSummary{
				{
					Epoch:    1,
					Step:     3,
					EvalPass: 2,
					Trigger:  "step",
					Improved: true,
					Eval: &eosruntime.EmbeddingEvalMetrics{
						RetrievalNDCGAt10:    0.41,
						RetrievalMAPAt100:    0.32,
						RetrievalRecallAt100: 0.73,
						RetrievalEval: &eosruntime.RetrievalEvalMetrics{
							Schema:  eosruntime.RetrievalEvalMetricsSchema,
							Dataset: "tiny",
							Backend: "cpu",
							Inputs: eosruntime.RetrievalEvalInputMetrics{
								Documents:   4,
								Queries:     2,
								ScoredPairs: 8,
							},
							Throughput: eosruntime.RetrievalEvalThroughput{
								ElapsedSeconds:     2,
								ScoresPerSecond:    4,
								QueriesPerSecond:   1,
								DocumentsPerSecond: 2,
							},
						},
					},
				},
			},
		},
		eosruntime.EmbeddingTrainPackagePaths{},
		nil,
	)
	if len(payload.EvalHistory) != 1 {
		t.Fatalf("eval history len = %d, want 1", len(payload.EvalHistory))
	}
	record := payload.EvalHistory[0]
	if record.Epoch != 1 || record.Step != 3 || record.EvalPass != 2 || record.Trigger != "step" || !record.Improved {
		t.Fatalf("eval history metadata = %+v", record)
	}
	if record.Eval == nil || record.Eval.RetrievalNDCGAt10 != 0.41 || record.Eval.RetrievalMAPAt100 != 0.32 || record.Eval.RetrievalRecallAt100 != 0.73 {
		t.Fatalf("eval history retrieval metrics = %+v", record.Eval)
	}

	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	var got struct {
		EvalHistory []struct {
			Epoch    int    `json:"epoch"`
			Step     int    `json:"step"`
			EvalPass int    `json:"eval_pass"`
			Trigger  string `json:"trigger"`
			Improved bool   `json:"improved"`
			Eval     struct {
				RetrievalNDCG   float32 `json:"retrieval_ndcg_at_10"`
				RetrievalMAP    float32 `json:"retrieval_map_at_100"`
				RetrievalRecall float32 `json:"retrieval_recall_at_100"`
				RetrievalEval   *struct {
					Backend string `json:"backend"`
					Inputs  struct {
						Documents   int   `json:"documents"`
						Queries     int   `json:"queries"`
						ScoredPairs int64 `json:"scored_pairs"`
					} `json:"inputs"`
					Throughput struct {
						ScoresPerSecond float64 `json:"scores_per_second"`
					} `json:"throughput"`
				} `json:"retrieval_eval"`
			} `json:"eval"`
		} `json:"eval_history"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal payload: %v\n%s", err, string(data))
	}
	if len(got.EvalHistory) != 1 || got.EvalHistory[0].Eval.RetrievalNDCG != 0.41 || got.EvalHistory[0].Eval.RetrievalMAP != 0.32 || got.EvalHistory[0].Eval.RetrievalRecall != 0.73 {
		t.Fatalf("json eval history = %+v", got.EvalHistory)
	}
	if got.EvalHistory[0].Eval.RetrievalEval == nil || got.EvalHistory[0].Eval.RetrievalEval.Backend != "cpu" || got.EvalHistory[0].Eval.RetrievalEval.Inputs.ScoredPairs != 8 {
		t.Fatalf("json eval history retrieval eval = %+v", got.EvalHistory[0].Eval.RetrievalEval)
	}
}

func TestPrintFinalRetrievalEvalIncludesBackendCountsAndThroughput(t *testing.T) {
	output := captureStdout(t, func() {
		printFinalRetrievalEval(&eosruntime.EmbeddingEvalMetrics{
			RetrievalEval: &eosruntime.RetrievalEvalMetrics{
				Dataset: "tiny",
				Backend: "cpu",
				Inputs: eosruntime.RetrievalEvalInputMetrics{
					Documents:     3,
					Queries:       2,
					RelevantPairs: 2,
					ScoredPairs:   6,
				},
				Throughput: eosruntime.RetrievalEvalThroughput{
					ElapsedSeconds:       1.25,
					DocumentEmbedSeconds: 0.5,
					QueryEmbedSeconds:    0.25,
					ScoreSeconds:         0.1,
					DocumentsPerSecond:   6,
					QueriesPerSecond:     8,
					ScoresPerSecond:      60,
				},
			},
		})
	})
	for _, want := range []string{
		"final retrieval eval: dataset=tiny backend=cpu docs=3 queries=2 relevant_pairs=2 scored_pairs=6",
		"elapsed=1.250s doc_embed=0.500s query_embed=0.250s score=0.100s",
		"docs_per_sec=6.00 queries_per_sec=8.00 scores_per_sec=60.00",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunCompareTrainMetricsReportsCurrentAndBaselineDeltas(t *testing.T) {
	dir := t.TempDir()
	currentPath := filepath.Join(dir, "current.metrics.json")
	baselinePath := filepath.Join(dir, "baseline.metrics.json")
	current := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
			PairCount:          128,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      300,
			OptimizerStepsPerSecond: 0.15,
			PairsPerSecond:          150000,
			ElapsedSeconds:          10,
		},
		Accelerators: trainAcceleratorsJSON{Forward: "cuda", Optimizer: "cuda", Activation: "cuda", Contrastive: "cuda"},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    4,
			ActivationCalls:     3,
			ContrastiveCalls:    2,
		},
	}
	baseline := current
	baseline.Artifact = "baseline.mll"
	baseline.FinalEval = &evalMetricsJSON{
		Top1Accuracy:       0.8,
		Top5Accuracy:       0.95,
		Top10Accuracy:      0.99,
		MeanReciprocalRank: 0.9,
		ROCAUC:             0.7,
		ScoreMargin:        0.10,
		Loss:               0.13,
		MeanPositiveRank:   1.3,
		PairCount:          128,
	}
	baseline.Throughput.TrainPairsPerSecond = 100000
	baseline.ProfileDelta.MatMulRuns = 1500
	writeMetricsJSONForTest(t, currentPath, current)
	writeMetricsJSONForTest(t, baselinePath, baseline)

	output := captureRunOutput(t, []string{"compare-train-metrics", currentPath, baselinePath})
	for _, want := range []string{
		"identity: schema=manta.embedding_train_metrics.v1 command=train-embed mode=eval artifact=current.mll",
		"quality: top1=0.900000",
		"throughput: train_pairs/s=120000.00",
		"accelerators: forward=cuda optimizer=cuda activation=cuda contrastive=cuda",
		"profile_delta: matmul_runs=1000",
		"baseline: " + baselinePath,
		"quality_delta: top1=+0.100000",
		"throughput_delta: train_pairs/s=+20000.00",
		"profile_delta_delta: matmul_runs=-500",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("compare output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsReportsDeviceBackedEfficiency(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100000,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:          90,
			TrainSeconds:            80,
			EvalSeconds:             10,
			TrainPairsPerSecond:     120000,
			EvalPairsPerSecond:      50000,
			OptimizerStepsPerSecond: 0.5,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "host",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   500,
			MatMulRunDownloadMB: 250,
			OptimizerUpdates:    10,
			OptimizerSyncs:      20,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"backend: forward=cuda optimizer=cuda activation=host contrastive=cuda",
		"efficiency: matmul_runs/update=100.00 pairs/matmul_run=100.00 optimizer_syncs/update=2.00",
		"transfer: total_mb=750.00 mb/matmul_run=0.7500 kb/pair=7.6800",
		"finding: ok production-critical accelerators are device-backed",
		"finding: note activation accelerator is host",
		"diagnosis: OK warnings=0 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnHostFallbacks(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Workload: trainWorkloadJSON{
			ActualTrainPairs: 100,
		},
		Throughput: trainThroughputJSON{
			ElapsedSeconds:      10,
			TrainSeconds:        10,
			TrainPairsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "host",
			Optimizer:   "host",
			Activation:  "host",
			Contrastive: "host",
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn production-critical accelerators include host fallback: forward=host optimizer=host contrastive=host",
		"finding: warn training run recorded zero optimizer updates",
		"finding: warn training pairs were processed but train_pairs/s is zero",
		"diagnosis: WARN warnings=3 notes=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("warning diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunDiagnoseTrainMetricsWarnsOnMissingOptimizerStepRate(t *testing.T) {
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "train",
		Artifact: "current.mll",
		Throughput: trainThroughputJSON{
			TrainSeconds:            2,
			TrainPairsPerSecond:     500,
			OptimizerStepsPerSecond: 0,
		},
		Accelerators: trainAcceleratorsJSON{
			Forward:     "cuda",
			Optimizer:   "cuda",
			Activation:  "cuda",
			Contrastive: "cuda",
		},
		ProfileDelta: trainProfileDeltaJSON{
			OptimizerUpdates: 2,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"diagnose-train-metrics", metricsPath})
	for _, want := range []string{
		"finding: warn optimizer updates were recorded but optimizer_steps/s is zero",
		"diagnosis: WARN warnings=1 notes=0",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("optimizer-rate diagnosis output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksThresholdFile(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			Top1Accuracy:       0.9,
			Top5Accuracy:       0.98,
			Top10Accuracy:      1,
			MeanReciprocalRank: 0.95,
			ROCAUC:             0.73,
			ScoreMargin:        0.12,
			Loss:               0.11,
			MeanPositiveRank:   1.1,
		},
		Throughput: trainThroughputJSON{
			TrainPairsPerSecond:     120000,
			OptimizerStepsPerSecond: 0.15,
		},
		ProfileDelta: trainProfileDeltaJSON{
			MatMulRuns:          1000,
			MatMulRunUploadMB:   100,
			MatMulRunDownloadMB: 80,
			OptimizerUpdates:    0,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	thresholds := "" +
		"EOS_MIN_MRR=0.90\n" +
		"EOS_MIN_TOP1=0.80\n" +
		"EOS_MAX_MEAN_RANK=1.20\n" +
		"EOS_MIN_TRAIN_PAIRS_PER_SEC=100000\n" +
		"EOS_MAX_MATMUL_RUNS=2000\n"
	if err := os.WriteFile(thresholdsPath, []byte(thresholds), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	for _, want := range []string{
		"metrics: " + metricsPath,
		"thresholds: " + thresholdsPath,
		"scope: all",
		"pass: mrr=0.95 >= 0.9",
		"pass: train_pairs/s=120000 >= 100000",
		"pass: matmul_runs=1000 <= 2000",
		"gate: PASS checks=5",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsChecksEvalOnlyOptimizerUpdates(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "eval.metrics.json")
	metrics := trainMetricsJSON{
		Schema:       "manta.embedding_train_metrics.v1",
		Command:      "train-embed",
		Mode:         "eval",
		Artifact:     "current.mll",
		ProfileDelta: trainProfileDeltaJSON{OptimizerUpdates: 0},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)

	output := captureRunOutput(t, []string{"gate-train-metrics", "--scope", "eval-only", metricsPath})
	for _, want := range []string{
		"scope: eval-only",
		"pass: optimizer_updates=0 == 0 (eval-only)",
		"gate: PASS checks=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("eval-only gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateTrainMetricsFailsMissedThreshold(t *testing.T) {
	clearTrainMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "current.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := trainMetricsJSON{
		Schema:   "manta.embedding_train_metrics.v1",
		Command:  "train-embed",
		Mode:     "eval",
		Artifact: "current.mll",
		FinalEval: &evalMetricsJSON{
			MeanReciprocalRank: 0.5,
		},
	}
	writeMetricsJSONForTest(t, metricsPath, metrics)
	if err := os.WriteFile(thresholdsPath, []byte("EOS_MIN_MRR=0.90\n"), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output, err := captureRunOutputAndError(t, []string{"gate-train-metrics", "--thresholds", thresholdsPath, metricsPath})
	if err == nil {
		t.Fatalf("gate unexpectedly passed\noutput:\n%s", output)
	}
	for _, want := range []string{
		"fail: mrr=0.5 >= 0.9",
		"gate: FAIL checks=1 failed=1",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("failed gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateRetrievalMetricsChecksDatasetThresholds(t *testing.T) {
	clearRetrievalMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "scifact.retrieval.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "scifact",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10:    0.23,
			MRRAt10:     0.22,
			RecallAt10:  0.32,
			RecallAt100: 0.60,
		},
		Throughput: eosruntime.RetrievalEvalThroughput{
			ScoresPerSecond: 8000000,
		},
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal retrieval metrics: %v", err)
	}
	if err := os.WriteFile(metricsPath, data, 0o644); err != nil {
		t.Fatalf("write retrieval metrics: %v", err)
	}
	thresholds := "" +
		"EOS_MIN_RETRIEVAL_NDCG10_SCIFACT=0.22843\n" +
		"EOS_MIN_RETRIEVAL_MRR10_SCIFACT=0.213567\n" +
		"EOS_MIN_RETRIEVAL_SCORES_PER_SEC=7000000\n"
	if err := os.WriteFile(thresholdsPath, []byte(thresholds), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-retrieval-metrics", "--thresholds", thresholdsPath, metricsPath})
	for _, want := range []string{
		"dataset: scifact",
		"pass: ndcg_at_10=0.23 >= 0.22843",
		"pass: mrr_at_10=0.22 >= 0.213567",
		"pass: scores/s=8e+06 >= 7e+06",
		"retrieval gate: PASS checks=3",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("retrieval gate output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunGateRetrievalMetricsAllowsRoundedEquality(t *testing.T) {
	clearRetrievalMetricGateEnv(t)
	dir := t.TempDir()
	metricsPath := filepath.Join(dir, "scifact.retrieval.metrics.json")
	thresholdsPath := filepath.Join(dir, "thresholds.env")
	metrics := eosruntime.RetrievalEvalMetrics{
		Schema:  eosruntime.RetrievalEvalMetricsSchema,
		Dataset: "scifact",
		Quality: eosruntime.RetrievalEvalQualityMetrics{
			NDCGAt10: 0.22842998825189667,
		},
	}
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal retrieval metrics: %v", err)
	}
	if err := os.WriteFile(metricsPath, data, 0o644); err != nil {
		t.Fatalf("write retrieval metrics: %v", err)
	}
	if err := os.WriteFile(thresholdsPath, []byte("EOS_MIN_RETRIEVAL_NDCG10_SCIFACT=0.228430\n"), 0o644); err != nil {
		t.Fatalf("write thresholds: %v", err)
	}

	output := captureRunOutput(t, []string{"gate-retrieval-metrics", "--thresholds", thresholdsPath, metricsPath})
	if !strings.Contains(output, "retrieval gate: PASS checks=1") {
		t.Fatalf("rounded equality gate did not pass\noutput:\n%s", output)
	}
}

func clearTrainMetricGateEnv(t *testing.T) {
	t.Helper()
	for _, threshold := range trainMetricThresholds {
		t.Setenv(threshold.Env, "")
	}
}

func clearRetrievalMetricGateEnv(t *testing.T) {
	t.Helper()
	for _, threshold := range retrievalMetricThresholds {
		t.Setenv(threshold.Env, "")
		t.Setenv(threshold.Env+"_SCIFACT", "")
	}
}

func writeScoreboardForTest(t *testing.T, path string, rows []retrievalScoreboardRow) {
	t.Helper()
	data, err := json.Marshal(retrievalScoreboard{
		Schema: "manta.embedder_scoreboard.v1",
		Rows:   rows,
	})
	if err != nil {
		t.Fatalf("marshal scoreboard: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write scoreboard: %v", err)
	}
}

func writeMetricsJSONForTest(t *testing.T, path string, metrics trainMetricsJSON) {
	t.Helper()
	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("marshal metrics json: %v", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write metrics json: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	examples := []eosruntime.EmbeddingTextContrastiveExample{
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
		{Query: "ab", Positive: "ab"},
		{Query: "cd", Positive: "cd"},
	}
	if err := eosruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingTextContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextContrastivePackageWithLabeledEvalPairs(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-text.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-text.jsonl")
	trainData := "" +
		"{\"query\":\"ab\",\"positive\":\"ab\"}\n" +
		"{\"query\":\"cd\",\"positive\":\"cd\"}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text dataset: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed text with labeled eval: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextPairwisePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-pairs.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	trainData := "" +
		"{\"source\":\"scifact\",\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"source\":\"scifact\",\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n" +
		"{\"source\":\"nfcorpus\",\"query\":\"cd\",\"document\":\"cd\",\"label\":1}\n" +
		"{\"source\":\"nfcorpus\",\"query\":\"cd\",\"document\":\"ab\",\"label\":-1}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text pairs: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--pairwise-train", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"trained package",
		"train=4 pairwise examples",
		"eval=2 pairwise examples",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("pairwise train output missing %q\noutput:\n%s", want, output)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedFitsTextHardNegativePackage(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-pairs.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	trainData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n" +
		"{\"query\":\"cd\",\"document\":\"cd\",\"label\":1}\n" +
		"{\"query\":\"cd\",\"document\":\"ab\",\"label\":-1}\n"
	evalData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write train text pairs: %v", err)
	}
	if err := os.WriteFile(evalPath, []byte(evalData), 0o644); err != nil {
		t.Fatalf("write eval text pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1", "--hard-negative-source-weights", "scifact=1,nfcorpus=2", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"trained package",
		"train=2 hard_negative_contrastive examples",
		"eval=2 pairwise examples",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("hard-negative train output missing %q\noutput:\n%s", want, output)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainEmbedCompactHardNegativeMetricsProfileOptimizerUpdates(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "compact.mll")
	if err := run([]string{
		"init-model",
		"--architecture", eosruntime.EmbeddingArchitectureCompactTransformerV1,
		"--vocab-size", "32",
		"--max-seq", "12",
		"--model-dim", "8",
		"--output-dim", "4",
		"--hidden-dim", "16",
		"--attention-heads", "1",
		"--encoder-repeats", "1",
		"--contrastive-loss", "grouped_infonce",
		"--temperature", "0.05",
		path,
	}); err != nil {
		t.Fatalf("run init-model compact: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	if err := os.WriteFile(corpusPath, []byte("alpha beta positive negative gamma delta\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "32", "--min-freq", "1", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	trainPath := filepath.Join(dir, "hard-negatives.jsonl")
	trainData := "" +
		"{\"query\":\"alpha beta\",\"positive\":\"alpha beta positive\",\"negatives\":[\"gamma delta negative\"]}\n" +
		"{\"query\":\"gamma delta\",\"positive\":\"gamma delta positive\",\"negatives\":[\"alpha beta negative\"]}\n"
	if err := os.WriteFile(trainPath, []byte(trainData), 0o644); err != nil {
		t.Fatalf("write hard-negative train data: %v", err)
	}

	evalMetricsPath := filepath.Join(dir, "eval.metrics.json")
	if err := run([]string{"train-embed", "--eval-only", "--hard-negative-train", "--hard-negatives-per-query", "1", "--metrics-json", evalMetricsPath, path, trainPath}); err != nil {
		t.Fatalf("run compact eval-only hard-negative train-embed: %v", err)
	}
	evalMetrics := readTrainMetricsProfileForTest(t, evalMetricsPath)
	if evalMetrics.Summary.StepsRun != 0 || evalMetrics.ProfileDelta.OptimizerUpdates != 0 {
		t.Fatalf("compact eval-only metrics steps=%d optimizer_updates=%d, want zero updates", evalMetrics.Summary.StepsRun, evalMetrics.ProfileDelta.OptimizerUpdates)
	}

	trainMetricsPath := filepath.Join(dir, "train.metrics.json")
	if err := run([]string{"train-embed", "--hard-negative-train", "--hard-negatives-per-query", "1", "--epochs", "1", "--batch-size", "2", "--shuffle=false", "--contrastive-loss", "grouped_infonce", "--temperature", "0.05", "--metrics-json", trainMetricsPath, path, trainPath}); err != nil {
		t.Fatalf("run compact hard-negative train-embed: %v", err)
	}
	trainMetrics := readTrainMetricsProfileForTest(t, trainMetricsPath)
	if trainMetrics.Summary.StepsRun != 1 {
		t.Fatalf("compact train steps_run = %d, want 1", trainMetrics.Summary.StepsRun)
	}
	if trainMetrics.ProfileDelta.OptimizerUpdates <= 0 {
		t.Fatalf("compact train optimizer_updates = %d, want positive", trainMetrics.ProfileDelta.OptimizerUpdates)
	}
}

func TestRunTrainEmbedPlanOnlyCountsGroupedTextHardNegativeEvalPairs(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-hard.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-hard.jsonl")
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(trainPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Query: "ab", Positive: "ab", Negatives: []string{"cd", "a"}},
	}); err != nil {
		t.Fatalf("write grouped train hard negatives: %v", err)
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(evalPath, []eosruntime.EmbeddingTextHardNegativeExample{
		{Query: "ab", Positive: "ab", Negatives: []string{"cd", "a"}},
	}); err != nil {
		t.Fatalf("write grouped eval hard negatives: %v", err)
	}

	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--tokenizer", tokenizerPath, "--hard-negative-train", "--hard-negatives-per-query", "2", "--epochs", "1", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"train=1 hard_negative_contrastive examples",
		"eval=4 pairwise examples",
		"eval_pairs/pass=4",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("hard-negative plan-only output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedRejectsScoreSpectrumMutualExclusion(t *testing.T) {
	path := writeTrainableArtifact(t)
	trainPath := filepath.Join(t.TempDir(), "train-score-spectrum.jsonl")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(false)); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	for _, args := range [][]string{
		{"train-embed", "--score-spectrum-train", "--pairwise-train", "--no-tokenizer", path, trainPath},
		{"train-embed", "--score-spectrum-train", "--hard-negative-train", "--no-tokenizer", path, trainPath},
	} {
		_, err := captureRunOutputAndError(t, args)
		if err == nil || !strings.Contains(err.Error(), "--score-spectrum-train") {
			t.Fatalf("args %v error = %v, want score-spectrum mutual exclusion", args, err)
		}
	}
	_, err := captureRunOutputAndError(t, []string{"train-embed", "--allow-research-only-score-spectrum", "--no-tokenizer", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "requires --score-spectrum-train") {
		t.Fatalf("allow research without score-spectrum error = %v", err)
	}
}

func TestRunTrainEmbedListwiseGeometryPlanOnlyAndValidation(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[UNK]", "a", "b"},
	}
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := writeTinyCLIListwiseGeometryJSONL(t, false)
	evalPath := writeTinyCLIListwiseGeometryJSONL(t, false)

	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--epochs", "2", "--batch-size", "1", path, trainPath})
	for _, want := range []string{
		"train=1 listwise_geometry examples",
		"train_pairs/epoch=4",
		"pairs(planned=8 actual=0)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("listwise plan output missing %q\noutput:\n%s", want, output)
		}
	}

	output = captureRunOutput(t, []string{"train-embed", "--plan-only", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--epochs", "2", "--batch-size", "1", path, trainPath, evalPath})
	for _, want := range []string{
		"train=1 listwise_geometry examples",
		"eval=2 listwise_geometry examples",
		"eval_pairs/pass=4",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("listwise train+eval plan output missing %q\noutput:\n%s", want, output)
		}
	}

	output = captureRunOutput(t, []string{"train-embed", "--plan-only", "--eval-only", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--batch-size", "1", path, trainPath})
	for _, want := range []string{
		"train=0 listwise_geometry examples",
		"eval=2 listwise_geometry examples",
		"eval_pairs/pass=4",
		"pairs(planned=4 actual=0)",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("listwise eval-only plan output missing %q\noutput:\n%s", want, output)
		}
	}

	_, err := captureRunOutputAndError(t, []string{"train-embed", "--listwise-geometry-train", "--no-tokenizer", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "requires text tokenization") {
		t.Fatalf("no-tokenizer listwise error = %v, want text tokenization rejection", err)
	}
	_, err = captureRunOutputAndError(t, []string{"train-embed", "--allow-research-only-listwise-geometry", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "requires --listwise-geometry-train") {
		t.Fatalf("allow research without listwise error = %v", err)
	}
	_, err = captureRunOutputAndError(t, []string{"train-embed", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--allow-research-only-listwise-geometry", "--epochs", "1", "--batch-size", "1", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "must be explicitly research-only") {
		t.Fatalf("allow research with non-research row error = %v, want strict listwise rejection", err)
	}
}

func TestRunTrainEmbedListwiseGeometryResearchOnlyGate(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[UNK]", "a", "b"},
	}
	tokenizerPath := filepath.Join(t.TempDir(), "tokenizer.mll")
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	trainPath := writeTinyCLIListwiseGeometryJSONL(t, true)

	_, err := captureRunOutputAndError(t, []string{"train-embed", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--epochs", "1", "--batch-size", "1", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "research-only") {
		t.Fatalf("research-only without flag error = %v", err)
	}
	defaultMetricsPath := filepath.Join(t.TempDir(), "default.metrics.json")
	output := captureRunOutput(t, []string{"train-embed", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--allow-research-only-listwise-geometry", "--epochs", "1", "--batch-size", "1", "--metrics-json", defaultMetricsPath, path, trainPath})
	if !strings.Contains(output, "trained package") || !strings.Contains(output, "train=1 listwise_geometry examples") {
		t.Fatalf("listwise train output unexpected:\n%s", output)
	}
	var defaultMetrics struct {
		Config struct {
			MovementDiagnostics bool `json:"movement_diagnostics"`
		} `json:"config"`
		FinalTrain struct {
			Movement *trainMovementMetricsJSON `json:"movement,omitempty"`
		} `json:"final_train"`
	}
	data, err := os.ReadFile(defaultMetricsPath)
	if err != nil {
		t.Fatalf("read default metrics: %v", err)
	}
	if err := json.Unmarshal(data, &defaultMetrics); err != nil {
		t.Fatalf("decode default metrics: %v\n%s", err, data)
	}
	if defaultMetrics.Config.MovementDiagnostics {
		t.Fatalf("default movement diagnostics config = true, want false")
	}
	if defaultMetrics.FinalTrain.Movement != nil {
		t.Fatalf("default movement metrics = %+v, want omitted", defaultMetrics.FinalTrain.Movement)
	}

	diagnosticMetricsPath := filepath.Join(t.TempDir(), "diagnostic.metrics.json")
	output = captureRunOutput(t, []string{"train-embed", "--tokenizer", tokenizerPath, "--listwise-geometry-train", "--allow-research-only-listwise-geometry", "--movement-diagnostics", "--epochs", "1", "--batch-size", "1", "--metrics-json", diagnosticMetricsPath, path, trainPath})
	if !strings.Contains(output, "metrics: "+diagnosticMetricsPath) {
		t.Fatalf("diagnostic listwise output missing metrics path:\n%s", output)
	}
	var diagnosticMetrics struct {
		Config struct {
			MovementDiagnostics bool `json:"movement_diagnostics"`
		} `json:"config"`
		FinalTrain struct {
			Movement *trainMovementMetricsJSON `json:"movement,omitempty"`
		} `json:"final_train"`
	}
	data, err = os.ReadFile(diagnosticMetricsPath)
	if err != nil {
		t.Fatalf("read diagnostic metrics: %v", err)
	}
	if err := json.Unmarshal(data, &diagnosticMetrics); err != nil {
		t.Fatalf("decode diagnostic metrics: %v\n%s", err, data)
	}
	if !diagnosticMetrics.Config.MovementDiagnostics {
		t.Fatalf("diagnostic movement diagnostics config = false, want true")
	}
	if diagnosticMetrics.FinalTrain.Movement == nil || diagnosticMetrics.FinalTrain.Movement.Gradient.NonzeroCount <= 0 || diagnosticMetrics.FinalTrain.Movement.ParameterDelta.NonzeroCount <= 0 {
		t.Fatalf("diagnostic movement metrics = %+v, want nonzero counts", diagnosticMetrics.FinalTrain.Movement)
	}

	manifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(path))
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if !manifest.ListwiseGeometry.ListwiseGeometryResearchOnly || manifest.ListwiseGeometry.ListwiseGeometryBatchCount != 1 {
		t.Fatalf("listwise package policy = %+v, want research-only batch count 1", manifest.ListwiseGeometry)
	}
}

func TestRunTrainEmbedRejectsExplicitScoreSpectrumTurboQuantCompactObjectives(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-score-spectrum.jsonl")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(false)); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}

	_, err := captureRunOutputAndError(t, []string{"train-embed", "--score-spectrum-train", "--no-tokenizer", "--epochs", "1", "--batch-size", "2", "--turboquant-compact-objectives", "2:2=0.25", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "does not support TurboQuant compact objectives") {
		t.Fatalf("explicit compact objective error = %v, want score-spectrum rejection", err)
	}
}

func TestRunTrainEmbedScoreSpectrumResearchOnlyGate(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train-score-spectrum.jsonl")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(true), eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: true}); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	_, err := captureRunOutputAndError(t, []string{"train-embed", "--score-spectrum-train", "--no-tokenizer", "--epochs", "1", "--batch-size", "2", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "research-only") {
		t.Fatalf("research-only without flag error = %v", err)
	}
	output := captureRunOutput(t, []string{"train-embed", "--score-spectrum-train", "--allow-research-only-score-spectrum", "--no-tokenizer", "--epochs", "1", "--batch-size", "2", path, trainPath})
	if !strings.Contains(output, "trained package") || !strings.Contains(output, "train=2 score_spectrum_grouped examples") {
		t.Fatalf("score-spectrum train output unexpected:\n%s", output)
	}
	manifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(path))
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	if !manifest.ScoreSpectrum.ScoreSpectrumResearchOnly || manifest.ScoreSpectrum.ScoreSpectrumRowCount != 2 {
		t.Fatalf("score-spectrum package policy = %+v, want research-only row count 2", manifest.ScoreSpectrum)
	}
}

func TestRunTrainEmbedPlanOnlyScoreSpectrumWorkload(t *testing.T) {
	path := writeTrainableArtifact(t)
	trainPath := filepath.Join(t.TempDir(), "train-score-spectrum.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval-pairs.jsonl")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(false)); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingPairExamplesFile(evalPath, []eosruntime.EmbeddingPairExample{
		{LeftTokens: []int32{1}, LeftMask: []int32{1}, RightTokens: []int32{1}, RightMask: []int32{1}, Target: 1},
	}); err != nil {
		t.Fatalf("write eval pairs: %v", err)
	}
	output := captureRunOutput(t, []string{"train-embed", "--plan-only", "--score-spectrum-train", "--no-tokenizer", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath})
	for _, want := range []string{
		"train=2 score_spectrum_grouped examples",
		"train_pairs/epoch=4",
		"eval=1 pairwise examples",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("score-spectrum plan output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunTrainEmbedScoreSpectrumNativeEvalAndRecoveryFlags(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	dir := t.TempDir()
	trainPath := filepath.Join(dir, "train-score-spectrum.jsonl")
	evalPath := filepath.Join(dir, "eval-score-spectrum.jsonl")
	metricsPath := filepath.Join(dir, "metrics.json")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(false)); err != nil {
		t.Fatalf("write train score-spectrum dataset: %v", err)
	}
	evalExamples := tinyCLIScoreSpectrumExamples(false)
	selected := 0
	evalExamples[0].SelectedPositiveIndex = &selected
	evalExamples[0].PositiveIndexes = nil
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(evalPath, evalExamples); err != nil {
		t.Fatalf("write eval score-spectrum dataset: %v", err)
	}

	output := captureRunOutput(t, []string{
		"train-embed",
		"--score-spectrum-train",
		"--score-spectrum-eval", evalPath,
		"--score-spectrum-loss-mode", "hard_soft_recovery",
		"--score-spectrum-recovery-weight", "1.25",
		"--score-spectrum-recovery-margin", "0.05",
		"--score-spectrum-recovery-top-k", "1",
		"--score-spectrum-recovery-tau", "0.05",
		"--select-metric", "score_spectrum_any_positive_top1",
		"--no-tokenizer",
		"--epochs", "1",
		"--batch-size", "2",
		"--metrics-json", metricsPath,
		path,
		trainPath,
	})
	if !strings.Contains(output, "final score-spectrum eval:") {
		t.Fatalf("score-spectrum native eval output missing final eval:\n%s", output)
	}
	var got struct {
		Config struct {
			ScoreSpectrumLossMode       string  `json:"score_spectrum_loss_mode"`
			ScoreSpectrumRecoveryWeight float32 `json:"score_spectrum_recovery_weight"`
			ScoreSpectrumRecoveryTopK   int     `json:"score_spectrum_recovery_top_k"`
		} `json:"config"`
		FinalScoreSpectrumEval *struct {
			RowCount                   int `json:"row_count"`
			CandidateCount             int `json:"candidate_count"`
			TargetDistributionRowCount int `json:"target_distribution_row_count"`
		} `json:"final_score_spectrum_eval"`
		BestScoreSpectrumEval *struct {
			RowCount int `json:"row_count"`
		} `json:"best_score_spectrum_eval"`
	}
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode metrics: %v\n%s", err, data)
	}
	if got.Config.ScoreSpectrumLossMode != "hard_soft_recovery" || got.Config.ScoreSpectrumRecoveryWeight != 1.25 || got.Config.ScoreSpectrumRecoveryTopK != 1 {
		t.Fatalf("score-spectrum config JSON = %+v, want recovery flags", got.Config)
	}
	if got.FinalScoreSpectrumEval == nil || got.FinalScoreSpectrumEval.RowCount != 2 || got.FinalScoreSpectrumEval.CandidateCount != 4 {
		t.Fatalf("final score-spectrum eval JSON = %+v, want 2 rows/4 candidates", got.FinalScoreSpectrumEval)
	}
	if got.FinalScoreSpectrumEval.TargetDistributionRowCount != 2 {
		t.Fatalf("target distribution row count = %d, want 2", got.FinalScoreSpectrumEval.TargetDistributionRowCount)
	}
	if got.BestScoreSpectrumEval == nil || got.BestScoreSpectrumEval.RowCount != 2 {
		t.Fatalf("best score-spectrum eval JSON = %+v, want row_count=2", got.BestScoreSpectrumEval)
	}
}

func TestRunTrainEmbedRejectsInvalidScoreSpectrumRecoveryFlags(t *testing.T) {
	path := writeTrainableArtifact(t)
	trainPath := filepath.Join(t.TempDir(), "train-score-spectrum.jsonl")
	if err := eosruntime.WriteEmbeddingScoreSpectrumExamplesFile(trainPath, tinyCLIScoreSpectrumExamples(false)); err != nil {
		t.Fatalf("write score-spectrum dataset: %v", err)
	}
	_, err := captureRunOutputAndError(t, []string{"train-embed", "--score-spectrum-train", "--score-spectrum-loss-mode", "recovery", "--score-spectrum-recovery-top-k", "-1", "--no-tokenizer", path, trainPath})
	if err == nil || !strings.Contains(err.Error(), "score-spectrum-recovery-top-k") {
		t.Fatalf("invalid recovery top-k error = %v, want rejection", err)
	}
}

func tinyCLIScoreSpectrumExamples(researchOnly bool) []eosruntime.EmbeddingScoreSpectrumExample {
	examples := []eosruntime.EmbeddingScoreSpectrumExample{
		{
			RowID:                   "row-a",
			QueryTokens:             []int32{1},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{1}, {2}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			ReleaseTrainAllowed:     true,
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
			SourceArtifactHash:      "hash-a",
		},
		{
			RowID:                   "row-b",
			QueryTokens:             []int32{2},
			QueryMask:               []int32{1},
			CandidateTokens:         [][]int32{{2}, {1}},
			CandidateMasks:          [][]int32{{1}, {1}},
			PositiveIndexes:         []int{0},
			HardNegativeEligible:    []bool{false, true},
			TargetProbabilities:     []float32{1, 0},
			ReleaseTrainAllowed:     true,
			CommercialUseAllowed:    true,
			TrainAllowedForResearch: false,
			SourceArtifactHash:      "hash-b",
		},
	}
	if researchOnly {
		for i := range examples {
			examples[i].ReleaseTrainAllowed = false
			examples[i].CommercialUseAllowed = false
			examples[i].TrainAllowedForResearch = true
		}
	}
	return examples
}

func writeTinyCLIListwiseGeometryJSONL(t *testing.T, researchOnly bool) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "train-listwise-geometry.jsonl")
	release := "true"
	commercial := "true"
	if researchOnly {
		release = "false"
		commercial = "false"
	}
	data := fmt.Sprintf(`{"schema":"eos.listwise_geometry_batch.v1","batch_id":"batch-0","examples":[{"row_id":"r0","source":"unit","query_id":"q0","positive_doc_id":"d0","negative_doc_ids":["d1"]},{"row_id":"r1","source":"unit","query_id":"q1","positive_doc_id":"d1","negative_doc_ids":["d0"]}],"queries":[{"id":"q0","text":"a"},{"id":"q1","text":"b"}],"documents":[{"id":"d0","text":"a"},{"id":"d1","text":"b"}],"teacher_similarity":[[0.9,0.1],[0.2,0.8]],"score":"cosine","train_allowed_for_research":%t,"release_train_allowed":%s,"commercial_use_allowed":%s}`+"\n", researchOnly, release, commercial)
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatalf("write listwise geometry dataset: %v", err)
	}
	return path
}

func TestRunTokenizeEmbedHardNegativeMode(t *testing.T) {
	path := writeTrainableArtifact(t)
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	inputPath := filepath.Join(t.TempDir(), "pairs.jsonl")
	outputPath := filepath.Join(t.TempDir(), "hard.tokens.jsonl")
	inputData := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"query\":\"ab\",\"document\":\"cd\",\"label\":-1}\n"
	if err := os.WriteFile(inputPath, []byte(inputData), 0o644); err != nil {
		t.Fatalf("write input pairs: %v", err)
	}

	output := captureRunOutput(t, []string{"tokenize-embed", "--mode", "hard-negative", path, inputPath, outputPath})
	if !strings.Contains(output, "tokenized hard-negative examples: 1") {
		t.Fatalf("tokenize hard-negative output unexpected:\n%s", output)
	}
	examples, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(outputPath)
	if err != nil {
		t.Fatalf("read tokenized hard-negative output: %v", err)
	}
	if len(examples) != 1 || len(examples[0].NegativeTokens) != 1 {
		t.Fatalf("tokenized examples = %+v", examples)
	}
}

func TestRunInspectReadsPackageManifest(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"inspect", path}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
}

func TestRunInspectShowsRepeatedEncoderEmbeddingDetails(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.eos")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")
	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	output := captureRunOutput(t, []string{"inspect", artifactPath})
	for _, want := range []string{
		"embedding manifest:",
		"embedding model: encoder-trainable-q8x2 pooled=embed_pooled batch=embed_pooled_batch output=result/f16",
		"encoder repeats: 2",
		"tokenizer: vocab=32768 max_sequence=256",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("inspect output missing %q\noutput:\n%s", want, output)
		}
	}
}

func TestRunExportMLLWritesContainer(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	if err := run([]string{"export-mll", path}); err != nil {
		t.Fatalf("run export-mll: %v", err)
	}
	outPath := eosruntime.DefaultMLLPath(path)
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("expected MLL file %q: %v", outPath, err)
	}
	if _, err := mll.ReadFile(outPath, mll.WithDigestVerification()); err != nil {
		t.Fatalf("read exported MLL: %v", err)
	}
}

func TestRunTrainTokenizerWritesSiblingTokenizer(t *testing.T) {
	path := writeTrainableArtifact(t)
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(corpusPath, []byte("ab ab cd ab cd\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "12", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	tokenizerPath := eosruntime.DefaultTokenizerPath(path)
	if _, err := os.Stat(tokenizerPath); err != nil {
		t.Fatalf("expected tokenizer file %q: %v", tokenizerPath, err)
	}
	if _, err := eosruntime.ReadTokenizerFile(tokenizerPath); err != nil {
		t.Fatalf("read tokenizer file: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Tokenizer.VocabSize != 12 {
		t.Fatalf("expected manifest vocab size to preserve requested contract, got %d", manifest.Tokenizer.VocabSize)
	}
}

func TestRunTrainTokenizerPreservesCompactPackageVocabContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.mll")
	if err := run([]string{
		"init-model",
		"--architecture", eosruntime.EmbeddingArchitectureCompactTransformerV1,
		"--vocab-size", "32",
		"--max-seq", "12",
		"--model-dim", "8",
		"--output-dim", "4",
		"--hidden-dim", "16",
		"--attention-heads", "1",
		"--encoder-repeats", "1",
		path,
	}); err != nil {
		t.Fatalf("run init-model compact: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(corpusPath, []byte("alpha beta alpha beta gamma\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "32", "--min-freq", "1", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	tokenizer, err := eosruntime.ReadTokenizerFile(eosruntime.DefaultTokenizerPath(path))
	if err != nil {
		t.Fatalf("read tokenizer: %v", err)
	}
	if len(tokenizer.Tokens) != 32 {
		t.Fatalf("tokenizer tokens = %d, want compact package vocab contract 32", len(tokenizer.Tokens))
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(path))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.Tokenizer.VocabSize != 32 {
		t.Fatalf("manifest vocab size = %d, want 32", manifest.Tokenizer.VocabSize)
	}
	state, err := eosruntime.LoadCompactEmbeddingTrainStateFromPackage(path)
	if err != nil {
		t.Fatalf("load compact train state after tokenizer sync: %v", err)
	}
	if got := state.TokenEmbedding.Tensor.Shape[0]; got != 32 {
		t.Fatalf("compact token embedding rows = %d, want 32", got)
	}
}

func TestRunTokenizeEmbedWritesTokenDatasets(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	tokenizer := eosruntime.TokenizerFile{
		Version: eosruntime.TokenizerFileVersion,
		Tokens:  []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"},
	}
	if err := tokenizer.WriteFile(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}

	dir := t.TempDir()
	trainTextPath := filepath.Join(dir, "train-text.jsonl")
	evalTextPath := filepath.Join(dir, "eval-text.jsonl")
	trainTokenPath := filepath.Join(dir, "train-token.jsonl")
	evalTokenPath := filepath.Join(dir, "eval-token.jsonl")
	if err := os.WriteFile(trainTextPath, []byte("{\"query\":\"ab\",\"positive\":\"ab\"}\n"), 0o644); err != nil {
		t.Fatalf("write train text: %v", err)
	}
	evalText := "" +
		"{\"query\":\"ab\",\"document\":\"ab\",\"label\":1}\n" +
		"{\"left\":\"ab\",\"right\":\"cd\",\"label\":0}\n"
	if err := os.WriteFile(evalTextPath, []byte(evalText), 0o644); err != nil {
		t.Fatalf("write eval text: %v", err)
	}

	if err := run([]string{"tokenize-embed", "--mode", "contrastive", path, trainTextPath, trainTokenPath}); err != nil {
		t.Fatalf("run tokenize contrastive: %v", err)
	}
	if err := run([]string{"tokenize-embed", "--mode", "pair", path, evalTextPath, evalTokenPath}); err != nil {
		t.Fatalf("run tokenize pair: %v", err)
	}
	if _, err := eosruntime.ReadEmbeddingContrastiveExamplesFile(trainTokenPath); err != nil {
		t.Fatalf("read tokenized train: %v", err)
	}
	pairs, err := eosruntime.ReadEmbeddingPairExamplesFile(evalTokenPath)
	if err != nil {
		t.Fatalf("read tokenized eval: %v", err)
	}
	if len(pairs) != 2 || pairs[0].Target != 1 || pairs[1].Target != 0 {
		t.Fatalf("tokenized eval targets = %+v", pairs)
	}
	output := captureRunOutput(t, []string{"train-embed", "--eval-only", "--no-tokenizer", path, evalTokenPath})
	if !strings.Contains(output, "evaluated package") || !strings.Contains(output, "eval=2 pairwise examples") {
		t.Fatalf("eval-only token output missing expected summary:\n%s", output)
	}
}

func TestRunMineTextPairsWritesTrainAndEvalFiles(t *testing.T) {
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "5", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if _, err := eosruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath); err != nil {
		t.Fatalf("read mined train pairs: %v", err)
	}
	evalSet, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
	if err != nil {
		t.Fatalf("read mined eval pairs: %v", err)
	}
	var positives, negatives int
	for _, example := range evalSet {
		if example.Target > 0 {
			positives++
		} else if example.Target == 0 {
			negatives++
		}
	}
	if positives == 0 || negatives == 0 {
		t.Fatalf("expected mined eval set to include both classes, got positives=%d negatives=%d", positives, negatives)
	}
}

func TestRunMineTextPairsThenTrainEmbedFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"ab ab cd. cd ab cd.\n" +
		"cd cd ab. ab cd ab.\n" +
		"ab cd ef. ef cd ab.\n" +
		"ef ef ab. ab ef ef.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-tokenizer", "--vocab-size", "16", path, corpusPath}); err != nil {
		t.Fatalf("run train-tokenizer: %v", err)
	}
	trainPath := filepath.Join(t.TempDir(), "train.jsonl")
	evalPath := filepath.Join(t.TempDir(), "eval.jsonl")
	if err := run([]string{"mine-text-pairs", "--min-chars", "2", "--eval-pairs", "2", corpusPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run mine-text-pairs: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", path, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed from mined text: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusFlow(t *testing.T) {
	path := writeTrainableArtifact(t)
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", path}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "" +
		"alpha beta gamma. gamma delta epsilon.\n" +
		"delta epsilon zeta. eta theta iota.\n" +
		"kappa lambda mu. nu xi omicron.\n" +
		"pi rho sigma. tau upsilon phi.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "20", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "5", "--eval-pairs", "2", path, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultTokenizerPath(path)); err != nil {
		t.Fatalf("expected tokenizer file: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultMinedTrainPairsPath(path)); err != nil {
		t.Fatalf("expected mined train pairs: %v", err)
	}
	if _, err := os.Stat(eosruntime.DefaultMinedEvalPairsPath(path)); err != nil {
		t.Fatalf("expected mined eval pairs: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

func TestRunTrainCorpusRepeatedEncoderExampleFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := copyExampleFile(t, dir, "encoder_trainable_q8x2.eos")
	artifactPath := filepath.Join(dir, "encoder_trainable_q8x2.mll")
	copyExampleFile(t, dir, "encoder_trainable_q8x2.embedding.mll")

	if err := run([]string{"compile", srcPath, artifactPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=16", "--dim", "H=32", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	corpusPath := filepath.Join(dir, "corpus.txt")
	corpus := "" +
		"Eos trains and serves compact transformer encoders.\n" +
		"CorkScrewDB needs a small default model with strong retrieval quality.\n" +
		"Quantized embeddings should be fast, portable, and cheap to ship.\n" +
		"Native CUDA training should reuse weights, activations, and optimizer state.\n" +
		"Metal parity matters later, but the package path must already be clean.\n" +
		"Attention, residuals, and layernorm make the encoder more realistic.\n"
	if err := os.WriteFile(corpusPath, []byte(corpus), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	if err := run([]string{"train-corpus", "--vocab-size", "48", "--min-freq", "1", "--epochs", "2", "--batch-size", "2", "--min-chars", "12", "--eval-pairs", "3", artifactPath, corpusPath}); err != nil {
		t.Fatalf("run train-corpus: %v", err)
	}
	if err := run([]string{"inspect", artifactPath}); err != nil {
		t.Fatalf("run inspect: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.DefaultEmbeddingManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	profile, err := eosruntime.ReadEmbeddingTrainProfileFile(eosruntime.DefaultEmbeddingTrainProfilePath(artifactPath))
	if err != nil {
		t.Fatalf("read training profile: %v", err)
	}
	if profile.Step == 0 {
		t.Fatal("expected non-zero training profile step")
	}
	if profile.ForwardBackend != "" && profile.ForwardResidency.MatMul.BindCalls == 0 {
		t.Fatal("expected matmul bind activity in repeated encoder train profile")
	}
	for _, candidate := range []string{
		eosruntime.DefaultTokenizerPath(artifactPath),
		eosruntime.DefaultMinedTrainPairsPath(artifactPath),
		eosruntime.DefaultMinedEvalPairsPath(artifactPath),
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected generated training artifact %q: %v", candidate, err)
		}
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained repeated-encoder package: %v", err)
	}
}

func TestRunCompileDefaultMLLThenTrainFlow(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "tiny_train_embed.eos")
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	if err := os.WriteFile(srcPath, source, 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}
	if err := run([]string{"compile", srcPath}); err != nil {
		t.Fatalf("run compile: %v", err)
	}
	artifactPath := filepath.Join(dir, "tiny_train_embed.mll")
	if _, err := os.Stat(artifactPath); err != nil {
		t.Fatalf("expected compiled .mll artifact %q: %v", artifactPath, err)
	}
	manifest := eosruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: eosruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(eosruntime.DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	if err := run([]string{"init-train", "--dim", "D=4", "--dim", "E=3", artifactPath}); err != nil {
		t.Fatalf("run init-train: %v", err)
	}
	trainPath := filepath.Join(dir, "train.jsonl")
	evalPath := filepath.Join(dir, "eval.jsonl")
	examples := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{1, 2}, PositiveTokens: []int32{1, 2}},
		{QueryTokens: []int32{2, 3}, PositiveTokens: []int32{2, 3}},
		{QueryTokens: []int32{3, 4}, PositiveTokens: []int32{3, 4}},
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 5}},
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(trainPath, examples); err != nil {
		t.Fatalf("write train dataset: %v", err)
	}
	if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(evalPath, examples); err != nil {
		t.Fatalf("write eval dataset: %v", err)
	}
	if err := run([]string{"train-embed", "--epochs", "2", "--batch-size", "2", artifactPath, trainPath, evalPath}); err != nil {
		t.Fatalf("run train-embed: %v", err)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(artifactPath); err != nil {
		t.Fatalf("reload trained package: %v", err)
	}
}

type trainMetricsProfileForTest struct {
	Summary struct {
		StepsRun int `json:"steps_run"`
	} `json:"summary"`
	ProfileDelta struct {
		OptimizerUpdates int64 `json:"optimizer_updates"`
	} `json:"profile_delta"`
}

func readTrainMetricsProfileForTest(t *testing.T, path string) trainMetricsProfileForTest {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read train metrics %q: %v", path, err)
	}
	var got trainMetricsProfileForTest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("decode train metrics %q: %v\n%s", path, err, string(data))
	}
	return got
}

func writeTrainableArtifact(t *testing.T) string {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> q8[E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> q8[B, E] {
    let embeddings = gather(token_embedding, tokens)
    let projected = @matmul(embeddings, projection)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(t.TempDir(), "tiny_train_embed.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := eosruntime.EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: eosruntime.TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(eosruntime.DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

func copyExampleFile(t *testing.T, dir, name string) string {
	t.Helper()
	srcPath := filepath.Join("..", "..", "examples", name)
	data, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatalf("read example file %q: %v", srcPath, err)
	}
	dstPath := filepath.Join(dir, name)
	if err := os.WriteFile(dstPath, data, 0o644); err != nil {
		t.Fatalf("write example file %q: %v", dstPath, err)
	}
	return dstPath
}

func captureRunOutput(t *testing.T, args []string) string {
	t.Helper()
	output, runErr := captureRunOutputAndError(t, args)
	if runErr != nil {
		t.Fatalf("run %v: %v\noutput:\n%s", args, runErr, output)
	}
	return output
}

func captureRunOutputAndError(t *testing.T, args []string) (string, error) {
	t.Helper()
	origStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	os.Stdout = writer
	defer func() {
		os.Stdout = origStdout
	}()
	runErr := run(args)
	if err := writer.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	return string(data), runErr
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	origStdout := os.Stdout
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	os.Stdout = writer
	defer func() {
		os.Stdout = origStdout
	}()
	fn()
	if err := writer.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	if err := reader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	return string(data)
}

// captureRunStderrAndOutput runs args and returns both the stdout and stderr
// output alongside any error, for tests asserting on warning/log lines that
// train-embed writes to stderr (e.g. the retrieval-gated select-metric
// auto-upgrade log line and the restore-best pairwise selection warning).
func captureRunStderrAndOutput(t *testing.T, args []string) (stdout string, stderr string, runErr error) {
	t.Helper()
	origStdout := os.Stdout
	origStderr := os.Stderr
	outReader, outWriter, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stdout pipe: %v", err)
	}
	errReader, errWriter, err := os.Pipe()
	if err != nil {
		t.Fatalf("create stderr pipe: %v", err)
	}
	os.Stdout = outWriter
	os.Stderr = errWriter
	defer func() {
		os.Stdout = origStdout
		os.Stderr = origStderr
	}()
	runErr = run(args)
	if err := outWriter.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	if err := errWriter.Close(); err != nil {
		t.Fatalf("close stderr writer: %v", err)
	}
	outData, err := io.ReadAll(outReader)
	if err != nil {
		t.Fatalf("read stdout capture: %v", err)
	}
	errData, err := io.ReadAll(errReader)
	if err != nil {
		t.Fatalf("read stderr capture: %v", err)
	}
	if err := outReader.Close(); err != nil {
		t.Fatalf("close stdout reader: %v", err)
	}
	if err := errReader.Close(); err != nil {
		t.Fatalf("close stderr reader: %v", err)
	}
	return string(outData), string(errData), runErr
}

func commandTestSHA256File(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
