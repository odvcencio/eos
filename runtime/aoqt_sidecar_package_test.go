package eosruntime

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestWriteAOQTSidecarCandidatePackageLoadsAndBindsPolicy(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))

	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT candidate package: %v", err)
	}
	if result.Paths.PostPoolTransformPath != DefaultPostPoolTransformPath(result.Paths.ArtifactPath) {
		t.Fatalf("post-pool path = %q, want default", result.Paths.PostPoolTransformPath)
	}
	if !result.PackageManifest.HasFileRole(EmbeddingPostPoolTransformRole) {
		t.Fatal("candidate package manifest does not list post_pool_transform")
	}
	if !result.PackageManifest.AOQTTransform.Enabled || !result.PackageManifest.AOQTTransform.ResearchOnly || result.PackageManifest.AOQTTransform.ReleaseTrainAllowed || result.PackageManifest.AOQTTransform.CommercialUseAllowed || result.PackageManifest.AOQTTransform.FreeOpenReleaseAllowed || result.PackageManifest.AOQTTransform.QualityClaim {
		t.Fatalf("AOQT policy gates = %+v, want research-only/no release claims", result.PackageManifest.AOQTTransform)
	}
	if result.PackageManifest.AOQTTransform.AnchorArtifactSHA256 != cfg.ExpectedAnchorArtifactSHA256 || result.PackageManifest.AOQTTransform.AnchorPackageManifestSHA256 != cfg.ExpectedAnchorPackageManifestSHA256 {
		t.Fatalf("AOQT policy anchor hashes = %+v, want config hashes", result.PackageManifest.AOQTTransform)
	}
	loadedPackage, err := ReadPackageManifestFile(result.Paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read candidate package manifest: %v", err)
	}
	if loadedPackage.AOQTTransform.TransformSHA256 == "" || loadedPackage.AOQTTransform.PairingsSHA256 == "" || loadedPackage.AOQTTransform.AnglesSHA256 == "" {
		t.Fatalf("AOQT policy hashes missing after MLL round trip: %+v", loadedPackage.AOQTTransform)
	}
	if err := loadedPackage.VerifyFiles(map[string]string{
		"artifact":                     result.Paths.ArtifactPath,
		"embedding_manifest":           result.Paths.ManifestPath,
		"tokenizer":                    result.Paths.TokenizerPath,
		"weights":                      result.Paths.WeightFilePath,
		"memory_plan":                  result.Paths.MemoryPlanPath,
		EmbeddingPostPoolTransformRole: result.Paths.PostPoolTransformPath,
	}); err != nil {
		t.Fatalf("verify candidate package manifest: %v", err)
	}

	manifest, err := ReadEmbeddingManifestFile(result.Paths.ManifestPath)
	if err != nil {
		t.Fatalf("read candidate embedding manifest: %v", err)
	}
	if manifest.PostPoolTransform != EmbeddingPostPoolTransformAOQTGivens {
		t.Fatalf("post_pool_transform = %q, want %q", manifest.PostPoolTransform, EmbeddingPostPoolTransformAOQTGivens)
	}

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath)
	if err != nil {
		t.Fatalf("load AOQT candidate package: %v", err)
	}
	out, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed AOQT candidate: %v", err)
	}
	assertTensorClose(t, out.Embeddings, []int{2}, []float32{-1, 0})
}

func TestRebuildSiblingPackageManifestPreservesAOQTPolicyAndLoadGuards(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT candidate package: %v", err)
	}

	rebuilt, manifestPath, err := RebuildSiblingPackageManifest(result.Paths.ArtifactPath)
	if err != nil {
		t.Fatalf("rebuild AOQT package manifest: %v", err)
	}
	if manifestPath != result.Paths.PackageManifestPath {
		t.Fatalf("rebuilt manifest path = %q, want %q", manifestPath, result.Paths.PackageManifestPath)
	}
	if !rebuilt.HasFileRole(EmbeddingPostPoolTransformRole) {
		t.Fatal("rebuilt manifest does not list AOQT post-pool transform role")
	}
	if !reflect.DeepEqual(rebuilt.AOQTTransform, result.AOQTPolicy) {
		t.Fatalf("rebuilt AOQT policy = %+v, want %+v", rebuilt.AOQTTransform, result.AOQTPolicy)
	}
	if err := rebuilt.VerifyFiles(map[string]string{
		"artifact":                     result.Paths.ArtifactPath,
		"embedding_manifest":           result.Paths.ManifestPath,
		"tokenizer":                    result.Paths.TokenizerPath,
		"weights":                      result.Paths.WeightFilePath,
		"memory_plan":                  result.Paths.MemoryPlanPath,
		EmbeddingPostPoolTransformRole: result.Paths.PostPoolTransformPath,
	}); err != nil {
		t.Fatalf("verify rebuilt AOQT package manifest: %v", err)
	}
	transformSHA, _, err := fileHash(result.Paths.PostPoolTransformPath)
	if err != nil {
		t.Fatalf("hash rebuilt AOQT sidecar: %v", err)
	}
	if rebuilt.AOQTTransform.TransformSHA256 != transformSHA {
		t.Fatalf("rebuilt transform_sha256 = %q, want %q", rebuilt.AOQTTransform.TransformSHA256, transformSHA)
	}

	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err != nil {
		t.Fatalf("load rebuilt AOQT candidate package: %v", err)
	}
	if err := os.WriteFile(result.Paths.PostPoolTransformPath, []byte(`{"version":"tampered"}`+"\n"), 0o644); err != nil {
		t.Fatalf("tamper AOQT sidecar: %v", err)
	}
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("load tampered rebuilt AOQT sidecar error = %v, want sha256 mismatch", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsExistingOutputBase(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	out := filepath.Join(t.TempDir(), "candidate.mll")
	if err := os.WriteFile(out, []byte("occupied\n"), 0o644); err != nil {
		t.Fatalf("write occupied output: %v", err)
	}
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, out)
	if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("write candidate over existing output error = %v, want already exists", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsExistingAuthoredSibling(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	out := filepath.Join(t.TempDir(), "candidate.mll")
	if err := os.WriteFile(DefaultEmbeddingManifestPath(out), []byte("occupied\n"), 0o644); err != nil {
		t.Fatalf("write occupied embedding sibling: %v", err)
	}
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, out)
	if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("write candidate over existing sibling error = %v, want already exists", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsAnchorHashMismatch(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	cfg.ExpectedAnchorArtifactSHA256 = strings.Repeat("0", 64)
	if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), "anchor artifact sha256 mismatch") {
		t.Fatalf("write candidate with bad anchor hash error = %v, want mismatch", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsMandatoryProvenanceMissing(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	for _, tc := range []struct {
		name string
		edit func(*AOQTSidecarCandidatePackageConfig)
		want string
	}{
		{
			name: "dataset_manifest_sha256",
			edit: func(cfg *AOQTSidecarCandidatePackageConfig) { cfg.DatasetManifestSHA256 = "" },
			want: "dataset manifest sha256 is required",
		},
		{
			name: "compatibility_digest",
			edit: func(cfg *AOQTSidecarCandidatePackageConfig) { cfg.CompatibilityDigest = "" },
			want: "compatibility digest is required",
		},
		{
			name: "qrels_coverage",
			edit: func(cfg *AOQTSidecarCandidatePackageConfig) { cfg.QrelsSHA256ByDataset = nil },
			want: "qrels sha256 dataset coverage is required",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
			tc.edit(&cfg)
			if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("write candidate missing provenance error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsHardlinks(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	cfg.UseHardlinks = true
	if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), "hardlinks are not supported") {
		t.Fatalf("write candidate with hardlinks error = %v, want rejection", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageRejectsResearchOnlyAnchorEmbedding(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	manifest, err := ReadPackageManifestFile(DefaultPackageManifestPath(anchor))
	if err != nil {
		t.Fatalf("read anchor package manifest: %v", err)
	}
	manifest.ScoreSpectrum = EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:        true,
		ScoreSpectrumResearchOnly: true,
		TrainAllowedForResearch:   true,
		ReleaseTrainAllowed:       false,
		CommercialUseAllowed:      false,
		SourceArtifactHashes:      []string{strings.Repeat("a", 64)},
		ScoreSpectrumRowCount:     1,
	}
	if err := manifest.WriteFile(DefaultPackageManifestPath(anchor)); err != nil {
		t.Fatalf("rewrite research-only anchor manifest: %v", err)
	}
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	if _, err := WriteAOQTSidecarCandidatePackage(cfg); err == nil || !strings.Contains(err.Error(), "research-only score-spectrum") {
		t.Fatalf("write candidate from research-only anchor error = %v, want research-only guard", err)
	}
}

func TestWriteAOQTSidecarCandidatePackageCleansCreatedOutputsAfterInjectedFailure(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	out := filepath.Join(t.TempDir(), "candidate.mll")
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, out)
	anchorManifest := mustReadPackageManifestForAOQTTest(t, DefaultPackageManifestPath(anchor))
	anchorPaths, err := packageRolePaths(anchor, anchorManifest)
	if err != nil {
		t.Fatalf("anchor package paths: %v", err)
	}
	oldWriter := writeAOQTCandidatePackageManifestExclusive
	writeAOQTCandidatePackageManifestExclusive = func(manifest PackageManifest, path string, cleanup *aoqtCandidateCleanup) error {
		data, err := encodePackageManifestMLL(manifest)
		if err != nil {
			return err
		}
		if err := writeAOQTExclusiveFile(path, data, 0o644, cleanup); err != nil {
			return err
		}
		return fmt.Errorf("injected package manifest post-create failure")
	}
	defer func() {
		writeAOQTCandidatePackageManifestExclusive = oldWriter
	}()

	_, err = WriteAOQTSidecarCandidatePackage(cfg)
	if err == nil || !strings.Contains(err.Error(), "injected package manifest post-create failure") {
		t.Fatalf("write candidate injected failure error = %v, want post-create injected failure", err)
	}
	for role, path := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(out, true)) {
		if path == "" {
			continue
		}
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Fatalf("candidate output %q residue at %s after injected failure: %v", role, path, err)
		}
	}
	for role, path := range anchorPaths {
		if _, err := os.Lstat(path); err != nil {
			t.Fatalf("anchor role %q was removed or became unreadable at %s: %v", role, path, err)
		}
	}
}

func TestAOQTCandidateCleanupSkipsSwappedOutputPaths(t *testing.T) {
	dir := t.TempDir()
	registered := filepath.Join(dir, "candidate.mll")
	foreign := filepath.Join(dir, "foreign.mll")
	if err := os.WriteFile(foreign, []byte("foreign\n"), 0o644); err != nil {
		t.Fatalf("write foreign file: %v", err)
	}

	cleanup := newAOQTCandidateCleanup()
	out, err := os.OpenFile(registered, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		t.Fatalf("create registered file: %v", err)
	}
	cleanup.createdFile(registered, out)
	if err := out.Close(); err != nil {
		t.Fatalf("close registered file: %v", err)
	}
	if err := os.Remove(registered); err != nil {
		t.Fatalf("remove registered file before swap: %v", err)
	}
	if err := os.WriteFile(registered, []byte("replacement\n"), 0o644); err != nil {
		t.Fatalf("write replacement file: %v", err)
	}

	cleanup.run()
	if got, err := os.ReadFile(registered); err != nil || string(got) != "replacement\n" {
		t.Fatalf("swapped regular file was removed or changed: data=%q err=%v", string(got), err)
	}
	if got, err := os.ReadFile(foreign); err != nil || string(got) != "foreign\n" {
		t.Fatalf("foreign file was removed or changed: data=%q err=%v", string(got), err)
	}

	symlinkPath := filepath.Join(dir, "candidate-symlink.mll")
	cleanup = newAOQTCandidateCleanup()
	out, err = os.OpenFile(symlinkPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		t.Fatalf("create symlink-registered file: %v", err)
	}
	cleanup.createdFile(symlinkPath, out)
	if err := out.Close(); err != nil {
		t.Fatalf("close symlink-registered file: %v", err)
	}
	if err := os.Remove(symlinkPath); err != nil {
		t.Fatalf("remove symlink-registered file before swap: %v", err)
	}
	if err := os.Symlink(foreign, symlinkPath); err != nil {
		t.Fatalf("create replacement symlink: %v", err)
	}

	cleanup.run()
	if info, err := os.Lstat(symlinkPath); err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("swapped symlink was removed or changed: info=%v err=%v", info, err)
	}
	if got, err := os.ReadFile(foreign); err != nil || string(got) != "foreign\n" {
		t.Fatalf("symlink target was removed or changed: data=%q err=%v", string(got), err)
	}
}

func TestAOQTCandidatePackageLoadFailsClosedForOmittedPolicy(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT candidate package: %v", err)
	}
	packageManifest := result.PackageManifest
	packageManifest.AOQTTransform = AOQTTransformPolicy{}
	data, err := encodePackageManifestMLL(packageManifest)
	if err != nil {
		t.Fatalf("encode package manifest without AOQT policy: %v", err)
	}
	if err := os.WriteFile(result.Paths.PackageManifestPath, data, 0o644); err != nil {
		t.Fatalf("rewrite package manifest without AOQT policy: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err == nil || !strings.Contains(err.Error(), "requires enabled AOQT transform policy") {
		t.Fatalf("load omitted AOQT policy error = %v, want policy guard", err)
	}
}

func TestAOQTCandidatePackageLoadFailsClosedForPolicyTransformHashMismatch(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT candidate package: %v", err)
	}
	packageManifest := result.PackageManifest
	packageManifest.AOQTTransform.TransformSHA256 = strings.Repeat("9", 64)
	if err := packageManifest.WriteFile(result.Paths.PackageManifestPath); err != nil {
		t.Fatalf("rewrite package manifest with bad AOQT transform hash: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err == nil || !strings.Contains(err.Error(), "AOQT transform policy transform_sha256 mismatch") {
		t.Fatalf("load bad AOQT policy hash error = %v, want transform_sha256 mismatch", err)
	}
}

func TestAOQTCandidatePackageLoadFailsClosedForMissingOrTamperedSidecar(t *testing.T) {
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "candidate.mll"))
	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT candidate package: %v", err)
	}

	if err := os.Remove(result.Paths.PostPoolTransformPath); err != nil {
		t.Fatalf("remove AOQT sidecar: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err == nil || !strings.Contains(err.Error(), "verify") {
		t.Fatalf("load missing AOQT sidecar error = %v, want verify failure", err)
	}

	if err := cfg.Transform.WriteFile(result.Paths.PostPoolTransformPath); err != nil {
		t.Fatalf("restore AOQT sidecar: %v", err)
	}
	if err := os.WriteFile(result.Paths.PostPoolTransformPath, []byte(`{"version":"tampered"}`+"\n"), 0o644); err != nil {
		t.Fatalf("tamper AOQT sidecar: %v", err)
	}
	if _, err := rt.LoadEmbeddingPackage(context.Background(), result.Paths.ArtifactPath); err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("load tampered AOQT sidecar error = %v, want sha256 mismatch", err)
	}
}

func TestAOQTTransformPolicyRejectsReleaseClaims(t *testing.T) {
	policy := AOQTTransformPolicy{
		Schema:                      AOQTTransformPolicySchema,
		Enabled:                     true,
		ResearchOnly:                true,
		ResearchTrainAllowed:        true,
		AnchorArtifactSHA256:        strings.Repeat("1", 64),
		AnchorPackageManifestSHA256: strings.Repeat("2", 64),
		AnchorEmbeddingSpaceID:      "tiny-space",
		DatasetManifestSHA256:       strings.Repeat("6", 64),
		QrelsSHA256ByDataset:        map[string]string{"synthetic": strings.Repeat("7", 64)},
		CompatibilityDigest:         strings.Repeat("8", 64),
		TransformSHA256:             strings.Repeat("3", 64),
		PairingsSHA256:              strings.Repeat("4", 64),
		AnglesSHA256:                strings.Repeat("5", 64),
	}
	for _, mutate := range []func(*AOQTTransformPolicy){
		func(p *AOQTTransformPolicy) { p.ReleaseTrainAllowed = true },
		func(p *AOQTTransformPolicy) { p.CommercialUseAllowed = true },
		func(p *AOQTTransformPolicy) { p.FreeOpenReleaseAllowed = true },
		func(p *AOQTTransformPolicy) { p.QualityClaim = true },
	} {
		bad := policy
		mutate(&bad)
		if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "quality_claim=false") {
			t.Fatalf("AOQT release-claim policy error = %v, want fail-closed legal gate", err)
		}
	}
}

func writeTinyAOQTAnchorPackage(t *testing.T) string {
	t.Helper()
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "anchor.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	if err := tinyMaskedEmbeddingManifest().WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	weights := NewWeightFile(map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF16([]int{3, 2}, []float32{
			1, 0,
			0, 1,
			1, 1,
		}),
		"projection": backend.NewTensorF16([]int{2, 2}, []float32{
			1, 0,
			0, 1,
		}),
	})
	if err := weights.WriteFile(DefaultWeightFilePath(artifactPath)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	if err := tinyEmbeddingTokenizerFile().WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{})
	if err := plan.WriteFile(DefaultMemoryPlanPath(artifactPath)); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}
	packageManifest, err := BuildPackageManifest(PackageEmbedding, bundle.Artifact, map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": DefaultEmbeddingManifestPath(artifactPath),
		"tokenizer":          DefaultTokenizerPath(artifactPath),
		"weights":            DefaultWeightFilePath(artifactPath),
		"memory_plan":        DefaultMemoryPlanPath(artifactPath),
	})
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	if err := packageManifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}
	return artifactPath
}

func tinyAOQTCandidatePackageConfig(t *testing.T, anchor, out string) AOQTSidecarCandidatePackageConfig {
	t.Helper()
	anchorSHA, _, err := fileHash(anchor)
	if err != nil {
		t.Fatalf("hash anchor: %v", err)
	}
	packageSHA, _, err := fileHash(DefaultPackageManifestPath(anchor))
	if err != nil {
		t.Fatalf("hash anchor package manifest: %v", err)
	}
	transform := tinyAOQTGivensTransform(2, math.Pi/2)
	return AOQTSidecarCandidatePackageConfig{
		AnchorArtifactPath:                  anchor,
		OutputArtifactPath:                  out,
		ExpectedAnchorArtifactSHA256:        anchorSHA,
		ExpectedAnchorPackageManifestSHA256: packageSHA,
		AnchorEmbeddingSpaceID:              "tiny-space-for-package-writer",
		DatasetManifestSHA256:               strings.Repeat("1", 64),
		QrelsSHA256ByDataset:                map[string]string{"synthetic": strings.Repeat("2", 64)},
		CompatibilityDigest:                 strings.Repeat("3", 64),
		Transform:                           transform,
	}
}

func mustReadPackageManifestForAOQTTest(t *testing.T, path string) PackageManifest {
	t.Helper()
	manifest, err := ReadPackageManifestFile(path)
	if err != nil {
		t.Fatalf("read package manifest: %v", err)
	}
	return manifest
}

func attachTinyAOQTPolicyForTest(t *testing.T, manifest *PackageManifest, artifactPath string, transform AOQTGivensTransform) {
	t.Helper()
	artifactSHA, _, err := fileHash(artifactPath)
	if err != nil {
		t.Fatalf("hash artifact: %v", err)
	}
	transformSHA, _, err := fileHash(DefaultPostPoolTransformPath(artifactPath))
	if err != nil {
		t.Fatalf("hash AOQT transform: %v", err)
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		t.Fatalf("hash AOQT pairings: %v", err)
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		t.Fatalf("hash AOQT angles: %v", err)
	}
	manifest.AOQTTransform = AOQTTransformPolicy{
		Schema:                      AOQTTransformPolicySchema,
		Enabled:                     true,
		ResearchOnly:                true,
		ResearchTrainAllowed:        true,
		ReleaseTrainAllowed:         false,
		CommercialUseAllowed:        false,
		FreeOpenReleaseAllowed:      false,
		QualityClaim:                false,
		AnchorArtifactSHA256:        artifactSHA,
		AnchorPackageManifestSHA256: strings.Repeat("4", 64),
		AnchorEmbeddingSpaceID:      "tiny-space-for-package-loader",
		DatasetManifestSHA256:       strings.Repeat("5", 64),
		QrelsSHA256ByDataset:        map[string]string{"synthetic": strings.Repeat("6", 64)},
		CompatibilityDigest:         strings.Repeat("7", 64),
		TransformSHA256:             transformSHA,
		PairingsSHA256:              pairingsSHA,
		AnglesSHA256:                anglesSHA,
	}
}
