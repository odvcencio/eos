package eosruntime

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestAOQTGivensTransformIdentityNoOp(t *testing.T) {
	transform := tinyAOQTGivensTransform(4, 0)
	in := []float32{0.25, -0.5, 0.75, 1}
	got, err := transform.ApplyVector(in)
	if err != nil {
		t.Fatalf("apply identity AOQT: %v", err)
	}
	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("identity AOQT changed coord %d: got %.9g want %.9g", i, got[i], in[i])
		}
	}
}

func TestAOQTGivensTransformOrthogonalityAndDenseDotInvariance(t *testing.T) {
	transform := AOQTGivensTransform{
		Version:  AOQTTransformVersion,
		Kind:     EmbeddingPostPoolTransformAOQTGivens,
		Dim:      384,
		Seed:     191,
		AngleCap: 0.04,
		Stages: []AOQTStage{
			tinySequentialAOQTStage(384, 0.031),
			tinySequentialAOQTStage(384, -0.017),
		},
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		t.Fatalf("orthogonality audit: %v", err)
	}
	if orth > 1e-6 {
		t.Fatalf("orthogonality = %.12g, want <= 1e-6", orth)
	}

	rng := rand.New(rand.NewSource(384191))
	var maxDelta float64
	for i := 0; i < 64; i++ {
		q := randomFloat32Vector(rng, 384)
		d := randomFloat32Vector(rng, 384)
		tq, err := transform.ApplyVector(q)
		if err != nil {
			t.Fatalf("apply query %d: %v", i, err)
		}
		td, err := transform.ApplyVector(d)
		if err != nil {
			t.Fatalf("apply doc %d: %v", i, err)
		}
		delta := math.Abs(float64(dotFloat32(q, d) - dotFloat32(tq, td)))
		if delta > maxDelta {
			maxDelta = delta
		}
	}
	if maxDelta > 2e-6 {
		t.Fatalf("max dense dot delta = %.12g, want <= 2e-6", maxDelta)
	}
}

func TestEmbeddingManifestPostPoolTransformRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "embed_aoqt.embedding.mll")
	want := tinyEmbeddingManifest()
	want.PostPoolTransform = EmbeddingPostPoolTransformAOQTGivens
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if got.PostPoolTransform != EmbeddingPostPoolTransformAOQTGivens {
		t.Fatalf("post_pool_transform = %q, want %q", got.PostPoolTransform, EmbeddingPostPoolTransformAOQTGivens)
	}
}

func TestAOQTGivensTransformAuditFieldsMustMatchComputedValues(t *testing.T) {
	transform := tinyAOQTGivensTransform(4, 0.125)
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		t.Fatalf("pairings hash: %v", err)
	}
	angles, err := transform.AnglesSHA256()
	if err != nil {
		t.Fatalf("angles hash: %v", err)
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		t.Fatalf("orthogonality: %v", err)
	}
	transform.Audit = AOQTAuditInfo{
		PairingsSHA256: pairings,
		AnglesSHA256:   angles,
		Orthogonality:  &orth,
	}
	if err := transform.Validate(); err != nil {
		t.Fatalf("validate matching audit: %v", err)
	}

	badPairings := transform
	badPairings.Audit.PairingsSHA256 = strings.Repeat("0", 64)
	if err := badPairings.Validate(); err == nil || !strings.Contains(err.Error(), "pairings_sha256 mismatch") {
		t.Fatalf("bad pairings audit error = %v, want mismatch", err)
	}

	badAngles := transform
	badAngles.Audit.AnglesSHA256 = strings.Repeat("1", 64)
	if err := badAngles.Validate(); err == nil || !strings.Contains(err.Error(), "angles_sha256 mismatch") {
		t.Fatalf("bad angles audit error = %v, want mismatch", err)
	}

	badOrth := transform
	wrongOrth := orth + 1e-6
	badOrth.Audit.Orthogonality = &wrongOrth
	if err := badOrth.Validate(); err == nil || !strings.Contains(err.Error(), "orthogonality mismatch") {
		t.Fatalf("bad orthogonality audit error = %v, want mismatch", err)
	}
}

func TestLoadEmbeddingPackageAppliesAOQTOnceForRaggedBatch(t *testing.T) {
	artifactPath := writeTinyAOQTPackage(t, tinyAOQTGivensTransform(2, math.Pi/2), true)

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("load AOQT embedding package: %v", err)
	}
	result, err := model.EmbedBatch(context.Background(), [][]int32{{0, 2}, {1}})
	if err != nil {
		t.Fatalf("embed AOQT ragged batch: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2, 2}, []float32{
		-0.35355338, 0.8535534,
		-1, 0,
	})
}

func TestLoadEmbeddingPackageWithPathsRejectsAOQTWithoutPackageManifestPath(t *testing.T) {
	artifactPath := writeTinyAOQTPackage(t, tinyAOQTGivensTransform(2, 0), true)

	rt := New(cuda.New(), metal.New())
	_, err := rt.LoadEmbeddingPackageWithPaths(context.Background(), EmbeddingPackagePaths{
		ArtifactPath:          artifactPath,
		ManifestPath:          DefaultEmbeddingManifestPath(artifactPath),
		WeightFilePath:        DefaultWeightFilePath(artifactPath),
		MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
		PostPoolTransformPath: DefaultPostPoolTransformPath(artifactPath),
		PackageManifestPath:   "",
	})
	if err == nil || !strings.Contains(err.Error(), "package manifest path is required") {
		t.Fatalf("load explicit-path AOQT without package manifest path error = %v, want required package manifest path", err)
	}
}

func TestLoadEmbeddingPackageWithPathsAllowsLegacyWithoutPackageManifestPath(t *testing.T) {
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_masked_pooled.mll")
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

	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbeddingPackageWithPaths(context.Background(), EmbeddingPackagePaths{
		ArtifactPath:        artifactPath,
		ManifestPath:        DefaultEmbeddingManifestPath(artifactPath),
		WeightFilePath:      DefaultWeightFilePath(artifactPath),
		PackageManifestPath: "",
	})
	if err != nil {
		t.Fatalf("load legacy explicit-path package without package manifest path: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed legacy explicit-path package: %v", err)
	}
	assertTensorClose(t, result.Embeddings, []int{2}, []float32{0, 1})
}

func TestLoadEmbeddingPackageRejectsDeclaredMissingAOQTSidecar(t *testing.T) {
	artifactPath := writeTinyAOQTPackage(t, tinyAOQTGivensTransform(2, 0), false)

	rt := New(cuda.New(), metal.New())
	_, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err == nil || !strings.Contains(err.Error(), "does not list") {
		t.Fatalf("load missing AOQT sidecar error = %v, want package manifest role failure", err)
	}
}

func TestLoadEmbeddingPackageRejectsTamperedAOQTSidecar(t *testing.T) {
	artifactPath := writeTinyAOQTPackage(t, tinyAOQTGivensTransform(2, 0), true)
	if err := os.WriteFile(DefaultPostPoolTransformPath(artifactPath), []byte(`{"tampered":true}`+"\n"), 0o644); err != nil {
		t.Fatalf("tamper AOQT sidecar: %v", err)
	}

	rt := New(cuda.New(), metal.New())
	_, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err == nil || !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("load tampered AOQT sidecar error = %v, want sha256 mismatch", err)
	}
}

func writeTinyAOQTPackage(t *testing.T, transform AOQTGivensTransform, includeTransformRole bool) string {
	t.Helper()
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	dir := t.TempDir()
	artifactPath := filepath.Join(dir, "tiny_embed_masked_pooled.mll")
	if err := eosartifact.WriteFile(artifactPath, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.PostPoolTransform = EmbeddingPostPoolTransformAOQTGivens
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
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
	plan := NewMemoryPlan(bundle.Artifact, weights.Weights, MemoryPlanOptions{})
	if err := plan.WriteFile(DefaultMemoryPlanPath(artifactPath)); err != nil {
		t.Fatalf("write memory plan: %v", err)
	}
	if err := transform.WriteFile(DefaultPostPoolTransformPath(artifactPath)); err != nil {
		t.Fatalf("write AOQT sidecar: %v", err)
	}
	files := map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": DefaultEmbeddingManifestPath(artifactPath),
		"weights":            DefaultWeightFilePath(artifactPath),
		"memory_plan":        DefaultMemoryPlanPath(artifactPath),
	}
	if includeTransformRole {
		files[EmbeddingPostPoolTransformRole] = DefaultPostPoolTransformPath(artifactPath)
	}
	packageManifest, err := buildPackageManifestUnchecked(PackageEmbedding, bundle.Artifact, files)
	if err != nil {
		t.Fatalf("build package manifest: %v", err)
	}
	if includeTransformRole {
		attachTinyAOQTPolicyForTest(t, &packageManifest, artifactPath, transform)
	}
	if err := packageManifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
		t.Fatalf("write package manifest: %v", err)
	}
	return artifactPath
}

func tinyAOQTGivensTransform(dim int, angle float64) AOQTGivensTransform {
	return AOQTGivensTransform{
		Version:  AOQTTransformVersion,
		Kind:     EmbeddingPostPoolTransformAOQTGivens,
		Dim:      dim,
		Seed:     191,
		AngleCap: float32(math.Pi),
		Stages: []AOQTStage{
			tinySequentialAOQTStage(dim, float32(angle)),
		},
	}
}

func tinySequentialAOQTStage(dim int, angle float32) AOQTStage {
	pairs := make([][2]int, 0, dim/2)
	angles := make([]float32, 0, dim/2)
	for i := 0; i+1 < dim; i += 2 {
		pairs = append(pairs, [2]int{i, i + 1})
		angles = append(angles, angle)
	}
	return AOQTStage{Pairs: pairs, Angles: angles}
}

func randomFloat32Vector(rng *rand.Rand, dim int) []float32 {
	out := make([]float32, dim)
	for i := range out {
		out[i] = float32(rng.NormFloat64() * 0.02)
	}
	return out
}

func dotFloat32(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
