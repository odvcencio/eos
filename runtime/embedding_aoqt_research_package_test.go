package eosruntime

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
)

func TestLoadAOQTResearchEmbeddingPackageAllowsResearchLineageByOptIn(t *testing.T) {
	artifactPath := writeTinyAOQTResearchCandidatePackage(t)
	beforeManifest := mustReadPackageManifestForAOQTResearchTest(t, artifactPath)
	beforeFiles := snapshotAOQTResearchPackageFiles(t, artifactPath, beforeManifest)

	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath); err == nil || !strings.Contains(err.Error(), "research-only score-spectrum") {
		t.Fatalf("ordinary AOQT research candidate load error = %v, want research-only rejection", err)
	}
	model, err := rt.LoadAOQTResearchEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		t.Fatalf("explicit AOQT research candidate load: %v", err)
	}
	out, err := model.Embed(context.Background(), []int32{1})
	if err != nil {
		t.Fatalf("embed explicit AOQT research candidate: %v", err)
	}
	if out.Embeddings == nil || !reflect.DeepEqual(out.Embeddings.Shape, []int{2}) {
		t.Fatalf("explicit AOQT research candidate embedding shape = %v, want [2]", embeddingShapeForTest(out.Embeddings))
	}

	afterManifest := mustReadPackageManifestForAOQTResearchTest(t, artifactPath)
	if !reflect.DeepEqual(afterManifest, beforeManifest) {
		t.Fatalf("AOQT research candidate package metadata changed during load:\nbefore=%+v\nafter=%+v", beforeManifest, afterManifest)
	}
	afterFiles := snapshotAOQTResearchPackageFiles(t, artifactPath, afterManifest)
	if !reflect.DeepEqual(afterFiles, beforeFiles) {
		t.Fatalf("AOQT research candidate sibling files changed during load:\nbefore=%v\nafter=%v", beforeFiles, afterFiles)
	}
}

func TestLoadAOQTResearchEmbeddingPackageRejectsInvalidContracts(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func(*testing.T) string
		want  string
	}{
		{
			name: "non_aoqt",
			build: func(t *testing.T) string {
				return writeTinyResearchEmbeddingPackage(t)
			},
			want: "requires enabled AOQT transform policy",
		},
		{
			name: "falsified_anchor_artifact_hash",
			build: func(t *testing.T) string {
				artifactPath := writeTinyAOQTResearchCandidatePackage(t)
				manifest := mustReadPackageManifestForAOQTResearchTest(t, artifactPath)
				manifest.AOQTTransform.AnchorArtifactSHA256 = strings.Repeat("f", 64)
				if err := manifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
					t.Fatalf("rewrite falsified AOQT policy: %v", err)
				}
				return artifactPath
			},
			want: "anchor artifact sha256 mismatch",
		},
		{
			name: "missing_policy",
			build: func(t *testing.T) string {
				artifactPath := writeTinyAOQTResearchCandidatePackage(t)
				manifest := mustReadPackageManifestForAOQTResearchTest(t, artifactPath)
				manifest.AOQTTransform = AOQTTransformPolicy{}
				filtered := manifest.Files[:0]
				for _, item := range manifest.Files {
					if item.Role != EmbeddingPostPoolTransformRole {
						filtered = append(filtered, item)
					}
				}
				manifest.Files = filtered
				if err := manifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
					t.Fatalf("rewrite AOQT candidate without policy: %v", err)
				}
				return artifactPath
			},
			want: "requires enabled AOQT transform policy",
		},
		{
			name: "hash_substitution",
			build: func(t *testing.T) string {
				artifactPath := writeTinyAOQTResearchCandidatePackage(t)
				if err := os.WriteFile(DefaultPostPoolTransformPath(artifactPath), []byte(`{"version":"tampered"}`+"\n"), 0o644); err != nil {
					t.Fatalf("tamper AOQT transform: %v", err)
				}
				return artifactPath
			},
			want: "sha256 mismatch",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			artifactPath := tc.build(t)
			rt := New(cuda.New(), metal.New())
			if _, err := rt.LoadAOQTResearchEmbeddingPackage(context.Background(), artifactPath); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("explicit AOQT research candidate load error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestLoadAOQTResearchEmbeddingPackageDoesNotUseSealedFallback(t *testing.T) {
	artifactPath := writeTinyResearchEmbeddingPackage(t)
	sealedPath, err := ExportPackageToMLL(artifactPath, "")
	if err != nil {
		t.Fatalf("export research-only package: %v", err)
	}
	rt := New(cuda.New(), metal.New())
	if _, err := rt.LoadEmbeddingPackage(context.Background(), sealedPath); err == nil || !strings.Contains(err.Error(), "research-only score-spectrum") {
		t.Fatalf("ordinary sealed research package load error = %v, want research-only rejection", err)
	}
	if _, err := rt.LoadAOQTResearchEmbeddingPackage(context.Background(), sealedPath); err == nil || !strings.Contains(err.Error(), "requires a native package manifest") {
		t.Fatalf("explicit sealed research package load error = %v, want native-package rejection", err)
	}
}

func writeTinyAOQTResearchCandidatePackage(t *testing.T) string {
	t.Helper()
	anchor := writeTinyAOQTAnchorPackage(t)
	cfg := tinyAOQTCandidatePackageConfig(t, anchor, filepath.Join(t.TempDir(), "research-candidate.mll"))
	result, err := WriteAOQTSidecarCandidatePackage(cfg)
	if err != nil {
		t.Fatalf("write AOQT research candidate: %v", err)
	}
	manifest := result.PackageManifest
	manifest.ScoreSpectrum = EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:        true,
		ScoreSpectrumResearchOnly: true,
		TrainAllowedForResearch:   true,
		ReleaseTrainAllowed:       false,
		CommercialUseAllowed:      false,
		SourceArtifactHashes:      []string{strings.Repeat("a", 64)},
		ScoreSpectrumRowCount:     1,
	}
	manifest.ListwiseGeometry = EmbeddingListwiseGeometryPolicy{
		ListwiseGeometryTrain:        true,
		ListwiseGeometryResearchOnly: true,
		TrainAllowedForResearch:      true,
		ReleaseTrainAllowed:          false,
		CommercialUseAllowed:         false,
		SourceArtifactHashes:         []string{strings.Repeat("b", 64)},
		ListwiseGeometryBatchCount:   1,
	}
	if err := manifest.WriteFile(result.Paths.PackageManifestPath); err != nil {
		t.Fatalf("write AOQT research lineage package manifest: %v", err)
	}
	return result.Paths.ArtifactPath
}

func writeTinyResearchEmbeddingPackage(t *testing.T) string {
	t.Helper()
	artifactPath := writeTinyAOQTAnchorPackage(t)
	manifest := mustReadPackageManifestForAOQTResearchTest(t, artifactPath)
	manifest.ScoreSpectrum = EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:        true,
		ScoreSpectrumResearchOnly: true,
		TrainAllowedForResearch:   true,
		ReleaseTrainAllowed:       false,
		CommercialUseAllowed:      false,
		SourceArtifactHashes:      []string{strings.Repeat("c", 64)},
		ScoreSpectrumRowCount:     1,
	}
	if err := manifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
		t.Fatalf("write non-AOQT research package manifest: %v", err)
	}
	return artifactPath
}

func mustReadPackageManifestForAOQTResearchTest(t *testing.T, artifactPath string) PackageManifest {
	t.Helper()
	manifest, err := ReadPackageManifestFile(DefaultPackageManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read AOQT research package manifest: %v", err)
	}
	return manifest
}

func snapshotAOQTResearchPackageFiles(t *testing.T, artifactPath string, manifest PackageManifest) map[string]string {
	t.Helper()
	paths, err := packageRolePaths(artifactPath, manifest)
	if err != nil {
		t.Fatalf("resolve AOQT research package files: %v", err)
	}
	paths["package_manifest"] = DefaultPackageManifestPath(artifactPath)
	hashes := make(map[string]string, len(paths))
	for role, path := range paths {
		sum, _, err := fileHash(path)
		if err != nil {
			t.Fatalf("hash AOQT research package %q: %v", role, err)
		}
		hashes[role] = sum
	}
	return hashes
}

func embeddingShapeForTest(tensor *backend.Tensor) []int {
	if tensor == nil {
		return nil
	}
	return tensor.Shape
}
