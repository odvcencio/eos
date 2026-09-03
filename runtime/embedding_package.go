package eosruntime

import (
	"context"
	"fmt"
	"os"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// EmbeddingPackagePaths names the files that make up a packaged embedding model.
type EmbeddingPackagePaths struct {
	ArtifactPath          string
	ManifestPath          string
	TokenizerPath         string
	WeightFilePath        string
	MemoryPlanPath        string
	PackageManifestPath   string
	PostPoolTransformPath string
}

// LoadEmbeddingPackage loads a packaged embedding model from sibling artifact, manifest, and weight files.
func (rt *Runtime) LoadEmbeddingPackage(ctx context.Context, artifactPath string) (*EmbeddingModel, error) {
	if model, ok, err := rt.tryLoadSealedEmbeddingPackage(ctx, artifactPath); err != nil {
		return nil, err
	} else if ok {
		return model, nil
	}
	return rt.LoadEmbeddingPackageWithPaths(
		ctx,
		EmbeddingPackagePaths{
			ArtifactPath:          artifactPath,
			ManifestPath:          ResolveEmbeddingManifestPath(artifactPath),
			TokenizerPath:         DefaultTokenizerPath(artifactPath),
			WeightFilePath:        DefaultWeightFilePath(artifactPath),
			MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
			PackageManifestPath:   ResolvePackageManifestPath(artifactPath),
			PostPoolTransformPath: DefaultPostPoolTransformPath(artifactPath),
		},
	)
}

func (rt *Runtime) tryLoadSealedEmbeddingPackage(ctx context.Context, path string) (*EmbeddingModel, bool, error) {
	reader, meta, err := readSealedEosMLL(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, nil
	}
	if _, ok := meta.JSONFiles["embedding_manifest"]; !ok {
		return nil, false, nil
	}
	pkg, ok, err := sealedEmbeddingPackageFromReader(reader, meta)
	if err != nil || !ok {
		return nil, ok, err
	}
	opts := pkg.Weights.LoadOptions()
	if pkg.PackageManifest != nil {
		if err := rejectResearchOnlyRestrictedEmbeddingPackage(*pkg.PackageManifest); err != nil {
			return nil, true, err
		}
		opts = append(opts, WithPackageManifest(*pkg.PackageManifest))
	}
	if pkg.MemoryPlan != nil {
		opts = append(opts, WithMemoryPlan(*pkg.MemoryPlan))
	}
	model, err := rt.LoadEmbedding(ctx, pkg.Module, pkg.Manifest, opts...)
	if err != nil {
		return nil, true, err
	}
	if pkg.Tokenizer != nil {
		if err := model.attachTokenizer(*pkg.Tokenizer); err != nil {
			return nil, true, err
		}
	}
	return model, true, nil
}

// LoadEmbeddingPackageWithPaths loads a packaged embedding model from explicit artifact, manifest, and weight files.
func (rt *Runtime) LoadEmbeddingPackageWithPaths(ctx context.Context, paths EmbeddingPackagePaths) (*EmbeddingModel, error) {
	opts := make([]LoadOption, 0, 5)
	manifest, err := ReadEmbeddingManifestFile(paths.ManifestPath)
	if err != nil {
		return nil, err
	}
	if paths.PackageManifestPath != "" {
		if _, err := os.Stat(paths.PackageManifestPath); err == nil {
			packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
			if err != nil {
				return nil, err
			}
			if err := rejectResearchOnlyRestrictedEmbeddingPackage(packageManifest); err != nil {
				return nil, err
			}
			if manifest.requiresPostPoolTransform() && !packageManifest.HasFileRole(EmbeddingPostPoolTransformRole) {
				return nil, fmt.Errorf("embedding manifest declares %q but package manifest does not list %q", manifest.PostPoolTransform, EmbeddingPostPoolTransformRole)
			}
			verifyPaths := map[string]string{
				"artifact":           paths.ArtifactPath,
				"embedding_manifest": paths.ManifestPath,
				"tokenizer":          paths.TokenizerPath,
				"weights":            paths.WeightFilePath,
				"memory_plan":        paths.MemoryPlanPath,
			}
			if manifest.requiresPostPoolTransform() {
				verifyPaths[EmbeddingPostPoolTransformRole] = paths.PostPoolTransformPath
			} else if packageManifest.HasFileRole(EmbeddingPostPoolTransformRole) {
				verifyPaths[EmbeddingPostPoolTransformRole] = paths.PostPoolTransformPath
			}
			if packageManifest.Kind == PackageTraining {
				verifyPaths["train_manifest"] = DefaultEmbeddingTrainManifestPath(paths.ArtifactPath)
				verifyPaths["checkpoint"] = DefaultEmbeddingCheckpointPath(paths.ArtifactPath)
				verifyPaths["train_profile"] = DefaultEmbeddingTrainProfilePath(paths.ArtifactPath)
			}
			if err := packageManifest.VerifyFiles(verifyPaths); err != nil {
				return nil, err
			}
			opts = append(opts, WithPackageManifest(packageManifest))
		} else if manifest.requiresPostPoolTransform() {
			return nil, fmt.Errorf("embedding manifest declares %q but package manifest is missing", manifest.PostPoolTransform)
		}
	} else if manifest.requiresPostPoolTransform() {
		return nil, fmt.Errorf("embedding manifest declares %q but package manifest path is required", manifest.PostPoolTransform)
	}
	if manifest.requiresPostPoolTransform() {
		transform, err := ReadAOQTGivensTransformFile(paths.PostPoolTransformPath)
		if err != nil {
			return nil, fmt.Errorf("read post-pool transform: %w", err)
		}
		opts = append(opts, WithPostPoolTransform(transform))
	}
	weightFile, err := ReadWeightFile(paths.WeightFilePath)
	if err != nil {
		return nil, err
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		return nil, err
	}
	opts = append(opts, weightFile.LoadOptions()...)
	if paths.MemoryPlanPath != "" {
		if _, err := os.Stat(paths.MemoryPlanPath); err == nil {
			plan, err := ReadMemoryPlanFile(paths.MemoryPlanPath)
			if err != nil {
				return nil, err
			}
			opts = append(opts, WithMemoryPlan(plan))
		}
	}
	model, err := rt.LoadEmbedding(ctx, mod, manifest, opts...)
	if err != nil {
		return nil, err
	}
	if tokenizerFile, ok, err := readOptionalTokenizerFile(paths.TokenizerPath); err != nil {
		return nil, err
	} else if ok {
		if err := model.attachTokenizer(tokenizerFile); err != nil {
			return nil, err
		}
	}
	return model, nil
}

func rejectResearchOnlyRestrictedEmbeddingPackage(packageManifest PackageManifest) error {
	if packageManifest.Kind == PackageEmbedding && packageManifest.ScoreSpectrum.ScoreSpectrumResearchOnly {
		return fmt.Errorf("embedding package was trained with research-only score-spectrum data; use training package load for continued research training")
	}
	if packageManifest.Kind == PackageEmbedding && packageManifest.ListwiseGeometry.ListwiseGeometryResearchOnly {
		return fmt.Errorf("embedding package was trained with research-only listwise geometry data; use training package load for continued research training")
	}
	return nil
}

func readOptionalTokenizerFile(path string) (TokenizerFile, bool, error) {
	if path == "" {
		return TokenizerFile{}, false, nil
	}
	tokenizer, err := ReadTokenizerFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return TokenizerFile{}, false, nil
		}
		return TokenizerFile{}, false, err
	}
	return tokenizer, true, nil
}
