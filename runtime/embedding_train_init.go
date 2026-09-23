package eosruntime

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

// EmbeddingTrainInitOptions controls training-package initialization from an artifact.
type EmbeddingTrainInitOptions struct {
	Seed                    int64
	ShapeSizes              map[string]int
	BootstrapArtifactPath   string
	BootstrapCheckpointPath string
	BootstrapInferencePath  string
	BootstrapInferenceWiden bool
	BootstrapTailInit       string
	BootstrapTailScale      float64
}

// InitializeEmbeddingTrainerPackageFromSealedInference initializes a fresh training package from a sealed inference package while preserving the sealed source graph.
func InitializeEmbeddingTrainerPackageFromSealedInference(artifactPath, bootstrapPath string, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	if artifactPath == "" {
		return EmbeddingTrainPackagePaths{}, fmt.Errorf("output artifact path is required")
	}
	if !opts.BootstrapInferenceWiden {
		if err := validateExactBootstrapTailOptions(opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	source, err := ReadSealedEmbeddingPackage(bootstrapPath)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, fmt.Errorf("read bootstrap inference artifact %q: %w", bootstrapPath, err)
	}
	manifest := source.Manifest.normalizedForModule(source.Module)
	manifest = inferLegacyInferenceBootstrapManifest(manifest, source.Weights.Weights)
	if err := manifest.ValidateModule(source.Module); err != nil {
		return EmbeddingTrainPackagePaths{}, fmt.Errorf("bootstrap inference embedding manifest: %w", err)
	}
	if source.Tokenizer != nil {
		if _, err := NewBPETokenizer(*source.Tokenizer, manifest.Tokenizer); err != nil {
			return EmbeddingTrainPackagePaths{}, fmt.Errorf("bootstrap inference tokenizer: %w", err)
		}
	}
	sourceShapeSizes := inferenceBootstrapShapeSizes(manifest, source.Weights.Weights)
	if opts.BootstrapInferenceWiden {
		if err := validateBootstrapTailInitOptions(opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
		var err error
		manifest, sourceShapeSizes, err = widenedInferenceBootstrapManifestAndShapes(source.Module, manifest, source.Weights.Weights, opts.ShapeSizes)
		if err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	mod, err := cloneEmbeddingInitModule(source.Module)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if err := eosartifact.WriteFile(artifactPath, mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(artifactPath)); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if source.Tokenizer != nil {
		if err := source.Tokenizer.WriteFile(DefaultTokenizerPath(artifactPath)); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	opts.BootstrapInferencePath = bootstrapPath
	opts.BootstrapArtifactPath = ""
	opts.BootstrapCheckpointPath = ""
	if opts.BootstrapInferenceWiden {
		opts.ShapeSizes = mergeTrainingShapeSizes(sourceShapeSizes, nil)
	} else {
		opts.ShapeSizes = mergeTrainingShapeSizes(sourceShapeSizes, opts.ShapeSizes)
	}
	paths, err := InitializeEmbeddingTrainerPackageWithManifest(artifactPath, manifest, cfg, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if err := preserveSealedBootstrapPackagePolicies(paths, source.PackageManifest); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return paths, nil
}

// InitializeEmbeddingTrainerPackage reads an artifact plus its sibling embedding manifest, initializes trainable weights, and writes a training package.
func InitializeEmbeddingTrainerPackage(artifactPath string, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	manifest, err := ReadEmbeddingManifestFile(ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return InitializeEmbeddingTrainerPackageWithManifest(artifactPath, manifest, EmbeddingTrainConfig{}, opts)
}

// InitializeEmbeddingTrainerPackageWithManifest initializes a training package from an explicit embedding manifest and trainer config.
func InitializeEmbeddingTrainerPackageWithManifest(artifactPath string, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	mod, err := eosartifact.ReadFile(artifactPath)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	manifest = manifest.normalized()
	if manifest.ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 {
		return initializeGenericEmbeddingTrainerPackage(artifactPath, mod, manifest, cfg, opts)
	}
	trainManifest := EmbeddingTrainManifest{
		Name:      manifest.Name,
		Embedding: manifest,
		Config:    cfg,
	}
	if err := trainManifest.ValidateModule(mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weights, err := initializedTrainingWeights(mod, manifest, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if opts.hasBootstrapSource() {
		if err := bootstrapTrainingWeights(weights, mod, manifest, opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	trainer, err := NewEmbeddingTrainer(mod, manifest, weights, cfg)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return trainer.WriteTrainingPackage(artifactPath)
}

func initializeGenericEmbeddingTrainerPackage(artifactPath string, mod *eosartifact.Module, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, opts EmbeddingTrainInitOptions) (EmbeddingTrainPackagePaths, error) {
	manifest = manifest.normalizedForModule(mod)
	if err := manifest.ValidateModule(mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if err := validateGenericTrainableEmbeddingModule(mod, manifest); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weights, err := initializedTrainingWeights(mod, manifest, opts)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	if opts.hasBootstrapSource() {
		if err := bootstrapTrainingWeights(weights, mod, manifest, opts); err != nil {
			return EmbeddingTrainPackagePaths{}, err
		}
	}
	return writeGenericEmbeddingTrainingPackage(artifactPath, mod, manifest, cfg, weights)
}

func validateGenericTrainableEmbeddingModule(mod *eosartifact.Module, manifest EmbeddingManifest) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return fmt.Errorf("generic embedding package initialization requires architecture_version=%q", EmbeddingArchitectureCompactTransformerV1)
	}
	if manifest.ParameterTying != EmbeddingParameterTyingUntied {
		return fmt.Errorf("%s package initialization requires parameter_tying=%q", manifest.ArchitectureVersion, EmbeddingParameterTyingUntied)
	}
	for _, param := range mod.Params {
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		if !param.Trainable {
			return fmt.Errorf("param %q is not trainable", param.Name)
		}
	}
	if manifest.OutputProjectionParam != "" {
		if err := validateEmbeddingParam(mod, manifest.OutputProjectionParam); err != nil {
			return err
		}
	}
	return nil
}

func writeGenericEmbeddingTrainingPackage(artifactPath string, mod *eosartifact.Module, manifest EmbeddingManifest, cfg EmbeddingTrainConfig, weights map[string]*backend.Tensor) (EmbeddingTrainPackagePaths, error) {
	if err := eosartifact.WriteFile(artifactPath, mod); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	embeddingManifestPath := DefaultEmbeddingManifestPath(artifactPath)
	if err := manifest.WriteFile(embeddingManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	weightPath := DefaultWeightFilePath(artifactPath)
	if err := NewWeightFile(weights).WriteFile(weightPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	memoryPlan := NewMemoryPlan(mod, weights, MemoryPlanOptions{})
	memoryPlanPath := DefaultMemoryPlanPath(artifactPath)
	if err := memoryPlan.WriteFile(memoryPlanPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainManifest := EmbeddingTrainManifest{
		Name:      manifest.Name,
		Embedding: manifest,
		Config:    cfg,
	}
	trainManifestPath := DefaultEmbeddingTrainManifestPath(artifactPath)
	if err := trainManifest.WriteFile(trainManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	checkpoint := genericEmbeddingTrainCheckpoint(manifest, cfg, weights)
	checkpointPath := DefaultEmbeddingCheckpointPath(artifactPath)
	if err := checkpoint.WriteFile(checkpointPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	trainProfilePath := DefaultEmbeddingTrainProfilePath(artifactPath)
	if err := (EmbeddingTrainProfile{Version: EmbeddingTrainProfileVersion}).WriteFile(trainProfilePath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageFiles := map[string]string{
		"artifact":           artifactPath,
		"embedding_manifest": embeddingManifestPath,
		"weights":            weightPath,
		"memory_plan":        memoryPlanPath,
		"train_manifest":     trainManifestPath,
		"checkpoint":         checkpointPath,
		"train_profile":      trainProfilePath,
	}
	tokenizerPath := DefaultTokenizerPath(artifactPath)
	if _, err := os.Stat(tokenizerPath); err == nil {
		packageFiles["tokenizer"] = tokenizerPath
	}
	packageManifest, err := BuildPackageManifest(PackageTraining, mod, packageFiles)
	if err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	packageManifestPath := DefaultPackageManifestPath(artifactPath)
	if err := packageManifest.WriteFile(packageManifestPath); err != nil {
		return EmbeddingTrainPackagePaths{}, err
	}
	return EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: embeddingManifestPath,
		TokenizerPath:         tokenizerPath,
		WeightFilePath:        weightPath,
		MemoryPlanPath:        memoryPlanPath,
		TrainManifestPath:     trainManifestPath,
		CheckpointPath:        checkpointPath,
		TrainProfilePath:      trainProfilePath,
		PackageManifestPath:   packageManifestPath,
	}, nil
}

func genericEmbeddingTrainCheckpoint(manifest EmbeddingManifest, cfg EmbeddingTrainConfig, weights map[string]*backend.Tensor) EmbeddingTrainCheckpoint {
	tensors := make(map[string]*backend.Tensor, len(weights))
	moments := make(map[string]*backend.Tensor, len(weights)*2)
	for name, tensor := range weights {
		if tensor == nil {
			continue
		}
		master := tensorAsMasterF32(tensor)
		tensors[name] = master
		moments[name+"_moment_1"] = zeroLikeMaster(master)
		moments[name+"_moment_2"] = zeroLikeMaster(master)
	}
	return EmbeddingTrainCheckpoint{
		Version:       EmbeddingTrainCheckpointVersion,
		Manifest:      manifest,
		Config:        cfg,
		Tensors:       tensors,
		MomentTensors: moments,
	}
}

func initializedTrainingWeights(mod *eosartifact.Module, manifest EmbeddingManifest, opts EmbeddingTrainInitOptions) (map[string]*backend.Tensor, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	seed := opts.Seed
	if seed == 0 {
		seed = 1
	}
	rng := rand.New(rand.NewSource(seed))
	weights := make(map[string]*backend.Tensor, len(mod.Params))
	for _, param := range mod.Params {
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return nil, fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		shape, err := resolveTrainingInitShape(param.Type.Tensor.Shape, manifest.Tokenizer, opts.ShapeSizes)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		tensor, err := initializedWeightTensor(param.Type.Tensor.DType, shape, rng)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", param.Name, err)
		}
		if manifest.roleConditioned() && param.Name == manifest.RoleEmbeddingParam {
			tensor = zeroInitializedTensor(param.Type.Tensor.DType, shape)
		}
		weights[param.Name] = tensor
	}
	return weights, nil
}

func zeroInitializedTensor(dtype string, shape []int) *backend.Tensor {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	data := make([]float32, n)
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data)
	case "q4":
		return backend.NewTensorQ4(shape, data)
	case "q8":
		return backend.NewTensorQ8(shape, data)
	default:
		return backend.NewTensorF32(shape, data)
	}
}

func resolveTrainingInitShape(shape []string, tokenizer TokenizerManifest, sizes map[string]int) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		if n, err := strconv.Atoi(dim); err == nil {
			if n <= 0 {
				return nil, fmt.Errorf("shape dim %q must be positive", dim)
			}
			out[i] = n
			continue
		}
		if sizes != nil && sizes[dim] > 0 {
			out[i] = sizes[dim]
			continue
		}
		switch dim {
		case "V":
			if tokenizer.VocabSize > 0 {
				out[i] = tokenizer.VocabSize
				continue
			}
		case "T":
			if tokenizer.MaxSequence > 0 {
				out[i] = tokenizer.MaxSequence
				continue
			}
		}
		return nil, fmt.Errorf("unresolved symbolic dim %q", dim)
	}
	return out, nil
}

func initializedWeightTensor(dtype string, shape []int, rng *rand.Rand) (*backend.Tensor, error) {
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("shape %v is invalid", shape)
		}
		n *= dim
	}
	scale := initializerScale(shape)
	data := make([]float32, n)
	for i := range data {
		data[i] = (rng.Float32()*2 - 1) * scale
	}
	switch dtype {
	case "f16":
		return backend.NewTensorF16(shape, data), nil
	case "f32":
		return backend.NewTensorF32(shape, data), nil
	case "q4":
		return backend.NewTensorQ4(shape, data), nil
	case "q8":
		return backend.NewTensorQ8(shape, data), nil
	default:
		return nil, fmt.Errorf("unsupported weight dtype %q", dtype)
	}
}

func initializerScale(shape []int) float32 {
	if len(shape) >= 2 {
		fanIn := shape[len(shape)-2]
		fanOut := shape[len(shape)-1]
		if fanIn > 0 && fanOut > 0 {
			return float32(math.Sqrt(2.0 / float64(fanIn+fanOut)))
		}
	}
	if len(shape) == 1 && shape[0] > 0 {
		return float32(1.0 / math.Sqrt(float64(shape[0])))
	}
	return 0.02
}

func bootstrapTrainingWeights(weights map[string]*backend.Tensor, targetMod *eosartifact.Module, targetManifest EmbeddingManifest, opts EmbeddingTrainInitOptions) error {
	if opts.BootstrapInferencePath != "" {
		if opts.BootstrapArtifactPath != "" || opts.BootstrapCheckpointPath != "" {
			return fmt.Errorf("bootstrap inference path is mutually exclusive with bootstrap artifact path and checkpoint path")
		}
		return bootstrapTrainingWeightsFromInference(weights, targetMod, targetManifest, opts)
	}
	checkpointPath, err := resolveBootstrapCheckpointPath(opts)
	if err != nil {
		return err
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(checkpointPath)
	if err != nil {
		return fmt.Errorf("read bootstrap checkpoint %q: %w", checkpointPath, err)
	}
	copies := []struct {
		role       string
		targetName string
		source     *backend.Tensor
	}{
		{role: "token_embedding", targetName: targetManifest.TokenEmbeddingParam, source: checkpoint.TokenEmbedding},
		{role: "role_embedding", targetName: targetManifest.RoleEmbeddingParam, source: checkpoint.RoleEmbedding},
		{role: "attention_query", targetName: targetManifest.AttentionQueryParam, source: checkpoint.AttentionQuery},
		{role: "attention_key", targetName: targetManifest.AttentionKeyParam, source: checkpoint.AttentionKey},
		{role: "attention_value", targetName: targetManifest.AttentionValueParam, source: checkpoint.AttentionValue},
		{role: "attention_output", targetName: targetManifest.AttentionOutputParam, source: checkpoint.AttentionOutput},
		{role: "hidden_projection", targetName: targetManifest.HiddenProjectionParam, source: checkpoint.HiddenProjection},
		{role: "projection", targetName: targetManifest.ProjectionParam, source: checkpoint.Projection},
	}
	copied := map[string]struct{}{}
	for _, copySpec := range copies {
		if copySpec.targetName == "" || copySpec.source == nil {
			continue
		}
		target := weights[copySpec.targetName]
		if target == nil {
			return fmt.Errorf("bootstrap target tensor %q for role %s is missing", copySpec.targetName, copySpec.role)
		}
		if err := copyOverlappingTensor(target, copySpec.source); err != nil {
			return fmt.Errorf("bootstrap %s: %w", copySpec.role, err)
		}
		copied[copySpec.targetName] = struct{}{}
	}
	targetNames := make([]string, 0, len(weights))
	for name := range weights {
		targetNames = append(targetNames, name)
	}
	sort.Strings(targetNames)
	for _, name := range targetNames {
		if _, ok := copied[name]; ok {
			continue
		}
		source := checkpoint.Tensors[name]
		if source == nil {
			continue
		}
		if err := copyOverlappingTensor(weights[name], source); err != nil {
			return fmt.Errorf("bootstrap generic tensor %q into target %q: %w", name, name, err)
		}
	}
	return nil
}

func bootstrapTrainingWeightsFromInference(weights map[string]*backend.Tensor, targetMod *eosartifact.Module, targetManifest EmbeddingManifest, opts EmbeddingTrainInitOptions) error {
	path := opts.BootstrapInferencePath
	if path == "" {
		return fmt.Errorf("bootstrap inference path is required")
	}
	source, err := ReadSealedEmbeddingPackage(path)
	if err != nil {
		return fmt.Errorf("read bootstrap inference artifact %q: %w", path, err)
	}
	sourceManifest := source.Manifest.normalizedForModule(source.Module)
	sourceManifest = inferLegacyInferenceBootstrapManifest(sourceManifest, source.Weights.Weights)
	targetManifest = targetManifest.normalizedForModule(targetMod)
	if opts.BootstrapInferenceWiden {
		if err := validateBootstrapTailInitOptions(opts); err != nil {
			return err
		}
		if _, _, err := widenedInferenceBootstrapManifestAndShapes(targetMod, sourceManifest, source.Weights.Weights, opts.ShapeSizes); err != nil {
			return fmt.Errorf("bootstrap inference artifact %q is incompatible for widen: %w", path, err)
		}
		if normalizeBootstrapTailInit(opts.BootstrapTailInit) == "projection-tail" {
			if err := validateProjectionTailBootstrapContract(weights, source.Weights.Weights, sourceManifest); err != nil {
				return fmt.Errorf("bootstrap inference artifact %q projection-tail contract: %w", path, err)
			}
		}
	} else {
		if err := validateExactBootstrapTailOptions(opts); err != nil {
			return err
		}
		if err := validateInferenceBootstrapManifest(targetManifest, sourceManifest); err != nil {
			return fmt.Errorf("bootstrap inference artifact %q is incompatible: %w", path, err)
		}
	}
	targetNames := make([]string, 0, len(weights))
	for name := range weights {
		targetNames = append(targetNames, name)
	}
	sort.Strings(targetNames)
	for _, name := range targetNames {
		target := weights[name]
		if target == nil {
			return fmt.Errorf("bootstrap inference target tensor %q is nil", name)
		}
		sourceTensor := source.Weights.Weights[name]
		if sourceTensor == nil {
			return fmt.Errorf("bootstrap inference source tensor %q is missing", name)
		}
		if opts.BootstrapInferenceWiden {
			switch normalizeBootstrapTailInit(opts.BootstrapTailInit) {
			case "zero", "projection-tail":
				zeroTensorData(target)
			case "random":
			default:
				return fmt.Errorf("unsupported bootstrap tail init %q", opts.BootstrapTailInit)
			}
			if err := copyOverlappingTensor(target, sourceTensor); err != nil {
				return fmt.Errorf("bootstrap inference tensor %q: %w", name, err)
			}
			if normalizeBootstrapTailInit(opts.BootstrapTailInit) == "projection-tail" {
				if err := applyProjectionTailBootstrapTensor(target, sourceTensor, name, sourceManifest, opts); err != nil {
					return fmt.Errorf("bootstrap inference tensor %q projection-tail: %w", name, err)
				}
			}
		} else {
			if err := copyExactBootstrapTensor(target, sourceTensor); err != nil {
				return fmt.Errorf("bootstrap inference tensor %q: %w", name, err)
			}
		}
	}
	return nil
}

func validateExactBootstrapTailOptions(opts EmbeddingTrainInitOptions) error {
	if strings.TrimSpace(opts.BootstrapTailInit) != "" {
		return fmt.Errorf("bootstrap tail init requires bootstrap inference widen")
	}
	if opts.BootstrapTailScale != 0 {
		return fmt.Errorf("bootstrap tail scale requires bootstrap inference widen")
	}
	return nil
}

func validateBootstrapTailInitOptions(opts EmbeddingTrainInitOptions) error {
	switch normalizeBootstrapTailInit(opts.BootstrapTailInit) {
	case "zero", "random":
		if opts.BootstrapTailScale != 0 {
			return fmt.Errorf("bootstrap tail scale requires bootstrap tail init projection-tail")
		}
	case "projection-tail":
		if opts.BootstrapTailScale != 0 {
			if math.IsNaN(opts.BootstrapTailScale) || math.IsInf(opts.BootstrapTailScale, 0) || opts.BootstrapTailScale <= 0 || opts.BootstrapTailScale > 0.10 {
				return fmt.Errorf("bootstrap tail scale must be finite and in (0, 0.10] for projection-tail, got %g", opts.BootstrapTailScale)
			}
		}
	default:
		return fmt.Errorf("unsupported bootstrap tail init %q", opts.BootstrapTailInit)
	}
	return nil
}

func validateProjectionTailBootstrapContract(targetWeights, sourceWeights map[string]*backend.Tensor, manifest EmbeddingManifest) error {
	hiddenName := manifest.HiddenProjectionParam
	projectionName := manifest.ProjectionParam
	if hiddenName == "" || projectionName == "" {
		return fmt.Errorf("recognized hidden projection and projection params are required")
	}
	sourceHidden := sourceWeights[hiddenName]
	targetHidden := targetWeights[hiddenName]
	sourceProjection := sourceWeights[projectionName]
	targetProjection := targetWeights[projectionName]
	if sourceHidden == nil || targetHidden == nil {
		return fmt.Errorf("hidden projection tensor %q is required", hiddenName)
	}
	if sourceProjection == nil || targetProjection == nil {
		return fmt.Errorf("projection tensor %q is required", projectionName)
	}
	if len(sourceHidden.Shape) != 2 || len(targetHidden.Shape) != 2 {
		return fmt.Errorf("hidden projection tensor %q rank source=%d target=%d, want 2", hiddenName, len(sourceHidden.Shape), len(targetHidden.Shape))
	}
	if len(sourceProjection.Shape) != 2 || len(targetProjection.Shape) != 2 {
		return fmt.Errorf("projection tensor %q rank source=%d target=%d, want 2", projectionName, len(sourceProjection.Shape), len(targetProjection.Shape))
	}
	if targetHidden.Shape[0] <= sourceHidden.Shape[0] || targetHidden.Shape[1] <= sourceHidden.Shape[1] {
		return fmt.Errorf("hidden projection tensor %q must grow D/H, source=%v target=%v", hiddenName, sourceHidden.Shape, targetHidden.Shape)
	}
	if targetProjection.Shape[0] <= sourceProjection.Shape[0] || targetProjection.Shape[1] <= sourceProjection.Shape[1] {
		return fmt.Errorf("projection tensor %q must grow H/D, source=%v target=%v", projectionName, sourceProjection.Shape, targetProjection.Shape)
	}
	if sourceHidden.Shape[0] != sourceProjection.Shape[1] || sourceHidden.Shape[1] != sourceProjection.Shape[0] {
		return fmt.Errorf("source hidden/projection shapes are not D,H and H,D: hidden=%v projection=%v", sourceHidden.Shape, sourceProjection.Shape)
	}
	if targetHidden.Shape[0] != targetProjection.Shape[1] || targetHidden.Shape[1] != targetProjection.Shape[0] {
		return fmt.Errorf("target hidden/projection shapes are not D,H and H,D: hidden=%v projection=%v", targetHidden.Shape, targetProjection.Shape)
	}
	return nil
}

func applyProjectionTailBootstrapTensor(target, source *backend.Tensor, name string, manifest EmbeddingManifest, opts EmbeddingTrainInitOptions) error {
	if target == nil || source == nil {
		return fmt.Errorf("nil tensor")
	}
	if len(target.Shape) != 2 || len(source.Shape) != 2 {
		return nil
	}
	switch name {
	case manifest.HiddenProjectionParam:
		scale := projectionTailNoiseScale(target.Shape, opts.BootstrapTailScale)
		oldD, oldH := source.Shape[0], source.Shape[1]
		newH := target.Shape[1]
		for row := 0; row < oldD; row++ {
			for col := oldH; col < newH; col++ {
				target.F32[row*newH+col] = deterministicProjectionTailNoise(opts.Seed, name, row, col, scale)
			}
		}
	case manifest.ProjectionParam:
		scale := projectionTailNoiseScale(target.Shape, opts.BootstrapTailScale)
		oldD := source.Shape[1]
		newD := target.Shape[1]
		for row := 0; row < target.Shape[0]; row++ {
			for col := oldD; col < newD; col++ {
				target.F32[row*newD+col] = deterministicProjectionTailNoise(opts.Seed, name, row, col, scale)
			}
		}
	}
	return nil
}

func projectionTailNoiseScale(shape []int, configured float64) float32 {
	scale := configured
	if scale == 0 {
		scale = 0.03
	}
	return float32(scale) * initializerScale(shape)
}

func deterministicProjectionTailNoise(seed int64, tensorName string, row, col int, scale float32) float32 {
	if seed == 0 {
		seed = 1
	}
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(seed))
	_, _ = h.Write(buf[:])
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(tensorName))
	_, _ = h.Write([]byte{0})
	binary.LittleEndian.PutUint64(buf[:], uint64(row))
	_, _ = h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], uint64(col))
	_, _ = h.Write(buf[:])
	unit := float64(h.Sum64()>>11) * (1.0 / (1 << 53))
	return float32(unit*2-1) * scale
}

func widenedInferenceBootstrapManifestAndShapes(mod *eosartifact.Module, sourceManifest EmbeddingManifest, sourceWeights map[string]*backend.Tensor, overrides map[string]int) (EmbeddingManifest, map[string]int, error) {
	if mod == nil {
		return EmbeddingManifest{}, nil, fmt.Errorf("nil bootstrap inference module")
	}
	sourceShapeSizes := inferenceBootstrapShapeSizes(sourceManifest, sourceWeights)
	sourceD := sourceShapeSizes["D"]
	sourceH := sourceShapeSizes["H"]
	if sourceD <= 0 || sourceH <= 0 {
		return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference source must expose positive D/H dimensions, got D=%d H=%d", sourceD, sourceH)
	}
	targetD := overrides["D"]
	targetH := overrides["H"]
	if targetD <= 0 || targetH <= 0 {
		return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference widen requires target D and H shape sizes")
	}
	if targetD <= sourceD {
		return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference widen requires model_dim growth: target=%d source=%d", targetD, sourceD)
	}
	if targetH <= sourceH {
		return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference widen requires ffn_dim growth: target=%d source=%d", targetH, sourceH)
	}
	for _, alias := range []string{"O", "E"} {
		if overrides[alias] > 0 && overrides[alias] != targetD {
			return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference widen requires %s=%d to match model_dim, got %d", alias, targetD, overrides[alias])
		}
	}
	if sourceManifest.OutputDim != 0 && sourceManifest.OutputDim != sourceD {
		return EmbeddingManifest{}, nil, fmt.Errorf("bootstrap inference widen requires source output_dim tied to model_dim, got output_dim=%d model_dim=%d", sourceManifest.OutputDim, sourceD)
	}
	targetShapeSizes := mergeTrainingShapeSizes(sourceShapeSizes, map[string]int{
		"D": targetD,
		"H": targetH,
		"O": targetD,
		"E": targetD,
	})
	if err := validateInferenceWidenGraphShapes(mod, sourceManifest.Tokenizer, sourceShapeSizes, targetShapeSizes); err != nil {
		return EmbeddingManifest{}, nil, err
	}
	target := sourceManifest
	target.ModelDim = targetD
	target.OutputDim = targetD
	target.FFNDim = targetH
	if target.AttentionHeads <= 0 {
		target.AttentionHeads = 1
	}
	if target.AttentionHeads != sourceManifest.AttentionHeads {
		return EmbeddingManifest{}, nil, fmt.Errorf("attention_heads target=%d source=%d", target.AttentionHeads, sourceManifest.AttentionHeads)
	}
	if targetD%target.AttentionHeads != 0 {
		return EmbeddingManifest{}, nil, fmt.Errorf("model_dim %d must be divisible by attention_heads %d", targetD, target.AttentionHeads)
	}
	target.HeadDim = targetD / target.AttentionHeads
	if target.ArchitectureVersion == "" {
		target.ArchitectureVersion = EmbeddingArchitectureLegacyV1
	}
	if target.ParameterTying == "" {
		target.ParameterTying = EmbeddingParameterTyingLegacyTied
	}
	if err := target.ValidateModule(mod); err != nil {
		return EmbeddingManifest{}, nil, fmt.Errorf("widened bootstrap inference embedding manifest: %w", err)
	}
	return target, targetShapeSizes, nil
}

func validateInferenceWidenGraphShapes(mod *eosartifact.Module, tokenizer TokenizerManifest, sourceSizes, targetSizes map[string]int) error {
	for _, param := range mod.Params {
		if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			return fmt.Errorf("param %q is not a tensor weight", param.Name)
		}
		if err := validateInferenceWidenShape("param "+param.Name, param.Type.Tensor.Shape, tokenizer, sourceSizes, targetSizes); err != nil {
			return err
		}
	}
	for _, buffer := range mod.Buffers {
		if err := validateInferenceWidenShape("buffer "+buffer.Name, buffer.Shape, tokenizer, sourceSizes, targetSizes); err != nil {
			return err
		}
	}
	for _, entry := range mod.EntryPoints {
		for _, input := range entry.Inputs {
			if err := validateInferenceWidenValueType("entrypoint "+entry.Name+" input "+input.Name, input.Type, tokenizer, sourceSizes, targetSizes); err != nil {
				return err
			}
		}
		for _, output := range entry.Outputs {
			if err := validateInferenceWidenValueType("entrypoint "+entry.Name+" output "+output.Name, output.Type, tokenizer, sourceSizes, targetSizes); err != nil {
				return err
			}
		}
	}
	for _, kernel := range mod.Kernels {
		for _, input := range kernel.Inputs {
			if err := validateInferenceWidenValueType("kernel "+kernel.Name+" input "+input.Name, input.Type, tokenizer, sourceSizes, targetSizes); err != nil {
				return err
			}
		}
		for _, output := range kernel.Outputs {
			if err := validateInferenceWidenValueType("kernel "+kernel.Name+" output "+output.Name, output.Type, tokenizer, sourceSizes, targetSizes); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateInferenceWidenValueType(surface string, valueType eosartifact.ValueType, tokenizer TokenizerManifest, sourceSizes, targetSizes map[string]int) error {
	if valueType.Tensor != nil {
		return validateInferenceWidenShape(surface, valueType.Tensor.Shape, tokenizer, sourceSizes, targetSizes)
	}
	if valueType.CandidatePack != nil {
		return validateInferenceWidenShape(surface, valueType.CandidatePack.Shape, tokenizer, sourceSizes, targetSizes)
	}
	return nil
}

func validateInferenceWidenShape(surface string, shape []string, tokenizer TokenizerManifest, sourceSizes, targetSizes map[string]int) error {
	for i, dim := range shape {
		sourceDim, sourceOK := resolveInferenceWidenGraphDim(dim, tokenizer, sourceSizes)
		targetDim, targetOK := resolveInferenceWidenGraphDim(dim, tokenizer, targetSizes)
		if sourceOK != targetOK {
			return fmt.Errorf("%s dim %d (%q) resolves inconsistently between source and target", surface, i, dim)
		}
		if !sourceOK {
			continue
		}
		if sourceDim == targetDim {
			if isConcreteOldWidenDim(dim, sourceSizes, targetSizes) {
				return fmt.Errorf("%s dim %d is concrete old widened dimension %q; use symbolic D/H/O/E for graph-preserving widen", surface, i, dim)
			}
			continue
		}
		if !isInferenceWidenDim(dim) {
			return fmt.Errorf("%s dim %d (%q) changed source=%d target=%d; only D/H/O/E may widen", surface, i, dim, sourceDim, targetDim)
		}
		if targetDim < sourceDim {
			return fmt.Errorf("%s dim %d shrank source=%d target=%d", surface, i, sourceDim, targetDim)
		}
	}
	return nil
}

func resolveInferenceWidenGraphDim(dim string, tokenizer TokenizerManifest, sizes map[string]int) (int, bool) {
	if n, err := strconv.Atoi(dim); err == nil {
		return n, n > 0
	}
	if sizes != nil && sizes[dim] > 0 {
		return sizes[dim], true
	}
	switch dim {
	case "V":
		if tokenizer.VocabSize > 0 {
			return tokenizer.VocabSize, true
		}
	case "T":
		if tokenizer.MaxSequence > 0 {
			return tokenizer.MaxSequence, true
		}
	}
	return 0, false
}

func isConcreteOldWidenDim(dim string, sourceSizes, targetSizes map[string]int) bool {
	n, err := strconv.Atoi(dim)
	if err != nil || n <= 0 {
		return false
	}
	for _, symbol := range []string{"D", "H", "O", "E"} {
		if sourceSizes[symbol] > 0 && targetSizes[symbol] > 0 && sourceSizes[symbol] != targetSizes[symbol] && n == sourceSizes[symbol] {
			return true
		}
	}
	return false
}

func isInferenceWidenDim(dim string) bool {
	switch dim {
	case "D", "H", "O", "E":
		return true
	default:
		return false
	}
}

func normalizeBootstrapTailInit(mode string) string {
	mode = strings.TrimSpace(strings.ToLower(mode))
	if mode == "" {
		return "zero"
	}
	return mode
}

func zeroTensorData(tensor *backend.Tensor) {
	if tensor == nil {
		return
	}
	for i := range tensor.F32 {
		tensor.F32[i] = 0
	}
}

func cloneEmbeddingInitModule(mod *eosartifact.Module) (*eosartifact.Module, error) {
	if mod == nil {
		return nil, fmt.Errorf("nil module")
	}
	data, err := json.Marshal(mod)
	if err != nil {
		return nil, err
	}
	var out eosartifact.Module
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func preserveSealedBootstrapPackagePolicies(paths EmbeddingTrainPackagePaths, sourcePackageManifest *PackageManifest) error {
	if sourcePackageManifest == nil {
		return nil
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		return fmt.Errorf("read target train manifest for bootstrap policy preservation: %w", err)
	}
	trainManifest.ScoreSpectrum = cloneScoreSpectrumPackagePolicy(sourcePackageManifest.ScoreSpectrum)
	trainManifest.ListwiseGeometry = cloneListwiseGeometryPackagePolicy(sourcePackageManifest.ListwiseGeometry)
	if err := trainManifest.WriteFile(paths.TrainManifestPath); err != nil {
		return fmt.Errorf("write target train manifest with bootstrap policies: %w", err)
	}
	packageManifest, _, err := RebuildSiblingPackageManifest(paths.ArtifactPath)
	if err != nil {
		return fmt.Errorf("rebuild target package manifest for bootstrap policy preservation: %w", err)
	}
	packageManifest.ScoreSpectrum = cloneScoreSpectrumPackagePolicy(sourcePackageManifest.ScoreSpectrum)
	packageManifest.ListwiseGeometry = cloneListwiseGeometryPackagePolicy(sourcePackageManifest.ListwiseGeometry)
	if err := packageManifest.WriteFile(paths.PackageManifestPath); err != nil {
		return fmt.Errorf("write target package manifest with bootstrap policies: %w", err)
	}
	return nil
}

func cloneScoreSpectrumPackagePolicy(policy EmbeddingScoreSpectrumPolicy) EmbeddingScoreSpectrumPolicy {
	policy.SourceArtifactHashes = append([]string(nil), policy.SourceArtifactHashes...)
	policy.AutoClearedObjectives = append([]string(nil), policy.AutoClearedObjectives...)
	policy.IsolatedInheritedObjectives = append([]string(nil), policy.IsolatedInheritedObjectives...)
	return policy
}

func cloneListwiseGeometryPackagePolicy(policy EmbeddingListwiseGeometryPolicy) EmbeddingListwiseGeometryPolicy {
	policy.SourceArtifactHashes = append([]string(nil), policy.SourceArtifactHashes...)
	policy.AutoClearedObjectives = append([]string(nil), policy.AutoClearedObjectives...)
	policy.IsolatedInheritedObjectives = append([]string(nil), policy.IsolatedInheritedObjectives...)
	return policy
}

func validateInferenceBootstrapManifest(target, source EmbeddingManifest) error {
	if target.ArchitectureVersion != source.ArchitectureVersion {
		return fmt.Errorf("architecture_version target=%q source=%q", target.ArchitectureVersion, source.ArchitectureVersion)
	}
	if target.ParameterTying != source.ParameterTying {
		return fmt.Errorf("parameter_tying target=%q source=%q", target.ParameterTying, source.ParameterTying)
	}
	if target.ModelDim != source.ModelDim {
		return fmt.Errorf("model_dim target=%d source=%d", target.ModelDim, source.ModelDim)
	}
	if target.OutputDim != source.OutputDim {
		return fmt.Errorf("output_dim target=%d source=%d", target.OutputDim, source.OutputDim)
	}
	if target.AttentionHeads != source.AttentionHeads {
		return fmt.Errorf("attention_heads target=%d source=%d", target.AttentionHeads, source.AttentionHeads)
	}
	if target.HeadDim != source.HeadDim {
		return fmt.Errorf("head_dim target=%d source=%d", target.HeadDim, source.HeadDim)
	}
	if target.FFNDim != source.FFNDim {
		return fmt.Errorf("ffn_dim target=%d source=%d", target.FFNDim, source.FFNDim)
	}
	if target.EncoderRepeats != source.EncoderRepeats {
		return fmt.Errorf("encoder_repeats target=%d source=%d", target.EncoderRepeats, source.EncoderRepeats)
	}
	if target.Tokenizer.VocabSize != source.Tokenizer.VocabSize {
		return fmt.Errorf("tokenizer vocab_size target=%d source=%d", target.Tokenizer.VocabSize, source.Tokenizer.VocabSize)
	}
	if target.Tokenizer.MaxSequence != source.Tokenizer.MaxSequence {
		return fmt.Errorf("tokenizer max_sequence target=%d source=%d", target.Tokenizer.MaxSequence, source.Tokenizer.MaxSequence)
	}
	return nil
}

func inferLegacyInferenceBootstrapManifest(manifest EmbeddingManifest, weights map[string]*backend.Tensor) EmbeddingManifest {
	if manifest.ArchitectureVersion != "" {
		return manifest
	}
	if manifest.ParameterTying != "" && manifest.ParameterTying != EmbeddingParameterTyingLegacyTied {
		return manifest
	}
	tokenName := manifest.TokenEmbeddingParam
	if tokenName == "" {
		tokenName = "token_embedding"
	}
	projectionName := manifest.ProjectionParam
	if projectionName == "" {
		projectionName = "projection"
	}
	hiddenName := manifest.HiddenProjectionParam
	if hiddenName == "" {
		hiddenName = "ffn_up"
	}
	queryName := manifest.AttentionQueryParam
	if queryName == "" {
		queryName = "attn_q"
	}
	keyName := manifest.AttentionKeyParam
	if keyName == "" {
		keyName = "attn_k"
	}
	valueName := manifest.AttentionValueParam
	if valueName == "" {
		valueName = "attn_v"
	}
	outputName := manifest.AttentionOutputParam
	if outputName == "" {
		outputName = "attn_o"
	}
	tokenRows, modelDim, ok := tensorMatrixShape(weights[tokenName])
	if !ok || manifest.Tokenizer.VocabSize > 0 && tokenRows != manifest.Tokenizer.VocabSize {
		return manifest
	}
	hiddenRows, ffnDim, ok := tensorMatrixShape(weights[hiddenName])
	if !ok || hiddenRows != modelDim {
		return manifest
	}
	projRows, outputDim, ok := tensorMatrixShape(weights[projectionName])
	if !ok || projRows != ffnDim || outputDim != modelDim {
		return manifest
	}
	for _, name := range []string{queryName, keyName, valueName, outputName} {
		rows, cols, ok := tensorMatrixShape(weights[name])
		if !ok || rows != modelDim || cols != modelDim {
			return manifest
		}
	}
	manifest.ArchitectureVersion = EmbeddingArchitectureLegacyV1
	manifest.ParameterTying = EmbeddingParameterTyingLegacyTied
	manifest.ModelDim = modelDim
	manifest.OutputDim = outputDim
	manifest.FFNDim = ffnDim
	manifest.AttentionHeads = 1
	manifest.HeadDim = modelDim
	manifest.TokenEmbeddingParam = tokenName
	manifest.ProjectionParam = projectionName
	manifest.HiddenProjectionParam = hiddenName
	manifest.AttentionQueryParam = queryName
	manifest.AttentionKeyParam = keyName
	manifest.AttentionValueParam = valueName
	manifest.AttentionOutputParam = outputName
	return manifest
}

func tensorMatrixShape(tensor *backend.Tensor) (int, int, bool) {
	if tensor == nil || len(tensor.Shape) != 2 {
		return 0, 0, false
	}
	if tensor.Shape[0] <= 0 || tensor.Shape[1] <= 0 {
		return 0, 0, false
	}
	return tensor.Shape[0], tensor.Shape[1], true
}

func inferenceBootstrapShapeSizes(manifest EmbeddingManifest, weights map[string]*backend.Tensor) map[string]int {
	out := map[string]int{}
	if manifest.Tokenizer.VocabSize > 0 {
		out["V"] = manifest.Tokenizer.VocabSize
	}
	if manifest.Tokenizer.MaxSequence > 0 {
		out["T"] = manifest.Tokenizer.MaxSequence
	}
	if manifest.ModelDim > 0 {
		out["D"] = manifest.ModelDim
	}
	if manifest.FFNDim > 0 {
		out["H"] = manifest.FFNDim
	}
	if manifest.OutputDim > 0 {
		out["O"] = manifest.OutputDim
		out["E"] = manifest.OutputDim
	}
	if rows, cols, ok := tensorMatrixShape(weights[manifest.TokenEmbeddingParam]); ok {
		out["V"] = rows
		out["D"] = cols
	}
	if rows, cols, ok := tensorMatrixShape(weights[manifest.HiddenProjectionParam]); ok {
		if out["D"] == 0 {
			out["D"] = rows
		}
		out["H"] = cols
	}
	if rows, cols, ok := tensorMatrixShape(weights[manifest.ProjectionParam]); ok {
		if rows == out["H"] && cols > 0 {
			out["O"] = cols
			out["E"] = cols
		} else if rows == out["D"] && cols > 0 {
			out["O"] = cols
			out["E"] = cols
		}
	}
	return out
}

func mergeTrainingShapeSizes(base, overrides map[string]int) map[string]int {
	if len(base) == 0 && len(overrides) == 0 {
		return nil
	}
	out := make(map[string]int, len(base)+len(overrides))
	for name, value := range base {
		if value > 0 {
			out[name] = value
		}
	}
	for name, value := range overrides {
		if value > 0 {
			out[name] = value
		}
	}
	return out
}

func copyExactBootstrapTensor(target, source *backend.Tensor) error {
	if target == nil || source == nil {
		return fmt.Errorf("nil tensor")
	}
	if target.DType != source.DType {
		return fmt.Errorf("dtype mismatch target=%q source=%q", target.DType, source.DType)
	}
	if len(target.Shape) != len(source.Shape) {
		return fmt.Errorf("rank mismatch target=%d source=%d", len(target.Shape), len(source.Shape))
	}
	for i := range target.Shape {
		if target.Shape[i] != source.Shape[i] {
			return fmt.Errorf("shape mismatch target=%v source=%v", target.Shape, source.Shape)
		}
	}
	return copyOverlappingTensor(target, source)
}

func resolveBootstrapCheckpointPath(opts EmbeddingTrainInitOptions) (string, error) {
	if opts.BootstrapArtifactPath != "" && opts.BootstrapCheckpointPath != "" {
		return "", fmt.Errorf("bootstrap artifact path and checkpoint path are mutually exclusive")
	}
	if opts.BootstrapCheckpointPath != "" {
		if _, err := os.Stat(opts.BootstrapCheckpointPath); err != nil {
			return "", fmt.Errorf("bootstrap checkpoint %q: %w", opts.BootstrapCheckpointPath, err)
		}
		return opts.BootstrapCheckpointPath, nil
	}
	if opts.BootstrapArtifactPath == "" {
		return "", fmt.Errorf("bootstrap artifact path is required")
	}
	if _, err := os.Stat(opts.BootstrapArtifactPath); err != nil {
		return "", fmt.Errorf("bootstrap artifact %q: %w", opts.BootstrapArtifactPath, err)
	}
	if strings.HasSuffix(opts.BootstrapArtifactPath, ".embed-train.mll") {
		return opts.BootstrapArtifactPath, nil
	}
	checkpointPath := DefaultEmbeddingCheckpointPath(opts.BootstrapArtifactPath)
	if _, err := os.Stat(checkpointPath); err != nil {
		return "", fmt.Errorf("bootstrap checkpoint %q derived from artifact %q: %w", checkpointPath, opts.BootstrapArtifactPath, err)
	}
	return checkpointPath, nil
}

func (opts EmbeddingTrainInitOptions) hasBootstrapSource() bool {
	return opts.BootstrapArtifactPath != "" || opts.BootstrapCheckpointPath != "" || opts.BootstrapInferencePath != ""
}

func copyOverlappingTensor(target, source *backend.Tensor) error {
	if target == nil || source == nil {
		return fmt.Errorf("nil tensor")
	}
	if len(target.Shape) != len(source.Shape) {
		return fmt.Errorf("rank mismatch target=%d source=%d", len(target.Shape), len(source.Shape))
	}
	if len(target.F32) != target.Elements() {
		return fmt.Errorf("target tensor shape %v has %d f32 values, want %d", target.Shape, len(target.F32), target.Elements())
	}
	if len(source.F32) != source.Elements() {
		return fmt.Errorf("source tensor shape %v has %d f32 values, want %d", source.Shape, len(source.F32), source.Elements())
	}
	limits := make([]int, len(target.Shape))
	for i := range target.Shape {
		if target.Shape[i] <= 0 || source.Shape[i] <= 0 {
			return fmt.Errorf("invalid shapes target=%v source=%v", target.Shape, source.Shape)
		}
		limits[i] = min(target.Shape[i], source.Shape[i])
	}
	copyOverlappingTensorAtRank(target.F32, target.Shape, source.F32, source.Shape, limits, 0, 0, 0)
	return nil
}

func copyOverlappingTensorAtRank(targetData []float32, targetShape []int, sourceData []float32, sourceShape []int, limits []int, rank int, targetOffset int, sourceOffset int) {
	if rank == len(limits)-1 {
		copy(targetData[targetOffset:targetOffset+limits[rank]], sourceData[sourceOffset:sourceOffset+limits[rank]])
		return
	}
	targetStride := rowMajorStride(targetShape, rank)
	sourceStride := rowMajorStride(sourceShape, rank)
	for i := 0; i < limits[rank]; i++ {
		copyOverlappingTensorAtRank(targetData, targetShape, sourceData, sourceShape, limits, rank+1, targetOffset+i*targetStride, sourceOffset+i*sourceStride)
	}
}

func rowMajorStride(shape []int, dim int) int {
	stride := 1
	for i := dim + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	return stride
}
