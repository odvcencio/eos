package eosruntime

import (
	"fmt"
	"os"
	"regexp"
	"strconv"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

var compactLayerTensorNameRE = regexp.MustCompile(`^layer([0-9]+)_(attn_[qkvo]|ffn_(up|down))$`)

// CompactEmbeddingTrainTensor is one named compact train-state tensor plus any
// optimizer moments currently present in the checkpoint.
type CompactEmbeddingTrainTensor struct {
	Name    string
	Tensor  *backend.Tensor
	Moment1 *backend.Tensor
	Moment2 *backend.Tensor
}

// CompactEmbeddingTrainLayer is the typed per-layer compact_transformer_v1 state.
type CompactEmbeddingTrainLayer struct {
	Index           int
	AttentionQuery  CompactEmbeddingTrainTensor
	AttentionKey    CompactEmbeddingTrainTensor
	AttentionValue  CompactEmbeddingTrainTensor
	AttentionOutput CompactEmbeddingTrainTensor
	FFNUp           CompactEmbeddingTrainTensor
	FFNDown         CompactEmbeddingTrainTensor
}

// CompactEmbeddingTrainState is the structured state contract for
// compact_transformer_v1 checkpoints. It is validation/loading only; compact
// forward/backward training math remains unsupported.
type CompactEmbeddingTrainState struct {
	Manifest         EmbeddingManifest
	Config           EmbeddingTrainConfig
	Step             int
	TokenEmbedding   CompactEmbeddingTrainTensor
	RoleEmbedding    *CompactEmbeddingTrainTensor
	Layers           []CompactEmbeddingTrainLayer
	OutputProjection *CompactEmbeddingTrainTensor
}

// LoadCompactEmbeddingTrainStateFromPackage reads a compact training package and
// returns its typed checkpoint state without instantiating the legacy trainer.
func LoadCompactEmbeddingTrainStateFromPackage(artifactPath string) (*CompactEmbeddingTrainState, error) {
	return LoadCompactEmbeddingTrainStateFromPackageWithPaths(EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: ResolveEmbeddingManifestPath(artifactPath),
		TokenizerPath:         DefaultTokenizerPath(artifactPath),
		WeightFilePath:        DefaultWeightFilePath(artifactPath),
		MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
		TrainManifestPath:     ResolveEmbeddingTrainManifestPath(artifactPath),
		CheckpointPath:        DefaultEmbeddingCheckpointPath(artifactPath),
		TrainProfilePath:      DefaultEmbeddingTrainProfilePath(artifactPath),
		PackageManifestPath:   ResolvePackageManifestPath(artifactPath),
	})
}

// LoadCompactEmbeddingTrainStateFromPackageWithPaths reads a compact training
// package using explicit file paths and returns its typed checkpoint state.
func LoadCompactEmbeddingTrainStateFromPackageWithPaths(paths EmbeddingTrainPackagePaths) (*CompactEmbeddingTrainState, error) {
	if paths.PackageManifestPath != "" {
		if _, err := os.Stat(paths.PackageManifestPath); err == nil {
			manifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
			if err != nil {
				return nil, err
			}
			if err := manifest.VerifyFiles(map[string]string{
				"artifact":           paths.ArtifactPath,
				"embedding_manifest": paths.EmbeddingManifestPath,
				"tokenizer":          paths.TokenizerPath,
				"weights":            paths.WeightFilePath,
				"memory_plan":        paths.MemoryPlanPath,
				"train_manifest":     paths.TrainManifestPath,
				"checkpoint":         paths.CheckpointPath,
				"train_profile":      paths.TrainProfilePath,
			}); err != nil {
				return nil, err
			}
		}
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		return nil, err
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		return nil, err
	}
	embeddingManifest := trainManifest.Embedding.normalizedForModule(mod)
	if err := embeddingManifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		return nil, err
	}
	return LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, embeddingManifest)
}

// LoadCompactEmbeddingTrainStateFromCheckpoint validates a compact checkpoint
// against the supplied manifest metadata and returns typed state.
func LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint EmbeddingTrainCheckpoint, manifest EmbeddingManifest) (*CompactEmbeddingTrainState, error) {
	if err := checkpoint.Validate(); err != nil {
		return nil, err
	}
	manifest = manifest.normalized()
	if manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return nil, fmt.Errorf("compact train state requires architecture_version=%q, got %q", EmbeddingArchitectureCompactTransformerV1, manifest.ArchitectureVersion)
	}
	if checkpoint.Manifest.normalized().ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return nil, fmt.Errorf("compact train state checkpoint requires architecture_version=%q, got %q", EmbeddingArchitectureCompactTransformerV1, checkpoint.Manifest.normalized().ArchitectureVersion)
	}
	if err := validateCompactTrainManifestCompatibility(manifest, checkpoint.Manifest.normalized()); err != nil {
		return nil, err
	}
	if manifest.EncoderRepeats <= 0 {
		return nil, fmt.Errorf("compact train state encoder_repeats must be positive")
	}
	if manifest.ModelDim <= 0 {
		return nil, fmt.Errorf("compact train state model_dim must be positive")
	}
	if manifest.OutputDim <= 0 {
		return nil, fmt.Errorf("compact train state output_dim must be positive")
	}
	if manifest.FFNDim <= 0 {
		return nil, fmt.Errorf("compact train state ffn_dim must be positive")
	}
	if manifest.Tokenizer.VocabSize <= 0 {
		return nil, fmt.Errorf("compact train state tokenizer.vocab_size must be positive")
	}
	if err := validateCompactTrainTensorNames(manifest); err != nil {
		return nil, err
	}

	state := &CompactEmbeddingTrainState{
		Manifest: manifest,
		Config:   checkpoint.Config,
		Step:     checkpoint.Step,
		Layers:   make([]CompactEmbeddingTrainLayer, manifest.EncoderRepeats),
	}
	var err error
	state.TokenEmbedding, err = requireCompactTrainTensor(checkpoint, compactTokenEmbeddingName(manifest), []int{manifest.Tokenizer.VocabSize, manifest.ModelDim})
	if err != nil {
		return nil, err
	}
	if manifest.roleConditioned() {
		role, err := requireCompactTrainTensor(checkpoint, compactRoleEmbeddingName(manifest), []int{3, manifest.ModelDim})
		if err != nil {
			return nil, err
		}
		state.RoleEmbedding = &role
	}
	for i := 0; i < manifest.EncoderRepeats; i++ {
		layer, err := loadCompactTrainLayer(checkpoint, i, manifest.ModelDim, manifest.FFNDim)
		if err != nil {
			return nil, err
		}
		state.Layers[i] = layer
	}
	if err := validateCompactLayerSet(checkpoint, manifest.EncoderRepeats); err != nil {
		return nil, err
	}
	if manifest.OutputDim != manifest.ModelDim {
		name := compactOutputProjectionName(manifest)
		proj, err := requireCompactTrainTensor(checkpoint, name, []int{manifest.ModelDim, manifest.OutputDim})
		if err != nil {
			return nil, err
		}
		state.OutputProjection = &proj
	} else {
		if manifest.OutputProjectionParam != "" {
			return nil, fmt.Errorf("compact train state output_projection_param %q is invalid when output_dim (%d) equals model_dim (%d)", manifest.OutputProjectionParam, manifest.OutputDim, manifest.ModelDim)
		}
		if checkpoint.Tensors["output_projection"] != nil {
			return nil, fmt.Errorf("compact train state tensor %q is present but output_dim (%d) equals model_dim (%d)", "output_projection", manifest.OutputDim, manifest.ModelDim)
		}
	}
	return state, nil
}

func validateCompactTrainTensorNames(manifest EmbeddingManifest) error {
	names := []struct {
		role string
		name string
	}{
		{role: "token_embedding", name: compactTokenEmbeddingName(manifest)},
	}
	if manifest.roleConditioned() {
		names = append(names, struct {
			role string
			name string
		}{role: "role_embedding", name: compactRoleEmbeddingName(manifest)})
	}
	for i := 0; i < manifest.EncoderRepeats; i++ {
		for _, suffix := range []string{"attn_q", "attn_k", "attn_v", "attn_o", "ffn_up", "ffn_down"} {
			names = append(names, struct {
				role string
				name string
			}{
				role: fmt.Sprintf("layer%d_%s", i, suffix),
				name: compactLayerTensorName(i, suffix),
			})
		}
	}
	if manifest.OutputDim != manifest.ModelDim {
		names = append(names, struct {
			role string
			name string
		}{role: "output_projection", name: compactOutputProjectionName(manifest)})
	}
	seen := map[string]string{}
	for _, item := range names {
		if item.name == "" {
			return fmt.Errorf("compact train state tensor name for %s is required", item.role)
		}
		if prior, ok := seen[item.name]; ok {
			return fmt.Errorf("compact train state tensor name %q is used by both %s and %s", item.name, prior, item.role)
		}
		seen[item.name] = item.role
	}
	return nil
}

func loadCompactTrainLayer(checkpoint EmbeddingTrainCheckpoint, index, modelDim, ffnDim int) (CompactEmbeddingTrainLayer, error) {
	layer := CompactEmbeddingTrainLayer{Index: index}
	var err error
	layer.AttentionQuery, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "attn_q"), []int{modelDim, modelDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	layer.AttentionKey, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "attn_k"), []int{modelDim, modelDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	layer.AttentionValue, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "attn_v"), []int{modelDim, modelDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	layer.AttentionOutput, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "attn_o"), []int{modelDim, modelDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	layer.FFNUp, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "ffn_up"), []int{modelDim, ffnDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	layer.FFNDown, err = requireCompactTrainTensor(checkpoint, compactLayerTensorName(index, "ffn_down"), []int{ffnDim, modelDim})
	if err != nil {
		return CompactEmbeddingTrainLayer{}, err
	}
	return layer, nil
}

func requireCompactTrainTensor(checkpoint EmbeddingTrainCheckpoint, name string, shape []int) (CompactEmbeddingTrainTensor, error) {
	if name == "" {
		return CompactEmbeddingTrainTensor{}, fmt.Errorf("compact train state tensor name is required")
	}
	tensor := checkpoint.Tensors[name]
	if tensor == nil {
		return CompactEmbeddingTrainTensor{}, fmt.Errorf("compact train state tensor %q is required", name)
	}
	if !tensorShapeEquals(tensor, shape) {
		return CompactEmbeddingTrainTensor{}, fmt.Errorf("compact train state tensor %q shape %v, want %v", name, tensor.Shape, shape)
	}
	out := CompactEmbeddingTrainTensor{
		Name:   name,
		Tensor: tensor.Clone(),
	}
	moment1, err := optionalCompactMoment(checkpoint, name, "_moment_1", shape)
	if err != nil {
		return CompactEmbeddingTrainTensor{}, err
	}
	moment2, err := optionalCompactMoment(checkpoint, name, "_moment_2", shape)
	if err != nil {
		return CompactEmbeddingTrainTensor{}, err
	}
	out.Moment1 = moment1
	out.Moment2 = moment2
	return out, nil
}

func optionalCompactMoment(checkpoint EmbeddingTrainCheckpoint, baseName, suffix string, shape []int) (*backend.Tensor, error) {
	name := baseName + suffix
	moment := checkpoint.MomentTensors[name]
	if moment == nil {
		return nil, nil
	}
	if !tensorShapeEquals(moment, shape) {
		return nil, fmt.Errorf("compact train state moment tensor %q shape %v does not match base tensor %q shape %v", name, moment.Shape, baseName, shape)
	}
	return moment.Clone(), nil
}

func validateCompactLayerSet(checkpoint EmbeddingTrainCheckpoint, repeats int) error {
	seen := map[int]bool{}
	for name := range checkpoint.Tensors {
		matches := compactLayerTensorNameRE.FindStringSubmatch(name)
		if matches == nil {
			continue
		}
		index, err := strconv.Atoi(matches[1])
		if err != nil {
			return fmt.Errorf("compact train state tensor %q has invalid layer index", name)
		}
		if index >= repeats {
			return fmt.Errorf("compact train state tensor %q is outside encoder_repeats=%d", name, repeats)
		}
		seen[index] = true
	}
	for i := 0; i < repeats; i++ {
		if !seen[i] {
			return fmt.Errorf("compact train state layer%d tensors are required", i)
		}
	}
	return nil
}

func validateCompactTrainManifestCompatibility(manifest, checkpointManifest EmbeddingManifest) error {
	if checkpointManifest.ArchitectureVersion != manifest.ArchitectureVersion {
		return fmt.Errorf("compact train state checkpoint architecture_version %q does not match manifest %q", checkpointManifest.ArchitectureVersion, manifest.ArchitectureVersion)
	}
	if checkpointManifest.EncoderRepeats != manifest.EncoderRepeats {
		return fmt.Errorf("compact train state checkpoint encoder_repeats %d does not match manifest %d", checkpointManifest.EncoderRepeats, manifest.EncoderRepeats)
	}
	if checkpointManifest.ModelDim != manifest.ModelDim {
		return fmt.Errorf("compact train state checkpoint model_dim %d does not match manifest %d", checkpointManifest.ModelDim, manifest.ModelDim)
	}
	if checkpointManifest.OutputDim != manifest.OutputDim {
		return fmt.Errorf("compact train state checkpoint output_dim %d does not match manifest %d", checkpointManifest.OutputDim, manifest.OutputDim)
	}
	if checkpointManifest.FFNDim != manifest.FFNDim {
		return fmt.Errorf("compact train state checkpoint ffn_dim %d does not match manifest %d", checkpointManifest.FFNDim, manifest.FFNDim)
	}
	if checkpointManifest.Tokenizer.VocabSize != manifest.Tokenizer.VocabSize {
		return fmt.Errorf("compact train state checkpoint tokenizer.vocab_size %d does not match manifest %d", checkpointManifest.Tokenizer.VocabSize, manifest.Tokenizer.VocabSize)
	}
	return nil
}

func compactTokenEmbeddingName(manifest EmbeddingManifest) string {
	if manifest.TokenEmbeddingParam != "" {
		return manifest.TokenEmbeddingParam
	}
	return "token_embedding"
}

func compactRoleEmbeddingName(manifest EmbeddingManifest) string {
	if manifest.RoleEmbeddingParam != "" {
		return manifest.RoleEmbeddingParam
	}
	return "role_embedding"
}

func compactOutputProjectionName(manifest EmbeddingManifest) string {
	if manifest.OutputProjectionParam != "" {
		return manifest.OutputProjectionParam
	}
	if manifest.OutputDim != manifest.ModelDim {
		return "output_projection"
	}
	return ""
}

func compactLayerTensorName(index int, suffix string) string {
	return fmt.Sprintf("layer%d_%s", index, suffix)
}

func tensorShapeEquals(tensor *backend.Tensor, shape []int) bool {
	if tensor == nil || len(tensor.Shape) != len(shape) {
		return false
	}
	for i := range shape {
		if tensor.Shape[i] != shape[i] {
			return false
		}
	}
	return true
}
