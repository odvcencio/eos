package models

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	eosruntime "m31labs.dev/eos/runtime"
)

const DefaultEmbeddingModelName = "eos-embed-v1"

// DefaultEmbeddingPackageConfig controls Eos's built-in default
// embedding-model package initialization.
type DefaultEmbeddingPackageConfig struct {
	Name                    string
	VocabSize               int
	MaxSequence             int
	Architecture            string
	ModelDim                int
	OutputDim               int
	EmbeddingDim            int
	HiddenDim               int
	AttentionHeads          int
	EncoderRepeats          int
	Seed                    int64
	LearningRate            float32
	WeightDecay             float32
	WeightBits              int
	WeightDType             string
	Optimizer               string
	ContrastiveLoss         string
	Temperature             float32
	GroupedLossWeight       float32
	TeacherLossWeight       float32
	TeacherTemperature      float32
	BootstrapFrom           string
	BootstrapFromInference  string
	BootstrapInferenceWiden bool
	BootstrapTailInit       string
	BootstrapTailScale      float64
}

// InitDefaultEmbeddingPackage compiles Eos's default trainable embedding
// shape and writes a native training package ready for train-corpus/train-embed.
func InitDefaultEmbeddingPackage(path string, cfg DefaultEmbeddingPackageConfig) (eosruntime.EmbeddingTrainPackagePaths, error) {
	if path == "" {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("output artifact path is required")
	}
	raw := cfg
	cfg = cfg.normalized()
	if cfg.BootstrapInferenceWiden && raw.OutputDim > 0 && raw.OutputDim != cfg.ModelDim {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("--bootstrap-from-inference-widen requires --output-dim to match --model-dim; got output_dim=%d model_dim=%d", raw.OutputDim, cfg.ModelDim)
	}
	if !cfg.BootstrapInferenceWiden && strings.TrimSpace(raw.BootstrapTailInit) != "" {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("--bootstrap-tail-init requires --bootstrap-from-inference-widen")
	}
	if !cfg.BootstrapInferenceWiden && raw.BootstrapTailScale != 0 {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("--bootstrap-tail-scale requires --bootstrap-from-inference-widen")
	}
	if err := cfg.validate(); err != nil {
		return eosruntime.EmbeddingTrainPackagePaths{}, err
	}
	if cfg.BootstrapFrom != "" && cfg.BootstrapFromInference != "" {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("--bootstrap-from and --bootstrap-from-inference are mutually exclusive")
	}
	if cfg.BootstrapInferenceWiden && cfg.BootstrapFromInference == "" {
		return eosruntime.EmbeddingTrainPackagePaths{}, fmt.Errorf("--bootstrap-from-inference-widen requires --bootstrap-from-inference")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return eosruntime.EmbeddingTrainPackagePaths{}, err
	}
	if cfg.BootstrapFromInference != "" {
		initOpts := eosruntime.EmbeddingTrainInitOptions{
			Seed:                    cfg.Seed,
			BootstrapInferenceWiden: cfg.BootstrapInferenceWiden,
			BootstrapTailInit:       cfg.BootstrapTailInit,
			BootstrapTailScale:      cfg.BootstrapTailScale,
		}
		if cfg.BootstrapInferenceWiden {
			if err := validateBootstrapFromInferenceWidenOverrides(raw, cfg); err != nil {
				return eosruntime.EmbeddingTrainPackagePaths{}, err
			}
			initOpts.ShapeSizes = map[string]int{
				"D": cfg.ModelDim,
				"H": cfg.HiddenDim,
				"O": cfg.OutputDim,
				"E": cfg.OutputDim,
			}
		} else {
			if err := validateBootstrapFromInferenceGraphOverrides(raw); err != nil {
				return eosruntime.EmbeddingTrainPackagePaths{}, err
			}
		}
		return eosruntime.InitializeEmbeddingTrainerPackageFromSealedInference(path, cfg.BootstrapFromInference, cfg.trainConfig(), initOpts)
	}

	moduleName := moduleNameForModel(cfg.Name)
	var src []byte
	preset := cfg.preset()
	if cfg.Architecture == eosruntime.EmbeddingArchitectureCompactTransformerV1 {
		src = []byte(compactTransformerSource(cfg))
		preset = ""
	}
	bundle, err := compiler.Build(src, compiler.Options{
		ModuleName: moduleName,
		Preset:     preset,
	})
	if err != nil {
		return eosruntime.EmbeddingTrainPackagePaths{}, err
	}
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		return eosruntime.EmbeddingTrainPackagePaths{}, err
	}

	manifest := DefaultEmbeddingManifest(cfg)
	if err := manifest.WriteFile(eosruntime.DefaultEmbeddingManifestPath(path)); err != nil {
		return eosruntime.EmbeddingTrainPackagePaths{}, err
	}
	return eosruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg.trainConfig(), eosruntime.EmbeddingTrainInitOptions{
		Seed:                   cfg.Seed,
		BootstrapArtifactPath:  cfg.BootstrapFrom,
		BootstrapInferencePath: cfg.BootstrapFromInference,
		ShapeSizes: map[string]int{
			"D": cfg.ModelDim,
			"H": cfg.HiddenDim,
			"O": cfg.OutputDim,
		},
	})
}

func validateBootstrapFromInferenceGraphOverrides(cfg DefaultEmbeddingPackageConfig) error {
	if cfg.Name != "" ||
		cfg.VocabSize != 0 ||
		cfg.MaxSequence != 0 ||
		cfg.Architecture != "" ||
		cfg.ModelDim != 0 ||
		cfg.OutputDim != 0 ||
		cfg.EmbeddingDim != 0 ||
		cfg.HiddenDim != 0 ||
		cfg.WeightDType != "" ||
		cfg.AttentionHeads != 0 ||
		cfg.EncoderRepeats != 0 {
		return fmt.Errorf("--bootstrap-from-inference preserves the sealed source graph; omit graph-shaping flags such as --name, --vocab-size, --max-seq, --architecture, --model-dim, --embedding-dim, --output-dim, --hidden-dim, --weight-dtype, --attention-heads, and --encoder-repeats")
	}
	return nil
}

func validateBootstrapFromInferenceWidenOverrides(raw, cfg DefaultEmbeddingPackageConfig) error {
	if raw.BootstrapFromInference == "" {
		return fmt.Errorf("--bootstrap-from-inference-widen requires --bootstrap-from-inference")
	}
	if raw.ModelDim <= 0 || raw.HiddenDim <= 0 {
		return fmt.Errorf("--bootstrap-from-inference-widen requires --model-dim and --hidden-dim")
	}
	if raw.OutputDim > 0 && raw.OutputDim != cfg.ModelDim {
		return fmt.Errorf("--bootstrap-from-inference-widen requires --output-dim to match --model-dim; got output_dim=%d model_dim=%d", raw.OutputDim, cfg.ModelDim)
	}
	if raw.EmbeddingDim > 0 && raw.EmbeddingDim != cfg.ModelDim {
		return fmt.Errorf("--bootstrap-from-inference-widen requires --embedding-dim to match --model-dim; got embedding_dim=%d model_dim=%d", raw.EmbeddingDim, cfg.ModelDim)
	}
	if raw.Name != "" ||
		raw.VocabSize != 0 ||
		raw.MaxSequence != 0 ||
		raw.Architecture != "" ||
		raw.WeightDType != "" ||
		raw.AttentionHeads != 0 ||
		raw.EncoderRepeats != 0 {
		return fmt.Errorf("--bootstrap-from-inference-widen preserves the sealed source graph and tokenizer; only --model-dim, --embedding-dim, --output-dim, --hidden-dim, seed/training flags, --bootstrap-tail-init, and --bootstrap-tail-scale may be supplied")
	}
	tailInit := eosruntimeTailInit(raw.BootstrapTailInit)
	switch tailInit {
	case "zero", "random":
		if raw.BootstrapTailScale != 0 {
			return fmt.Errorf("--bootstrap-tail-scale requires --bootstrap-tail-init projection-tail")
		}
	case "projection-tail":
		if raw.BootstrapTailScale != 0 && (math.IsNaN(raw.BootstrapTailScale) || math.IsInf(raw.BootstrapTailScale, 0) || raw.BootstrapTailScale <= 0 || raw.BootstrapTailScale > 0.10) {
			return fmt.Errorf("--bootstrap-tail-scale must be finite and in (0, 0.10] for projection-tail, got %g", raw.BootstrapTailScale)
		}
	default:
		return fmt.Errorf("--bootstrap-tail-init must be zero, random, or projection-tail, got %q", raw.BootstrapTailInit)
	}
	return nil
}

func eosruntimeTailInit(mode string) string {
	mode = strings.TrimSpace(strings.ToLower(mode))
	if mode == "" {
		return "zero"
	}
	return mode
}

// DefaultEmbeddingManifest returns the serving/training contract for the
// built-in default embedding model shape.
func DefaultEmbeddingManifest(cfg DefaultEmbeddingPackageConfig) eosruntime.EmbeddingManifest {
	cfg = cfg.normalized()
	manifest := eosruntime.EmbeddingManifest{
		Name:                  cfg.Name,
		PooledEntry:           "embed_pooled",
		BatchEntry:            "embed_pooled_batch",
		EncoderRepeats:        cfg.EncoderRepeats,
		TokenInput:            "tokens",
		MaskInput:             "attention_mask",
		OutputName:            "result",
		OutputDType:           "f16",
		ArchitectureVersion:   cfg.Architecture,
		ModelDim:              cfg.ModelDim,
		OutputDim:             cfg.OutputDim,
		AttentionHeads:        cfg.AttentionHeads,
		HeadDim:               cfg.ModelDim / cfg.AttentionHeads,
		FFNDim:                cfg.HiddenDim,
		ParameterTying:        eosruntime.EmbeddingParameterTyingLegacyTied,
		TokenEmbeddingParam:   "token_embedding",
		RoleConditioning:      eosruntime.EmbeddingRoleConditioningAdditiveV1,
		RoleEmbeddingParam:    "role_embedding",
		RoleInput:             "role_ids",
		BatchRoleInput:        "role_ids",
		RawRoleIndex:          0,
		QueryRoleIndex:        1,
		DocumentRoleIndex:     2,
		AttentionQueryParam:   "attn_q",
		AttentionKeyParam:     "attn_k",
		AttentionValueParam:   "attn_v",
		AttentionOutputParam:  "attn_o",
		AttentionMaskMode:     eosruntime.EmbeddingAttentionMaskModeKey,
		AttentionScoreScale:   eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ,
		AttentionResidual:     true,
		AttentionLayerNorm:    true,
		PositionEncoding:      eosruntime.EmbeddingPositionEncodingRoPE,
		HiddenProjectionParam: "ffn_up",
		FFNResidual:           true,
		FFNLayerNorm:          true,
		ProjectionParam:       "projection",
		Tokenizer: eosruntime.TokenizerManifest{
			VocabSize:   cfg.VocabSize,
			MaxSequence: cfg.MaxSequence,
			PadID:       0,
			BOSID:       1,
			EOSID:       2,
			UnknownID:   3,
		},
	}
	if cfg.Architecture == eosruntime.EmbeddingArchitectureCompactTransformerV1 {
		manifest.ParameterTying = eosruntime.EmbeddingParameterTyingUntied
		manifest.AttentionQueryParam = "layer0_attn_q"
		manifest.AttentionKeyParam = "layer0_attn_k"
		manifest.AttentionValueParam = "layer0_attn_v"
		manifest.AttentionOutputParam = "layer0_attn_o"
		manifest.HiddenProjectionParam = "layer0_ffn_up"
		manifest.ProjectionParam = "layer0_ffn_down"
		if cfg.OutputDim != cfg.ModelDim {
			manifest.OutputProjectionParam = "output_projection"
		}
	}
	return manifest
}

func (cfg DefaultEmbeddingPackageConfig) normalized() DefaultEmbeddingPackageConfig {
	if cfg.Name == "" {
		cfg.Name = DefaultEmbeddingModelName
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 32768
	}
	if cfg.MaxSequence == 0 {
		cfg.MaxSequence = 256
	}
	if cfg.Architecture == "" {
		cfg.Architecture = eosruntime.EmbeddingArchitectureLegacyV1
	}
	if cfg.ModelDim == 0 {
		cfg.ModelDim = cfg.EmbeddingDim
	}
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = cfg.ModelDim
	}
	if cfg.ModelDim == 0 {
		cfg.ModelDim = 128
		cfg.EmbeddingDim = 128
	}
	if cfg.OutputDim == 0 {
		cfg.OutputDim = cfg.ModelDim
	}
	if cfg.HiddenDim == 0 {
		cfg.HiddenDim = cfg.ModelDim * 2
	}
	if cfg.AttentionHeads == 0 {
		cfg.AttentionHeads = 1
	}
	if cfg.EncoderRepeats <= 0 {
		cfg.EncoderRepeats = 2
	}
	if cfg.Seed == 0 {
		cfg.Seed = 1
	}
	if cfg.LearningRate == 0 {
		cfg.LearningRate = 0.005
	}
	if cfg.WeightDType == "" {
		cfg.WeightDType = "q8"
	}
	if cfg.WeightBits == 0 {
		// Default the fake-quant training grid to the declared param dtype so
		// the sealed packed payload is a fixed point of the QAT forward.
		if cfg.WeightDType == "q4" {
			cfg.WeightBits = 4
		} else {
			cfg.WeightBits = 8
		}
	}
	if cfg.Optimizer == "" {
		cfg.Optimizer = "adamw"
	}
	if cfg.ContrastiveLoss == "" {
		cfg.ContrastiveLoss = "infonce"
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.05
	}
	if cfg.TeacherTemperature == 0 {
		cfg.TeacherTemperature = 1
	}
	return cfg
}

func (cfg DefaultEmbeddingPackageConfig) validate() error {
	if cfg.VocabSize <= 0 {
		return fmt.Errorf("vocab size must be positive")
	}
	if cfg.MaxSequence <= 0 {
		return fmt.Errorf("max sequence must be positive")
	}
	switch cfg.Architecture {
	case eosruntime.EmbeddingArchitectureLegacyV1:
	case eosruntime.EmbeddingArchitectureCompactTransformerV1:
		if err := validateDefaultEmbeddingArchitectureDims(cfg); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported architecture %q", cfg.Architecture)
	}
	if cfg.ModelDim <= 0 {
		return fmt.Errorf("model dim must be positive")
	}
	if cfg.OutputDim <= 0 {
		return fmt.Errorf("output dim must be positive")
	}
	if cfg.EmbeddingDim > 0 && cfg.EmbeddingDim != cfg.ModelDim {
		return fmt.Errorf("embedding dim %d conflicts with model dim %d", cfg.EmbeddingDim, cfg.ModelDim)
	}
	if cfg.AttentionHeads <= 0 {
		return fmt.Errorf("attention heads must be positive")
	}
	if cfg.ModelDim%cfg.AttentionHeads != 0 {
		return fmt.Errorf("model_dim %d must be divisible by attention_heads %d", cfg.ModelDim, cfg.AttentionHeads)
	}
	if cfg.Architecture == eosruntime.EmbeddingArchitectureLegacyV1 && cfg.AttentionHeads != 1 {
		return fmt.Errorf("%s with attention_heads=%d is not supported by trainable package initialization yet", cfg.Architecture, cfg.AttentionHeads)
	}
	if cfg.Architecture == eosruntime.EmbeddingArchitectureLegacyV1 && cfg.OutputDim != cfg.ModelDim {
		return fmt.Errorf("%s with output_dim=%d is not supported by trainable package initialization yet", cfg.Architecture, cfg.OutputDim)
	}
	if cfg.HiddenDim <= 0 {
		return fmt.Errorf("hidden dim must be positive")
	}
	if cfg.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive")
	}
	if cfg.WeightBits <= 0 {
		return fmt.Errorf("weight bits must be positive")
	}
	if cfg.WeightDType != "q8" && cfg.WeightDType != "q4" {
		return fmt.Errorf("weight dtype must be q8 or q4, got %q", cfg.WeightDType)
	}
	if cfg.Temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	if cfg.GroupedLossWeight < 0 {
		return fmt.Errorf("grouped loss weight must be non-negative")
	}
	if cfg.TeacherLossWeight < 0 {
		return fmt.Errorf("teacher loss weight must be non-negative")
	}
	if cfg.TeacherTemperature <= 0 {
		return fmt.Errorf("teacher temperature must be positive")
	}
	return nil
}

func validateDefaultEmbeddingArchitectureDims(cfg DefaultEmbeddingPackageConfig) error {
	if cfg.ModelDim <= 0 {
		return fmt.Errorf("model dim must be positive")
	}
	if cfg.OutputDim <= 0 {
		return fmt.Errorf("output dim must be positive")
	}
	if cfg.AttentionHeads <= 0 {
		return fmt.Errorf("attention heads must be positive")
	}
	if cfg.ModelDim%cfg.AttentionHeads != 0 {
		return fmt.Errorf("model_dim %d must be divisible by attention_heads %d", cfg.ModelDim, cfg.AttentionHeads)
	}
	return nil
}

func (cfg DefaultEmbeddingPackageConfig) preset() compiler.Preset {
	if cfg.WeightDType == "q4" {
		return compiler.PresetEncoderTrainableQ4x2
	}
	return compiler.PresetEncoderTrainableQ8x2
}

func compactTransformerSource(cfg DefaultEmbeddingPackageConfig) string {
	var b strings.Builder
	dtype := cfg.WeightDType
	if dtype == "" {
		dtype = "q8"
	}
	fmt.Fprintf(&b, "param token_embedding: %s[V, D] @weight(\"weights/token_embedding\") @trainable\n", dtype)
	fmt.Fprintf(&b, "param role_embedding: %s[3, D] @weight(\"weights/role_embedding\") @trainable\n", dtype)
	for i := 0; i < cfg.EncoderRepeats; i++ {
		prefix := fmt.Sprintf("layer%d", i)
		fmt.Fprintf(&b, "param %s_attn_q: %s[D, D] @weight(\"weights/%s_attn_q\") @trainable\n", prefix, dtype, prefix)
		fmt.Fprintf(&b, "param %s_attn_k: %s[D, D] @weight(\"weights/%s_attn_k\") @trainable\n", prefix, dtype, prefix)
		fmt.Fprintf(&b, "param %s_attn_v: %s[D, D] @weight(\"weights/%s_attn_v\") @trainable\n", prefix, dtype, prefix)
		fmt.Fprintf(&b, "param %s_attn_o: %s[D, D] @weight(\"weights/%s_attn_o\") @trainable\n", prefix, dtype, prefix)
		fmt.Fprintf(&b, "param %s_ffn_up: %s[D, H] @weight(\"weights/%s_ffn_up\") @trainable\n", prefix, dtype, prefix)
		fmt.Fprintf(&b, "param %s_ffn_down: %s[H, D] @weight(\"weights/%s_ffn_down\") @trainable\n", prefix, dtype, prefix)
	}
	if cfg.OutputDim != cfg.ModelDim {
		fmt.Fprintf(&b, "param output_projection: %s[D, O] @weight(\"weights/output_projection\") @trainable\n", dtype)
	}
	b.WriteString("\n")
	b.WriteString("pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T], role_ids: i32[T]) -> f16[")
	if cfg.OutputDim != cfg.ModelDim {
		b.WriteString("O")
	} else {
		b.WriteString("D")
	}
	b.WriteString("] {\n")
	b.WriteString(compactTransformerPipelineBody(cfg))
	b.WriteString("}\n\n")
	b.WriteString("pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T], role_ids: i32[B, T]) -> f16[")
	if cfg.OutputDim != cfg.ModelDim {
		b.WriteString("B, O")
	} else {
		b.WriteString("B, D")
	}
	b.WriteString("] {\n")
	b.WriteString(compactTransformerPipelineBody(cfg))
	b.WriteString("}\n")
	return b.String()
}

func compactTransformerPipelineBody(cfg DefaultEmbeddingPackageConfig) string {
	var b strings.Builder
	b.WriteString("    let hidden_q = gather(token_embedding, tokens)\n")
	b.WriteString("    let role_hidden_q = gather(role_embedding, role_ids)\n")
	b.WriteString("    let hidden_f = dequant(hidden_q)\n")
	b.WriteString("    let role_hidden_f = dequant(role_hidden_q)\n")
	b.WriteString("    let hidden = rope(hidden_f + role_hidden_f)\n")
	prev := "hidden"
	for i := 0; i < cfg.EncoderRepeats; i++ {
		prefix := fmt.Sprintf("layer%d", i)
		fmt.Fprintf(&b, "    let %s_wq = dequant(%s_attn_q)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wk = dequant(%s_attn_k)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wv = dequant(%s_attn_v)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_wo = dequant(%s_attn_o)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_ffn_up_f = dequant(%s_ffn_up)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_ffn_down_f = dequant(%s_ffn_down)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_q = @matmul(%s, %s_wq)\n", prefix, prev, prefix)
		fmt.Fprintf(&b, "    let %s_k = @matmul(%s, %s_wk)\n", prefix, prev, prefix)
		fmt.Fprintf(&b, "    let %s_v = @matmul(%s, %s_wv)\n", prefix, prev, prefix)
		if cfg.AttentionHeads > 1 {
			fmt.Fprintf(&b, "    let %s_mixed = compact_multihead_attention_h%d(%s_q, %s_k, %s_v, attention_mask)\n", prefix, cfg.AttentionHeads, prefix, prefix, prefix)
		} else {
			fmt.Fprintf(&b, "    let %s_kt = transpose(%s_k)\n", prefix, prefix)
			fmt.Fprintf(&b, "    let %s_scores = @scaled_attention_scores(%s_q, %s_kt)\n", prefix, prefix, prefix)
			fmt.Fprintf(&b, "    let %s_probs = masked_softmax(%s_scores, attention_mask)\n", prefix, prefix)
			fmt.Fprintf(&b, "    let %s_mixed = @matmul(%s_probs, %s_v)\n", prefix, prefix, prefix)
		}
		fmt.Fprintf(&b, "    let %s_attended = @matmul(%s_mixed, %s_wo)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_attn_hidden = layernorm(%s_attended + %s)\n", prefix, prefix, prev)
		fmt.Fprintf(&b, "    let %s_ffn_hidden = @matmul(%s_attn_hidden, %s_ffn_up_f)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_activated = gelu(%s_ffn_hidden)\n", prefix, prefix)
		fmt.Fprintf(&b, "    let %s_projected = @matmul(%s_activated, %s_ffn_down_f)\n", prefix, prefix, prefix)
		fmt.Fprintf(&b, "    let %s_encoded = layernorm(%s_projected + %s_attn_hidden)\n", prefix, prefix, prefix)
		prev = prefix + "_encoded"
	}
	fmt.Fprintf(&b, "    let normalized = normalize(%s)\n", prev)
	if cfg.OutputDim != cfg.ModelDim {
		b.WriteString("    let output_projection_f = dequant(output_projection)\n")
		b.WriteString("    let output_projected = @matmul(normalized, output_projection_f)\n")
		b.WriteString("    return mean_pool(output_projected, attention_mask)\n")
		return b.String()
	}
	b.WriteString("    return mean_pool(normalized, attention_mask)\n")
	return b.String()
}

func (cfg DefaultEmbeddingPackageConfig) trainConfig() eosruntime.EmbeddingTrainConfig {
	return eosruntime.EmbeddingTrainConfig{
		LearningRate:       cfg.LearningRate,
		WeightDecay:        cfg.WeightDecay,
		WeightBits:         cfg.WeightBits,
		Optimizer:          cfg.Optimizer,
		ContrastiveLoss:    cfg.ContrastiveLoss,
		Temperature:        cfg.Temperature,
		GroupedLossWeight:  cfg.GroupedLossWeight,
		TeacherLossWeight:  cfg.TeacherLossWeight,
		TeacherTemperature: cfg.TeacherTemperature,
	}
}

func moduleNameForModel(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	var b strings.Builder
	lastUnderscore := false
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "manta_model"
	}
	return out
}
