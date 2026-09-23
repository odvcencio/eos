package models

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	eosartifact "m31labs.dev/eos/artifact/eos"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backends/vulkan"
	mll "m31labs.dev/mll"
)

func TestInitDefaultEmbeddingPackageCreatesTrainablePackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	for _, candidate := range []string{
		paths.ArtifactPath,
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
		paths.PackageManifestPath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.Name != DefaultEmbeddingModelName {
		t.Fatalf("manifest name = %q, want %q", manifest.Name, DefaultEmbeddingModelName)
	}
	if manifest.EncoderRepeats != 2 {
		t.Fatalf("encoder repeats = %d, want 2", manifest.EncoderRepeats)
	}
	if manifest.AttentionMaskMode != eosruntime.EmbeddingAttentionMaskModeKey {
		t.Fatalf("attention mask mode = %q, want %q", manifest.AttentionMaskMode, eosruntime.EmbeddingAttentionMaskModeKey)
	}
	if manifest.AttentionScoreScale != eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ {
		t.Fatalf("attention score scale = %q, want %q", manifest.AttentionScoreScale, eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ)
	}
	if manifest.PositionEncoding != eosruntime.EmbeddingPositionEncodingRoPE {
		t.Fatalf("position encoding = %q, want %q", manifest.PositionEncoding, eosruntime.EmbeddingPositionEncodingRoPE)
	}
	if manifest.ArchitectureVersion != eosruntime.EmbeddingArchitectureLegacyV1 ||
		manifest.ModelDim != 4 ||
		manifest.OutputDim != 4 ||
		manifest.AttentionHeads != 1 ||
		manifest.HeadDim != 4 ||
		manifest.FFNDim != 8 ||
		manifest.ParameterTying != eosruntime.EmbeddingParameterTyingLegacyTied {
		t.Fatalf("unexpected architecture metadata: %+v", manifest)
	}
	if manifest.Tokenizer.VocabSize != 16 || manifest.Tokenizer.MaxSequence != 8 {
		t.Fatalf("unexpected tokenizer contract: %+v", manifest.Tokenizer)
	}
	if manifest.Tokenizer.PadID != 0 || manifest.Tokenizer.BOSID != 1 || manifest.Tokenizer.EOSID != 2 || manifest.Tokenizer.UnknownID != 3 {
		t.Fatalf("unexpected tokenizer ids: %+v", manifest.Tokenizer)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.ContrastiveLoss != "infonce" {
		t.Fatalf("contrastive loss = %q, want infonce", checkpoint.Config.ContrastiveLoss)
	}
	if checkpoint.Config.Temperature != 0.05 {
		t.Fatalf("temperature = %f, want 0.05", checkpoint.Config.Temperature)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("load training package: %v", err)
	}
	if _, err := eosruntime.LoadCompactEmbeddingTrainStateFromPackage(path); err == nil || !strings.Contains(err.Error(), `compact train state requires architecture_version="compact_transformer_v1"`) {
		t.Fatalf("LoadCompactEmbeddingTrainStateFromPackage error = %v, want legacy package rejection", err)
	}
}

func TestInitDefaultEmbeddingPackageRejectsMixedBootstrapSources(t *testing.T) {
	path := filepath.Join(t.TempDir(), "target.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		BootstrapFrom:          filepath.Join(t.TempDir(), "trainable.mll"),
		BootstrapFromInference: filepath.Join(t.TempDir(), "sealed.mll"),
	})
	if err == nil {
		t.Fatal("expected mixed bootstrap source rejection")
	}
	if got := err.Error(); !strings.Contains(got, "--bootstrap-from and --bootstrap-from-inference are mutually exclusive") {
		t.Fatalf("error = %q, want mutual exclusion rejection", got)
	}
}

func TestInitDefaultEmbeddingPackageRejectsTailInitWithoutInferenceWiden(t *testing.T) {
	path := filepath.Join(t.TempDir(), "target.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		BootstrapTailInit: "zero",
	})
	if err == nil {
		t.Fatal("expected tail-init without widen rejection")
	}
	if got := err.Error(); !strings.Contains(got, "--bootstrap-tail-init requires --bootstrap-from-inference-widen") {
		t.Fatalf("error = %q, want tail-init/widen rejection", got)
	}
}

func TestInitDefaultEmbeddingPackageRejectsTailScaleWithoutInferenceWiden(t *testing.T) {
	path := filepath.Join(t.TempDir(), "target.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		BootstrapTailScale: 0.03,
	})
	if err == nil {
		t.Fatal("expected tail-scale without widen rejection")
	}
	if got := err.Error(); !strings.Contains(got, "--bootstrap-tail-scale requires --bootstrap-from-inference-widen") {
		t.Fatalf("error = %q, want tail-scale/widen rejection", got)
	}
}

func TestInitDefaultEmbeddingPackageRejectsUnknownInferenceWidenTailInit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "target.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		BootstrapFromInference:  "missing.sealed.mll",
		BootstrapInferenceWiden: true,
		ModelDim:                3,
		HiddenDim:               5,
		BootstrapTailInit:       "gaussian",
	})
	if err == nil {
		t.Fatal("expected unknown tail-init rejection")
	}
	if got := err.Error(); !strings.Contains(got, "--bootstrap-tail-init must be zero, random, or projection-tail") {
		t.Fatalf("error = %q, want tail-init value rejection", got)
	}
}

func TestInitDefaultEmbeddingPackageValidatesProjectionTailScale(t *testing.T) {
	for _, tc := range []struct {
		name     string
		tailInit string
		scale    float64
		want     string
	}{
		{"zero scale rejected", "zero", 0.03, "--bootstrap-tail-scale requires --bootstrap-tail-init projection-tail"},
		{"random scale rejected", "random", 0.03, "--bootstrap-tail-scale requires --bootstrap-tail-init projection-tail"},
		{"too large projection scale", "projection-tail", 0.11, "--bootstrap-tail-scale must be finite and in (0, 0.10]"},
		{"negative projection scale", "projection-tail", -0.01, "--bootstrap-tail-scale must be finite and in (0, 0.10]"},
		{"nan projection scale", "projection-tail", math.NaN(), "--bootstrap-tail-scale must be finite and in (0, 0.10]"},
		{"inf projection scale", "projection-tail", math.Inf(1), "--bootstrap-tail-scale must be finite and in (0, 0.10]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := InitDefaultEmbeddingPackage(filepath.Join(t.TempDir(), "target.mll"), DefaultEmbeddingPackageConfig{
				BootstrapFromInference:  "missing.sealed.mll",
				BootstrapInferenceWiden: true,
				ModelDim:                3,
				HiddenDim:               5,
				BootstrapTailInit:       tc.tailInit,
				BootstrapTailScale:      tc.scale,
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestInitDefaultEmbeddingPackageDefaultsOutputDimToModelDim(t *testing.T) {
	manifest := DefaultEmbeddingManifest(DefaultEmbeddingPackageConfig{
		ModelDim:  6,
		HiddenDim: 12,
	})
	if manifest.ModelDim != 6 || manifest.OutputDim != 6 {
		t.Fatalf("manifest dims = model:%d output:%d, want 6/6", manifest.ModelDim, manifest.OutputDim)
	}
}

func TestInitDefaultEmbeddingPackageCreatesCompactBootstrapPackage(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		VocabSize:      16,
		MaxSequence:    8,
		ModelDim:       8,
		OutputDim:      4,
		HiddenDim:      16,
		AttentionHeads: 1,
		EncoderRepeats: 3,
	})
	if err != nil {
		t.Fatalf("init compact default embedding package: %v", err)
	}
	for _, candidate := range []string{
		paths.ArtifactPath,
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
		paths.PackageManifestPath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
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
		manifest.EncoderRepeats != 3 ||
		manifest.OutputProjectionParam != "output_projection" {
		t.Fatalf("unexpected compact metadata: %+v", manifest)
	}
	if manifest.RoleConditioning != eosruntime.EmbeddingRoleConditioningAdditiveV1 ||
		manifest.PositionEncoding != eosruntime.EmbeddingPositionEncodingRoPE ||
		manifest.AttentionMaskMode != eosruntime.EmbeddingAttentionMaskModeKey ||
		manifest.AttentionScoreScale != eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ {
		t.Fatalf("unexpected compact conditioning/attention metadata: %+v", manifest)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	for _, name := range []string{"token_embedding", "role_embedding", "layer0_attn_q", "layer2_ffn_down", "output_projection"} {
		if checkpoint.Tensors[name] == nil {
			t.Fatalf("checkpoint missing generic tensor %q; tensors=%v", name, checkpoint.Tensors)
		}
		if checkpoint.MomentTensors[name+"_moment_1"] == nil || checkpoint.MomentTensors[name+"_moment_2"] == nil {
			t.Fatalf("checkpoint missing generic moments for %q", name)
		}
	}
	state, err := eosruntime.LoadCompactEmbeddingTrainStateFromPackage(path)
	if err != nil {
		t.Fatalf("load compact train state: %v", err)
	}
	if len(state.Layers) != 3 {
		t.Fatalf("compact state layer count = %d, want 3", len(state.Layers))
	}
	if got := state.TokenEmbedding.Tensor.Shape; len(got) != 2 || got[0] != 16 || got[1] != 8 {
		t.Fatalf("compact state token shape = %v, want [16 8]", got)
	}
	if got := state.Layers[2].FFNDown.Tensor.Shape; len(got) != 2 || got[0] != 16 || got[1] != 8 {
		t.Fatalf("compact state layer2 ffn_down shape = %v, want [16 8]", got)
	}
	if state.OutputProjection == nil {
		t.Fatal("compact state missing output projection")
	}
	if got := state.OutputProjection.Tensor.Shape; len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("compact state output projection shape = %v, want [8 4]", got)
	}
	trainer, err := eosruntime.LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load compact trainer package: %v", err)
	}
	evalSet := []eosruntime.EmbeddingPairExample{
		{LeftTokens: []int32{1, 4, 5}, RightTokens: []int32{1, 4, 5}, Target: 1},
		{LeftTokens: []int32{1, 4, 5}, RightTokens: []int32{5, 4, 1}, Target: -1},
	}
	beforeStep := trainer.TrainProfile().Step
	metrics, err := trainer.EvaluatePairs(evalSet)
	if err != nil {
		t.Fatalf("evaluate compact trainer package: %v", err)
	}
	if math.IsNaN(float64(metrics.Loss)) || math.IsInf(float64(metrics.Loss), 0) ||
		math.IsNaN(float64(metrics.AverageScore)) || math.IsInf(float64(metrics.AverageScore), 0) ||
		metrics.PairCount != len(evalSet) {
		t.Fatalf("compact eval metrics = %+v, want finite pair metrics", metrics)
	}
	summary, err := trainer.Fit(nil, evalSet, eosruntime.EmbeddingTrainRunConfig{EvalOnly: true})
	if err != nil {
		t.Fatalf("compact eval-only fit: %v", err)
	}
	if summary.StepsRun != 0 || trainer.TrainProfile().Step != beforeStep {
		t.Fatalf("compact eval-only steps run=%d trainer step=%d want unchanged %d", summary.StepsRun, trainer.TrainProfile().Step, beforeStep)
	}
	trainMetrics, err := trainer.TrainStep(evalSet)
	if err != nil {
		t.Fatalf("compact TrainStep: %v", err)
	}
	if math.IsNaN(float64(trainMetrics.Loss)) || math.IsInf(float64(trainMetrics.Loss), 0) ||
		math.IsNaN(float64(trainMetrics.AverageScore)) || math.IsInf(float64(trainMetrics.AverageScore), 0) ||
		trainMetrics.BatchSize != len(evalSet) {
		t.Fatalf("compact train metrics = %+v, want finite pair metrics", trainMetrics)
	}
	if got := trainer.TrainProfile().Step; got != beforeStep+1 {
		t.Fatalf("compact train step = %d, want %d", got, beforeStep+1)
	}
	metrics, err = trainer.EvaluatePairs(evalSet)
	if err != nil {
		t.Fatalf("evaluate compact trainer package after train: %v", err)
	}
	outPath := filepath.Join(t.TempDir(), "compact-roundtrip.mll")
	outPaths, err := trainer.WriteTrainingPackage(outPath)
	if err != nil {
		t.Fatalf("write compact training package: %v", err)
	}
	reloadedManifest, err := eosruntime.ReadEmbeddingManifestFile(outPaths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read round-trip compact manifest: %v", err)
	}
	if reloadedManifest.ArchitectureVersion != eosruntime.EmbeddingArchitectureCompactTransformerV1 ||
		reloadedManifest.EncoderRepeats != 3 ||
		reloadedManifest.OutputProjectionParam != "output_projection" {
		t.Fatalf("round-trip compact manifest not preserved: %+v", reloadedManifest)
	}
	reloadedCheckpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(outPaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read round-trip compact checkpoint: %v", err)
	}
	if reloadedCheckpoint.TokenEmbedding != nil || reloadedCheckpoint.Projection != nil {
		t.Fatalf("round-trip compact checkpoint populated legacy fields: token=%v projection=%v", reloadedCheckpoint.TokenEmbedding, reloadedCheckpoint.Projection)
	}
	if reloadedCheckpoint.Step != beforeStep+1 {
		t.Fatalf("round-trip compact checkpoint step = %d, want %d", reloadedCheckpoint.Step, beforeStep+1)
	}
	for _, name := range []string{"token_embedding", "role_embedding", "layer0_attn_q", "layer2_ffn_down", "output_projection"} {
		if reloadedCheckpoint.Tensors[name] == nil {
			t.Fatalf("round-trip checkpoint missing compact tensor %q", name)
		}
		if reloadedCheckpoint.MomentTensors[name+"_moment_1"] == nil || reloadedCheckpoint.MomentTensors[name+"_moment_2"] == nil {
			t.Fatalf("round-trip checkpoint missing compact moments for %q", name)
		}
	}
	reloadedState, err := eosruntime.LoadCompactEmbeddingTrainStateFromPackage(outPath)
	if err != nil {
		t.Fatalf("load round-trip compact train state: %v", err)
	}
	if len(reloadedState.Layers) != 3 || reloadedState.OutputProjection == nil {
		t.Fatalf("round-trip compact state = %+v, want 3 layers and output projection", reloadedState)
	}
	reloadedTrainer, err := eosruntime.LoadEmbeddingTrainerPackage(outPath)
	if err != nil {
		t.Fatalf("load round-trip compact trainer package: %v", err)
	}
	reloadedMetrics, err := reloadedTrainer.EvaluatePairs(evalSet)
	if err != nil {
		t.Fatalf("evaluate round-trip compact trainer package: %v", err)
	}
	if math.Abs(float64(metrics.Loss-reloadedMetrics.Loss)) > 0.000001 ||
		math.Abs(float64(metrics.AverageScore-reloadedMetrics.AverageScore)) > 0.000001 ||
		reloadedMetrics.PairCount != metrics.PairCount {
		t.Fatalf("round-trip compact metrics = %+v, want %+v", reloadedMetrics, metrics)
	}
	reloadedStep := reloadedTrainer.TrainProfile().Step
	if _, err := reloadedTrainer.TrainStep(evalSet); err != nil {
		t.Fatalf("round-trip compact TrainStep: %v", err)
	}
	if got := reloadedTrainer.TrainProfile().Step; got != reloadedStep+1 {
		t.Fatalf("round-trip compact train step = %d, want %d", got, reloadedStep+1)
	}
}

func TestInitDefaultEmbeddingPackageCompactParamContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-contract.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		VocabSize:      19,
		MaxSequence:    7,
		ModelDim:       8,
		OutputDim:      5,
		HiddenDim:      24,
		AttentionHeads: 1,
		EncoderRepeats: 2,
		WeightDType:    "q4",
	})
	if err != nil {
		t.Fatalf("init compact default embedding package: %v", err)
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read compact artifact: %v", err)
	}
	want := []struct {
		name  string
		shape []string
	}{
		{name: "token_embedding", shape: []string{"V", "D"}},
		{name: "role_embedding", shape: []string{"3", "D"}},
		{name: "layer0_attn_q", shape: []string{"D", "D"}},
		{name: "layer0_attn_k", shape: []string{"D", "D"}},
		{name: "layer0_attn_v", shape: []string{"D", "D"}},
		{name: "layer0_attn_o", shape: []string{"D", "D"}},
		{name: "layer0_ffn_up", shape: []string{"D", "H"}},
		{name: "layer0_ffn_down", shape: []string{"H", "D"}},
		{name: "layer1_attn_q", shape: []string{"D", "D"}},
		{name: "layer1_attn_k", shape: []string{"D", "D"}},
		{name: "layer1_attn_v", shape: []string{"D", "D"}},
		{name: "layer1_attn_o", shape: []string{"D", "D"}},
		{name: "layer1_ffn_up", shape: []string{"D", "H"}},
		{name: "layer1_ffn_down", shape: []string{"H", "D"}},
		{name: "output_projection", shape: []string{"D", "O"}},
	}
	if got := len(mod.Params); got != len(want) {
		t.Fatalf("compact param count = %d, want %d", got, len(want))
	}
	for i, spec := range want {
		param := mod.Params[i]
		if param.Name != spec.name {
			t.Fatalf("param[%d] name = %q, want %q", i, param.Name, spec.name)
		}
		if !param.Trainable {
			t.Fatalf("param %q is not trainable", param.Name)
		}
		if param.Type.Tensor == nil {
			t.Fatalf("param %q is not a tensor", param.Name)
		}
		if param.Type.Tensor.DType != "q4" {
			t.Fatalf("param %q dtype = %q, want q4", param.Name, param.Type.Tensor.DType)
		}
		if strings.Join(param.Type.Tensor.Shape, ",") != strings.Join(spec.shape, ",") {
			t.Fatalf("param %q shape = %v, want %v", param.Name, param.Type.Tensor.Shape, spec.shape)
		}
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read compact manifest: %v", err)
	}
	if manifest.AttentionQueryParam != "layer0_attn_q" ||
		manifest.AttentionKeyParam != "layer0_attn_k" ||
		manifest.AttentionValueParam != "layer0_attn_v" ||
		manifest.AttentionOutputParam != "layer0_attn_o" ||
		manifest.HiddenProjectionParam != "layer0_ffn_up" ||
		manifest.ProjectionParam != "layer0_ffn_down" ||
		manifest.OutputProjectionParam != "output_projection" {
		t.Fatalf("compact manifest params drifted from artifact contract: %+v", manifest)
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read compact checkpoint: %v", err)
	}
	for _, spec := range want {
		if checkpoint.Tensors[spec.name] == nil {
			t.Fatalf("checkpoint missing compact tensor %q", spec.name)
		}
		if checkpoint.MomentTensors[spec.name+"_moment_1"] == nil ||
			checkpoint.MomentTensors[spec.name+"_moment_2"] == nil {
			t.Fatalf("checkpoint missing compact moments for %q", spec.name)
		}
	}
}

func TestInitDefaultEmbeddingPackageCreatesCompactMultiHeadServingGraph(t *testing.T) {
	path := filepath.Join(t.TempDir(), "compact-multihead.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		VocabSize:      12,
		MaxSequence:    6,
		ModelDim:       4,
		OutputDim:      4,
		HiddenDim:      8,
		AttentionHeads: 2,
		EncoderRepeats: 1,
	})
	if err != nil {
		t.Fatalf("init compact multi-head package: %v", err)
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read compact multi-head artifact: %v", err)
	}
	if got := compactMultiheadAttentionHeadsForTest(mod); got != "2" {
		t.Fatalf("compact_multihead_attention num_attention_heads = %q, want 2", got)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read compact multi-head manifest: %v", err)
	}
	rt := eosruntime.New(vulkan.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		t.Fatalf("load compact multi-head embedding package: %v", err)
	}
	result, err := model.Embed(context.Background(), []int32{1, 4, 5})
	if err != nil {
		t.Fatalf("embed compact multi-head package: %v", err)
	}
	if result.Embeddings == nil || len(result.Embeddings.Shape) != 1 || result.Embeddings.Shape[0] != manifest.OutputDim {
		t.Fatalf("compact multi-head embedding shape = %v, want [%d]", result.Embeddings.Shape, manifest.OutputDim)
	}
}

func TestInitDefaultEmbeddingPackageRejectsInvalidHeadConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad-heads.mll")
	_, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Architecture:   eosruntime.EmbeddingArchitectureCompactTransformerV1,
		ModelDim:       7,
		OutputDim:      7,
		HiddenDim:      14,
		AttentionHeads: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "model_dim 7 must be divisible by attention_heads 2") {
		t.Fatalf("InitDefaultEmbeddingPackage error = %v, want head divisibility error", err)
	}
}

func TestInitDefaultEmbeddingPackageQ4DeclaresQ4Params(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	wantParams := map[string]bool{
		"token_embedding": true,
		"role_embedding":  true,
		"attn_q":          true,
		"attn_k":          true,
		"attn_v":          true,
		"attn_o":          true,
		"ffn_up":          true,
		"projection":      true,
	}
	if got := len(mod.Params); got != len(wantParams) {
		t.Fatalf("param count = %d, want %d", got, len(wantParams))
	}
	for _, param := range mod.Params {
		if !wantParams[param.Name] {
			t.Fatalf("unexpected param %q", param.Name)
		}
		if param.Type.Tensor == nil || param.Type.Tensor.DType != "q4" {
			t.Fatalf("param %q dtype = %+v, want q4 tensor", param.Name, param.Type)
		}
		if !param.Trainable {
			t.Fatalf("param %q is not trainable", param.Name)
		}
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Config.WeightBits != 4 {
		t.Fatalf("weight bits = %d, want 4", checkpoint.Config.WeightBits)
	}
	if _, err := eosruntime.LoadEmbeddingTrainerPackage(path); err != nil {
		t.Fatalf("load training package: %v", err)
	}
}

func TestDefaultEmbeddingPackageV2PackagedInferenceParity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v2-generated.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		Name:         "eos-embed-v2-generated",
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}

	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.AttentionMaskMode != eosruntime.EmbeddingAttentionMaskModeKey {
		t.Fatalf("attention mask mode = %q, want %q", manifest.AttentionMaskMode, eosruntime.EmbeddingAttentionMaskModeKey)
	}
	if manifest.AttentionScoreScale != eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ {
		t.Fatalf("attention score scale = %q, want %q", manifest.AttentionScoreScale, eosruntime.EmbeddingAttentionScoreScaleKeyDimRSQ)
	}
	if manifest.PositionEncoding != eosruntime.EmbeddingPositionEncodingRoPE {
		t.Fatalf("position encoding = %q, want %q", manifest.PositionEncoding, eosruntime.EmbeddingPositionEncodingRoPE)
	}

	mod, err := eosartifact.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read artifact: %v", err)
	}
	if !moduleHasKernelOpForTest(mod, "masked_softmax") {
		t.Fatal("generated v2 artifact is missing masked_softmax")
	}
	if !moduleHasKernelOpForTest(mod, "rope") {
		t.Fatal("generated v2 artifact is missing rope")
	}
	if moduleRequiresCapabilityForTest(mod, eosartifact.CapabilityDeviceExecution) {
		t.Fatalf("generated v2 artifact unexpectedly declares device-native serving capability: %v", mod.Requirements.Capabilities)
	}

	rt := eosruntime.New(vulkan.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		t.Fatalf("load generated v2 embedding package: %v", err)
	}
	if got := model.Backend(); got == "" {
		t.Fatal("expected selected backend")
	}

	base, err := model.Embed(context.Background(), []int32{4, 5})
	if err != nil {
		t.Fatalf("embed base: %v", err)
	}
	padded, err := model.Embed(context.Background(), []int32{4, 5, 0, 0})
	if err != nil {
		t.Fatalf("embed padded: %v", err)
	}
	assertEmbeddingCloseForTest(t, "right-padding", padded.Embeddings.F32, base.Embeddings.F32, 1e-5)

	ordered, err := model.Embed(context.Background(), []int32{4, 5, 6})
	if err != nil {
		t.Fatalf("embed ordered: %v", err)
	}
	reordered, err := model.Embed(context.Background(), []int32{6, 5, 4})
	if err != nil {
		t.Fatalf("embed reordered: %v", err)
	}
	if maxAbs, l2 := embeddingDiffStatsForTest(ordered.Embeddings.F32, reordered.Embeddings.F32); maxAbs <= 1e-6 && l2 <= 1e-6 {
		t.Fatalf("v2 packaged embed is not order-sensitive: max_abs=%.9g l2=%.9g ordered=%v reordered=%v", maxAbs, l2, ordered.Embeddings.F32, reordered.Embeddings.F32)
	}

	batches := [][]int32{
		{4, 5, 0, 0},
		{6, 7, 8, 0},
		{9, 10, 11, 12},
	}
	batchResult, err := model.EmbedBatch(context.Background(), batches)
	if err != nil {
		t.Fatalf("embed batch: %v", err)
	}
	if got, want := batchResult.Embeddings.Shape, []int{len(batches), len(base.Embeddings.F32)}; len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("batch embedding shape = %v, want %v", got, want)
	}
	rowWidth := batchResult.Embeddings.Shape[1]
	for i, tokens := range batches {
		perExample, err := model.Embed(context.Background(), tokens)
		if err != nil {
			t.Fatalf("embed batch item %d: %v", i, err)
		}
		row := batchResult.Embeddings.F32[i*rowWidth : (i+1)*rowWidth]
		assertEmbeddingCloseForTest(t, "batch row", row, perExample.Embeddings.F32, 1e-5)
	}
}

func TestInitDefaultEmbeddingPackageRejectsUnknownWeightDType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		WeightDType:  "int4",
	}); err == nil {
		t.Fatal("expected weight dtype error")
	}
}

func moduleHasKernelOpForTest(mod *eosartifact.Module, op string) bool {
	if mod == nil {
		return false
	}
	for _, kernel := range mod.Kernels {
		for _, bodyOp := range kernel.Body {
			if bodyOp.Op == op {
				return true
			}
		}
	}
	return false
}

func compactMultiheadAttentionHeadsForTest(mod *eosartifact.Module) string {
	if mod == nil {
		return ""
	}
	for _, kernel := range mod.Kernels {
		for _, bodyOp := range kernel.Body {
			if bodyOp.Op == "compact_multihead_attention" {
				return bodyOp.Attributes["num_attention_heads"]
			}
		}
	}
	return ""
}

func moduleRequiresCapabilityForTest(mod *eosartifact.Module, capability string) bool {
	if mod == nil {
		return false
	}
	for _, got := range mod.Requirements.Capabilities {
		if got == capability {
			return true
		}
	}
	return false
}

func assertEmbeddingCloseForTest(t *testing.T, label string, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len = %d, want %d", label, len(got), len(want))
	}
	if maxAbs, l2 := embeddingDiffStatsForTest(got, want); maxAbs > tolerance || l2 > tolerance {
		t.Fatalf("%s mismatch: max_abs=%.9g l2=%.9g got=%v want=%v", label, maxAbs, l2, got, want)
	}
}

func embeddingDiffStatsForTest(a, b []float32) (float32, float32) {
	maxAbs := float32(0)
	sumSquares := float64(0)
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxAbs {
			maxAbs = diff
		}
		sumSquares += float64(diff * diff)
	}
	return maxAbs, float32(math.Sqrt(sumSquares))
}

func TestDefaultEmbeddingPackageQ4TrainsContrastive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	}); err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	trainer, err := eosruntime.LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	checkpoint, err := trainer.Checkpoint()
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if checkpoint.Config.WeightBits != 4 {
		t.Fatalf("trainer weight bits = %d, want 4", checkpoint.Config.WeightBits)
	}
	trainSet := []eosruntime.EmbeddingContrastiveExample{
		{QueryTokens: []int32{4, 5}, PositiveTokens: []int32{4, 4}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
		{QueryTokens: []int32{6, 7}, PositiveTokens: []int32{6, 6}, QueryMask: []int32{1, 1}, PositiveMask: []int32{1, 1}},
	}
	summary, err := trainer.FitContrastive(trainSet, trainSet, eosruntime.EmbeddingTrainRunConfig{
		Epochs:    2,
		BatchSize: 2,
		Seed:      7,
	})
	if err != nil {
		t.Fatalf("fit contrastive: %v", err)
	}
	if summary.FinalEval == nil {
		t.Fatal("expected final eval metrics")
	}
}

func TestExportDefaultEmbeddingPackageQ4SealsPackedTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	if _, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:    16,
		MaxSequence:  8,
		EmbeddingDim: 4,
		HiddenDim:    8,
		Seed:         7,
		WeightDType:  "q4",
	}); err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}

	outPath, err := eosruntime.ExportPackageToMLL(path, "")
	if err != nil {
		t.Fatalf("ExportPackageToMLL: %v", err)
	}
	reader, err := mll.ReadFile(outPath, mll.WithDigestVerification())
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if reader.Profile() != mll.ProfileSealed {
		t.Fatalf("profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}

	xmtaBody, ok := reader.Section(eosartifact.MLLTagXMTA)
	if !ok {
		t.Fatal("missing XMTA section")
	}
	meta, err := eosartifact.DecodeMLLMetadata(xmtaBody)
	if err != nil {
		t.Fatalf("decode XMTA: %v", err)
	}
	if got := meta.LogicalTensorDType["token_embedding"]; got != "q4" {
		t.Fatalf("logical dtype for token_embedding = %q, want q4", got)
	}

	strgBody, _ := reader.Section(mll.TagSTRG)
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		t.Fatalf("ReadStringTable: %v", err)
	}
	tnsrBody, _ := reader.Section(mll.TagTNSR)
	tnsrSection, err := mll.ReadTnsrSection(tnsrBody)
	if err != nil {
		t.Fatalf("ReadTnsrSection: %v", err)
	}
	if got := len(tnsrSection.Tensors); got != 8 {
		t.Fatalf("tensor count = %d, want 8", got)
	}

	// q4 tensors are sealed as real packed payloads: storage dtype Q4, two
	// offset-binary nibbles per byte, and a per-tensor scale in XMTA.
	for _, entry := range tnsrSection.Tensors {
		name := strg.At(entry.NameIdx)
		if meta.LogicalTensorDType[name] != "q4" {
			t.Fatalf("tensor %q logical dtype = %q, want q4", name, meta.LogicalTensorDType[name])
		}
		if entry.DType != mll.DTypeQ4 {
			t.Fatalf("tensor %q stored dtype = %d, want packed q4 (%d)", name, entry.DType, mll.DTypeQ4)
		}
		elements := uint64(1)
		for _, dim := range entry.Shape {
			elements *= dim
		}
		if want := (elements + 1) / 2; uint64(len(entry.Data)) != want {
			t.Fatalf("tensor %q packed bytes = %d, want %d", name, len(entry.Data), want)
		}
		scale, ok := meta.TensorScales[name]
		if !ok {
			t.Fatalf("tensor %q missing packed scale (scales=%v)", name, meta.TensorScales)
		}
		if name == "role_embedding" {
			if scale != 0 {
				t.Fatalf("tensor %q scale = %g, want zero for zero-initialized role embeddings", name, scale)
			}
		} else if scale <= 0 {
			t.Fatalf("tensor %q scale = %g, want positive", name, scale)
		}
	}
}

func TestInitDefaultEmbeddingPackageHonorsEncoderRepeats(t *testing.T) {
	path := filepath.Join(t.TempDir(), "eos-embed-v1.mll")
	paths, err := InitDefaultEmbeddingPackage(path, DefaultEmbeddingPackageConfig{
		VocabSize:      16,
		MaxSequence:    8,
		EmbeddingDim:   4,
		HiddenDim:      8,
		EncoderRepeats: 3,
		Seed:           7,
	})
	if err != nil {
		t.Fatalf("init default embedding package: %v", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read embedding manifest: %v", err)
	}
	if manifest.EncoderRepeats != 3 {
		t.Fatalf("encoder repeats = %d, want 3", manifest.EncoderRepeats)
	}
}

func TestDefaultEmbedderAssetInfoAndVerify(t *testing.T) {
	info, err := DefaultEmbedderAssetInfo("")
	if err != nil {
		t.Fatalf("default embedder asset info: %v", err)
	}
	if info.AssetID != DefaultEmbedderAssetID {
		t.Fatalf("asset id = %q, want %q", info.AssetID, DefaultEmbedderAssetID)
	}
	if info.ModelName != DefaultEmbeddingModelName {
		t.Fatalf("model name = %q, want %q", info.ModelName, DefaultEmbeddingModelName)
	}
	if filepath.Base(info.ArtifactPath) != DefaultEmbedderArtifactFilename {
		t.Fatalf("artifact path = %q", info.ArtifactPath)
	}
	if filepath.Base(info.TokenizerPath) != DefaultEmbedderTokenizerFilename {
		t.Fatalf("tokenizer path = %q", info.TokenizerPath)
	}
	report, err := VerifyDefaultEmbedderAsset("")
	if err != nil {
		t.Fatalf("verify default embedder asset: %v", err)
	}
	if !report.OK || len(report.Files) != 2 {
		t.Fatalf("unexpected verification report: %+v", report)
	}
	for _, check := range report.Files {
		if !check.OK || check.SHA256 != check.ExpectedSHA256 || check.Bytes <= 0 {
			t.Fatalf("bad file check: %+v", check)
		}
	}
}

func TestImportedEmbedderCandidateAssetInfoOverrideIsNonDefault(t *testing.T) {
	override := filepath.Join(t.TempDir(), "fixture.imported.mll")
	info, err := ImportedEmbedderCandidateAssetInfo("", override)
	if err != nil {
		t.Fatalf("imported candidate asset info: %v", err)
	}
	if info.CandidateID != ImportedEmbedderCandidateID {
		t.Fatalf("candidate id = %q, want %q", info.CandidateID, ImportedEmbedderCandidateID)
	}
	if info.PublicID != ImportedEmbedderCandidatePublicID {
		t.Fatalf("public id = %q, want %q", info.PublicID, ImportedEmbedderCandidatePublicID)
	}
	if info.ModelName != ImportedEmbedderCandidateModelName {
		t.Fatalf("model name = %q, want %q", info.ModelName, ImportedEmbedderCandidateModelName)
	}
	if info.ModelName != ImportedEmbedderCandidatePublicID || info.ModelName == DefaultEmbeddingModelName {
		t.Fatalf("public model name = %q, default legacy name = %q", info.ModelName, DefaultEmbeddingModelName)
	}
	if info.DisplayName != ImportedEmbedderCandidatePublicName {
		t.Fatalf("display name = %q, want %q", info.DisplayName, ImportedEmbedderCandidatePublicName)
	}
	if info.LegacyModelName != ImportedEmbedderCandidateLegacyModelName {
		t.Fatalf("legacy model name = %q, want %q", info.LegacyModelName, ImportedEmbedderCandidateLegacyModelName)
	}
	if info.SourceModel != ImportedEmbedderCandidateSourceModel {
		t.Fatalf("source model = %q", info.SourceModel)
	}
	if info.Status != ImportedEmbedderCandidateStatus {
		t.Fatalf("status = %q", info.Status)
	}
	if info.QualityClaim || info.DefaultAliasChanged {
		t.Fatalf("candidate unexpectedly claims default or quality: %+v", info)
	}
	if info.PackagePath != override {
		t.Fatalf("package path = %q, want %q", info.PackagePath, override)
	}
	if info.PackageSHA256 != ImportedEmbedderCandidatePackageSHA256 || info.PackageIdentity != ImportedEmbedderCandidatePackageIdentity {
		t.Fatalf("candidate expected identity changed: %+v", info)
	}
	if info.SourceSnapshotCommit != ImportedEmbedderCandidateSourceSnapshot || info.UpstreamModelURL != ImportedEmbedderCandidateUpstreamModelURL {
		t.Fatalf("candidate provenance metadata changed: %+v", info)
	}
	if info.LicenseID != ImportedEmbedderCandidateLicenseID || !info.LicenseNoticeRequired || !info.ProvenanceNoticeRequired || info.Attribution != ImportedEmbedderCandidateAttribution {
		t.Fatalf("candidate license metadata changed: %+v", info)
	}
	if info.RoleContractSchema != eosruntime.PretrainedBERTRetrievalRoleContractSchema ||
		info.QueryRole != ImportedEmbedderCandidateQueryRole ||
		info.QueryPrefix != ImportedEmbedderCandidateQueryPrefix ||
		info.DocumentRole != ImportedEmbedderCandidateDocumentRole ||
		info.DocumentPrefix != ImportedEmbedderCandidateDocumentPrefix ||
		info.Pooling != ImportedEmbedderCandidatePooling ||
		info.Normalization != ImportedEmbedderCandidateNormalization ||
		info.MaxLength != ImportedEmbedderCandidateMaxLength ||
		info.NativeDim != ImportedEmbedderCandidateNativeDim {
		t.Fatalf("candidate role contract metadata changed: %+v", info)
	}
}

func TestImportedEmbedderCandidateNoticePacketContainsProvenance(t *testing.T) {
	noticePath := filepath.Join("..", "NOTICE")
	noticeBytes, err := os.ReadFile(noticePath)
	if err != nil {
		t.Fatalf("read %s: %v", noticePath, err)
	}
	notice := string(noticeBytes)
	for _, want := range []string{
		"Imported BGE non-default candidate",
		ImportedEmbedderCandidateID,
		ImportedEmbedderCandidateSourceModel,
		ImportedEmbedderCandidateSourceSnapshot,
		ImportedEmbedderCandidatePackageSHA256,
		ImportedEmbedderCandidatePackageIdentity,
		ImportedEmbedderCandidateQueryPrefix,
		"quality_claim=false",
		"default_alias_changed=false",
		"Copyright (c) 2022 staoxiao",
		"Permission is hereby granted, free of charge",
	} {
		if !strings.Contains(notice, want) {
			t.Fatalf("NOTICE missing %q", want)
		}
	}
}

func TestImportedEmbedderCandidateAssetInfoRootUsesRunArtifactPath(t *testing.T) {
	root := t.TempDir()
	packagePath := filepath.Join(root, ImportedEmbedderCandidatePackageRelativePath)
	if err := os.MkdirAll(filepath.Dir(packagePath), 0o755); err != nil {
		t.Fatalf("mkdir package dir: %v", err)
	}
	if err := os.WriteFile(packagePath, []byte("fixture"), 0o644); err != nil {
		t.Fatalf("write package fixture: %v", err)
	}
	info, err := ImportedEmbedderCandidateAssetInfo(root, "")
	if err != nil {
		t.Fatalf("imported candidate asset info with root: %v", err)
	}
	want := filepath.Join(root, ImportedEmbedderCandidatePackageRelativePath)
	if info.PackagePath != want {
		t.Fatalf("package path = %q, want %q", info.PackagePath, want)
	}
}

func TestImportedEmbedderCandidateAssetInfoRequiresExplicitPath(t *testing.T) {
	_, err := ImportedEmbedderCandidateAssetInfo("", "")
	if err == nil || !strings.Contains(err.Error(), "package path is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}
