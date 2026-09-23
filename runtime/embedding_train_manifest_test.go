package eosruntime

import (
	"path/filepath"
	"strings"
	"testing"

	"m31labs.dev/eos/compiler"
)

func TestDefaultEmbeddingTrainManifestPath(t *testing.T) {
	got := DefaultEmbeddingTrainManifestPath("/tmp/tiny_train_embed_q8.mll")
	if want := "/tmp/tiny_train_embed_q8.train.mll"; got != want {
		t.Fatalf("train manifest path = %q, want %q", got, want)
	}
}

func TestEmbeddingTrainManifestRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:                          0.05,
			WeightBits:                            8,
			Optimizer:                             "adamw",
			Beta1:                                 0.9,
			Beta2:                                 0.999,
			Epsilon:                               1e-8,
			ContrastiveLoss:                       "infonce",
			Temperature:                           0.05,
			MatryoshkaDims:                        []int{64, 128},
			MatryoshkaWeights:                     []float32{0.5, 1},
			TurboQuantPrefixBits:                  []int{2, 4},
			TurboQuantPrefixWeight:                0.25,
			TurboQuantPrefixSeed:                  DefaultTurboQuantMultiVectorQuantizerSeed,
			TurboQuantPrefixScoreMode:             TurboQuantPrefixScoreModePreparedIP,
			TurboQuantTopKObjectives:              []TurboQuantPrefixObjective{{Dim: 4, BitWidth: 3, Weight: 0.5}},
			TurboQuantTopKLoss:                    TurboQuantTopKLossLambdaNDCG,
			TurboQuantTopKCutoff:                  7,
			TurboQuantTopKTau:                     0.08,
			TurboQuantTopKMargin:                  0.01,
			TurboQuantTopKNegativeMask:            TurboQuantTopKNegativeMaskQ3,
			TurboQuantTopKRecallWeight:            0.2,
			TurboQuantTopKRecallCutoff:            100,
			TurboQuantTopKRecallTau:               0.07,
			TurboQuantTopKRecallMargin:            0.03,
			TurboQuantTopKRecallNegativeMask:      TurboQuantTopKNegativeMaskBM25,
			ScoreSpectrumLossMode:                 ScoreSpectrumLossModeHardSoftRecovery,
			ScoreSpectrumRecoveryWeight:           1.5,
			ScoreSpectrumRecoveryMargin:           0.2,
			ScoreSpectrumRecoveryTopK:             3,
			ScoreSpectrumRecoveryTau:              0.07,
			ScoreSpectrumActivationMicrobatchSize: 7,
		},
		ScoreSpectrum: EmbeddingScoreSpectrumPolicy{
			ScoreSpectrumTrain:        true,
			ScoreSpectrumResearchOnly: true,
			TrainAllowedForResearch:   true,
			ReleaseTrainAllowed:       false,
			CommercialUseAllowed:      false,
			SourceArtifactHashes:      []string{"bbb", "aaa"},
			ScoreSpectrumRowCount:     2,
			AutoClearedObjectives:     []string{"turboquant_prefix_seed", "matryoshka"},
			IsolatedInheritedObjectives: []string{
				"turboquant_prefix_seed",
				"matryoshka",
			},
		},
		ListwiseGeometry: EmbeddingListwiseGeometryPolicy{
			ListwiseGeometryTrain:        true,
			ListwiseGeometryResearchOnly: true,
			TrainAllowedForResearch:      true,
			ReleaseTrainAllowed:          false,
			CommercialUseAllowed:         false,
			SourceArtifactHashes:         []string{"lg-bbb", "lg-aaa"},
			ListwiseGeometryBatchCount:   4,
			AutoClearedObjectives:        []string{"turboquant_prefix_seed", "matryoshka"},
			IsolatedInheritedObjectives: []string{
				"turboquant_prefix_seed",
				"matryoshka",
			},
		},
	}
	want.Embedding.ArchitectureVersion = EmbeddingArchitectureLegacyV1
	want.Embedding.ModelDim = 8
	want.Embedding.OutputDim = 4
	want.Embedding.AttentionHeads = 2
	want.Embedding.HeadDim = 4
	want.Embedding.FFNDim = 32
	want.Embedding.ParameterTying = EmbeddingParameterTyingUntied
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if got.Name != want.Name || got.Embedding.Name != want.Embedding.Name || got.Config.Optimizer != want.Config.Optimizer || got.Config.ContrastiveLoss != want.Config.ContrastiveLoss || got.Config.Temperature != want.Config.Temperature {
		t.Fatalf("manifest mismatch:\nwant: %+v\ngot:  %+v", want, got)
	}
	if got.Embedding.ArchitectureVersion != want.Embedding.ArchitectureVersion ||
		got.Embedding.ModelDim != want.Embedding.ModelDim ||
		got.Embedding.OutputDim != want.Embedding.OutputDim ||
		got.Embedding.AttentionHeads != want.Embedding.AttentionHeads ||
		got.Embedding.HeadDim != want.Embedding.HeadDim ||
		got.Embedding.FFNDim != want.Embedding.FFNDim ||
		got.Embedding.ParameterTying != want.Embedding.ParameterTying {
		t.Fatalf("embedding architecture metadata mismatch:\nwant: %+v\ngot:  %+v", want.Embedding, got.Embedding)
	}
	if len(got.Config.TurboQuantPrefixBits) != 2 || got.Config.TurboQuantPrefixBits[0] != 2 || got.Config.TurboQuantPrefixBits[1] != 4 {
		t.Fatalf("turboquant prefix bits = %v, want [2 4]", got.Config.TurboQuantPrefixBits)
	}
	if got.Config.TurboQuantPrefixWeight != 0.25 {
		t.Fatalf("turboquant prefix weight = %f, want 0.25", got.Config.TurboQuantPrefixWeight)
	}
	if got.Config.TurboQuantPrefixSeed != DefaultTurboQuantMultiVectorQuantizerSeed {
		t.Fatalf("turboquant prefix seed = %d, want %d", got.Config.TurboQuantPrefixSeed, DefaultTurboQuantMultiVectorQuantizerSeed)
	}
	if got.Config.TurboQuantPrefixScoreMode != TurboQuantPrefixScoreModePreparedIP {
		t.Fatalf("turboquant prefix score mode = %q, want %q", got.Config.TurboQuantPrefixScoreMode, TurboQuantPrefixScoreModePreparedIP)
	}
	if formatted := FormatTurboQuantPrefixObjectives(got.Config.TurboQuantTopKObjectives); formatted != "4:3=0.5" {
		t.Fatalf("turboquant top-k objectives = %q", formatted)
	}
	if got.Config.TurboQuantTopKLoss != TurboQuantTopKLossLambdaNDCG || got.Config.TurboQuantTopKCutoff != 7 || got.Config.TurboQuantTopKTau != 0.08 || got.Config.TurboQuantTopKMargin != 0.01 || got.Config.TurboQuantTopKNegativeMask != TurboQuantTopKNegativeMaskQ3 {
		t.Fatalf("turboquant top-k config = %+v, want lambdandcg cutoff=7 tau=0.08 margin=0.01 mask=q3", got.Config)
	}
	if got.Config.TurboQuantTopKRecallWeight != 0.2 || got.Config.TurboQuantTopKRecallCutoff != 100 || got.Config.TurboQuantTopKRecallTau != 0.07 || got.Config.TurboQuantTopKRecallMargin != 0.03 || got.Config.TurboQuantTopKRecallNegativeMask != TurboQuantTopKNegativeMaskBM25 {
		t.Fatalf("turboquant top-k recall config = %+v, want weight=0.2 cutoff=100 tau=0.07 margin=0.03 mask=bm25", got.Config)
	}
	if got.Config.ScoreSpectrumLossMode != ScoreSpectrumLossModeHardSoftRecovery || got.Config.ScoreSpectrumRecoveryWeight != 1.5 || got.Config.ScoreSpectrumRecoveryMargin != 0.2 || got.Config.ScoreSpectrumRecoveryTopK != 3 || got.Config.ScoreSpectrumRecoveryTau != 0.07 {
		t.Fatalf("score-spectrum recovery config = %+v, want hard_soft_recovery weight=1.5 margin=0.2 topK=3 tau=0.07", got.Config)
	}
	if got.Config.ScoreSpectrumActivationMicrobatchSize != 7 {
		t.Fatalf("score-spectrum activation microbatch size = %d, want 7", got.Config.ScoreSpectrumActivationMicrobatchSize)
	}
	if !got.ScoreSpectrum.ScoreSpectrumTrain || !got.ScoreSpectrum.ScoreSpectrumResearchOnly || !got.ScoreSpectrum.TrainAllowedForResearch || got.ScoreSpectrum.ReleaseTrainAllowed || got.ScoreSpectrum.CommercialUseAllowed {
		t.Fatalf("score-spectrum policy mismatch: %+v", got.ScoreSpectrum)
	}
	if got.ScoreSpectrum.ScoreSpectrumRowCount != 2 || len(got.ScoreSpectrum.SourceArtifactHashes) != 2 || got.ScoreSpectrum.SourceArtifactHashes[0] != "aaa" || got.ScoreSpectrum.SourceArtifactHashes[1] != "bbb" {
		t.Fatalf("score-spectrum provenance mismatch: %+v", got.ScoreSpectrum)
	}
	if len(got.ScoreSpectrum.AutoClearedObjectives) != 2 || got.ScoreSpectrum.AutoClearedObjectives[0] != "matryoshka" || got.ScoreSpectrum.AutoClearedObjectives[1] != "turboquant_prefix_seed" {
		t.Fatalf("auto-cleared objectives = %v", got.ScoreSpectrum.AutoClearedObjectives)
	}
	if len(got.ScoreSpectrum.IsolatedInheritedObjectives) != 2 || got.ScoreSpectrum.IsolatedInheritedObjectives[0] != "matryoshka" || got.ScoreSpectrum.IsolatedInheritedObjectives[1] != "turboquant_prefix_seed" {
		t.Fatalf("isolated inherited objectives = %v", got.ScoreSpectrum.IsolatedInheritedObjectives)
	}
	if !got.ListwiseGeometry.ListwiseGeometryTrain || !got.ListwiseGeometry.ListwiseGeometryResearchOnly || !got.ListwiseGeometry.TrainAllowedForResearch || got.ListwiseGeometry.ReleaseTrainAllowed || got.ListwiseGeometry.CommercialUseAllowed {
		t.Fatalf("listwise geometry policy mismatch: %+v", got.ListwiseGeometry)
	}
	if got.ListwiseGeometry.ListwiseGeometryBatchCount != 4 || len(got.ListwiseGeometry.SourceArtifactHashes) != 2 || got.ListwiseGeometry.SourceArtifactHashes[0] != "lg-aaa" || got.ListwiseGeometry.SourceArtifactHashes[1] != "lg-bbb" {
		t.Fatalf("listwise geometry provenance mismatch: %+v", got.ListwiseGeometry)
	}
	if len(got.ListwiseGeometry.AutoClearedObjectives) != 2 || got.ListwiseGeometry.AutoClearedObjectives[0] != "matryoshka" || got.ListwiseGeometry.AutoClearedObjectives[1] != "turboquant_prefix_seed" {
		t.Fatalf("listwise auto-cleared objectives = %v", got.ListwiseGeometry.AutoClearedObjectives)
	}
	if len(got.ListwiseGeometry.IsolatedInheritedObjectives) != 2 || got.ListwiseGeometry.IsolatedInheritedObjectives[0] != "matryoshka" || got.ListwiseGeometry.IsolatedInheritedObjectives[1] != "turboquant_prefix_seed" {
		t.Fatalf("listwise isolated inherited objectives = %v", got.ListwiseGeometry.IsolatedInheritedObjectives)
	}
}

func TestEmbeddingTrainManifestRoundTripTurboQuantPrefixObjectives(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:      0.05,
			WeightBits:        8,
			Optimizer:         "adamw",
			Beta1:             0.9,
			Beta2:             0.999,
			Epsilon:           1e-8,
			ContrastiveLoss:   "infonce",
			Temperature:       0.05,
			MatryoshkaDims:    []int{64, 128},
			MatryoshkaWeights: []float32{0.5, 1},
			TurboQuantPrefixObjectives: []TurboQuantPrefixObjective{
				{Dim: 128, BitWidth: 4, Weight: 0.5},
				{Dim: 64, BitWidth: 2, Weight: 0},
			},
			TurboQuantPrefixSeed:      DefaultTurboQuantMultiVectorQuantizerSeed,
			TurboQuantPrefixScoreMode: TurboQuantPrefixScoreModePreparedIP,
		},
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if formatted := FormatTurboQuantPrefixObjectives(got.Config.TurboQuantPrefixObjectives); formatted != "64:2=0,128:4=0.5" {
		t.Fatalf("turboquant prefix objectives = %q", formatted)
	}
}

func TestEmbeddingTrainManifestRoundTripTurboQuantRankMarginObjectives(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:    0.05,
			WeightBits:      8,
			Optimizer:       "adamw",
			ContrastiveLoss: "infonce",
			Temperature:     0.05,
			MatryoshkaDims:  []int{64, 128},
			TurboQuantRankMarginObjectives: []TurboQuantPrefixObjective{
				{Dim: 128, BitWidth: 4, Weight: 0.1},
			},
			TurboQuantRankMargin:          0.03,
			TurboQuantRankMarginLoss:      TurboQuantRankMarginLossSoftplus,
			TurboQuantRankMarginReduction: TurboQuantRankMarginReductionMeanEligible,
			TurboQuantRankMarginTau:       0.04,
			TurboQuantPrefixSeed:          DefaultTurboQuantMultiVectorQuantizerSeed,
			TurboQuantPrefixScoreMode:     TurboQuantPrefixScoreModePreparedIP,
		},
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if formatted := FormatTurboQuantPrefixObjectives(got.Config.TurboQuantRankMarginObjectives); formatted != "128:4=0.1" {
		t.Fatalf("turboquant rank-margin objectives = %q", formatted)
	}
	if got.Config.TurboQuantRankMargin != 0.03 {
		t.Fatalf("turboquant rank margin = %f, want 0.03", got.Config.TurboQuantRankMargin)
	}
	if got.Config.TurboQuantRankMarginLoss != TurboQuantRankMarginLossSoftplus || got.Config.TurboQuantRankMarginReduction != TurboQuantRankMarginReductionMeanEligible || got.Config.TurboQuantRankMarginTau != 0.04 {
		t.Fatalf("turboquant rank-margin shape = %q/%q/%f, want softplus/mean_eligible/0.04", got.Config.TurboQuantRankMarginLoss, got.Config.TurboQuantRankMarginReduction, got.Config.TurboQuantRankMarginTau)
	}
}

func TestEmbeddingTrainManifestRoundTripTurboQuantCompactObjectives(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny_train_embed_q8.train.mll")
	want := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config: EmbeddingTrainConfig{
			LearningRate:    0.05,
			WeightBits:      8,
			Optimizer:       "adamw",
			ContrastiveLoss: "infonce",
			Temperature:     0.05,
			MatryoshkaDims:  []int{64, 128},
			TurboQuantCompactObjectives: []TurboQuantPrefixObjective{
				{Dim: 128, BitWidth: 4, Weight: 0.05},
				{Dim: 64, BitWidth: 2, Weight: 0},
			},
			TurboQuantPrefixSeed: DefaultTurboQuantMultiVectorQuantizerSeed,
		},
	}
	if err := want.WriteFile(path); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	got, err := ReadEmbeddingTrainManifestFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if formatted := FormatTurboQuantPrefixObjectives(got.Config.TurboQuantCompactObjectives); formatted != "64:2=0,128:4=0.05" {
		t.Fatalf("turboquant compact objectives = %q", formatted)
	}
}

func TestEmbeddingTrainManifestValidateModule(t *testing.T) {
	src := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T]) -> f16[E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T]) -> f16[B, E] {
    let hidden_q = gather(token_embedding, tokens)
    let hidden = dequant(hidden_q)
    let projection_f = dequant(projection)
    let projected = @matmul(hidden, projection_f)
    let normalized = normalize(projected)
    return mean_pool(normalized, attention_mask)
}
`)

	bundle, err := compiler.Build(src, compiler.Options{ModuleName: "tiny_train_embed_q8"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	manifest := EmbeddingTrainManifest{
		Name:      "tiny_train_embed_q8",
		Embedding: tinyMaskedEmbeddingManifest(),
		Config:    EmbeddingTrainConfig{LearningRate: 0.05},
	}
	if err := manifest.ValidateModule(bundle.Artifact); err != nil {
		t.Fatalf("validate train manifest: %v", err)
	}

	unsupported := manifest
	unsupported.Embedding.ArchitectureVersion = EmbeddingArchitectureCompactTransformerV1
	unsupported.Embedding.ModelDim = 2
	unsupported.Embedding.OutputDim = 2
	unsupported.Embedding.AttentionHeads = 1
	unsupported.Embedding.HeadDim = 2
	unsupported.Embedding.ParameterTying = EmbeddingParameterTyingUntied
	if err := unsupported.ValidateModule(bundle.Artifact); err == nil || !strings.Contains(err.Error(), "compact_transformer_v1 is not supported by trainable package initialization yet") {
		t.Fatalf("compact ValidateModule error = %v, want unsupported compact error", err)
	}
}
