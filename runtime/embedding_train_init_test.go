package eosruntime

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
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

func TestInitializeEmbeddingTrainerPackageWithManifestCreatesPackage(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> f16[E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> f16[B, E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
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
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	if err := manifest.WriteFile(DefaultEmbeddingManifestPath(path)); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	paths, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       7,
		ShapeSizes: map[string]int{"D": 4, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize training package: %v", err)
	}
	for _, candidate := range []string{
		paths.EmbeddingManifestPath,
		paths.WeightFilePath,
		paths.MemoryPlanPath,
		paths.TrainManifestPath,
		paths.CheckpointPath,
		paths.TrainProfilePath,
	} {
		if _, err := os.Stat(candidate); err != nil {
			t.Fatalf("expected package file %q: %v", candidate, err)
		}
	}
	trainer, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load training package: %v", err)
	}
	if got := trainer.tokenEmbed.Shape; len(got) != 2 || got[0] != 8 || got[1] != 4 {
		t.Fatalf("token embedding shape = %v, want [8 4]", got)
	}
	if got := trainer.projection.Shape; len(got) != 2 || got[0] != 4 || got[1] != 3 {
		t.Fatalf("projection shape = %v, want [4 3]", got)
	}
}

func TestInitializeEmbeddingTrainerPackageRejectsUnresolvedShapes(t *testing.T) {
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> f16[E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> f16[B, E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
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
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer:           TokenizerManifest{VocabSize: 8},
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{}); err == nil {
		t.Fatal("expected unresolved symbolic dim error")
	}
}

func TestInitializeEmbeddingTrainerPackageNormalizesRoleManifestBeforeWeights(t *testing.T) {
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "role_train_init"})
	if err != nil {
		t.Fatalf("build role source: %v", err)
	}
	path := filepath.Join(t.TempDir(), "role_train_init.mll")
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := tinyRoleEmbeddingManifest()
	manifest.Name = "role_train_init"
	manifest.RoleEmbeddingParam = ""
	manifest.RoleInput = ""
	manifest.BatchRoleInput = ""
	paths, err := InitializeEmbeddingTrainerPackageWithManifest(path, manifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       7,
		ShapeSizes: map[string]int{"D": 2},
	})
	if err != nil {
		t.Fatalf("initialize role package: %v", err)
	}
	trainer, err := LoadEmbeddingTrainerPackage(path)
	if err != nil {
		t.Fatalf("load role package: %v", err)
	}
	if trainer.roleEmbed == nil || trainer.roleParam.Name != "role_embedding" {
		t.Fatalf("role tensor/param = %q %+v", trainer.roleParam.Name, trainer.roleEmbed)
	}
	for i, v := range trainer.roleEmbed.F32 {
		if v != 0 {
			t.Fatalf("role embedding[%d] = %f, want zero init", i, v)
		}
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	if checkpoint.Manifest.RoleEmbeddingParam != "role_embedding" || checkpoint.RoleEmbedding == nil {
		t.Fatalf("checkpoint role manifest/tensor = %q %+v", checkpoint.Manifest.RoleEmbeddingParam, checkpoint.RoleEmbedding)
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapsDefaultedRoleEmbedding(t *testing.T) {
	dir := t.TempDir()
	bundle, err := compiler.Build([]byte(tinyRoleEmbeddingSource()), compiler.Options{ModuleName: "role_bootstrap_init"})
	if err != nil {
		t.Fatalf("build role source: %v", err)
	}
	sourcePath := filepath.Join(dir, "role_bootstrap_source.mll")
	if err := eosartifact.WriteFile(sourcePath, bundle.Artifact); err != nil {
		t.Fatalf("write source artifact: %v", err)
	}
	sourceManifest := tinyRoleEmbeddingManifest()
	sourceManifest.Name = "role_bootstrap_init"
	sourcePaths, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2},
	})
	if err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sourceCheckpoint, err := ReadEmbeddingTrainCheckpointFile(sourcePaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read source checkpoint: %v", err)
	}
	sourceCheckpoint.RoleEmbedding = backend.NewTensorF32([]int{3, 2}, []float32{
		0, 0,
		11, 12,
		21, 22,
	})
	if err := sourceCheckpoint.WriteFile(sourcePaths.CheckpointPath); err != nil {
		t.Fatalf("rewrite source checkpoint: %v", err)
	}

	targetPath := filepath.Join(dir, "role_bootstrap_target.mll")
	if err := eosartifact.WriteFile(targetPath, bundle.Artifact); err != nil {
		t.Fatalf("write target artifact: %v", err)
	}
	targetManifest := tinyRoleEmbeddingManifest()
	targetManifest.Name = "role_bootstrap_init"
	targetManifest.RoleEmbeddingParam = ""
	targetManifest.RoleInput = ""
	targetManifest.BatchRoleInput = ""
	targetPaths, err := InitializeEmbeddingTrainerPackageWithManifest(targetPath, targetManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:                  7,
		BootstrapArtifactPath: sourcePath,
		ShapeSizes:            map[string]int{"D": 2},
	})
	if err != nil {
		t.Fatalf("initialize target package: %v", err)
	}
	targetCheckpoint, err := ReadEmbeddingTrainCheckpointFile(targetPaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	for i, got := range targetCheckpoint.RoleEmbedding.F32 {
		if want := sourceCheckpoint.RoleEmbedding.F32[i]; got != want {
			t.Fatalf("target role embedding[%d] = %f, want bootstrap %f", i, got, want)
		}
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapCopiesOverlap(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_embed.mll")
	sourcePaths, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	})
	if err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sourceCheckpoint, err := ReadEmbeddingTrainCheckpointFile(sourcePaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read source checkpoint: %v", err)
	}
	sourceCheckpoint.Step = 17
	sourceCheckpoint.TokenEmbedding = backend.NewTensorF32([]int{8, 2}, []float32{
		101, 102,
		103, 104,
		105, 106,
		107, 108,
		109, 110,
		111, 112,
		113, 114,
		115, 116,
	})
	sourceCheckpoint.Projection = backend.NewTensorF32([]int{2, 2}, []float32{
		201, 202,
		203, 204,
	})
	sourceCheckpoint.TokenMoment1 = backend.NewTensorF32([]int{8, 2}, filledFloat32(16, 9))
	sourceCheckpoint.TokenMoment2 = backend.NewTensorF32([]int{8, 2}, filledFloat32(16, 8))
	sourceCheckpoint.ProjMoment1 = backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 7))
	sourceCheckpoint.ProjMoment2 = backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 6))
	if err := sourceCheckpoint.WriteFile(sourcePaths.CheckpointPath); err != nil {
		t.Fatalf("rewrite source checkpoint: %v", err)
	}

	targetPath, targetManifest := buildTinyTrainInitArtifact(t, dir, "target_embed.mll")
	baselinePath, baselineManifest := buildTinyTrainInitArtifact(t, dir, "baseline_embed.mll")
	baselinePaths, err := InitializeEmbeddingTrainerPackageWithManifest(baselinePath, baselineManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       11,
		ShapeSizes: map[string]int{"D": 3, "E": 4},
	})
	if err != nil {
		t.Fatalf("initialize baseline package: %v", err)
	}
	targetPaths, err := InitializeEmbeddingTrainerPackageWithManifest(targetPath, targetManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:                  11,
		BootstrapArtifactPath: sourcePath,
		ShapeSizes:            map[string]int{"D": 3, "E": 4},
	})
	if err != nil {
		t.Fatalf("initialize target package with bootstrap: %v", err)
	}

	baseline, err := ReadEmbeddingTrainCheckpointFile(baselinePaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read baseline checkpoint: %v", err)
	}
	target, err := ReadEmbeddingTrainCheckpointFile(targetPaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	if target.Step != 0 {
		t.Fatalf("target step = %d, want 0", target.Step)
	}
	assertFloat32SliceEqual(t, target.TokenEmbedding.Shape, []int{8, 3})
	assertFloat32SliceEqual(t, target.Projection.Shape, []int{3, 4})

	for row := 0; row < 8; row++ {
		for col := 0; col < 2; col++ {
			got := target.TokenEmbedding.F32[row*3+col]
			want := sourceCheckpoint.TokenEmbedding.F32[row*2+col]
			if got != want {
				t.Fatalf("token overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		if got, want := target.TokenEmbedding.F32[row*3+2], baseline.TokenEmbedding.F32[row*3+2]; got != want {
			t.Fatalf("token non-overlap[%d,2] = %f, want initialized baseline %f", row, got, want)
		}
	}
	for row := 0; row < 2; row++ {
		for col := 0; col < 2; col++ {
			got := target.Projection.F32[row*4+col]
			want := sourceCheckpoint.Projection.F32[row*2+col]
			if got != want {
				t.Fatalf("projection overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		for col := 2; col < 4; col++ {
			got := target.Projection.F32[row*4+col]
			want := baseline.Projection.F32[row*4+col]
			if got != want {
				t.Fatalf("projection non-overlap[%d,%d] = %f, want initialized baseline %f", row, col, got, want)
			}
		}
	}
	for col := 0; col < 4; col++ {
		got := target.Projection.F32[2*4+col]
		want := baseline.Projection.F32[2*4+col]
		if got != want {
			t.Fatalf("projection extra row[2,%d] = %f, want initialized baseline %f", col, got, want)
		}
	}
	for i, v := range target.TokenMoment1.F32 {
		if v != 0 {
			t.Fatalf("target token moment 1[%d] = %f, want zero", i, v)
		}
	}
	for i, v := range target.ProjMoment2.F32 {
		if v != 0 {
			t.Fatalf("target projection moment 2[%d] = %f, want zero", i, v)
		}
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapFromInferenceCopiesExactWeights(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_inference.mll")
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sourceWeights, err := ReadWeightFile(DefaultWeightFilePath(sourcePath))
	if err != nil {
		t.Fatalf("read source weights: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_inference.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}

	targetPath, targetManifest := buildTinyTrainInitArtifact(t, dir, "target_from_inference.mll")
	targetPaths, err := InitializeEmbeddingTrainerPackageWithManifest(targetPath, targetManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:                   11,
		BootstrapInferencePath: sealedPath,
		ShapeSizes:             map[string]int{"D": 2, "E": 2},
	})
	if err != nil {
		t.Fatalf("initialize target package from sealed inference: %v", err)
	}
	target, err := ReadEmbeddingTrainCheckpointFile(targetPaths.CheckpointPath)
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	if target.Step != 0 {
		t.Fatalf("target step = %d, want 0", target.Step)
	}
	assertFloat32ValuesInit(t, target.TokenEmbedding.F32, sourceWeights.Weights[sourceManifest.TokenEmbeddingParam].F32)
	assertFloat32ValuesInit(t, target.Projection.F32, sourceWeights.Weights[sourceManifest.ProjectionParam].F32)
	assertFloat32ValuesInit(t, target.TokenMoment1.F32, filledFloat32(len(target.TokenMoment1.F32), 0))
	assertFloat32ValuesInit(t, target.ProjMoment2.F32, filledFloat32(len(target.ProjMoment2.F32), 0))
}

func TestInitializeEmbeddingTrainerPackageBootstrapFromInferenceRejectsManifestMismatch(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_inference_mismatch.mll")
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_inference_mismatch.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}

	targetPath, targetManifest := buildTinyTrainInitArtifact(t, dir, "target_inference_mismatch.mll")
	targetManifest.Tokenizer.MaxSequence = sourceManifest.Tokenizer.MaxSequence + 1
	_, err := InitializeEmbeddingTrainerPackageWithManifest(targetPath, targetManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:                   11,
		BootstrapInferencePath: sealedPath,
		ShapeSizes:             map[string]int{"D": 2, "E": 2},
	})
	if err == nil {
		t.Fatal("expected sealed inference bootstrap mismatch error")
	}
	if got := err.Error(); !strings.Contains(got, "bootstrap inference artifact") || !strings.Contains(got, "tokenizer max_sequence") {
		t.Fatalf("error = %q, want bootstrap inference max_sequence mismatch", got)
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferencePreservesGraphTokenizerAndVectors(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_preserve_graph.mll")
	sourceTokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b", "c", "d"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	if err := sourceTokenizer.WriteFile(DefaultTokenizerPath(sourcePath)); err != nil {
		t.Fatalf("write source tokenizer: %v", err)
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_preserve_graph.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}

	targetPath := filepath.Join(dir, "target_preserve_graph.mll")
	paths, err := InitializeEmbeddingTrainerPackageFromSealedInference(targetPath, sealedPath, EmbeddingTrainConfig{LearningRate: 0.03}, EmbeddingTrainInitOptions{Seed: 11})
	if err != nil {
		t.Fatalf("initialize target from sealed inference: %v", err)
	}
	targetCheckpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read target checkpoint: %v", err)
	}
	if targetCheckpoint.Step != 0 {
		t.Fatalf("target step = %d, want 0", targetCheckpoint.Step)
	}
	sourceWeights, err := ReadWeightFile(DefaultWeightFilePath(sourcePath))
	if err != nil {
		t.Fatalf("read source weights: %v", err)
	}
	assertFloat32ValuesInit(t, targetCheckpoint.TokenEmbedding.F32, sourceWeights.Weights[sourceManifest.TokenEmbeddingParam].F32)
	assertFloat32ValuesInit(t, targetCheckpoint.Projection.F32, sourceWeights.Weights[sourceManifest.ProjectionParam].F32)
	if got, want := sha256FileForTest(t, paths.TokenizerPath), sha256FileForTest(t, DefaultTokenizerPath(sourcePath)); got != want {
		t.Fatalf("target tokenizer sha256 = %s, want source tokenizer %s", got, want)
	}

	rt := New(cuda.New(), metal.New())
	sourceModel, err := rt.LoadEmbeddingPackage(context.Background(), sealedPath)
	if err != nil {
		t.Fatalf("load sealed source model: %v", err)
	}
	targetModel, err := rt.LoadEmbeddingPackage(context.Background(), targetPath)
	if err != nil {
		t.Fatalf("load target trainable model as embedding package: %v", err)
	}
	for _, text := range []string{"a", "b c"} {
		sourceResult, err := sourceModel.EmbedTextWithRole(context.Background(), text, EmbeddingRoleRaw)
		if err != nil {
			t.Fatalf("embed sealed source text %q: %v", text, err)
		}
		targetResult, err := targetModel.EmbedTextWithRole(context.Background(), text, EmbeddingRoleRaw)
		if err != nil {
			t.Fatalf("embed target text %q: %v", text, err)
		}
		assertEmbeddingClose(t, targetResult.Embeddings.F32, sourceResult.Embeddings.F32, 1e-6)
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferencePreservesPackagePolicies(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyTrainInitArtifact(t, dir, "source_policy.mll")
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "E": 2},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	wantScore, wantListwise := writeResearchPolicyPackageManifestForTest(t, sourcePath)
	sealedPath := filepath.Join(dir, "source_policy.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}

	targetPath := filepath.Join(dir, "target_policy.mll")
	paths, err := InitializeEmbeddingTrainerPackageFromSealedInference(targetPath, sealedPath, EmbeddingTrainConfig{LearningRate: 0.03}, EmbeddingTrainInitOptions{Seed: 11})
	if err != nil {
		t.Fatalf("initialize target from sealed inference: %v", err)
	}
	assertPackagePoliciesForTest(t, paths, wantScore, wantListwise)
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceWidensSymbolicGraph(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyLegacyAttentionTrainInitArtifact(t, dir, "source_widen.mll")
	sourceTokenizer := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "a", "b"},
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	if err := sourceTokenizer.WriteFile(DefaultTokenizerPath(sourcePath)); err != nil {
		t.Fatalf("write source tokenizer: %v", err)
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "H": 4},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	wantScore, wantListwise := writeResearchPolicyPackageManifestForTest(t, sourcePath)
	sourceWeights, err := ReadWeightFile(DefaultWeightFilePath(sourcePath))
	if err != nil {
		t.Fatalf("read source weights: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_widen.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}
	sourceModuleJSON, err := os.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read source module: %v", err)
	}

	targetPath := filepath.Join(dir, "target_widen.mll")
	paths, err := InitializeEmbeddingTrainerPackageFromSealedInference(targetPath, sealedPath, EmbeddingTrainConfig{LearningRate: 0.03}, EmbeddingTrainInitOptions{
		Seed:                    11,
		BootstrapInferenceWiden: true,
		BootstrapTailInit:       "zero",
		ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 3, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize widened target from sealed inference: %v", err)
	}
	manifest, err := ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		t.Fatalf("read widened manifest: %v", err)
	}
	if manifest.ModelDim != 3 || manifest.OutputDim != 3 || manifest.FFNDim != 5 || manifest.HeadDim != 3 {
		t.Fatalf("widened manifest dims = model:%d output:%d ffn:%d head:%d, want 3/3/5/3", manifest.ModelDim, manifest.OutputDim, manifest.FFNDim, manifest.HeadDim)
	}
	if got, want := sha256FileForTest(t, paths.TokenizerPath), sha256FileForTest(t, DefaultTokenizerPath(sourcePath)); got != want {
		t.Fatalf("target tokenizer sha256 = %s, want source tokenizer %s", got, want)
	}
	assertPackagePoliciesForTest(t, paths, wantScore, wantListwise)
	targetModuleJSON, err := os.ReadFile(paths.ArtifactPath)
	if err != nil {
		t.Fatalf("read target module: %v", err)
	}
	if sha256.Sum256(targetModuleJSON) != sha256.Sum256(sourceModuleJSON) {
		t.Fatal("widened target changed the sealed source module graph")
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read widened checkpoint: %v", err)
	}
	if checkpoint.Step != 0 {
		t.Fatalf("widened checkpoint step = %d, want 0", checkpoint.Step)
	}
	assertFloat32SliceEqual(t, checkpoint.TokenEmbedding.Shape, []int{6, 3})
	assertFloat32SliceEqual(t, checkpoint.HiddenProjection.Shape, []int{3, 5})
	assertFloat32SliceEqual(t, checkpoint.Projection.Shape, []int{5, 3})
	for row := 0; row < 6; row++ {
		for col := 0; col < 2; col++ {
			got := checkpoint.TokenEmbedding.F32[row*3+col]
			want := sourceWeights.Weights["token_embedding"].F32[row*2+col]
			if got != want {
				t.Fatalf("token overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		if got := checkpoint.TokenEmbedding.F32[row*3+2]; got != 0 {
			t.Fatalf("token zero tail[%d,2] = %f, want 0", row, got)
		}
	}
	for row := 0; row < 4; row++ {
		for col := 0; col < 2; col++ {
			got := checkpoint.Projection.F32[row*3+col]
			want := sourceWeights.Weights["projection"].F32[row*2+col]
			if got != want {
				t.Fatalf("projection overlap[%d,%d] = %f, want %f", row, col, got, want)
			}
		}
		if got := checkpoint.Projection.F32[row*3+2]; got != 0 {
			t.Fatalf("projection zero tail[%d,2] = %f, want 0", row, got)
		}
	}
	for col := 0; col < 3; col++ {
		if got := checkpoint.Projection.F32[4*3+col]; got != 0 {
			t.Fatalf("projection extra zero row[4,%d] = %f, want 0", col, got)
		}
	}
	for i, v := range checkpoint.TokenMoment1.F32 {
		if v != 0 {
			t.Fatalf("token moment 1[%d] = %f, want zero", i, v)
		}
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceWidenRejectsConcreteOldGraphDim(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyLegacyAttentionTrainInitArtifact(t, dir, "source_widen_concrete_dim.mll")
	mod, err := eosartifact.ReadFile(sourcePath)
	if err != nil {
		t.Fatalf("read source artifact: %v", err)
	}
	for i := range mod.EntryPoints {
		for j := range mod.EntryPoints[i].Outputs {
			if mod.EntryPoints[i].Outputs[j].Type.Tensor == nil {
				continue
			}
			if len(mod.EntryPoints[i].Outputs[j].Type.Tensor.Shape) == 1 {
				mod.EntryPoints[i].Outputs[j].Type.Tensor.Shape = []string{"2"}
			} else if len(mod.EntryPoints[i].Outputs[j].Type.Tensor.Shape) == 2 {
				mod.EntryPoints[i].Outputs[j].Type.Tensor.Shape = []string{"B", "2"}
			}
		}
	}
	if err := eosartifact.WriteFile(sourcePath, mod); err != nil {
		t.Fatalf("rewrite concrete-dim source artifact: %v", err)
	}
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "H": 4},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_widen_concrete_dim.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}
	_, err = InitializeEmbeddingTrainerPackageFromSealedInference(filepath.Join(dir, "target_concrete_dim.mll"), sealedPath, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{
		BootstrapInferenceWiden: true,
		ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 3, "E": 3},
	})
	if err == nil || !strings.Contains(err.Error(), "concrete old widened dimension") {
		t.Fatalf("concrete old dim error = %v, want graph-visible concrete old dim rejection", err)
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceWidenRandomTailsStayInitialized(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyLegacyAttentionTrainInitArtifact(t, dir, "source_widen_random.mll")
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "H": 4},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_widen_random.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}
	targetPath := filepath.Join(dir, "target_widen_random.mll")
	paths, err := InitializeEmbeddingTrainerPackageFromSealedInference(targetPath, sealedPath, EmbeddingTrainConfig{LearningRate: 0.03}, EmbeddingTrainInitOptions{
		Seed:                    11,
		BootstrapInferenceWiden: true,
		BootstrapTailInit:       "random",
		ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 3, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize random-tail widened target from sealed inference: %v", err)
	}
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read random-tail widened checkpoint: %v", err)
	}
	nonZeroTail := false
	for row := 0; row < checkpoint.TokenEmbedding.Shape[0]; row++ {
		if checkpoint.TokenEmbedding.F32[row*3+2] != 0 {
			nonZeroTail = true
			break
		}
	}
	if !nonZeroTail {
		t.Fatal("random tail initialization left all token tail values at zero")
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceProjectionTailRegionsAndDeterminism(t *testing.T) {
	dir := t.TempDir()
	sealedPath, sourcePath, sourceManifest := buildTinyLegacyAttentionSealedTrainInitSource(t, dir, "source_projection_tail.mll")
	wantScore, wantListwise := writeResearchPolicyPackageManifestForTest(t, sourcePath)
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source with policy: %v", err)
	}
	sourceWeights, err := ReadWeightFile(DefaultWeightFilePath(sourcePath))
	if err != nil {
		t.Fatalf("read source weights: %v", err)
	}

	pathsA := initializeProjectionTailTargetForTest(t, filepath.Join(dir, "target_projection_tail_a.mll"), sealedPath, 19, 0.04)
	pathsB := initializeProjectionTailTargetForTest(t, filepath.Join(dir, "target_projection_tail_b.mll"), sealedPath, 19, 0.04)
	pathsC := initializeProjectionTailTargetForTest(t, filepath.Join(dir, "target_projection_tail_c.mll"), sealedPath, 23, 0.04)
	checkpointA, err := ReadEmbeddingTrainCheckpointFile(pathsA.CheckpointPath)
	if err != nil {
		t.Fatalf("read projection-tail checkpoint: %v", err)
	}
	checkpointB, err := ReadEmbeddingTrainCheckpointFile(pathsB.CheckpointPath)
	if err != nil {
		t.Fatalf("read second projection-tail checkpoint: %v", err)
	}
	checkpointC, err := ReadEmbeddingTrainCheckpointFile(pathsC.CheckpointPath)
	if err != nil {
		t.Fatalf("read different-seed projection-tail checkpoint: %v", err)
	}
	assertPackagePoliciesForTest(t, pathsA, wantScore, wantListwise)
	if checkpointA.Step != 0 {
		t.Fatalf("projection-tail checkpoint step = %d, want 0", checkpointA.Step)
	}
	assertFloat32SliceEqual(t, checkpointA.TokenEmbedding.Shape, []int{6, 3})
	assertFloat32SliceEqual(t, checkpointA.RoleEmbedding.Shape, []int{3, 3})
	assertFloat32SliceEqual(t, checkpointA.AttentionQuery.Shape, []int{3, 3})
	assertFloat32SliceEqual(t, checkpointA.HiddenProjection.Shape, []int{3, 5})
	assertFloat32SliceEqual(t, checkpointA.Projection.Shape, []int{5, 3})

	assertProjectionTailOverlapAndForbiddenZerosForTest(t, "token", checkpointA.TokenEmbedding, sourceWeights.Weights[sourceManifest.TokenEmbeddingParam], 3)
	assertProjectionTailOverlapAndForbiddenZerosForTest(t, "role", checkpointA.RoleEmbedding, sourceWeights.Weights[sourceManifest.RoleEmbeddingParam], 3)
	for _, item := range []struct {
		name   string
		target *backend.Tensor
		source *backend.Tensor
	}{
		{"attn_q", checkpointA.AttentionQuery, sourceWeights.Weights[sourceManifest.AttentionQueryParam]},
		{"attn_k", checkpointA.AttentionKey, sourceWeights.Weights[sourceManifest.AttentionKeyParam]},
		{"attn_v", checkpointA.AttentionValue, sourceWeights.Weights[sourceManifest.AttentionValueParam]},
		{"attn_o", checkpointA.AttentionOutput, sourceWeights.Weights[sourceManifest.AttentionOutputParam]},
	} {
		assertProjectionTailOverlapAndForbiddenZerosForTest(t, item.name, item.target, item.source, 3)
	}

	oldD, oldH := 2, 4
	for row := 0; row < 3; row++ {
		for col := 0; col < 5; col++ {
			got := checkpointA.HiddenProjection.F32[row*5+col]
			switch {
			case row < oldD && col < oldH:
				want := sourceWeights.Weights[sourceManifest.HiddenProjectionParam].F32[row*oldH+col]
				if got != want {
					t.Fatalf("ffn_up overlap[%d,%d] = %f, want %f", row, col, got, want)
				}
			case row < oldD && col >= oldH:
				want := deterministicProjectionTailNoise(19, sourceManifest.HiddenProjectionParam, row, col, projectionTailNoiseScale([]int{3, 5}, 0.04))
				if got != want || got == 0 {
					t.Fatalf("ffn_up projection-tail[%d,%d] = %f, want nonzero deterministic %f", row, col, got, want)
				}
			default:
				if got != 0 {
					t.Fatalf("ffn_up forbidden new-D region[%d,%d] = %f, want 0", row, col, got)
				}
			}
		}
	}
	for row := 0; row < 5; row++ {
		for col := 0; col < 3; col++ {
			got := checkpointA.Projection.F32[row*3+col]
			switch {
			case row < oldH && col < oldD:
				want := sourceWeights.Weights[sourceManifest.ProjectionParam].F32[row*oldD+col]
				if got != want {
					t.Fatalf("projection overlap[%d,%d] = %f, want %f", row, col, got, want)
				}
			case col >= oldD:
				want := deterministicProjectionTailNoise(19, sourceManifest.ProjectionParam, row, col, projectionTailNoiseScale([]int{5, 3}, 0.04))
				if got != want || got == 0 {
					t.Fatalf("projection projection-tail[%d,%d] = %f, want nonzero deterministic %f", row, col, got, want)
				}
			default:
				if got != 0 {
					t.Fatalf("projection forbidden new-H/old-D region[%d,%d] = %f, want 0", row, col, got)
				}
			}
		}
	}
	assertFloat32ValuesInit(t, checkpointA.HiddenMoment1.F32, make([]float32, len(checkpointA.HiddenMoment1.F32)))
	assertFloat32ValuesInit(t, checkpointA.ProjMoment2.F32, make([]float32, len(checkpointA.ProjMoment2.F32)))
	assertFloat32ValuesInit(t, checkpointB.HiddenProjection.F32, checkpointA.HiddenProjection.F32)
	assertFloat32ValuesInit(t, checkpointB.Projection.F32, checkpointA.Projection.F32)
	if checkpointC.HiddenProjection.F32[oldH] == checkpointA.HiddenProjection.F32[oldH] {
		t.Fatal("projection-tail hidden tail did not change across different seeds")
	}
	if checkpointC.Projection.F32[oldD] == checkpointA.Projection.F32[oldD] {
		t.Fatal("projection-tail projection tail did not change across different seeds")
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceProjectionTailDefaultScaleAndValidation(t *testing.T) {
	dir := t.TempDir()
	sealedPath, _, sourceManifest := buildTinyLegacyAttentionSealedTrainInitSource(t, dir, "source_projection_tail_scale.mll")
	paths := initializeProjectionTailTargetForTest(t, filepath.Join(dir, "target_projection_tail_default_scale.mll"), sealedPath, 31, 0)
	checkpoint, err := ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		t.Fatalf("read default-scale projection-tail checkpoint: %v", err)
	}
	want := deterministicProjectionTailNoise(31, sourceManifest.HiddenProjectionParam, 0, 4, projectionTailNoiseScale([]int{3, 5}, 0))
	if got := checkpoint.HiddenProjection.F32[4]; got != want || got == 0 {
		t.Fatalf("default-scale ffn_up tail[0,4] = %f, want nonzero deterministic %f", got, want)
	}

	for _, tc := range []struct {
		name  string
		mode  string
		scale float64
		want  string
	}{
		{"scale with zero", "zero", 0.03, "bootstrap tail scale requires bootstrap tail init projection-tail"},
		{"scale with random", "random", 0.03, "bootstrap tail scale requires bootstrap tail init projection-tail"},
		{"negative projection scale", "projection-tail", -0.01, "bootstrap tail scale must be finite"},
		{"too large projection scale", "projection-tail", 0.11, "bootstrap tail scale must be finite"},
		{"nan projection scale", "projection-tail", math.NaN(), "bootstrap tail scale must be finite"},
		{"inf projection scale", "projection-tail", math.Inf(1), "bootstrap tail scale must be finite"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := InitializeEmbeddingTrainerPackageFromSealedInference(filepath.Join(dir, tc.name+".mll"), sealedPath, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{
				Seed:                    7,
				BootstrapInferenceWiden: true,
				BootstrapTailInit:       tc.mode,
				BootstrapTailScale:      tc.scale,
				ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 3, "E": 3},
			})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	if err := validateProjectionTailBootstrapContract(map[string]*backend.Tensor{}, map[string]*backend.Tensor{}, EmbeddingManifest{}); err == nil || !strings.Contains(err.Error(), "recognized hidden projection") {
		t.Fatalf("missing projection-tail contract error = %v, want fail-closed recognized-param rejection", err)
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceExactRejectsTailFlags(t *testing.T) {
	dir := t.TempDir()
	sealedPath, _, _ := buildTinyLegacyAttentionSealedTrainInitSource(t, dir, "source_exact_tail_flags.mll")
	for _, tc := range []struct {
		name string
		opts EmbeddingTrainInitOptions
		want string
	}{
		{"tail init", EmbeddingTrainInitOptions{BootstrapTailInit: "projection-tail"}, "bootstrap tail init requires bootstrap inference widen"},
		{"tail scale", EmbeddingTrainInitOptions{BootstrapTailScale: 0.03}, "bootstrap tail scale requires bootstrap inference widen"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := InitializeEmbeddingTrainerPackageFromSealedInference(filepath.Join(dir, tc.name+".mll"), sealedPath, EmbeddingTrainConfig{}, tc.opts)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestInitializeEmbeddingTrainerPackageFromSealedInferenceWidenRejectsShrinkAndUntiedOutput(t *testing.T) {
	dir := t.TempDir()
	sourcePath, sourceManifest := buildTinyLegacyAttentionTrainInitArtifact(t, dir, "source_widen_reject.mll")
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "H": 4},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, "source_widen_reject.sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}
	_, err := InitializeEmbeddingTrainerPackageFromSealedInference(filepath.Join(dir, "target_shrink.mll"), sealedPath, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{
		BootstrapInferenceWiden: true,
		ShapeSizes:              map[string]int{"D": 1, "H": 5, "O": 1, "E": 1},
	})
	if err == nil || !strings.Contains(err.Error(), "model_dim growth") {
		t.Fatalf("shrink error = %v, want model_dim growth rejection", err)
	}
	_, err = InitializeEmbeddingTrainerPackageFromSealedInference(filepath.Join(dir, "target_untied_output.mll"), sealedPath, EmbeddingTrainConfig{}, EmbeddingTrainInitOptions{
		BootstrapInferenceWiden: true,
		ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 2, "E": 2},
	})
	if err == nil || !strings.Contains(err.Error(), "match model_dim") {
		t.Fatalf("untied output error = %v, want output/model tie rejection", err)
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapCopiesGenericExactName(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := writeSyntheticBootstrapCheckpoint(t, dir, map[string]*backend.Tensor{
		"layers.0.attn_q": backend.NewTensorF32([]int{2, 2}, []float32{
			31, 32,
			33, 34,
		}),
	}, nil)
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -1)),
		"projection":      backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -2)),
		"layers.0.attn_q": backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 0)),
	}
	if err := bootstrapTrainingWeights(weights, nil, tinyBootstrapManifest(), EmbeddingTrainInitOptions{BootstrapCheckpointPath: checkpointPath}); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	assertFloat32ValuesInit(t, weights["layers.0.attn_q"].F32, []float32{31, 32, 33, 34})
}

func TestInitializeEmbeddingTrainerPackageBootstrapCopiesGenericOverlapAndIgnoresMoments(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := writeSyntheticBootstrapCheckpoint(t, dir, map[string]*backend.Tensor{
		"layers.0.attn_q": backend.NewTensorF32([]int{2, 2}, []float32{
			41, 42,
			43, 44,
		}),
	}, map[string]*backend.Tensor{
		"layers.0.attn_q_moment_1": backend.NewTensorF32([]int{3, 3}, filledFloat32(9, 99)),
	})
	weights := map[string]*backend.Tensor{
		"token_embedding":          backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -1)),
		"projection":               backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -2)),
		"layers.0.attn_q":          backend.NewTensorF32([]int{3, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}),
		"layers.0.attn_q_moment_1": backend.NewTensorF32([]int{3, 3}, filledFloat32(9, 7)),
	}
	if err := bootstrapTrainingWeights(weights, nil, tinyBootstrapManifest(), EmbeddingTrainInitOptions{BootstrapCheckpointPath: checkpointPath}); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	assertFloat32ValuesInit(t, weights["layers.0.attn_q"].F32, []float32{
		41, 42, 3,
		43, 44, 6,
		7, 8, 9,
	})
	assertFloat32ValuesInit(t, weights["layers.0.attn_q_moment_1"].F32, filledFloat32(9, 7))
}

func TestInitializeEmbeddingTrainerPackageBootstrapRejectsRankMismatch(t *testing.T) {
	target := backend.NewTensorF32([]int{2, 2}, []float32{1, 2, 3, 4})
	source := backend.NewTensorF32([]int{2, 1, 2}, []float32{1, 2, 3, 4})
	if err := copyOverlappingTensor(target, source); err == nil {
		t.Fatal("expected rank mismatch error")
	}
}

func TestInitializeEmbeddingTrainerPackageBootstrapRejectsGenericRankMismatch(t *testing.T) {
	dir := t.TempDir()
	checkpointPath := writeSyntheticBootstrapCheckpoint(t, dir, map[string]*backend.Tensor{
		"layers.0.attn_q": backend.NewTensorF32([]int{2, 1, 2}, []float32{1, 2, 3, 4}),
	}, nil)
	weights := map[string]*backend.Tensor{
		"token_embedding": backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -1)),
		"projection":      backend.NewTensorF32([]int{2, 2}, filledFloat32(4, -2)),
		"layers.0.attn_q": backend.NewTensorF32([]int{2, 2}, filledFloat32(4, 0)),
	}
	err := bootstrapTrainingWeights(weights, nil, tinyBootstrapManifest(), EmbeddingTrainInitOptions{BootstrapCheckpointPath: checkpointPath})
	if err == nil {
		t.Fatal("expected generic rank mismatch error")
	}
	if got := err.Error(); !strings.Contains(got, `bootstrap generic tensor "layers.0.attn_q"`) || !strings.Contains(got, "rank mismatch") {
		t.Fatalf("error = %q, want generic tensor name and rank mismatch", got)
	}
}

func buildTinyLegacyAttentionTrainInitArtifact(t *testing.T, dir, name string) (string, EmbeddingManifest) {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param role_embedding: q8[3, D] @weight("weights/role_embedding") @trainable
param attn_q: q8[D, D] @weight("weights/attn_q") @trainable
param attn_k: q8[D, D] @weight("weights/attn_k") @trainable
param attn_v: q8[D, D] @weight("weights/attn_v") @trainable
param attn_o: q8[D, D] @weight("weights/attn_o") @trainable
param ffn_up: q8[D, H] @weight("weights/ffn_up") @trainable
param projection: q8[H, D] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T], attention_mask: i32[T], role_ids: i32[T]) -> f16[D] {
    let token_q = gather(token_embedding, tokens)
    let role_q = gather(role_embedding, role_ids)
    let token_f = dequant(token_q)
    let role_f = dequant(role_q)
    let hidden = rope(token_f + role_f)
    let wq = dequant(attn_q)
    let wk = dequant(attn_k)
    let wv = dequant(attn_v)
    let wo = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq)
    let k = @matmul(hidden, wk)
    let v = @matmul(hidden, wv)
    let kt = transpose(k)
    let scores = @scaled_attention_scores(q, kt)
    let probs = masked_softmax(scores, attention_mask)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo)
    let attended_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attended_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attended_hidden)
    return mean_pool(normalize(encoded), attention_mask)
}

pipeline embed_pooled_batch(tokens: i32[B, T], attention_mask: i32[B, T], role_ids: i32[B, T]) -> f16[B, D] {
    let token_q = gather(token_embedding, tokens)
    let role_q = gather(role_embedding, role_ids)
    let token_f = dequant(token_q)
    let role_f = dequant(role_q)
    let hidden = rope(token_f + role_f)
    let wq = dequant(attn_q)
    let wk = dequant(attn_k)
    let wv = dequant(attn_v)
    let wo = dequant(attn_o)
    let ffn_up_f = dequant(ffn_up)
    let projection_f = dequant(projection)
    let q = @matmul(hidden, wq)
    let k = @matmul(hidden, wk)
    let v = @matmul(hidden, wv)
    let kt = transpose(k)
    let scores = @scaled_attention_scores(q, kt)
    let probs = masked_softmax(scores, attention_mask)
    let mixed = @matmul(probs, v)
    let attended = @matmul(mixed, wo)
    let attended_hidden = layernorm(attended + hidden)
    let ffn_hidden = @matmul(attended_hidden, ffn_up_f)
    let activated = gelu(ffn_hidden)
    let projected = @matmul(activated, projection_f)
    let encoded = layernorm(projected + attended_hidden)
    return mean_pool(normalize(encoded), attention_mask)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_legacy_attention_embed"})
	if err != nil {
		t.Fatalf("build legacy attention source: %v", err)
	}
	path := filepath.Join(dir, name)
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	return path, EmbeddingManifest{
		Name:                  "tiny-legacy-attention",
		PooledEntry:           "embed_pooled",
		BatchEntry:            "embed_pooled_batch",
		EncoderRepeats:        1,
		TokenInput:            "tokens",
		MaskInput:             "attention_mask",
		OutputName:            "result",
		OutputDType:           "f16",
		ArchitectureVersion:   EmbeddingArchitectureLegacyV1,
		ModelDim:              2,
		OutputDim:             2,
		AttentionHeads:        1,
		HeadDim:               2,
		FFNDim:                4,
		ParameterTying:        EmbeddingParameterTyingLegacyTied,
		TokenEmbeddingParam:   "token_embedding",
		RoleConditioning:      EmbeddingRoleConditioningAdditiveV1,
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
		AttentionMaskMode:     EmbeddingAttentionMaskModeKey,
		AttentionScoreScale:   EmbeddingAttentionScoreScaleKeyDimRSQ,
		AttentionResidual:     true,
		AttentionLayerNorm:    true,
		PositionEncoding:      EmbeddingPositionEncodingRoPE,
		HiddenProjectionParam: "ffn_up",
		FFNResidual:           true,
		FFNLayerNorm:          true,
		ProjectionParam:       "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   6,
			MaxSequence: 6,
			PadID:       0,
			BOSID:       1,
			EOSID:       2,
			UnknownID:   3,
		},
	}
}

func writeResearchPolicyPackageManifestForTest(t *testing.T, artifactPath string) (EmbeddingScoreSpectrumPolicy, EmbeddingListwiseGeometryPolicy) {
	t.Helper()
	packageManifest, err := ReadPackageManifestFile(DefaultPackageManifestPath(artifactPath))
	if err != nil {
		t.Fatalf("read source package manifest: %v", err)
	}
	score := EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:          true,
		ScoreSpectrumResearchOnly:   true,
		TrainAllowedForResearch:     true,
		ReleaseTrainAllowed:         false,
		CommercialUseAllowed:        false,
		SourceArtifactHashes:        []string{"score-a", "score-b"},
		ScoreSpectrumRowCount:       7,
		AutoClearedObjectives:       []string{"matryoshka"},
		IsolatedInheritedObjectives: []string{"turboquant_prefix_seed"},
	}
	listwise := EmbeddingListwiseGeometryPolicy{
		ListwiseGeometryTrain:        true,
		ListwiseGeometryResearchOnly: true,
		TrainAllowedForResearch:      true,
		ReleaseTrainAllowed:          false,
		CommercialUseAllowed:         false,
		SourceArtifactHashes:         []string{"listwise-a"},
		ListwiseGeometryBatchCount:   5,
		AutoClearedObjectives:        []string{"matryoshka"},
		IsolatedInheritedObjectives:  []string{"turboquant_prefix_seed"},
	}
	packageManifest.ScoreSpectrum = score
	packageManifest.ListwiseGeometry = listwise
	if err := packageManifest.WriteFile(DefaultPackageManifestPath(artifactPath)); err != nil {
		t.Fatalf("write source package manifest: %v", err)
	}
	return score, listwise
}

func assertPackagePoliciesForTest(t *testing.T, paths EmbeddingTrainPackagePaths, wantScore EmbeddingScoreSpectrumPolicy, wantListwise EmbeddingListwiseGeometryPolicy) {
	t.Helper()
	packageManifest, err := ReadPackageManifestFile(paths.PackageManifestPath)
	if err != nil {
		t.Fatalf("read target package manifest: %v", err)
	}
	trainManifest, err := ReadEmbeddingTrainManifestFile(paths.TrainManifestPath)
	if err != nil {
		t.Fatalf("read target train manifest: %v", err)
	}
	for label, got := range map[string]EmbeddingScoreSpectrumPolicy{
		"package": packageManifest.ScoreSpectrum,
		"train":   trainManifest.ScoreSpectrum,
	} {
		if !got.ScoreSpectrumResearchOnly || !got.TrainAllowedForResearch || got.ReleaseTrainAllowed || got.CommercialUseAllowed {
			t.Fatalf("%s score-spectrum policy gates = %+v, want research-only non-commercial", label, got)
		}
		if got.ScoreSpectrumRowCount != wantScore.ScoreSpectrumRowCount || strings.Join(got.SourceArtifactHashes, ",") != strings.Join(wantScore.SourceArtifactHashes, ",") {
			t.Fatalf("%s score-spectrum lineage = %+v, want %+v", label, got, wantScore)
		}
	}
	for label, got := range map[string]EmbeddingListwiseGeometryPolicy{
		"package": packageManifest.ListwiseGeometry,
		"train":   trainManifest.ListwiseGeometry,
	} {
		if !got.ListwiseGeometryResearchOnly || !got.TrainAllowedForResearch || got.ReleaseTrainAllowed || got.CommercialUseAllowed {
			t.Fatalf("%s listwise policy gates = %+v, want research-only non-commercial", label, got)
		}
		if got.ListwiseGeometryBatchCount != wantListwise.ListwiseGeometryBatchCount || strings.Join(got.SourceArtifactHashes, ",") != strings.Join(wantListwise.SourceArtifactHashes, ",") {
			t.Fatalf("%s listwise lineage = %+v, want %+v", label, got, wantListwise)
		}
	}
}

func buildTinyLegacyAttentionSealedTrainInitSource(t *testing.T, dir, name string) (string, string, EmbeddingManifest) {
	t.Helper()
	sourcePath, sourceManifest := buildTinyLegacyAttentionTrainInitArtifact(t, dir, name)
	if _, err := InitializeEmbeddingTrainerPackageWithManifest(sourcePath, sourceManifest, EmbeddingTrainConfig{LearningRate: 0.02}, EmbeddingTrainInitOptions{
		Seed:       3,
		ShapeSizes: map[string]int{"D": 2, "H": 4},
	}); err != nil {
		t.Fatalf("initialize source package: %v", err)
	}
	sealedPath := filepath.Join(dir, strings.TrimSuffix(name, filepath.Ext(name))+".sealed.mll")
	if _, err := ExportPackageToMLL(sourcePath, sealedPath); err != nil {
		t.Fatalf("export sealed source: %v", err)
	}
	return sealedPath, sourcePath, sourceManifest
}

func initializeProjectionTailTargetForTest(t *testing.T, path, sealedPath string, seed int64, scale float64) EmbeddingTrainPackagePaths {
	t.Helper()
	paths, err := InitializeEmbeddingTrainerPackageFromSealedInference(path, sealedPath, EmbeddingTrainConfig{LearningRate: 0.03}, EmbeddingTrainInitOptions{
		Seed:                    seed,
		BootstrapInferenceWiden: true,
		BootstrapTailInit:       "projection-tail",
		BootstrapTailScale:      scale,
		ShapeSizes:              map[string]int{"D": 3, "H": 5, "O": 3, "E": 3},
	})
	if err != nil {
		t.Fatalf("initialize projection-tail target: %v", err)
	}
	return paths
}

func assertProjectionTailOverlapAndForbiddenZerosForTest(t *testing.T, name string, target, source *backend.Tensor, targetCols int) {
	t.Helper()
	if target == nil || source == nil {
		t.Fatalf("%s tensor missing: target=%v source=%v", name, target, source)
	}
	sourceRows, sourceCols := source.Shape[0], source.Shape[1]
	for row := 0; row < target.Shape[0]; row++ {
		for col := 0; col < target.Shape[1]; col++ {
			got := target.F32[row*targetCols+col]
			if row < sourceRows && col < sourceCols {
				want := source.F32[row*sourceCols+col]
				if got != want {
					t.Fatalf("%s overlap[%d,%d] = %f, want %f", name, row, col, got, want)
				}
			} else if got != 0 {
				t.Fatalf("%s forbidden tail[%d,%d] = %f, want 0", name, row, col, got)
			}
		}
	}
}

func buildTinyTrainInitArtifact(t *testing.T, dir, name string) (string, EmbeddingManifest) {
	t.Helper()
	source := []byte(`
param token_embedding: q8[V, D] @weight("weights/token_embedding") @trainable
param projection: q8[D, E] @weight("weights/projection") @trainable

pipeline embed_pooled(tokens: i32[T]) -> f16[E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
    return mean_pool(projected)
}

pipeline embed_pooled_batch(tokens: i32[B, T]) -> f16[B, E] {
    let embeddings_q = gather(token_embedding, tokens)
    let embeddings = dequant(embeddings_q)
    let projection_f = dequant(projection)
    let projected = @matmul(embeddings, projection_f)
    return mean_pool(projected)
}
`)
	bundle, err := compiler.Build(source, compiler.Options{ModuleName: "tiny_train_embed"})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	path := filepath.Join(dir, name)
	if err := eosartifact.WriteFile(path, bundle.Artifact); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	manifest := EmbeddingManifest{
		Name:                "tiny-train-embed",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "f16",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   8,
			MaxSequence: 8,
			PadID:       0,
		},
	}
	return path, manifest
}

func filledFloat32(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func writeSyntheticBootstrapCheckpoint(t *testing.T, dir string, tensors, moments map[string]*backend.Tensor) string {
	t.Helper()
	path := filepath.Join(dir, "source.embed-train.mll")
	checkpoint := EmbeddingTrainCheckpoint{
		Version:        EmbeddingTrainCheckpointVersion,
		Manifest:       tinyBootstrapManifest(),
		Config:         EmbeddingTrainConfig{LearningRate: 0.01},
		TokenEmbedding: backend.NewTensorF32([]int{2, 2}, []float32{11, 12, 13, 14}),
		Projection:     backend.NewTensorF32([]int{2, 2}, []float32{21, 22, 23, 24}),
		Tensors:        tensors,
		MomentTensors:  moments,
	}
	if err := checkpoint.WriteFile(path); err != nil {
		t.Fatalf("write synthetic checkpoint: %v", err)
	}
	return path
}

func tinyBootstrapManifest() EmbeddingManifest {
	return EmbeddingManifest{
		Name:                "tiny-bootstrap",
		PooledEntry:         "embed_pooled",
		BatchEntry:          "embed_pooled_batch",
		TokenInput:          "tokens",
		OutputName:          "result",
		OutputDType:         "q8",
		TokenEmbeddingParam: "token_embedding",
		ProjectionParam:     "projection",
		Tokenizer: TokenizerManifest{
			VocabSize:   2,
			MaxSequence: 2,
			PadID:       0,
		},
	}
}

func assertFloat32SliceEqual(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("shape rank = %d, want %d: got %v want %v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("shape = %v, want %v", got, want)
		}
	}
}

func assertFloat32ValuesInit(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("value count = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func assertEmbeddingClose(t *testing.T, got, want []float32, tolerance float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("embedding length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > tolerance {
			t.Fatalf("embedding[%d] = %f, want %f (diff %f > %f)", i, got[i], want[i], diff, tolerance)
		}
	}
}

func sha256FileForTest(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return fmt.Sprintf("%x", sha256.Sum256(data))
}
