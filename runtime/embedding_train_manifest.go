package eosruntime

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

const EmbeddingTrainManifestVersion = "manta/train-manifest/v0alpha1"

// EmbeddingTrainManifest describes the native training contract for an embedding module.
type EmbeddingTrainManifest struct {
	Name             string                          `json:"name,omitempty"`
	Embedding        EmbeddingManifest               `json:"embedding"`
	Config           EmbeddingTrainConfig            `json:"config"`
	ScoreSpectrum    EmbeddingScoreSpectrumPolicy    `json:"score_spectrum,omitempty"`
	ListwiseGeometry EmbeddingListwiseGeometryPolicy `json:"listwise_geometry,omitempty"`
}

// EmbeddingScoreSpectrumPolicy records train-time provenance and usage gates
// for score-spectrum data carried through training and package manifests.
type EmbeddingScoreSpectrumPolicy struct {
	ScoreSpectrumTrain          bool     `json:"score_spectrum_train,omitempty"`
	ScoreSpectrumResearchOnly   bool     `json:"score_spectrum_research_only,omitempty"`
	TrainAllowedForResearch     bool     `json:"train_allowed_for_research,omitempty"`
	ReleaseTrainAllowed         bool     `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed        bool     `json:"commercial_use_allowed,omitempty"`
	SourceArtifactHashes        []string `json:"source_artifact_hashes,omitempty"`
	ScoreSpectrumRowCount       int      `json:"score_spectrum_row_count,omitempty"`
	AutoClearedObjectives       []string `json:"auto_cleared_objectives,omitempty"`
	IsolatedInheritedObjectives []string `json:"isolated_inherited_objectives,omitempty"`
}

// EmbeddingListwiseGeometryPolicy records train-time provenance and usage
// gates for listwise geometry data carried through training and package manifests.
type EmbeddingListwiseGeometryPolicy struct {
	ListwiseGeometryTrain        bool     `json:"listwise_geometry_train,omitempty"`
	ListwiseGeometryResearchOnly bool     `json:"listwise_geometry_research_only,omitempty"`
	TrainAllowedForResearch      bool     `json:"train_allowed_for_research,omitempty"`
	ReleaseTrainAllowed          bool     `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed         bool     `json:"commercial_use_allowed,omitempty"`
	SourceArtifactHashes         []string `json:"source_artifact_hashes,omitempty"`
	ListwiseGeometryBatchCount   int      `json:"listwise_geometry_batch_count,omitempty"`
	AutoClearedObjectives        []string `json:"auto_cleared_objectives,omitempty"`
	IsolatedInheritedObjectives  []string `json:"isolated_inherited_objectives,omitempty"`
}

// DefaultEmbeddingTrainManifestPath returns the conventional sibling train-manifest path for an .mll artifact.
func DefaultEmbeddingTrainManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".train.mll")
}

func ResolveEmbeddingTrainManifestPath(artifactPath string) string {
	return DefaultEmbeddingTrainManifestPath(artifactPath)
}

// ReadEmbeddingTrainManifestFile decodes an authored MLL training manifest.
func ReadEmbeddingTrainManifestFile(path string) (EmbeddingTrainManifest, error) {
	doc, err := readAuthoredManifestMLL(path, "train_manifest", EmbeddingTrainManifestVersion)
	if err != nil {
		return EmbeddingTrainManifest{}, err
	}
	return embeddingTrainManifestFromDoc(doc)
}

// WriteFile writes the training manifest as an authored MLL container.
func (m EmbeddingTrainManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "train_manifest", EmbeddingTrainManifestVersion, m.nameOrDefault(), "Eos training manifest", m.mllValues())
}

func (m EmbeddingTrainManifest) normalized() EmbeddingTrainManifest {
	m.Embedding = m.Embedding.normalized()
	return m
}

// ValidateModule checks that a module satisfies the training contract.
func (m EmbeddingTrainManifest) ValidateModule(mod *eosartifact.Module) error {
	m = m.normalized()
	m.Embedding = m.Embedding.normalizedForModule(mod)
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if err := m.Embedding.ValidateModule(mod); err != nil {
		return err
	}
	if err := m.Embedding.ValidateLegacyEmbeddingTrainerSupported(); err != nil {
		return err
	}
	if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.TokenEmbeddingParam); err != nil {
		return err
	}
	if err := validateTrainableAttentionParams(mod, m.Embedding); err != nil {
		return err
	}
	if m.Embedding.HiddenProjectionParam != "" {
		if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.HiddenProjectionParam); err != nil {
			return err
		}
	}
	if _, err := requireTrainableEmbeddingParam(mod, m.Embedding.ProjectionParam); err != nil {
		return err
	}
	return nil
}

func validateTrainableAttentionParams(mod *eosartifact.Module, manifest EmbeddingManifest) error {
	names := []string{
		manifest.AttentionQueryParam,
		manifest.AttentionKeyParam,
		manifest.AttentionValueParam,
		manifest.AttentionOutputParam,
	}
	set := 0
	for _, name := range names {
		if name != "" {
			set++
		}
	}
	if set == 0 {
		return nil
	}
	if set != len(names) {
		return fmt.Errorf("attention params must declare query, key, value, and output together")
	}
	for _, name := range names {
		if _, err := requireTrainableEmbeddingParam(mod, name); err != nil {
			return err
		}
	}
	return nil
}

func (m EmbeddingTrainManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	if m.Embedding.Name != "" {
		return m.Embedding.Name
	}
	return "train_manifest"
}

func (m EmbeddingTrainManifest) mllValues() map[string]authoredManifestValue {
	values := map[string]authoredManifestValue{
		"name":                                              authoredString(m.Name),
		"config.optimizer":                                  authoredString(m.Config.Optimizer),
		"config.weight_bits":                                authoredInt(int64(m.Config.WeightBits)),
		"config.learning_rate":                              authoredFloat(float64(m.Config.LearningRate)),
		"config.weight_decay":                               authoredFloat(float64(m.Config.WeightDecay)),
		"config.beta1":                                      authoredFloat(float64(m.Config.Beta1)),
		"config.beta2":                                      authoredFloat(float64(m.Config.Beta2)),
		"config.epsilon":                                    authoredFloat(float64(m.Config.Epsilon)),
		"config.contrastive_loss":                           authoredString(m.Config.ContrastiveLoss),
		"config.temperature":                                authoredFloat(float64(m.Config.Temperature)),
		"config.grouped_loss_weight":                        authoredFloat(float64(m.Config.GroupedLossWeight)),
		"config.teacher_loss_weight":                        authoredFloat(float64(m.Config.TeacherLossWeight)),
		"config.teacher_temperature":                        authoredFloat(float64(m.Config.TeacherTemperature)),
		"config.matryoshka_dims":                            authoredString(formatMatryoshkaDims(m.Config.MatryoshkaDims)),
		"config.matryoshka_weights":                         authoredString(formatMatryoshkaWeights(m.Config.MatryoshkaWeights)),
		"config.turboquant_prefix_bits":                     authoredString(formatIntList(m.Config.TurboQuantPrefixBits)),
		"config.turboquant_prefix_objectives":               authoredString(FormatTurboQuantPrefixObjectives(m.Config.TurboQuantPrefixObjectives)),
		"config.turboquant_prefix_weight":                   authoredFloat(float64(m.Config.TurboQuantPrefixWeight)),
		"config.turboquant_prefix_seed":                     authoredInt(m.Config.TurboQuantPrefixSeed),
		"config.turboquant_prefix_score_mode":               authoredString(m.Config.TurboQuantPrefixScoreMode),
		"config.turboquant_compact_objectives":              authoredString(FormatTurboQuantPrefixObjectives(m.Config.TurboQuantCompactObjectives)),
		"config.turboquant_rank_margin_objectives":          authoredString(FormatTurboQuantPrefixObjectives(m.Config.TurboQuantRankMarginObjectives)),
		"config.turboquant_rank_margin":                     authoredFloat(float64(m.Config.TurboQuantRankMargin)),
		"config.turboquant_rank_margin_loss":                authoredString(m.Config.TurboQuantRankMarginLoss),
		"config.turboquant_rank_margin_reduction":           authoredString(m.Config.TurboQuantRankMarginReduction),
		"config.turboquant_rank_margin_tau":                 authoredFloat(float64(m.Config.TurboQuantRankMarginTau)),
		"config.turboquant_topk_objectives":                 authoredString(FormatTurboQuantPrefixObjectives(m.Config.TurboQuantTopKObjectives)),
		"config.turboquant_topk_loss":                       authoredString(m.Config.TurboQuantTopKLoss),
		"config.turboquant_topk_cutoff":                     authoredInt(int64(m.Config.TurboQuantTopKCutoff)),
		"config.turboquant_topk_tau":                        authoredFloat(float64(m.Config.TurboQuantTopKTau)),
		"config.turboquant_topk_margin":                     authoredFloat(float64(m.Config.TurboQuantTopKMargin)),
		"config.turboquant_topk_negative_mask":              authoredString(m.Config.TurboQuantTopKNegativeMask),
		"config.turboquant_topk_recall_weight":              authoredFloat(float64(m.Config.TurboQuantTopKRecallWeight)),
		"config.turboquant_topk_recall_cutoff":              authoredInt(int64(m.Config.TurboQuantTopKRecallCutoff)),
		"config.turboquant_topk_recall_tau":                 authoredFloat(float64(m.Config.TurboQuantTopKRecallTau)),
		"config.turboquant_topk_recall_margin":              authoredFloat(float64(m.Config.TurboQuantTopKRecallMargin)),
		"config.turboquant_topk_recall_negative_mask":       authoredString(m.Config.TurboQuantTopKRecallNegativeMask),
		"config.score_spectrum_loss_mode":                   authoredString(m.Config.ScoreSpectrumLossMode),
		"config.score_spectrum_recovery_weight":             authoredFloat(float64(m.Config.ScoreSpectrumRecoveryWeight)),
		"config.score_spectrum_recovery_margin":             authoredFloat(float64(m.Config.ScoreSpectrumRecoveryMargin)),
		"config.score_spectrum_recovery_top_k":              authoredInt(int64(m.Config.ScoreSpectrumRecoveryTopK)),
		"config.score_spectrum_recovery_tau":                authoredFloat(float64(m.Config.ScoreSpectrumRecoveryTau)),
		"config.score_spectrum_activation_microbatch_size":  authoredInt(int64(m.Config.ScoreSpectrumActivationMicrobatchSize)),
		"score_spectrum.score_spectrum_train":               authoredBool(m.ScoreSpectrum.ScoreSpectrumTrain),
		"score_spectrum.score_spectrum_research_only":       authoredBool(m.ScoreSpectrum.ScoreSpectrumResearchOnly),
		"score_spectrum.train_allowed_for_research":         authoredBool(m.ScoreSpectrum.TrainAllowedForResearch),
		"score_spectrum.release_train_allowed":              authoredBool(m.ScoreSpectrum.ReleaseTrainAllowed),
		"score_spectrum.commercial_use_allowed":             authoredBool(m.ScoreSpectrum.CommercialUseAllowed),
		"score_spectrum.source_artifact_hashes":             authoredString(formatScoreSpectrumSourceHashes(m.ScoreSpectrum.SourceArtifactHashes)),
		"score_spectrum.score_spectrum_row_count":           authoredInt(int64(m.ScoreSpectrum.ScoreSpectrumRowCount)),
		"score_spectrum.auto_cleared_objectives":            authoredString(formatScoreSpectrumObjectiveNames(m.ScoreSpectrum.AutoClearedObjectives)),
		"score_spectrum.isolated_inherited_objectives":      authoredString(formatScoreSpectrumObjectiveNames(m.ScoreSpectrum.IsolatedInheritedObjectives)),
		"listwise_geometry.listwise_geometry_train":         authoredBool(m.ListwiseGeometry.ListwiseGeometryTrain),
		"listwise_geometry.listwise_geometry_research_only": authoredBool(m.ListwiseGeometry.ListwiseGeometryResearchOnly),
		"listwise_geometry.train_allowed_for_research":      authoredBool(m.ListwiseGeometry.TrainAllowedForResearch),
		"listwise_geometry.release_train_allowed":           authoredBool(m.ListwiseGeometry.ReleaseTrainAllowed),
		"listwise_geometry.commercial_use_allowed":          authoredBool(m.ListwiseGeometry.CommercialUseAllowed),
		"listwise_geometry.source_artifact_hashes":          authoredString(formatScoreSpectrumSourceHashes(m.ListwiseGeometry.SourceArtifactHashes)),
		"listwise_geometry.listwise_geometry_batch_count":   authoredInt(int64(m.ListwiseGeometry.ListwiseGeometryBatchCount)),
		"listwise_geometry.auto_cleared_objectives":         authoredString(formatScoreSpectrumObjectiveNames(m.ListwiseGeometry.AutoClearedObjectives)),
		"listwise_geometry.isolated_inherited_objectives":   authoredString(formatScoreSpectrumObjectiveNames(m.ListwiseGeometry.IsolatedInheritedObjectives)),
	}
	for key, value := range m.Embedding.mllValues() {
		values["embedding."+key] = value
	}
	return values
}

func embeddingTrainManifestFromDoc(doc authoredManifestDoc) (EmbeddingTrainManifest, error) {
	var manifest EmbeddingTrainManifest
	var err error
	if manifest.Name, _, err = doc.string("name"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	embeddingDoc := authoredManifestDoc{values: map[string]authoredManifestValue{}}
	for key, value := range doc.values {
		if len(key) > len("embedding.") && key[:len("embedding.")] == "embedding." {
			embeddingDoc.values[key[len("embedding."):]] = value
		}
	}
	manifest.Embedding, err = embeddingManifestFromDoc(embeddingDoc)
	if err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if manifest.Config.Optimizer, _, err = doc.string("config.optimizer"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if value, _, err := doc.int("config.weight_bits"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.WeightBits = int(value)
	}
	if value, _, err := doc.float("config.learning_rate"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.LearningRate = float32(value)
	}
	if value, _, err := doc.float("config.weight_decay"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.WeightDecay = float32(value)
	}
	if value, _, err := doc.float("config.beta1"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Beta1 = float32(value)
	}
	if value, _, err := doc.float("config.beta2"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Beta2 = float32(value)
	}
	if value, _, err := doc.float("config.epsilon"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Epsilon = float32(value)
	}
	if manifest.Config.ContrastiveLoss, _, err = doc.string("config.contrastive_loss"); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if value, _, err := doc.float("config.temperature"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.Temperature = float32(value)
	}
	if value, _, err := doc.float("config.grouped_loss_weight"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else {
		manifest.Config.GroupedLossWeight = float32(value)
	}
	if value, ok, err := doc.float("config.teacher_loss_weight"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TeacherLossWeight = float32(value)
	}
	if value, ok, err := doc.float("config.teacher_temperature"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TeacherTemperature = float32(value)
	}
	if value, ok, err := doc.string("config.matryoshka_dims"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		dims, err := parseMatryoshkaDims(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.MatryoshkaDims = dims
	}
	if value, ok, err := doc.string("config.matryoshka_weights"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		weights, err := parseMatryoshkaWeights(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.MatryoshkaWeights = weights
	}
	if value, ok, err := doc.string("config.turboquant_prefix_bits"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		bits, err := parseMatryoshkaDims(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantPrefixBits = bits
	}
	if value, ok, err := doc.string("config.turboquant_prefix_objectives"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		objectives, err := ParseTurboQuantPrefixObjectives(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantPrefixObjectives = objectives
	}
	if value, ok, err := doc.float("config.turboquant_prefix_weight"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantPrefixWeight = float32(value)
	}
	if value, ok, err := doc.int("config.turboquant_prefix_seed"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantPrefixSeed = value
	}
	if value, ok, err := doc.string("config.turboquant_prefix_score_mode"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		mode, err := normalizeTurboQuantPrefixScoreMode(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantPrefixScoreMode = mode
	}
	if value, ok, err := doc.string("config.turboquant_compact_objectives"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		objectives, err := ParseTurboQuantPrefixObjectives(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantCompactObjectives = objectives
	}
	if value, ok, err := doc.string("config.turboquant_rank_margin_objectives"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		objectives, err := ParseTurboQuantPrefixObjectives(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantRankMarginObjectives = objectives
	}
	if value, ok, err := doc.float("config.turboquant_rank_margin"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantRankMargin = float32(value)
	}
	if value, ok, err := doc.string("config.turboquant_rank_margin_loss"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantRankMarginLoss = value
	}
	if value, ok, err := doc.string("config.turboquant_rank_margin_reduction"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantRankMarginReduction = value
	}
	if value, ok, err := doc.float("config.turboquant_rank_margin_tau"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantRankMarginTau = float32(value)
	}
	if value, ok, err := doc.string("config.turboquant_topk_objectives"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		objectives, err := ParseTurboQuantPrefixObjectives(value)
		if err != nil {
			return EmbeddingTrainManifest{}, err
		}
		manifest.Config.TurboQuantTopKObjectives = objectives
	}
	if value, ok, err := doc.string("config.turboquant_topk_loss"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKLoss = value
	}
	if value, ok, err := doc.int("config.turboquant_topk_cutoff"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKCutoff = int(value)
	}
	if value, ok, err := doc.float("config.turboquant_topk_tau"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKTau = float32(value)
	}
	if value, ok, err := doc.float("config.turboquant_topk_margin"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKMargin = float32(value)
	}
	if value, ok, err := doc.string("config.turboquant_topk_negative_mask"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKNegativeMask = value
	}
	if value, ok, err := doc.float("config.turboquant_topk_recall_weight"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKRecallWeight = float32(value)
	}
	if value, ok, err := doc.int("config.turboquant_topk_recall_cutoff"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKRecallCutoff = int(value)
	}
	if value, ok, err := doc.float("config.turboquant_topk_recall_tau"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKRecallTau = float32(value)
	}
	if value, ok, err := doc.float("config.turboquant_topk_recall_margin"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKRecallMargin = float32(value)
	}
	if value, ok, err := doc.string("config.turboquant_topk_recall_negative_mask"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.TurboQuantTopKRecallNegativeMask = value
	}
	if value, ok, err := doc.string("config.score_spectrum_loss_mode"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumLossMode = value
	}
	if value, ok, err := doc.float("config.score_spectrum_recovery_weight"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumRecoveryWeight = float32(value)
	}
	if value, ok, err := doc.float("config.score_spectrum_recovery_margin"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumRecoveryMargin = float32(value)
	}
	if value, ok, err := doc.int("config.score_spectrum_recovery_top_k"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumRecoveryTopK = int(value)
	}
	if value, ok, err := doc.float("config.score_spectrum_recovery_tau"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumRecoveryTau = float32(value)
	}
	if value, ok, err := doc.int("config.score_spectrum_activation_microbatch_size"); err != nil {
		return EmbeddingTrainManifest{}, err
	} else if ok {
		manifest.Config.ScoreSpectrumActivationMicrobatchSize = int(value)
	}
	if manifest.ScoreSpectrum, err = scoreSpectrumPolicyFromAuthoredDoc(doc, "score_spectrum."); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	if manifest.ListwiseGeometry, err = listwiseGeometryPolicyFromAuthoredDoc(doc, "listwise_geometry."); err != nil {
		return EmbeddingTrainManifest{}, err
	}
	return manifest, nil
}

func scoreSpectrumPolicyFromAuthoredDoc(doc authoredManifestDoc, prefix string) (EmbeddingScoreSpectrumPolicy, error) {
	var policy EmbeddingScoreSpectrumPolicy
	var err error
	if policy.ScoreSpectrumTrain, _, err = doc.bool(prefix + "score_spectrum_train"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	}
	if policy.ScoreSpectrumResearchOnly, _, err = doc.bool(prefix + "score_spectrum_research_only"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	}
	if policy.TrainAllowedForResearch, _, err = doc.bool(prefix + "train_allowed_for_research"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	}
	if policy.ReleaseTrainAllowed, _, err = doc.bool(prefix + "release_train_allowed"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	}
	if policy.CommercialUseAllowed, _, err = doc.bool(prefix + "commercial_use_allowed"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	}
	if value, _, err := doc.string(prefix + "source_artifact_hashes"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	} else {
		policy.SourceArtifactHashes = parseScoreSpectrumSourceHashes(value)
	}
	if value, _, err := doc.int(prefix + "score_spectrum_row_count"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	} else {
		policy.ScoreSpectrumRowCount = int(value)
	}
	if value, _, err := doc.string(prefix + "auto_cleared_objectives"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	} else {
		policy.AutoClearedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	if value, _, err := doc.string(prefix + "isolated_inherited_objectives"); err != nil {
		return EmbeddingScoreSpectrumPolicy{}, err
	} else {
		policy.IsolatedInheritedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	return policy, nil
}

func packageScoreSpectrumPolicy(policy EmbeddingScoreSpectrumPolicy) EmbeddingScoreSpectrumPolicy {
	return EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:          policy.ScoreSpectrumTrain,
		ScoreSpectrumResearchOnly:   policy.ScoreSpectrumResearchOnly,
		TrainAllowedForResearch:     policy.TrainAllowedForResearch,
		ReleaseTrainAllowed:         policy.ReleaseTrainAllowed,
		CommercialUseAllowed:        policy.CommercialUseAllowed,
		SourceArtifactHashes:        normalizeScoreSpectrumSourceHashes(policy.SourceArtifactHashes),
		ScoreSpectrumRowCount:       policy.ScoreSpectrumRowCount,
		AutoClearedObjectives:       normalizeScoreSpectrumObjectiveNames(policy.AutoClearedObjectives),
		IsolatedInheritedObjectives: normalizeScoreSpectrumObjectiveNames(policy.IsolatedInheritedObjectives),
	}
}

func listwiseGeometryPolicyFromAuthoredDoc(doc authoredManifestDoc, prefix string) (EmbeddingListwiseGeometryPolicy, error) {
	var policy EmbeddingListwiseGeometryPolicy
	var err error
	if policy.ListwiseGeometryTrain, _, err = doc.bool(prefix + "listwise_geometry_train"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	}
	if policy.ListwiseGeometryResearchOnly, _, err = doc.bool(prefix + "listwise_geometry_research_only"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	}
	if policy.TrainAllowedForResearch, _, err = doc.bool(prefix + "train_allowed_for_research"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	}
	if policy.ReleaseTrainAllowed, _, err = doc.bool(prefix + "release_train_allowed"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	}
	if policy.CommercialUseAllowed, _, err = doc.bool(prefix + "commercial_use_allowed"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	}
	if value, _, err := doc.string(prefix + "source_artifact_hashes"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	} else {
		policy.SourceArtifactHashes = parseScoreSpectrumSourceHashes(value)
	}
	if value, _, err := doc.int(prefix + "listwise_geometry_batch_count"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	} else {
		policy.ListwiseGeometryBatchCount = int(value)
	}
	if value, _, err := doc.string(prefix + "auto_cleared_objectives"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	} else {
		policy.AutoClearedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	if value, _, err := doc.string(prefix + "isolated_inherited_objectives"); err != nil {
		return EmbeddingListwiseGeometryPolicy{}, err
	} else {
		policy.IsolatedInheritedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	return policy, nil
}

func packageListwiseGeometryPolicy(policy EmbeddingListwiseGeometryPolicy) EmbeddingListwiseGeometryPolicy {
	return EmbeddingListwiseGeometryPolicy{
		ListwiseGeometryTrain:        policy.ListwiseGeometryTrain,
		ListwiseGeometryResearchOnly: policy.ListwiseGeometryResearchOnly,
		TrainAllowedForResearch:      policy.TrainAllowedForResearch,
		ReleaseTrainAllowed:          policy.ReleaseTrainAllowed,
		CommercialUseAllowed:         policy.CommercialUseAllowed,
		SourceArtifactHashes:         normalizeScoreSpectrumSourceHashes(policy.SourceArtifactHashes),
		ListwiseGeometryBatchCount:   policy.ListwiseGeometryBatchCount,
		AutoClearedObjectives:        normalizeScoreSpectrumObjectiveNames(policy.AutoClearedObjectives),
		IsolatedInheritedObjectives:  normalizeScoreSpectrumObjectiveNames(policy.IsolatedInheritedObjectives),
	}
}

func formatScoreSpectrumSourceHashes(values []string) string {
	values = normalizeScoreSpectrumSourceHashes(values)
	if len(values) == 0 {
		return ""
	}
	return strings.Join(values, ",")
}

func parseScoreSpectrumSourceHashes(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	return normalizeScoreSpectrumSourceHashes(strings.Split(raw, ","))
}

func formatScoreSpectrumObjectiveNames(values []string) string {
	values = normalizeScoreSpectrumObjectiveNames(values)
	if len(values) == 0 {
		return ""
	}
	return strings.Join(values, ",")
}

func parseScoreSpectrumObjectiveNames(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	return normalizeScoreSpectrumObjectiveNames(strings.Split(raw, ","))
}

func normalizeScoreSpectrumSourceHashes(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func normalizeScoreSpectrumObjectiveNames(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func formatMatryoshkaDims(dims []int) string {
	return formatIntList(dims)
}

func formatIntList(values []int) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ",")
}

func formatMatryoshkaWeights(weights []float32) string {
	if len(weights) == 0 {
		return ""
	}
	parts := make([]string, 0, len(weights))
	for _, weight := range weights {
		parts = append(parts, strconv.FormatFloat(float64(weight), 'g', -1, 32))
	}
	return strings.Join(parts, ",")
}

func parseMatryoshkaDims(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	dims := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		dim, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("parse matryoshka dim %q: %w", part, err)
		}
		dims = append(dims, dim)
	}
	return dims, nil
}

func parseMatryoshkaWeights(raw string) ([]float32, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	weights := make([]float32, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		weight, err := strconv.ParseFloat(part, 32)
		if err != nil {
			return nil, fmt.Errorf("parse matryoshka weight %q: %w", part, err)
		}
		weights = append(weights, float32(weight))
	}
	return weights, nil
}
