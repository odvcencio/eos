package eosruntime

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"m31labs.dev/turboquant"
)

const (
	AOQTSidecarMaterializerPreflightSchema = "eos.q3_aoqt_sidecar_materializer_preflight.v1"
	AOQTSidecarMaterializerQuantSeed       = int64(5581486560434873699)
	AOQTSidecarMaterializerTopologySeed    = int64(191)
)

type AOQTSidecarMaterializeConfig struct {
	PlanPath                   string
	ExclusionQIDPaths          []string
	VectorPaths                []string
	ScoreEvidencePaths         []string
	QrelsPaths                 []string
	RowJSONLPath               string
	ManifestJSONPath           string
	PreflightJSONPath          string
	AnchorEmbeddingSpaceID     string
	CreatedAtUTC               string
	AllowResearchOnly          bool
	ExpectedTurboQuantSeed     int64
	ExpectedTopologySeed       int64
	AnchorArtifactPath         string
	AnchorArtifactSHA256       string
	PackageManifestSHA256      string
	TrainingContract           string
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy
}

type AOQTSidecarMaterializePreflight struct {
	Schema                     string                                 `json:"schema"`
	CreatedAtUTC               string                                 `json:"created_at_utc"`
	QualityClaim               bool                                   `json:"quality_claim"`
	ResearchOnly               bool                                   `json:"research_only"`
	ActualTrainingRan          bool                                   `json:"actual_training_ran"`
	ActualEvalRan              bool                                   `json:"actual_eval_ran"`
	ActualVectorExportRan      bool                                   `json:"actual_vector_export_ran"`
	PlanSHA256                 string                                 `json:"plan_sha256"`
	InputSHA256                map[string]string                      `json:"input_sha256"`
	RowCount                   int                                    `json:"row_count"`
	CandidateCount             int                                    `json:"candidate_count"`
	PairCount                  int                                    `json:"pair_count"`
	TurboQuantSeed             int64                                  `json:"turboquant_seed"`
	TopologySeed               int64                                  `json:"topology_seed"`
	QuantScoreSurface          string                                 `json:"quant_score_surface"`
	QrelsSHA256ByDataset       map[string]string                      `json:"qrels_sha256_by_dataset"`
	SourceSemantics            []string                               `json:"source_semantics"`
	Outputs                    map[string]string                      `json:"outputs"`
	CalibrationManifestSHA256  string                                 `json:"calibration_manifest_sha256"`
	LegalGates                 AOQTSidecarLegalGates                  `json:"legal_gates"`
	ObjectiveContract          AOQTSidecarObjectiveContract           `json:"objective_contract"`
	TrainingContract           string                                 `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy `json:"candidate_eligibility_policy,omitempty"`
}

type aoqtMaterializePlan struct {
	Schema                   string `json:"schema"`
	Mode                     string `json:"mode"`
	ActualCacheGenerationRan bool   `json:"actual_cache_generation_ran"`
	ActualTrainingRan        bool   `json:"actual_training_ran"`
	ActualEvalRan            bool   `json:"actual_eval_ran"`
	Seed                     int64  `json:"seed"`
	TopologySeed             int64  `json:"topology_seed"`
	TurboQuant               struct {
		ScoreMode    string `json:"score_mode"`
		Seed         int64  `json:"seed"`
		RequiredBits []int  `json:"required_bits"`
		TopK         int    `json:"top_k"`
	} `json:"turboquant"`
	Topology   aoqtMaterializeTopology `json:"topology"`
	LegalScope struct {
		TrainAllowedForResearch bool `json:"train_allowed_for_research"`
		ReleaseTrainAllowed     bool `json:"release_train_allowed"`
		CommercialUseAllowed    bool `json:"commercial_use_allowed"`
		FreeOpenReleaseAllowed  bool `json:"free_open_release_allowed"`
		QualityClaim            bool `json:"quality_claim"`
	} `json:"legal_scope"`
	Anchor struct {
		Path           string `json:"path"`
		ManifestSHA256 string `json:"manifest_sha256"`
		PackageSHA256  string `json:"package_sha256"`
	} `json:"anchor"`
	Exclusions struct {
		Sets                        []aoqtMaterializeExclusionSet `json:"sets"`
		ExcludedQIDCountByDataset   map[string]int                `json:"excluded_qid_count_by_dataset"`
		ExcludedQIDsByDatasetSHA256 string                        `json:"excluded_qids_by_dataset_sha256"`
	} `json:"exclusions"`
	SelectionPolicy struct {
		Split                 string   `json:"split"`
		Datasets              []string `json:"datasets"`
		RequiredScoreBits     []int    `json:"required_score_bits"`
		ScoreMode             string   `json:"score_mode"`
		TurboQuantSeed        int64    `json:"turboquant_seed"`
		Top10GuardLimit       int      `json:"top10_guard_limit"`
		NFBoundaryWindow      []int    `json:"nf_boundary_window"`
		OfficialExclusionMode string   `json:"official_exclusion_mode"`
		ForbiddenSplits       []string `json:"forbidden_splits"`
	} `json:"selection_policy"`
	Rows       []aoqtMaterializePlanRow `json:"rows"`
	Provenance struct {
		QrelsSHA256ByDataset map[string]string `json:"qrels_sha256_by_dataset"`
	} `json:"provenance"`
}

type aoqtMaterializeTopology struct {
	ID              string  `json:"id"`
	Dim             int     `json:"dim"`
	Stages          int     `json:"stages"`
	PairsPerStage   int     `json:"pairs_per_stage"`
	AngleCount      int     `json:"angle_count"`
	AngleCapDefault float32 `json:"angle_cap_default"`
	AngleCapHard    float32 `json:"angle_cap_hard"`
}

type aoqtMaterializeExclusionSet struct {
	Name                string         `json:"name"`
	Path                string         `json:"path"`
	QIDCountByDataset   map[string]int `json:"qid_count_by_dataset"`
	QIDsByDatasetSHA256 string         `json:"qids_by_dataset_sha256"`
	ManifestSHA256      string         `json:"manifest_sha256"`
	Provenance          map[string]any `json:"provenance"`
}

type aoqtMaterializeExclusionManifest struct {
	SourcePath         string              `json:"-"`
	SHA256             string              `json:"-"`
	Schema             string              `json:"schema"`
	Name               string              `json:"name"`
	QIDsByDataset      map[string][]string `json:"qids_by_dataset"`
	SourceSHA256       string              `json:"source_sha256,omitempty"`
	SourceSHA256ByFile map[string]string   `json:"source_sha256_by_file,omitempty"`
}

type aoqtMaterializeExclusionValidation struct {
	ExcludedQIDs map[string]map[string]bool
	InputSHA256  map[string]string
}

type aoqtMaterializePlanRow struct {
	RowID           string   `json:"row_id"`
	Dataset         string   `json:"dataset"`
	Bits            int      `json:"bits"`
	Bucket          string   `json:"bucket"`
	QID             string   `json:"qid"`
	RankWindow      []int    `json:"rank_window"`
	CandidateDocIDs []string `json:"candidate_doc_ids"`
}

type aoqtMaterializeVectorRecord struct {
	Dataset          string    `json:"dataset"`
	Role             string    `json:"role"`
	ID               string    `json:"id"`
	VectorID         string    `json:"vector_id"`
	EmbeddingSpaceID string    `json:"embedding_space_id"`
	Vector           []float32 `json:"vector"`
}

type aoqtMaterializeScoreEvidenceFile struct {
	SourcePath string `json:"-"`
	SHA256     string `json:"-"`
	Schema     string `json:"schema"`
	Dataset    string `json:"dataset"`
	Bits       int    `json:"bits"`
	Split      string `json:"split"`
	ScoreMode  string `json:"score_mode"`
	TopK       int    `json:"top_k"`
	TurboQuant struct {
		ScoreMode string `json:"score_mode"`
		Bits      int    `json:"bits"`
		Seed      int64  `json:"seed"`
	} `json:"turboquant"`
	Rows []aoqtMaterializeScoreEvidenceRow `json:"rows"`
}

type aoqtMaterializeScoreEvidenceRow struct {
	QID  string                            `json:"qid"`
	Docs []aoqtMaterializeScoreEvidenceDoc `json:"docs"`
}

type aoqtMaterializeScoreEvidenceDoc struct {
	DocID string  `json:"doc_id"`
	Rank  int     `json:"rank"`
	Gain  float32 `json:"gain"`
	Score float32 `json:"score,omitempty"`
}

type aoqtMaterializeQrel struct {
	Dataset string
	QID     string
	DocID   string
	Gain    float32
}

type aoqtMaterializeCandidate struct {
	DocID     string
	VectorID  string
	Vector    []float32
	Gain      float32
	Source    string
	FirstSeen int
}

func MaterializeAOQTSidecarCalibration(cfg AOQTSidecarMaterializeConfig) (AOQTSidecarMaterializePreflight, error) {
	cfg = normalizeAOQTMaterializeConfig(cfg)
	if err := validateAOQTMaterializeConfig(cfg); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	plan, planSHA, err := loadAOQTMaterializePlan(cfg.PlanPath)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	if plan.Seed != cfg.ExpectedTopologySeed {
		return AOQTSidecarMaterializePreflight{}, fmt.Errorf("AOQT materializer plan seed = %d, want topology seed %d", plan.Seed, cfg.ExpectedTopologySeed)
	}
	exclusions, err := validateAOQTMaterializePlanBindings(plan, cfg)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	inputHashes := map[string]string{cfg.PlanPath: planSHA}
	for path, sum := range exclusions.InputSHA256 {
		inputHashes[path] = sum
	}
	for _, path := range append(append([]string{}, cfg.VectorPaths...), append(cfg.ScoreEvidencePaths, cfg.QrelsPaths...)...) {
		sum, err := sha256FileAOQT(path)
		if err != nil {
			return AOQTSidecarMaterializePreflight{}, err
		}
		inputHashes[path] = sum
	}
	qrels, qrelsByDataset, err := loadAOQTMaterializeQrels(cfg.QrelsPaths)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	if err := validateAOQTMaterializeQrelsBinding(plan, qrelsByDataset); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	queries, docs, vectorHashes, err := loadAOQTMaterializeVectors(cfg.VectorPaths, cfg.AnchorEmbeddingSpaceID)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	evidence, err := loadAOQTMaterializeScoreEvidence(cfg.ScoreEvidencePaths, cfg.ExpectedTurboQuantSeed)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	rows, err := buildAOQTMaterializeRows(plan, planSHA, queries, docs, qrels, evidence, qrelsByDataset)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	topology, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, cfg.ExpectedTopologySeed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	pairings, err := topology.PairingsSHA256()
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	rowIDs := make([]string, 0, len(rows))
	for _, row := range rows {
		rowIDs = append(rowIDs, row.RowID)
	}
	manifest := AOQTSidecarCalibrationManifest{
		Schema:                      AOQTSidecarManifestSchema,
		CreatedAtUTC:                cfg.CreatedAtUTC,
		AnchorArtifactPath:          firstNonEmptyAOQT(cfg.AnchorArtifactPath, plan.Anchor.Path),
		AnchorArtifactSHA256:        firstNonEmptyAOQT(cfg.AnchorArtifactSHA256, plan.Anchor.PackageSHA256),
		AnchorPackageManifestSHA256: firstNonEmptyAOQT(cfg.PackageManifestSHA256, plan.Anchor.ManifestSHA256),
		AnchorEmbeddingSpaceID:      cfg.AnchorEmbeddingSpaceID,
		Dim:                         AOQTSidecarDim,
		Topology: AOQTSidecarTopologyBinding{
			Kind:           AOQTTopologyKindGivensV1,
			Dim:            AOQTSidecarDim,
			Stages:         AOQTSidecarStages,
			PairsPerStage:  AOQTSidecarPairsPerStage,
			AngleCount:     AOQTSidecarAngleCount,
			Seed:           cfg.ExpectedTopologySeed,
			PairingsSHA256: pairings,
		},
		TurboQuantSeed: cfg.ExpectedTurboQuantSeed,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: AOQTSidecarDefaultGainBit, Seed: cfg.ExpectedTurboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
			{BitWidth: AOQTSidecarDefaultGuardBit5, Seed: cfg.ExpectedTurboQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true},
		},
		QrelsSHA256ByDataset: qrelsByDataset,
		SplitProof:           aoqtMaterializeSplitProof(plan),
		SourceArtifactHashes: aoqtMaterializeRowSourceHashes(rows),
		VectorCacheHashes:    sortedMapValuesAOQT(vectorHashes),
		RowCount:             len(rows),
		RowIDSHA256:          aoqtRowIDSHA256(rowIDs),
		CompatibilityDigest:  sha256StringsAOQT([]string{planSHA, cfg.AnchorEmbeddingSpaceID, fmt.Sprint(cfg.ExpectedTopologySeed), fmt.Sprint(cfg.ExpectedTurboQuantSeed)}),
		LegalGates:           AOQTSidecarLegalGates{ResearchTrainAllowed: true},
	}
	for i := range rows {
		rows[i].SplitProof = manifest.SplitProof
		rows[i].CompatibilityDigest = manifest.CompatibilityDigest
	}
	manifest.ObjectiveContract = AOQTSidecarPreparedIPObjectiveConfig{
		TurboQuantSeed:   cfg.ExpectedTurboQuantSeed,
		NFBoundarySource: "nf_boundary80_120",
	}.ObjectiveContract(sumAOQTRowWeights(rows))
	manifest.TrainingContract = cfg.TrainingContract
	manifest.CandidateEligibilityPolicy = cloneAOQTSidecarCandidateEligibilityPolicy(cfg.CandidateEligibilityPolicy)
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	if err := writeAOQTCalibrationRows(cfg.RowJSONLPath, rows); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	if err := writeJSONFileAOQT(cfg.ManifestJSONPath, manifest); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	manifestSHA, err := AOQTSidecarManifestSHA256(manifest)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	preflight := AOQTSidecarMaterializePreflight{
		Schema:                     AOQTSidecarMaterializerPreflightSchema,
		CreatedAtUTC:               cfg.CreatedAtUTC,
		QualityClaim:               false,
		ResearchOnly:               true,
		PlanSHA256:                 planSHA,
		InputSHA256:                inputHashes,
		RowCount:                   len(rows),
		CandidateCount:             countAOQTCandidates(rows),
		PairCount:                  countAOQTPairs(rows),
		TurboQuantSeed:             cfg.ExpectedTurboQuantSeed,
		TopologySeed:               cfg.ExpectedTopologySeed,
		QuantScoreSurface:          AOQTSidecarPreparedIPScoreSurface,
		QrelsSHA256ByDataset:       qrelsByDataset,
		SourceSemantics:            []string{"q3_top10", "q5_top10", "nf_boundary80_120"},
		Outputs:                    map[string]string{"rows_jsonl": cfg.RowJSONLPath, "manifest_json": cfg.ManifestJSONPath, "preflight_json": cfg.PreflightJSONPath},
		CalibrationManifestSHA256:  manifestSHA,
		LegalGates:                 manifest.LegalGates,
		ObjectiveContract:          manifest.ObjectiveContract,
		TrainingContract:           manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(manifest.CandidateEligibilityPolicy),
	}
	if err := writeJSONFileAOQT(cfg.PreflightJSONPath, preflight); err != nil {
		return AOQTSidecarMaterializePreflight{}, err
	}
	return preflight, nil
}

func normalizeAOQTMaterializeConfig(cfg AOQTSidecarMaterializeConfig) AOQTSidecarMaterializeConfig {
	if cfg.CreatedAtUTC == "" {
		cfg.CreatedAtUTC = time.Now().UTC().Format(time.RFC3339)
	}
	if cfg.ExpectedTurboQuantSeed == 0 {
		cfg.ExpectedTurboQuantSeed = AOQTSidecarMaterializerQuantSeed
	}
	if cfg.ExpectedTopologySeed == 0 {
		cfg.ExpectedTopologySeed = AOQTSidecarMaterializerTopologySeed
	}
	return cfg
}

func validateAOQTMaterializeConfig(cfg AOQTSidecarMaterializeConfig) error {
	required := []struct{ name, value string }{
		{"plan", cfg.PlanPath}, {"rows-jsonl", cfg.RowJSONLPath}, {"manifest-json", cfg.ManifestJSONPath},
		{"preflight-json", cfg.PreflightJSONPath}, {"anchor-embedding-space-id", cfg.AnchorEmbeddingSpaceID},
	}
	for _, item := range required {
		if strings.TrimSpace(item.value) == "" {
			return fmt.Errorf("AOQT materializer requires --%s", item.name)
		}
	}
	if !cfg.AllowResearchOnly {
		return fmt.Errorf("AOQT materializer requires --allow-research-only-aoqt")
	}
	if cfg.ExpectedTurboQuantSeed != AOQTSidecarMaterializerQuantSeed {
		return fmt.Errorf("AOQT materializer turboquant seed = %d, want %d", cfg.ExpectedTurboQuantSeed, AOQTSidecarMaterializerQuantSeed)
	}
	if cfg.ExpectedTopologySeed != AOQTSidecarMaterializerTopologySeed {
		return fmt.Errorf("AOQT materializer topology seed = %d, want %d", cfg.ExpectedTopologySeed, AOQTSidecarMaterializerTopologySeed)
	}
	if len(cfg.VectorPaths) == 0 || len(cfg.ScoreEvidencePaths) == 0 || len(cfg.QrelsPaths) == 0 {
		return fmt.Errorf("AOQT materializer requires vector, score-evidence, and qrels inputs")
	}
	if _, err := AOQTSidecarTrainingContractPolicy(cfg.TrainingContract, cfg.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT materializer training contract: %w", err)
	}
	return nil
}

func loadAOQTMaterializePlan(path string) (aoqtMaterializePlan, string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return aoqtMaterializePlan{}, "", err
	}
	var plan aoqtMaterializePlan
	if err := json.Unmarshal(data, &plan); err != nil {
		return aoqtMaterializePlan{}, "", err
	}
	if plan.Schema != "eos.aoqt_stage2_calibration_plan.v1" {
		return aoqtMaterializePlan{}, "", fmt.Errorf("AOQT materializer plan schema %q is not supported", plan.Schema)
	}
	if len(plan.Rows) == 0 {
		return aoqtMaterializePlan{}, "", fmt.Errorf("AOQT materializer plan rows are required")
	}
	sum := sha256.Sum256(data)
	return plan, hex.EncodeToString(sum[:]), nil
}

func validateAOQTMaterializePlanBindings(plan aoqtMaterializePlan, cfg AOQTSidecarMaterializeConfig) (aoqtMaterializeExclusionValidation, error) {
	if plan.Mode != "plan_only" {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan mode %q is not supported, want plan_only", plan.Mode)
	}
	if plan.ActualCacheGenerationRan || plan.ActualTrainingRan || plan.ActualEvalRan {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan must not claim cache generation, training, or eval ran")
	}
	if plan.TopologySeed != cfg.ExpectedTopologySeed {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan topology_seed = %d, want %d", plan.TopologySeed, cfg.ExpectedTopologySeed)
	}
	if plan.TurboQuant.ScoreMode != "prepared_ip" || plan.TurboQuant.Seed != cfg.ExpectedTurboQuantSeed || plan.TurboQuant.TopK < 120 || !intSetEqualAOQT(plan.TurboQuant.RequiredBits, []int{3, 5}) {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan TurboQuant prepared-IP binding mismatch")
	}
	if err := validateAOQTMaterializeTopology(plan.Topology); err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	if !plan.LegalScope.TrainAllowedForResearch || plan.LegalScope.ReleaseTrainAllowed || plan.LegalScope.CommercialUseAllowed || plan.LegalScope.FreeOpenReleaseAllowed || plan.LegalScope.QualityClaim {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan legal_scope must be research-only with no release, commercial, free-open, or quality claim")
	}
	if plan.SelectionPolicy.Split != "train" ||
		plan.SelectionPolicy.ScoreMode != "prepared_ip" ||
		plan.SelectionPolicy.TurboQuantSeed != cfg.ExpectedTurboQuantSeed ||
		plan.SelectionPolicy.Top10GuardLimit != 10 ||
		!intSetEqualAOQT(plan.SelectionPolicy.RequiredScoreBits, []int{3, 5}) ||
		!intSliceEqualAOQT(plan.SelectionPolicy.NFBoundaryWindow, []int{80, 120}) ||
		plan.SelectionPolicy.OfficialExclusionMode != "dataset_scoped_qid_only" {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan selection_policy binding mismatch")
	}
	if !stringSetEqualAOQT(plan.SelectionPolicy.Datasets, requiredAOQTMaterializeExclusionDatasets()) {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan selection_policy datasets must be exactly fiqa,nfcorpus,scifact")
	}
	if !stringSetIncludesAOQT(plan.SelectionPolicy.ForbiddenSplits, []string{"dev", "reserve", "official", "test", "eval", "heldout"}) {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan selection_policy forbidden split binding mismatch")
	}
	exclusions, err := validateAOQTMaterializePlanExclusions(plan, cfg)
	if err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	if !stringSetEqualAOQT(sortedStringMapKeysAOQT(plan.Provenance.QrelsSHA256ByDataset), requiredAOQTMaterializeExclusionDatasets()) {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan provenance.qrels_sha256_by_dataset must be exactly fiqa,nfcorpus,scifact")
	}
	for dataset, sum := range plan.Provenance.QrelsSHA256ByDataset {
		if strings.TrimSpace(dataset) == "" {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan qrels dataset is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT materializer plan qrels sha256"); err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
	}
	if err := validateAOQTMaterializePlanRows(plan.Rows, exclusions.ExcludedQIDs); err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	return exclusions, nil
}

func validateAOQTMaterializeTopology(topology aoqtMaterializeTopology) error {
	if topology.ID != AOQTTopologyKindGivensV1 ||
		topology.Dim != AOQTSidecarDim ||
		topology.Stages != AOQTSidecarStages ||
		topology.PairsPerStage != AOQTSidecarPairsPerStage ||
		topology.AngleCount != AOQTSidecarAngleCount ||
		topology.AngleCapDefault != AOQTSidecarDefaultAngleCap ||
		topology.AngleCapHard != AOQTSidecarHardMaxAngleCap {
		return fmt.Errorf("AOQT materializer plan topology binding mismatch")
	}
	return nil
}

func validateAOQTMaterializePlanExclusions(plan aoqtMaterializePlan, cfg AOQTSidecarMaterializeConfig) (aoqtMaterializeExclusionValidation, error) {
	sets := plan.Exclusions.Sets
	if len(sets) == 0 {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan qid-only exclusions are required")
	}
	if err := validateAOQTMaterializeDatasetCounts(plan.Exclusions.ExcludedQIDCountByDataset, "AOQT materializer plan aggregate excluded_qid_count_by_dataset"); err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	if err := validateAOQTSHA256(plan.Exclusions.ExcludedQIDsByDatasetSHA256, "AOQT materializer plan aggregate excluded_qids_by_dataset_sha256"); err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	seen := map[string]bool{}
	setsByName := map[string]aoqtMaterializeExclusionSet{}
	for _, set := range sets {
		if strings.TrimSpace(set.Name) == "" || strings.TrimSpace(set.Path) == "" {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan invalid qid-only exclusion proof for %q", set.Name)
		}
		if err := validateAOQTSHA256(set.ManifestSHA256, "AOQT materializer exclusion manifest sha256"); err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		if err := validateAOQTSHA256(set.QIDsByDatasetSHA256, "AOQT materializer exclusion qids_by_dataset_sha256"); err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		if err := validateAOQTMaterializeDatasetCounts(set.QIDCountByDataset, "AOQT materializer plan qid-only exclusion "+set.Name+" qid_count_by_dataset"); err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		if seen[set.Name] {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan duplicate qid-only exclusion %q", set.Name)
		}
		seen[set.Name] = true
		setsByName[set.Name] = set
	}
	for _, required := range []string{"dev4", "reserve4", "official-test"} {
		if !seen[required] {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer plan missing required qid-only exclusion %q", required)
		}
	}
	manifestPaths := cfg.ExclusionQIDPaths
	if len(manifestPaths) == 0 {
		for _, set := range sets {
			manifestPaths = append(manifestPaths, resolveAOQTMaterializePlanPath(cfg.PlanPath, set.Path))
		}
	}
	return validateAOQTMaterializeStrictExclusionManifests(manifestPaths, setsByName, plan.Exclusions.ExcludedQIDCountByDataset, plan.Exclusions.ExcludedQIDsByDatasetSHA256)
}

func validateAOQTMaterializeDatasetCounts(counts map[string]int, label string) error {
	if !stringSetEqualAOQT(sortedIntMapKeysAOQT(counts), requiredAOQTMaterializeExclusionDatasets()) {
		return fmt.Errorf("%s must cover exactly fiqa,nfcorpus,scifact", label)
	}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		if counts[dataset] <= 0 {
			return fmt.Errorf("%s has non-positive count for dataset %q", label, dataset)
		}
	}
	return nil
}

func validateAOQTMaterializeStrictExclusionManifests(paths []string, setsByName map[string]aoqtMaterializeExclusionSet, aggregateCounts map[string]int, aggregateSHA string) (aoqtMaterializeExclusionValidation, error) {
	if len(paths) == 0 {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer requires strict exclusion manifest inputs")
	}
	inputSHA := map[string]string{}
	excluded := map[string]map[string]bool{}
	seen := map[string]bool{}
	for _, path := range paths {
		manifest, err := loadAOQTMaterializeExclusionManifest(path)
		if err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		inputSHA[path] = manifest.SHA256
		set, ok := setsByName[manifest.Name]
		if !ok {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer exclusion manifest %q is not present in plan audit", manifest.Name)
		}
		if seen[manifest.Name] {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer duplicate strict exclusion manifest %q", manifest.Name)
		}
		seen[manifest.Name] = true
		if manifest.SHA256 != set.ManifestSHA256 {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer exclusion manifest %q sha256 mismatch: file %s plan %s", manifest.Name, manifest.SHA256, set.ManifestSHA256)
		}
		counts, err := validateAOQTMaterializeExclusionQIDsByDataset(manifest.Name, manifest.QIDsByDataset, excluded)
		if err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		if !intMapEqualAOQT(counts, set.QIDCountByDataset) {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer exclusion manifest %q qid_count_by_dataset mismatch", manifest.Name)
		}
		gotQIDSHA, err := sha256JSONAOQT(canonicalAOQTMaterializeQIDsByDataset(manifest.QIDsByDataset))
		if err != nil {
			return aoqtMaterializeExclusionValidation{}, err
		}
		if gotQIDSHA != set.QIDsByDatasetSHA256 {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer exclusion manifest %q qids_by_dataset_sha256 mismatch", manifest.Name)
		}
	}
	for name := range setsByName {
		if !seen[name] {
			return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer missing strict exclusion manifest %q", name)
		}
	}
	aggregate := boolExclusionMapToSortedQIDsAOQT(excluded)
	if !intMapEqualAOQT(countAOQTMaterializeQIDsByDataset(aggregate), aggregateCounts) {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer aggregate excluded_qid_count_by_dataset mismatch")
	}
	gotAggregateSHA, err := sha256JSONAOQT(aggregate)
	if err != nil {
		return aoqtMaterializeExclusionValidation{}, err
	}
	if gotAggregateSHA != aggregateSHA {
		return aoqtMaterializeExclusionValidation{}, fmt.Errorf("AOQT materializer aggregate excluded_qids_by_dataset_sha256 mismatch")
	}
	return aoqtMaterializeExclusionValidation{ExcludedQIDs: excluded, InputSHA256: inputSHA}, nil
}

func loadAOQTMaterializeExclusionManifest(path string) (aoqtMaterializeExclusionManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return aoqtMaterializeExclusionManifest{}, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: invalid exclusion manifest JSON: %w", path, err)
	}
	if _, exists := raw["qids"]; exists {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: ambiguous legacy global qids are not allowed; use qids_by_dataset", path)
	}
	allowed := map[string]bool{"schema": true, "name": true, "qids_by_dataset": true, "source_sha256": true, "source_sha256_by_file": true}
	for key := range raw {
		if !allowed[key] {
			return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: exclusion manifest must be qid-only; unexpected field %q", path, key)
		}
	}
	var manifest aoqtMaterializeExclusionManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: invalid exclusion manifest JSON: %w", path, err)
	}
	sum := sha256.Sum256(data)
	manifest.SourcePath = path
	manifest.SHA256 = hex.EncodeToString(sum[:])
	if manifest.Schema != "eos.aoqt_stage2.exclusion_qids.v1" {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: exclusion manifest schema %q is not supported", path, manifest.Schema)
	}
	if strings.TrimSpace(manifest.Name) == "" {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: exclusion manifest name is required", path)
	}
	if manifest.SourceSHA256 == "" && len(manifest.SourceSHA256ByFile) == 0 {
		return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: exclusion manifest source hash provenance is required", path)
	}
	if manifest.SourceSHA256 != "" {
		if err := validateAOQTSHA256(manifest.SourceSHA256, path+" source_sha256"); err != nil {
			return aoqtMaterializeExclusionManifest{}, err
		}
	}
	for key, value := range manifest.SourceSHA256ByFile {
		if strings.TrimSpace(key) == "" {
			return aoqtMaterializeExclusionManifest{}, fmt.Errorf("%s: source_sha256_by_file has empty key", path)
		}
		if err := validateAOQTSHA256(value, path+" source_sha256_by_file"); err != nil {
			return aoqtMaterializeExclusionManifest{}, err
		}
	}
	if _, err := validateAOQTMaterializeExclusionQIDsByDataset(manifest.Name, manifest.QIDsByDataset, nil); err != nil {
		return aoqtMaterializeExclusionManifest{}, err
	}
	return manifest, nil
}

func validateAOQTMaterializeExclusionQIDsByDataset(name string, qidsByDataset map[string][]string, excluded map[string]map[string]bool) (map[string]int, error) {
	if len(qidsByDataset) == 0 {
		return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q requires qids_by_dataset", name)
	}
	counts := map[string]int{}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		qids, ok := qidsByDataset[dataset]
		if !ok {
			return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q missing qids_by_dataset[%q]", name, dataset)
		}
		if len(qids) == 0 {
			return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q has empty qids_by_dataset[%q]", name, dataset)
		}
		counts[dataset] = len(qids)
	}
	for dataset, qids := range qidsByDataset {
		if !stringInSetAOQT(dataset, requiredAOQTMaterializeExclusionDatasets()) {
			return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q has unsupported dataset %q", name, dataset)
		}
		seenQIDs := map[string]bool{}
		for _, qid := range qids {
			if err := validateAOQTMaterializeQID(qid); err != nil {
				return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q has invalid qid for dataset %q: %w", name, dataset, err)
			}
			if seenQIDs[qid] {
				return nil, fmt.Errorf("AOQT materializer qid-only exclusion %q duplicate qid %q for dataset %q", name, qid, dataset)
			}
			seenQIDs[qid] = true
			if excluded != nil && excluded[dataset] == nil {
				excluded[dataset] = map[string]bool{}
			}
			if excluded != nil {
				excluded[dataset][qid] = true
			}
		}
	}
	return counts, nil
}

func validateAOQTMaterializeQID(qid string) error {
	if qid == "" || strings.ContainsFunc(qid, unicode.IsSpace) {
		return fmt.Errorf("qids must be non-empty strings without whitespace")
	}
	return nil
}

func requiredAOQTMaterializeExclusionDatasets() []string {
	return []string{"fiqa", "nfcorpus", "scifact"}
}

func validateAOQTMaterializePlanRows(rows []aoqtMaterializePlanRow, excludedQIDs map[string]map[string]bool) error {
	if len(rows) == 0 {
		return fmt.Errorf("AOQT materializer plan rows are required")
	}
	seenRows := map[string]bool{}
	rowDatasets := map[string]bool{}
	guardCoverage := map[string]bool{}
	for _, row := range rows {
		if strings.TrimSpace(row.RowID) == "" || strings.TrimSpace(row.Dataset) == "" || strings.TrimSpace(row.QID) == "" {
			return fmt.Errorf("AOQT materializer plan row identity fields are required")
		}
		if !stringInSetAOQT(row.Dataset, requiredAOQTMaterializeExclusionDatasets()) {
			return fmt.Errorf("AOQT materializer plan row %q has unsupported dataset %q", row.RowID, row.Dataset)
		}
		if seenRows[row.RowID] {
			return fmt.Errorf("AOQT materializer plan duplicate row_id %q", row.RowID)
		}
		seenRows[row.RowID] = true
		rowDatasets[row.Dataset] = true
		if excludedQIDs[row.Dataset][row.QID] {
			return fmt.Errorf("AOQT materializer plan row %q qid %q is excluded for dataset %q", row.RowID, row.QID, row.Dataset)
		}
		if _, err := aoqtMaterializePlanRowSource(row); err != nil {
			return err
		}
		guardCoverage[row.Dataset+"\x00"+fmt.Sprint(row.Bits)+"\x00"+row.Bucket] = true
		if len(row.CandidateDocIDs) != row.RankWindow[1]-row.RankWindow[0]+1 {
			return fmt.Errorf("AOQT materializer plan row %q candidate count does not match rank_window", row.RowID)
		}
		seenDocs := map[string]bool{}
		for _, docID := range row.CandidateDocIDs {
			if strings.TrimSpace(docID) == "" {
				return fmt.Errorf("AOQT materializer plan row %q has empty candidate doc_id", row.RowID)
			}
			if seenDocs[docID] {
				return fmt.Errorf("AOQT materializer plan row %q has duplicate candidate doc_id %q", row.RowID, docID)
			}
			seenDocs[docID] = true
		}
	}
	if !stringSetEqualAOQT(sortedBoolMapKeysAOQT(rowDatasets), requiredAOQTMaterializeExclusionDatasets()) {
		return fmt.Errorf("AOQT materializer plan rows must cover exactly fiqa,nfcorpus,scifact")
	}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		for _, bits := range []int{3, 5} {
			if !guardCoverage[dataset+"\x00"+fmt.Sprint(bits)+"\x00top10_guard"] {
				return fmt.Errorf("AOQT materializer plan rows missing top10 guard coverage for %s q%d", dataset, bits)
			}
		}
	}
	if !guardCoverage["nfcorpus\x003\x00nf_boundary80_120_guard"] {
		return fmt.Errorf("AOQT materializer plan rows missing NFCorpus q3 boundary rank 80..120 coverage")
	}
	if guardCoverage["nfcorpus\x005\x00nf_boundary80_120_guard"] {
		return fmt.Errorf("AOQT materializer plan rows must not include NFCorpus q5 boundary coverage")
	}
	return nil
}

func aoqtMaterializePlanRowSource(row aoqtMaterializePlanRow) (string, error) {
	if row.Bucket == "top10_guard" && row.Bits == 3 && intSliceEqualAOQT(row.RankWindow, []int{1, 10}) {
		return "q3_top10", nil
	}
	if row.Bucket == "top10_guard" && row.Bits == 5 && intSliceEqualAOQT(row.RankWindow, []int{1, 10}) {
		return "q5_top10", nil
	}
	if row.Bucket == "nf_boundary80_120_guard" && row.Dataset == "nfcorpus" && row.Bits == 3 && intSliceEqualAOQT(row.RankWindow, []int{80, 120}) {
		return "nf_boundary80_120", nil
	}
	return "", fmt.Errorf("AOQT materializer plan row %q has unsupported bucket/bits/window binding", row.RowID)
}

func validateAOQTMaterializeQrelsBinding(plan aoqtMaterializePlan, qrelsByDataset map[string]string) error {
	if len(plan.Provenance.QrelsSHA256ByDataset) != len(qrelsByDataset) {
		return fmt.Errorf("AOQT qrels sha256 dataset coverage mismatch: plan has %d datasets, inputs have %d", len(plan.Provenance.QrelsSHA256ByDataset), len(qrelsByDataset))
	}
	if !stringSetEqualAOQT(sortedStringMapKeysAOQT(qrelsByDataset), requiredAOQTMaterializeExclusionDatasets()) {
		return fmt.Errorf("AOQT qrels inputs must cover exactly fiqa,nfcorpus,scifact")
	}
	for dataset, want := range plan.Provenance.QrelsSHA256ByDataset {
		got := qrelsByDataset[dataset]
		if got == "" {
			return fmt.Errorf("AOQT qrels sha256 missing input dataset %q", dataset)
		}
		if got != want {
			return fmt.Errorf("AOQT qrels sha256 mismatch for dataset %q: file %s plan %s", dataset, got, want)
		}
	}
	return nil
}

func loadAOQTMaterializeVectors(paths []string, embeddingSpaceID string) (map[string]aoqtMaterializeVectorRecord, map[string]aoqtMaterializeVectorRecord, map[string]string, error) {
	queries := map[string]aoqtMaterializeVectorRecord{}
	docs := map[string]aoqtMaterializeVectorRecord{}
	hashes := map[string]string{}
	for _, path := range paths {
		sum, err := sha256FileAOQT(path)
		if err != nil {
			return nil, nil, nil, err
		}
		hashes[path] = sum
		file, err := os.Open(path)
		if err != nil {
			return nil, nil, nil, err
		}
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 1024), 16*1024*1024)
		lineNo := 0
		for scanner.Scan() {
			lineNo++
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rec aoqtMaterializeVectorRecord
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: invalid vector JSONL: %w", path, lineNo, err)
			}
			if rec.VectorID == "" {
				rec.VectorID = rec.ID
			}
			if rec.EmbeddingSpaceID != embeddingSpaceID {
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: embedding_space_id = %q, want %q", path, lineNo, rec.EmbeddingSpaceID, embeddingSpaceID)
			}
			if len(rec.Vector) != AOQTSidecarDim {
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: vector dim = %d, want %d", path, lineNo, len(rec.Vector), AOQTSidecarDim)
			}
			if err := validateAOQTFiniteVector(rec.Vector, "vector"); err != nil {
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			if err := validateAOQTUnitVector(rec.Vector, "vector"); err != nil {
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			key := rec.Dataset + "\x00" + rec.ID
			switch rec.Role {
			case "query":
				if _, exists := queries[key]; exists {
					_ = file.Close()
					return nil, nil, nil, fmt.Errorf("%s:%d: duplicate query vector %q/%q", path, lineNo, rec.Dataset, rec.ID)
				}
				queries[key] = rec
			case "doc":
				if _, exists := docs[key]; exists {
					_ = file.Close()
					return nil, nil, nil, fmt.Errorf("%s:%d: duplicate doc vector %q/%q", path, lineNo, rec.Dataset, rec.ID)
				}
				docs[key] = rec
			default:
				_ = file.Close()
				return nil, nil, nil, fmt.Errorf("%s:%d: vector role must be query or doc", path, lineNo)
			}
		}
		if err := scanner.Err(); err != nil {
			_ = file.Close()
			return nil, nil, nil, err
		}
		_ = file.Close()
	}
	return queries, docs, hashes, nil
}

func loadAOQTMaterializeScoreEvidence(paths []string, quantSeed int64) (map[string]map[int]aoqtMaterializeScoreEvidenceFile, error) {
	out := map[string]map[int]aoqtMaterializeScoreEvidenceFile{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var ev aoqtMaterializeScoreEvidenceFile
		if err := json.Unmarshal(data, &ev); err != nil {
			return nil, fmt.Errorf("%s: invalid score evidence JSON: %w", path, err)
		}
		sum := sha256.Sum256(data)
		ev.SourcePath = path
		ev.SHA256 = hex.EncodeToString(sum[:])
		if ev.Schema != "eos.aoqt_stage2.score_cache_manifest.v1" {
			return nil, fmt.Errorf("%s: score evidence schema %q is not supported", path, ev.Schema)
		}
		if ev.Split != "train" || ev.ScoreMode != "prepared_ip" || ev.TopK < 120 {
			return nil, fmt.Errorf("%s: score evidence must be train prepared_ip top_k>=120", path)
		}
		if ev.Bits != 3 && ev.Bits != 5 {
			return nil, fmt.Errorf("%s: score evidence bits = %d, want 3 or 5", path, ev.Bits)
		}
		if ev.TurboQuant.ScoreMode != "prepared_ip" || ev.TurboQuant.Bits != ev.Bits || ev.TurboQuant.Seed != quantSeed {
			return nil, fmt.Errorf("%s: TurboQuant prepared-IP config binding mismatch", path)
		}
		seenQIDs := map[string]bool{}
		for _, row := range ev.Rows {
			if strings.TrimSpace(row.QID) == "" {
				return nil, fmt.Errorf("%s: score evidence qid is required", path)
			}
			if seenQIDs[row.QID] {
				return nil, fmt.Errorf("%s: duplicate score evidence qid %q", path, row.QID)
			}
			seenQIDs[row.QID] = true
			seenDocs := map[string]bool{}
			for _, doc := range row.Docs {
				if strings.TrimSpace(doc.DocID) == "" {
					return nil, fmt.Errorf("%s: score evidence doc_id is required for qid %q", path, row.QID)
				}
				if seenDocs[doc.DocID] {
					return nil, fmt.Errorf("%s: duplicate score evidence doc_id %q for qid %q", path, doc.DocID, row.QID)
				}
				seenDocs[doc.DocID] = true
			}
		}
		if out[ev.Dataset] == nil {
			out[ev.Dataset] = map[int]aoqtMaterializeScoreEvidenceFile{}
		}
		if _, exists := out[ev.Dataset][ev.Bits]; exists {
			return nil, fmt.Errorf("duplicate score evidence for %s q%d", ev.Dataset, ev.Bits)
		}
		out[ev.Dataset][ev.Bits] = ev
	}
	return out, nil
}

func loadAOQTMaterializeQrels(paths []string) (map[string]float32, map[string]string, error) {
	out := map[string]float32{}
	hashes := map[string]string{}
	for _, path := range paths {
		sum, err := sha256FileAOQT(path)
		if err != nil {
			return nil, nil, err
		}
		dataset := strings.TrimSuffix(strings.TrimSuffix(path[strings.LastIndex(path, "/")+1:], ".jsonl"), ".qrels")
		file, err := os.Open(path)
		if err != nil {
			return nil, nil, err
		}
		reader := bufio.NewReader(file)
		lineNo := 0
		for {
			lineBytes, err := reader.ReadBytes('\n')
			if err != nil && err != io.EOF {
				_ = file.Close()
				return nil, nil, err
			}
			line := strings.TrimSpace(string(lineBytes))
			if line != "" {
				lineNo++
				qrel, parseErr := parseAOQTMaterializeQrel(line, dataset)
				if parseErr != nil {
					_ = file.Close()
					return nil, nil, fmt.Errorf("%s:%d: %w", path, lineNo, parseErr)
				}
				out[qrel.Dataset+"\x00"+qrel.QID+"\x00"+qrel.DocID] = qrel.Gain
				hashes[qrel.Dataset] = sum
			}
			if err == io.EOF {
				break
			}
		}
		_ = file.Close()
	}
	if len(out) == 0 {
		return nil, nil, fmt.Errorf("AOQT qrels are empty")
	}
	return out, hashes, nil
}

func parseAOQTMaterializeQrel(line, fallbackDataset string) (aoqtMaterializeQrel, error) {
	if strings.HasPrefix(line, "{") {
		var row struct {
			Dataset string  `json:"dataset"`
			QID     string  `json:"qid"`
			DocID   string  `json:"doc_id"`
			Gain    float32 `json:"gain"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return aoqtMaterializeQrel{}, err
		}
		if row.Dataset == "" {
			row.Dataset = fallbackDataset
		}
		if row.QID == "" || row.DocID == "" || row.Gain < 0 || !isFinite32(row.Gain) {
			return aoqtMaterializeQrel{}, fmt.Errorf("invalid qrel")
		}
		return aoqtMaterializeQrel{Dataset: row.Dataset, QID: row.QID, DocID: row.DocID, Gain: row.Gain}, nil
	}
	fields := strings.Fields(line)
	if len(fields) < 4 {
		return aoqtMaterializeQrel{}, fmt.Errorf("qrel must be JSONL or TREC qid unused doc_id gain")
	}
	gain, err := parseFloat32AOQT(fields[3])
	if err != nil || gain < 0 {
		return aoqtMaterializeQrel{}, fmt.Errorf("invalid qrel gain")
	}
	return aoqtMaterializeQrel{Dataset: fallbackDataset, QID: fields[0], DocID: fields[2], Gain: gain}, nil
}

func buildAOQTMaterializeRows(plan aoqtMaterializePlan, planSHA string, queries, docs map[string]aoqtMaterializeVectorRecord, qrels map[string]float32, evidence map[string]map[int]aoqtMaterializeScoreEvidenceFile, qrelsByDataset map[string]string) ([]AOQTSidecarCalibrationRow, error) {
	var out []AOQTSidecarCalibrationRow
	for _, planRow := range plan.Rows {
		byBits := evidence[planRow.Dataset]
		if byBits == nil || byBits[planRow.Bits].Dataset == "" {
			return nil, fmt.Errorf("AOQT materializer missing q%d evidence for dataset %q", planRow.Bits, planRow.Dataset)
		}
		ev := byBits[planRow.Bits]
		source, err := aoqtMaterializePlanRowSource(planRow)
		if err != nil {
			return nil, err
		}
		evidenceRow, ok := findAOQTEvidenceRow(ev, planRow.QID)
		if !ok {
			return nil, fmt.Errorf("AOQT materializer missing q%d evidence row for %s/%s", planRow.Bits, planRow.Dataset, planRow.QID)
		}
		candidates, err := materializeAOQTPlannedCandidates(planRow, evidenceRow, docs, qrels, source)
		if err != nil {
			return nil, err
		}
		query, ok := queries[planRow.Dataset+"\x00"+planRow.QID]
		if !ok {
			return nil, fmt.Errorf("AOQT materializer missing query vector for %s/%s", planRow.Dataset, planRow.QID)
		}
		sourceHash := aoqtMaterializeSourceArtifactHash(planSHA, ev.SHA256, planRow)
		row, err := newAOQTMaterializeCalibrationRow(planRow.RowID, planRow.Bucket, planRow.Dataset, planRow.QID, query, candidates, qrelsByDataset[planRow.Dataset], sourceHash)
		if err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].RowID < out[j].RowID })
	if len(out) == 0 {
		return nil, fmt.Errorf("AOQT materializer built no calibration rows")
	}
	return out, nil
}

func materializeAOQTPlannedCandidates(planRow aoqtMaterializePlanRow, evidenceRow aoqtMaterializeScoreEvidenceRow, docs map[string]aoqtMaterializeVectorRecord, qrels map[string]float32, source string) ([]aoqtMaterializeCandidate, error) {
	start, end := planRow.RankWindow[0], planRow.RankWindow[1]
	if len(evidenceRow.Docs) < end {
		return nil, fmt.Errorf("AOQT materializer q%d evidence for %s/%s has %d docs, want at least %d", planRow.Bits, planRow.Dataset, planRow.QID, len(evidenceRow.Docs), end)
	}
	window := evidenceRow.Docs[start-1 : end]
	out := make([]aoqtMaterializeCandidate, 0, len(planRow.CandidateDocIDs))
	for i, doc := range window {
		wantDocID := planRow.CandidateDocIDs[i]
		wantRank := start + i
		if doc.Rank != wantRank {
			return nil, fmt.Errorf("AOQT materializer q%d evidence rank %d outside exact Stage3 rank %d for %s/%s row %q", planRow.Bits, doc.Rank, wantRank, planRow.Dataset, planRow.QID, planRow.RowID)
		}
		if doc.DocID != wantDocID {
			return nil, fmt.Errorf("AOQT materializer q%d evidence doc %q at rank %d does not match Stage3 plan doc %q for row %q", planRow.Bits, doc.DocID, wantRank, wantDocID, planRow.RowID)
		}
		vec, ok := docs[planRow.Dataset+"\x00"+doc.DocID]
		if !ok {
			return nil, fmt.Errorf("AOQT materializer missing doc vector for %s/%s", planRow.Dataset, doc.DocID)
		}
		gain := qrels[planRow.Dataset+"\x00"+planRow.QID+"\x00"+doc.DocID]
		out = append(out, aoqtMaterializeCandidate{DocID: doc.DocID, VectorID: vec.VectorID, Vector: vec.Vector, Gain: gain, Source: source, FirstSeen: wantRank})
	}
	return out, nil
}

func findAOQTEvidenceRow(ev aoqtMaterializeScoreEvidenceFile, qid string) (aoqtMaterializeScoreEvidenceRow, bool) {
	for _, row := range ev.Rows {
		if row.QID == qid {
			return row, true
		}
	}
	return aoqtMaterializeScoreEvidenceRow{}, false
}

func newAOQTMaterializeCalibrationRow(rowID, guardClass, dataset, qid string, query aoqtMaterializeVectorRecord, candidates []aoqtMaterializeCandidate, qrelsSHA, sourceHash string) (AOQTSidecarCalibrationRow, error) {
	row := AOQTSidecarCalibrationRow{
		Schema:             AOQTSidecarRowSchema,
		RowID:              rowID,
		Dataset:            dataset,
		QueryID:            qid,
		QueryVectorID:      query.VectorID,
		QueryVectorSHA256:  aoqtVectorSHA256(query.Vector),
		QueryVector:        append([]float32(nil), query.Vector...),
		GuardClass:         guardClass,
		SourceArtifactHash: sourceHash,
		QrelsSHA256:        qrelsSHA,
		LegalGates:         AOQTSidecarLegalGates{ResearchTrainAllowed: true},
	}
	for _, candidate := range candidates {
		row.CandidateDocIDs = append(row.CandidateDocIDs, candidate.DocID)
		row.CandidateVectorIDs = append(row.CandidateVectorIDs, candidate.VectorID)
		row.CandidateVectorSHA256 = append(row.CandidateVectorSHA256, aoqtVectorSHA256(candidate.Vector))
		row.CandidateVectors = append(row.CandidateVectors, append([]float32(nil), candidate.Vector...))
		row.QrelGains = append(row.QrelGains, candidate.Gain)
		row.CandidateSources = append(row.CandidateSources, candidate.Source)
	}
	row.AnchorScores.Dense = denseAOQTScores(row.QueryVector, row.CandidateVectors)
	row.AnchorScores.Q3 = preparedAOQTScores(row.QueryVector, row.CandidateVectors, 3, AOQTSidecarMaterializerQuantSeed)
	row.AnchorScores.Q5 = preparedAOQTScores(row.QueryVector, row.CandidateVectors, 5, AOQTSidecarMaterializerQuantSeed)
	row.AnchorRanks.Dense = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Dense)
	row.AnchorRanks.Q3 = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Q3)
	row.AnchorRanks.Q5 = ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Q5)
	row.EligiblePairMask = positiveZeroMaskAOQT(row.QrelGains)
	row.Weights = AOQTSidecarRowWeights{
		Q3Gain:         1,
		Q3OrderGuard:   0.5,
		Q3ScoreDistill: 0.25,
		Q5OrderGuard:   0.5,
		Q5ScoreDistill: 0.25,
	}
	for _, source := range row.CandidateSources {
		if source == "nf_boundary80_120" {
			row.Weights.NFBoundaryGuard = 0.75
			break
		}
	}
	if countAOQTGainEligiblePairs(row.QrelGains, row.EligiblePairMask) == 0 {
		return AOQTSidecarCalibrationRow{}, fmt.Errorf("AOQT materializer row %q has no positive-vs-zero eligible pairs", rowID)
	}
	if row.Weights.NFBoundaryGuard > 0 && countAOQTNFBoundaryGuardPairs(row.AnchorRanks.Q3, row.CandidateSources, "nf_boundary80_120", row.EligiblePairMask) == 0 {
		return AOQTSidecarCalibrationRow{}, fmt.Errorf("AOQT materializer row %q positive NF boundary guard weight is inert", rowID)
	}
	return row, nil
}

func denseAOQTScores(query []float32, candidates [][]float32) []float32 {
	out := make([]float32, len(candidates))
	for i, candidate := range candidates {
		out[i] = dotAOQT(query, candidate)
	}
	return out
}

func preparedAOQTScores(query []float32, candidates [][]float32, bitWidth int, seed int64) []float32 {
	q := turboquant.NewIPWithSeed(AOQTSidecarDim, bitWidth, seed)
	prepared := q.PrepareQuery(query)
	out := make([]float32, len(candidates))
	for i, candidate := range candidates {
		out[i] = q.InnerProductPrepared(q.Quantize(candidate), prepared)
	}
	return out
}

func ranksAOQT(docIDs []string, scores []float32) []int {
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if scores[a] != scores[b] {
			return scores[a] > scores[b]
		}
		return docIDs[a] < docIDs[b]
	})
	ranks := make([]int, len(scores))
	for rank, idx := range order {
		ranks[idx] = rank + 1
	}
	return ranks
}

func positiveZeroMaskAOQT(gains []float32) [][]bool {
	mask := make([][]bool, len(gains))
	for i := range mask {
		mask[i] = make([]bool, len(gains))
		for j := range gains {
			mask[i][j] = i != j && ((gains[i] > 0 && gains[j] == 0) || (gains[i] == 0 && gains[j] > 0))
		}
	}
	return mask
}

func writeAOQTCalibrationRows(path string, rows []AOQTSidecarCalibrationRow) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(file)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			_ = file.Close()
			return err
		}
	}
	return file.Close()
}

func writeJSONFileAOQT(path string, payload any) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(file)
	enc.SetIndent("", "  ")
	if err := enc.Encode(payload); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

func sha256FileAOQT(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func sha256StringsAOQT(values []string) string {
	h := sha256.New()
	items := append([]string(nil), values...)
	sort.Strings(items)
	for _, value := range items {
		_, _ = h.Write([]byte(value))
		_, _ = h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func aoqtMaterializeSplitProof(plan aoqtMaterializePlan) AOQTSidecarTrainOnlySplitProof {
	identities := append([]string(nil), requiredAOQTSplitExclusionIdentities...)
	sets := append([]aoqtMaterializeExclusionSet(nil), plan.Exclusions.Sets...)
	sort.Slice(sets, func(i, j int) bool { return sets[i].Name < sets[j].Name })
	for _, set := range sets {
		identities = append(identities, aoqtMaterializeExclusionIdentity(set))
	}
	identities = append(identities, "dataset-scoped-qid-only:aggregate:"+plan.Exclusions.ExcludedQIDsByDatasetSHA256+":"+mustSHA256JSONAOQT(plan.Exclusions.ExcludedQIDCountByDataset))
	return AOQTSidecarTrainOnlySplitProof{
		Split:               "train",
		TrainOnly:           true,
		ProofSHA256:         sha256StringsAOQT(identities),
		ExclusionIdentities: identities,
	}
}

func aoqtMaterializeExclusionIdentity(set aoqtMaterializeExclusionSet) string {
	return "dataset-scoped-qid-only:" + set.Name + ":" + set.ManifestSHA256 + ":counts:" + mustSHA256JSONAOQT(set.QIDCountByDataset) + ":qids_by_dataset:" + set.QIDsByDatasetSHA256
}

func resolveAOQTMaterializePlanPath(planPath, path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return filepath.Join(filepath.Dir(planPath), path)
}

func canonicalAOQTMaterializeQIDsByDataset(qidsByDataset map[string][]string) map[string][]string {
	out := map[string][]string{}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		qids := append([]string(nil), qidsByDataset[dataset]...)
		sort.Strings(qids)
		out[dataset] = qids
	}
	return out
}

func boolExclusionMapToSortedQIDsAOQT(excluded map[string]map[string]bool) map[string][]string {
	out := map[string][]string{}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		for qid := range excluded[dataset] {
			out[dataset] = append(out[dataset], qid)
		}
		sort.Strings(out[dataset])
	}
	return out
}

func countAOQTMaterializeQIDsByDataset(qidsByDataset map[string][]string) map[string]int {
	out := map[string]int{}
	for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		out[dataset] = len(qidsByDataset[dataset])
	}
	return out
}

func sha256JSONAOQT(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func mustSHA256JSONAOQT(value any) string {
	sum, err := sha256JSONAOQT(value)
	if err != nil {
		panic(err)
	}
	return sum
}

func aoqtMaterializeSourceArtifactHash(planSHA, scoreSHA string, row aoqtMaterializePlanRow) string {
	return sha256StringsAOQT([]string{
		"plan:" + planSHA,
		"score:" + scoreSHA,
		"row:" + row.RowID,
		"dataset:" + row.Dataset,
		"bits:" + fmt.Sprint(row.Bits),
		"bucket:" + row.Bucket,
		"qid:" + row.QID,
		"rank_window:" + fmt.Sprint(row.RankWindow),
		"candidate_doc_ids:" + strings.Join(row.CandidateDocIDs, "\n"),
	})
}

func aoqtMaterializeRowSourceHashes(rows []AOQTSidecarCalibrationRow) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(rows))
	for _, row := range rows {
		if !seen[row.SourceArtifactHash] {
			seen[row.SourceArtifactHash] = true
			out = append(out, row.SourceArtifactHash)
		}
	}
	sort.Strings(out)
	return out
}

func sortedMapValuesAOQT(values map[string]string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func intSliceEqualAOQT(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func intSetEqualAOQT(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	left := append([]int(nil), a...)
	right := append([]int(nil), b...)
	sort.Ints(left)
	sort.Ints(right)
	return intSliceEqualAOQT(left, right)
}

func stringSetIncludesAOQT(values, required []string) bool {
	seen := map[string]bool{}
	for _, value := range values {
		seen[value] = true
	}
	for _, value := range required {
		if !seen[value] {
			return false
		}
	}
	return true
}

func stringSetEqualAOQT(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	left := append([]string(nil), a...)
	right := append([]string(nil), b...)
	sort.Strings(left)
	sort.Strings(right)
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func intMapEqualAOQT(a, b map[string]int) bool {
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		if b[key] != value {
			return false
		}
	}
	return true
}

func sortedStringMapKeysAOQT(values map[string]string) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func sortedIntMapKeysAOQT(values map[string]int) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func sortedBoolMapKeysAOQT(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func stringInSetAOQT(value string, set []string) bool {
	for _, item := range set {
		if value == item {
			return true
		}
	}
	return false
}

func sortedKeysAOQT(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func countAOQTCandidates(rows []AOQTSidecarCalibrationRow) int {
	total := 0
	for _, row := range rows {
		total += len(row.CandidateDocIDs)
	}
	return total
}

func countAOQTPairs(rows []AOQTSidecarCalibrationRow) int {
	total := 0
	for _, row := range rows {
		total += countEligibleAOQTPairs(row)
	}
	return total
}

func parseFloat32AOQT(value string) (float32, error) {
	parsed, err := strconvParseFloatAOQT(value)
	if err != nil {
		return 0, err
	}
	if math.IsNaN(float64(parsed)) || math.IsInf(float64(parsed), 0) {
		return 0, fmt.Errorf("not finite")
	}
	return parsed, nil
}

func strconvParseFloatAOQT(value string) (float32, error) {
	var out float64
	_, err := fmt.Sscanf(value, "%f", &out)
	return float32(out), err
}

func firstNonEmptyAOQT(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
