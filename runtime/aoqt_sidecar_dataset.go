package eosruntime

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
)

const (
	AOQTSidecarManifestSchema = "eos.q3_aoqt_sidecar_manifest.v1"
	AOQTSidecarRowSchema      = "eos.q3_aoqt_sidecar_row.v1"

	AOQTTopologyKindGivensV1 = EmbeddingPostPoolTransformAOQTGivens

	AOQTSidecarDim              = 384
	AOQTSidecarStages           = 8
	AOQTSidecarPairsPerStage    = AOQTSidecarDim / 2
	AOQTSidecarAngleCount       = AOQTSidecarStages * AOQTSidecarPairsPerStage
	AOQTSidecarDefaultAngleCap  = float32(0.04)
	AOQTSidecarHardMaxAngleCap  = float32(0.08)
	AOQTSidecarDefaultGainBit   = 3
	AOQTSidecarDefaultGuardBit3 = 3
	AOQTSidecarDefaultGuardBit5 = 5

	AOQTSidecarPreparedIPScoreSurface = "turboquant.ip.prepared_v1"
	AOQTSidecarDenseAnchorTolerance   = float32(1e-6)
	AOQTSidecarQuantAnchorTolerance   = float32(1e-6)
	AOQTSidecarUnitVectorTolerance    = float32(2e-5)
)

var requiredAOQTSplitExclusionIdentities = []string{
	"dev",
	"dev4",
	"reserve",
	"reserve4",
	"test",
	"official",
	"official-test",
	"proxy",
}

type AOQTSidecarLegalGates struct {
	ResearchTrainAllowed bool `json:"research_train_allowed"`
	ReleaseTrainAllowed  bool `json:"release_train_allowed"`
	CommercialUseAllowed bool `json:"commercial_use_allowed"`
	FreeOpenUseAllowed   bool `json:"free_open_release_allowed"`
}

type AOQTSidecarQuantSurface struct {
	BitWidth      int    `json:"bit_width"`
	Seed          int64  `json:"seed"`
	ScoreSurface  string `json:"score_surface"`
	PreparedQuery bool   `json:"prepared_query"`
}

type AOQTSidecarTopologyBinding struct {
	Kind           string `json:"kind"`
	Dim            int    `json:"dim"`
	Stages         int    `json:"stages"`
	PairsPerStage  int    `json:"pairs_per_stage"`
	AngleCount     int    `json:"angle_count"`
	Seed           int64  `json:"seed"`
	PairingsSHA256 string `json:"pairings_sha256"`
}

type AOQTSidecarTrainOnlySplitProof struct {
	Split               string   `json:"split"`
	TrainOnly           bool     `json:"train_only"`
	ProofSHA256         string   `json:"proof_sha256"`
	ExclusionIdentities []string `json:"exclusion_identities"`
}

type AOQTSidecarObjectiveContract struct {
	Dim              int                   `json:"dim"`
	TurboQuantSeed   int64                 `json:"turboquant_seed"`
	GainBit          int                   `json:"gain_bit"`
	Q3GuardBit       int                   `json:"q3_guard_bit"`
	Q5GuardBit       int                   `json:"q5_guard_bit"`
	ScoreSurface     string                `json:"score_surface"`
	GainCutoff       int                   `json:"gain_cutoff"`
	GainTau          float32               `json:"gain_tau"`
	GainMargin       float32               `json:"gain_margin"`
	GuardTau         float32               `json:"guard_tau"`
	GuardMargin      float32               `json:"guard_margin"`
	ScoreDistillTau  float32               `json:"score_distill_tau"`
	NFBoundarySource string                `json:"nf_boundary_source"`
	WeightSums       AOQTSidecarRowWeights `json:"weight_sums"`
}

type AOQTSidecarCalibrationManifest struct {
	Schema                      string                         `json:"schema"`
	CreatedAtUTC                string                         `json:"created_at_utc,omitempty"`
	AnchorArtifactPath          string                         `json:"anchor_artifact_path,omitempty"`
	AnchorArtifactSHA256        string                         `json:"anchor_artifact_sha256"`
	AnchorPackageManifestSHA256 string                         `json:"anchor_package_manifest_sha256"`
	AnchorEmbeddingSpaceID      string                         `json:"anchor_embedding_space_id"`
	Dim                         int                            `json:"dim"`
	Topology                    AOQTSidecarTopologyBinding     `json:"topology"`
	TurboQuantSeed              int64                          `json:"turboquant_seed"`
	QuantSurfaces               []AOQTSidecarQuantSurface      `json:"quant_surfaces"`
	ObjectiveContract           AOQTSidecarObjectiveContract   `json:"objective_contract"`
	QrelsSHA256ByDataset        map[string]string              `json:"qrels_sha256_by_dataset"`
	SplitProof                  AOQTSidecarTrainOnlySplitProof `json:"split_proof"`
	SourceArtifactHashes        []string                       `json:"source_artifact_hashes"`
	VectorCacheHashes           []string                       `json:"vector_cache_hashes"`
	RowCount                    int                            `json:"row_count"`
	RowIDSHA256                 string                         `json:"row_id_sha256"`
	CompatibilityDigest         string                         `json:"compatibility_digest"`
	LegalGates                  AOQTSidecarLegalGates          `json:"legal_gates"`
	Extra                       map[string]json.RawMessage     `json:"extra,omitempty"`
	// TrainingContract and CandidateEligibilityPolicy are omitted for legacy
	// manifests. A non-empty named contract is a new, explicitly materialized
	// training policy and is included in the canonical manifest digest.
	TrainingContract           string                                 `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy `json:"candidate_eligibility_policy,omitempty"`
}

type AOQTSidecarCalibrationRow struct {
	Schema                string                         `json:"schema"`
	RowID                 string                         `json:"row_id"`
	Dataset               string                         `json:"dataset"`
	QueryID               string                         `json:"query_id"`
	QueryVectorID         string                         `json:"query_vector_id"`
	QueryVectorSHA256     string                         `json:"query_vector_sha256"`
	QueryVector           []float32                      `json:"query_vector,omitempty"`
	CandidateDocIDs       []string                       `json:"candidate_doc_ids"`
	CandidateVectorIDs    []string                       `json:"candidate_vector_ids"`
	CandidateVectorSHA256 []string                       `json:"candidate_vector_sha256"`
	CandidateVectors      [][]float32                    `json:"candidate_vectors,omitempty"`
	QrelGains             []float32                      `json:"qrel_gains"`
	CandidateSources      []string                       `json:"candidate_sources"`
	EligiblePairMask      [][]bool                       `json:"eligible_pair_mask,omitempty"`
	GuardClass            string                         `json:"guard_class"`
	AnchorScores          AOQTSidecarAnchorScores        `json:"anchor_scores"`
	AnchorRanks           AOQTSidecarAnchorRanks         `json:"anchor_ranks"`
	Weights               AOQTSidecarRowWeights          `json:"weights"`
	SourceArtifactHash    string                         `json:"source_artifact_hash"`
	QrelsSHA256           string                         `json:"qrels_sha256"`
	SplitProof            AOQTSidecarTrainOnlySplitProof `json:"split_proof"`
	CompatibilityDigest   string                         `json:"compatibility_digest"`
	LegalGates            AOQTSidecarLegalGates          `json:"legal_gates"`
	Extra                 map[string]json.RawMessage     `json:"extra,omitempty"`
}

type AOQTSidecarAnchorScores struct {
	Dense []float32 `json:"dense"`
	Q3    []float32 `json:"q3"`
	Q5    []float32 `json:"q5"`
}

type AOQTSidecarAnchorRanks struct {
	Dense []int `json:"dense"`
	Q3    []int `json:"q3"`
	Q5    []int `json:"q5"`
}

type AOQTSidecarRowWeights struct {
	Q3Gain          float32 `json:"q3_gain"`
	Q3OrderGuard    float32 `json:"q3_order_guard"`
	Q3ScoreDistill  float32 `json:"q3_score_distill"`
	Q5OrderGuard    float32 `json:"q5_order_guard"`
	Q5ScoreDistill  float32 `json:"q5_score_distill"`
	NFBoundaryGuard float32 `json:"nf_boundary_guard"`
}

type AOQTSidecarCalibrationSet struct {
	Manifest AOQTSidecarCalibrationManifest
	Rows     []AOQTSidecarCalibrationRow
}

func (set AOQTSidecarCalibrationSet) Validate() error {
	if err := set.Manifest.Validate(); err != nil {
		return err
	}
	if len(set.Rows) != set.Manifest.RowCount {
		return fmt.Errorf("AOQT calibration row count = %d, want manifest row_count %d", len(set.Rows), set.Manifest.RowCount)
	}
	rowIDs := make([]string, 0, len(set.Rows))
	seen := map[string]bool{}
	for i, row := range set.Rows {
		if err := row.Validate(set.Manifest); err != nil {
			return fmt.Errorf("AOQT calibration row %d: %w", i, err)
		}
		if seen[row.RowID] {
			return fmt.Errorf("AOQT calibration duplicate row_id %q", row.RowID)
		}
		if i > 0 && row.RowID <= set.Rows[i-1].RowID {
			return fmt.Errorf("AOQT calibration rows must be strictly sorted by row_id")
		}
		seen[row.RowID] = true
		rowIDs = append(rowIDs, row.RowID)
	}
	if got := aoqtRowIDSHA256(rowIDs); got != set.Manifest.RowIDSHA256 {
		return fmt.Errorf("AOQT calibration row_id_sha256 mismatch")
	}
	if err := validateAOQTCalibrationObjectiveContract(set.Manifest.ObjectiveContract, set.Rows); err != nil {
		return err
	}
	return nil
}

func (m AOQTSidecarCalibrationManifest) Validate() error {
	if m.Schema != AOQTSidecarManifestSchema {
		return fmt.Errorf("AOQT manifest schema %q is not supported, want %q", m.Schema, AOQTSidecarManifestSchema)
	}
	if m.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT manifest dim = %d, want %d", m.Dim, AOQTSidecarDim)
	}
	if err := m.Topology.Validate(); err != nil {
		return err
	}
	if m.Topology.Dim != m.Dim {
		return fmt.Errorf("AOQT manifest topology dim = %d, want manifest dim %d", m.Topology.Dim, m.Dim)
	}
	if m.TurboQuantSeed == 0 {
		return fmt.Errorf("AOQT manifest turboquant_seed is required")
	}
	if err := validateAOQTQuantSurfaces(m.QuantSurfaces, m.TurboQuantSeed); err != nil {
		return err
	}
	if err := m.ObjectiveContract.Validate(m); err != nil {
		return err
	}
	if _, err := AOQTSidecarTrainingContractPolicy(m.TrainingContract, m.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT manifest training contract: %w", err)
	}
	if len(m.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT manifest qrels_sha256_by_dataset is required")
	}
	for dataset, sum := range m.QrelsSHA256ByDataset {
		if strings.TrimSpace(dataset) == "" {
			return fmt.Errorf("AOQT manifest qrels dataset name is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT manifest qrels sha256"); err != nil {
			return err
		}
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"anchor_artifact_sha256", m.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", m.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", m.AnchorEmbeddingSpaceID},
		{"compatibility_digest", m.CompatibilityDigest},
		{"row_id_sha256", m.RowIDSHA256},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("AOQT manifest %s is required", field.name)
		}
	}
	if err := validateAOQTSHA256(m.AnchorArtifactSHA256, "AOQT manifest anchor_artifact_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.AnchorPackageManifestSHA256, "AOQT manifest anchor_package_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.RowIDSHA256, "AOQT manifest row_id_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(m.CompatibilityDigest, "AOQT manifest compatibility_digest"); err != nil {
		return err
	}
	if err := m.SplitProof.Validate("AOQT manifest split_proof"); err != nil {
		return err
	}
	if err := validateAOQTHashList(m.SourceArtifactHashes, "AOQT manifest source_artifact_hashes"); err != nil {
		return err
	}
	if err := validateAOQTHashList(m.VectorCacheHashes, "AOQT manifest vector_cache_hashes"); err != nil {
		return err
	}
	if m.RowCount <= 0 {
		return fmt.Errorf("AOQT manifest row_count must be positive")
	}
	return validateAOQTResearchOnlyLegalGates(m.LegalGates, "AOQT manifest")
}

func (p AOQTSidecarTrainOnlySplitProof) Validate(label string) error {
	if strings.TrimSpace(p.Split) != "train" || !p.TrainOnly {
		return fmt.Errorf("%s must declare split=train and train_only=true", label)
	}
	if err := validateAOQTSHA256(p.ProofSHA256, label+".proof_sha256"); err != nil {
		return err
	}
	if len(p.ExclusionIdentities) == 0 {
		return fmt.Errorf("%s exclusion_identities are required", label)
	}
	seen := map[string]bool{}
	for _, identity := range p.ExclusionIdentities {
		if strings.TrimSpace(identity) == "" {
			return fmt.Errorf("%s exclusion identity is required", label)
		}
		seen[identity] = true
	}
	for _, required := range requiredAOQTSplitExclusionIdentities {
		if !seen[required] {
			return fmt.Errorf("%s missing required exclusion identity %q", label, required)
		}
	}
	return nil
}

func (b AOQTSidecarTopologyBinding) Validate() error {
	if b.Kind != AOQTTopologyKindGivensV1 {
		return fmt.Errorf("AOQT topology kind %q is not supported, want %q", b.Kind, AOQTTopologyKindGivensV1)
	}
	if b.Dim != AOQTSidecarDim {
		return fmt.Errorf("AOQT topology dim = %d, want %d", b.Dim, AOQTSidecarDim)
	}
	if b.Stages != AOQTSidecarStages {
		return fmt.Errorf("AOQT topology stages = %d, want %d", b.Stages, AOQTSidecarStages)
	}
	if b.PairsPerStage != AOQTSidecarPairsPerStage {
		return fmt.Errorf("AOQT topology pairs_per_stage = %d, want %d", b.PairsPerStage, AOQTSidecarPairsPerStage)
	}
	if b.AngleCount != AOQTSidecarAngleCount {
		return fmt.Errorf("AOQT topology angle_count = %d, want %d", b.AngleCount, AOQTSidecarAngleCount)
	}
	if b.Seed == 0 {
		return fmt.Errorf("AOQT topology seed is required")
	}
	return validateAOQTSHA256(b.PairingsSHA256, "AOQT topology pairings_sha256")
}

func (c AOQTSidecarObjectiveContract) Validate(manifest AOQTSidecarCalibrationManifest) error {
	if c.Dim != manifest.Dim {
		return fmt.Errorf("AOQT objective contract dim = %d, want manifest dim %d", c.Dim, manifest.Dim)
	}
	if c.TurboQuantSeed != manifest.TurboQuantSeed {
		return fmt.Errorf("AOQT objective contract turboquant_seed = %d, want manifest turboquant_seed %d", c.TurboQuantSeed, manifest.TurboQuantSeed)
	}
	if c.ScoreSurface != AOQTSidecarPreparedIPScoreSurface {
		return fmt.Errorf("AOQT objective contract score_surface = %q, want %q", c.ScoreSurface, AOQTSidecarPreparedIPScoreSurface)
	}
	for _, item := range []struct {
		name string
		bit  int
	}{
		{"gain_bit", c.GainBit},
		{"q3_guard_bit", c.Q3GuardBit},
		{"q5_guard_bit", c.Q5GuardBit},
	} {
		if item.bit < 2 || item.bit > 8 {
			return fmt.Errorf("AOQT objective contract %s = %d, want 2..8", item.name, item.bit)
		}
		if !aoqtManifestHasQuantSurface(manifest.QuantSurfaces, item.bit, c.TurboQuantSeed) {
			return fmt.Errorf("AOQT objective contract %s bit_width %d is not declared as prepared-IP quant surface", item.name, item.bit)
		}
	}
	if c.GainCutoff <= 0 {
		return fmt.Errorf("AOQT objective contract gain_cutoff must be positive")
	}
	for _, item := range []struct {
		name  string
		value float32
	}{
		{"gain_tau", c.GainTau},
		{"gain_margin", c.GainMargin},
		{"guard_tau", c.GuardTau},
		{"guard_margin", c.GuardMargin},
		{"score_distill_tau", c.ScoreDistillTau},
	} {
		if !isFinite32(item.value) {
			return fmt.Errorf("AOQT objective contract %s must be finite", item.name)
		}
	}
	if c.GainTau <= 0 || c.GuardTau <= 0 || c.ScoreDistillTau <= 0 {
		return fmt.Errorf("AOQT objective contract taus must be positive")
	}
	if strings.TrimSpace(c.NFBoundarySource) == "" {
		return fmt.Errorf("AOQT objective contract nf_boundary_source is required")
	}
	return validateAOQTRowWeights(c.WeightSums)
}

func (r AOQTSidecarCalibrationRow) Validate(manifest AOQTSidecarCalibrationManifest) error {
	if r.Schema != AOQTSidecarRowSchema {
		return fmt.Errorf("schema %q is not supported, want %q", r.Schema, AOQTSidecarRowSchema)
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"row_id", r.RowID},
		{"dataset", r.Dataset},
		{"query_id", r.QueryID},
		{"query_vector_id", r.QueryVectorID},
		{"source_artifact_hash", r.SourceArtifactHash},
		{"qrels_sha256", r.QrelsSHA256},
		{"compatibility_digest", r.CompatibilityDigest},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("%s is required", field.name)
		}
	}
	if r.CompatibilityDigest != manifest.CompatibilityDigest {
		return fmt.Errorf("compatibility_digest mismatch")
	}
	if !aoqtSplitProofEqual(r.SplitProof, manifest.SplitProof) {
		return fmt.Errorf("split_proof mismatch")
	}
	wantQrels, ok := manifest.QrelsSHA256ByDataset[r.Dataset]
	if !ok {
		return fmt.Errorf("dataset %q is not declared in qrels_sha256_by_dataset", r.Dataset)
	}
	if r.QrelsSHA256 != wantQrels {
		return fmt.Errorf("qrels_sha256 mismatch for dataset %q", r.Dataset)
	}
	if err := validateAOQTSHA256(r.QueryVectorSHA256, "query_vector_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(r.QrelsSHA256, "qrels_sha256"); err != nil {
		return err
	}
	if err := validateAOQTResearchOnlyLegalGates(r.LegalGates, "AOQT row"); err != nil {
		return err
	}
	if err := r.SplitProof.Validate("split_proof"); err != nil {
		return err
	}
	n := len(r.CandidateDocIDs)
	if n == 0 {
		return fmt.Errorf("candidate_doc_ids must not be empty")
	}
	if len(r.CandidateVectorIDs) != n || len(r.CandidateVectorSHA256) != n || len(r.QrelGains) != n || len(r.CandidateSources) != n {
		return fmt.Errorf("candidate metadata lengths must match candidate_doc_ids")
	}
	if len(r.QueryVector) != manifest.Dim {
		return fmt.Errorf("query_vector dim = %d, want %d", len(r.QueryVector), manifest.Dim)
	}
	if err := validateAOQTUnitVector(r.QueryVector, "query_vector"); err != nil {
		return err
	}
	if len(r.CandidateVectors) != n {
		return fmt.Errorf("candidate_vectors length = %d, want %d", len(r.CandidateVectors), n)
	}
	if got := aoqtVectorSHA256(r.QueryVector); got != r.QueryVectorSHA256 {
		return fmt.Errorf("query_vector_sha256 does not match query_vector")
	}
	if !slices.Contains(manifest.SourceArtifactHashes, r.SourceArtifactHash) {
		return fmt.Errorf("source_artifact_hash is not declared in manifest source_artifact_hashes")
	}
	for i := 0; i < n; i++ {
		if strings.TrimSpace(r.CandidateDocIDs[i]) == "" || strings.TrimSpace(r.CandidateVectorIDs[i]) == "" {
			return fmt.Errorf("candidate %d id fields are required", i)
		}
		if err := validateAOQTSHA256(r.CandidateVectorSHA256[i], fmt.Sprintf("candidate_vector_sha256[%d]", i)); err != nil {
			return err
		}
		if !isFinite32(r.QrelGains[i]) {
			return fmt.Errorf("qrel_gains[%d] must be finite", i)
		}
		if len(r.CandidateVectors[i]) != manifest.Dim {
			return fmt.Errorf("candidate_vectors[%d] dim = %d, want %d", i, len(r.CandidateVectors[i]), manifest.Dim)
		}
		if err := validateAOQTFiniteVector(r.CandidateVectors[i], fmt.Sprintf("candidate_vectors[%d]", i)); err != nil {
			return err
		}
		if err := validateAOQTUnitVector(r.CandidateVectors[i], fmt.Sprintf("candidate_vectors[%d]", i)); err != nil {
			return err
		}
		if got := aoqtVectorSHA256(r.CandidateVectors[i]); got != r.CandidateVectorSHA256[i] {
			return fmt.Errorf("candidate_vector_sha256[%d] does not match candidate_vectors[%d]", i, i)
		}
	}
	if err := validateAOQTFiniteVector(r.QueryVector, "query_vector"); err != nil {
		return err
	}
	if err := validateAOQTScoreLengths(r.AnchorScores.Dense, n, "anchor_scores.dense"); err != nil {
		return err
	}
	for i := 0; i < n; i++ {
		want := dotAOQT(r.QueryVector, r.CandidateVectors[i])
		if float32(math.Abs(float64(want-r.AnchorScores.Dense[i]))) > AOQTSidecarDenseAnchorTolerance {
			return fmt.Errorf("anchor_scores.dense[%d] = %.9g, want pre-transform dot %.9g within %.9g", i, r.AnchorScores.Dense[i], want, AOQTSidecarDenseAnchorTolerance)
		}
	}
	if err := validateAOQTScoreLengths(r.AnchorScores.Q3, n, "anchor_scores.q3"); err != nil {
		return err
	}
	if err := validateAOQTScoreLengths(r.AnchorScores.Q5, n, "anchor_scores.q5"); err != nil {
		return err
	}
	if err := validateAOQTRankLengths(r.AnchorRanks.Dense, n, "anchor_ranks.dense"); err != nil {
		return err
	}
	if err := validateAOQTRankLengths(r.AnchorRanks.Q3, n, "anchor_ranks.q3"); err != nil {
		return err
	}
	if err := validateAOQTRankLengths(r.AnchorRanks.Q5, n, "anchor_ranks.q5"); err != nil {
		return err
	}
	if err := validateAOQTDeclaredAnchors(r, manifest); err != nil {
		return err
	}
	if len(r.EligiblePairMask) > 0 {
		if len(r.EligiblePairMask) != n {
			return fmt.Errorf("eligible_pair_mask rows = %d, want %d", len(r.EligiblePairMask), n)
		}
		for i := range r.EligiblePairMask {
			if len(r.EligiblePairMask[i]) != n {
				return fmt.Errorf("eligible_pair_mask row %d length = %d, want %d", i, len(r.EligiblePairMask[i]), n)
			}
			if r.EligiblePairMask[i][i] {
				return fmt.Errorf("eligible_pair_mask row %d marks self-pair eligible", i)
			}
		}
	}
	if err := validateAOQTRowWeights(r.Weights); err != nil {
		return err
	}
	return validateAOQTRowObjectiveCoverage(r, manifest.ObjectiveContract)
}

func validateAOQTQuantSurfaces(surfaces []AOQTSidecarQuantSurface, seed int64) error {
	if len(surfaces) == 0 {
		return fmt.Errorf("AOQT manifest quant_surfaces are required")
	}
	seen := map[int]bool{}
	for i, surface := range surfaces {
		if surface.BitWidth < 1 || surface.BitWidth > 8 {
			return fmt.Errorf("AOQT quant surface %d bit_width = %d, want 1..8", i, surface.BitWidth)
		}
		if surface.Seed != seed {
			return fmt.Errorf("AOQT quant surface %d seed = %d, want turboquant_seed %d", i, surface.Seed, seed)
		}
		if surface.ScoreSurface != AOQTSidecarPreparedIPScoreSurface {
			return fmt.Errorf("AOQT quant surface %d score_surface = %q, want %q", i, surface.ScoreSurface, AOQTSidecarPreparedIPScoreSurface)
		}
		if !surface.PreparedQuery {
			return fmt.Errorf("AOQT quant surface %d must bind prepared_query=true", i)
		}
		seen[surface.BitWidth] = true
	}
	for _, bit := range []int{AOQTSidecarDefaultGainBit, AOQTSidecarDefaultGuardBit3, AOQTSidecarDefaultGuardBit5} {
		if !seen[bit] {
			return fmt.Errorf("AOQT quant surfaces must include bit width %d", bit)
		}
	}
	return nil
}

func aoqtManifestHasQuantSurface(surfaces []AOQTSidecarQuantSurface, bitWidth int, seed int64) bool {
	for _, surface := range surfaces {
		if surface.BitWidth == bitWidth && surface.Seed == seed && surface.ScoreSurface == AOQTSidecarPreparedIPScoreSurface && surface.PreparedQuery {
			return true
		}
	}
	return false
}

func validateAOQTHashList(values []string, label string) error {
	if len(values) == 0 {
		return fmt.Errorf("%s are required", label)
	}
	seen := map[string]bool{}
	for i, value := range values {
		if err := validateAOQTSHA256(value, fmt.Sprintf("%s[%d]", label, i)); err != nil {
			return err
		}
		if seen[value] {
			return fmt.Errorf("%s duplicate hash %q", label, value)
		}
		seen[value] = true
	}
	return nil
}

func aoqtSplitProofEqual(a, b AOQTSidecarTrainOnlySplitProof) bool {
	return a.Split == b.Split &&
		a.TrainOnly == b.TrainOnly &&
		a.ProofSHA256 == b.ProofSHA256 &&
		slices.Equal(a.ExclusionIdentities, b.ExclusionIdentities)
}

func validateAOQTRowWeights(weights AOQTSidecarRowWeights) error {
	values := []struct {
		name  string
		value float32
	}{
		{"q3_gain", weights.Q3Gain},
		{"q3_order_guard", weights.Q3OrderGuard},
		{"q3_score_distill", weights.Q3ScoreDistill},
		{"q5_order_guard", weights.Q5OrderGuard},
		{"q5_score_distill", weights.Q5ScoreDistill},
		{"nf_boundary_guard", weights.NFBoundaryGuard},
	}
	var active bool
	for _, item := range values {
		if !isFinite32(item.value) || item.value < 0 {
			return fmt.Errorf("weight %s must be finite and non-negative", item.name)
		}
		active = active || item.value > 0
	}
	if !active {
		return fmt.Errorf("at least one AOQT row weight must be positive")
	}
	return nil
}

func validateAOQTResearchOnlyLegalGates(gates AOQTSidecarLegalGates, label string) error {
	if !gates.ResearchTrainAllowed || gates.ReleaseTrainAllowed || gates.CommercialUseAllowed || gates.FreeOpenUseAllowed {
		return fmt.Errorf("%s requires research_train_allowed=true, release_train_allowed=false, commercial_use_allowed=false, free_open_release_allowed=false", label)
	}
	return nil
}

func validateAOQTScoreLengths(scores []float32, want int, name string) error {
	if len(scores) != want {
		return fmt.Errorf("%s length = %d, want %d", name, len(scores), want)
	}
	for i, score := range scores {
		if !isFinite32(score) {
			return fmt.Errorf("%s[%d] must be finite", name, i)
		}
	}
	return nil
}

func validateAOQTRankLengths(ranks []int, want int, name string) error {
	if len(ranks) != want {
		return fmt.Errorf("%s length = %d, want %d", name, len(ranks), want)
	}
	seen := make([]bool, want+1)
	for i, rank := range ranks {
		if rank <= 0 || rank > want {
			return fmt.Errorf("%s[%d] = %d, want unique permutation rank in 1..%d", name, i, rank, want)
		}
		if seen[rank] {
			return fmt.Errorf("%s repeats rank %d; ranks must be a unique 1..%d permutation", name, rank, want)
		}
		seen[rank] = true
	}
	return nil
}

func validateAOQTDeclaredAnchors(row AOQTSidecarCalibrationRow, manifest AOQTSidecarCalibrationManifest) error {
	if err := validateAOQTRanksMatchScores(row.AnchorRanks.Dense, ranksAOQT(row.CandidateDocIDs, row.AnchorScores.Dense), "anchor_ranks.dense"); err != nil {
		return err
	}
	q3 := newAOQTPreparedIPSurface(row.QueryVector, row.CandidateVectors, manifest.Dim, manifest.ObjectiveContract.Q3GuardBit, manifest.TurboQuantSeed)
	if err := validateAOQTScoresMatchSurface(row.AnchorScores.Q3, q3.scores, "anchor_scores.q3"); err != nil {
		return err
	}
	if err := validateAOQTRanksMatchScores(row.AnchorRanks.Q3, ranksAOQT(row.CandidateDocIDs, q3.scores), "anchor_ranks.q3"); err != nil {
		return err
	}
	q5 := newAOQTPreparedIPSurface(row.QueryVector, row.CandidateVectors, manifest.Dim, manifest.ObjectiveContract.Q5GuardBit, manifest.TurboQuantSeed)
	if err := validateAOQTScoresMatchSurface(row.AnchorScores.Q5, q5.scores, "anchor_scores.q5"); err != nil {
		return err
	}
	return validateAOQTRanksMatchScores(row.AnchorRanks.Q5, ranksAOQT(row.CandidateDocIDs, q5.scores), "anchor_ranks.q5")
}

func validateAOQTScoresMatchSurface(got, want []float32, label string) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if float32(math.Abs(float64(got[i]-want[i]))) > AOQTSidecarQuantAnchorTolerance {
			return fmt.Errorf("%s[%d] = %.9g, want prepared-IP score %.9g within %.9g", label, i, got[i], want[i], AOQTSidecarQuantAnchorTolerance)
		}
	}
	return nil
}

func validateAOQTRanksMatchScores(got, want []int, label string) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s length = %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("%s[%d] = %d, want recomputed rank %d using score-desc/doc-id-asc tie policy", label, i, got[i], want[i])
		}
	}
	return nil
}

func validateAOQTFiniteVector(vec []float32, name string) error {
	for i, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%s[%d] must be finite", name, i)
		}
	}
	return nil
}

func validateAOQTUnitVector(vec []float32, name string) error {
	norm := vectorNorm(vec)
	if float32(math.Abs(float64(norm-1))) > AOQTSidecarUnitVectorTolerance {
		return fmt.Errorf("%s norm = %.9g, want unit vector within %.9g so prepared-IP raw-vector scoring is unambiguous", name, norm, AOQTSidecarUnitVectorTolerance)
	}
	return nil
}

func validateAOQTRowObjectiveCoverage(row AOQTSidecarCalibrationRow, contract AOQTSidecarObjectiveContract) error {
	if row.Weights.Q3Gain > 0 {
		gainPairs := countAOQTGainEligiblePairs(row.QrelGains, row.EligiblePairMask)
		if gainPairs == 0 {
			return fmt.Errorf("row %q q3_gain weight %.9g has no usable gain-ordered eligible pairs", row.RowID, row.Weights.Q3Gain)
		}
	}
	if row.Weights.Q3OrderGuard > 0 {
		pairs := countAOQTRankOrderedPairs(row.AnchorRanks.Q3, row.EligiblePairMask)
		if pairs == 0 {
			return fmt.Errorf("row %q q3_order_guard weight %.9g has no usable anchor-rank guard pairs", row.RowID, row.Weights.Q3OrderGuard)
		}
	}
	if row.Weights.Q5OrderGuard > 0 {
		pairs := countAOQTRankOrderedPairs(row.AnchorRanks.Q5, row.EligiblePairMask)
		if pairs == 0 {
			return fmt.Errorf("row %q q5_order_guard weight %.9g has no usable anchor-rank guard pairs", row.RowID, row.Weights.Q5OrderGuard)
		}
	}
	if row.Weights.NFBoundaryGuard > 0 {
		pairs := countAOQTNFBoundaryGuardPairs(row.AnchorRanks.Q3, row.CandidateSources, contract.NFBoundarySource, row.EligiblePairMask)
		if pairs == 0 {
			return fmt.Errorf("row %q nf_boundary_guard weight %.9g has no usable NF boundary guard pairs for source %q", row.RowID, row.Weights.NFBoundaryGuard, contract.NFBoundarySource)
		}
	}
	if row.Weights.Q3ScoreDistill > 0 && len(row.CandidateDocIDs) < 2 {
		return fmt.Errorf("row %q q3_score_distill weight %.9g requires at least two candidates", row.RowID, row.Weights.Q3ScoreDistill)
	}
	if row.Weights.Q5ScoreDistill > 0 && len(row.CandidateDocIDs) < 2 {
		return fmt.Errorf("row %q q5_score_distill weight %.9g requires at least two candidates", row.RowID, row.Weights.Q5ScoreDistill)
	}
	return nil
}

func validateAOQTCalibrationObjectiveContract(contract AOQTSidecarObjectiveContract, rows []AOQTSidecarCalibrationRow) error {
	sums := sumAOQTRowWeights(rows)
	if !aoqtRowWeightsNearEqual(contract.WeightSums, sums) {
		return fmt.Errorf("AOQT objective contract weight_sums do not match calibration rows")
	}
	return nil
}

func sumAOQTRowWeights(rows []AOQTSidecarCalibrationRow) AOQTSidecarRowWeights {
	var out AOQTSidecarRowWeights
	for _, row := range rows {
		out.Q3Gain += row.Weights.Q3Gain
		out.Q3OrderGuard += row.Weights.Q3OrderGuard
		out.Q3ScoreDistill += row.Weights.Q3ScoreDistill
		out.Q5OrderGuard += row.Weights.Q5OrderGuard
		out.Q5ScoreDistill += row.Weights.Q5ScoreDistill
		out.NFBoundaryGuard += row.Weights.NFBoundaryGuard
	}
	return out
}

func aoqtRowWeightsNearEqual(a, b AOQTSidecarRowWeights) bool {
	return float32Near(a.Q3Gain, b.Q3Gain, 1e-6) &&
		float32Near(a.Q3OrderGuard, b.Q3OrderGuard, 1e-6) &&
		float32Near(a.Q3ScoreDistill, b.Q3ScoreDistill, 1e-6) &&
		float32Near(a.Q5OrderGuard, b.Q5OrderGuard, 1e-6) &&
		float32Near(a.Q5ScoreDistill, b.Q5ScoreDistill, 1e-6) &&
		float32Near(a.NFBoundaryGuard, b.NFBoundaryGuard, 1e-6)
}

func float32Near(a, b, tolerance float32) bool {
	return float32(math.Abs(float64(a-b))) <= tolerance
}

func countAOQTGainEligiblePairs(gains []float32, eligiblePairs [][]bool) int {
	var count int
	for high := range gains {
		for low := range gains {
			if gains[high] <= gains[low] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[high][low] {
				continue
			}
			count++
		}
	}
	return count
}

func countAOQTRankOrderedPairs(ranks []int, eligiblePairs [][]bool) int {
	var count int
	for high := range ranks {
		for low := range ranks {
			if high == low || ranks[high] >= ranks[low] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[high][low] {
				continue
			}
			count++
		}
	}
	return count
}

func countAOQTNFBoundaryGuardPairs(ranks []int, sources []string, boundarySource string, eligiblePairs [][]bool) int {
	if len(ranks) != len(sources) {
		return 0
	}
	var count int
	for boundary := range ranks {
		if sources[boundary] != boundarySource {
			continue
		}
		for other := range ranks {
			if boundary == other || ranks[boundary] >= ranks[other] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[boundary][other] {
				continue
			}
			count++
		}
	}
	return count
}

func validateAOQTSHA256(value, label string) error {
	value = strings.TrimSpace(value)
	if len(value) != sha256.Size*2 {
		return fmt.Errorf("%s must be %d hex chars", label, sha256.Size*2)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("%s is invalid: %w", label, err)
	}
	return nil
}

func aoqtRowIDSHA256(rowIDs []string) string {
	ids := append([]string(nil), rowIDs...)
	slices.Sort(ids)
	h := sha256.New()
	for _, id := range ids {
		_, _ = h.Write([]byte(id))
		_, _ = h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func aoqtVectorSHA256(vec []float32) string {
	h := sha256.New()
	var buf [4]byte
	for _, v := range vec {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func cloneAOQTRawMessageMap(in map[string]json.RawMessage) map[string]json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]json.RawMessage, len(in))
	for k, v := range in {
		out[k] = append(json.RawMessage(nil), v...)
	}
	return out
}
