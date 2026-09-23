package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
)

const (
	AOQTSidecarResidualLowRankScreenSchema         = "eos.aoqt.v10_residual_lowrank_screen.v1"
	AOQTSidecarOptimizerModeResidualLowRankScreen  = AOQTResidualLowRankTransformVersion
	AOQTResidualLowRankScheduleVersion             = "aoqt-v10-rank-role-alpha-canonical-24-v1"
	AOQTResidualLowRankSelectionSeed               = "aoqt-v10-residual-lowrank-seed-v1"
	AOQTResidualLowRankCandidateCap                = 24
	AOQTResidualLowRankObjectiveEvalMax            = 25
	AOQTResidualLowRankTurboQuantSeed              = int64(5581486560434873699)
	AOQTResidualLowRankTopK                        = 120
	AOQTResidualLowRankCanonicalV7R6ReceiptSHA256  = "41857a7d14ca87c589f3a51aab097a2fd5f6911cb8eb30ed1728ece1b792f63d"
	AOQTResidualLowRankCanonicalV8ReceiptSHA256    = "4bb0cdf9e756340b3c308336313034104a188bb16dc7b32394ceeaa51769d07b"
	AOQTResidualLowRankCanonicalV9ReceiptSHA256    = "710176938e43924d424c284fdc42b798d6e2a25cc4a5cc4409f079071eef40cc"
	AOQTResidualLowRankRealEvaluatorStatusDeferred = "real_evaluator_integration_deferred_fail_closed_v1"
)

type AOQTSidecarResidualLowRankScreenConfig struct {
	OutputReceiptJSONPath string
	Mode                  string
	ProbeOnly             bool
	DryRunOnly            bool

	BaseDocVectorPath   string
	BaseQueryVectorPath string
	ExpectedDocSHA256   string
	ExpectedQuerySHA256 string

	CanonicalV7R6ReceiptPath       string
	CanonicalV7R6ReceiptFileSHA256 string
	CanonicalV8ReceiptPath         string
	CanonicalV8ReceiptFileSHA256   string
	CanonicalV9ReceiptPath         string
	CanonicalV9ReceiptFileSHA256   string

	ExpectedAnchorArtifactSHA256        string
	ExpectedAnchorPackageManifestSHA256 string
	ExpectedAnchorEmbeddingSpaceID      string
	ExpectedCompatibilityDigest         string
	ExpectedWorkloadSHA256              string
	ExpectedQrelsSHA256                 string
	ExpectedTurboQuantBits              []int
	ExpectedTurboQuantSeed              int64
	ExpectedTopK                        int
	ExpectedPerQueryTopK                int
	ConstructionSeed                    string
}

type AOQTResidualLowRankEndpoint struct {
	Ordinal              int                                 `json:"ordinal"`
	Rank                 int                                 `json:"rank"`
	Role                 string                              `json:"role"`
	Alpha                float64                             `json:"alpha"`
	Seed                 string                              `json:"seed"`
	ParameterSHA256      string                              `json:"parameter_sha256"`
	CandidateDocSHA256   string                              `json:"candidate_doc_sha256,omitempty"`
	CandidateQuerySHA256 string                              `json:"candidate_query_sha256,omitempty"`
	EvaluatorStatus      string                              `json:"evaluator_status"`
	BaselineReproduced   bool                                `json:"baseline_reproduced"`
	Metrics              *AOQTResidualLowRankEndpointMetrics `json:"metrics,omitempty"`
	TransactionEligible  bool                                `json:"transaction_eligible"`
	Reason               string                              `json:"reason"`
	RankOrder            int                                 `json:"rank_order"`
}

type AOQTResidualLowRankEndpointMetrics struct {
	Q3NDCGAt10Delta    float64 `json:"q3_ndcg_at_10_delta"`
	DenseLossDelta     float64 `json:"dense_loss_delta"`
	Q5LossDelta        float64 `json:"q5_loss_delta"`
	Q3RecallDelta      float64 `json:"q3_recall_delta"`
	PerDomainSafe      bool    `json:"per_domain_safe"`
	NFBoundarySafe     bool    `json:"nf_boundary_safe"`
	PerQuerySHA256     string  `json:"per_query_sha256"`
	DenseMetricsSHA256 string  `json:"dense_metrics_sha256"`
	Q3MetricsSHA256    string  `json:"q3_metrics_sha256"`
	Q5MetricsSHA256    string  `json:"q5_metrics_sha256"`
	NFMetricsSHA256    string  `json:"nf_metrics_sha256"`
}

type AOQTResidualLowRankScreenBinding struct {
	BaseDocVectorPath              string `json:"base_doc_vector_path"`
	BaseDocVectorSHA256            string `json:"base_doc_vector_sha256"`
	BaseQueryVectorPath            string `json:"base_query_vector_path"`
	BaseQueryVectorSHA256          string `json:"base_query_vector_sha256"`
	CanonicalV7R6ReceiptPath       string `json:"canonical_v7_r6_receipt_path"`
	CanonicalV7R6ReceiptFileSHA256 string `json:"canonical_v7_r6_receipt_file_sha256"`
	CanonicalV8ReceiptPath         string `json:"canonical_v8_receipt_path"`
	CanonicalV8ReceiptFileSHA256   string `json:"canonical_v8_receipt_file_sha256"`
	CanonicalV9ReceiptPath         string `json:"canonical_v9_receipt_path"`
	CanonicalV9ReceiptFileSHA256   string `json:"canonical_v9_receipt_file_sha256"`
	AnchorArtifactSHA256           string `json:"anchor_artifact_sha256"`
	AnchorPackageManifestSHA256    string `json:"anchor_package_manifest_sha256"`
	AnchorEmbeddingSpaceID         string `json:"anchor_embedding_space_id"`
	CompatibilityDigest            string `json:"compatibility_digest"`
	WorkloadSHA256                 string `json:"workload_sha256"`
	QrelsSHA256                    string `json:"qrels_sha256"`
	TurboQuantBits                 []int  `json:"turboquant_bits"`
	TurboQuantSeed                 int64  `json:"turboquant_seed"`
	TopK                           int    `json:"top_k"`
	PerQueryTopK                   int    `json:"per_query_top_k"`
}

type AOQTSidecarResidualLowRankScreenReceipt struct {
	Schema                         string                           `json:"schema"`
	Mode                           string                           `json:"mode"`
	Required                       bool                             `json:"required"`
	ProbeOnly                      bool                             `json:"probe_only"`
	DryRunOnly                     bool                             `json:"dry_run_only"`
	Passed                         bool                             `json:"passed"`
	FailureReason                  string                           `json:"failure_reason"`
	TransformVersion               string                           `json:"transform_version"`
	ScheduleVersion                string                           `json:"schedule_version"`
	SelectionSeedSHA256            string                           `json:"selection_seed_sha256"`
	Dim                            int                              `json:"dim"`
	Ranks                          []int                            `json:"ranks"`
	Roles                          []string                         `json:"roles"`
	Magnitudes                     []float64                        `json:"magnitudes"`
	CandidateCap                   int                              `json:"candidate_cap"`
	ObjectiveEvaluationMax         int                              `json:"objective_evaluation_max"`
	ObjectiveEvaluationCount       int                              `json:"objective_evaluation_count"`
	BaselineReproduced             bool                             `json:"baseline_reproduced"`
	FullVerificationCompleted      bool                             `json:"full_verification_completed"`
	RealEvaluatorIntegrationStatus string                           `json:"real_evaluator_integration_status"`
	ResidualScheduleSHA256         string                           `json:"residual_schedule_sha256"`
	EndpointHashChain              string                           `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256    string                           `json:"endpoint_hash_chain_tail_sha256"`
	FeasibleEndpointCount          int                              `json:"feasible_endpoint_count"`
	SelectedEndpointOrdinals       []int                            `json:"selected_endpoint_ordinals"`
	Binding                        AOQTResidualLowRankScreenBinding `json:"binding"`
	Endpoints                      []AOQTResidualLowRankEndpoint    `json:"endpoints"`
	QualityClaim                   bool                             `json:"quality_claim"`
	ReleaseClaim                   bool                             `json:"release_claim"`
	OfficialClaim                  bool                             `json:"official_claim"`
	OfficialHeldoutGate            bool                             `json:"official_heldout_gate"`
	CommercialClaim                bool                             `json:"commercial_claim"`
	FreeOpenUseClaim               bool                             `json:"free_open_use_claim"`
}

type AOQTSidecarResidualLowRankScreenResult struct {
	Receipt         AOQTSidecarResidualLowRankScreenReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

func RunAOQTResidualLowRankScreen(cfg AOQTSidecarResidualLowRankScreenConfig) (AOQTSidecarResidualLowRankScreenResult, error) {
	result := AOQTSidecarResidualLowRankScreenResult{}
	if cfg.Mode == "" {
		cfg.Mode = AOQTSidecarOptimizerModeResidualLowRankScreen
	}
	if cfg.Mode != AOQTSidecarOptimizerModeResidualLowRankScreen || !cfg.ProbeOnly || !cfg.DryRunOnly {
		return result, fmt.Errorf("AOQT residual low-rank requires mode=%s, probe-only=true, dry-run-only=true", AOQTSidecarOptimizerModeResidualLowRankScreen)
	}
	if strings.TrimSpace(cfg.ConstructionSeed) == "" {
		cfg.ConstructionSeed = AOQTResidualLowRankSelectionSeed
	}
	if cfg.ConstructionSeed != AOQTResidualLowRankSelectionSeed {
		return result, fmt.Errorf("AOQT residual low-rank construction seed must be canonical %q", AOQTResidualLowRankSelectionSeed)
	}
	binding, err := aoqtResidualLowRankResolveBinding(cfg)
	if err != nil {
		return result, err
	}
	receipt, err := BuildAOQTResidualLowRankDryRunReceipt(binding, cfg.ConstructionSeed)
	if err != nil {
		return result, err
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := WriteAOQTResidualLowRankScreenReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
			return result, err
		}
		result.ReceiptJSONPath = cfg.OutputReceiptJSONPath
	} else if err := receipt.ValidateBound(); err != nil {
		return result, err
	}
	receiptSHA, err := receipt.SHA256()
	if err != nil {
		return result, err
	}
	result.Receipt, result.ReceiptSHA256 = receipt, receiptSHA
	return result, nil
}

func BuildAOQTResidualLowRankDryRunReceipt(binding AOQTResidualLowRankScreenBinding, seed string) (AOQTSidecarResidualLowRankScreenReceipt, error) {
	if strings.TrimSpace(seed) == "" {
		seed = AOQTResidualLowRankSelectionSeed
	}
	if seed != AOQTResidualLowRankSelectionSeed {
		return AOQTSidecarResidualLowRankScreenReceipt{}, fmt.Errorf("AOQT residual low-rank construction seed must be canonical %q", AOQTResidualLowRankSelectionSeed)
	}
	endpoints := make([]AOQTResidualLowRankEndpoint, 0, AOQTResidualLowRankCandidateCap)
	ordinal := 1
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				params, err := NewAOQTResidualLowRankParameters(rank, role, alpha, seed)
				if err != nil {
					return AOQTSidecarResidualLowRankScreenReceipt{}, err
				}
				paramSHA, err := params.SHA256()
				if err != nil {
					return AOQTSidecarResidualLowRankScreenReceipt{}, err
				}
				endpoints = append(endpoints, AOQTResidualLowRankEndpoint{
					Ordinal: ordinal, Rank: rank, Role: role, Alpha: alpha, Seed: seed,
					ParameterSHA256: paramSHA, EvaluatorStatus: AOQTResidualLowRankRealEvaluatorStatusDeferred,
					BaselineReproduced: false, TransactionEligible: false,
					Reason: "real_evaluator_integration_deferred", RankOrder: ordinal,
				})
				ordinal++
			}
		}
	}
	scheduleSHA, err := aoqtResidualLowRankScheduleSHA256(endpoints)
	if err != nil {
		return AOQTSidecarResidualLowRankScreenReceipt{}, err
	}
	chain, tail, err := aoqtResidualLowRankEndpointHashChain(endpoints)
	if err != nil {
		return AOQTSidecarResidualLowRankScreenReceipt{}, err
	}
	selectionSeedSHA, err := aoqtResidualLowRankSelectionSeedSHA256()
	if err != nil {
		return AOQTSidecarResidualLowRankScreenReceipt{}, err
	}
	return AOQTSidecarResidualLowRankScreenReceipt{
		Schema: AOQTSidecarResidualLowRankScreenSchema, Mode: AOQTSidecarOptimizerModeResidualLowRankScreen,
		Required: true, ProbeOnly: true, DryRunOnly: true, Passed: false,
		FailureReason:    "real evaluator integration deferred; no baseline reproduction or endpoint metrics are authoritative",
		TransformVersion: AOQTResidualLowRankTransformVersion, ScheduleVersion: AOQTResidualLowRankScheduleVersion,
		SelectionSeedSHA256: selectionSeedSHA, Dim: AOQTResidualLowRankDim,
		Ranks: AOQTResidualLowRankRanks(), Roles: AOQTResidualLowRankRoles(), Magnitudes: AOQTResidualLowRankMagnitudes(),
		CandidateCap: AOQTResidualLowRankCandidateCap, ObjectiveEvaluationMax: AOQTResidualLowRankObjectiveEvalMax,
		ObjectiveEvaluationCount: 0, BaselineReproduced: false, FullVerificationCompleted: false,
		RealEvaluatorIntegrationStatus: AOQTResidualLowRankRealEvaluatorStatusDeferred,
		ResidualScheduleSHA256:         scheduleSHA, EndpointHashChain: chain, EndpointHashChainTailSHA256: tail,
		FeasibleEndpointCount: 0, SelectedEndpointOrdinals: nil, Binding: binding, Endpoints: endpoints,
	}, nil
}

func (r AOQTSidecarResidualLowRankScreenReceipt) SHA256() (string, error) {
	return aoqtCanonicalSHA256(r)
}

func (r AOQTSidecarResidualLowRankScreenReceipt) Validate() error {
	return r.validate(false)
}

func (r AOQTSidecarResidualLowRankScreenReceipt) ValidateBound() error {
	if err := r.validate(true); err != nil {
		return err
	}
	return rereadAOQTResidualLowRankBinding(r.Binding)
}

func (r AOQTSidecarResidualLowRankScreenReceipt) validate(requireBinding bool) error {
	if r.Schema != AOQTSidecarResidualLowRankScreenSchema || r.Mode != AOQTSidecarOptimizerModeResidualLowRankScreen || !r.Required || !r.ProbeOnly || !r.DryRunOnly {
		return fmt.Errorf("AOQT residual low-rank receipt must use required probe-only dry-run V10 schema and mode")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim || r.FreeOpenUseClaim {
		return fmt.Errorf("AOQT residual low-rank claims and official heldout gate must be false")
	}
	if r.Passed || r.BaselineReproduced || r.FullVerificationCompleted || r.ObjectiveEvaluationCount != 0 || r.FeasibleEndpointCount != 0 || len(r.SelectedEndpointOrdinals) != 0 {
		return fmt.Errorf("AOQT residual low-rank dry-run receipt cannot pass, reproduce baseline, complete verification, or select endpoints")
	}
	if strings.TrimSpace(r.FailureReason) == "" {
		return fmt.Errorf("AOQT residual low-rank failed receipt requires failure_reason")
	}
	if r.TransformVersion != AOQTResidualLowRankTransformVersion || r.ScheduleVersion != AOQTResidualLowRankScheduleVersion || r.RealEvaluatorIntegrationStatus != AOQTResidualLowRankRealEvaluatorStatusDeferred {
		return fmt.Errorf("AOQT residual low-rank contract/status version binding is unsupported")
	}
	if r.Dim != AOQTResidualLowRankDim || r.CandidateCap != AOQTResidualLowRankCandidateCap || r.ObjectiveEvaluationMax != AOQTResidualLowRankObjectiveEvalMax {
		return fmt.Errorf("AOQT residual low-rank D384/cap binding is unsupported")
	}
	if !reflect.DeepEqual(r.Ranks, AOQTResidualLowRankRanks()) || !reflect.DeepEqual(r.Roles, AOQTResidualLowRankRoles()) || !reflect.DeepEqual(r.Magnitudes, AOQTResidualLowRankMagnitudes()) {
		return fmt.Errorf("AOQT residual low-rank canonical rank/role/magnitude schedule mismatch")
	}
	selectionSeedSHA, err := aoqtResidualLowRankSelectionSeedSHA256()
	if err != nil || r.SelectionSeedSHA256 != selectionSeedSHA {
		return fmt.Errorf("AOQT residual low-rank selection seed hash must replay the canonical construction seed")
	}
	if r.Binding.CanonicalV7R6ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV7R6ReceiptSHA256 || r.Binding.CanonicalV8ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV8ReceiptSHA256 || r.Binding.CanonicalV9ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV9ReceiptSHA256 {
		return fmt.Errorf("AOQT residual low-rank terminal V7/V8/V9 receipt hashes must match canonical pins")
	}
	for _, item := range []struct{ name, value string }{
		{"selection_seed_sha256", r.SelectionSeedSHA256},
		{"residual_schedule_sha256", r.ResidualScheduleSHA256},
		{"endpoint_hash_chain_tail_sha256", r.EndpointHashChainTailSHA256},
		{"base_doc_vector_sha256", r.Binding.BaseDocVectorSHA256},
		{"base_query_vector_sha256", r.Binding.BaseQueryVectorSHA256},
		{"canonical_v7_r6_receipt_file_sha256", r.Binding.CanonicalV7R6ReceiptFileSHA256},
		{"canonical_v8_receipt_file_sha256", r.Binding.CanonicalV8ReceiptFileSHA256},
		{"canonical_v9_receipt_file_sha256", r.Binding.CanonicalV9ReceiptFileSHA256},
		{"anchor_artifact_sha256", r.Binding.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", r.Binding.AnchorPackageManifestSHA256},
		{"compatibility_digest", r.Binding.CompatibilityDigest},
		{"workload_sha256", r.Binding.WorkloadSHA256},
		{"qrels_sha256", r.Binding.QrelsSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT residual low-rank "+item.name); err != nil {
			return err
		}
	}
	if strings.TrimSpace(r.Binding.AnchorEmbeddingSpaceID) == "" || len(r.Binding.TurboQuantBits) == 0 || r.Binding.TurboQuantSeed == 0 || r.Binding.TopK <= 0 || r.Binding.PerQueryTopK <= 0 {
		return fmt.Errorf("AOQT residual low-rank evaluator identity/workload binding is incomplete")
	}
	if err := validateTurboQuantRetrievalBits(r.Binding.TurboQuantBits); err != nil {
		return fmt.Errorf("AOQT residual low-rank TurboQuant bits: %w", err)
	}
	if !reflect.DeepEqual(r.Binding.TurboQuantBits, []int{3, 5}) || r.Binding.TurboQuantSeed != AOQTResidualLowRankTurboQuantSeed || r.Binding.TopK != AOQTResidualLowRankTopK || r.Binding.PerQueryTopK != AOQTResidualLowRankTopK {
		return fmt.Errorf("AOQT residual low-rank evaluator must bind TurboQuant bits [3,5], seed %d, and topK/perQueryTopK %d", AOQTResidualLowRankTurboQuantSeed, AOQTResidualLowRankTopK)
	}
	if len(r.Endpoints) != AOQTResidualLowRankCandidateCap {
		return fmt.Errorf("AOQT residual low-rank endpoint count = %d, want %d", len(r.Endpoints), AOQTResidualLowRankCandidateCap)
	}
	if err := validateAOQTResidualLowRankEndpointSchedule(r.Endpoints); err != nil {
		return err
	}
	scheduleSHA, err := aoqtResidualLowRankScheduleSHA256(r.Endpoints)
	if err != nil || scheduleSHA != r.ResidualScheduleSHA256 {
		return fmt.Errorf("AOQT residual low-rank schedule hash mismatch")
	}
	chain, tail, err := aoqtResidualLowRankEndpointHashChain(r.Endpoints)
	if err != nil || chain != r.EndpointHashChain || tail != r.EndpointHashChainTailSHA256 {
		return fmt.Errorf("AOQT residual low-rank endpoint hash chain mismatch")
	}
	for _, endpoint := range r.Endpoints {
		if err := endpoint.ValidateDryRun(); err != nil {
			return err
		}
	}
	if requireBinding {
		for _, item := range []struct{ name, value string }{
			{"base doc vector path", r.Binding.BaseDocVectorPath},
			{"base query vector path", r.Binding.BaseQueryVectorPath},
			{"canonical V7-r6 receipt path", r.Binding.CanonicalV7R6ReceiptPath},
			{"canonical V8 receipt path", r.Binding.CanonicalV8ReceiptPath},
			{"canonical V9 receipt path", r.Binding.CanonicalV9ReceiptPath},
		} {
			if strings.TrimSpace(item.value) == "" {
				return fmt.Errorf("AOQT residual low-rank bound receipt requires %s", item.name)
			}
		}
	}
	return nil
}

func (e AOQTResidualLowRankEndpoint) ValidateDryRun() error {
	if e.Ordinal <= 0 || e.RankOrder != e.Ordinal {
		return fmt.Errorf("AOQT residual low-rank endpoint ordinal/rank order mismatch")
	}
	if !containsIntAOQTResidual(e.Rank, AOQTResidualLowRankRanks()) || !containsStringAOQTResidual(e.Role, AOQTResidualLowRankRoles()) || !containsFloat64AOQTResidual(e.Alpha, AOQTResidualLowRankMagnitudes()) {
		return fmt.Errorf("AOQT residual low-rank endpoint %d has noncanonical rank/role/alpha", e.Ordinal)
	}
	if strings.TrimSpace(e.Seed) == "" {
		return fmt.Errorf("AOQT residual low-rank endpoint %d seed is empty", e.Ordinal)
	}
	if e.Seed != AOQTResidualLowRankSelectionSeed {
		return fmt.Errorf("AOQT residual low-rank endpoint %d seed must be canonical", e.Ordinal)
	}
	params, err := NewAOQTResidualLowRankParameters(e.Rank, e.Role, e.Alpha, e.Seed)
	if err != nil {
		return err
	}
	sha, err := params.SHA256()
	if err != nil || sha != e.ParameterSHA256 {
		return fmt.Errorf("AOQT residual low-rank endpoint %d parameter hash mismatch", e.Ordinal)
	}
	if e.EvaluatorStatus != AOQTResidualLowRankRealEvaluatorStatusDeferred || e.BaselineReproduced || e.Metrics != nil || e.TransactionEligible || e.Reason != "real_evaluator_integration_deferred" || e.CandidateDocSHA256 != "" || e.CandidateQuerySHA256 != "" {
		return fmt.Errorf("AOQT residual low-rank endpoint %d must remain dry-run non-authoritative", e.Ordinal)
	}
	return nil
}

func WriteAOQTResidualLowRankScreenReceipt(path string, receipt AOQTSidecarResidualLowRankScreenReceipt) error {
	if err := receipt.ValidateBound(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := writeExclusiveAOQTVectorCacheFile(path, data, 0o644); err != nil {
		return err
	}
	sha := sha256HexAOQTResidual(data)
	if err := writeExclusiveAOQTVectorCacheFile(path+".sha256", []byte(sha+"  "+filepathBaseAOQTResidual(path)+"\n"), 0o644); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}

func aoqtResidualLowRankResolveBinding(cfg AOQTSidecarResidualLowRankScreenConfig) (AOQTResidualLowRankScreenBinding, error) {
	if cfg.CanonicalV7R6ReceiptFileSHA256 == "" {
		cfg.CanonicalV7R6ReceiptFileSHA256 = AOQTResidualLowRankCanonicalV7R6ReceiptSHA256
	}
	if cfg.CanonicalV8ReceiptFileSHA256 == "" {
		cfg.CanonicalV8ReceiptFileSHA256 = AOQTResidualLowRankCanonicalV8ReceiptSHA256
	}
	if cfg.CanonicalV9ReceiptFileSHA256 == "" {
		cfg.CanonicalV9ReceiptFileSHA256 = AOQTResidualLowRankCanonicalV9ReceiptSHA256
	}
	for _, item := range []struct{ name, value string }{
		{"base doc vector path", cfg.BaseDocVectorPath},
		{"base query vector path", cfg.BaseQueryVectorPath},
		{"canonical V7-r6 receipt path", cfg.CanonicalV7R6ReceiptPath},
		{"canonical V8 receipt path", cfg.CanonicalV8ReceiptPath},
		{"canonical V9 receipt path", cfg.CanonicalV9ReceiptPath},
		{"anchor embedding space id", cfg.ExpectedAnchorEmbeddingSpaceID},
	} {
		if strings.TrimSpace(item.value) == "" {
			return AOQTResidualLowRankScreenBinding{}, fmt.Errorf("AOQT residual low-rank requires %s", item.name)
		}
	}
	docAudit, err := validateAOQTResidualLowRankCacheFile(cfg.BaseDocVectorPath)
	if err != nil {
		return AOQTResidualLowRankScreenBinding{}, fmt.Errorf("AOQT residual low-rank base doc cache validation: %w", err)
	}
	queryAudit, err := validateAOQTResidualLowRankCacheFile(cfg.BaseQueryVectorPath)
	if err != nil {
		return AOQTResidualLowRankScreenBinding{}, fmt.Errorf("AOQT residual low-rank base query cache validation: %w", err)
	}
	if cfg.ExpectedDocSHA256 != "" && docAudit.SHA256 != cfg.ExpectedDocSHA256 {
		return AOQTResidualLowRankScreenBinding{}, fmt.Errorf("AOQT residual low-rank base doc vector hash mismatch")
	}
	if cfg.ExpectedQuerySHA256 != "" && queryAudit.SHA256 != cfg.ExpectedQuerySHA256 {
		return AOQTResidualLowRankScreenBinding{}, fmt.Errorf("AOQT residual low-rank base query vector hash mismatch")
	}
	v7SHA, err := hashAOQTResidualRequiredFile(cfg.CanonicalV7R6ReceiptPath, cfg.CanonicalV7R6ReceiptFileSHA256, "canonical V7-r6 receipt")
	if err != nil {
		return AOQTResidualLowRankScreenBinding{}, err
	}
	v8SHA, err := hashAOQTResidualRequiredFile(cfg.CanonicalV8ReceiptPath, cfg.CanonicalV8ReceiptFileSHA256, "canonical V8 receipt")
	if err != nil {
		return AOQTResidualLowRankScreenBinding{}, err
	}
	v9SHA, err := hashAOQTResidualRequiredFile(cfg.CanonicalV9ReceiptPath, cfg.CanonicalV9ReceiptFileSHA256, "canonical V9 receipt")
	if err != nil {
		return AOQTResidualLowRankScreenBinding{}, err
	}
	binding := AOQTResidualLowRankScreenBinding{
		BaseDocVectorPath: cfg.BaseDocVectorPath, BaseDocVectorSHA256: docAudit.SHA256,
		BaseQueryVectorPath: cfg.BaseQueryVectorPath, BaseQueryVectorSHA256: queryAudit.SHA256,
		CanonicalV7R6ReceiptPath: cfg.CanonicalV7R6ReceiptPath, CanonicalV7R6ReceiptFileSHA256: v7SHA,
		CanonicalV8ReceiptPath: cfg.CanonicalV8ReceiptPath, CanonicalV8ReceiptFileSHA256: v8SHA,
		CanonicalV9ReceiptPath: cfg.CanonicalV9ReceiptPath, CanonicalV9ReceiptFileSHA256: v9SHA,
		AnchorArtifactSHA256: cfg.ExpectedAnchorArtifactSHA256, AnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID: cfg.ExpectedAnchorEmbeddingSpaceID, CompatibilityDigest: cfg.ExpectedCompatibilityDigest,
		WorkloadSHA256: cfg.ExpectedWorkloadSHA256, QrelsSHA256: cfg.ExpectedQrelsSHA256,
		TurboQuantBits: append([]int(nil), cfg.ExpectedTurboQuantBits...), TurboQuantSeed: cfg.ExpectedTurboQuantSeed, TopK: cfg.ExpectedTopK, PerQueryTopK: cfg.ExpectedPerQueryTopK,
	}
	if err := validateAOQTResidualLowRankBindingHashes(binding); err != nil {
		return AOQTResidualLowRankScreenBinding{}, err
	}
	return binding, nil
}

func validateAOQTResidualLowRankBindingHashes(binding AOQTResidualLowRankScreenBinding) error {
	if binding.CanonicalV7R6ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV7R6ReceiptSHA256 || binding.CanonicalV8ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV8ReceiptSHA256 || binding.CanonicalV9ReceiptFileSHA256 != AOQTResidualLowRankCanonicalV9ReceiptSHA256 {
		return fmt.Errorf("AOQT residual low-rank terminal V7/V8/V9 receipt hashes must match canonical pins")
	}
	for _, item := range []struct{ name, value string }{
		{"anchor_artifact_sha256", binding.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", binding.AnchorPackageManifestSHA256},
		{"compatibility_digest", binding.CompatibilityDigest},
		{"workload_sha256", binding.WorkloadSHA256},
		{"qrels_sha256", binding.QrelsSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT residual low-rank "+item.name); err != nil {
			return err
		}
	}
	if err := validateTurboQuantRetrievalBits(binding.TurboQuantBits); err != nil {
		return err
	}
	if !reflect.DeepEqual(binding.TurboQuantBits, []int{3, 5}) || binding.TurboQuantSeed != AOQTResidualLowRankTurboQuantSeed || binding.TopK != AOQTResidualLowRankTopK || binding.PerQueryTopK != AOQTResidualLowRankTopK {
		return fmt.Errorf("AOQT residual low-rank evaluator must bind TurboQuant bits [3,5], seed %d, and topK/perQueryTopK %d", AOQTResidualLowRankTurboQuantSeed, AOQTResidualLowRankTopK)
	}
	return nil
}

func rereadAOQTResidualLowRankBinding(binding AOQTResidualLowRankScreenBinding) error {
	for _, item := range []struct{ path, sha, label string }{
		{binding.BaseDocVectorPath, binding.BaseDocVectorSHA256, "base doc vector"},
		{binding.BaseQueryVectorPath, binding.BaseQueryVectorSHA256, "base query vector"},
		{binding.CanonicalV7R6ReceiptPath, binding.CanonicalV7R6ReceiptFileSHA256, "canonical V7-r6 receipt"},
		{binding.CanonicalV8ReceiptPath, binding.CanonicalV8ReceiptFileSHA256, "canonical V8 receipt"},
		{binding.CanonicalV9ReceiptPath, binding.CanonicalV9ReceiptFileSHA256, "canonical V9 receipt"},
	} {
		sha, err := sha256FileHexAOQTVectorCache(item.path)
		if err != nil {
			return fmt.Errorf("AOQT residual low-rank reread %s: %w", item.label, err)
		}
		if sha != item.sha {
			return fmt.Errorf("AOQT residual low-rank %s hash mismatch", item.label)
		}
	}
	if _, err := validateAOQTResidualLowRankCacheFile(binding.BaseDocVectorPath); err != nil {
		return fmt.Errorf("AOQT residual low-rank bound base doc cache validation: %w", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(binding.BaseQueryVectorPath); err != nil {
		return fmt.Errorf("AOQT residual low-rank bound base query cache validation: %w", err)
	}
	return validateAOQTResidualLowRankBindingHashes(binding)
}

func hashAOQTResidualRequiredFile(path, expectedSHA, label string) (string, error) {
	sha, err := sha256FileHexAOQTVectorCache(path)
	if err != nil {
		return "", fmt.Errorf("AOQT residual low-rank %s reread: %w", label, err)
	}
	if expectedSHA != "" && sha != expectedSHA {
		return "", fmt.Errorf("AOQT residual low-rank %s hash mismatch", label)
	}
	return sha, nil
}

func validateAOQTResidualLowRankEndpointSchedule(endpoints []AOQTResidualLowRankEndpoint) error {
	index := 0
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				if index >= len(endpoints) {
					return fmt.Errorf("AOQT residual low-rank omitted endpoint at canonical schedule position %d", index+1)
				}
				e := endpoints[index]
				if e.Ordinal != index+1 || e.Rank != rank || e.Role != role || e.Alpha != alpha {
					return fmt.Errorf("AOQT residual low-rank endpoint %d does not match canonical schedule position", index+1)
				}
				index++
			}
		}
	}
	return nil
}

func aoqtResidualLowRankScheduleSHA256(endpoints []AOQTResidualLowRankEndpoint) (string, error) {
	type item struct {
		Ordinal         int     `json:"ordinal"`
		Rank            int     `json:"rank"`
		Role            string  `json:"role"`
		Alpha           float64 `json:"alpha"`
		Seed            string  `json:"seed"`
		ParameterSHA256 string  `json:"parameter_sha256"`
	}
	items := make([]item, len(endpoints))
	for i, endpoint := range endpoints {
		items[i] = item{endpoint.Ordinal, endpoint.Rank, endpoint.Role, endpoint.Alpha, endpoint.Seed, endpoint.ParameterSHA256}
	}
	return aoqtCanonicalSHA256(struct {
		Version   string `json:"version"`
		Endpoints []item `json:"endpoints"`
	}{AOQTResidualLowRankScheduleVersion, items})
}

func aoqtResidualLowRankEndpointHashChain(endpoints []AOQTResidualLowRankEndpoint) (string, string, error) {
	hashes := make([]string, len(endpoints))
	for i, endpoint := range endpoints {
		hash, err := aoqtCanonicalSHA256(endpoint)
		if err != nil {
			return "", "", err
		}
		hashes[i] = hash
	}
	chainData, err := json.Marshal(hashes)
	if err != nil {
		return "", "", err
	}
	return string(chainData), sha256HexAOQTResidual(chainData), nil
}

func aoqtResidualLowRankSelectionSeedSHA256() (string, error) {
	return aoqtCanonicalSHA256(struct {
		Version         string `json:"version"`
		ScheduleVersion string `json:"schedule_version"`
		Seed            string `json:"seed"`
	}{AOQTResidualLowRankTransformVersion, AOQTResidualLowRankScheduleVersion, AOQTResidualLowRankSelectionSeed})
}

func filepathBaseAOQTResidual(path string) string {
	path = strings.TrimRight(path, string(os.PathSeparator))
	idx := strings.LastIndex(path, string(os.PathSeparator))
	if idx >= 0 {
		return path[idx+1:]
	}
	return path
}
