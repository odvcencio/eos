package eosruntime

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

const (
	AOQTV10bCacheScreenReceiptSchema   = "eos.aoqt.v10b_cache_screen_receipt.v1"
	AOQTV10bCacheScreenVersion         = "eos-d384-v10b-cache-native-screen-v1"
	AOQTV10bCacheScreenScheduleVersion = AOQTV10bGradientSVDScheduleVersion
	AOQTV10bCacheScreenQuantizerSeed   = int64(5581486560434873699)
	AOQTV10bCacheScreenTopK            = 120
	AOQTV10bCacheScreenPerQueryTopK    = 120
)

var aoqtV10bCacheScreenBits = []int{3, 5}

type AOQTV10bCacheScreenCase struct {
	Domain                    string
	Dataset                   string
	CorpusPath                string
	QueriesPath               string
	QrelsPath                 string
	DocVectorPath             string
	QueryVectorPath           string
	ExpectedCorpusSHA256      string
	ExpectedQueriesSHA256     string
	ExpectedQrelsSHA256       string
	ExpectedDocVectorSHA256   string
	ExpectedQueryVectorSHA256 string
	WorkloadSHA256            string
}

type AOQTV10bCacheScreenConfig struct {
	SourcePath            string
	Source                *AOQTV10bGradientSVDSource
	SourceFileSHA256      string
	Cases                 []AOQTV10bCacheScreenCase
	WorkDir               string
	OutputReceiptJSONPath string
	ReviewAuthority       string
}

type AOQTV10bCacheScreenRunResult struct {
	Receipt         AOQTV10bCacheScreenReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

type AOQTV10bCacheScreenReceipt struct {
	Schema                    string                         `json:"schema"`
	Version                   string                         `json:"version"`
	SourceSchema              string                         `json:"source_schema"`
	SourceVersion             string                         `json:"source_version"`
	DerivationVersion         string                         `json:"derivation_version"`
	ScheduleVersion           string                         `json:"schedule_version"`
	ResearchOnly              bool                           `json:"research_only"`
	ProbeOnly                 bool                           `json:"probe_only"`
	Passed                    bool                           `json:"passed"`
	FailureReason             string                         `json:"failure_reason"`
	ReviewAuthority           string                         `json:"review_authority"`
	Dim                       int                            `json:"dim"`
	Bits                      []int                          `json:"bits"`
	QuantizerSeed             int64                          `json:"quantizer_seed"`
	TopK                      int                            `json:"top_k"`
	PerQueryTopK              int                            `json:"per_query_top_k"`
	RerankPolicy              string                         `json:"rerank_policy"`
	DenseNDCGAt10MinDelta     float64                        `json:"dense_ndcg_at_10_min_delta"`
	Q3NDCGAt10MinDelta        float64                        `json:"q3_ndcg_at_10_min_delta"`
	Q3RecallAt100MinDelta     float64                        `json:"q3_recall_at_100_min_delta"`
	Q5NDCGAt10MinDelta        float64                        `json:"q5_ndcg_at_10_min_delta"`
	PerQuerySafetyMinDelta    float64                        `json:"per_query_safety_min_delta"`
	NFCorpusBoundaryMinDelta  float64                        `json:"nfcorpus_boundary_min_delta"`
	ObjectiveEvaluationCount  int                            `json:"objective_evaluation_count"`
	FullVerificationCompleted bool                           `json:"full_verification_completed"`
	MetricGatePassedCount     int                            `json:"metric_gate_passed_count"`
	FeasibleEndpointCount     int                            `json:"feasible_endpoint_count"`
	SelectedEndpointOrdinals  []int                          `json:"selected_endpoint_ordinals"`
	SourceBinding             AOQTV10bCacheScreenSourceBind  `json:"source_binding"`
	CaseBindings              []AOQTV10bCacheScreenCaseBind  `json:"case_bindings"`
	EndpointHashChain         string                         `json:"endpoint_hash_chain"`
	EndpointHashChainTail     string                         `json:"endpoint_hash_chain_tail_sha256"`
	Endpoints                 []AOQTV10bCacheScreenEndpoint  `json:"endpoints"`
	QualityClaim              bool                           `json:"quality_claim"`
	ReleaseClaim              bool                           `json:"release_claim"`
	OfficialClaim             bool                           `json:"official_claim"`
	OfficialHeldoutGate       bool                           `json:"official_heldout_gate"`
	CommercialClaim           bool                           `json:"commercial_claim"`
	FreeOpenUseClaim          bool                           `json:"free_open_use_claim"`
}

type AOQTV10bCacheScreenSourceBind struct {
	SourcePath               string `json:"source_path,omitempty"`
	SourceFileSHA256         string `json:"source_file_sha256"`
	SourceSHA256             string `json:"source_sha256"`
	FoldID                   string `json:"fold_id"`
	TrainRowsSHA256          string `json:"train_rows_sha256"`
	SplitManifestSHA256      string `json:"split_manifest_sha256"`
	Q3GradientSHA256         string `json:"q3_gradient_sha256"`
	ProtectedGradientSHA256  string `json:"protected_gradient_matrix_sha256"`
	ProjectedOperatorSHA256  string `json:"projected_operator_sha256"`
	ParameterScheduleSHA256  string `json:"parameter_schedule_sha256"`
	ProtectedGradientNames   []string `json:"protected_gradient_names"`
}

type AOQTV10bCacheScreenCaseBind struct {
	Domain                  string `json:"domain"`
	Dataset                 string `json:"dataset"`
	CorpusPath              string `json:"corpus_path"`
	CorpusSHA256            string `json:"corpus_sha256"`
	QueriesPath             string `json:"queries_path"`
	QueriesSHA256           string `json:"queries_sha256"`
	QrelsPath               string `json:"qrels_path"`
	QrelsSHA256             string `json:"qrels_sha256"`
	DocVectorPath           string `json:"doc_vector_path"`
	DocVectorSHA256         string `json:"doc_vector_sha256"`
	QueryVectorPath         string `json:"query_vector_path"`
	QueryVectorSHA256       string `json:"query_vector_sha256"`
	WorkloadSHA256          string `json:"workload_sha256"`
	DocumentCount           int    `json:"document_count"`
	QueryCount              int    `json:"query_count"`
	RelevantPairCount       int    `json:"relevant_pair_count"`
	BaselineMetricsSHA256    string `json:"baseline_metrics_sha256"`
	BaselinePerQuerySHA256   string `json:"baseline_per_query_sha256"`
	ValidationPolicy        string `json:"validation_policy"`
}

type AOQTV10bCacheScreenEndpoint struct {
	Ordinal                int                               `json:"ordinal"`
	Rank                   int                               `json:"rank"`
	Role                   string                            `json:"role"`
	Alpha                  float64                           `json:"alpha"`
	ParameterSHA256        string                            `json:"parameter_sha256"`
	TransformedCacheSHA256 string                            `json:"transformed_cache_sha256"`
	MetricGatePassed       bool                              `json:"metric_gate_passed"`
	TransactionEligible    bool                              `json:"transaction_eligible"`
	Reason                 string                            `json:"reason"`
	FailureReasons         []string                          `json:"failure_reasons,omitempty"`
	Cases                  []AOQTV10bCacheScreenEndpointCase `json:"cases"`
}

type AOQTV10bCacheScreenEndpointCase struct {
	Domain                   string  `json:"domain"`
	Dataset                  string  `json:"dataset"`
	CandidateDocVectorSHA256 string  `json:"candidate_doc_vector_sha256"`
	CandidateQueryVectorSHA256 string `json:"candidate_query_vector_sha256"`
	CandidateMetricsSHA256   string  `json:"candidate_metrics_sha256"`
	CandidatePerQuerySHA256  string  `json:"candidate_per_query_sha256"`
	DenseNDCGAt10Delta       float64 `json:"dense_ndcg_at_10_delta"`
	Q3NDCGAt10Delta          float64 `json:"q3_ndcg_at_10_delta"`
	Q3RecallAt100Delta       float64 `json:"q3_recall_at_100_delta"`
	Q5NDCGAt10Delta          float64 `json:"q5_ndcg_at_10_delta"`
	MinPerQueryNDCGAt10Delta float64 `json:"min_per_query_ndcg_at_10_delta"`
	NFCorpusBoundaryDelta    float64 `json:"nfcorpus_boundary_delta,omitempty"`
	MetricGatePassed         bool    `json:"metric_gate_passed"`
}

func AOQTV10bCacheScreenBits() []int { return append([]int(nil), aoqtV10bCacheScreenBits...) }

func RunAOQTV10bCacheScreen(ctx context.Context, cfg AOQTV10bCacheScreenConfig) (AOQTV10bCacheScreenRunResult, error) {
	source, sourceFileSHA, err := aoqtV10bCacheScreenSource(cfg)
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	sourceSHA, q3SHA, protectedNames, protectedSHA, projectedSHA, _, err := aoqtV10bProjectedOperator(source)
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	if strings.TrimSpace(sourceFileSHA) == "" {
		sourceFileSHA = sourceSHA
	}
	if strings.TrimSpace(cfg.WorkDir) == "" {
		return AOQTV10bCacheScreenRunResult{}, fmt.Errorf("AOQT V10b cache screen work dir is required")
	}
	if len(cfg.Cases) == 0 {
		return AOQTV10bCacheScreenRunResult{}, fmt.Errorf("AOQT V10b cache screen requires at least one dataset case")
	}
	workRoot, err := os.MkdirTemp(cfg.WorkDir, "aoqt-v10b-cache-screen-")
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, fmt.Errorf("create AOQT V10b cache work root: %w", err)
	}
	cleanupWorkRoot := true
	defer func() {
		if cleanupWorkRoot {
			_ = os.RemoveAll(workRoot)
		}
	}()
	caseStates := make([]aoqtV10bCacheScreenCaseState, len(cfg.Cases))
	caseBindings := make([]AOQTV10bCacheScreenCaseBind, len(cfg.Cases))
	for i, c := range cfg.Cases {
		state, binding, err := aoqtV10bCacheScreenPrepareCase(ctx, c, workRoot, i)
		if err != nil {
			return AOQTV10bCacheScreenRunResult{}, err
		}
		caseStates[i] = state
		caseBindings[i] = binding
	}
	endpoints := make([]AOQTV10bCacheScreenEndpoint, 0, AOQTV10bGradientSVDCandidateCap)
	ordinal := 1
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				params, err := DeriveAOQTV10bGradientSVDParameters(source, rank, role, alpha)
				if err != nil {
					return AOQTV10bCacheScreenRunResult{}, err
				}
				endpoint, err := aoqtV10bCacheScreenEvaluateEndpoint(ctx, workRoot, ordinal, params, caseStates)
				if err != nil {
					return AOQTV10bCacheScreenRunResult{}, err
				}
				endpoints = append(endpoints, endpoint)
				ordinal++
			}
		}
	}
	metricPassed := 0
	for _, endpoint := range endpoints {
		if endpoint.MetricGatePassed {
			metricPassed++
		}
	}
	scheduleSHA, err := aoqtV10bCacheScreenScheduleSHA256(endpoints)
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	chain, tail, err := aoqtV10bCacheScreenEndpointHashChain(endpoints)
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	receipt := AOQTV10bCacheScreenReceipt{
		Schema: AOQTV10bCacheScreenReceiptSchema, Version: AOQTV10bCacheScreenVersion, SourceSchema: AOQTV10bGradientSVDSchema, SourceVersion: AOQTV10bGradientSVDVersion, DerivationVersion: AOQTV10bGradientSVDDerivationVersion, ScheduleVersion: AOQTV10bCacheScreenScheduleVersion,
		ResearchOnly: true, ProbeOnly: true, Passed: false, FailureReason: "research-only cache-native V10b screen requires independent review before any transaction eligibility", ReviewAuthority: strings.TrimSpace(cfg.ReviewAuthority),
		Dim: AOQTV10bGradientSVDDim, Bits: AOQTV10bCacheScreenBits(), QuantizerSeed: AOQTV10bCacheScreenQuantizerSeed, TopK: AOQTV10bCacheScreenTopK, PerQueryTopK: AOQTV10bCacheScreenPerQueryTopK, RerankPolicy: "none",
		DenseNDCGAt10MinDelta: -0.0005, Q3NDCGAt10MinDelta: 0.0003, Q3RecallAt100MinDelta: 0, Q5NDCGAt10MinDelta: -0.001, PerQuerySafetyMinDelta: 0, NFCorpusBoundaryMinDelta: 0,
		ObjectiveEvaluationCount: len(endpoints), FullVerificationCompleted: true, MetricGatePassedCount: metricPassed, FeasibleEndpointCount: 0, SelectedEndpointOrdinals: []int{},
		SourceBinding: AOQTV10bCacheScreenSourceBind{SourcePath: cfg.SourcePath, SourceFileSHA256: sourceFileSHA, SourceSHA256: sourceSHA, FoldID: source.FoldID, TrainRowsSHA256: source.TrainRowsSHA256, SplitManifestSHA256: source.SplitManifestSHA256, Q3GradientSHA256: q3SHA, ProtectedGradientSHA256: protectedSHA, ProjectedOperatorSHA256: projectedSHA, ParameterScheduleSHA256: scheduleSHA, ProtectedGradientNames: protectedNames},
		CaseBindings: caseBindings, EndpointHashChain: chain, EndpointHashChainTail: tail, Endpoints: endpoints,
	}
	if err := receipt.ValidateWithSource(source); err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	var result AOQTV10bCacheScreenRunResult
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := WriteAOQTV10bCacheScreenReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
			return AOQTV10bCacheScreenRunResult{}, err
		}
		result.ReceiptJSONPath = cfg.OutputReceiptJSONPath
	}
	sha, err := receipt.SHA256()
	if err != nil {
		return AOQTV10bCacheScreenRunResult{}, err
	}
	result.Receipt = receipt
	result.ReceiptSHA256 = sha
	cleanupWorkRoot = false
	return result, nil
}

func (r AOQTV10bCacheScreenReceipt) SHA256() (string, error) { return aoqtCanonicalSHA256(r) }

func WriteAOQTV10bCacheScreenReceipt(path string, receipt AOQTV10bCacheScreenReceipt) error {
	if err := receipt.Validate(); err != nil {
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

func (r AOQTV10bCacheScreenReceipt) Validate() error {
	if r.Schema != AOQTV10bCacheScreenReceiptSchema || r.Version != AOQTV10bCacheScreenVersion || r.SourceSchema != AOQTV10bGradientSVDSchema || r.SourceVersion != AOQTV10bGradientSVDVersion || r.DerivationVersion != AOQTV10bGradientSVDDerivationVersion || r.ScheduleVersion != AOQTV10bCacheScreenScheduleVersion {
		return fmt.Errorf("AOQT V10b cache screen receipt schema/version is unsupported")
	}
	if !r.ResearchOnly || !r.ProbeOnly || r.Passed || r.FeasibleEndpointCount != 0 || len(r.SelectedEndpointOrdinals) != 0 {
		return fmt.Errorf("AOQT V10b cache screen receipt must remain research-only probe fail-closed evidence")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim || r.FreeOpenUseClaim {
		return fmt.Errorf("AOQT V10b cache screen claims and heldout gate must be false")
	}
	if r.Dim != AOQTV10bGradientSVDDim || !reflect.DeepEqual(r.Bits, AOQTV10bCacheScreenBits()) || r.QuantizerSeed != AOQTV10bCacheScreenQuantizerSeed || r.TopK != AOQTV10bCacheScreenTopK || r.PerQueryTopK != AOQTV10bCacheScreenPerQueryTopK || r.RerankPolicy != "none" {
		return fmt.Errorf("AOQT V10b cache screen evaluator config mismatch")
	}
	if r.DenseNDCGAt10MinDelta != -0.0005 || r.Q3NDCGAt10MinDelta != 0.0003 || r.Q3RecallAt100MinDelta != 0 || r.Q5NDCGAt10MinDelta != -0.001 || r.PerQuerySafetyMinDelta != 0 || r.NFCorpusBoundaryMinDelta != 0 {
		return fmt.Errorf("AOQT V10b cache screen gate predicate mismatch")
	}
	for _, item := range []struct{ name, value string }{{"source_file_sha256", r.SourceBinding.SourceFileSHA256}, {"source_sha256", r.SourceBinding.SourceSHA256}, {"train_rows_sha256", r.SourceBinding.TrainRowsSHA256}, {"split_manifest_sha256", r.SourceBinding.SplitManifestSHA256}, {"q3_gradient_sha256", r.SourceBinding.Q3GradientSHA256}, {"protected_gradient_matrix_sha256", r.SourceBinding.ProtectedGradientSHA256}, {"projected_operator_sha256", r.SourceBinding.ProjectedOperatorSHA256}, {"parameter_schedule_sha256", r.SourceBinding.ParameterScheduleSHA256}, {"endpoint_hash_chain_tail_sha256", r.EndpointHashChainTail}} {
		if err := validateAOQTSHA256(item.value, "AOQT V10b cache screen "+item.name); err != nil {
			return err
		}
	}
	if strings.TrimSpace(r.SourceBinding.FoldID) == "" || len(r.CaseBindings) == 0 || len(r.Endpoints) != AOQTV10bGradientSVDCandidateCap || r.ObjectiveEvaluationCount != len(r.Endpoints) || !r.FullVerificationCompleted {
		return fmt.Errorf("AOQT V10b cache screen receipt binding/counts incomplete")
	}
	for _, c := range r.CaseBindings {
		if strings.TrimSpace(c.Domain) == "" || strings.TrimSpace(c.Dataset) == "" || c.DocumentCount == 0 || c.QueryCount == 0 || c.RelevantPairCount == 0 || c.ValidationPolicy == "" {
			return fmt.Errorf("AOQT V10b cache screen case binding incomplete")
		}
		for _, item := range []struct{ name, value string }{{"corpus", c.CorpusSHA256}, {"queries", c.QueriesSHA256}, {"qrels", c.QrelsSHA256}, {"doc_vector", c.DocVectorSHA256}, {"query_vector", c.QueryVectorSHA256}, {"workload", c.WorkloadSHA256}, {"baseline_metrics", c.BaselineMetricsSHA256}, {"baseline_per_query", c.BaselinePerQuerySHA256}} {
			if err := validateAOQTSHA256(item.value, "AOQT V10b cache case "+item.name); err != nil {
				return err
			}
		}
	}
	if err := aoqtV10bCacheScreenValidateEndpointSchedule(r.Endpoints); err != nil {
		return err
	}
	scheduleSHA, err := aoqtV10bCacheScreenScheduleSHA256(r.Endpoints)
	if err != nil || scheduleSHA != r.SourceBinding.ParameterScheduleSHA256 {
		return fmt.Errorf("AOQT V10b cache screen parameter schedule hash mismatch")
	}
	chain, tail, err := aoqtV10bCacheScreenEndpointHashChain(r.Endpoints)
	if err != nil || chain != r.EndpointHashChain || tail != r.EndpointHashChainTail {
		return fmt.Errorf("AOQT V10b cache screen endpoint hash chain mismatch")
	}
	metricPassed := 0
	for _, e := range r.Endpoints {
		if e.TransactionEligible || e.Reason != "independent_review_required_fail_closed" {
			return fmt.Errorf("AOQT V10b cache screen endpoint %d must remain transaction-ineligible", e.Ordinal)
		}
		if e.MetricGatePassed {
			metricPassed++
		}
		if err := validateAOQTSHA256(e.ParameterSHA256, "AOQT V10b cache endpoint parameter"); err != nil {
			return err
		}
		if err := validateAOQTSHA256(e.TransformedCacheSHA256, "AOQT V10b cache endpoint transformed cache"); err != nil {
			return err
		}
		if len(e.Cases) != len(r.CaseBindings) {
			return fmt.Errorf("AOQT V10b cache screen endpoint %d case count mismatch", e.Ordinal)
		}
	}
	if metricPassed != r.MetricGatePassedCount {
		return fmt.Errorf("AOQT V10b cache screen metric gate count mismatch")
	}
	return nil
}

func (r AOQTV10bCacheScreenReceipt) ValidateWithSource(source AOQTV10bGradientSVDSource) error {
	if err := r.Validate(); err != nil {
		return err
	}
	sourceSHA, q3SHA, names, protectedSHA, projectedSHA, _, err := aoqtV10bProjectedOperator(source)
	if err != nil {
		return err
	}
	if r.SourceBinding.SourceSHA256 != sourceSHA || r.SourceBinding.FoldID != source.FoldID || r.SourceBinding.TrainRowsSHA256 != source.TrainRowsSHA256 || r.SourceBinding.SplitManifestSHA256 != source.SplitManifestSHA256 || r.SourceBinding.Q3GradientSHA256 != q3SHA || r.SourceBinding.ProtectedGradientSHA256 != protectedSHA || r.SourceBinding.ProjectedOperatorSHA256 != projectedSHA || !reflect.DeepEqual(r.SourceBinding.ProtectedGradientNames, names) {
		return fmt.Errorf("AOQT V10b cache screen source replay mismatch")
	}
	for _, endpoint := range r.Endpoints {
		params, err := DeriveAOQTV10bGradientSVDParameters(source, endpoint.Rank, endpoint.Role, endpoint.Alpha)
		if err != nil {
			return err
		}
		sha, err := params.SHA256()
		if err != nil || sha != endpoint.ParameterSHA256 {
			return fmt.Errorf("AOQT V10b cache screen endpoint %d parameter replay hash mismatch", endpoint.Ordinal)
		}
	}
	return nil
}

type aoqtV10bCacheScreenCaseState struct {
	cfg         AOQTV10bCacheScreenCase
	binding     AOQTV10bCacheScreenCaseBind
	baseline    TurboQuantRetrievalEvalMetrics
	perQuery    map[string]RetrievalEvalQualityMetrics
	perQuerySHA string
}

func aoqtV10bCacheScreenSource(cfg AOQTV10bCacheScreenConfig) (AOQTV10bGradientSVDSource, string, error) {
	if cfg.Source != nil && strings.TrimSpace(cfg.SourcePath) != "" {
		return AOQTV10bGradientSVDSource{}, "", fmt.Errorf("AOQT V10b cache screen accepts source object or source path, not both")
	}
	if cfg.Source != nil {
		if err := cfg.Source.Validate(); err != nil {
			return AOQTV10bGradientSVDSource{}, "", err
		}
		return *cfg.Source, strings.TrimSpace(cfg.SourceFileSHA256), nil
	}
	return ReadAOQTV10bGradientSVDSourceFile(cfg.SourcePath)
}

func aoqtV10bCacheScreenPrepareCase(ctx context.Context, c AOQTV10bCacheScreenCase, workRoot string, index int) (aoqtV10bCacheScreenCaseState, AOQTV10bCacheScreenCaseBind, error) {
	c.Domain = strings.TrimSpace(c.Domain)
	c.Dataset = strings.TrimSpace(c.Dataset)
	if c.Domain == "" || c.Dataset == "" {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("AOQT V10b cache screen case domain and dataset are required")
	}
	if c.CorpusPath == "" || c.QueriesPath == "" || c.QrelsPath == "" || c.DocVectorPath == "" || c.QueryVectorPath == "" {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("AOQT V10b cache screen case paths are required")
	}
	qrels, qrelsSHA, err := readBEIRQrelsWithSHA256(c.QrelsPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	corpus, err := readTurboQuantRetrievalCorpus(RetrievalEvalConfig{CorpusPath: c.CorpusPath}, qrels)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	queries, _, err := readBEIRQueries(c.QueriesPath, qrels, 0)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	docAudit, err := validateAOQTResidualLowRankCacheFile(c.DocVectorPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("validate baseline document cache: %w", err)
	}
	queryAudit, err := validateAOQTResidualLowRankCacheFile(c.QueryVectorPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("validate baseline query cache: %w", err)
	}
	if err := aoqtV10bValidateCacheCoverage(c.DocVectorPath, retrievalIDs(corpus), "document"); err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	if err := aoqtV10bValidateCacheCoverage(c.QueryVectorPath, retrievalIDs(queries), "query"); err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	corpusSHA, err := sha256FileHexAOQTVectorCache(c.CorpusPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	queriesSHA, err := sha256FileHexAOQTVectorCache(c.QueriesPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	for _, check := range []struct{ label, got, want string }{
		{"corpus", corpusSHA, c.ExpectedCorpusSHA256}, {"queries", queriesSHA, c.ExpectedQueriesSHA256}, {"qrels", qrelsSHA, c.ExpectedQrelsSHA256}, {"document cache", docAudit.SHA256, c.ExpectedDocVectorSHA256}, {"query cache", queryAudit.SHA256, c.ExpectedQueryVectorSHA256},
	} {
		if strings.TrimSpace(check.want) != "" && check.got != check.want {
			return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("AOQT V10b cache screen %s sha256 mismatch", check.label)
		}
	}
	workloadSHA := strings.TrimSpace(c.WorkloadSHA256)
	if workloadSHA == "" {
		workloadSHA, err = aoqtCanonicalSHA256(struct {
			Domain, Dataset, CorpusSHA256, QueriesSHA256, QrelsSHA256, DocVectorSHA256, QueryVectorSHA256 string
		}{c.Domain, c.Dataset, corpusSHA, queriesSHA, qrelsSHA, docAudit.SHA256, queryAudit.SHA256})
		if err != nil {
			return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
		}
	} else if err := validateAOQTSHA256(workloadSHA, "AOQT V10b cache screen workload"); err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	perQueryPath := filepath.Join(workRoot, fmt.Sprintf("case-%03d-baseline-per-query.jsonl", index+1))
	baseline, err := EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage(ctx, aoqtV10bCaseEvalConfig(c, c.DocVectorPath, c.QueryVectorPath, perQueryPath), AOQTV10bCacheScreenBits(), []int{}, TurboQuantRerankStorageDense)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, fmt.Errorf("evaluate baseline cache: %w", err)
	}
	baselineSHA, err := aoqtCanonicalSHA256(baseline)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	perQuerySHA, err := sha256FileHexAOQTVectorCache(perQueryPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	perQueryRows, err := aoqtV10bReadPerQueryQualities(perQueryPath)
	if err != nil {
		return aoqtV10bCacheScreenCaseState{}, AOQTV10bCacheScreenCaseBind{}, err
	}
	relevantPairs := 0
	for _, rels := range qrels {
		relevantPairs += len(rels)
	}
	binding := AOQTV10bCacheScreenCaseBind{Domain: c.Domain, Dataset: c.Dataset, CorpusPath: c.CorpusPath, CorpusSHA256: corpusSHA, QueriesPath: c.QueriesPath, QueriesSHA256: queriesSHA, QrelsPath: c.QrelsPath, QrelsSHA256: qrelsSHA, DocVectorPath: c.DocVectorPath, DocVectorSHA256: docAudit.SHA256, QueryVectorPath: c.QueryVectorPath, QueryVectorSHA256: queryAudit.SHA256, WorkloadSHA256: workloadSHA, DocumentCount: len(corpus), QueryCount: len(queries), RelevantPairCount: relevantPairs, BaselineMetricsSHA256: baselineSHA, BaselinePerQuerySHA256: perQuerySHA, ValidationPolicy: "D384_unit_norm_finite_complete_ids_no_duplicates_before_and_after_transform"}
	return aoqtV10bCacheScreenCaseState{cfg: c, binding: binding, baseline: baseline, perQuery: perQueryRows, perQuerySHA: perQuerySHA}, binding, nil
}

func aoqtV10bCacheScreenEvaluateEndpoint(ctx context.Context, workRoot string, ordinal int, params AOQTV10bGradientSVDParameters, cases []aoqtV10bCacheScreenCaseState) (AOQTV10bCacheScreenEndpoint, error) {
	paramSHA, err := params.SHA256()
	if err != nil {
		return AOQTV10bCacheScreenEndpoint{}, err
	}
	endpointDir := filepath.Join(workRoot, fmt.Sprintf("endpoint-%03d", ordinal))
	if err := os.Mkdir(endpointDir, 0o755); err != nil {
		return AOQTV10bCacheScreenEndpoint{}, err
	}
	endpoint := AOQTV10bCacheScreenEndpoint{Ordinal: ordinal, Rank: params.Rank, Role: params.Role, Alpha: params.Alpha, ParameterSHA256: paramSHA, TransactionEligible: false, Reason: "independent_review_required_fail_closed"}
	caseResults := make([]AOQTV10bCacheScreenEndpointCase, 0, len(cases))
	failures := map[string]struct{}{}
	cacheHashes := make([]string, 0, len(cases)*2)
	for i, state := range cases {
		docOut := filepath.Join(endpointDir, fmt.Sprintf("case-%03d-doc.jsonl", i+1))
		queryOut := filepath.Join(endpointDir, fmt.Sprintf("case-%03d-query.jsonl", i+1))
		sidecarOut := filepath.Join(endpointDir, fmt.Sprintf("case-%03d-transform.json", i+1))
		legacy := AOQTResidualLowRankParameters{Schema: AOQTResidualLowRankTransformSchema, Version: AOQTResidualLowRankTransformVersion, Dim: params.Dim, Rank: params.Rank, Role: params.Role, Alpha: params.Alpha, Seed: "v10b:" + params.SourceSHA256, A: params.A, B: params.B}
		binding, err := TransformAOQTResidualLowRankVectorCaches(AOQTResidualLowRankCacheTransformConfig{Parameters: legacy, DocVectorPath: state.cfg.DocVectorPath, QueryVectorPath: state.cfg.QueryVectorPath, OutputDocPath: docOut, OutputQueryPath: queryOut, SidecarPath: sidecarOut, Dataset: state.cfg.Dataset})
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		if err := aoqtV10bValidateCacheCoverage(docOut, aoqtV10bMustRetrievalIDs(state.binding.DocumentCount, state.cfg.CorpusPath, state.cfg.QrelsPath, true), "transformed document"); err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		if err := aoqtV10bValidateCacheCoverage(queryOut, aoqtV10bMustRetrievalIDs(state.binding.QueryCount, state.cfg.QueriesPath, state.cfg.QrelsPath, false), "transformed query"); err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		perQueryPath := filepath.Join(endpointDir, fmt.Sprintf("case-%03d-per-query.jsonl", i+1))
		candidate, err := EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage(ctx, aoqtV10bCaseEvalConfig(state.cfg, docOut, queryOut, perQueryPath), AOQTV10bCacheScreenBits(), []int{}, TurboQuantRerankStorageDense)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, fmt.Errorf("evaluate endpoint %d case %s: %w", ordinal, state.cfg.Domain, err)
		}
		candidateSHA, err := aoqtCanonicalSHA256(candidate)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		perQuerySHA, err := sha256FileHexAOQTVectorCache(perQueryPath)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		candidatePerQuery, err := aoqtV10bReadPerQueryQualities(perQueryPath)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		minPQ, err := aoqtV10bMinPerQueryDelta(state.perQuery, candidatePerQuery)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		q3, q5, err := aoqtV10bRows(candidate, state.baseline)
		if err != nil {
			return AOQTV10bCacheScreenEndpoint{}, err
		}
		result := AOQTV10bCacheScreenEndpointCase{Domain: state.cfg.Domain, Dataset: state.cfg.Dataset, CandidateDocVectorSHA256: binding.Outputs.DocVectorSHA256, CandidateQueryVectorSHA256: binding.Outputs.QueryVectorSHA256, CandidateMetricsSHA256: candidateSHA, CandidatePerQuerySHA256: perQuerySHA, DenseNDCGAt10Delta: candidate.Dense.Quality.NDCGAt10 - state.baseline.Dense.Quality.NDCGAt10, Q3NDCGAt10Delta: q3.NDCGAt10Delta, Q3RecallAt100Delta: q3.RecallAt100Delta, Q5NDCGAt10Delta: q5.NDCGAt10Delta, MinPerQueryNDCGAt10Delta: minPQ}
		if strings.EqualFold(state.cfg.Domain, "nfcorpus") || strings.EqualFold(state.cfg.Dataset, "nfcorpus") {
			result.NFCorpusBoundaryDelta = minPQ
		}
		result.MetricGatePassed = result.DenseNDCGAt10Delta >= -0.0005 && result.Q3NDCGAt10Delta >= 0.0003 && result.Q3RecallAt100Delta >= 0 && result.Q5NDCGAt10Delta >= -0.001 && result.MinPerQueryNDCGAt10Delta >= 0 && (!strings.EqualFold(state.cfg.Domain, "nfcorpus") || result.NFCorpusBoundaryDelta >= 0)
		if !result.MetricGatePassed {
			for _, failure := range aoqtV10bEndpointFailures(result) {
				failures[failure] = struct{}{}
			}
		}
		cacheHashes = append(cacheHashes, binding.Outputs.DocVectorSHA256, binding.Outputs.QueryVectorSHA256)
		caseResults = append(caseResults, result)
	}
	endpoint.Cases = caseResults
	endpoint.MetricGatePassed = len(failures) == 0
	endpoint.FailureReasons = sortedKeysAOQTV10b(failures)
	endpoint.TransformedCacheSHA256, err = aoqtCanonicalSHA256(cacheHashes)
	if err != nil {
		return AOQTV10bCacheScreenEndpoint{}, err
	}
	return endpoint, nil
}

func aoqtV10bCaseEvalConfig(c AOQTV10bCacheScreenCase, docPath, queryPath, perQueryPath string) RetrievalEvalConfig {
	return RetrievalEvalConfig{DatasetName: c.Dataset, CorpusPath: c.CorpusPath, QueriesPath: c.QueriesPath, QrelsPath: c.QrelsPath, DocVectorPath: docPath, QueryVectorPath: queryPath, BackendName: "aoqt-v10b-cache", TopK: AOQTV10bCacheScreenTopK, PerQueryTopK: AOQTV10bCacheScreenPerQueryTopK, PerQueryJSONLPath: perQueryPath, QuantizerSeed: AOQTV10bCacheScreenQuantizerSeed}
}

func aoqtV10bRows(candidate, baseline TurboQuantRetrievalEvalMetrics) (TurboQuantRetrievalBitMetrics, TurboQuantRetrievalBitMetrics, error) {
	rows := map[int]TurboQuantRetrievalBitMetrics{}
	baseRows := map[int]TurboQuantRetrievalBitMetrics{}
	for _, row := range candidate.Rows {
		rows[row.Bits] = row
	}
	for _, row := range baseline.Rows {
		baseRows[row.Bits] = row
	}
	q3, ok3 := rows[3]
	q5, ok5 := rows[5]
	base3, bok3 := baseRows[3]
	base5, bok5 := baseRows[5]
	if !ok3 || !ok5 || !bok3 || !bok5 {
		return q3, q5, fmt.Errorf("AOQT V10b cache screen expected q3 and q5 rows")
	}
	q3.NDCGAt10Delta = q3.Quality.NDCGAt10 - base3.Quality.NDCGAt10
	q3.RecallAt100Delta = q3.Quality.RecallAt100 - base3.Quality.RecallAt100
	q5.NDCGAt10Delta = q5.Quality.NDCGAt10 - base5.Quality.NDCGAt10
	q5.RecallAt100Delta = q5.Quality.RecallAt100 - base5.Quality.RecallAt100
	return q3, q5, nil
}

func aoqtV10bEndpointFailures(result AOQTV10bCacheScreenEndpointCase) []string {
	var failures []string
	if result.DenseNDCGAt10Delta < -0.0005 { failures = append(failures, "dense_ndcg_at_10") }
	if result.Q3NDCGAt10Delta < 0.0003 { failures = append(failures, "q3_ndcg_at_10") }
	if result.Q3RecallAt100Delta < 0 { failures = append(failures, "q3_recall_at_100") }
	if result.Q5NDCGAt10Delta < -0.001 { failures = append(failures, "q5_ndcg_at_10") }
	if result.MinPerQueryNDCGAt10Delta < 0 { failures = append(failures, "per_query_safety") }
	if strings.EqualFold(result.Domain, "nfcorpus") && result.NFCorpusBoundaryDelta < 0 { failures = append(failures, "nfcorpus_boundary") }
	return failures
}

func aoqtV10bValidateCacheCoverage(path string, orderedIDs []string, label string) error {
	audit, err := validateAOQTResidualLowRankCacheFile(path)
	if err != nil {
		return fmt.Errorf("validate %s cache: %w", label, err)
	}
	seen := map[string]struct{}{}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" { continue }
		id, _, err := readAOQTResidualLowRankCacheRecordIdentityAndVector([]byte(line))
		if err != nil { return err }
		seen[id] = struct{}{}
	}
	if err := scanner.Err(); err != nil { return err }
	var missing []string
	for _, id := range orderedIDs {
		if _, ok := seen[id]; !ok { missing = append(missing, id) }
	}
	if len(missing) != 0 {
		sort.Strings(missing)
		return fmt.Errorf("AOQT V10b cache screen %s cache missing %d required ids: %s", label, len(missing), strings.Join(missing, ", "))
	}
	if audit.Count < len(orderedIDs) {
		return fmt.Errorf("AOQT V10b cache screen %s cache count %d below required ids %d", label, audit.Count, len(orderedIDs))
	}
	return nil
}

func aoqtV10bMustRetrievalIDs(_ int, recordPath, qrelsPath string, corpus bool) []string {
	qrels, _ := readBEIRQrels(qrelsPath)
	if corpus {
		records, _ := readTurboQuantRetrievalCorpus(RetrievalEvalConfig{CorpusPath: recordPath}, qrels)
		return retrievalIDs(records)
	}
	records, _, _ := readBEIRQueries(recordPath, qrels, 0)
	return retrievalIDs(records)
}

func aoqtV10bReadPerQueryQualities(path string) (map[string]RetrievalEvalQualityMetrics, error) {
	file, err := os.Open(path)
	if err != nil { return nil, err }
	defer file.Close()
	out := map[string]RetrievalEvalQualityMetrics{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" { continue }
		var row TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if row.Method != "dense" {
			continue
		}
		if row.QueryID == "" {
			return nil, fmt.Errorf("%s:%d: empty query id", path, lineNo)
		}
		if _, ok := out[row.QueryID]; ok {
			return nil, fmt.Errorf("%s:%d: duplicate dense per-query row %q", path, lineNo, row.QueryID)
		}
		out[row.QueryID] = row.Quality
	}
	if err := scanner.Err(); err != nil { return nil, err }
	if len(out) == 0 { return nil, fmt.Errorf("%s: no dense per-query rows", path) }
	return out, nil
}

func aoqtV10bMinPerQueryDelta(baseline, candidate map[string]RetrievalEvalQualityMetrics) (float64, error) {
	min := math.Inf(1)
	for qid, base := range baseline {
		got, ok := candidate[qid]
		if !ok {
			return 0, fmt.Errorf("AOQT V10b cache screen candidate per-query evidence missing %q", qid)
		}
		delta := got.NDCGAt10 - base.NDCGAt10
		if delta < min { min = delta }
	}
	if !isFinite64(min) {
		return 0, fmt.Errorf("AOQT V10b cache screen per-query delta is not finite")
	}
	return min, nil
}

func aoqtV10bCacheScreenValidateEndpointSchedule(endpoints []AOQTV10bCacheScreenEndpoint) error {
	index := 0
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				if index >= len(endpoints) {
					return fmt.Errorf("AOQT V10b cache screen omitted endpoint at canonical schedule position %d", index+1)
				}
				e := endpoints[index]
				if e.Ordinal != index+1 || e.Rank != rank || e.Role != role || e.Alpha != alpha {
					return fmt.Errorf("AOQT V10b cache screen endpoint %d does not match canonical schedule position", index+1)
				}
				index++
			}
		}
	}
	return nil
}

func aoqtV10bCacheScreenScheduleSHA256(endpoints []AOQTV10bCacheScreenEndpoint) (string, error) {
	type item struct {
		Ordinal int `json:"ordinal"`
		Rank int `json:"rank"`
		Role string `json:"role"`
		Alpha float64 `json:"alpha"`
		ParameterSHA256 string `json:"parameter_sha256"`
	}
	items := make([]item, len(endpoints))
	for i, endpoint := range endpoints {
		items[i] = item{endpoint.Ordinal, endpoint.Rank, endpoint.Role, endpoint.Alpha, endpoint.ParameterSHA256}
	}
	return aoqtCanonicalSHA256(struct {
		Version string `json:"version"`
		Endpoints []item `json:"endpoints"`
	}{AOQTV10bCacheScreenScheduleVersion, items})
}

func aoqtV10bCacheScreenEndpointHashChain(endpoints []AOQTV10bCacheScreenEndpoint) (string, string, error) {
	hashes := make([]string, len(endpoints))
	for i, endpoint := range endpoints {
		hash, err := aoqtCanonicalSHA256(endpoint)
		if err != nil { return "", "", err }
		hashes[i] = hash
	}
	data, err := json.Marshal(hashes)
	if err != nil { return "", "", err }
	sum := sha256.Sum256(data)
	return string(data), hex.EncodeToString(sum[:]), nil
}

func sortedKeysAOQTV10b(values map[string]struct{}) []string {
	out := make([]string, 0, len(values))
	for value := range values { out = append(out, value) }
	sort.Strings(out)
	return out
}
