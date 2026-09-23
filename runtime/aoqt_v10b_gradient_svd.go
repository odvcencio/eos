package eosruntime

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
)

const (
	AOQTV10bGradientSVDSchema                 = "eos.aoqt.v10b_gradient_svd_source.v1"
	AOQTV10bGradientSVDParameterSchema        = "eos.aoqt.v10b_gradient_svd_parameters.v1"
	AOQTV10bGradientSVDReceiptSchema          = "eos.aoqt.v10b_gradient_svd_receipt.v1"
	AOQTV10bGradientSVDVersion                = "eos-d384-residual-lowrank-gradient-svd-v10b"
	AOQTV10bGradientSVDDerivationVersion      = "protected-safe-covariance-power-svd-v1"
	AOQTV10bGradientSVDOperatorPolicy         = "per-vector-unit-tangent-descent-second-moment-v1"
	AOQTV10bGradientSVDScheduleVersion        = "aoqt-v10b-gradient-svd-rank-role-alpha-canonical-24-v1"
	AOQTV10bGradientSVDEvaluatorStatusBlocked = "cache_native_evaluator_replay_required_fail_closed_v1"
	AOQTV10bGradientSVDDim                    = AOQTResidualLowRankDim
	AOQTV10bGradientSVDCandidateCap           = AOQTResidualLowRankCandidateCap
	AOQTV10bCanonicalV7R6ReceiptFileSHA256    = "41857a7d14ca87c589f3a51aab097a2fd5f6911cb8eb30ed1728ece1b792f63d"
	AOQTV10bCanonicalV8ReceiptFileSHA256      = "4bb0cdf9e756340b3c308336313034104a188bb16dc7b32394ceeaa51769d07b"
	AOQTV10bCanonicalV9ReceiptFileSHA256      = "710176938e43924d424c284fdc42b798d6e2a25cc4a5cc4409f079071eef40cc"
)

type AOQTV10bNamedGradient struct {
	Name string    `json:"name"`
	Grad []float64 `json:"grad"`
}

type AOQTV10bGradientSVDSource struct {
	Schema              string                  `json:"schema"`
	Version             string                  `json:"version"`
	Dim                 int                     `json:"dim"`
	FoldID              string                  `json:"fold_id"`
	TrainRowsSHA256     string                  `json:"train_rows_sha256"`
	SplitManifestSHA256 string                  `json:"split_manifest_sha256"`
	OperatorPolicy      string                  `json:"operator_policy"`
	Q3Gradient          []float64               `json:"q3_gradient"`
	ProtectedGradients  []AOQTV10bNamedGradient `json:"protected_gradients"`
	Q3TargetCovariance  [][]float64             `json:"q3_target_covariance"`
}

type AOQTV10bGradientSVDParameters struct {
	Schema                  string      `json:"schema"`
	Version                 string      `json:"version"`
	DerivationVersion       string      `json:"derivation_version"`
	Dim                     int         `json:"dim"`
	Rank                    int         `json:"rank"`
	Role                    string      `json:"role"`
	Alpha                   float64     `json:"alpha"`
	SourceSHA256            string      `json:"source_sha256"`
	Q3GradientSHA256        string      `json:"q3_gradient_sha256"`
	ProtectedSHA256         string      `json:"protected_gradient_matrix_sha256"`
	ProjectedOperatorSHA256 string      `json:"projected_operator_sha256"`
	Eigenvalues             []float64   `json:"eigenvalues"`
	A                       [][]float32 `json:"a"`
	B                       [][]float32 `json:"b"`
}

type AOQTV10bGradientSVDEndpoint struct {
	Ordinal             int     `json:"ordinal"`
	Rank                int     `json:"rank"`
	Role                string  `json:"role"`
	Alpha               float64 `json:"alpha"`
	ParameterSHA256     string  `json:"parameter_sha256"`
	EvaluatorStatus     string  `json:"evaluator_status"`
	TransactionEligible bool    `json:"transaction_eligible"`
	Reason              string  `json:"reason"`
	RankOrder           int     `json:"rank_order"`
}

type AOQTV10bGradientSVDBinding struct {
	SourcePath                     string `json:"source_path,omitempty"`
	SourceFileSHA256               string `json:"source_file_sha256"`
	SourceSHA256                   string `json:"source_sha256"`
	FoldID                         string `json:"fold_id"`
	TrainRowsSHA256                string `json:"train_rows_sha256"`
	SplitManifestSHA256            string `json:"split_manifest_sha256"`
	CanonicalV7R6ReceiptPath       string `json:"canonical_v7_r6_receipt_path,omitempty"`
	CanonicalV7R6ReceiptFileSHA256 string `json:"canonical_v7_r6_receipt_file_sha256"`
	CanonicalV8ReceiptPath         string `json:"canonical_v8_receipt_path,omitempty"`
	CanonicalV8ReceiptFileSHA256   string `json:"canonical_v8_receipt_file_sha256"`
	CanonicalV9ReceiptPath         string `json:"canonical_v9_receipt_path,omitempty"`
	CanonicalV9ReceiptFileSHA256   string `json:"canonical_v9_receipt_file_sha256"`
}

type AOQTV10bGradientSVDReceipt struct {
	Schema                        string                        `json:"schema"`
	Version                       string                        `json:"version"`
	DerivationVersion             string                        `json:"derivation_version"`
	ScheduleVersion               string                        `json:"schedule_version"`
	Required                      bool                          `json:"required"`
	ProbeOnly                     bool                          `json:"probe_only"`
	DryRunOnly                    bool                          `json:"dry_run_only"`
	Passed                        bool                          `json:"passed"`
	FailureReason                 string                        `json:"failure_reason"`
	Dim                           int                           `json:"dim"`
	Ranks                         []int                         `json:"ranks"`
	Roles                         []string                      `json:"roles"`
	Magnitudes                    []float64                     `json:"magnitudes"`
	CandidateCap                  int                           `json:"candidate_cap"`
	ObjectiveEvaluationCount      int                           `json:"objective_evaluation_count"`
	FullVerificationCompleted     bool                          `json:"full_verification_completed"`
	OperatorPolicy                string                        `json:"operator_policy"`
	Q3GradientSHA256              string                        `json:"q3_gradient_sha256"`
	ProtectedGradientNames        []string                      `json:"protected_gradient_names"`
	ProtectedGradientMatrixSHA256 string                        `json:"protected_gradient_matrix_sha256"`
	ProjectedOperatorSHA256       string                        `json:"projected_operator_sha256"`
	ParameterScheduleSHA256       string                        `json:"parameter_schedule_sha256"`
	EndpointHashChain             string                        `json:"endpoint_hash_chain"`
	EndpointHashChainTailSHA256   string                        `json:"endpoint_hash_chain_tail_sha256"`
	FeasibleEndpointCount         int                           `json:"feasible_endpoint_count"`
	SelectedEndpointOrdinals      []int                         `json:"selected_endpoint_ordinals"`
	Binding                       AOQTV10bGradientSVDBinding    `json:"binding"`
	Endpoints                     []AOQTV10bGradientSVDEndpoint `json:"endpoints"`
	QualityClaim                  bool                          `json:"quality_claim"`
	ReleaseClaim                  bool                          `json:"release_claim"`
	OfficialClaim                 bool                          `json:"official_claim"`
	OfficialHeldoutGate           bool                          `json:"official_heldout_gate"`
	CommercialClaim               bool                          `json:"commercial_claim"`
	FreeOpenUseClaim              bool                          `json:"free_open_use_claim"`
}

type AOQTV10bGradientSVDRunConfig struct {
	SourcePath                     string
	OutputReceiptJSONPath          string
	CanonicalV7R6ReceiptPath       string
	CanonicalV7R6ReceiptFileSHA256 string
	CanonicalV8ReceiptPath         string
	CanonicalV8ReceiptFileSHA256   string
	CanonicalV9ReceiptPath         string
	CanonicalV9ReceiptFileSHA256   string
}

type AOQTV10bGradientSVDRunResult struct {
	Receipt         AOQTV10bGradientSVDReceipt
	ReceiptSHA256   string
	ReceiptJSONPath string
}

func BuildAOQTV10bGradientSVDSourceFromCalibrationSet(set AOQTSidecarCalibrationSet, objective AOQTSidecarVectorObjective, foldID, trainRowsSHA256, splitManifestSHA256 string) (AOQTV10bGradientSVDSource, error) {
	if err := set.Validate(); err != nil {
		return AOQTV10bGradientSVDSource{}, err
	}
	if strings.TrimSpace(foldID) == "" {
		return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b fold_id is required")
	}
	if err := validateAOQTSHA256(trainRowsSHA256, "AOQT V10b train rows sha256"); err != nil {
		return AOQTV10bGradientSVDSource{}, err
	}
	if err := validateAOQTSHA256(splitManifestSHA256, "AOQT V10b split manifest sha256"); err != nil {
		return AOQTV10bGradientSVDSource{}, err
	}
	provider, ok := objective.(AOQTSidecarPreparedIPObjectiveContractProvider)
	if !ok {
		return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b source extraction requires the concrete prepared-IP objective")
	}
	switch objective.(type) {
	case AOQTSidecarPreparedIPObjective, *AOQTSidecarPreparedIPObjective:
	default:
		return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b source extraction requires the concrete prepared-IP objective")
	}
	if got := provider.AOQTPreparedIPObjectiveConfig().ObjectiveContract(set.Manifest.ObjectiveContract.WeightSums); !reflect.DeepEqual(got, set.Manifest.ObjectiveContract) {
		return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b prepared-IP objective contract does not match calibration manifest")
	}
	q3Gradient, q3Covariance, err := aoqtV10bComponentVectorGradientAndCovariance(set.Rows, objective, "q3_gain")
	if err != nil {
		return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b q3 vector gradient: %w", err)
	}
	protectedComponents := aoqtActiveProtectedComponentNames(set.Manifest.ObjectiveContract.WeightSums)
	protected := make([]AOQTV10bNamedGradient, 0, len(protectedComponents))
	for _, component := range protectedComponents {
		gradient, _, err := aoqtV10bComponentVectorGradientAndCovariance(set.Rows, objective, component)
		if err != nil {
			return AOQTV10bGradientSVDSource{}, fmt.Errorf("AOQT V10b protected vector gradient %s: %w", component, err)
		}
		protected = append(protected, AOQTV10bNamedGradient{Name: component, Grad: gradient})
	}
	source := AOQTV10bGradientSVDSource{Schema: AOQTV10bGradientSVDSchema, Version: AOQTV10bGradientSVDVersion, Dim: AOQTV10bGradientSVDDim, FoldID: foldID, TrainRowsSHA256: trainRowsSHA256, SplitManifestSHA256: splitManifestSHA256, OperatorPolicy: AOQTV10bGradientSVDOperatorPolicy, Q3Gradient: q3Gradient, ProtectedGradients: protected, Q3TargetCovariance: q3Covariance}
	if err := source.Validate(); err != nil {
		return AOQTV10bGradientSVDSource{}, err
	}
	return source, nil
}

func DeriveAOQTV10bGradientSVDParameters(source AOQTV10bGradientSVDSource, rank int, role string, alpha float64) (AOQTV10bGradientSVDParameters, error) {
	sourceSHA, q3SHA, protectedNames, protectedSHA, projectedSHA, matrix, err := aoqtV10bProjectedOperator(source)
	if err != nil {
		return AOQTV10bGradientSVDParameters{}, err
	}
	p := AOQTV10bGradientSVDParameters{Schema: AOQTV10bGradientSVDParameterSchema, Version: AOQTV10bGradientSVDVersion, DerivationVersion: AOQTV10bGradientSVDDerivationVersion, Dim: AOQTV10bGradientSVDDim, Rank: rank, Role: role, Alpha: alpha, SourceSHA256: sourceSHA, Q3GradientSHA256: q3SHA, ProtectedSHA256: protectedSHA, ProjectedOperatorSHA256: projectedSHA}
	_ = protectedNames
	if err := p.validateHeader(); err != nil {
		return AOQTV10bGradientSVDParameters{}, err
	}
	vecs, vals, err := aoqtV10bTopEigenvectors(matrix, rank, sourceSHA)
	if err != nil {
		return AOQTV10bGradientSVDParameters{}, err
	}
	p.Eigenvalues = vals
	p.A = make([][]float32, rank)
	p.B = make([][]float32, AOQTV10bGradientSVDDim)
	for k := 0; k < rank; k++ {
		p.A[k] = make([]float32, AOQTV10bGradientSVDDim)
		for d := 0; d < AOQTV10bGradientSVDDim; d++ {
			p.A[k][d] = float32(vecs[k][d])
		}
	}
	for d := 0; d < AOQTV10bGradientSVDDim; d++ {
		p.B[d] = make([]float32, rank)
		for k := 0; k < rank; k++ {
			p.B[d][k] = float32(vecs[k][d] * math.Sqrt(math.Max(vals[k], 0)))
		}
	}
	if err := p.Validate(); err != nil {
		return AOQTV10bGradientSVDParameters{}, err
	}
	return p, nil
}

func (p AOQTV10bGradientSVDParameters) Validate() error {
	if err := p.validateHeader(); err != nil {
		return err
	}
	for _, item := range []struct{ name, value string }{{"source_sha256", p.SourceSHA256}, {"q3_gradient_sha256", p.Q3GradientSHA256}, {"protected_gradient_matrix_sha256", p.ProtectedSHA256}, {"projected_operator_sha256", p.ProjectedOperatorSHA256}} {
		if err := validateAOQTSHA256(item.value, "AOQT V10b "+item.name); err != nil {
			return err
		}
	}
	if len(p.Eigenvalues) != p.Rank || len(p.A) != p.Rank || len(p.B) != p.Dim {
		return fmt.Errorf("AOQT V10b parameter shape does not match rank/dim")
	}
	for k := range p.A {
		if len(p.A[k]) != p.Dim {
			return fmt.Errorf("AOQT V10b A[%d] dim = %d, want %d", k, len(p.A[k]), p.Dim)
		}
		if err := validateAOQTFiniteVector(p.A[k], fmt.Sprintf("AOQT V10b A[%d]", k)); err != nil {
			return err
		}
		if math.Abs(aoqtResidualNorm(p.A[k])-1) > 2e-5 {
			return fmt.Errorf("AOQT V10b A[%d] must be unit-normalized", k)
		}
		if !isFinite64(p.Eigenvalues[k]) || p.Eigenvalues[k] <= 0 {
			return fmt.Errorf("AOQT V10b eigenvalue[%d] must be finite and positive", k)
		}
	}
	for d := range p.B {
		if len(p.B[d]) != p.Rank {
			return fmt.Errorf("AOQT V10b B[%d] rank dim = %d, want %d", d, len(p.B[d]), p.Rank)
		}
		if err := validateAOQTFiniteVector(p.B[d], fmt.Sprintf("AOQT V10b B[%d]", d)); err != nil {
			return err
		}
	}
	return nil
}

func (p AOQTV10bGradientSVDParameters) validateHeader() error {
	if p.Schema != AOQTV10bGradientSVDParameterSchema || p.Version != AOQTV10bGradientSVDVersion || p.DerivationVersion != AOQTV10bGradientSVDDerivationVersion {
		return fmt.Errorf("AOQT V10b parameter schema/version is unsupported")
	}
	if p.Dim != AOQTV10bGradientSVDDim {
		return fmt.Errorf("AOQT V10b dim = %d, want %d", p.Dim, AOQTV10bGradientSVDDim)
	}
	if !containsIntAOQTResidual(p.Rank, AOQTResidualLowRankRanks()) || !containsStringAOQTResidual(p.Role, AOQTResidualLowRankRoles()) || !containsFloat64AOQTResidual(p.Alpha, AOQTResidualLowRankMagnitudes()) {
		return fmt.Errorf("AOQT V10b rank/role/alpha is noncanonical")
	}
	return nil
}

func (p AOQTV10bGradientSVDParameters) SHA256() (string, error) { return aoqtCanonicalSHA256(p) }

func (p AOQTV10bGradientSVDParameters) Apply(vector []float32) ([]float32, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	legacy := AOQTResidualLowRankParameters{Schema: AOQTResidualLowRankTransformSchema, Version: AOQTResidualLowRankTransformVersion, Dim: p.Dim, Rank: p.Rank, Role: p.Role, Alpha: p.Alpha, Seed: "v10b:" + p.SourceSHA256, A: p.A, B: p.B}
	return legacy.Apply(vector)
}

func RunAOQTV10bGradientSVDScreen(cfg AOQTV10bGradientSVDRunConfig) (AOQTV10bGradientSVDRunResult, error) {
	var result AOQTV10bGradientSVDRunResult
	source, sourceFileSHA, err := ReadAOQTV10bGradientSVDSourceFile(cfg.SourcePath)
	if err != nil {
		return result, err
	}
	sourceSHA, err := aoqtCanonicalSHA256(source)
	if err != nil {
		return result, err
	}
	binding := AOQTV10bGradientSVDBinding{SourcePath: cfg.SourcePath, SourceFileSHA256: sourceFileSHA, SourceSHA256: sourceSHA, FoldID: source.FoldID, TrainRowsSHA256: source.TrainRowsSHA256, SplitManifestSHA256: source.SplitManifestSHA256, CanonicalV7R6ReceiptPath: cfg.CanonicalV7R6ReceiptPath, CanonicalV8ReceiptPath: cfg.CanonicalV8ReceiptPath, CanonicalV9ReceiptPath: cfg.CanonicalV9ReceiptPath}
	if binding.CanonicalV7R6ReceiptFileSHA256, err = hashAOQTResidualRequiredFile(cfg.CanonicalV7R6ReceiptPath, cfg.CanonicalV7R6ReceiptFileSHA256, "canonical V7-r6 receipt"); err != nil {
		return result, err
	}
	if binding.CanonicalV8ReceiptFileSHA256, err = hashAOQTResidualRequiredFile(cfg.CanonicalV8ReceiptPath, cfg.CanonicalV8ReceiptFileSHA256, "canonical V8 receipt"); err != nil {
		return result, err
	}
	if binding.CanonicalV9ReceiptFileSHA256, err = hashAOQTResidualRequiredFile(cfg.CanonicalV9ReceiptPath, cfg.CanonicalV9ReceiptFileSHA256, "canonical V9 receipt"); err != nil {
		return result, err
	}
	receipt, err := BuildAOQTV10bGradientSVDReceipt(binding, source)
	if err != nil {
		return result, err
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) != "" {
		if err := WriteAOQTV10bGradientSVDReceipt(cfg.OutputReceiptJSONPath, receipt); err != nil {
			return result, err
		}
		result.ReceiptJSONPath = cfg.OutputReceiptJSONPath
	}
	sha, err := receipt.SHA256()
	if err != nil {
		return result, err
	}
	result.Receipt = receipt
	result.ReceiptSHA256 = sha
	return result, nil
}

func ReadAOQTV10bGradientSVDSourceFile(path string) (AOQTV10bGradientSVDSource, string, error) {
	if strings.TrimSpace(path) == "" {
		return AOQTV10bGradientSVDSource{}, "", fmt.Errorf("AOQT V10b source path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return AOQTV10bGradientSVDSource{}, "", err
	}
	var source AOQTV10bGradientSVDSource
	if err := json.Unmarshal(data, &source); err != nil {
		return AOQTV10bGradientSVDSource{}, "", err
	}
	sha := sha256HexAOQTResidual(data)
	if err := source.Validate(); err != nil {
		return AOQTV10bGradientSVDSource{}, "", err
	}
	return source, sha, nil
}

func BuildAOQTV10bGradientSVDReceipt(binding AOQTV10bGradientSVDBinding, source AOQTV10bGradientSVDSource) (AOQTV10bGradientSVDReceipt, error) {
	sourceSHA, q3SHA, names, protectedSHA, projectedSHA, _, err := aoqtV10bProjectedOperator(source)
	if err != nil {
		return AOQTV10bGradientSVDReceipt{}, err
	}
	if binding.SourceFileSHA256 == "" {
		binding.SourceFileSHA256 = sourceSHA
	}
	if binding.SourceSHA256 == "" {
		binding.SourceSHA256 = sourceSHA
	}
	endpoints := make([]AOQTV10bGradientSVDEndpoint, 0, AOQTV10bGradientSVDCandidateCap)
	ordinal := 1
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				params, err := DeriveAOQTV10bGradientSVDParameters(source, rank, role, alpha)
				if err != nil {
					return AOQTV10bGradientSVDReceipt{}, err
				}
				paramSHA, err := params.SHA256()
				if err != nil {
					return AOQTV10bGradientSVDReceipt{}, err
				}
				endpoints = append(endpoints, AOQTV10bGradientSVDEndpoint{Ordinal: ordinal, Rank: rank, Role: role, Alpha: alpha, ParameterSHA256: paramSHA, EvaluatorStatus: AOQTV10bGradientSVDEvaluatorStatusBlocked, TransactionEligible: false, Reason: "cache_native_evaluator_not_authorized", RankOrder: ordinal})
				ordinal++
			}
		}
	}
	scheduleSHA, err := aoqtV10bScheduleSHA256(endpoints)
	if err != nil {
		return AOQTV10bGradientSVDReceipt{}, err
	}
	chain, tail, err := aoqtV10bEndpointHashChain(endpoints)
	if err != nil {
		return AOQTV10bGradientSVDReceipt{}, err
	}
	receipt := AOQTV10bGradientSVDReceipt{Schema: AOQTV10bGradientSVDReceiptSchema, Version: AOQTV10bGradientSVDVersion, DerivationVersion: AOQTV10bGradientSVDDerivationVersion, ScheduleVersion: AOQTV10bGradientSVDScheduleVersion, Required: true, ProbeOnly: true, DryRunOnly: true, Passed: false, FailureReason: "cache-native evaluator replay not authorized; V10b parameters are train-derived foundation only", Dim: AOQTV10bGradientSVDDim, Ranks: AOQTResidualLowRankRanks(), Roles: AOQTResidualLowRankRoles(), Magnitudes: AOQTResidualLowRankMagnitudes(), CandidateCap: AOQTV10bGradientSVDCandidateCap, ObjectiveEvaluationCount: 0, FullVerificationCompleted: false, OperatorPolicy: AOQTV10bGradientSVDOperatorPolicy, Q3GradientSHA256: q3SHA, ProtectedGradientNames: names, ProtectedGradientMatrixSHA256: protectedSHA, ProjectedOperatorSHA256: projectedSHA, ParameterScheduleSHA256: scheduleSHA, EndpointHashChain: chain, EndpointHashChainTailSHA256: tail, FeasibleEndpointCount: 0, Binding: binding, Endpoints: endpoints}
	if err := receipt.ValidateWithSource(source); err != nil {
		return AOQTV10bGradientSVDReceipt{}, err
	}
	return receipt, nil
}

func WriteAOQTV10bGradientSVDReceipt(path string, receipt AOQTV10bGradientSVDReceipt) error {
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

func (s AOQTV10bGradientSVDSource) Validate() error {
	if s.Schema != AOQTV10bGradientSVDSchema || s.Version != AOQTV10bGradientSVDVersion {
		return fmt.Errorf("AOQT V10b source schema/version is unsupported")
	}
	if s.Dim != AOQTV10bGradientSVDDim {
		return fmt.Errorf("AOQT V10b source dim = %d, want %d", s.Dim, AOQTV10bGradientSVDDim)
	}
	if strings.TrimSpace(s.FoldID) == "" {
		return fmt.Errorf("AOQT V10b source fold_id is required")
	}
	if s.OperatorPolicy != AOQTV10bGradientSVDOperatorPolicy {
		return fmt.Errorf("AOQT V10b source operator_policy is unsupported")
	}
	if err := validateAOQTSHA256(s.TrainRowsSHA256, "AOQT V10b train_rows_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(s.SplitManifestSHA256, "AOQT V10b split_manifest_sha256"); err != nil {
		return err
	}
	if len(s.Q3Gradient) != s.Dim || !aoqtV9AllFinite(s.Q3Gradient) || aoqtV9MaxAbs(s.Q3Gradient) == 0 {
		return fmt.Errorf("AOQT V10b q3 gradient must be D384 finite nonzero")
	}
	seen := map[string]bool{}
	for i, g := range s.ProtectedGradients {
		if strings.TrimSpace(g.Name) == "" || seen[g.Name] {
			return fmt.Errorf("AOQT V10b protected gradient[%d] has empty or duplicate name", i)
		}
		seen[g.Name] = true
		if len(g.Grad) != s.Dim || !aoqtV9AllFinite(g.Grad) {
			return fmt.Errorf("AOQT V10b protected gradient[%d] must be D384 finite", i)
		}
	}
	if len(s.Q3TargetCovariance) != s.Dim {
		return fmt.Errorf("AOQT V10b q3 target covariance rows = %d, want %d", len(s.Q3TargetCovariance), s.Dim)
	}
	for i, row := range s.Q3TargetCovariance {
		if len(row) != s.Dim || !aoqtV9AllFinite(row) {
			return fmt.Errorf("AOQT V10b q3 target covariance row[%d] must be D384 finite", i)
		}
		for j := 0; j < i; j++ {
			if math.Abs(row[j]-s.Q3TargetCovariance[j][i]) > 1e-8 {
				return fmt.Errorf("AOQT V10b q3 target covariance must be symmetric")
			}
		}
	}
	return nil
}

func aoqtV10bComponentVectorGradientAndCovariance(rows []AOQTSidecarCalibrationRow, objective AOQTSidecarVectorObjective, component string) ([]float64, [][]float64, error) {
	total := make([]float64, AOQTV10bGradientSVDDim)
	covariance := make([][]float64, AOQTV10bGradientSVDDim)
	for i := range covariance {
		covariance[i] = make([]float64, AOQTV10bGradientSVDDim)
	}
	for _, row := range rows {
		masked := row
		masked.Weights = aoqtOnlyComponentWeights(row.Weights, component)
		result, err := objective.EvaluateAOQT(AOQTSidecarObjectiveInput{Row: aoqtObjectiveRowView(masked), Query: append([]float32(nil), row.QueryVector...), Candidates: cloneAOQTVectors(row.CandidateVectors)})
		if err != nil {
			return nil, nil, err
		}
		if !isFinite32(result.Loss) {
			return nil, nil, fmt.Errorf("AOQT V10b component %s loss must be finite", component)
		}
		if err := validateAOQTObjectiveActivation(result.Activation, "AOQT V10b objective activation"); err != nil {
			return nil, nil, err
		}
		if err := validateAOQTObjectiveComponents(result.Components, "AOQT V10b objective components"); err != nil {
			return nil, nil, err
		}
		if len(result.QueryGrad) != AOQTV10bGradientSVDDim || len(result.CandidateGrads) != len(row.CandidateVectors) {
			return nil, nil, fmt.Errorf("AOQT V10b component %s vector gradient shape mismatch", component)
		}
		queryTangent, err := aoqtV10bTangentDescent(row.QueryVector, result.QueryGrad, "query grad")
		if err != nil {
			return nil, nil, err
		}
		aoqtV10bAccumulateVector(total, queryTangent)
		aoqtV10bAccumulateOuter(covariance, queryTangent)
		for ci, grad := range result.CandidateGrads {
			if len(grad) != AOQTV10bGradientSVDDim {
				return nil, nil, fmt.Errorf("AOQT V10b candidate grad[%d] dim = %d, want %d", ci, len(grad), AOQTV10bGradientSVDDim)
			}
			tangent, err := aoqtV10bTangentDescent(row.CandidateVectors[ci], grad, fmt.Sprintf("candidate grad[%d]", ci))
			if err != nil {
				return nil, nil, err
			}
			aoqtV10bAccumulateVector(total, tangent)
			aoqtV10bAccumulateOuter(covariance, tangent)
		}
	}
	for i := range covariance {
		for j := 0; j < i; j++ {
			covariance[j][i] = covariance[i][j]
		}
	}
	if !aoqtV9AllFinite(total) || aoqtV9MaxAbs(total) == 0 {
		return nil, nil, fmt.Errorf("AOQT V10b component %s produced zero/nonfinite vector gradient", component)
	}
	return total, covariance, nil
}

func aoqtV10bTangentDescent(vector, grad []float32, label string) ([]float64, error) {
	if len(vector) != AOQTV10bGradientSVDDim || len(grad) != AOQTV10bGradientSVDDim {
		return nil, fmt.Errorf("AOQT V10b %s tangent dim mismatch", label)
	}
	if err := validateAOQTFiniteVector(vector, "AOQT V10b "+label+" vector"); err != nil {
		return nil, err
	}
	if err := validateAOQTFiniteVector(grad, "AOQT V10b "+label); err != nil {
		return nil, err
	}
	norm := aoqtResidualNorm(vector)
	if math.Abs(norm-1) > float64(AOQTSidecarUnitVectorTolerance) {
		return nil, fmt.Errorf("AOQT V10b %s vector must be unit-normalized", label)
	}
	dot := 0.0
	for i := range vector {
		dot += float64(vector[i]) * float64(grad[i])
	}
	out := make([]float64, AOQTV10bGradientSVDDim)
	for i := range out {
		out[i] = -(float64(grad[i]) - dot*float64(vector[i]))
		if !aoqtV9Finite(out[i]) {
			return nil, fmt.Errorf("AOQT V10b %s tangent[%d] must be finite", label, i)
		}
	}
	return out, nil
}

func aoqtV10bAccumulateVector(dst, src []float64) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func aoqtV10bAccumulateOuter(dst [][]float64, src []float64) {
	for i := range src {
		for j := 0; j <= i; j++ {
			dst[i][j] += src[i] * src[j]
		}
	}
}

func (r AOQTV10bGradientSVDReceipt) SHA256() (string, error) { return aoqtCanonicalSHA256(r) }

func (r AOQTV10bGradientSVDReceipt) Validate() error {
	if r.Schema != AOQTV10bGradientSVDReceiptSchema || r.Version != AOQTV10bGradientSVDVersion || r.DerivationVersion != AOQTV10bGradientSVDDerivationVersion || r.ScheduleVersion != AOQTV10bGradientSVDScheduleVersion {
		return fmt.Errorf("AOQT V10b receipt schema/version is unsupported")
	}
	if !r.Required || !r.ProbeOnly || !r.DryRunOnly || r.Passed || r.FullVerificationCompleted || r.ObjectiveEvaluationCount != 0 || r.FeasibleEndpointCount != 0 || len(r.SelectedEndpointOrdinals) != 0 {
		return fmt.Errorf("AOQT V10b receipt must remain required probe-only dry-run fail-closed evidence")
	}
	if r.QualityClaim || r.ReleaseClaim || r.OfficialClaim || r.OfficialHeldoutGate || r.CommercialClaim || r.FreeOpenUseClaim {
		return fmt.Errorf("AOQT V10b receipt claims and heldout gate must be false")
	}
	if strings.TrimSpace(r.FailureReason) == "" || r.Dim != AOQTV10bGradientSVDDim || r.CandidateCap != AOQTV10bGradientSVDCandidateCap || r.OperatorPolicy != AOQTV10bGradientSVDOperatorPolicy {
		return fmt.Errorf("AOQT V10b receipt binding is incomplete")
	}
	if !reflect.DeepEqual(r.Ranks, AOQTResidualLowRankRanks()) || !reflect.DeepEqual(r.Roles, AOQTResidualLowRankRoles()) || !reflect.DeepEqual(r.Magnitudes, AOQTResidualLowRankMagnitudes()) {
		return fmt.Errorf("AOQT V10b canonical rank/role/magnitude schedule mismatch")
	}
	for _, item := range []struct{ name, value string }{{"source_file_sha256", r.Binding.SourceFileSHA256}, {"source_sha256", r.Binding.SourceSHA256}, {"train_rows_sha256", r.Binding.TrainRowsSHA256}, {"split_manifest_sha256", r.Binding.SplitManifestSHA256}, {"canonical_v7_r6_receipt_file_sha256", r.Binding.CanonicalV7R6ReceiptFileSHA256}, {"canonical_v8_receipt_file_sha256", r.Binding.CanonicalV8ReceiptFileSHA256}, {"canonical_v9_receipt_file_sha256", r.Binding.CanonicalV9ReceiptFileSHA256}, {"q3_gradient_sha256", r.Q3GradientSHA256}, {"protected_gradient_matrix_sha256", r.ProtectedGradientMatrixSHA256}, {"projected_operator_sha256", r.ProjectedOperatorSHA256}, {"parameter_schedule_sha256", r.ParameterScheduleSHA256}, {"endpoint_hash_chain_tail_sha256", r.EndpointHashChainTailSHA256}} {
		if err := validateAOQTSHA256(item.value, "AOQT V10b "+item.name); err != nil {
			return err
		}
	}
	if strings.TrimSpace(r.Binding.FoldID) == "" {
		return fmt.Errorf("AOQT V10b fold binding is required")
	}
	if r.Binding.CanonicalV7R6ReceiptFileSHA256 != AOQTV10bCanonicalV7R6ReceiptFileSHA256 ||
		r.Binding.CanonicalV8ReceiptFileSHA256 != AOQTV10bCanonicalV8ReceiptFileSHA256 ||
		r.Binding.CanonicalV9ReceiptFileSHA256 != AOQTV10bCanonicalV9ReceiptFileSHA256 {
		return fmt.Errorf("AOQT V10b receipt must bind exact terminal V7-r6/V8/V9 receipt hashes")
	}
	if len(r.Endpoints) != AOQTV10bGradientSVDCandidateCap {
		return fmt.Errorf("AOQT V10b endpoint count = %d, want %d", len(r.Endpoints), AOQTV10bGradientSVDCandidateCap)
	}
	if err := aoqtV10bValidateEndpointSchedule(r.Endpoints); err != nil {
		return err
	}
	scheduleSHA, err := aoqtV10bScheduleSHA256(r.Endpoints)
	if err != nil || scheduleSHA != r.ParameterScheduleSHA256 {
		return fmt.Errorf("AOQT V10b parameter schedule hash mismatch")
	}
	chain, tail, err := aoqtV10bEndpointHashChain(r.Endpoints)
	if err != nil || chain != r.EndpointHashChain || tail != r.EndpointHashChainTailSHA256 {
		return fmt.Errorf("AOQT V10b endpoint hash chain mismatch")
	}
	for _, endpoint := range r.Endpoints {
		if endpoint.EvaluatorStatus != AOQTV10bGradientSVDEvaluatorStatusBlocked || endpoint.TransactionEligible || endpoint.Reason != "cache_native_evaluator_not_authorized" || endpoint.RankOrder != endpoint.Ordinal {
			return fmt.Errorf("AOQT V10b endpoint %d must remain fail-closed without evaluator metrics", endpoint.Ordinal)
		}
	}
	return nil
}

func (r AOQTV10bGradientSVDReceipt) ValidateWithSource(source AOQTV10bGradientSVDSource) error {
	if err := r.Validate(); err != nil {
		return err
	}
	sourceSHA, q3SHA, names, protectedSHA, projectedSHA, _, err := aoqtV10bProjectedOperator(source)
	if err != nil {
		return err
	}
	if r.Binding.SourceSHA256 != sourceSHA || r.Binding.FoldID != source.FoldID || r.Binding.TrainRowsSHA256 != source.TrainRowsSHA256 || r.Binding.SplitManifestSHA256 != source.SplitManifestSHA256 || r.OperatorPolicy != source.OperatorPolicy || r.Q3GradientSHA256 != q3SHA || !reflect.DeepEqual(r.ProtectedGradientNames, names) || r.ProtectedGradientMatrixSHA256 != protectedSHA || r.ProjectedOperatorSHA256 != projectedSHA {
		return fmt.Errorf("AOQT V10b receipt source replay mismatch")
	}
	for _, endpoint := range r.Endpoints {
		params, err := DeriveAOQTV10bGradientSVDParameters(source, endpoint.Rank, endpoint.Role, endpoint.Alpha)
		if err != nil {
			return err
		}
		sha, err := params.SHA256()
		if err != nil || sha != endpoint.ParameterSHA256 {
			return fmt.Errorf("AOQT V10b endpoint %d parameter replay hash mismatch", endpoint.Ordinal)
		}
	}
	return nil
}

func aoqtV10bProjectedOperator(source AOQTV10bGradientSVDSource) (string, string, []string, string, string, [][]float64, error) {
	if err := source.Validate(); err != nil {
		return "", "", nil, "", "", nil, err
	}
	sourceSHA, err := aoqtCanonicalSHA256(source)
	if err != nil {
		return "", "", nil, "", "", nil, err
	}
	q3SHA, err := aoqtCanonicalSHA256(source.Q3Gradient)
	if err != nil {
		return "", "", nil, "", "", nil, err
	}
	names := make([]string, len(source.ProtectedGradients))
	protectedRows := make([][]float64, len(source.ProtectedGradients))
	for i, g := range source.ProtectedGradients {
		names[i] = g.Name
		protectedRows[i] = append([]float64(nil), g.Grad...)
	}
	protectedSHA, err := aoqtCanonicalSHA256(struct {
		Names     []string                `json:"names"`
		Gradients []AOQTV10bNamedGradient `json:"gradients"`
	}{names, source.ProtectedGradients})
	if err != nil {
		return "", "", nil, "", "", nil, err
	}
	projected := make([][]float64, source.Dim)
	for i := 0; i < source.Dim; i++ {
		projected[i] = append([]float64(nil), source.Q3TargetCovariance[i]...)
	}
	basis, err := aoqtV9OrthonormalRows(protectedRows, source.Dim)
	if err != nil {
		return "", "", nil, "", "", nil, err
	}
	for j := 0; j < source.Dim; j++ {
		col := make([]float64, source.Dim)
		for i := range col {
			col[i] = projected[i][j]
		}
		col = aoqtV10bProjectAwayBasis(col, basis)
		for i := range col {
			projected[i][j] = col[i]
		}
	}
	for i := 0; i < source.Dim; i++ {
		projected[i] = aoqtV10bProjectAwayBasis(projected[i], basis)
	}
	for i := 0; i < source.Dim; i++ {
		for j := 0; j < i; j++ {
			value := 0.5 * (projected[i][j] + projected[j][i])
			if math.Abs(value) < 1e-15 {
				value = 0
			}
			projected[i][j], projected[j][i] = value, value
		}
	}
	projectedSHA, err := aoqtCanonicalSHA256(projected)
	if err != nil {
		return "", "", nil, "", "", nil, err
	}
	return sourceSHA, q3SHA, names, protectedSHA, projectedSHA, projected, nil
}

func aoqtV10bProjectAwayBasis(vector []float64, basis [][]float64) []float64 {
	out := append([]float64(nil), vector...)
	for pass := 0; pass < 2; pass++ {
		for _, q := range basis {
			scale := aoqtV9Dot(out, q)
			for i := range out {
				out[i] -= scale * q[i]
			}
		}
	}
	return out
}

func aoqtV10bTopEigenvectors(matrix [][]float64, rank int, sourceSHA string) ([][]float64, []float64, error) {
	work := make([][]float64, len(matrix))
	for i := range matrix {
		work[i] = append([]float64(nil), matrix[i]...)
	}
	vecs := make([][]float64, 0, rank)
	vals := make([]float64, 0, rank)
	for k := 0; k < rank; k++ {
		v := aoqtV10bDeterministicStart(len(work), sourceSHA, k)
		for _, prev := range vecs {
			v = aoqtV10bProjectAwayVector(v, prev)
		}
		if err := aoqtV10bNormalizeL2(v); err != nil {
			return nil, nil, err
		}
		for iter := 0; iter < 96; iter++ {
			next := aoqtV10bMatVec(work, v)
			for _, prev := range vecs {
				next = aoqtV10bProjectAwayVector(next, prev)
			}
			if aoqtV9MaxAbs(next) <= 1e-14 {
				break
			}
			if err := aoqtV10bNormalizeL2(next); err != nil {
				return nil, nil, err
			}
			v = next
		}
		lambda := aoqtV9Dot(v, aoqtV10bMatVec(work, v))
		if !aoqtV9Finite(lambda) || lambda <= 1e-12 {
			return nil, nil, fmt.Errorf("AOQT V10b projected operator has insufficient positive rank for rank %d", rank)
		}
		aoqtV10bCanonicalSign(v)
		vecs = append(vecs, v)
		vals = append(vals, lambda)
		for i := range work {
			for j := range work[i] {
				work[i][j] -= lambda * v[i] * v[j]
			}
		}
	}
	return vecs, vals, nil
}

func aoqtV10bMatVec(matrix [][]float64, vector []float64) []float64 {
	out := make([]float64, len(vector))
	for i, row := range matrix {
		for j, value := range row {
			out[i] += value * vector[j]
		}
	}
	return out
}

func aoqtV10bProjectAwayVector(vector, unit []float64) []float64 {
	out := append([]float64(nil), vector...)
	scale := aoqtV9Dot(out, unit)
	for i := range out {
		out[i] -= scale * unit[i]
	}
	return out
}

func aoqtV10bNormalizeL2(vector []float64) error {
	norm := math.Sqrt(aoqtV9Dot(vector, vector))
	if !aoqtV9Finite(norm) || norm <= 0 {
		return fmt.Errorf("AOQT V10b vector norm is invalid")
	}
	for i := range vector {
		vector[i] /= norm
	}
	return nil
}

func aoqtV10bCanonicalSign(vector []float64) {
	index := 0
	maxAbs := -1.0
	for i, value := range vector {
		abs := math.Abs(value)
		if abs > maxAbs {
			maxAbs = abs
			index = i
		}
	}
	if vector[index] < 0 {
		for i := range vector {
			vector[i] = -vector[i]
		}
	}
}

func aoqtV10bDeterministicStart(dim int, sourceSHA string, component int) []float64 {
	out := make([]float64, dim)
	for i := range out {
		h := sha256.New()
		h.Write([]byte(AOQTV10bGradientSVDDerivationVersion))
		h.Write([]byte{0})
		h.Write([]byte(sourceSHA))
		var buf [16]byte
		binary.LittleEndian.PutUint64(buf[:8], uint64(component))
		binary.LittleEndian.PutUint64(buf[8:], uint64(i))
		h.Write(buf[:])
		sum := h.Sum(nil)
		u := binary.LittleEndian.Uint64(sum[:8]) >> 11
		out[i] = (float64(u)/float64(uint64(1)<<53))*2 - 1
	}
	return out
}

func aoqtV10bValidateEndpointSchedule(endpoints []AOQTV10bGradientSVDEndpoint) error {
	index := 0
	for _, rank := range AOQTResidualLowRankRanks() {
		for _, role := range AOQTResidualLowRankRoles() {
			for _, alpha := range AOQTResidualLowRankMagnitudes() {
				if index >= len(endpoints) {
					return fmt.Errorf("AOQT V10b omitted endpoint at canonical schedule position %d", index+1)
				}
				e := endpoints[index]
				if e.Ordinal != index+1 || e.Rank != rank || e.Role != role || e.Alpha != alpha {
					return fmt.Errorf("AOQT V10b endpoint %d does not match canonical schedule position", index+1)
				}
				index++
			}
		}
	}
	return nil
}

func aoqtV10bScheduleSHA256(endpoints []AOQTV10bGradientSVDEndpoint) (string, error) {
	type item struct {
		Ordinal         int     `json:"ordinal"`
		Rank            int     `json:"rank"`
		Role            string  `json:"role"`
		Alpha           float64 `json:"alpha"`
		ParameterSHA256 string  `json:"parameter_sha256"`
	}
	items := make([]item, len(endpoints))
	for i, endpoint := range endpoints {
		items[i] = item{endpoint.Ordinal, endpoint.Rank, endpoint.Role, endpoint.Alpha, endpoint.ParameterSHA256}
	}
	return aoqtCanonicalSHA256(struct {
		Version   string `json:"version"`
		Endpoints []item `json:"endpoints"`
	}{AOQTV10bGradientSVDScheduleVersion, items})
}

func aoqtV10bEndpointHashChain(endpoints []AOQTV10bGradientSVDEndpoint) (string, string, error) {
	hashes := make([]string, len(endpoints))
	for i, endpoint := range endpoints {
		hash, err := aoqtCanonicalSHA256(endpoint)
		if err != nil {
			return "", "", err
		}
		hashes[i] = hash
	}
	data, err := json.Marshal(hashes)
	if err != nil {
		return "", "", err
	}
	return string(data), sha256HexAOQTResidual(data), nil
}

func AOQTV10bGradientSVDProtectedNames(source AOQTV10bGradientSVDSource) []string {
	names := make([]string, len(source.ProtectedGradients))
	for i := range source.ProtectedGradients {
		names[i] = source.ProtectedGradients[i].Name
	}
	return names
}
