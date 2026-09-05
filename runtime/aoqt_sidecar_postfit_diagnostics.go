package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// AOQTSidecarPostFitFailClosedDiagnosticsSchema is distinct from the
// zero-accepted-step diagnostics schema.  It describes a successful Fit whose
// final candidate-eligibility check rejected the resulting metrics.
const AOQTSidecarPostFitFailClosedDiagnosticsSchema = "eos.q3_aoqt_sidecar_postfit_failclosed_diagnostics.v1"

const AOQTSidecarPostFitCandidateEligibilityFailureKind = "post_fit_candidate_eligibility"

type AOQTSidecarPostFitFailClosedDiagnostics struct {
	Schema             string                          `json:"schema"`
	FailureKind        string                          `json:"failure_kind"`
	Error              string                          `json:"error"`
	Metrics            AOQTSidecarRunMetrics           `json:"metrics"`
	IOReport           AOQTSidecarCalibrationIOReport  `json:"io_report"`
	Preflight          AOQTSidecarMaterializePreflight `json:"preflight"`
	PreflightPath      string                          `json:"preflight_path"`
	PreflightSHA256    string                          `json:"preflight_sha256"`
	QualityClaim       bool                            `json:"quality_claim"`
	CandidatePublished bool                            `json:"candidate_published"`
}

func AOQTPostFitFailClosedDiagnosticsPath(metricsPath string) string {
	return metricsPath + ".postfit-failclosed.json"
}

// newAOQTPostFitFailClosedDiagnostics constructs evidence only after the
// unchanged eligibility validator independently reproduces the supplied
// failure.  It intentionally requires the non-plan accepted-step receipt
// chain so a plan, zero-safe-step failure, or caller-authored error cannot be
// relabeled as a post-fit diagnostic.
func newAOQTPostFitFailClosedDiagnostics(
	metrics AOQTSidecarRunMetrics,
	ioReport AOQTSidecarCalibrationIOReport,
	preflight AOQTSidecarMaterializePreflight,
	preflightPath string,
	preflightSHA256 string,
	policy AOQTSidecarCandidateEligibilityPolicy,
	eligibilityFailure error,
) (AOQTSidecarPostFitFailClosedDiagnostics, error) {
	if eligibilityFailure == nil {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, fmt.Errorf("AOQT post-fit diagnostics require the original eligibility failure")
	}
	if err := metrics.Validate(); err != nil {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, fmt.Errorf("AOQT post-fit diagnostics metrics are invalid: %w", err)
	}
	if err := validateAOQTPostFitAcceptedReceiptEvidence(metrics); err != nil {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, err
	}
	reproduced := ValidateAOQTSidecarCandidateEligibility(metrics, policy)
	if reproduced == nil {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, fmt.Errorf("AOQT post-fit diagnostics cannot record an eligible metrics result")
	}
	if reproduced.Error() != eligibilityFailure.Error() {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, fmt.Errorf("AOQT post-fit eligibility failure does not match unchanged validator: got %q, want %q", eligibilityFailure.Error(), reproduced.Error())
	}
	diagnostics := AOQTSidecarPostFitFailClosedDiagnostics{
		Schema:             AOQTSidecarPostFitFailClosedDiagnosticsSchema,
		FailureKind:        AOQTSidecarPostFitCandidateEligibilityFailureKind,
		Error:              eligibilityFailure.Error(),
		Metrics:            metrics,
		IOReport:           ioReport,
		Preflight:          preflight,
		PreflightPath:      preflightPath,
		PreflightSHA256:    preflightSHA256,
		QualityClaim:       false,
		CandidatePublished: false,
	}
	if err := diagnostics.Validate(); err != nil {
		return AOQTSidecarPostFitFailClosedDiagnostics{}, err
	}
	return diagnostics, nil
}

func validateAOQTPostFitAcceptedReceiptEvidence(metrics AOQTSidecarRunMetrics) error {
	if metrics.Plan.PlanOnly {
		return fmt.Errorf("AOQT post-fit diagnostics require a non-plan metrics result")
	}
	if metrics.Summary.Steps <= 0 {
		return fmt.Errorf("AOQT post-fit diagnostics require at least one accepted safe optimizer step")
	}
	optimizer := metrics.Summary.OptimizerDiagnostics
	if optimizer == nil {
		return fmt.Errorf("AOQT post-fit diagnostics require optimizer diagnostics")
	}
	if optimizer.AcceptedSteps <= 0 || optimizer.AcceptedProposals <= 0 {
		return fmt.Errorf("AOQT post-fit diagnostics require accepted optimizer receipts")
	}
	if optimizer.ProposalReceipts == nil || len(*optimizer.ProposalReceipts) == 0 {
		return fmt.Errorf("AOQT post-fit diagnostics require non-nil proposal receipts")
	}
	return nil
}

func (d AOQTSidecarPostFitFailClosedDiagnostics) Validate() error {
	if d.Schema != AOQTSidecarPostFitFailClosedDiagnosticsSchema {
		return fmt.Errorf("AOQT post-fit diagnostics schema %q is not supported, want %q", d.Schema, AOQTSidecarPostFitFailClosedDiagnosticsSchema)
	}
	if d.FailureKind != AOQTSidecarPostFitCandidateEligibilityFailureKind {
		return fmt.Errorf("AOQT post-fit diagnostics failure_kind %q is not supported, want %q", d.FailureKind, AOQTSidecarPostFitCandidateEligibilityFailureKind)
	}
	if strings.TrimSpace(d.Error) == "" {
		return fmt.Errorf("AOQT post-fit diagnostics error is required")
	}
	if d.QualityClaim {
		return fmt.Errorf("AOQT post-fit diagnostics quality_claim must be false")
	}
	if d.CandidatePublished {
		return fmt.Errorf("AOQT post-fit diagnostics candidate_published must be false")
	}
	if err := d.Metrics.Validate(); err != nil {
		return fmt.Errorf("AOQT post-fit diagnostics metrics: %w", err)
	}
	if err := validateAOQTPostFitAcceptedReceiptEvidence(d.Metrics); err != nil {
		return err
	}
	if strings.TrimSpace(d.PreflightPath) == "" {
		return fmt.Errorf("AOQT post-fit diagnostics preflight_path is required")
	}
	if err := validateAOQTSHA256(d.PreflightSHA256, "AOQT post-fit diagnostics preflight_sha256"); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflightRawBinding(d.Preflight, d.PreflightPath, d.PreflightSHA256); err != nil {
		return err
	}
	if err := validateAOQTFailClosedIOReport(d.IOReport, d.Metrics.Plan, d.Metrics.Inputs, d.Metrics.Topology, d.Metrics.ObjectiveContract, d.Metrics.LegalGates); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflight(d.Preflight, d.PreflightPath, d.IOReport, d.Metrics.Inputs, d.Metrics.Topology, d.Metrics.ObjectiveContract, d.Metrics.LegalGates); err != nil {
		return err
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.Metrics.TrainingContract, d.Metrics.CandidateEligibilityPolicy, d.IOReport.TrainingContract, d.IOReport.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT post-fit diagnostics IO report training contract: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.Metrics.TrainingContract, d.Metrics.CandidateEligibilityPolicy, d.Preflight.TrainingContract, d.Preflight.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT post-fit diagnostics preflight training contract: %w", err)
	}
	policy, err := AOQTSidecarTrainingContractPolicy(d.Metrics.TrainingContract, d.Metrics.CandidateEligibilityPolicy)
	if err != nil {
		return fmt.Errorf("AOQT post-fit diagnostics metrics training contract: %w", err)
	}
	reproduced := ValidateAOQTSidecarCandidateEligibility(d.Metrics, policy)
	if reproduced == nil {
		return fmt.Errorf("AOQT post-fit diagnostics cannot record an eligible metrics result")
	}
	if reproduced.Error() != d.Error {
		return fmt.Errorf("AOQT post-fit diagnostics error does not match unchanged validator: got %q, want %q", d.Error, reproduced.Error())
	}
	return nil
}

func writeAOQTPostFitFailClosedDiagnosticsFile(path string, diagnostics AOQTSidecarPostFitFailClosedDiagnostics) error {
	if err := diagnostics.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(diagnostics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT post-fit diagnostics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT post-fit diagnostics parent: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".aoqt-postfit-failclosed-*")
	if err != nil {
		return fmt.Errorf("create AOQT post-fit diagnostics temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	defer cleanup()
	if err := tmp.Chmod(0o644); err != nil {
		return fmt.Errorf("set AOQT post-fit diagnostics temporary permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write AOQT post-fit diagnostics temporary file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync AOQT post-fit diagnostics temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close AOQT post-fit diagnostics temporary file: %w", err)
	}
	if err := os.Link(tmpPath, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT post-fit diagnostics output %q already exists", path)
		}
		return fmt.Errorf("create AOQT post-fit diagnostics output %q exclusively: %w", path, err)
	}
	return nil
}

func ensureAOQTPostFitFailClosedDiagnosticsAbsent(metricsPath string) error {
	path := AOQTPostFitFailClosedDiagnosticsPath(metricsPath)
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT sidecar post-fit fail-closed diagnostics output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}
