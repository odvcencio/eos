package eosruntime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
)

type AOQTSidecarTrainRunnerConfig struct {
	ManifestPath                           string
	RowsJSONLPath                          string
	PreflightJSONPath                      string
	MetricsJSONPath                        string
	OutputArtifactPath                     string
	DevOutputDir                           string
	ExpectedManifestSHA256                 string
	ExpectedRowsSHA256                     string
	ExpectedPreflightSHA256                string
	ExpectedSplitManifestSHA256            string
	ExpectedAnchorArtifactSHA256           string
	ExpectedAnchorPackageManifestSHA256    string
	ExpectedAnchorEmbeddingSpaceID         string
	ExpectedCompatibilityDigest            string
	ExpectedSourceArtifactHashes           []string
	ExpectedVectorCacheHashes              []string
	OptimizerMode                          string
	SplitManifestPath                      string
	FoldID                                 string
	PlanOnly                               bool
	AllowResearchOnly                      bool
	AllowDevOnly                           bool
	RequireForwardConsistencyProbe         bool
	ForwardConsistencyProbeOnly            bool
	ForwardConsistencyProbeReceiptJSONPath string
	// RequireActualDirectionProbe enables the V7-r3 actual-direction endpoint
	// sweep. ActualDirectionProbeRequired is retained as a source-compatible
	// alias for callers that mirror AOQTSidecarTrainConfig.
	RequireActualDirectionProbe         bool
	ActualDirectionProbeRequired        bool
	ActualDirectionProbeOnly            bool
	ActualDirectionProbeReceiptJSONPath string
	// RequireActualCoordinateProbe enables the preregistered V7-r4 scalar
	// endpoint sweep. V7-r4 is probe-only until a later authorization changes
	// the trainer contract.
	RequireActualCoordinateProbe         bool
	ActualCoordinateProbeRequired        bool
	ActualCoordinateProbeOnly            bool
	ActualCoordinateProbeReceiptJSONPath string
	// V7-r5 consumes the root-authorized, bound V7-r4 receipt and writes a
	// separate bound full-training receipt. The longer *JSONPath names are
	// canonical; aliases keep trainer-oriented callers source-compatible.
	RequireActualCoordinateTrain             bool
	ActualCoordinateTrainRequired            bool
	ActualCoordinateTrainR4ReceiptJSONPath   string
	ActualCoordinateTrainR4ReceiptPath       string
	ActualCoordinateTrainR4ReceiptSHA256     string
	CanonicalR4ReceiptJSONPath               string
	CanonicalR4ReceiptPath                   string
	ExpectedCanonicalR4ReceiptSHA256         string
	CanonicalR4ReceiptSHA256                 string
	ActualCoordinateTrainReceiptJSONPath     string
	ActualCoordinateTrainReceiptPath         string
	ActualCoordinateFullTrainReceiptJSONPath string
	ActualCoordinateFullTrainReceiptPath     string
	MaxSteps                                 int
	LearningRate                             float32
}

type AOQTSidecarTrainRunnerResult struct {
	IOReport                               AOQTSidecarCalibrationIOReport
	Preflight                              AOQTSidecarMaterializePreflight
	Metrics                                AOQTSidecarRunMetrics
	PackageResult                          *AOQTSidecarCandidatePackageResult
	DevEvidence                            *AOQTSidecarDevEvidence
	DevFailureDiagnostics                  *AOQTSidecarDevFailClosedDiagnostics
	FailureDiagnostics                     *AOQTSidecarFailClosedDiagnostics
	PostFitFailureDiagnostics              *AOQTSidecarPostFitFailClosedDiagnostics
	ForwardConsistencyProbe                *AOQTSidecarForwardConsistencyProbeReceipt
	ForwardConsistencyProbeSHA256          string
	ForwardConsistencyProbeReceiptJSONPath string
	ActualDirectionProbe                   *AOQTSidecarActualDirectionProbeReceipt
	ActualDirectionProbeSHA256             string
	ActualDirectionProbeReceiptJSONPath    string
	ActualCoordinateProbe                  *AOQTSidecarActualCoordinateProbeReceipt
	ActualCoordinateProbeSHA256            string
	ActualCoordinateProbeReceiptJSONPath   string
	ActualCoordinateTrain                  *AOQTSidecarActualCoordinateTrainReceipt
	ActualCoordinateTrainSHA256            string
	ActualCoordinateTrainReceiptJSONPath   string
}

const AOQTSidecarDevEvidenceSchema = "eos.q3_aoqt_sidecar_dev_evidence.v1"

type AOQTSidecarDevSplitBinding struct {
	Schema                       string            `json:"schema"`
	SplitManifestSchema          string            `json:"split_manifest_schema"`
	SplitManifestPath            string            `json:"split_manifest_path"`
	SplitManifestFileSHA256      string            `json:"split_manifest_file_sha256"`
	SplitManifestSHA256          string            `json:"split_manifest_sha256"`
	SplitManifestPayloadSHA256   string            `json:"split_manifest_payload_sha256"`
	SourcePlanSHA256             string            `json:"source_plan_sha256"`
	SourcePlanRowsSHA256         string            `json:"source_plan_rows_sha256,omitempty"`
	SourcePlanRowIDSHA256        string            `json:"source_plan_row_ids_sha256,omitempty"`
	OfficialQIDRegistrySHA256    string            `json:"official_qid_registry_sha256"`
	OfficialRegistrySourceSHA256 string            `json:"official_registry_source_sha256"`
	FoldID                       string            `json:"fold_id"`
	TrainRowCount                int               `json:"train_row_count"`
	TrainRowIDSHA256             string            `json:"train_row_ids_sha256"`
	TrainQIDCountByDataset       map[string]int    `json:"train_qid_count_by_dataset"`
	TrainQIDSetSHA256ByDataset   map[string]string `json:"train_qid_set_sha256_by_dataset"`
	TrainQIDsByDatasetSHA256     string            `json:"train_qids_by_dataset_sha256"`
	MaterializedManifestSHA256   string            `json:"materialized_manifest_sha256"`
	MaterializedRowsSHA256       string            `json:"materialized_rows_sha256"`
	MaterializedPreflightSHA256  string            `json:"materialized_preflight_sha256"`
	MaterializedRowIDSHA256      string            `json:"materialized_row_ids_sha256"`
	MaterializedRowCount         int               `json:"materialized_row_count"`
}

const (
	AOQTV7DevSplitManifestSchema  = "eos.aoqt.v7_dev_split_manifest.v1"
	AOQTV7TrainSplitBindingSchema = "eos.aoqt.v7_train_split_binding.v1"
)

type AOQTSidecarDevEvidence struct {
	Schema          string                      `json:"schema"`
	DevOnly         bool                        `json:"dev_only"`
	OptimizerMode   string                      `json:"optimizer_mode"`
	MetricsPath     string                      `json:"metrics_path"`
	MetricsSHA256   string                      `json:"metrics_sha256"`
	PreflightPath   string                      `json:"preflight_path"`
	PreflightSHA256 string                      `json:"preflight_sha256"`
	TransformPath   string                      `json:"transform_path"`
	TransformSHA256 string                      `json:"transform_sha256"`
	PairingsSHA256  string                      `json:"pairings_sha256"`
	AnglesSHA256    string                      `json:"angles_sha256"`
	SplitBinding    *AOQTSidecarDevSplitBinding `json:"split_binding,omitempty"`
}

type aoqtRunnerObjectiveFactory func(AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error)

// v2 binds fail-closed optimizer evidence to the protected-cone coordinate
// audit contract. Older v1 diagnostics are intentionally not relabeled.
const AOQTSidecarFailClosedDiagnosticsSchema = "eos.q3_aoqt_sidecar_failclosed_diagnostics.v2"

type AOQTSidecarFailClosedDiagnostics struct {
	Schema                     string                                 `json:"schema"`
	FailureKind                string                                 `json:"failure_kind"`
	Error                      string                                 `json:"error"`
	RejectionSummary           string                                 `json:"rejection_summary"`
	Plan                       AOQTSidecarWorkPlan                    `json:"plan"`
	Inputs                     AOQTSidecarRunMetricInputs             `json:"inputs"`
	IOReport                   AOQTSidecarCalibrationIOReport         `json:"io_report"`
	Preflight                  AOQTSidecarMaterializePreflight        `json:"preflight"`
	PreflightPath              string                                 `json:"preflight_path"`
	PreflightSHA256            string                                 `json:"preflight_sha256"`
	Topology                   AOQTSidecarTopologyBinding             `json:"topology"`
	ObjectiveContract          AOQTSidecarObjectiveContract           `json:"objective_contract"`
	LegalGates                 AOQTSidecarLegalGates                  `json:"legal_gates"`
	Summary                    AOQTSidecarTrainSummary                `json:"summary"`
	QualityClaim               bool                                   `json:"quality_claim"`
	TrainingContract           string                                 `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy `json:"candidate_eligibility_policy,omitempty"`
}

// AOQTSidecarDevFailClosedDiagnostics is a dev-only failure receipt. It is
// separate from the production fail-closed schema because a V7-r2 run may
// fail during its train-only consistency probe before any optimizer step.
const AOQTSidecarDevFailClosedDiagnosticsSchema = "eos.aoqt.v7_dev_failclosed_diagnostics.v1"

type AOQTSidecarDevFailClosedDiagnostics struct {
	Schema                     string                                     `json:"schema"`
	FailureKind                string                                     `json:"failure_kind"`
	Error                      string                                     `json:"error"`
	RejectionSummary           string                                     `json:"rejection_summary,omitempty"`
	Plan                       AOQTSidecarWorkPlan                        `json:"plan"`
	Inputs                     AOQTSidecarRunMetricInputs                 `json:"inputs"`
	IOReport                   AOQTSidecarCalibrationIOReport             `json:"io_report"`
	Preflight                  AOQTSidecarMaterializePreflight            `json:"preflight"`
	PreflightPath              string                                     `json:"preflight_path"`
	PreflightSHA256            string                                     `json:"preflight_sha256"`
	SplitBinding               *AOQTSidecarDevSplitBinding                `json:"split_binding,omitempty"`
	Topology                   AOQTSidecarTopologyBinding                 `json:"topology"`
	ObjectiveContract          AOQTSidecarObjectiveContract               `json:"objective_contract"`
	LegalGates                 AOQTSidecarLegalGates                      `json:"legal_gates"`
	Summary                    AOQTSidecarTrainSummary                    `json:"summary"`
	ForwardConsistencyProbe    *AOQTSidecarForwardConsistencyProbeReceipt `json:"forward_consistency_probe,omitempty"`
	ActualDirectionProbe       *AOQTSidecarActualDirectionProbeReceipt    `json:"actual_direction_probe,omitempty"`
	ActualCoordinateTrain      *AOQTSidecarActualCoordinateTrainReceipt   `json:"actual_coordinate_train,omitempty"`
	QualityClaim               bool                                       `json:"quality_claim"`
	ReleaseClaim               bool                                       `json:"release_claim"`
	OfficialClaim              bool                                       `json:"official_claim"`
	OfficialHeldoutGate        bool                                       `json:"official_heldout_gate"`
	CommercialClaim            bool                                       `json:"commercial_claim"`
	TrainingContract           string                                     `json:"training_contract,omitempty"`
	CandidateEligibilityPolicy *AOQTSidecarCandidateEligibilityPolicy     `json:"candidate_eligibility_policy,omitempty"`
}

func RunAOQTSidecarTraining(cfg AOQTSidecarTrainRunnerConfig) (AOQTSidecarTrainRunnerResult, error) {
	return runAOQTSidecarTraining(cfg, aoqtRunnerObjectiveFromContract)
}

func runAOQTSidecarTraining(cfg AOQTSidecarTrainRunnerConfig, objectiveFactory aoqtRunnerObjectiveFactory) (AOQTSidecarTrainRunnerResult, error) {
	if err := validateAOQTSidecarTrainRunnerConfig(cfg); err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	topology, err := expectedAOQTSidecarTrainTopology()
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	set, ioReport, err := LoadAOQTSidecarCalibrationSet(AOQTSidecarCalibrationIOConfig{
		ManifestPath:                        cfg.ManifestPath,
		RowsJSONLPath:                       cfg.RowsJSONLPath,
		ExpectedManifestSHA256:              cfg.ExpectedManifestSHA256,
		ExpectedRowsSHA256:                  cfg.ExpectedRowsSHA256,
		ExpectedAnchorArtifactSHA256:        cfg.ExpectedAnchorArtifactSHA256,
		ExpectedAnchorPackageManifestSHA256: cfg.ExpectedAnchorPackageManifestSHA256,
		ExpectedAnchorEmbeddingSpaceID:      cfg.ExpectedAnchorEmbeddingSpaceID,
		ExpectedCompatibilityDigest:         cfg.ExpectedCompatibilityDigest,
		ExpectedTurboQuantSeed:              AOQTSidecarMaterializerQuantSeed,
		ExpectedTopology:                    topology,
		ExpectedSourceArtifactHashes:        append([]string(nil), cfg.ExpectedSourceArtifactHashes...),
		ExpectedVectorCacheHashes:           append([]string(nil), cfg.ExpectedVectorCacheHashes...),
	})
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	preflight, preflightSHA256, err := loadAndValidateAOQTTrainPreflight(cfg, set, ioReport)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	splitBinding, err := validateAOQTV7DevSplitBinding(cfg, set, ioReport, preflight, preflightSHA256)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	var canonicalR4Receipt *AOQTSidecarActualCoordinateProbeReceipt
	if cfg.actualCoordinateTrainMode() {
		canonicalR4Receipt, err = loadAOQTV7R5CanonicalR4Receipt(cfg, set, ioReport, preflightSHA256, splitBinding)
		if err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	trainerConfig := AOQTSidecarTrainConfig{
		PairingSeed:                     AOQTSidecarMaterializerTopologySeed,
		WorkplanSeed:                    AOQTSidecarMaterializerQuantSeed,
		OptimizerMode:                   cfg.OptimizerMode,
		PlanOnly:                        cfg.PlanOnly,
		MaxSteps:                        cfg.MaxSteps,
		LearningRate:                    cfg.LearningRate,
		ForwardConsistencyProbeRequired: cfg.RequireForwardConsistencyProbe,
		ForwardConsistencyProbeOnly:     cfg.ForwardConsistencyProbeOnly,
		ForwardConsistencyProbeFoldID:   cfg.FoldID,
		ForwardConsistencyProbeSplitManifestSHA256: cfg.ExpectedSplitManifestSHA256,
		ActualDirectionProbeRequired:               cfg.actualDirectionProbeRequired(),
		ActualDirectionProbeOnly:                   cfg.ActualDirectionProbeOnly,
		ActualDirectionProbeFoldID:                 cfg.FoldID,
		ActualDirectionProbeSplitManifestSHA256:    cfg.ExpectedSplitManifestSHA256,
		ActualCoordinateProbeRequired:              cfg.actualCoordinateProbeRequired(),
		ActualCoordinateProbeOnly:                  cfg.ActualCoordinateProbeOnly,
		ActualCoordinateProbeFoldID:                cfg.FoldID,
		ActualCoordinateProbeSplitManifestSHA256:   cfg.ExpectedSplitManifestSHA256,
		AngleCap:                                   AOQTSidecarDefaultAngleCap,
		MaxAngleCap:                                AOQTSidecarHardMaxAngleCap,
		CaptureProposalReceipts:                    true,
		TrainingContract:                           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy:                 cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	}
	// Keep the canonical r4 authorization fields out of ordinary, r4 probe,
	// and plan-only trainer configs.  The expected r5 pin is intentionally
	// populated only for the exact r5 lane; passing it through unconditionally
	// would make the trainer interpret every legacy run as an r5 request.
	if cfg.actualCoordinateTrainMode() {
		trainerConfig.ActualCoordinateTrainRequired = cfg.actualCoordinateTrainRequired()
		trainerConfig.ActualCoordinateTrainR4Receipt = canonicalR4Receipt
		trainerConfig.ActualCoordinateTrainR4ReceiptPath = cfg.canonicalR4ReceiptPath()
		trainerConfig.ActualCoordinateTrainR4ReceiptSHA256 = cfg.canonicalR4ReceiptExpectedSHA256()
		trainerConfig.CanonicalR4Receipt = canonicalR4Receipt
		trainerConfig.CanonicalR4ReceiptPath = cfg.canonicalR4ReceiptPath()
		trainerConfig.CanonicalR4ReceiptSHA256 = cfg.canonicalR4ReceiptExpectedSHA256()
	}
	trainer, err := NewAOQTSidecarTrainer(trainerConfig)
	if err != nil {
		return AOQTSidecarTrainRunnerResult{}, err
	}
	var summary AOQTSidecarTrainSummary
	// r5 failures are serialized through one path, including failures that
	// happen before Fit receives an objective and failures discovered while
	// writing/validating post-fit artifacts. This closure is installed only
	// after authoritative materialization, preflight, split, and canonical r4
	// authorization have been loaded, so its receipt can be fully bound.
	finishR5Failure := func(failure error) (AOQTSidecarTrainRunnerResult, error) {
		if !cfg.actualCoordinateTrainMode() {
			return AOQTSidecarTrainRunnerResult{}, failure
		}
		if failure == nil {
			failure = fmt.Errorf("AOQT V7-r5 actual-coordinate training failed")
		}
		if summary.ActualCoordinateTrain == nil || summary.OptimizerDiagnostics == nil {
			seeded, seedErr := trainer.Fit(set, nil)
			if seeded.ActualCoordinateTrain != nil && seeded.OptimizerDiagnostics != nil {
				summary = seeded
			} else if seedErr != nil {
				return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; failed to initialize AOQT V7-r5 failure artifacts: %v", failure, seedErr)
			}
		}
		if summary.ActualCoordinateTrain == nil || summary.OptimizerDiagnostics == nil {
			return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; AOQT V7-r5 failure artifacts are missing", failure)
		}
		bindTrain := func() error {
			return bindAOQTActualCoordinateTrainReceipt(summary.ActualCoordinateTrain, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding, cfg.canonicalR4ReceiptPath(), cfg.canonicalR4ReceiptExpectedSHA256())
		}
		// Bind before finalization to populate source hashes on early failures,
		// then bind again because finalization rebuilds the step-chain hash.
		if bindErr := bindTrain(); bindErr != nil {
			failure = fmt.Errorf("%w; failed to bind AOQT V7-r5 failure receipt: %v", failure, bindErr)
		}
		diagnostics := *summary.OptimizerDiagnostics
		receipt := *summary.ActualCoordinateTrain
		summary, _ = trainer.finishV7R5Failure(summary, diagnostics, receipt, failure)
		if bindErr := bindTrain(); bindErr != nil {
			failure = fmt.Errorf("%w; failed to bind AOQT V7-r5 failure receipt: %v", failure, bindErr)
		} else if trainSHA, hashErr := summary.ActualCoordinateTrain.SHA256(); hashErr != nil {
			failure = fmt.Errorf("%w; failed to hash AOQT V7-r5 failure receipt: %v", failure, hashErr)
		} else {
			summary.ActualCoordinateTrainSHA256 = trainSHA
		}
		// A failed r5 run must never leave a metrics artifact behind. This also
		// rolls back metrics when a later dev-evidence/write step fails.
		if removeErr := os.Remove(cfg.MetricsJSONPath); removeErr != nil && !os.IsNotExist(removeErr) {
			failure = fmt.Errorf("%w; failed to remove AOQT metrics output %q: %v", failure, cfg.MetricsJSONPath, removeErr)
		}
		result := AOQTSidecarTrainRunnerResult{
			IOReport: ioReport, Preflight: preflight,
			ActualCoordinateTrain:                summary.ActualCoordinateTrain,
			ActualCoordinateTrainSHA256:          summary.ActualCoordinateTrainSHA256,
			ActualCoordinateTrainReceiptJSONPath: cfg.actualCoordinateTrainReceiptPath(),
		}
		receiptPath := cfg.actualCoordinateTrainReceiptPath()
		if strings.TrimSpace(receiptPath) != "" {
			// The normal success path writes the receipt before dev evidence. If a
			// later operation fails, replace that run-owned output with the bound
			// failed receipt.
			if removeErr := os.Remove(receiptPath); removeErr != nil && !os.IsNotExist(removeErr) {
				failure = fmt.Errorf("%w; failed to replace AOQT V7-r5 receipt output %q: %v", failure, receiptPath, removeErr)
			} else if writeErr := writeAOQTActualCoordinateTrainReceiptFile(receiptPath, *summary.ActualCoordinateTrain); writeErr != nil {
				failure = fmt.Errorf("%w; failed to write AOQT V7-r5 failure receipt: %v", failure, writeErr)
			}
		}
		devDiagnostics, diagnosticsErr := newAOQTDevFailClosedDiagnostics(set, ioReport, preflight, cfg.PreflightJSONPath, preflightSHA256, splitBinding, summary, failure)
		if diagnosticsErr != nil {
			return result, fmt.Errorf("%w; failed to construct AOQT dev fail-closed diagnostics: %v", failure, diagnosticsErr)
		}
		result.DevFailureDiagnostics = &devDiagnostics
		if writeErr := writeAOQTDevFailClosedDiagnosticsFile(AOQTDevFailClosedDiagnosticsPath(cfg.MetricsJSONPath), devDiagnostics); writeErr != nil {
			return result, fmt.Errorf("%w; failed to write AOQT dev fail-closed diagnostics: %v", failure, writeErr)
		}
		return result, failure
	}
	var objective AOQTSidecarVectorObjective
	var objectiveErr error
	if !cfg.PlanOnly {
		if objectiveFactory == nil {
			objectiveErr = fmt.Errorf("AOQT objective factory is required for non-plan training")
		} else {
			objective, objectiveErr = objectiveFactory(set.Manifest.ObjectiveContract)
		}
	}
	if objectiveErr != nil {
		if cfg.actualCoordinateTrainMode() {
			// Fit(nil) initializes the r5 receipt/diagnostics; preserve the
			// factory's actual error as the durable failure reason below.
			if seeded, seedErr := trainer.Fit(set, nil); seeded.ActualCoordinateTrain != nil && seeded.OptimizerDiagnostics != nil {
				summary = seeded
			} else if seedErr != nil {
				return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; failed to initialize AOQT V7-r5 failure artifacts: %v", objectiveErr, seedErr)
			}
			err = objectiveErr
		} else {
			return AOQTSidecarTrainRunnerResult{}, objectiveErr
		}
	} else {
		summary, err = trainer.Fit(set, objective)
	}
	if summary.ForwardConsistencyProbe != nil {
		if bindErr := bindAOQTForwardConsistencyProbeReceipt(summary.ForwardConsistencyProbe, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding); bindErr != nil {
			if err == nil {
				err = bindErr
			} else {
				err = fmt.Errorf("%w; failed to bind forward-consistency probe receipt: %v", err, bindErr)
			}
		} else if probeSHA, hashErr := summary.ForwardConsistencyProbe.SHA256(); hashErr == nil {
			summary.ForwardConsistencyProbeSHA256 = probeSHA
		} else if err == nil {
			err = hashErr
		}
	}
	if summary.ActualDirectionProbe != nil {
		if bindErr := bindAOQTActualDirectionProbeReceipt(summary.ActualDirectionProbe, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding); bindErr != nil {
			if err == nil {
				err = bindErr
			} else {
				err = fmt.Errorf("%w; failed to bind actual-direction probe receipt: %v", err, bindErr)
			}
		} else if probeSHA, hashErr := summary.ActualDirectionProbe.SHA256(); hashErr == nil {
			summary.ActualDirectionProbeSHA256 = probeSHA
			if provenanceErr := bindAOQTActualDirectionProposalReceipts(summary.OptimizerDiagnostics, summary.ActualDirectionProbe, probeSHA); provenanceErr != nil && err == nil {
				err = provenanceErr
			} else if provenanceErr == nil {
				if hashErr := refreshAOQTActualDirectionDiagnosticsHash(&summary); hashErr != nil && err == nil {
					err = hashErr
				}
			}
		} else if err == nil {
			err = hashErr
		}
	}
	if summary.ActualCoordinateProbe != nil {
		if bindErr := bindAOQTActualCoordinateProbeReceipt(summary.ActualCoordinateProbe, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding); bindErr != nil {
			if err == nil {
				err = bindErr
			} else {
				err = fmt.Errorf("%w; failed to bind actual-coordinate probe receipt: %v", err, bindErr)
			}
		} else if probeSHA, hashErr := summary.ActualCoordinateProbe.SHA256(); hashErr == nil {
			summary.ActualCoordinateProbeSHA256 = probeSHA
		} else if err == nil {
			err = hashErr
		}
	}
	if summary.ActualCoordinateTrain != nil {
		if bindErr := bindAOQTActualCoordinateTrainReceipt(summary.ActualCoordinateTrain, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding, cfg.canonicalR4ReceiptPath(), cfg.canonicalR4ReceiptExpectedSHA256()); bindErr != nil {
			if err == nil {
				err = bindErr
			} else {
				err = fmt.Errorf("%w; failed to bind actual-coordinate full-training receipt: %v", err, bindErr)
			}
		} else if trainSHA, hashErr := summary.ActualCoordinateTrain.SHA256(); hashErr == nil {
			summary.ActualCoordinateTrainSHA256 = trainSHA
		} else if err == nil {
			err = hashErr
		}
	}
	if cfg.ActualCoordinateProbeOnly {
		result := AOQTSidecarTrainRunnerResult{
			IOReport:                             ioReport,
			Preflight:                            preflight,
			ActualCoordinateProbe:                summary.ActualCoordinateProbe,
			ActualCoordinateProbeSHA256:          summary.ActualCoordinateProbeSHA256,
			ActualCoordinateProbeReceiptJSONPath: cfg.ActualCoordinateProbeReceiptJSONPath,
		}
		if summary.ActualCoordinateProbe == nil {
			if err == nil {
				err = fmt.Errorf("AOQT actual-coordinate probe-only run produced no receipt")
			}
			return result, err
		}
		if writeErr := writeAOQTActualCoordinateProbeReceiptFile(cfg.ActualCoordinateProbeReceiptJSONPath, *summary.ActualCoordinateProbe); writeErr != nil {
			if err == nil {
				err = writeErr
			} else {
				err = fmt.Errorf("%w; failed to write actual-coordinate probe receipt: %v", err, writeErr)
			}
		}
		return result, err
	}
	if cfg.ActualDirectionProbeOnly {
		result := AOQTSidecarTrainRunnerResult{
			IOReport:                            ioReport,
			Preflight:                           preflight,
			ActualDirectionProbe:                summary.ActualDirectionProbe,
			ActualDirectionProbeSHA256:          summary.ActualDirectionProbeSHA256,
			ActualDirectionProbeReceiptJSONPath: cfg.ActualDirectionProbeReceiptJSONPath,
		}
		if summary.ActualDirectionProbe == nil {
			if err == nil {
				err = fmt.Errorf("AOQT actual-direction probe-only run produced no receipt")
			}
			return result, err
		}
		if writeErr := writeAOQTActualDirectionProbeReceiptFile(cfg.ActualDirectionProbeReceiptJSONPath, *summary.ActualDirectionProbe); writeErr != nil {
			if err == nil {
				err = writeErr
			} else {
				err = fmt.Errorf("%w; failed to write actual-direction probe receipt: %v", err, writeErr)
			}
		}
		return result, err
	}
	if cfg.ForwardConsistencyProbeOnly {
		result := AOQTSidecarTrainRunnerResult{
			IOReport:                               ioReport,
			Preflight:                              preflight,
			ForwardConsistencyProbe:                summary.ForwardConsistencyProbe,
			ForwardConsistencyProbeSHA256:          summary.ForwardConsistencyProbeSHA256,
			ForwardConsistencyProbeReceiptJSONPath: cfg.ForwardConsistencyProbeReceiptJSONPath,
		}
		if summary.ForwardConsistencyProbe == nil {
			if err == nil {
				err = fmt.Errorf("AOQT forward-consistency probe-only run produced no receipt")
			}
			return result, err
		}
		if writeErr := writeAOQTForwardConsistencyProbeReceiptFile(cfg.ForwardConsistencyProbeReceiptJSONPath, *summary.ForwardConsistencyProbe); writeErr != nil {
			if err == nil {
				err = writeErr
			} else {
				err = fmt.Errorf("%w; failed to write forward-consistency probe receipt: %v", err, writeErr)
			}
		}
		return result, err
	}
	if err != nil && cfg.actualCoordinateTrainMode() {
		return finishR5Failure(err)
	}
	if err != nil {
		result := AOQTSidecarTrainRunnerResult{IOReport: ioReport, Preflight: preflight, ActualCoordinateTrain: summary.ActualCoordinateTrain, ActualCoordinateTrainSHA256: summary.ActualCoordinateTrainSHA256, ActualCoordinateTrainReceiptJSONPath: cfg.actualCoordinateTrainReceiptPath()}
		if cfg.actualCoordinateTrainMode() && summary.ActualCoordinateTrain != nil {
			if writeErr := writeAOQTActualCoordinateTrainReceiptFile(cfg.actualCoordinateTrainReceiptPath(), *summary.ActualCoordinateTrain); writeErr != nil {
				err = fmt.Errorf("%w; failed to write actual-coordinate full-training receipt: %v", err, writeErr)
			}
		}
		if !cfg.PlanOnly && cfg.isDevOnlyRun() {
			diagnostics, diagnosticsErr := newAOQTDevFailClosedDiagnostics(set, ioReport, preflight, cfg.PreflightJSONPath, preflightSHA256, splitBinding, summary, err)
			if diagnosticsErr != nil {
				return result, fmt.Errorf("%w; failed to construct AOQT dev fail-closed diagnostics: %v", err, diagnosticsErr)
			}
			result.DevFailureDiagnostics = &diagnostics
			if writeErr := writeAOQTDevFailClosedDiagnosticsFile(AOQTDevFailClosedDiagnosticsPath(cfg.MetricsJSONPath), diagnostics); writeErr != nil {
				return result, fmt.Errorf("%w; failed to write AOQT dev fail-closed diagnostics: %v", err, writeErr)
			}
		} else if !cfg.PlanOnly {
			diagnostics, diagnosticsErr := newAOQTFailClosedDiagnostics(set, ioReport, preflight, cfg.PreflightJSONPath, preflightSHA256, summary, err)
			if diagnosticsErr != nil {
				return result, fmt.Errorf("%w; failed to construct AOQT fail-closed diagnostics: %v", err, diagnosticsErr)
			}
			result.FailureDiagnostics = &diagnostics
			if writeErr := writeAOQTFailClosedDiagnosticsFile(AOQTFailClosedDiagnosticsPath(cfg.MetricsJSONPath), diagnostics); writeErr != nil {
				return result, fmt.Errorf("%w; failed to write AOQT fail-closed diagnostics: %v", err, writeErr)
			}
		}
		return result, err
	}
	metrics, err := NewAOQTSidecarRunMetrics(set, summary)
	if err != nil {
		if cfg.actualCoordinateTrainMode() {
			return finishR5Failure(err)
		}
		return AOQTSidecarTrainRunnerResult{}, err
	}
	if splitBinding != nil {
		binding := *splitBinding
		metrics.SplitBinding = &binding
	}
	if metrics.Summary.ForwardConsistencyProbe != nil {
		if err := bindAOQTForwardConsistencyProbeReceipt(metrics.Summary.ForwardConsistencyProbe, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		probeSHA, err := metrics.Summary.ForwardConsistencyProbe.SHA256()
		if err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		metrics.Summary.ForwardConsistencyProbeSHA256 = probeSHA
	}
	if metrics.Summary.ActualDirectionProbe != nil {
		if err := bindAOQTActualDirectionProbeReceipt(metrics.Summary.ActualDirectionProbe, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		probeSHA, err := metrics.Summary.ActualDirectionProbe.SHA256()
		if err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		metrics.Summary.ActualDirectionProbeSHA256 = probeSHA
		if err := bindAOQTActualDirectionProposalReceipts(metrics.Summary.OptimizerDiagnostics, metrics.Summary.ActualDirectionProbe, probeSHA); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		if err := refreshAOQTActualDirectionDiagnosticsHash(&metrics.Summary); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	if metrics.Summary.ActualCoordinateTrain != nil {
		if err := bindAOQTActualCoordinateTrainReceipt(metrics.Summary.ActualCoordinateTrain, set, ioReport, cfg.PreflightJSONPath, preflightSHA256, splitBinding, cfg.canonicalR4ReceiptPath(), cfg.canonicalR4ReceiptExpectedSHA256()); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		trainSHA, err := metrics.Summary.ActualCoordinateTrain.SHA256()
		if err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
		metrics.Summary.ActualCoordinateTrainSHA256 = trainSHA
	}
	if err := metrics.Validate(); err != nil {
		if cfg.actualCoordinateTrainMode() {
			summary = metrics.Summary
			return finishR5Failure(err)
		}
		return AOQTSidecarTrainRunnerResult{}, err
	}
	expectedLearningRate := normalizedAOQTSidecarTrainConfig(AOQTSidecarTrainConfig{LearningRate: cfg.LearningRate}).LearningRate
	if metrics.Plan.LearningRate != expectedLearningRate {
		if cfg.actualCoordinateTrainMode() {
			summary = metrics.Summary
			return finishR5Failure(fmt.Errorf("AOQT runner work plan learning_rate = %.9g, want configured learning_rate %.9g", metrics.Plan.LearningRate, expectedLearningRate))
		}
		return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("AOQT runner work plan learning_rate = %.9g, want configured learning_rate %.9g", metrics.Plan.LearningRate, expectedLearningRate)
	}
	if !cfg.PlanOnly && !cfg.isDevOnlyRun() {
		policy, err := AOQTSidecarTrainingContractPolicy(metrics.TrainingContract, metrics.CandidateEligibilityPolicy)
		if err != nil {
			return AOQTSidecarTrainRunnerResult{}, err
		}
		if eligibilityErr := ValidateAOQTSidecarCandidateEligibility(metrics, policy); eligibilityErr != nil {
			result := AOQTSidecarTrainRunnerResult{IOReport: ioReport, Preflight: preflight}
			diagnostics, diagnosticsErr := newAOQTPostFitFailClosedDiagnostics(metrics, ioReport, preflight, cfg.PreflightJSONPath, preflightSHA256, policy, eligibilityErr)
			if diagnosticsErr != nil {
				return result, fmt.Errorf("%w; failed to construct AOQT post-fit fail-closed diagnostics: %v", eligibilityErr, diagnosticsErr)
			}
			result.PostFitFailureDiagnostics = &diagnostics
			if writeErr := writeAOQTPostFitFailClosedDiagnosticsFile(AOQTPostFitFailClosedDiagnosticsPath(cfg.MetricsJSONPath), diagnostics); writeErr != nil {
				return result, fmt.Errorf("%w; failed to write AOQT post-fit fail-closed diagnostics: %v", eligibilityErr, writeErr)
			}
			return result, eligibilityErr
		}
	}
	if err := writeAOQTRunMetricsFile(cfg.MetricsJSONPath, metrics); err != nil {
		if cfg.actualCoordinateTrainMode() {
			summary = metrics.Summary
			return finishR5Failure(err)
		}
		return AOQTSidecarTrainRunnerResult{}, err
	}
	result := AOQTSidecarTrainRunnerResult{IOReport: ioReport, Preflight: preflight, Metrics: metrics, ActualCoordinateTrain: metrics.Summary.ActualCoordinateTrain, ActualCoordinateTrainSHA256: metrics.Summary.ActualCoordinateTrainSHA256, ActualCoordinateTrainReceiptJSONPath: cfg.actualCoordinateTrainReceiptPath()}
	if cfg.actualCoordinateTrainMode() && metrics.Summary.ActualCoordinateTrain != nil {
		if err := writeAOQTActualCoordinateTrainReceiptFile(cfg.actualCoordinateTrainReceiptPath(), *metrics.Summary.ActualCoordinateTrain); err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			return AOQTSidecarTrainRunnerResult{}, err
		}
	}
	if cfg.isDevOnlyRun() {
		devEvidence, err := writeAOQTSidecarDevEvidence(cfg, preflightSHA256, trainer.Transform(), splitBinding)
		if err != nil {
			if cfg.actualCoordinateTrainMode() {
				summary = metrics.Summary
				return finishR5Failure(err)
			}
			if removeErr := os.Remove(cfg.MetricsJSONPath); removeErr != nil && !os.IsNotExist(removeErr) {
				return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; failed to roll back AOQT metrics output %q: %v", err, cfg.MetricsJSONPath, removeErr)
			}
			return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; rolled back AOQT metrics output %q", err, cfg.MetricsJSONPath)
		}
		result.DevEvidence = &devEvidence
	} else if !cfg.PlanOnly {
		packageResult, err := WriteAOQTSidecarCandidatePackage(AOQTSidecarCandidatePackageConfig{
			AnchorArtifactPath:                  set.Manifest.AnchorArtifactPath,
			OutputArtifactPath:                  cfg.OutputArtifactPath,
			ExpectedAnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			ExpectedAnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:              set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:               metrics.Inputs.DatasetManifestSHA256,
			QrelsSHA256ByDataset:                cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:                 set.Manifest.CompatibilityDigest,
			Transform:                           trainer.Transform(),
		})
		if err != nil {
			if removeErr := os.Remove(cfg.MetricsJSONPath); removeErr != nil && !os.IsNotExist(removeErr) {
				return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; failed to roll back AOQT metrics output %q: %v", err, cfg.MetricsJSONPath, removeErr)
			}
			return AOQTSidecarTrainRunnerResult{}, fmt.Errorf("%w; rolled back AOQT metrics output %q", err, cfg.MetricsJSONPath)
		}
		result.PackageResult = &packageResult
	}
	return result, nil
}

func aoqtRunnerObjectiveFromContract(contract AOQTSidecarObjectiveContract) (AOQTSidecarVectorObjective, error) {
	objective, err := objectiveFromAOQTContract(contract)
	if err != nil {
		return nil, err
	}
	return objective, nil
}

func validateAOQTSidecarTrainRunnerConfig(cfg AOQTSidecarTrainRunnerConfig) error {
	if cfg.ForwardConsistencyProbeOnly && (cfg.ActualDirectionProbeOnly || cfg.ActualCoordinateProbeOnly) {
		return fmt.Errorf("AOQT forward-consistency and actual probe-only modes are mutually exclusive")
	}
	if cfg.ActualDirectionProbeOnly && cfg.ActualCoordinateProbeOnly {
		return fmt.Errorf("AOQT actual-direction and actual-coordinate probe-only modes are mutually exclusive")
	}
	if !cfg.AllowResearchOnly {
		return fmt.Errorf("AOQT sidecar training requires explicit research-only train authorization")
	}
	if err := validateAOQTTrainRunnerDevMode(cfg); err != nil {
		return err
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"manifest", cfg.ManifestPath},
		{"rows", cfg.RowsJSONLPath},
		{"preflight", cfg.PreflightJSONPath},
		{"expected-manifest-sha256", cfg.ExpectedManifestSHA256},
		{"expected-rows-sha256", cfg.ExpectedRowsSHA256},
		{"expected-preflight-sha256", cfg.ExpectedPreflightSHA256},
		{"expected-anchor-artifact-sha256", cfg.ExpectedAnchorArtifactSHA256},
		{"expected-anchor-package-manifest-sha256", cfg.ExpectedAnchorPackageManifestSHA256},
		{"anchor-embedding-space-id", cfg.ExpectedAnchorEmbeddingSpaceID},
		{"compatibility-digest", cfg.ExpectedCompatibilityDigest},
	} {
		if strings.TrimSpace(item.value) == "" {
			return fmt.Errorf("AOQT sidecar training requires --%s", item.name)
		}
	}
	if !cfg.ForwardConsistencyProbeOnly && !cfg.ActualDirectionProbeOnly && !cfg.ActualCoordinateProbeOnly {
		if strings.TrimSpace(cfg.MetricsJSONPath) == "" {
			return fmt.Errorf("AOQT sidecar training requires --metrics-json")
		}
		if err := ensureAOQTFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
			return err
		}
		if cfg.isDevOnlyRun() {
			if err := ensureAOQTDevFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if cfg.actualCoordinateTrainMode() {
				if err := ensureAOQTActualCoordinateTrainReceiptAbsent(cfg.actualCoordinateTrainReceiptPath()); err != nil {
					return err
				}
			}
		}
		if err := ensureAOQTPostFitFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
			return err
		}
		if err := ensureAOQTRunMetricsAbsent(cfg.MetricsJSONPath); err != nil {
			return err
		}
	} else if cfg.ForwardConsistencyProbeOnly {
		if strings.TrimSpace(cfg.ForwardConsistencyProbeReceiptJSONPath) == "" {
			return fmt.Errorf("AOQT forward-consistency probe-only mode requires --forward-consistency-probe-receipt-json")
		}
		if strings.TrimSpace(cfg.DevOutputDir) != "" {
			return fmt.Errorf("AOQT forward-consistency probe-only mode must not configure --dev-output-dir")
		}
		if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT forward-consistency probe-only mode must not configure --output")
		}
		if cfg.MaxSteps < 0 {
			return fmt.Errorf("AOQT forward-consistency probe-only max steps must be non-negative")
		}
		if err := ensureAOQTForwardConsistencyProbeReceiptAbsent(cfg.ForwardConsistencyProbeReceiptJSONPath); err != nil {
			return err
		}
		if strings.TrimSpace(cfg.MetricsJSONPath) != "" {
			if err := ensureAOQTFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTDevFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTPostFitFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTRunMetricsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
		}
	} else if cfg.ActualDirectionProbeOnly {
		if strings.TrimSpace(cfg.ActualDirectionProbeReceiptJSONPath) == "" {
			return fmt.Errorf("AOQT actual-direction probe-only mode requires --actual-direction-probe-receipt-json")
		}
		if strings.TrimSpace(cfg.DevOutputDir) != "" {
			return fmt.Errorf("AOQT actual-direction probe-only mode must not configure --dev-output-dir")
		}
		if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT actual-direction probe-only mode must not configure --output")
		}
		if cfg.MaxSteps < 0 {
			return fmt.Errorf("AOQT actual-direction probe-only max steps must be non-negative")
		}
		if err := ensureAOQTActualDirectionProbeReceiptAbsent(cfg.ActualDirectionProbeReceiptJSONPath); err != nil {
			return err
		}
		if strings.TrimSpace(cfg.MetricsJSONPath) != "" {
			if err := ensureAOQTFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTDevFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTPostFitFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTRunMetricsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
		}
	} else {
		if strings.TrimSpace(cfg.ActualCoordinateProbeReceiptJSONPath) == "" {
			return fmt.Errorf("AOQT actual-coordinate probe-only mode requires --actual-coordinate-probe-receipt-json")
		}
		if strings.TrimSpace(cfg.DevOutputDir) != "" {
			return fmt.Errorf("AOQT actual-coordinate probe-only mode must not configure --dev-output-dir")
		}
		if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT actual-coordinate probe-only mode must not configure --output")
		}
		if cfg.MaxSteps < 0 {
			return fmt.Errorf("AOQT actual-coordinate probe-only max steps must be non-negative")
		}
		if err := ensureAOQTActualCoordinateProbeReceiptAbsent(cfg.ActualCoordinateProbeReceiptJSONPath); err != nil {
			return err
		}
		if strings.TrimSpace(cfg.MetricsJSONPath) != "" {
			if err := ensureAOQTFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTDevFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTPostFitFailClosedDiagnosticsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
			if err := ensureAOQTRunMetricsAbsent(cfg.MetricsJSONPath); err != nil {
				return err
			}
		}
	}
	if err := validateAOQTTrainRunnerOutputPathDistinctness(cfg); err != nil {
		return err
	}
	if cfg.PlanOnly {
		if strings.TrimSpace(cfg.OptimizerMode) != "" {
			return fmt.Errorf("AOQT sidecar plan-only does not accept --optimizer-mode")
		}
		if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT sidecar plan-only writes metrics only; --output is not allowed")
		}
		if cfg.MaxSteps < 0 {
			return fmt.Errorf("AOQT sidecar max steps must be non-negative")
		}
	} else if cfg.ForwardConsistencyProbeOnly || cfg.ActualDirectionProbeOnly || cfg.ActualCoordinateProbeOnly {
		if cfg.PlanOnly {
			return fmt.Errorf("AOQT forward-consistency probe-only mode cannot be plan-only")
		}
	} else {
		if !cfg.isDevOnlyRun() && strings.TrimSpace(cfg.OutputArtifactPath) == "" {
			return fmt.Errorf("AOQT sidecar non-plan training requires --output")
		}
		if cfg.isDevOnlyRun() && strings.TrimSpace(cfg.OutputArtifactPath) != "" {
			return fmt.Errorf("AOQT V7 dev-only training writes dev evidence only; --output candidate package is not allowed")
		}
		if cfg.MaxSteps <= 0 {
			return fmt.Errorf("AOQT sidecar non-plan training requires positive --max-steps")
		}
		if cfg.isDevOnlyRun() {
			if err := ensureAOQTDevEvidenceAbsent(cfg.DevOutputDir); err != nil {
				return err
			}
		} else {
			if err := ensureAOQTOutputArtifactAbsent(cfg.OutputArtifactPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func (cfg AOQTSidecarTrainRunnerConfig) isDevOnlyRun() bool {
	return isAOQTV7DevOptimizerMode(cfg.OptimizerMode)
}

func (cfg AOQTSidecarTrainRunnerConfig) actualDirectionProbeRequired() bool {
	return cfg.RequireActualDirectionProbe || cfg.ActualDirectionProbeRequired || cfg.ActualDirectionProbeOnly
}

func (cfg AOQTSidecarTrainRunnerConfig) actualCoordinateProbeRequired() bool {
	return cfg.RequireActualCoordinateProbe || cfg.ActualCoordinateProbeRequired || cfg.ActualCoordinateProbeOnly
}

func (cfg AOQTSidecarTrainRunnerConfig) actualCoordinateTrainMode() bool {
	return cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain
}

func (cfg AOQTSidecarTrainRunnerConfig) actualCoordinateTrainAuthorizationRequested() bool {
	return cfg.RequireActualCoordinateTrain || cfg.ActualCoordinateTrainRequired ||
		strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptJSONPath) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptPath) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateTrainR4ReceiptSHA256) != "" ||
		strings.TrimSpace(cfg.CanonicalR4ReceiptJSONPath) != "" ||
		strings.TrimSpace(cfg.CanonicalR4ReceiptPath) != "" ||
		strings.TrimSpace(cfg.CanonicalR4ReceiptSHA256) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateTrainReceiptJSONPath) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateTrainReceiptPath) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateFullTrainReceiptJSONPath) != "" ||
		strings.TrimSpace(cfg.ActualCoordinateFullTrainReceiptPath) != ""
}

func validateAOQTR5AliasConsistency(label string, values ...string) error {
	want := ""
	for _, value := range values {
		if strings.TrimSpace(value) == "" {
			continue
		}
		if want == "" {
			want = value
			continue
		}
		if value != want {
			return fmt.Errorf("AOQT V7-r5 %s aliases disagree", label)
		}
	}
	return nil
}

func (cfg AOQTSidecarTrainRunnerConfig) actualCoordinateTrainRequired() bool {
	return cfg.RequireActualCoordinateTrain || cfg.ActualCoordinateTrainRequired || cfg.actualCoordinateTrainMode()
}

func (cfg AOQTSidecarTrainRunnerConfig) canonicalR4ReceiptPath() string {
	for _, path := range []string{cfg.ActualCoordinateTrainR4ReceiptJSONPath, cfg.ActualCoordinateTrainR4ReceiptPath, cfg.CanonicalR4ReceiptJSONPath, cfg.CanonicalR4ReceiptPath} {
		if strings.TrimSpace(path) != "" {
			return path
		}
	}
	return ""
}

func (cfg AOQTSidecarTrainRunnerConfig) canonicalR4ReceiptExpectedSHA256() string {
	for _, value := range []string{cfg.ExpectedCanonicalR4ReceiptSHA256, cfg.CanonicalR4ReceiptSHA256, cfg.ActualCoordinateTrainR4ReceiptSHA256} {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return AOQTV7R5CanonicalR4ReceiptSHA256
}

func (cfg AOQTSidecarTrainRunnerConfig) actualCoordinateTrainReceiptPath() string {
	for _, path := range []string{cfg.ActualCoordinateTrainReceiptJSONPath, cfg.ActualCoordinateTrainReceiptPath, cfg.ActualCoordinateFullTrainReceiptJSONPath, cfg.ActualCoordinateFullTrainReceiptPath} {
		if strings.TrimSpace(path) != "" {
			return path
		}
	}
	return ""
}

func validateAOQTTrainRunnerDevMode(cfg AOQTSidecarTrainRunnerConfig) error {
	if cfg.actualCoordinateTrainAuthorizationRequested() && !cfg.actualCoordinateTrainMode() {
		return fmt.Errorf("AOQT V7-r5 actual-coordinate training authorization requires --optimizer-mode=%s", AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain)
	}
	switch strings.TrimSpace(cfg.OptimizerMode) {
	case "":
		if cfg.AllowDevOnly || strings.TrimSpace(cfg.DevOutputDir) != "" || cfg.RequireForwardConsistencyProbe || cfg.ForwardConsistencyProbeOnly || cfg.actualDirectionProbeRequired() || cfg.actualCoordinateProbeRequired() || cfg.actualCoordinateTrainAuthorizationRequested() {
			return fmt.Errorf("AOQT dev-only flags require --optimizer-mode=%s", AOQTSidecarOptimizerModeV7DevTrustRegion)
		}
	case AOQTSidecarOptimizerModeV7DevTrustRegion, AOQTSidecarOptimizerModeV7R2DevTrustRegion, AOQTSidecarOptimizerModeV7R3DevActualDirection, AOQTSidecarOptimizerModeV7R4DevActualCoordinate, AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain:
		if cfg.OptimizerMode != strings.TrimSpace(cfg.OptimizerMode) {
			return fmt.Errorf("AOQT optimizer mode must be canonical without surrounding whitespace")
		}
		if !cfg.AllowDevOnly {
			return fmt.Errorf("AOQT V7 dev optimizer mode requires explicit dev-only authorization")
		}
		if !cfg.ForwardConsistencyProbeOnly && !cfg.ActualDirectionProbeOnly && !cfg.ActualCoordinateProbeOnly && strings.TrimSpace(cfg.DevOutputDir) == "" {
			return fmt.Errorf("AOQT V7 dev optimizer mode requires --dev-output-dir")
		}
		if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R2DevTrustRegion && !cfg.RequireForwardConsistencyProbe {
			return fmt.Errorf("AOQT V7-r2 optimizer mode requires --require-forward-consistency-probe")
		}
		if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7DevTrustRegion && cfg.RequireForwardConsistencyProbe {
			return fmt.Errorf("AOQT forward-consistency probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R2DevTrustRegion)
		}
		if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
			if cfg.RequireForwardConsistencyProbe || cfg.ForwardConsistencyProbeOnly {
				return fmt.Errorf("AOQT V7-r3 actual-direction mode cannot use the V7-r2 forward-consistency probe")
			}
			if !cfg.actualDirectionProbeRequired() {
				return fmt.Errorf("AOQT V7-r3 optimizer mode requires --require-actual-direction-probe")
			}
		}
		if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R4DevActualCoordinate {
			if cfg.RequireForwardConsistencyProbe || cfg.ForwardConsistencyProbeOnly || cfg.actualDirectionProbeRequired() {
				return fmt.Errorf("AOQT V7-r4 actual-coordinate mode cannot use an older probe")
			}
			if !cfg.ActualCoordinateProbeOnly || !cfg.actualCoordinateProbeRequired() {
				return fmt.Errorf("AOQT V7-r4 optimizer mode requires --actual-coordinate-probe-only")
			}
		}
		if cfg.actualCoordinateTrainMode() {
			if cfg.RequireForwardConsistencyProbe || cfg.ForwardConsistencyProbeOnly || cfg.actualDirectionProbeRequired() || cfg.actualCoordinateProbeRequired() {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training mode cannot use an older probe")
			}
			if cfg.PlanOnly {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training mode cannot be plan-only")
			}
			if cfg.MaxSteps != AOQTV7R5ActualCoordinateTrainMaxSteps {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires max_steps exactly %d", AOQTV7R5ActualCoordinateTrainMaxSteps)
			}
			if cfg.LearningRate != float32(0.01) {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires learning rate exactly 0.01")
			}
			if err := validateAOQTR5AliasConsistency("canonical r4 receipt path", cfg.ActualCoordinateTrainR4ReceiptJSONPath, cfg.ActualCoordinateTrainR4ReceiptPath, cfg.CanonicalR4ReceiptJSONPath, cfg.CanonicalR4ReceiptPath); err != nil {
				return err
			}
			if err := validateAOQTR5AliasConsistency("canonical r4 receipt sha256", cfg.ExpectedCanonicalR4ReceiptSHA256, cfg.CanonicalR4ReceiptSHA256, cfg.ActualCoordinateTrainR4ReceiptSHA256); err != nil {
				return err
			}
			if err := validateAOQTR5AliasConsistency("full-training receipt output path", cfg.ActualCoordinateTrainReceiptJSONPath, cfg.ActualCoordinateTrainReceiptPath, cfg.ActualCoordinateFullTrainReceiptJSONPath, cfg.ActualCoordinateFullTrainReceiptPath); err != nil {
				return err
			}
			if strings.TrimSpace(cfg.canonicalR4ReceiptPath()) == "" {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires the canonical V7-r4 receipt path")
			}
			if got := cfg.canonicalR4ReceiptExpectedSHA256(); got != AOQTV7R5CanonicalR4ReceiptSHA256 {
				return fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt sha256 pin = %q, want %q", got, AOQTV7R5CanonicalR4ReceiptSHA256)
			}
			if strings.TrimSpace(cfg.actualCoordinateTrainReceiptPath()) == "" {
				return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires a full-training receipt output path")
			}
		}
		if cfg.actualDirectionProbeRequired() && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection {
			return fmt.Errorf("AOQT actual-direction probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R3DevActualDirection)
		}
		if cfg.actualCoordinateProbeRequired() && cfg.OptimizerMode != AOQTSidecarOptimizerModeV7R4DevActualCoordinate {
			return fmt.Errorf("AOQT actual-coordinate probe requires optimizer_mode=%s", AOQTSidecarOptimizerModeV7R4DevActualCoordinate)
		}
	default:
		return fmt.Errorf("AOQT optimizer_mode %q is unsupported", cfg.OptimizerMode)
	}
	splitProvided := strings.TrimSpace(cfg.SplitManifestPath) != "" || strings.TrimSpace(cfg.ExpectedSplitManifestSHA256) != "" || strings.TrimSpace(cfg.FoldID) != ""
	if splitProvided && !cfg.isDevOnlyRun() {
		return fmt.Errorf("AOQT split manifest/fold binding is only allowed for explicit V7 dev-only training")
	}
	if splitProvided {
		if strings.TrimSpace(cfg.SplitManifestPath) == "" {
			return fmt.Errorf("AOQT V7 dev split binding requires --split-manifest")
		}
		if strings.TrimSpace(cfg.ExpectedSplitManifestSHA256) == "" {
			return fmt.Errorf("AOQT V7 dev split binding requires --expected-split-manifest-sha256")
		}
		if strings.TrimSpace(cfg.FoldID) == "" {
			return fmt.Errorf("AOQT V7 dev split binding requires --fold-id")
		}
		if err := validateAOQTSHA256(cfg.ExpectedSplitManifestSHA256, "AOQT V7 dev expected split manifest sha256"); err != nil {
			return err
		}
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R2DevTrustRegion && !splitProvided {
		return fmt.Errorf("AOQT V7-r2 forward-consistency probe requires split manifest/fold binding")
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection && !splitProvided {
		return fmt.Errorf("AOQT V7-r3 actual-direction probe requires split manifest/fold binding")
	}
	if cfg.OptimizerMode == AOQTSidecarOptimizerModeV7R4DevActualCoordinate && !splitProvided {
		return fmt.Errorf("AOQT V7-r4 actual-coordinate probe requires split manifest/fold binding")
	}
	if cfg.actualCoordinateTrainMode() && !splitProvided {
		return fmt.Errorf("AOQT V7-r5 actual-coordinate training requires split manifest/fold binding")
	}
	return nil
}

func expectedAOQTSidecarTrainTopology() (AOQTSidecarTopologyBinding, error) {
	transform, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, AOQTSidecarMaterializerTopologySeed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		return AOQTSidecarTopologyBinding{}, err
	}
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarTopologyBinding{}, err
	}
	return AOQTSidecarTopologyBinding{
		Kind:           AOQTTopologyKindGivensV1,
		Dim:            AOQTSidecarDim,
		Stages:         AOQTSidecarStages,
		PairsPerStage:  AOQTSidecarPairsPerStage,
		AngleCount:     AOQTSidecarAngleCount,
		Seed:           AOQTSidecarMaterializerTopologySeed,
		PairingsSHA256: pairings,
	}, nil
}

func loadAndValidateAOQTTrainPreflight(cfg AOQTSidecarTrainRunnerConfig, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport) (AOQTSidecarMaterializePreflight, string, error) {
	if err := validateAOQTSHA256(cfg.ExpectedPreflightSHA256, "AOQT sidecar expected preflight sha256"); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", err
	}
	data, err := os.ReadFile(cfg.PreflightJSONPath)
	if err != nil {
		return AOQTSidecarMaterializePreflight{}, "", err
	}
	preflightSHA256 := sha256BytesAOQT(data)
	if preflightSHA256 != cfg.ExpectedPreflightSHA256 {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight sha256 mismatch")
	}
	var preflight AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(data, &preflight); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("%s: invalid AOQT materializer preflight JSON: %w", cfg.PreflightJSONPath, err)
	}
	if preflight.Schema != AOQTSidecarMaterializerPreflightSchema {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight schema %q is not supported", preflight.Schema)
	}
	if preflight.QualityClaim || !preflight.ResearchOnly || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight must be research-only and record no training/eval/vector export execution")
	}
	if preflight.TurboQuantSeed != AOQTSidecarMaterializerQuantSeed {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight turboquant seed = %d, want %d", preflight.TurboQuantSeed, AOQTSidecarMaterializerQuantSeed)
	}
	if preflight.TopologySeed != AOQTSidecarMaterializerTopologySeed {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight topology seed = %d, want %d", preflight.TopologySeed, AOQTSidecarMaterializerTopologySeed)
	}
	if preflight.RowCount != len(set.Rows) || preflight.CandidateCount != ioReport.CandidateCount || preflight.PairCount != ioReport.PairCount {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight workload counts do not match calibration IO")
	}
	if preflight.CalibrationManifestSHA256 != ioReport.ManifestSHA256 {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight calibration manifest sha256 mismatch")
	}
	if !aoqtObjectiveContractsEqual(preflight.ObjectiveContract, set.Manifest.ObjectiveContract) {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight objective contract mismatch")
	}
	if err := validateAOQTSidecarTrainingContractBinding(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy, preflight.TrainingContract, preflight.CandidateEligibilityPolicy); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight training contract mismatch: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(set.Manifest.TrainingContract, set.Manifest.CandidateEligibilityPolicy, ioReport.TrainingContract, ioReport.CandidateEligibilityPolicy); err != nil {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar IO training contract mismatch: %w", err)
	}
	if preflight.LegalGates != set.Manifest.LegalGates {
		return AOQTSidecarMaterializePreflight{}, "", fmt.Errorf("AOQT sidecar preflight legal gates mismatch")
	}
	return preflight, preflightSHA256, nil
}

func validateAOQTV7DevSplitBinding(cfg AOQTSidecarTrainRunnerConfig, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflight AOQTSidecarMaterializePreflight, preflightSHA256 string) (*AOQTSidecarDevSplitBinding, error) {
	if strings.TrimSpace(cfg.SplitManifestPath) == "" {
		return nil, nil
	}
	data, err := os.ReadFile(cfg.SplitManifestPath)
	if err != nil {
		return nil, fmt.Errorf("read AOQT V7 dev split manifest: %w", err)
	}
	fileSHA := sha256BytesAOQT(data)
	if fileSHA != cfg.ExpectedSplitManifestSHA256 {
		return nil, fmt.Errorf("AOQT V7 dev split manifest sha256 mismatch")
	}
	var manifest map[string]any
	if err := strictUnmarshalAOQTUseNumber(data, &manifest); err != nil {
		return nil, fmt.Errorf("%s: invalid AOQT V7 dev split manifest JSON: %w", cfg.SplitManifestPath, err)
	}
	if got := requireAOQTStringField(manifest, "schema", "AOQT V7 dev split manifest schema"); got != AOQTV7DevSplitManifestSchema {
		return nil, fmt.Errorf("AOQT V7 dev split manifest schema %q is not supported, want %q", got, AOQTV7DevSplitManifestSchema)
	}
	for _, field := range []string{"actual_training_ran", "actual_eval_ran", "actual_official_data_eval_ran"} {
		if got, ok := manifest[field].(bool); !ok || got {
			return nil, fmt.Errorf("AOQT V7 dev split manifest %s must be false", field)
		}
	}
	payloadSHA, err := aoqtV7SplitPayloadSHA256(manifest)
	if err != nil {
		return nil, err
	}
	provenance, err := requireAOQTMapField(manifest, "provenance", "AOQT V7 dev split manifest provenance")
	if err != nil {
		return nil, err
	}
	if declared := requireAOQTStringField(provenance, "manifest_sha256", "AOQT V7 dev split manifest provenance.manifest_sha256"); declared != payloadSHA {
		return nil, fmt.Errorf("AOQT V7 dev split manifest provenance.manifest_sha256 does not bind split payload")
	}
	if err := validateAOQTSHA256(payloadSHA, "AOQT V7 dev split manifest payload sha256"); err != nil {
		return nil, err
	}
	sourcePlan, err := requireAOQTMapField(manifest, "source_plan", "AOQT V7 dev split manifest source_plan")
	if err != nil {
		return nil, err
	}
	officialRegistry, err := requireAOQTMapField(manifest, "official_qid_registry", "AOQT V7 dev split manifest official_qid_registry")
	if err != nil {
		return nil, err
	}
	fold, err := requireAOQTV7SplitFold(manifest, cfg.FoldID)
	if err != nil {
		return nil, err
	}
	train, err := requireAOQTMapField(fold, "train", "AOQT V7 dev split manifest fold train")
	if err != nil {
		return nil, err
	}
	rowIDs, err := requireAOQTStringArrayField(train, "row_ids", "AOQT V7 dev split manifest fold train row_ids")
	if err != nil {
		return nil, err
	}
	if len(rowIDs) == 0 {
		return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train.row_ids is empty", cfg.FoldID)
	}
	if !slices.IsSorted(rowIDs) {
		return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train.row_ids must be sorted", cfg.FoldID)
	}
	trainRowCount, err := requireAOQTIntField(train, "row_count", "AOQT V7 dev split manifest fold train row_count")
	if err != nil {
		return nil, err
	}
	if trainRowCount != len(rowIDs) {
		return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train.row_count = %d, want row_ids count %d", cfg.FoldID, trainRowCount, len(rowIDs))
	}
	trainRowIDsSHA, err := sha256JSONAOQT(rowIDs)
	if err != nil {
		return nil, fmt.Errorf("hash AOQT V7 dev split fold train row_ids: %w", err)
	}
	if declared := requireAOQTStringField(train, "row_ids_sha256", "AOQT V7 dev split manifest fold train row_ids_sha256"); declared != trainRowIDsSHA {
		return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train.row_ids_sha256 does not bind train.row_ids", cfg.FoldID)
	}
	materializedRowIDs := make([]string, 0, len(set.Rows))
	for _, row := range set.Rows {
		materializedRowIDs = append(materializedRowIDs, row.RowID)
	}
	if !slices.Equal(materializedRowIDs, rowIDs) {
		return nil, fmt.Errorf("AOQT V7 dev split fold %q train row_ids do not exactly match materialized calibration rows", cfg.FoldID)
	}
	materializedRowIDSHA := aoqtRowIDSHA256(materializedRowIDs)
	if materializedRowIDSHA != set.Manifest.RowIDSHA256 {
		return nil, fmt.Errorf("AOQT V7 dev split materialized row_id_sha256 mismatch")
	}
	if set.Manifest.RowCount != trainRowCount || preflight.RowCount != trainRowCount || ioReport.RowCount != trainRowCount {
		return nil, fmt.Errorf("AOQT V7 dev split fold %q train row_count does not match materialized manifest/rows/preflight", cfg.FoldID)
	}
	qidsByDataset, err := requireAOQTStringArrayMapField(train, "qids_by_dataset", "AOQT V7 dev split manifest fold train qids_by_dataset")
	if err != nil {
		return nil, err
	}
	qidCounts, err := requireAOQTIntMapField(train, "qid_count_by_dataset", "AOQT V7 dev split manifest fold train qid_count_by_dataset")
	if err != nil {
		return nil, err
	}
	for dataset, qids := range qidsByDataset {
		if !slices.IsSorted(qids) {
			return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train qids_by_dataset.%s must be sorted", cfg.FoldID, dataset)
		}
		if qidCounts[dataset] != len(qids) {
			return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train qid_count_by_dataset.%s does not match qids", cfg.FoldID, dataset)
		}
	}
	qidsByDatasetSHA, err := sha256JSONAOQT(qidsByDataset)
	if err != nil {
		return nil, fmt.Errorf("hash AOQT V7 dev split fold train qids_by_dataset: %w", err)
	}
	if declared := requireAOQTStringField(train, "qids_by_dataset_sha256", "AOQT V7 dev split manifest fold train qids_by_dataset_sha256"); declared != qidsByDatasetSHA {
		return nil, fmt.Errorf("AOQT V7 dev split manifest fold %q train.qids_by_dataset_sha256 does not bind qids", cfg.FoldID)
	}
	qidHashes := make(map[string]string, len(qidsByDataset))
	for dataset, qids := range qidsByDataset {
		sum, err := sha256JSONAOQT(qids)
		if err != nil {
			return nil, fmt.Errorf("hash AOQT V7 dev split fold train qids for %s: %w", dataset, err)
		}
		qidHashes[dataset] = sum
	}
	binding := AOQTSidecarDevSplitBinding{
		Schema:                       AOQTV7TrainSplitBindingSchema,
		SplitManifestSchema:          AOQTV7DevSplitManifestSchema,
		SplitManifestPath:            cfg.SplitManifestPath,
		SplitManifestFileSHA256:      fileSHA,
		SplitManifestSHA256:          fileSHA,
		SplitManifestPayloadSHA256:   payloadSHA,
		SourcePlanSHA256:             requireAOQTStringField(sourcePlan, "sha256", "AOQT V7 dev split manifest source_plan.sha256"),
		SourcePlanRowsSHA256:         optionalAOQTStringField(sourcePlan, "rows_sha256"),
		SourcePlanRowIDSHA256:        optionalAOQTStringField(sourcePlan, "row_ids_sha256"),
		OfficialQIDRegistrySHA256:    requireAOQTStringField(officialRegistry, "manifest_sha256", "AOQT V7 dev split manifest official_qid_registry.manifest_sha256"),
		OfficialRegistrySourceSHA256: requireAOQTStringField(officialRegistry, "source_sha256", "AOQT V7 dev split manifest official_qid_registry.source_sha256"),
		FoldID:                       cfg.FoldID,
		TrainRowCount:                trainRowCount,
		TrainRowIDSHA256:             trainRowIDsSHA,
		TrainQIDCountByDataset:       cloneAOQTIntMap(qidCounts),
		TrainQIDSetSHA256ByDataset:   cloneAOQTStringMap(qidHashes),
		TrainQIDsByDatasetSHA256:     qidsByDatasetSHA,
		MaterializedManifestSHA256:   ioReport.ManifestSHA256,
		MaterializedRowsSHA256:       ioReport.RowsSHA256,
		MaterializedPreflightSHA256:  preflightSHA256,
		MaterializedRowIDSHA256:      materializedRowIDSHA,
		MaterializedRowCount:         len(set.Rows),
	}
	if err := binding.Validate(); err != nil {
		return nil, err
	}
	return &binding, nil
}

func strictUnmarshalAOQTUseNumber(data []byte, out any) error {
	if err := rejectDuplicateObjectKeysAOQT(data); err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var extra struct{}
	if err := dec.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing JSON payload")
	}
	return nil
}

func aoqtV7SplitPayloadSHA256(manifest map[string]any) (string, error) {
	data, err := json.Marshal(manifest)
	if err != nil {
		return "", fmt.Errorf("encode AOQT V7 dev split manifest for hash: %w", err)
	}
	var comparable map[string]any
	if err := strictUnmarshalAOQTUseNumber(data, &comparable); err != nil {
		return "", fmt.Errorf("normalize AOQT V7 dev split manifest for hash: %w", err)
	}
	provenance, ok := comparable["provenance"].(map[string]any)
	if !ok {
		return "", fmt.Errorf("AOQT V7 dev split manifest provenance is required")
	}
	delete(provenance, "manifest_sha256")
	return sha256JSONAOQT(comparable)
}

func requireAOQTV7SplitFold(manifest map[string]any, foldID string) (map[string]any, error) {
	folds, ok := manifest["folds"].([]any)
	if !ok || len(folds) == 0 {
		return nil, fmt.Errorf("AOQT V7 dev split manifest folds must be a non-empty array")
	}
	for index, value := range folds {
		fold, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("AOQT V7 dev split manifest folds[%d] must be an object", index)
		}
		if requireAOQTStringField(fold, "name", "AOQT V7 dev split manifest fold name") == foldID {
			return fold, nil
		}
	}
	return nil, fmt.Errorf("AOQT V7 dev split manifest does not contain fold %q", foldID)
}

func requireAOQTMapField(parent map[string]any, key, label string) (map[string]any, error) {
	value, ok := parent[key].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", label)
	}
	return value, nil
}

func requireAOQTStringField(parent map[string]any, key, label string) string {
	value, _ := parent[key].(string)
	if strings.TrimSpace(value) == "" {
		return ""
	}
	return value
}

func optionalAOQTStringField(parent map[string]any, key string) string {
	value, _ := parent[key].(string)
	return value
}

func requireAOQTIntField(parent map[string]any, key, label string) (int, error) {
	switch value := parent[key].(type) {
	case json.Number:
		number, err := strconv.ParseInt(value.String(), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("%s must be an integer", label)
		}
		return int(number), nil
	case float64:
		if value != float64(int(value)) {
			return 0, fmt.Errorf("%s must be an integer", label)
		}
		return int(value), nil
	case int:
		return value, nil
	default:
		return 0, fmt.Errorf("%s must be an integer", label)
	}
}

func requireAOQTStringArrayField(parent map[string]any, key, label string) ([]string, error) {
	values, ok := parent[key].([]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an array", label)
	}
	out := make([]string, 0, len(values))
	for index, value := range values {
		text, ok := value.(string)
		if !ok || strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("%s[%d] must be a non-empty string", label, index)
		}
		out = append(out, text)
	}
	return out, nil
}

func requireAOQTStringArrayMapField(parent map[string]any, key, label string) (map[string][]string, error) {
	raw, ok := parent[key].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", label)
	}
	out := make(map[string][]string, len(raw))
	for dataset, value := range raw {
		child := map[string]any{"value": value}
		qids, err := requireAOQTStringArrayField(child, "value", label+"."+dataset)
		if err != nil {
			return nil, err
		}
		out[dataset] = qids
	}
	return out, nil
}

func requireAOQTIntMapField(parent map[string]any, key, label string) (map[string]int, error) {
	raw, ok := parent[key].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", label)
	}
	out := make(map[string]int, len(raw))
	for dataset, value := range raw {
		child := map[string]any{"value": value}
		number, err := requireAOQTIntField(child, "value", label+"."+dataset)
		if err != nil {
			return nil, err
		}
		out[dataset] = number
	}
	return out, nil
}

func cloneAOQTIntMap(in map[string]int) map[string]int {
	if in == nil {
		return nil
	}
	out := make(map[string]int, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func objectiveFromAOQTContract(contract AOQTSidecarObjectiveContract) (AOQTSidecarPreparedIPObjective, error) {
	return NewAOQTSidecarPreparedIPObjective(AOQTSidecarPreparedIPObjectiveConfig{
		Dim:              contract.Dim,
		TurboQuantSeed:   contract.TurboQuantSeed,
		GainBit:          contract.GainBit,
		Q3GuardBit:       contract.Q3GuardBit,
		Q5GuardBit:       contract.Q5GuardBit,
		GainCutoff:       contract.GainCutoff,
		GainTau:          contract.GainTau,
		GainMargin:       contract.GainMargin,
		GuardTau:         contract.GuardTau,
		GuardMargin:      contract.GuardMargin,
		ScoreDistillTau:  contract.ScoreDistillTau,
		NFBoundarySource: contract.NFBoundarySource,
	})
}

func validateAOQTTrainRunnerOutputPathDistinctness(cfg AOQTSidecarTrainRunnerConfig) error {
	paths := map[string]string{
		"metrics":                              cfg.MetricsJSONPath,
		"fail-closed diagnostics":              AOQTFailClosedDiagnosticsPath(cfg.MetricsJSONPath),
		"post-fit fail-closed diagnostics":     AOQTPostFitFailClosedDiagnosticsPath(cfg.MetricsJSONPath),
		"manifest input":                       cfg.ManifestPath,
		"rows input":                           cfg.RowsJSONLPath,
		"preflight input":                      cfg.PreflightJSONPath,
		"forward-consistency probe receipt":    cfg.ForwardConsistencyProbeReceiptJSONPath,
		"actual-direction probe receipt":       cfg.ActualDirectionProbeReceiptJSONPath,
		"actual-coordinate probe receipt":      cfg.ActualCoordinateProbeReceiptJSONPath,
		"canonical r4 receipt input":           cfg.canonicalR4ReceiptPath(),
		"actual-coordinate full-train receipt": cfg.actualCoordinateTrainReceiptPath(),
	}
	if cfg.isDevOnlyRun() {
		paths["dev fail-closed diagnostics"] = AOQTDevFailClosedDiagnosticsPath(cfg.MetricsJSONPath)
	}
	if strings.TrimSpace(cfg.OutputArtifactPath) != "" {
		for role, path := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(cfg.OutputArtifactPath, true)) {
			paths[role] = path
		}
	}
	if strings.TrimSpace(cfg.DevOutputDir) != "" {
		for role, path := range aoqtDevEvidencePaths(cfg.DevOutputDir) {
			paths[role] = path
		}
	}
	roles := make([]string, 0, len(paths))
	for role := range paths {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	seen := make(map[string]string, len(paths))
	for _, role := range roles {
		path := paths[role]
		if strings.TrimSpace(path) == "" {
			continue
		}
		canonical, err := canonicalAOQTPath(path)
		if err != nil {
			return fmt.Errorf("resolve AOQT sidecar %s output path %q: %w", role, path, err)
		}
		if prior, ok := seen[canonical]; ok {
			return fmt.Errorf("AOQT sidecar output path collision: %s %q conflicts with %s %q", role, path, prior, paths[prior])
		}
		seen[canonical] = role
	}
	return nil
}

// canonicalAOQTPath compares both lexical paths and symlink-resolved paths.
// Output validation runs before candidate files exist, so resolve the nearest
// existing parent when the complete path is not yet present.
func canonicalAOQTPath(path string) (string, error) {
	abs, err := filepath.Abs(filepath.Clean(path))
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		return filepath.Clean(resolved), nil
	}
	dir, base := filepath.Split(abs)
	if resolvedDir, err := filepath.EvalSymlinks(filepath.Clean(dir)); err == nil {
		return filepath.Join(resolvedDir, base), nil
	}
	return abs, nil
}

// NewAOQTSidecarFailClosedDiagnostics is retained for source compatibility,
// but cannot construct evidence safely without the authentic raw preflight
// file and hash. The training runner is the only supported constructor.
func NewAOQTSidecarFailClosedDiagnostics(set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, summary AOQTSidecarTrainSummary, failure error) (AOQTSidecarFailClosedDiagnostics, error) {
	return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require authentic materializer preflight provenance; construct them through the training runner")
}

func newAOQTFailClosedDiagnostics(set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflight AOQTSidecarMaterializePreflight, preflightPath, preflightSHA256 string, summary AOQTSidecarTrainSummary, failure error) (AOQTSidecarFailClosedDiagnostics, error) {
	if failure == nil {
		return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require a failure")
	}
	if err := set.Validate(); err != nil {
		return AOQTSidecarFailClosedDiagnostics{}, err
	}
	if summary.OptimizerDiagnostics == nil || summary.OptimizerDiagnostics.AcceptedSteps != 0 {
		return AOQTSidecarFailClosedDiagnostics{}, fmt.Errorf("AOQT fail-closed diagnostics require zero accepted optimizer steps")
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return AOQTSidecarFailClosedDiagnostics{}, err
	}
	diagnostics := AOQTSidecarFailClosedDiagnostics{
		Schema:           AOQTSidecarFailClosedDiagnosticsSchema,
		FailureKind:      "zero_accepted_safe_steps",
		Error:            failure.Error(),
		RejectionSummary: summary.OptimizerDiagnostics.RejectionSummary(),
		Plan:             summary.Plan,
		Inputs: AOQTSidecarRunMetricInputs{
			AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:       manifestHash,
			QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:         set.Manifest.CompatibilityDigest,
		},
		IOReport:                   ioReport,
		Preflight:                  preflight,
		PreflightPath:              preflightPath,
		PreflightSHA256:            preflightSHA256,
		Topology:                   set.Manifest.Topology,
		ObjectiveContract:          set.Manifest.ObjectiveContract,
		LegalGates:                 set.Manifest.LegalGates,
		Summary:                    summary,
		QualityClaim:               false,
		TrainingContract:           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	}
	return diagnostics, diagnostics.Validate()
}

func newAOQTDevFailClosedDiagnostics(set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflight AOQTSidecarMaterializePreflight, preflightPath, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding, summary AOQTSidecarTrainSummary, failure error) (AOQTSidecarDevFailClosedDiagnostics, error) {
	if failure == nil {
		return AOQTSidecarDevFailClosedDiagnostics{}, fmt.Errorf("AOQT dev fail-closed diagnostics require a failure")
	}
	if !isAOQTV7DevOptimizerMode(summary.OptimizerMode) {
		return AOQTSidecarDevFailClosedDiagnostics{}, fmt.Errorf("AOQT dev fail-closed diagnostics require a V7 dev optimizer mode")
	}
	if err := set.Validate(); err != nil {
		return AOQTSidecarDevFailClosedDiagnostics{}, err
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return AOQTSidecarDevFailClosedDiagnostics{}, err
	}
	failureKind := "dev_training_failed"
	if summary.ForwardConsistencyProbe != nil && !summary.ForwardConsistencyProbe.Passed {
		failureKind = "probe_failed"
	}
	if summary.ActualDirectionProbe != nil && !summary.ActualDirectionProbe.Passed {
		failureKind = "actual_direction_probe_failed"
	}
	if isAOQTV7R5ActualCoordinateMode(summary.OptimizerMode) && summary.ActualCoordinateTrain != nil && !summary.ActualCoordinateTrain.Passed {
		failureKind = "actual_coordinate_train_failed"
	}
	diagnostics := AOQTSidecarDevFailClosedDiagnostics{
		Schema:      AOQTSidecarDevFailClosedDiagnosticsSchema,
		FailureKind: failureKind,
		Error:       failure.Error(),
		Plan:        summary.Plan,
		Inputs: AOQTSidecarRunMetricInputs{
			AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
			AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
			AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
			DatasetManifestSHA256:       manifestHash,
			QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
			CompatibilityDigest:         set.Manifest.CompatibilityDigest,
		},
		IOReport:                   ioReport,
		Preflight:                  preflight,
		PreflightPath:              preflightPath,
		PreflightSHA256:            preflightSHA256,
		SplitBinding:               cloneAOQTDevSplitBinding(splitBinding),
		Topology:                   set.Manifest.Topology,
		ObjectiveContract:          set.Manifest.ObjectiveContract,
		LegalGates:                 set.Manifest.LegalGates,
		Summary:                    summary,
		ForwardConsistencyProbe:    summary.ForwardConsistencyProbe,
		ActualDirectionProbe:       summary.ActualDirectionProbe,
		ActualCoordinateTrain:      summary.ActualCoordinateTrain,
		QualityClaim:               false,
		ReleaseClaim:               false,
		OfficialClaim:              false,
		OfficialHeldoutGate:        false,
		CommercialClaim:            false,
		TrainingContract:           set.Manifest.TrainingContract,
		CandidateEligibilityPolicy: cloneAOQTSidecarCandidateEligibilityPolicy(set.Manifest.CandidateEligibilityPolicy),
	}
	if diagnostics.Summary.OptimizerDiagnostics != nil {
		diagnostics.RejectionSummary = diagnostics.Summary.OptimizerDiagnostics.RejectionSummary()
		if diagnostics.ActualDirectionProbe != nil && !diagnostics.ActualDirectionProbe.Passed && diagnostics.RejectionSummary != "" && !strings.Contains(diagnostics.Error, diagnostics.RejectionSummary) {
			diagnostics.Error += ": " + diagnostics.RejectionSummary
		}
	}
	return diagnostics, diagnostics.Validate()
}

func cloneAOQTDevSplitBinding(binding *AOQTSidecarDevSplitBinding) *AOQTSidecarDevSplitBinding {
	if binding == nil {
		return nil
	}
	clone := *binding
	clone.TrainQIDCountByDataset = cloneAOQTIntMap(binding.TrainQIDCountByDataset)
	clone.TrainQIDSetSHA256ByDataset = cloneAOQTStringMap(binding.TrainQIDSetSHA256ByDataset)
	return &clone
}

func (d AOQTSidecarDevFailClosedDiagnostics) Validate() error {
	if d.Schema != AOQTSidecarDevFailClosedDiagnosticsSchema {
		return fmt.Errorf("AOQT dev fail-closed diagnostics schema %q is unsupported, want %q", d.Schema, AOQTSidecarDevFailClosedDiagnosticsSchema)
	}
	if strings.TrimSpace(d.Error) == "" {
		return fmt.Errorf("AOQT dev fail-closed diagnostics error is required")
	}
	if d.QualityClaim || d.ReleaseClaim || d.OfficialClaim || d.OfficialHeldoutGate || d.CommercialClaim || d.Summary.QualityClaim {
		return fmt.Errorf("AOQT dev fail-closed diagnostics quality/release/official/commercial claims must be false")
	}
	if d.FailureKind != "dev_training_failed" && d.FailureKind != "probe_failed" && d.FailureKind != "actual_direction_probe_failed" && d.FailureKind != "actual_coordinate_train_failed" {
		return fmt.Errorf("AOQT dev fail-closed diagnostics failure_kind %q is unsupported", d.FailureKind)
	}
	if d.Plan != d.Summary.Plan {
		return fmt.Errorf("AOQT dev fail-closed diagnostics plan must exactly match summary.plan")
	}
	if d.Plan.PlanOnly || d.Plan.StepCount <= 0 {
		return fmt.Errorf("AOQT dev fail-closed diagnostics require a non-plan optimizer run")
	}
	if !isAOQTV7DevOptimizerMode(d.Plan.OptimizerMode) || d.Plan.OptimizerMode != d.Summary.OptimizerMode {
		return fmt.Errorf("AOQT dev fail-closed diagnostics require a matching V7 dev optimizer mode")
	}
	if d.Summary.ForwardConsistencyProbeRequired != d.Plan.ForwardConsistencyProbeRequired {
		return fmt.Errorf("AOQT dev fail-closed diagnostics probe-required flag must match plan")
	}
	if d.Summary.OptimizerDiagnostics == nil {
		return fmt.Errorf("AOQT dev fail-closed diagnostics optimizer diagnostics are required")
	}
	optimizer := *d.Summary.OptimizerDiagnostics
	if !optimizer.DevOnly || optimizer.OptimizerMode != d.Plan.OptimizerMode {
		return fmt.Errorf("AOQT dev fail-closed diagnostics optimizer diagnostics must be dev-only and match plan")
	}
	if optimizer.PlannedSteps != d.Plan.StepCount {
		return fmt.Errorf("AOQT dev fail-closed diagnostics planned_steps = %d, want plan step_count %d", optimizer.PlannedSteps, d.Plan.StepCount)
	}
	if !isAOQTV7R5ActualCoordinateMode(d.Plan.OptimizerMode) {
		if optimizer.ProposalReceipts == nil {
			return fmt.Errorf("AOQT dev fail-closed diagnostics proposal receipts are required")
		}
		if err := validateAOQTOptimizerPathDiagnostics(optimizer, "AOQT dev fail-closed diagnostics"); err != nil {
			return err
		}
	} else if err := validateAOQTV7R5OptimizerDiagnostics(d.Plan, d.Summary, optimizer); err != nil {
		return err
	}
	if strings.TrimSpace(d.Summary.OptimizerDiagnosticsSHA256) == "" {
		return fmt.Errorf("AOQT dev fail-closed diagnostics optimizer diagnostics sha256 is required")
	}
	optimizerSHA, err := optimizer.SHA256()
	if err != nil {
		return err
	}
	if optimizerSHA != d.Summary.OptimizerDiagnosticsSHA256 {
		return fmt.Errorf("AOQT dev fail-closed diagnostics optimizer diagnostics sha256 mismatch")
	}
	if err := validateAOQTMetricInputs(d.Inputs); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflightRawBinding(d.Preflight, d.PreflightPath, d.PreflightSHA256); err != nil {
		return err
	}
	if err := validateAOQTFailClosedIOReport(d.IOReport, d.Plan, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflight(d.Preflight, d.PreflightPath, d.IOReport, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	if err := d.Topology.Validate(); err != nil {
		return err
	}
	if err := validateAOQTMetricsPlanTopology(d.Plan, d.Topology); err != nil {
		return err
	}
	if !aoqtObjectiveContractsEqual(d.ObjectiveContract, d.Summary.ObjectiveContract) {
		return fmt.Errorf("AOQT dev fail-closed diagnostics objective contract must match summary")
	}
	if err := validateAOQTFailClosedObjectiveContract(d.ObjectiveContract, d.Topology); err != nil {
		return err
	}
	if err := validateAOQTResearchOnlyLegalGates(d.LegalGates, "AOQT dev fail-closed diagnostics"); err != nil {
		return err
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.Summary.TrainingContract, d.Summary.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT dev fail-closed diagnostics training contract: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.IOReport.TrainingContract, d.IOReport.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT dev fail-closed diagnostics IO training contract: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.Preflight.TrainingContract, d.Preflight.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT dev fail-closed diagnostics preflight training contract: %w", err)
	}
	if d.SplitBinding != nil {
		if err := d.SplitBinding.Validate(); err != nil {
			return fmt.Errorf("AOQT dev fail-closed diagnostics split binding: %w", err)
		}
		if d.SplitBinding.MaterializedManifestSHA256 != d.Inputs.DatasetManifestSHA256 || d.SplitBinding.MaterializedRowCount != d.Plan.RowCount {
			return fmt.Errorf("AOQT dev fail-closed diagnostics split binding does not match inputs/plan")
		}
	} else if d.Plan.OptimizerMode == AOQTSidecarOptimizerModeV7R2DevTrustRegion || d.Plan.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection || isAOQTV7R5ActualCoordinateMode(d.Plan.OptimizerMode) {
		return fmt.Errorf("AOQT V7 dev fail-closed diagnostics require split binding")
	}
	if (d.ForwardConsistencyProbe == nil) != (d.Summary.ForwardConsistencyProbe == nil) {
		return fmt.Errorf("AOQT dev fail-closed diagnostics probe must match summary probe presence")
	}
	if d.ForwardConsistencyProbe != nil {
		if !reflect.DeepEqual(d.ForwardConsistencyProbe, d.Summary.ForwardConsistencyProbe) {
			return fmt.Errorf("AOQT dev fail-closed diagnostics probe must exactly match summary probe")
		}
		if err := d.ForwardConsistencyProbe.ValidateBound(); err != nil {
			return err
		}
		probeSHA, err := d.ForwardConsistencyProbe.SHA256()
		if err != nil {
			return err
		}
		if probeSHA != d.Summary.ForwardConsistencyProbeSHA256 {
			return fmt.Errorf("AOQT dev fail-closed diagnostics forward-consistency probe sha256 mismatch")
		}
		if d.FailureKind == "probe_failed" && d.ForwardConsistencyProbe.Passed {
			return fmt.Errorf("AOQT dev fail-closed diagnostics probe_failed requires a failed probe receipt")
		}
		if d.FailureKind == "dev_training_failed" && !d.ForwardConsistencyProbe.Passed {
			return fmt.Errorf("AOQT dev fail-closed diagnostics dev_training_failed cannot carry a failed probe receipt")
		}
	} else if d.Summary.ForwardConsistencyProbeRequired {
		return fmt.Errorf("AOQT V7-r2 dev fail-closed diagnostics require a probe receipt")
	}
	if (d.ActualDirectionProbe == nil) != (d.Summary.ActualDirectionProbe == nil) {
		return fmt.Errorf("AOQT dev fail-closed diagnostics actual-direction probe must match summary probe presence")
	}
	if d.ActualDirectionProbe != nil {
		if !reflect.DeepEqual(d.ActualDirectionProbe, d.Summary.ActualDirectionProbe) {
			return fmt.Errorf("AOQT dev fail-closed diagnostics actual-direction probe must exactly match summary probe")
		}
		if err := d.ActualDirectionProbe.ValidateBound(); err != nil {
			return err
		}
		probeSHA, err := d.ActualDirectionProbe.SHA256()
		if err != nil {
			return err
		}
		if probeSHA != d.Summary.ActualDirectionProbeSHA256 {
			return fmt.Errorf("AOQT dev fail-closed diagnostics actual-direction probe sha256 mismatch")
		}
		if err := validateAOQTActualDirectionProposalProvenance(*d.ActualDirectionProbe, d.Summary.ActualDirectionProbeSHA256, optimizer, "AOQT dev fail-closed diagnostics"); err != nil {
			return err
		}
		if d.FailureKind == "actual_direction_probe_failed" && d.ActualDirectionProbe.Passed {
			return fmt.Errorf("AOQT dev fail-closed diagnostics actual_direction_probe_failed requires a failed actual-direction probe")
		}
		if d.FailureKind == "dev_training_failed" && !d.ActualDirectionProbe.Passed {
			return fmt.Errorf("AOQT dev fail-closed diagnostics dev_training_failed cannot carry a failed actual-direction probe")
		}
	} else if d.Summary.ActualDirectionProbeRequired {
		return fmt.Errorf("AOQT V7-r3 dev fail-closed diagnostics require an actual-direction probe receipt")
	}
	if isAOQTV7R5ActualCoordinateMode(d.Plan.OptimizerMode) {
		if d.FailureKind != "actual_coordinate_train_failed" {
			return fmt.Errorf("AOQT V7-r5 dev fail-closed diagnostics require actual_coordinate_train_failed")
		}
		if d.ActualCoordinateTrain == nil || d.Summary.ActualCoordinateTrain == nil {
			return fmt.Errorf("AOQT V7-r5 dev fail-closed diagnostics require a full-training receipt")
		}
		if !reflect.DeepEqual(d.ActualCoordinateTrain, d.Summary.ActualCoordinateTrain) {
			return fmt.Errorf("AOQT V7-r5 dev fail-closed diagnostics full-training receipt must match summary")
		}
		if d.ActualCoordinateTrain.Passed {
			return fmt.Errorf("AOQT V7-r5 actual_coordinate_train_failed requires a failed receipt")
		}
		if err := d.ActualCoordinateTrain.ValidateBound(); err != nil {
			return err
		}
		trainSHA, err := d.ActualCoordinateTrain.SHA256()
		if err != nil {
			return err
		}
		if trainSHA != d.Summary.ActualCoordinateTrainSHA256 {
			return fmt.Errorf("AOQT V7-r5 dev fail-closed diagnostics full-training receipt sha256 mismatch")
		}
	} else if d.ActualCoordinateTrain != nil || d.Summary.ActualCoordinateTrain != nil || strings.TrimSpace(d.Summary.ActualCoordinateTrainSHA256) != "" {
		return fmt.Errorf("AOQT non-r5 dev fail-closed diagnostics must not include a full-training receipt")
	}
	return nil
}

func (d AOQTSidecarFailClosedDiagnostics) Validate() error {
	if d.Schema != AOQTSidecarFailClosedDiagnosticsSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics schema %q is not supported, want %q", d.Schema, AOQTSidecarFailClosedDiagnosticsSchema)
	}
	if d.QualityClaim || d.Summary.QualityClaim {
		return fmt.Errorf("AOQT fail-closed diagnostics quality_claim must be false")
	}
	if d.FailureKind != "zero_accepted_safe_steps" {
		return fmt.Errorf("AOQT fail-closed diagnostics failure_kind %q is not supported", d.FailureKind)
	}
	if strings.TrimSpace(d.Error) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics error is required")
	}
	if strings.TrimSpace(d.RejectionSummary) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics rejection_summary is required")
	}
	if strings.TrimSpace(d.PreflightPath) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight_path is required")
	}
	if err := validateAOQTSHA256(d.PreflightSHA256, "AOQT fail-closed diagnostics preflight_sha256"); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflightRawBinding(d.Preflight, d.PreflightPath, d.PreflightSHA256); err != nil {
		return err
	}
	if d.Plan != d.Summary.Plan {
		return fmt.Errorf("AOQT fail-closed diagnostics plan must exactly match summary.plan")
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.Summary.TrainingContract, d.Summary.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics training contract: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.IOReport.TrainingContract, d.IOReport.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report training contract: %w", err)
	}
	if err := validateAOQTSidecarTrainingContractBinding(d.TrainingContract, d.CandidateEligibilityPolicy, d.Preflight.TrainingContract, d.Preflight.CandidateEligibilityPolicy); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight training contract: %w", err)
	}
	if d.Plan.PlanOnly {
		return fmt.Errorf("AOQT fail-closed diagnostics require a non-plan optimizer run")
	}
	if d.Summary.Steps != 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics summary.steps = %d, want 0", d.Summary.Steps)
	}
	if d.Plan.RowCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.row_count must be positive")
	}
	if d.Plan.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.pair_count must be positive")
	}
	if d.Plan.StepCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics plan.step_count must be positive")
	}
	if err := d.Topology.Validate(); err != nil {
		return err
	}
	if err := validateAOQTMetricsPlanTopology(d.Plan, d.Topology); err != nil {
		return err
	}
	if !aoqtObjectiveContractsEqual(d.ObjectiveContract, d.Summary.ObjectiveContract) {
		return fmt.Errorf("AOQT fail-closed diagnostics objective_contract must exactly match summary.objective_contract")
	}
	if err := validateAOQTFailClosedObjectiveContract(d.ObjectiveContract, d.Topology); err != nil {
		return err
	}
	if len(aoqtActiveProtectedComponentNames(d.ObjectiveContract.WeightSums)) == 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics v2 require at least one active protected objective component")
	}
	if err := validateAOQTResearchOnlyLegalGates(d.LegalGates, "AOQT fail-closed diagnostics"); err != nil {
		return err
	}
	if d.Summary.OptimizerDiagnostics == nil {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics are required")
	}
	if err := validateAOQTObjectiveActivation(d.Summary.InitialObjectiveActivation, "AOQT fail-closed diagnostics summary.initial_objective_activation"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveActivation(d.Summary.FinalObjectiveActivation, "AOQT fail-closed diagnostics summary.final_objective_activation"); err != nil {
		return err
	}
	optimizer := *d.Summary.OptimizerDiagnostics
	if optimizer.PlannedSteps != d.Plan.StepCount {
		return fmt.Errorf("AOQT fail-closed diagnostics planned_steps = %d, want plan step_count %d", optimizer.PlannedSteps, d.Plan.StepCount)
	}
	if optimizer.AttemptedSteps != 1 {
		return fmt.Errorf("AOQT fail-closed diagnostics attempted_steps = %d, want 1", optimizer.AttemptedSteps)
	}
	if optimizer.AcceptedSteps != 0 || optimizer.AcceptedProposals != 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics must have zero accepted steps and proposals")
	}
	if optimizer.ExhaustedSteps != 1 {
		return fmt.Errorf("AOQT fail-closed diagnostics exhausted_steps = %d, want 1", optimizer.ExhaustedSteps)
	}
	if optimizer.MaxAttemptsPerStep != aoqtTransactionalMaxAttemptsPerStep {
		return fmt.Errorf("AOQT fail-closed diagnostics max_attempts_per_step = %d, want %d", optimizer.MaxAttemptsPerStep, aoqtTransactionalMaxAttemptsPerStep)
	}
	if optimizer.ProposalAttempts != optimizer.RejectedProposals || optimizer.Backtracks != optimizer.RejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics rejected proposal accounting mismatch")
	}
	if optimizer.ProposalAttempts <= 0 || optimizer.ProposalAttempts > optimizer.MaxAttemptsPerStep {
		return fmt.Errorf("AOQT fail-closed diagnostics proposal_attempts = %d outside expected range", optimizer.ProposalAttempts)
	}
	if optimizer.RejectedProposals < 0 || optimizer.Backtracks < 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics rejected proposal counts must be non-negative")
	}
	if optimizer.RejectionDiagnostics.CandidateEvaluations != optimizer.ProposalAttempts {
		return fmt.Errorf("AOQT fail-closed diagnostics candidate_evaluations = %d, want proposal_attempts %d", optimizer.RejectionDiagnostics.CandidateEvaluations, optimizer.ProposalAttempts)
	}
	if err := validateAOQTFailClosedOptimizerPathDiagnostics(d.Plan, d.ObjectiveContract.WeightSums, optimizer); err != nil {
		return err
	}
	if d.RejectionSummary != optimizer.RejectionSummary() {
		return fmt.Errorf("AOQT fail-closed diagnostics rejection_summary does not match optimizer diagnostics")
	}
	if !strings.Contains(d.Error, d.RejectionSummary) {
		return fmt.Errorf("AOQT fail-closed diagnostics error does not contain rejection_summary")
	}
	if strings.TrimSpace(d.Summary.OptimizerDiagnosticsSHA256) == "" {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics sha256 is required")
	}
	if err := validateAOQTSHA256(d.Summary.OptimizerDiagnosticsSHA256, "AOQT fail-closed diagnostics optimizer diagnostics sha256"); err != nil {
		return err
	}
	got, err := optimizer.SHA256()
	if err != nil {
		return err
	}
	if got != d.Summary.OptimizerDiagnosticsSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer diagnostics sha256 mismatch")
	}
	if err := validateAOQTObjectiveComponents(d.Summary.InitialObjectiveComponents, "AOQT fail-closed diagnostics summary.initial_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(d.Summary.InitialLoss, d.Summary.InitialObjectiveComponents, "AOQT fail-closed diagnostics summary.initial"); err != nil {
		return err
	}
	if err := validateAOQTObjectiveComponents(d.Summary.FinalObjectiveComponents, "AOQT fail-closed diagnostics summary.final_objective_components"); err != nil {
		return err
	}
	if err := validateAOQTLossMatchesComponents(d.Summary.FinalLoss, d.Summary.FinalObjectiveComponents, "AOQT fail-closed diagnostics summary.final"); err != nil {
		return err
	}
	if err := validateAOQTMetricInputs(d.Inputs); err != nil {
		return err
	}
	if err := validateAOQTFailClosedIOReport(d.IOReport, d.Plan, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	if err := validateAOQTFailClosedPreflight(d.Preflight, d.PreflightPath, d.IOReport, d.Inputs, d.Topology, d.ObjectiveContract, d.LegalGates); err != nil {
		return err
	}
	return nil
}

func validateAOQTFailClosedPreflightRawBinding(preflight AOQTSidecarMaterializePreflight, path, expectedSHA256 string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw file %q: %w", path, err)
	}
	if got := sha256BytesAOQT(raw); got != expectedSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw sha256 mismatch")
	}
	var loaded AOQTSidecarMaterializePreflight
	if err := strictUnmarshalAOQT(raw, &loaded); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight raw file %q is invalid: %w", path, err)
	}
	if !reflect.DeepEqual(loaded, preflight) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight does not match raw file %q", path)
	}
	return nil
}

func validateAOQTFailClosedOptimizerPathDiagnostics(plan AOQTSidecarWorkPlan, weights AOQTSidecarRowWeights, diagnostics AOQTSidecarOptimizerDiagnostics) error {
	if len(aoqtActiveProtectedComponentNames(weights)) == 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics v2 require at least one active protected objective component")
	}
	if err := validateAOQTOptimizerPathDiagnostics(diagnostics, "AOQT fail-closed diagnostics"); err != nil {
		return err
	}
	for _, item := range []struct {
		name  string
		value int
	}{
		{"adam_proposal_attempts", diagnostics.AdamProposalAttempts},
		{"adam_accepted_proposals", diagnostics.AdamAcceptedProposals},
		{"adam_rejected_proposals", diagnostics.AdamRejectedProposals},
		{"coordinate_proposal_attempts", diagnostics.CoordinateProposalAttempts},
		{"coordinate_accepted_proposals", diagnostics.CoordinateAcceptedProposals},
		{"coordinate_rejected_proposals", diagnostics.CoordinateRejectedProposals},
	} {
		if item.value < 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics %s must be non-negative", item.name)
		}
	}
	hasPathDiagnostics := diagnostics.AdamProposalAttempts != 0 ||
		diagnostics.AdamAcceptedProposals != 0 ||
		diagnostics.AdamRejectedProposals != 0 ||
		diagnostics.CoordinateProposalAttempts != 0 ||
		diagnostics.CoordinateAcceptedProposals != 0 ||
		diagnostics.CoordinateRejectedProposals != 0
	if !hasPathDiagnostics {
		if diagnostics.ProposalAttempts != 0 || diagnostics.AcceptedProposals != 0 || diagnostics.RejectedProposals != 0 || diagnostics.Backtracks != 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics top-level proposal accounting requires optimizer path diagnostics")
		}
		if diagnostics.CoordinateSearchPlanCount != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate audit is present without optimizer path diagnostics")
		}
		return nil
	}
	if diagnostics.CoordinateSearchPlanCount < 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_plan_count must be non-negative")
	}
	if diagnostics.CoordinateSearchPlanCount > diagnostics.AttemptedSteps {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_plan_count = %d exceeds attempted_steps %d", diagnostics.CoordinateSearchPlanCount, diagnostics.AttemptedSteps)
	}
	if diagnostics.CoordinateSearchPlanCount == 0 && (diagnostics.CoordinateProposalAttempts != 0 || diagnostics.CoordinateSearchStrategy != "" || diagnostics.CoordinateSearchOrderingHash != "" || diagnostics.CoordinateSearchLearningRate != 0 || diagnostics.CoordinateSearchAudit != "" || diagnostics.CoordinateSearchAuditChain != "" || diagnostics.CoordinateSearchHashChain != "" || diagnostics.CoordinateTopAngles != 0 || diagnostics.CoordinateMagnitudeCount != 0 || diagnostics.CoordinateBlockCount != 0) {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate audit is present without coordinate plan history")
	}
	if diagnostics.AdamProposalAttempts != diagnostics.AdamAcceptedProposals+diagnostics.AdamRejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics adam proposal accounting mismatch")
	}
	if diagnostics.CoordinateProposalAttempts != diagnostics.CoordinateAcceptedProposals+diagnostics.CoordinateRejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics coordinate proposal accounting mismatch")
	}
	if diagnostics.AdamProposalAttempts+diagnostics.CoordinateProposalAttempts != diagnostics.ProposalAttempts {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path proposal accounting mismatch")
	}
	if diagnostics.AdamAcceptedProposals+diagnostics.CoordinateAcceptedProposals != diagnostics.AcceptedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path accepted proposal accounting mismatch")
	}
	if diagnostics.AdamRejectedProposals+diagnostics.CoordinateRejectedProposals != diagnostics.RejectedProposals {
		return fmt.Errorf("AOQT fail-closed diagnostics optimizer-path rejected proposal accounting mismatch")
	}
	if diagnostics.CoordinateSearchPlanCount > 0 {
		wantStrategy := aoqtCoordinateSearchStrategyProtectedConeMicroTail
		wantMagnitudeCount := aoqtTransactionalCoordinateMicroTailMagnitudeCount
		if plan.OptimizerMode == AOQTSidecarOptimizerModeV7R3DevActualDirection {
			wantStrategy = aoqtCoordinateSearchStrategyV7R3ActualDirection
			wantMagnitudeCount = AOQTV7R3ActualDirectionProbeMagnitudeCount
		}
		if diagnostics.CoordinateSearchStrategy != wantStrategy {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_strategy = %q, want %q", diagnostics.CoordinateSearchStrategy, wantStrategy)
		}
		if diagnostics.CoordinateTopAngles <= 0 || diagnostics.CoordinateTopAngles > aoqtTransactionalCoordinateTopAngles {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_top_angles = %d outside expected range", diagnostics.CoordinateTopAngles)
		}
		if diagnostics.CoordinateMagnitudeCount != wantMagnitudeCount {
			return fmt.Errorf("AOQT fail-closed diagnostics protected coordinate_magnitude_count = %d, want exact count %d", diagnostics.CoordinateMagnitudeCount, wantMagnitudeCount)
		}
		if !isFinite32(diagnostics.CoordinateSearchLearningRate) || diagnostics.CoordinateSearchLearningRate <= 0 {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_search_learning_rate must be finite and positive")
		}
		if diagnostics.CoordinateBlockCount < 0 || diagnostics.CoordinateBlockCount > len(aoqtTransactionalCoordinateBlockSizes) {
			return fmt.Errorf("AOQT fail-closed diagnostics coordinate_block_count = %d outside expected range", diagnostics.CoordinateBlockCount)
		}
		if err := validateAOQTProtectedCoordinateSearchAudit(plan, weights, diagnostics); err != nil {
			return fmt.Errorf("AOQT fail-closed diagnostics protected coordinate audit: %w", err)
		}
	}
	return nil
}

func validateAOQTFailClosedObjectiveContract(contract AOQTSidecarObjectiveContract, topology AOQTSidecarTopologyBinding) error {
	manifest := AOQTSidecarCalibrationManifest{
		Dim:            topology.Dim,
		TurboQuantSeed: contract.TurboQuantSeed,
		QuantSurfaces: []AOQTSidecarQuantSurface{
			{BitWidth: contract.GainBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
			{BitWidth: contract.Q3GuardBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
			{BitWidth: contract.Q5GuardBit, Seed: contract.TurboQuantSeed, ScoreSurface: contract.ScoreSurface, PreparedQuery: true},
		},
	}
	if err := contract.Validate(manifest); err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics objective contract: %w", err)
	}
	return nil
}

func validateAOQTFailClosedIOReport(report AOQTSidecarCalibrationIOReport, plan AOQTSidecarWorkPlan, inputs AOQTSidecarRunMetricInputs, topology AOQTSidecarTopologyBinding, objective AOQTSidecarObjectiveContract, gates AOQTSidecarLegalGates) error {
	if report.Schema != AOQTSidecarCalibrationIOReportSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report schema %q is not supported, want %q", report.Schema, AOQTSidecarCalibrationIOReportSchema)
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"manifest_path", report.ManifestPath},
		{"rows_jsonl_path", report.RowsJSONLPath},
		{"manifest_sha256", report.ManifestSHA256},
		{"rows_sha256", report.RowsSHA256},
		{"anchor_artifact_sha256", report.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", report.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", report.AnchorEmbeddingSpaceID},
		{"compatibility_digest", report.CompatibilityDigest},
	} {
		if strings.TrimSpace(field.value) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics IO report %s is required", field.name)
		}
	}
	for _, item := range []struct {
		value string
		label string
	}{
		{report.ManifestSHA256, "AOQT fail-closed diagnostics IO report manifest_sha256"},
		{report.RowsSHA256, "AOQT fail-closed diagnostics IO report rows_sha256"},
		{report.AnchorArtifactSHA256, "AOQT fail-closed diagnostics IO report anchor_artifact_sha256"},
		{report.AnchorPackageManifestSHA256, "AOQT fail-closed diagnostics IO report anchor_package_manifest_sha256"},
		{report.CompatibilityDigest, "AOQT fail-closed diagnostics IO report compatibility_digest"},
	} {
		if err := validateAOQTSHA256(item.value, item.label); err != nil {
			return err
		}
	}
	rowsSHA256, err := sha256FileAOQT(report.RowsJSONLPath)
	if err != nil {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report rows file %q: %w", report.RowsJSONLPath, err)
	}
	if rowsSHA256 != report.RowsSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report rows_sha256 does not match rows file")
	}
	if report.ManifestSHA256 != inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report manifest_sha256 must match inputs.dataset_manifest_sha256")
	}
	if report.AnchorArtifactSHA256 != inputs.AnchorArtifactSHA256 || report.AnchorPackageManifestSHA256 != inputs.AnchorPackageManifestSHA256 || report.AnchorEmbeddingSpaceID != inputs.AnchorEmbeddingSpaceID || report.CompatibilityDigest != inputs.CompatibilityDigest {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report provenance must exactly match inputs")
	}
	if report.Topology != topology {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report topology must exactly match topology")
	}
	if !aoqtObjectiveContractsEqual(report.ObjectiveContract, objective) {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report objective_contract must exactly match objective_contract")
	}
	if report.LegalGates != gates {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report legal_gates must exactly match legal_gates")
	}
	if report.TurboQuantSeed != objective.TurboQuantSeed {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report turboquant_seed = %d, want objective contract seed %d", report.TurboQuantSeed, objective.TurboQuantSeed)
	}
	if report.RowCount <= 0 || report.CandidateCount <= 0 || report.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report workload counts must be positive")
	}
	if report.RowCount != plan.RowCount || report.PairCount != plan.PairCount {
		return fmt.Errorf("AOQT fail-closed diagnostics IO report workload counts must match plan")
	}
	return nil
}

func validateAOQTFailClosedPreflight(preflight AOQTSidecarMaterializePreflight, preflightPath string, report AOQTSidecarCalibrationIOReport, inputs AOQTSidecarRunMetricInputs, topology AOQTSidecarTopologyBinding, objective AOQTSidecarObjectiveContract, gates AOQTSidecarLegalGates) error {
	if preflight.Schema != AOQTSidecarMaterializerPreflightSchema {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight schema %q is not supported, want %q", preflight.Schema, AOQTSidecarMaterializerPreflightSchema)
	}
	if preflight.QualityClaim || !preflight.ResearchOnly || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight must be research-only and record no training/eval/vector export execution")
	}
	if err := validateAOQTSHA256(preflight.PlanSHA256, "AOQT fail-closed diagnostics preflight plan_sha256"); err != nil {
		return err
	}
	if len(preflight.InputSHA256) == 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight input_sha256 is required")
	}
	for path, sum := range preflight.InputSHA256 {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics preflight input_sha256 path is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT fail-closed diagnostics preflight input_sha256"); err != nil {
			return err
		}
	}
	planHashBound := false
	for _, sum := range preflight.InputSHA256 {
		if sum == preflight.PlanSHA256 {
			planHashBound = true
			break
		}
	}
	if !planHashBound {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight plan_sha256 is not bound by input_sha256")
	}
	if preflight.RowCount <= 0 || preflight.CandidateCount <= 0 || preflight.PairCount <= 0 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight workload counts must be positive")
	}
	if preflight.RowCount != report.RowCount || preflight.CandidateCount != report.CandidateCount || preflight.PairCount != report.PairCount {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight workload counts must match IO report")
	}
	if preflight.TurboQuantSeed != objective.TurboQuantSeed {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight turboquant_seed = %d, want objective contract seed %d", preflight.TurboQuantSeed, objective.TurboQuantSeed)
	}
	if preflight.TopologySeed != topology.Seed {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight topology_seed = %d, want topology seed %d", preflight.TopologySeed, topology.Seed)
	}
	if preflight.QuantScoreSurface != objective.ScoreSurface {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight quant_score_surface = %q, want objective score surface %q", preflight.QuantScoreSurface, objective.ScoreSurface)
	}
	if preflight.CalibrationManifestSHA256 != report.ManifestSHA256 || preflight.CalibrationManifestSHA256 != inputs.DatasetManifestSHA256 {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight calibration manifest sha256 must match IO report and inputs")
	}
	if !aoqtStringMapsEqual(preflight.QrelsSHA256ByDataset, inputs.QrelsSHA256ByDataset) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight qrels_sha256_by_dataset must exactly match inputs")
	}
	if preflight.LegalGates != gates {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight legal_gates must exactly match legal_gates")
	}
	if !aoqtObjectiveContractsEqual(preflight.ObjectiveContract, objective) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight objective_contract must exactly match objective_contract")
	}
	if !slices.Equal(preflight.SourceSemantics, []string{"q3_top10", "q5_top10", "nf_boundary80_120"}) {
		return fmt.Errorf("AOQT fail-closed diagnostics preflight source_semantics must bind q3_top10,q5_top10,nf_boundary80_120")
	}
	for _, key := range []string{"rows_jsonl", "manifest_json", "preflight_json"} {
		if strings.TrimSpace(preflight.Outputs[key]) == "" {
			return fmt.Errorf("AOQT fail-closed diagnostics preflight outputs.%s is required", key)
		}
	}
	if err := validateAOQTPathBinding(preflight.Outputs["rows_jsonl"], report.RowsJSONLPath, "AOQT fail-closed diagnostics preflight outputs.rows_jsonl", "IO report rows_jsonl_path"); err != nil {
		return err
	}
	if err := validateAOQTPathBinding(preflight.Outputs["manifest_json"], report.ManifestPath, "AOQT fail-closed diagnostics preflight outputs.manifest_json", "IO report manifest_path"); err != nil {
		return err
	}
	if err := validateAOQTPathBinding(preflight.Outputs["preflight_json"], preflightPath, "AOQT fail-closed diagnostics preflight outputs.preflight_json", "preflight path"); err != nil {
		return err
	}
	return nil
}

func validateAOQTPathBinding(declared, actual, declaredLabel, actualLabel string) error {
	declaredCanonical, err := canonicalAOQTPath(declared)
	if err != nil {
		return fmt.Errorf("%s path %q cannot be resolved: %w", declaredLabel, declared, err)
	}
	actualCanonical, err := canonicalAOQTPath(actual)
	if err != nil {
		return fmt.Errorf("%s %q cannot be resolved: %w", actualLabel, actual, err)
	}
	if declaredCanonical != actualCanonical {
		return fmt.Errorf("%s must bind %s", declaredLabel, actualLabel)
	}
	return nil
}

func aoqtStringMapsEqual(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func validateAOQTMetricInputs(inputs AOQTSidecarRunMetricInputs) error {
	for _, field := range []struct {
		name  string
		value string
	}{
		{"anchor_artifact_sha256", inputs.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", inputs.AnchorPackageManifestSHA256},
		{"anchor_embedding_space_id", inputs.AnchorEmbeddingSpaceID},
		{"dataset_manifest_sha256", inputs.DatasetManifestSHA256},
		{"compatibility_digest", inputs.CompatibilityDigest},
	} {
		if field.value == "" {
			return fmt.Errorf("AOQT run inputs.%s is required", field.name)
		}
	}
	if err := validateAOQTSHA256(inputs.AnchorArtifactSHA256, "AOQT run inputs.anchor_artifact_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.AnchorPackageManifestSHA256, "AOQT run inputs.anchor_package_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.DatasetManifestSHA256, "AOQT run inputs.dataset_manifest_sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(inputs.CompatibilityDigest, "AOQT run inputs.compatibility_digest"); err != nil {
		return err
	}
	if len(inputs.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT run inputs.qrels_sha256_by_dataset is required")
	}
	for dataset, sum := range inputs.QrelsSHA256ByDataset {
		if dataset == "" {
			return fmt.Errorf("AOQT run inputs qrels dataset name is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT run inputs.qrels_sha256_by_dataset"); err != nil {
			return err
		}
	}
	return nil
}

func bindAOQTForwardConsistencyProbeReceipt(receipt *AOQTSidecarForwardConsistencyProbeReceipt, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflightPath, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding) error {
	if receipt == nil {
		return nil
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return err
	}
	receipt.Binding.Inputs = AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestHash,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	receipt.Binding.IOReport = ioReport
	receipt.Binding.PreflightPath = preflightPath
	receipt.Binding.PreflightSHA256 = preflightSHA256
	receipt.Binding.SplitBinding = cloneAOQTDevSplitBinding(splitBinding)
	if receipt.Required && receipt.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT forward-consistency probe requires a split binding")
	}
	return receipt.ValidateBound()
}

func bindAOQTActualDirectionProbeReceipt(receipt *AOQTSidecarActualDirectionProbeReceipt, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflightPath, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding) error {
	if receipt == nil {
		return nil
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return err
	}
	receipt.Binding.Inputs = AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestHash,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	receipt.Binding.IOReport = ioReport
	receipt.Binding.PreflightPath = preflightPath
	receipt.Binding.PreflightSHA256 = preflightSHA256
	receipt.Binding.SplitBinding = cloneAOQTDevSplitBinding(splitBinding)
	receipt.Binding.ObjectiveContractSHA256 = receipt.ObjectiveContractSHA256
	receipt.Binding.EligibilityPolicySHA256 = receipt.EligibilityPolicySHA256
	receipt.Binding.ScheduleSHA256 = receipt.ScheduleSHA256
	receipt.Binding.RankingSHA256 = receipt.RankingSHA256
	if receipt.Required && receipt.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-direction probe requires a split binding")
	}
	return receipt.ValidateBound()
}

func bindAOQTActualCoordinateProbeReceipt(receipt *AOQTSidecarActualCoordinateProbeReceipt, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflightPath, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding) error {
	if receipt == nil {
		return nil
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return err
	}
	receipt.Binding.Inputs = AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestHash,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	receipt.Binding.IOReport = ioReport
	receipt.Binding.PreflightPath = preflightPath
	receipt.Binding.PreflightSHA256 = preflightSHA256
	receipt.Binding.SplitBinding = cloneAOQTDevSplitBinding(splitBinding)
	receipt.Binding.ObjectiveContractSHA256 = receipt.ObjectiveContractSHA256
	receipt.Binding.EligibilityPolicySHA256 = receipt.EligibilityPolicySHA256
	receipt.Binding.ScheduleSHA256 = receipt.ScheduleSHA256
	receipt.Binding.RankingSHA256 = receipt.RankingSHA256
	receipt.Binding.CoverageSHA256 = receipt.CoverageSHA256
	if receipt.Required && receipt.Binding.SplitBinding == nil {
		return fmt.Errorf("AOQT actual-coordinate probe requires a split binding")
	}
	return receipt.ValidateBound()
}

// loadAOQTV7R5CanonicalR4Receipt performs the authorization check before a
// trainer is constructed. The raw file hash, receipt validation, and all
// material input bindings must agree with the current run; a caller cannot
// substitute a merely well-formed r4 receipt from another calibration.
func loadAOQTV7R5CanonicalR4Receipt(cfg AOQTSidecarTrainRunnerConfig, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding) (*AOQTSidecarActualCoordinateProbeReceipt, error) {
	path := cfg.canonicalR4ReceiptPath()
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt path is required")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read AOQT V7-r5 canonical V7-r4 receipt %q: %w", path, err)
	}
	var receipt AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(data, &receipt); err != nil {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt %q is invalid: %w", path, err)
	}
	if err := receipt.ValidateBound(); err != nil {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt ValidateBound: %w", err)
	}
	fileSHA := sha256BytesAOQT(data)
	if fileSHA != AOQTV7R5CanonicalR4ReceiptSHA256 || fileSHA != cfg.canonicalR4ReceiptExpectedSHA256() {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt file sha256 = %s, want pinned %s", fileSHA, AOQTV7R5CanonicalR4ReceiptSHA256)
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return nil, err
	}
	expectedInputs := AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestHash,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	if !reflect.DeepEqual(receipt.Binding.IOReport, ioReport) || !reflect.DeepEqual(receipt.Binding.Inputs, expectedInputs) || receipt.Binding.PreflightPath != cfg.PreflightJSONPath || receipt.Binding.PreflightSHA256 != preflightSHA256 {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt is bound to different current inputs")
	}
	if splitBinding == nil || receipt.Binding.SplitBinding == nil || !reflect.DeepEqual(splitBinding, receipt.Binding.SplitBinding) {
		return nil, fmt.Errorf("AOQT V7-r5 canonical V7-r4 receipt split binding does not match current run")
	}
	return &receipt, nil
}

// bindAOQTActualCoordinateTrainReceipt binds every fresh r4 search receipt,
// then rehashes it and rebuilds the outer step chain. The trainer intentionally
// hashes unbound search receipts for the step-0 equality check; only the
// runner, after all source bindings are known, emits the durable bound chain.
func bindAOQTActualCoordinateTrainReceipt(receipt *AOQTSidecarActualCoordinateTrainReceipt, set AOQTSidecarCalibrationSet, ioReport AOQTSidecarCalibrationIOReport, preflightPath, preflightSHA256 string, splitBinding *AOQTSidecarDevSplitBinding, canonicalPath, canonicalSHA string) error {
	if receipt == nil {
		return nil
	}
	if canonicalSHA != AOQTV7R5CanonicalR4ReceiptSHA256 || receipt.CanonicalR4ReceiptSHA256 != AOQTV7R5CanonicalR4ReceiptSHA256 {
		return fmt.Errorf("AOQT V7-r5 full-training receipt canonical r4 sha256 is not the pinned authorization")
	}
	manifestHash, err := AOQTSidecarManifestSHA256(set.Manifest)
	if err != nil {
		return err
	}
	inputs := AOQTSidecarRunMetricInputs{
		AnchorArtifactSHA256:        set.Manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: set.Manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      set.Manifest.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       manifestHash,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(set.Manifest.QrelsSHA256ByDataset),
		CompatibilityDigest:         set.Manifest.CompatibilityDigest,
	}
	receipt.Binding.Inputs = inputs
	receipt.Binding.IOReport = ioReport
	receipt.Binding.PreflightPath = preflightPath
	receipt.Binding.PreflightSHA256 = preflightSHA256
	receipt.Binding.SplitBinding = cloneAOQTDevSplitBinding(splitBinding)
	receipt.Binding.CanonicalR4ReceiptPath = canonicalPath
	receipt.Binding.CanonicalR4ReceiptSHA256 = canonicalSHA
	canonicalData, err := os.ReadFile(canonicalPath)
	if err != nil {
		return fmt.Errorf("read AOQT V7-r5 canonical r4 receipt for file hash: %w", err)
	}
	receipt.CanonicalR4ReceiptFileSHA256 = sha256BytesAOQT(canonicalData)
	receipt.Binding.CanonicalR4ReceiptFileSHA256 = receipt.CanonicalR4ReceiptFileSHA256
	var canonicalReceipt AOQTSidecarActualCoordinateProbeReceipt
	if err := strictUnmarshalAOQT(canonicalData, &canonicalReceipt); err != nil {
		return fmt.Errorf("decode AOQT V7-r5 canonical r4 receipt for binding: %w", err)
	}
	if err := canonicalReceipt.Validate(); err != nil {
		return fmt.Errorf("validate AOQT V7-r5 canonical r4 receipt for binding: %w", err)
	}
	receipt.Binding.ObjectiveContractSHA256 = canonicalReceipt.Binding.ObjectiveContractSHA256
	receipt.Binding.EligibilityPolicySHA256 = canonicalReceipt.Binding.EligibilityPolicySHA256
	receipt.Binding.ScheduleSHA256 = canonicalReceipt.Binding.ScheduleSHA256
	receipt.Binding.RankingSHA256 = canonicalReceipt.Binding.RankingSHA256
	receipt.Binding.CoverageSHA256 = canonicalReceipt.Binding.CoverageSHA256
	for index := range receipt.StepsEvidence {
		searchReceipt := &receipt.StepsEvidence[index].Search.Receipt
		searchReceipt.Binding.Inputs = inputs
		searchReceipt.Binding.IOReport = ioReport
		searchReceipt.Binding.PreflightPath = preflightPath
		searchReceipt.Binding.PreflightSHA256 = preflightSHA256
		searchReceipt.Binding.SplitBinding = cloneAOQTDevSplitBinding(splitBinding)
		searchReceipt.Binding.ObjectiveContractSHA256 = searchReceipt.ObjectiveContractSHA256
		searchReceipt.Binding.EligibilityPolicySHA256 = searchReceipt.EligibilityPolicySHA256
		searchReceipt.Binding.ScheduleSHA256 = searchReceipt.ScheduleSHA256
		searchReceipt.Binding.RankingSHA256 = searchReceipt.RankingSHA256
		searchReceipt.Binding.CoverageSHA256 = searchReceipt.CoverageSHA256
		if index == 0 {
			// Step zero is required to be the identity-state replay of the
			// canonical r4 receipt, so its full replay can be checked directly.
			if err := searchReceipt.ValidateBound(); err != nil {
				return fmt.Errorf("AOQT V7-r5 step %d bound r4 search receipt: %w", index, err)
			}
		} else if err := searchReceipt.Validate(); err != nil {
			// Later searches intentionally run from the committed r5 state;
			// r4's identity-state ValidateBound replay is not the authoritative
			// state for those endpoints. Full source/contract binding is checked
			// independently below and again by the outer receipt validator.
			return fmt.Errorf("AOQT V7-r5 step %d r4 search receipt: %w", index, err)
		}
		if err := validateAOQTV7R5SearchBinding(*searchReceipt, receipt.Binding); err != nil {
			return fmt.Errorf("AOQT V7-r5 step %d r4 search binding: %w", index, err)
		}
		searchSHA, err := searchReceipt.SHA256()
		if err != nil {
			return err
		}
		receipt.StepsEvidence[index].Search.ReceiptSHA256 = searchSHA
	}
	if len(receipt.StepsEvidence) > 0 {
		first := receipt.StepsEvidence[0].Search.Receipt
		receipt.Binding.ObjectiveContractSHA256 = first.ObjectiveContractSHA256
		receipt.Binding.EligibilityPolicySHA256 = first.EligibilityPolicySHA256
		receipt.Binding.ScheduleSHA256 = first.ScheduleSHA256
		receipt.Binding.RankingSHA256 = first.RankingSHA256
		receipt.Binding.CoverageSHA256 = first.CoverageSHA256
	}
	receipt.StepHashChain, receipt.StepHashChainTailSHA256, err = aoqtV7R5StepHashChain(receipt.StepsEvidence)
	if err != nil {
		return err
	}
	return receipt.ValidateBound()
}

// bindAOQTActualDirectionProposalReceipts updates the per-step provenance
// after the probe receives its authoritative runner binding. The trainer
// first hashes an unbound probe so it can bind proposals during the same fit;
// the runner then rehashes the bound receipt and must carry that final hash
// into every coordinate proposal before diagnostics are validated/serialized.
func bindAOQTActualDirectionProposalReceipts(diagnostics *AOQTSidecarOptimizerDiagnostics, probe *AOQTSidecarActualDirectionProbeReceipt, probeSHA string) error {
	if diagnostics == nil || diagnostics.OptimizerMode != AOQTSidecarOptimizerModeV7R3DevActualDirection || diagnostics.ProposalReceipts == nil {
		return nil
	}
	if probe == nil {
		return fmt.Errorf("AOQT actual-direction proposal provenance requires a probe receipt")
	}
	if err := validateAOQTSHA256(probeSHA, "AOQT actual-direction proposal probe receipt sha256"); err != nil {
		return err
	}
	for index := range *diagnostics.ProposalReceipts {
		receipt := &(*diagnostics.ProposalReceipts)[index]
		if receipt.Kind != aoqtProposalKindCoordinate {
			continue
		}
		if receipt.Accepted || receipt.EndpointHashSHA256 != "" || receipt.RequestedDirectionSHA256 != "" || receipt.ActualDirectionSHA256 != "" {
			receipt.ActualDirectionProbeSHA256 = probeSHA
		}
	}
	return nil
}

func refreshAOQTActualDirectionDiagnosticsHash(summary *AOQTSidecarTrainSummary) error {
	if summary == nil || summary.OptimizerDiagnostics == nil {
		return nil
	}
	hash, err := summary.OptimizerDiagnostics.SHA256()
	if err != nil {
		return err
	}
	summary.OptimizerDiagnosticsSHA256 = hash
	return nil
}

func AOQTFailClosedDiagnosticsPath(metricsPath string) string {
	return metricsPath + ".failclosed.json"
}

func AOQTDevFailClosedDiagnosticsPath(metricsPath string) string {
	return metricsPath + ".dev.failclosed.json"
}

func ensureAOQTForwardConsistencyProbeReceiptAbsent(path string) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT forward-consistency probe receipt path is required")
	}
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT forward-consistency probe receipt output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func ensureAOQTActualDirectionProbeReceiptAbsent(path string) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT actual-direction probe receipt path is required")
	}
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT actual-direction probe receipt output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func ensureAOQTActualCoordinateProbeReceiptAbsent(path string) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT actual-coordinate probe receipt path is required")
	}
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT actual-coordinate probe receipt output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func ensureAOQTActualCoordinateTrainReceiptAbsent(path string) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("AOQT actual-coordinate full-training receipt path is required")
	}
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT actual-coordinate full-training receipt output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func writeAOQTForwardConsistencyProbeReceiptFile(path string, receipt AOQTSidecarForwardConsistencyProbeReceipt) error {
	if err := receipt.ValidateBound(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT forward-consistency probe receipt JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT forward-consistency probe receipt parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT forward-consistency probe receipt output %q already exists", path)
		}
		return fmt.Errorf("create AOQT forward-consistency probe receipt output %q: %w", path, err)
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write AOQT forward-consistency probe receipt output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync AOQT forward-consistency probe receipt output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close AOQT forward-consistency probe receipt output %q: %w", path, err)
	}
	removeOnFailure = false
	return nil
}

func writeAOQTActualDirectionProbeReceiptFile(path string, receipt AOQTSidecarActualDirectionProbeReceipt) error {
	if err := receipt.ValidateBound(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT actual-direction probe receipt JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT actual-direction probe receipt parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT actual-direction probe receipt output %q already exists", path)
		}
		return fmt.Errorf("create AOQT actual-direction probe receipt output %q: %w", path, err)
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write AOQT actual-direction probe receipt output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync AOQT actual-direction probe receipt output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close AOQT actual-direction probe receipt output %q: %w", path, err)
	}
	removeOnFailure = false
	return nil
}

func writeAOQTActualCoordinateProbeReceiptFile(path string, receipt AOQTSidecarActualCoordinateProbeReceipt) error {
	if err := receipt.ValidateBound(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT actual-coordinate probe receipt JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT actual-coordinate probe receipt parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT actual-coordinate probe receipt output %q already exists", path)
		}
		return fmt.Errorf("create AOQT actual-coordinate probe receipt output %q: %w", path, err)
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write AOQT actual-coordinate probe receipt output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync AOQT actual-coordinate probe receipt output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close AOQT actual-coordinate probe receipt output %q: %w", path, err)
	}
	removeOnFailure = false
	return nil
}

func writeAOQTActualCoordinateTrainReceiptFile(path string, receipt AOQTSidecarActualCoordinateTrainReceipt) error {
	if err := receipt.ValidateBound(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT actual-coordinate full-training receipt JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT actual-coordinate full-training receipt parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT actual-coordinate full-training receipt output %q already exists", path)
		}
		return fmt.Errorf("create AOQT actual-coordinate full-training receipt output %q: %w", path, err)
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write AOQT actual-coordinate full-training receipt output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync AOQT actual-coordinate full-training receipt output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close AOQT actual-coordinate full-training receipt output %q: %w", path, err)
	}
	removeOnFailure = false
	return nil
}

func writeAOQTFailClosedDiagnosticsFile(path string, diagnostics AOQTSidecarFailClosedDiagnostics) error {
	if err := diagnostics.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(diagnostics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT fail-closed diagnostics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT fail-closed diagnostics parent: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".aoqt-failclosed-*")
	if err != nil {
		return fmt.Errorf("create AOQT fail-closed diagnostics temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	defer cleanup()
	if err := tmp.Chmod(0o644); err != nil {
		return fmt.Errorf("set AOQT fail-closed diagnostics temporary permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close AOQT fail-closed diagnostics temporary file: %w", err)
	}
	if err := os.Link(tmpPath, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT fail-closed diagnostics output %q already exists", path)
		}
		return fmt.Errorf("create AOQT fail-closed diagnostics output %q exclusively: %w", path, err)
	}
	return nil
}

func writeAOQTDevFailClosedDiagnosticsFile(path string, diagnostics AOQTSidecarDevFailClosedDiagnostics) error {
	if err := diagnostics.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(diagnostics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT dev fail-closed diagnostics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT dev fail-closed diagnostics parent: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".aoqt-dev-failclosed-*")
	if err != nil {
		return fmt.Errorf("create AOQT dev fail-closed diagnostics temporary file: %w", err)
	}
	tmpPath := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
	}
	defer cleanup()
	if err := tmp.Chmod(0o644); err != nil {
		return fmt.Errorf("set AOQT dev fail-closed diagnostics temporary permissions: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("write AOQT dev fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync AOQT dev fail-closed diagnostics temporary file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close AOQT dev fail-closed diagnostics temporary file: %w", err)
	}
	if err := os.Link(tmpPath, path); err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT dev fail-closed diagnostics output %q already exists", path)
		}
		return fmt.Errorf("create AOQT dev fail-closed diagnostics output %q exclusively: %w", path, err)
	}
	return nil
}

func writeAOQTRunMetricsFile(path string, metrics AOQTSidecarRunMetrics) error {
	if err := metrics.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return fmt.Errorf("encode AOQT metrics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create AOQT metrics parent: %w", err)
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("AOQT sidecar metrics output %q already exists", path)
		}
		return fmt.Errorf("create AOQT metrics output %q exclusively: %w", path, err)
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("write AOQT metrics output %q: %w", path, err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("sync AOQT metrics output %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close AOQT metrics output %q: %w", path, err)
	}
	removeOnFailure = false
	return nil
}

func aoqtDevEvidencePaths(dir string) map[string]string {
	return map[string]string{
		"dev transform": filepath.Join(dir, "aoqt-sidecar-transform.json"),
		"dev evidence":  filepath.Join(dir, "aoqt-sidecar-dev-evidence.json"),
	}
}

func writeAOQTSidecarDevEvidence(cfg AOQTSidecarTrainRunnerConfig, preflightSHA256 string, transform AOQTGivensTransform, splitBinding *AOQTSidecarDevSplitBinding) (AOQTSidecarDevEvidence, error) {
	if !cfg.isDevOnlyRun() {
		return AOQTSidecarDevEvidence{}, fmt.Errorf("AOQT dev evidence requires explicit V7 dev-only optimizer mode")
	}
	if err := os.MkdirAll(cfg.DevOutputDir, 0o755); err != nil {
		return AOQTSidecarDevEvidence{}, fmt.Errorf("create AOQT dev output dir: %w", err)
	}
	paths := aoqtDevEvidencePaths(cfg.DevOutputDir)
	cleanup := newAOQTCandidateCleanup()
	defer cleanup.run()
	if err := writeAOQTGivensTransformExclusive(transform, paths["dev transform"], cleanup); err != nil {
		return AOQTSidecarDevEvidence{}, fmt.Errorf("write AOQT dev transform evidence: %w", err)
	}
	transformSHA, _, err := fileHash(paths["dev transform"])
	if err != nil {
		return AOQTSidecarDevEvidence{}, err
	}
	metricsSHA, _, err := fileHash(cfg.MetricsJSONPath)
	if err != nil {
		return AOQTSidecarDevEvidence{}, err
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarDevEvidence{}, err
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTSidecarDevEvidence{}, err
	}
	evidence := AOQTSidecarDevEvidence{
		Schema:          AOQTSidecarDevEvidenceSchema,
		DevOnly:         true,
		OptimizerMode:   cfg.OptimizerMode,
		MetricsPath:     cfg.MetricsJSONPath,
		MetricsSHA256:   metricsSHA,
		PreflightPath:   cfg.PreflightJSONPath,
		PreflightSHA256: preflightSHA256,
		TransformPath:   paths["dev transform"],
		TransformSHA256: transformSHA,
		PairingsSHA256:  pairingsSHA,
		AnglesSHA256:    anglesSHA,
	}
	if strings.TrimSpace(cfg.SplitManifestPath) != "" {
		if splitBinding == nil {
			return AOQTSidecarDevEvidence{}, fmt.Errorf("AOQT V7 dev split binding was not validated")
		}
		binding := *splitBinding
		evidence.SplitBinding = &binding
	}
	if err := evidence.Validate(); err != nil {
		return AOQTSidecarDevEvidence{}, err
	}
	data, err := json.MarshalIndent(evidence, "", "  ")
	if err != nil {
		return AOQTSidecarDevEvidence{}, fmt.Errorf("encode AOQT dev evidence JSON: %w", err)
	}
	data = append(data, '\n')
	if err := writeAOQTExclusiveFile(paths["dev evidence"], data, 0o644, cleanup); err != nil {
		return AOQTSidecarDevEvidence{}, fmt.Errorf("write AOQT dev evidence manifest: %w", err)
	}
	cleanup.commit()
	return evidence, nil
}

func (b AOQTSidecarDevSplitBinding) Validate() error {
	if b.Schema != AOQTV7TrainSplitBindingSchema {
		return fmt.Errorf("schema %q is not supported, want %q", b.Schema, AOQTV7TrainSplitBindingSchema)
	}
	if b.SplitManifestSchema != AOQTV7DevSplitManifestSchema {
		return fmt.Errorf("split_manifest_schema %q is not supported, want %q", b.SplitManifestSchema, AOQTV7DevSplitManifestSchema)
	}
	if strings.TrimSpace(b.SplitManifestPath) == "" {
		return fmt.Errorf("split_manifest_path is required")
	}
	if strings.TrimSpace(b.FoldID) == "" {
		return fmt.Errorf("fold_id is required")
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"split_manifest_file_sha256", b.SplitManifestFileSHA256},
		{"split_manifest_sha256", b.SplitManifestSHA256},
		{"split_manifest_payload_sha256", b.SplitManifestPayloadSHA256},
		{"source_plan_sha256", b.SourcePlanSHA256},
		{"official_qid_registry_sha256", b.OfficialQIDRegistrySHA256},
		{"official_registry_source_sha256", b.OfficialRegistrySourceSHA256},
		{"train_row_ids_sha256", b.TrainRowIDSHA256},
		{"train_qids_by_dataset_sha256", b.TrainQIDsByDatasetSHA256},
		{"materialized_manifest_sha256", b.MaterializedManifestSHA256},
		{"materialized_rows_sha256", b.MaterializedRowsSHA256},
		{"materialized_preflight_sha256", b.MaterializedPreflightSHA256},
		{"materialized_row_ids_sha256", b.MaterializedRowIDSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT V7 train split binding "+item.name); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"source_plan_rows_sha256", b.SourcePlanRowsSHA256},
		{"source_plan_row_ids_sha256", b.SourcePlanRowIDSHA256},
	} {
		if strings.TrimSpace(item.value) != "" {
			if err := validateAOQTSHA256(item.value, "AOQT V7 train split binding "+item.name); err != nil {
				return err
			}
		}
	}
	if b.SplitManifestSHA256 != b.SplitManifestFileSHA256 {
		return fmt.Errorf("split_manifest_sha256 must match split_manifest_file_sha256")
	}
	if b.TrainRowCount <= 0 || b.MaterializedRowCount <= 0 {
		return fmt.Errorf("train/materialized row counts must be positive")
	}
	if b.TrainRowCount != b.MaterializedRowCount {
		return fmt.Errorf("train_row_count must match materialized_row_count")
	}
	if len(b.TrainQIDCountByDataset) == 0 || len(b.TrainQIDSetSHA256ByDataset) == 0 {
		return fmt.Errorf("train qid count/hash maps are required")
	}
	if len(b.TrainQIDCountByDataset) != len(b.TrainQIDSetSHA256ByDataset) {
		return fmt.Errorf("train qid count/hash maps must cover the same datasets")
	}
	for dataset, count := range b.TrainQIDCountByDataset {
		if strings.TrimSpace(dataset) == "" {
			return fmt.Errorf("train qid dataset name is required")
		}
		if count < 0 {
			return fmt.Errorf("train qid count for %s must be non-negative", dataset)
		}
		sum, ok := b.TrainQIDSetSHA256ByDataset[dataset]
		if !ok {
			return fmt.Errorf("train qid hash missing for %s", dataset)
		}
		if err := validateAOQTSHA256(sum, "AOQT V7 train split binding train_qid_set_sha256_by_dataset"); err != nil {
			return err
		}
	}
	return nil
}

func (e AOQTSidecarDevEvidence) Validate() error {
	if e.Schema != AOQTSidecarDevEvidenceSchema {
		return fmt.Errorf("AOQT dev evidence schema %q is not supported, want %q", e.Schema, AOQTSidecarDevEvidenceSchema)
	}
	if !e.DevOnly {
		return fmt.Errorf("AOQT dev evidence dev_only must be true")
	}
	if !isAOQTV7DevOptimizerMode(e.OptimizerMode) {
		return fmt.Errorf("AOQT dev evidence optimizer_mode %q is not supported", e.OptimizerMode)
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"metrics_path", e.MetricsPath},
		{"preflight_path", e.PreflightPath},
		{"transform_path", e.TransformPath},
		{"pairings_sha256", e.PairingsSHA256},
		{"angles_sha256", e.AnglesSHA256},
		{"metrics_sha256", e.MetricsSHA256},
		{"preflight_sha256", e.PreflightSHA256},
		{"transform_sha256", e.TransformSHA256},
	} {
		if strings.TrimSpace(item.value) == "" {
			return fmt.Errorf("AOQT dev evidence %s is required", item.name)
		}
	}
	for _, item := range []struct {
		name  string
		value string
	}{
		{"metrics_sha256", e.MetricsSHA256},
		{"preflight_sha256", e.PreflightSHA256},
		{"transform_sha256", e.TransformSHA256},
		{"pairings_sha256", e.PairingsSHA256},
		{"angles_sha256", e.AnglesSHA256},
	} {
		if err := validateAOQTSHA256(item.value, "AOQT dev evidence "+item.name); err != nil {
			return err
		}
	}
	if e.SplitBinding != nil {
		if err := e.SplitBinding.Validate(); err != nil {
			return fmt.Errorf("AOQT dev evidence split binding: %w", err)
		}
	}
	return nil
}

func ensureAOQTOutputArtifactAbsent(path string) error {
	for role, candidate := range aoqtCandidatePathMap(aoqtCandidateOutputPaths(path, true)) {
		if candidate == "" {
			continue
		}
		if _, err := os.Lstat(candidate); err == nil {
			return fmt.Errorf("AOQT sidecar output %s %q already exists", role, candidate)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func ensureAOQTDevEvidenceAbsent(dir string) error {
	for role, path := range aoqtDevEvidencePaths(dir) {
		if _, err := os.Lstat(path); err == nil {
			return fmt.Errorf("AOQT sidecar %s output %q already exists", role, path)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func ensureAOQTFailClosedDiagnosticsAbsent(metricsPath string) error {
	path := AOQTFailClosedDiagnosticsPath(metricsPath)
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT sidecar fail-closed diagnostics output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func ensureAOQTDevFailClosedDiagnosticsAbsent(metricsPath string) error {
	path := AOQTDevFailClosedDiagnosticsPath(metricsPath)
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT sidecar dev fail-closed diagnostics output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func ensureAOQTRunMetricsAbsent(path string) error {
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT sidecar metrics output %q already exists", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}
