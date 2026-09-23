package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runTrainAOQTSidecar(args []string) error {
	// Keep the established train-aoqt-sidecar entry point useful for scripts
	// that dispatch every AOQT operation through one command.  V7-r6 is still
	// routed to its dedicated parser/runtime so it cannot fall through into the
	// training, metrics, or package paths below.
	if hasAOQTV7R6ProgressiveScreenFlag(args) {
		return runScreenAOQTV7R6(args)
	}
	fs := flag.NewFlagSet("train-aoqt-sidecar", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarTrainRunnerConfig
	var sourceHashes string
	var sourceHashesFile string
	var sourceHashesFileSHA256 string
	var vectorHashes string
	var learningRate float64
	fs.BoolVar(&cfg.PlanOnly, "plan-only", false, "validate AOQT training inputs and write metrics without optimizer or package output")
	fs.BoolVar(&cfg.AllowResearchOnly, "allow-research-only-aoqt", false, "explicitly allow research-only AOQT train rows; release/commercial/free-open gates must remain false")
	fs.BoolVar(&cfg.AllowDevOnly, "allow-dev-aoqt-v7", false, "explicitly allow AOQT V7 dev-only optimizer telemetry; never writes candidate package artifacts")
	fs.BoolVar(&cfg.RequireForwardConsistencyProbe, "require-forward-consistency-probe", false, "require the fixed V7-r2 train-only forward-consistency probe before optimization")
	fs.BoolVar(&cfg.ForwardConsistencyProbeOnly, "forward-consistency-probe-only", false, "run the authorized V7-r2 forward-consistency probe and write only its bound receipt")
	fs.BoolVar(&cfg.RequireActualDirectionProbe, "require-actual-direction-probe", false, "require the fixed V7-r3 actual-direction endpoint probe before optimization")
	fs.BoolVar(&cfg.ActualDirectionProbeOnly, "actual-direction-probe-only", false, "run the authorized V7-r3 actual-direction endpoint probe and write only its bound receipt")
	fs.BoolVar(&cfg.RequireActualCoordinateProbe, "require-actual-coordinate-probe", false, "require the preregistered V7-r4 actual-coordinate endpoint probe")
	fs.BoolVar(&cfg.ActualCoordinateProbeOnly, "actual-coordinate-probe-only", false, "run the authorized V7-r4 actual-coordinate probe and write only its bound receipt")
	fs.BoolVar(&cfg.RequireActualCoordinateTrain, "require-actual-coordinate-train", false, "require the authorized V7-r5 actual-coordinate full trainer")
	fs.BoolVar(&cfg.ActualCoordinateTrainRequired, "actual-coordinate-train-required", false, "alias for --require-actual-coordinate-train")
	fs.StringVar(&cfg.ManifestPath, "manifest", "", "strict AOQT calibration manifest JSON")
	fs.StringVar(&cfg.RowsJSONLPath, "rows", "", "strict AOQT calibration rows JSONL")
	fs.StringVar(&cfg.PreflightJSONPath, "preflight", "", "strict AOQT materializer preflight JSON")
	fs.StringVar(&cfg.MetricsJSONPath, "metrics-json", "", "write AOQT sidecar training metrics JSON")
	fs.StringVar(&cfg.OutputArtifactPath, "output", "", "write guarded AOQT candidate package artifact for non-plan training")
	fs.StringVar(&cfg.DevOutputDir, "dev-output-dir", "", "write AOQT V7 dev-only transform/evidence artifacts; requires an explicit V7 dev optimizer mode")
	fs.StringVar(&cfg.ExpectedManifestSHA256, "expected-manifest-sha256", "", "expected calibration manifest sha256")
	fs.StringVar(&cfg.ExpectedRowsSHA256, "expected-rows-sha256", "", "expected calibration rows JSONL sha256")
	fs.StringVar(&cfg.ExpectedPreflightSHA256, "expected-preflight-sha256", "", "expected materializer preflight sha256")
	fs.StringVar(&cfg.ExpectedSplitManifestSHA256, "expected-split-manifest-sha256", "", "expected AOQT V7 dev split manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorArtifactSHA256, "expected-anchor-artifact-sha256", "", "expected immutable anchor artifact sha256")
	fs.StringVar(&cfg.ExpectedAnchorPackageManifestSHA256, "expected-anchor-package-manifest-sha256", "", "expected immutable anchor package manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorEmbeddingSpaceID, "anchor-embedding-space-id", "", "expected anchor embedding space id")
	fs.StringVar(&cfg.ExpectedCompatibilityDigest, "compatibility-digest", "", "expected AOQT compatibility digest")
	fs.StringVar(&cfg.OptimizerMode, "optimizer-mode", "", "AOQT optimizer mode; empty is production-safe, aoqt-v7-dev-trust-region-v1 is V7-v1 dev-only, aoqt-v7-r2-dev-trust-region-v1 requires the fixed r2 probe, aoqt-v7-r3-dev-actual-direction-v1 requires the fixed actual-direction probe, aoqt-v7-r4-dev-actual-coordinate-v1 is probe-only, aoqt-v7-r5-dev-actual-coordinate-train-v1 is dev-only full training")
	fs.StringVar(&cfg.SplitManifestPath, "split-manifest", "", "AOQT V7 dev split manifest to bind into dev evidence")
	fs.StringVar(&cfg.FoldID, "fold-id", "", "AOQT V7 dev split fold id to bind into dev evidence")
	fs.StringVar(&cfg.ForwardConsistencyProbeReceiptJSONPath, "forward-consistency-probe-receipt-json", "", "write the bound V7-r2 forward-consistency probe receipt (probe-only mode)")
	// Shorter spelling retained as a CLI alias for scripts that name the
	// artifact by its role rather than the full probe contract.
	fs.StringVar(&cfg.ForwardConsistencyProbeReceiptJSONPath, "probe-receipt-json", "", "alias for --forward-consistency-probe-receipt-json")
	fs.StringVar(&cfg.ForwardConsistencyProbeReceiptJSONPath, "forward-consistency-probe-receipt", "", "alias for --forward-consistency-probe-receipt-json")
	fs.StringVar(&cfg.ActualDirectionProbeReceiptJSONPath, "actual-direction-probe-receipt-json", "", "write the bound V7-r3 actual-direction probe receipt (probe-only mode)")
	fs.StringVar(&cfg.ActualDirectionProbeReceiptJSONPath, "actual-direction-probe-receipt", "", "alias for --actual-direction-probe-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateProbeReceiptJSONPath, "actual-coordinate-probe-receipt-json", "", "write the bound V7-r4 actual-coordinate probe receipt (probe-only mode)")
	fs.StringVar(&cfg.ActualCoordinateProbeReceiptJSONPath, "actual-coordinate-probe-receipt", "", "alias for --actual-coordinate-probe-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateTrainR4ReceiptJSONPath, "actual-coordinate-train-r4-receipt-json", "", "read the bound canonical V7-r4 receipt authorized for V7-r5 training")
	fs.StringVar(&cfg.ActualCoordinateTrainR4ReceiptJSONPath, "canonical-r4-receipt-json", "", "alias for --actual-coordinate-train-r4-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateTrainR4ReceiptJSONPath, "actual-coordinate-train-r4-receipt", "", "alias for --actual-coordinate-train-r4-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateTrainR4ReceiptJSONPath, "canonical-r4-receipt", "", "alias for --actual-coordinate-train-r4-receipt-json")
	fs.StringVar(&cfg.ExpectedCanonicalR4ReceiptSHA256, "expected-canonical-r4-receipt-sha256", eosruntime.AOQTV7R5CanonicalR4ReceiptSHA256, "pinned raw-file sha256 of the canonical V7-r4 receipt")
	fs.StringVar(&cfg.ExpectedCanonicalR4ReceiptSHA256, "canonical-r4-receipt-sha256", eosruntime.AOQTV7R5CanonicalR4ReceiptSHA256, "alias for --expected-canonical-r4-receipt-sha256")
	fs.StringVar(&cfg.ExpectedCanonicalR4ReceiptSHA256, "actual-coordinate-train-r4-receipt-sha256", eosruntime.AOQTV7R5CanonicalR4ReceiptSHA256, "alias for --expected-canonical-r4-receipt-sha256")
	fs.StringVar(&cfg.ActualCoordinateTrainReceiptJSONPath, "actual-coordinate-train-receipt-json", "", "write the bound V7-r5 actual-coordinate full-training receipt")
	fs.StringVar(&cfg.ActualCoordinateTrainReceiptJSONPath, "actual-coordinate-full-train-receipt-json", "", "alias for --actual-coordinate-train-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateTrainReceiptJSONPath, "actual-coordinate-train-receipt", "", "alias for --actual-coordinate-train-receipt-json")
	fs.StringVar(&cfg.ActualCoordinateTrainReceiptJSONPath, "actual-coordinate-full-train-receipt", "", "alias for --actual-coordinate-train-receipt-json")
	fs.StringVar(&sourceHashes, "source-artifact-sha256", "", "comma-separated expected source artifact sha256 values; for large lists use --source-artifact-sha256-file")
	fs.StringVar(&sourceHashesFile, "source-artifact-sha256-file", "", "newline-delimited expected source artifact sha256 values")
	fs.StringVar(&sourceHashesFileSHA256, "source-artifact-sha256-file-sha256", "", "expected sha256 of --source-artifact-sha256-file")
	fs.StringVar(&vectorHashes, "vector-cache-sha256", "", "comma-separated expected vector cache sha256 values")
	fs.IntVar(&cfg.MaxSteps, "max-steps", 0, "positive optimizer steps for non-plan training")
	fs.Float64Var(&learningRate, "lr", 0, "AOQT optimizer learning rate")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos train-aoqt-sidecar [flags]")
	}
	var err error
	cfg.ExpectedSourceArtifactHashes, err = parseRequiredHashListBinding(sourceHashes, sourceHashesFile, sourceHashesFileSHA256, "source-artifact-sha256")
	if err != nil {
		return err
	}
	cfg.ExpectedVectorCacheHashes, err = parseRequiredHashList(vectorHashes, "vector-cache-sha256")
	if err != nil {
		return err
	}
	cfg.LearningRate = float32(learningRate)
	result, err := eosruntime.RunAOQTSidecarTraining(cfg)
	if err != nil {
		if cfg.OptimizerMode == eosruntime.AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain && result.ActualCoordinateTrainReceiptJSONPath != "" {
			fmt.Printf("AOQT actual-coordinate full-training receipt: %s\n", result.ActualCoordinateTrainReceiptJSONPath)
		}
		if cfg.ActualCoordinateProbeOnly && result.ActualCoordinateProbeReceiptJSONPath != "" {
			fmt.Printf("AOQT actual-coordinate probe receipt: %s\n", result.ActualCoordinateProbeReceiptJSONPath)
		}
		if cfg.ActualDirectionProbeOnly && result.ActualDirectionProbeReceiptJSONPath != "" {
			fmt.Printf("AOQT actual-direction probe receipt: %s\n", result.ActualDirectionProbeReceiptJSONPath)
		}
		if cfg.ForwardConsistencyProbeOnly && result.ForwardConsistencyProbeReceiptJSONPath != "" {
			fmt.Printf("AOQT forward-consistency probe receipt: %s\n", result.ForwardConsistencyProbeReceiptJSONPath)
		}
		return err
	}
	if cfg.ForwardConsistencyProbeOnly {
		fmt.Printf("AOQT forward-consistency probe receipt: %s\n", result.ForwardConsistencyProbeReceiptJSONPath)
		if result.ForwardConsistencyProbe != nil {
			fmt.Printf("probe: passed=%t sha256=%s\n", result.ForwardConsistencyProbe.Passed, result.ForwardConsistencyProbeSHA256)
		}
		return nil
	}
	if cfg.ActualDirectionProbeOnly {
		fmt.Printf("AOQT actual-direction probe receipt: %s\n", result.ActualDirectionProbeReceiptJSONPath)
		if result.ActualDirectionProbe != nil {
			fmt.Printf("probe: passed=%t sha256=%s endpoints=%d candidates=%d\n", result.ActualDirectionProbe.Passed, result.ActualDirectionProbeSHA256, result.ActualDirectionProbe.EndpointCount, result.ActualDirectionProbe.FullTransactionCandidateCount)
		}
		return nil
	}
	if cfg.ActualCoordinateProbeOnly {
		fmt.Printf("AOQT actual-coordinate probe receipt: %s\n", result.ActualCoordinateProbeReceiptJSONPath)
		if result.ActualCoordinateProbe != nil {
			fmt.Printf("probe: passed=%t sha256=%s endpoints=%d eligible=%d\n", result.ActualCoordinateProbe.Passed, result.ActualCoordinateProbeSHA256, result.ActualCoordinateProbe.EndpointCount, result.ActualCoordinateProbe.EligibleEndpointCount)
		}
		return nil
	}
	if cfg.OptimizerMode == eosruntime.AOQTSidecarOptimizerModeV7R5DevActualCoordinateTrain {
		fmt.Printf("AOQT actual-coordinate full-training receipt: %s\n", result.ActualCoordinateTrainReceiptJSONPath)
		if result.ActualCoordinateTrain != nil {
			fmt.Printf("train: passed=%t sha256=%s steps=%d/%d candidates=%d\n", result.ActualCoordinateTrain.Passed, result.ActualCoordinateTrainSHA256, result.ActualCoordinateTrain.AcceptedSteps, result.ActualCoordinateTrain.AttemptedSteps, result.ActualCoordinateTrain.FullCandidateEvaluationCount)
		}
	}
	fmt.Printf("AOQT sidecar metrics: %s\n", cfg.MetricsJSONPath)
	fmt.Printf("plan: rows=%d candidates=%d pairs=%d steps=%d plan_only=%t\n",
		result.IOReport.RowCount,
		result.IOReport.CandidateCount,
		result.IOReport.PairCount,
		result.Metrics.Plan.StepCount,
		result.Metrics.Plan.PlanOnly,
	)
	fmt.Printf("seeds: turboquant=%d topology=%d\n", result.IOReport.TurboQuantSeed, result.IOReport.Topology.Seed)
	if result.PackageResult != nil {
		fmt.Printf("candidate package: %s\n", result.PackageResult.Paths.ArtifactPath)
	}
	if result.DevEvidence != nil {
		fmt.Printf("dev evidence: %s\n", result.DevEvidence.TransformPath)
		fmt.Printf("dev evidence manifest: %s\n", filepath.Join(cfg.DevOutputDir, "aoqt-sidecar-dev-evidence.json"))
	}
	return nil
}

func hasAOQTV7R6ProgressiveScreenFlag(args []string) bool {
	for _, arg := range args {
		if arg == "--progressive-screen-only" || arg == "--optimizer-mode=aoqt-v7-r6-progressive-screen-v1" {
			return true
		}
	}
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--optimizer-mode" && args[i+1] == "aoqt-v7-r6-progressive-screen-v1" {
			return true
		}
	}
	return false
}

func parseRequiredHashListBinding(raw, filePath, fileSHA256, name string) ([]string, error) {
	if filePath != "" {
		if strings.TrimSpace(filePath) == "" {
			return nil, fmt.Errorf("--%s-file path is empty", name)
		}
		if strings.TrimSpace(raw) != "" {
			return nil, fmt.Errorf("--%s and --%s-file are mutually exclusive", name, name)
		}
		return parseRequiredHashListFile(filePath, fileSHA256, name+"-file")
	}
	if strings.TrimSpace(fileSHA256) != "" {
		return nil, fmt.Errorf("--%s-file-sha256 requires --%s-file", name, name)
	}
	return parseRequiredHashList(raw, name)
}

func parseRequiredHashList(raw, name string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("AOQT sidecar training requires --%s", name)
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value == "" {
			return nil, fmt.Errorf("--%s contains an empty sha256", name)
		}
		if err := validateCLIHashListSHA256(value, "--"+name); err != nil {
			return nil, err
		}
		if seen[value] {
			return nil, fmt.Errorf("--%s contains duplicate sha256 %q", name, value)
		}
		seen[value] = true
		out = append(out, value)
	}
	return out, nil
}

func parseRequiredHashListFile(path, expectedSHA256, name string) ([]string, error) {
	if strings.TrimSpace(expectedSHA256) == "" {
		return nil, fmt.Errorf("--%s-sha256 is required when --%s is used", name, name)
	}
	if err := validateCLIHashListSHA256(expectedSHA256, "--"+name+"-sha256"); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read --%s %q: %w", name, path, err)
	}
	sum := sha256.Sum256(data)
	gotSHA256 := hex.EncodeToString(sum[:])
	if gotSHA256 != expectedSHA256 {
		return nil, fmt.Errorf("--%s sha256 = %s, want %s", name, gotSHA256, expectedSHA256)
	}
	lines := strings.Split(string(data), "\n")
	out := make([]string, 0, len(lines))
	seen := map[string]bool{}
	for i, line := range lines {
		if line == "" && i == len(lines)-1 {
			continue
		}
		if strings.TrimSpace(line) == "" {
			return nil, fmt.Errorf("--%s contains an empty sha256 at line %d", name, i+1)
		}
		if line != strings.TrimSpace(line) {
			return nil, fmt.Errorf("--%s line %d must contain only a sha256 value", name, i+1)
		}
		if err := validateCLIHashListSHA256(line, fmt.Sprintf("--%s line %d", name, i+1)); err != nil {
			return nil, err
		}
		if len(out) > 0 && line < out[len(out)-1] {
			return nil, fmt.Errorf("--%s must be sorted in ascending sha256 order; line %d is out of order", name, i+1)
		}
		if seen[line] {
			return nil, fmt.Errorf("--%s contains duplicate sha256 %q at line %d", name, line, i+1)
		}
		seen[line] = true
		out = append(out, line)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("--%s must contain at least one sha256", name)
	}
	return out, nil
}

func validateCLIHashListSHA256(value, label string) error {
	if len(value) != sha256.Size*2 {
		return fmt.Errorf("%s must be %d hex chars", label, sha256.Size*2)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("%s is invalid: %w", label, err)
	}
	return nil
}

func rejectAOQTCandidateExportMLL(artifactPath string) error {
	manifest, err := eosruntime.ReadPackageManifestFile(eosruntime.DefaultPackageManifestPath(artifactPath))
	if err != nil {
		return nil
	}
	if manifest.AOQTTransform.Enabled || manifest.HasFileRole(eosruntime.EmbeddingPostPoolTransformRole) {
		return fmt.Errorf("export-mll rejects AOQT candidate packages; keep the sidecar package layout so AOQT policy/provenance gates remain enforceable")
	}
	return nil
}
