package main

import (
	"flag"
	"fmt"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

// runScreenAOQTV7R6 is intentionally a separate command from
// train-aoqt-sidecar.  The runtime API has no metrics/package/training output
// path, and this command requires both explicit dev/research acknowledgements
// and probe-only mode before it will run.
func runScreenAOQTV7R6(args []string) error {
	fs := flag.NewFlagSet("screen-aoqt-v7-r6", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarV7R6ProgressiveScreenConfig
	var sourceHashes, sourceHashesFile, sourceHashesFileSHA256, vectorHashes string
	var allowResearchOnly, allowDevOnly, probeOnly, progressiveScreenOnly, stopAfterReceipts bool
	var optimizerMode string
	fs.BoolVar(&allowResearchOnly, "allow-research-only-aoqt", false, "explicitly acknowledge research-only AOQT rows")
	fs.BoolVar(&allowDevOnly, "allow-dev-aoqt-v7", false, "explicitly acknowledge V7 dev-only screen")
	fs.BoolVar(&probeOnly, "probe-only", false, "required: run only the V7-r6 screen; no training/metrics/package output")
	fs.BoolVar(&progressiveScreenOnly, "progressive-screen-only", false, "explicitly select the preregistered progressive S512/S1024 screen")
	fs.BoolVar(&stopAfterReceipts, "stop-after-receipts", false, "stop after the S512/S1024 stage receipts; omit optional fixed-K full-objective verification")
	fs.StringVar(&optimizerMode, "optimizer-mode", eosruntime.AOQTSidecarOptimizerModeV7R6ProgressiveScreen, "must remain the V7-r6 progressive-screen mode")
	fs.StringVar(&cfg.ManifestPath, "manifest", "", "strict AOQT calibration manifest JSON")
	fs.StringVar(&cfg.RowsJSONLPath, "rows", "", "strict AOQT calibration rows JSONL")
	fs.StringVar(&cfg.PreflightJSONPath, "preflight", "", "strict AOQT materializer preflight JSON")
	fs.StringVar(&cfg.SplitManifestPath, "split-manifest", "", "AOQT V7 dev split manifest")
	fs.StringVar(&cfg.FoldID, "fold-id", "", "AOQT V7 dev split fold id")
	fs.StringVar(&cfg.ExpectedManifestSHA256, "expected-manifest-sha256", "", "expected calibration manifest sha256")
	fs.StringVar(&cfg.ExpectedRowsSHA256, "expected-rows-sha256", "", "expected calibration rows JSONL sha256")
	fs.StringVar(&cfg.ExpectedPreflightSHA256, "expected-preflight-sha256", "", "expected materializer preflight sha256")
	fs.StringVar(&cfg.ExpectedSplitManifestSHA256, "expected-split-manifest-sha256", "", "expected AOQT V7 dev split manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorArtifactSHA256, "expected-anchor-artifact-sha256", "", "expected immutable anchor artifact sha256")
	fs.StringVar(&cfg.ExpectedAnchorPackageManifestSHA256, "expected-anchor-package-manifest-sha256", "", "expected immutable anchor package manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorEmbeddingSpaceID, "anchor-embedding-space-id", "", "expected anchor embedding space id")
	fs.StringVar(&cfg.ExpectedCompatibilityDigest, "compatibility-digest", "", "expected AOQT compatibility digest")
	fs.StringVar(&sourceHashes, "source-artifact-sha256", "", "comma-separated expected source artifact sha256 values; for large lists use --source-artifact-sha256-file")
	fs.StringVar(&sourceHashesFile, "source-artifact-sha256-file", "", "newline-delimited expected source artifact sha256 values")
	fs.StringVar(&sourceHashesFileSHA256, "source-artifact-sha256-file-sha256", "", "expected sha256 of --source-artifact-sha256-file")
	fs.StringVar(&vectorHashes, "vector-cache-sha256", "", "comma-separated expected vector cache sha256 values")
	fs.StringVar(&cfg.CanonicalR4ReceiptPath, "canonical-r4-receipt-json", "", "bound canonical V7-r4 actual-coordinate receipt")
	fs.StringVar(&cfg.CanonicalR4ReceiptPath, "canonical-r4-receipt", "", "alias for --canonical-r4-receipt-json")
	fs.StringVar(&cfg.CanonicalR4ReceiptSHA256, "canonical-r4-receipt-sha256", "", "expected raw-file sha256 of the canonical V7-r4 receipt")
	fs.StringVar(&cfg.CanonicalR5ReceiptPath, "canonical-r5-receipt-json", "", "bound V7-r5 actual-coordinate receipt")
	fs.StringVar(&cfg.CanonicalR5ReceiptPath, "canonical-r5-receipt", "", "alias for --canonical-r5-receipt-json")
	fs.StringVar(&cfg.CanonicalR5ReceiptSHA256, "canonical-r5-receipt-sha256", "", "expected raw-file sha256 of the V7-r5 receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "progressive-screen-receipt-json", "", "write the bound V7-r6 screen receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "progressive-screen-receipt", "", "alias for --progressive-screen-receipt-json")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "receipt", "", "alias for --progressive-screen-receipt-json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos screen-aoqt-v7-r6 [flags]")
	}
	if !allowResearchOnly || !allowDevOnly || !probeOnly || !progressiveScreenOnly {
		return fmt.Errorf("V7-r6 screen requires --allow-research-only-aoqt, --allow-dev-aoqt-v7, --progressive-screen-only, and --probe-only")
	}
	if optimizerMode != eosruntime.AOQTSidecarOptimizerModeV7R6ProgressiveScreen {
		return fmt.Errorf("V7-r6 screen requires --optimizer-mode=%s", eosruntime.AOQTSidecarOptimizerModeV7R6ProgressiveScreen)
	}
	if stopAfterReceipts {
		return fmt.Errorf("V7-r6 file-backed CLI always requires full verification; stop-after-receipts is available only to in-memory tests")
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) == "" {
		return fmt.Errorf("V7-r6 screen requires --progressive-screen-receipt-json")
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
	cfg.Mode = eosruntime.AOQTSidecarOptimizerModeV7R6ProgressiveScreen
	cfg.ProbeOnly = true
	// The persisted/file-backed command is the authoritative screen path. It
	// always performs the fixed-K full verification; receipt-only execution is
	// intentionally limited to the in-memory runtime API and tests.
	cfg.StopAfterReceipts = false
	cfg.FullVerification = true
	result, err := eosruntime.RunAOQTV7R6ProgressiveScreen(cfg)
	if err != nil {
		if result.ReceiptJSONPath != "" {
			fmt.Printf("AOQT V7-r6 progressive screen receipt: %s\n", result.ReceiptJSONPath)
		}
		return err
	}
	fmt.Printf("AOQT V7-r6 progressive screen receipt: %s\n", result.ReceiptJSONPath)
	fmt.Printf("screen: passed=%t sha256=%s stages=S512:%d,S1024:%d endpoints=%d selected_k=%d objective_evaluations=%d full_verification=%t dense_gate=%s\n", result.Receipt.Passed, result.ReceiptSHA256, result.Receipt.StageS512.Sample.RowCount, result.Receipt.StageS1024.Sample.RowCount, len(result.Receipt.Endpoints), result.Receipt.SelectionK, result.Receipt.ObjectiveEvaluationCount, result.Receipt.FullVerification.Completed, result.Receipt.DenseGateStatus)
	return nil
}

// Keep the function name easy to discover in command-level tests and retain a
// spelling that mirrors the runtime API.
func runProgressiveScreenAOQTSidecar(args []string) error {
	return runScreenAOQTV7R6(args)
}
