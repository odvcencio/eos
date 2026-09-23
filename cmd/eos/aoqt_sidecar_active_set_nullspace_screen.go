package main

import (
	"flag"
	"fmt"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runScreenAOQTV9ActiveSetNullspace(args []string) error {
	fs := flag.NewFlagSet("screen-aoqt-v9-active-set-nullspace", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarV9ActiveSetNullspaceScreenConfig
	var sourceHashes, sourceHashesFile, sourceHashesFileSHA256, vectorHashes string
	var allowResearchOnly, allowDevOnly, probeOnly, activeSetOnly bool
	var optimizerMode string
	fs.BoolVar(&allowResearchOnly, "allow-research-only-aoqt", false, "explicitly acknowledge research-only AOQT rows")
	fs.BoolVar(&allowDevOnly, "allow-dev-aoqt-v9", false, "explicitly acknowledge V9 dev-only screen")
	fs.BoolVar(&probeOnly, "probe-only", false, "required: run only the V9 active-set screen; no training/metrics/package output")
	fs.BoolVar(&activeSetOnly, "active-set-nullspace-screen-only", false, "explicitly select the preregistered active-set nullspace screen")
	fs.StringVar(&optimizerMode, "optimizer-mode", eosruntime.AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen, "must remain the V9 active-set nullspace screen mode")
	fs.StringVar(&cfg.ManifestPath, "manifest", "", "strict AOQT calibration manifest JSON")
	fs.StringVar(&cfg.RowsJSONLPath, "rows", "", "strict AOQT calibration rows JSONL")
	fs.StringVar(&cfg.PreflightJSONPath, "preflight", "", "strict AOQT materializer preflight JSON")
	fs.StringVar(&cfg.SplitManifestPath, "split-manifest", "", "AOQT V7/V8/V9 dev split manifest")
	fs.StringVar(&cfg.FoldID, "fold-id", "", "AOQT dev split fold id")
	fs.StringVar(&cfg.ExpectedManifestSHA256, "expected-manifest-sha256", "", "expected calibration manifest sha256")
	fs.StringVar(&cfg.ExpectedRowsSHA256, "expected-rows-sha256", "", "expected calibration rows JSONL sha256")
	fs.StringVar(&cfg.ExpectedPreflightSHA256, "expected-preflight-sha256", "", "expected materializer preflight sha256")
	fs.StringVar(&cfg.ExpectedSplitManifestSHA256, "expected-split-manifest-sha256", "", "expected AOQT dev split manifest sha256")
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
	fs.StringVar(&cfg.CanonicalV8ReceiptPath, "canonical-v8-receipt-json", "", "bound terminal V8 constrained-basis receipt")
	fs.StringVar(&cfg.CanonicalV8ReceiptPath, "canonical-v8-receipt", "", "alias for --canonical-v8-receipt-json")
	fs.StringVar(&cfg.CanonicalV8ReceiptFileSHA256, "canonical-v8-receipt-sha256", eosruntime.AOQTV9CanonicalV8ReceiptFileSHA256, "expected raw-file sha256 of the canonical terminal V8 receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "active-set-nullspace-screen-receipt-json", "", "write the bound V9 active-set nullspace receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "active-set-nullspace-screen-receipt", "", "alias for --active-set-nullspace-screen-receipt-json")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "receipt", "", "alias for --active-set-nullspace-screen-receipt-json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos screen-aoqt-v9-active-set-nullspace [flags]")
	}
	if !allowResearchOnly || !allowDevOnly || !probeOnly || !activeSetOnly {
		return fmt.Errorf("V9 active-set nullspace screen requires --allow-research-only-aoqt, --allow-dev-aoqt-v9, --active-set-nullspace-screen-only, and --probe-only")
	}
	if optimizerMode != eosruntime.AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen {
		return fmt.Errorf("V9 active-set nullspace screen requires --optimizer-mode=%s", eosruntime.AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen)
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) == "" {
		return fmt.Errorf("V9 active-set nullspace screen requires --active-set-nullspace-screen-receipt-json")
	}
	if strings.TrimSpace(cfg.CanonicalV8ReceiptPath) == "" {
		return fmt.Errorf("V9 active-set nullspace screen requires --canonical-v8-receipt-json")
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
	cfg.Mode = eosruntime.AOQTSidecarOptimizerModeV9ActiveSetNullspaceScreen
	cfg.ProbeOnly = true
	result, err := eosruntime.RunAOQTV9ActiveSetNullspaceScreen(cfg)
	if err != nil {
		if result.ReceiptJSONPath != "" {
			fmt.Printf("AOQT V9 active-set nullspace screen receipt: %s\n", result.ReceiptJSONPath)
		}
		return err
	}
	fmt.Printf("AOQT V9 active-set nullspace screen receipt: %s\n", result.ReceiptJSONPath)
	fmt.Printf("screen: sha256=%s candidate_cap=%d objective_evaluations_max=%d\n", result.ReceiptSHA256, eosruntime.AOQTV9ActiveSetNullspaceCandidateCap, eosruntime.AOQTV9ActiveSetNullspaceObjectiveEvalMax)
	return nil
}
