package main

import (
	"flag"
	"fmt"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runScreenAOQTV8ConstrainedBasis(args []string) error {
	fs := flag.NewFlagSet("screen-aoqt-v8-constrained-basis", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarV8ConstrainedBasisScreenConfig
	var sourceHashes, sourceHashesFile, sourceHashesFileSHA256, vectorHashes string
	var allowResearchOnly, allowDevOnly, probeOnly, constrainedBasisOnly bool
	var optimizerMode string
	fs.BoolVar(&allowResearchOnly, "allow-research-only-aoqt", false, "explicitly acknowledge research-only AOQT rows")
	fs.BoolVar(&allowDevOnly, "allow-dev-aoqt-v8", false, "explicitly acknowledge V8 dev-only screen")
	fs.BoolVar(&probeOnly, "probe-only", false, "required: run only the V8 constrained-basis screen; no training/metrics/package output")
	fs.BoolVar(&constrainedBasisOnly, "constrained-basis-screen-only", false, "explicitly select the preregistered constrained-basis screen")
	fs.StringVar(&optimizerMode, "optimizer-mode", eosruntime.AOQTSidecarOptimizerModeV8ConstrainedBasisScreen, "must remain the V8 constrained-basis screen mode")
	fs.StringVar(&cfg.ManifestPath, "manifest", "", "strict AOQT calibration manifest JSON")
	fs.StringVar(&cfg.RowsJSONLPath, "rows", "", "strict AOQT calibration rows JSONL")
	fs.StringVar(&cfg.PreflightJSONPath, "preflight", "", "strict AOQT materializer preflight JSON")
	fs.StringVar(&cfg.SplitManifestPath, "split-manifest", "", "AOQT V7/V8 dev split manifest")
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
	fs.StringVar(&cfg.CanonicalV7R6ReceiptPath, "canonical-v7-r6-receipt-json", "", "bound terminal V7-r6 progressive-screen receipt")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptPath, "canonical-v7-r6-receipt", "", "alias for --canonical-v7-r6-receipt-json")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptFileSHA256, "canonical-v7-r6-receipt-sha256", eosruntime.AOQTV8CanonicalV7R6ReceiptFileSHA256, "expected raw-file sha256 of the canonical terminal V7-r6 receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "constrained-basis-screen-receipt-json", "", "write the bound V8 constrained-basis screen receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "constrained-basis-screen-receipt", "", "alias for --constrained-basis-screen-receipt-json")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "receipt", "", "alias for --constrained-basis-screen-receipt-json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos screen-aoqt-v8-constrained-basis [flags]")
	}
	if !allowResearchOnly || !allowDevOnly || !probeOnly || !constrainedBasisOnly {
		return fmt.Errorf("V8 constrained-basis screen requires --allow-research-only-aoqt, --allow-dev-aoqt-v8, --constrained-basis-screen-only, and --probe-only")
	}
	if optimizerMode != eosruntime.AOQTSidecarOptimizerModeV8ConstrainedBasisScreen {
		return fmt.Errorf("V8 constrained-basis screen requires --optimizer-mode=%s", eosruntime.AOQTSidecarOptimizerModeV8ConstrainedBasisScreen)
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) == "" {
		return fmt.Errorf("V8 constrained-basis screen requires --constrained-basis-screen-receipt-json")
	}
	if strings.TrimSpace(cfg.CanonicalV7R6ReceiptPath) == "" {
		return fmt.Errorf("V8 constrained-basis screen requires --canonical-v7-r6-receipt-json")
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
	cfg.Mode = eosruntime.AOQTSidecarOptimizerModeV8ConstrainedBasisScreen
	cfg.ProbeOnly = true
	result, err := eosruntime.RunAOQTV8ConstrainedBasisScreen(cfg)
	if err != nil {
		if result.ReceiptJSONPath != "" {
			fmt.Printf("AOQT V8 constrained-basis screen receipt: %s\n", result.ReceiptJSONPath)
		}
		return err
	}
	fmt.Printf("AOQT V8 constrained-basis screen receipt: %s\n", result.ReceiptJSONPath)
	fmt.Printf("screen: passed=%t sha256=%s endpoints=%d feasible=%d objective_evaluations=%d dense_invariant=%s candidate_cap=%d\n", result.Receipt.Passed, result.ReceiptSHA256, len(result.Receipt.Endpoints), result.Receipt.FeasibleEndpointCount, result.Receipt.ObjectiveEvaluationCount, result.Receipt.DenseInvariantStatus, result.Receipt.CandidateCap)
	return nil
}
