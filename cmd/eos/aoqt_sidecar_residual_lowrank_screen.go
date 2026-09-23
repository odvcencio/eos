package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runScreenAOQTResidualLowRank(args []string) error {
	fs := flag.NewFlagSet("screen-aoqt-v10-residual-lowrank", flag.ContinueOnError)
	var cfg eosruntime.AOQTSidecarResidualLowRankScreenConfig
	var allowResearchOnly, allowDevOnly, probeOnly, screenOnly, dryRunOnly bool
	var optimizerMode, bits string
	fs.BoolVar(&allowResearchOnly, "allow-research-only-aoqt", false, "explicitly acknowledge research-only AOQT rows")
	fs.BoolVar(&allowDevOnly, "allow-dev-aoqt-v10", false, "explicitly acknowledge V10 dev-only residual screen")
	fs.BoolVar(&probeOnly, "probe-only", false, "required: run only the V10 residual screen; no training/metrics/package output")
	fs.BoolVar(&screenOnly, "residual-lowrank-screen-only", false, "explicitly select the preregistered residual low-rank screen")
	fs.BoolVar(&dryRunOnly, "dry-run-only", false, "required: write only a fail-closed dry-run receipt")
	fs.StringVar(&optimizerMode, "optimizer-mode", eosruntime.AOQTSidecarOptimizerModeResidualLowRankScreen, "must remain the V10 residual low-rank mode")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "residual-lowrank-screen-receipt-json", "", "write the bound V10 residual low-rank receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "residual-lowrank-screen-receipt", "", "alias for --residual-lowrank-screen-receipt-json")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "receipt", "", "alias for --residual-lowrank-screen-receipt-json")
	fs.StringVar(&cfg.BaseDocVectorPath, "base-doc-vectors", "", "base D384 document vector cache JSONL")
	fs.StringVar(&cfg.BaseQueryVectorPath, "base-query-vectors", "", "base D384 query vector cache JSONL")
	fs.StringVar(&cfg.ExpectedDocSHA256, "expected-base-doc-vectors-sha256", "", "expected base document vector cache sha256")
	fs.StringVar(&cfg.ExpectedQuerySHA256, "expected-base-query-vectors-sha256", "", "expected base query vector cache sha256")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptPath, "canonical-v7-r6-receipt-json", "", "bound terminal V7-r6 receipt")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptFileSHA256, "canonical-v7-r6-receipt-sha256", eosruntime.AOQTResidualLowRankCanonicalV7R6ReceiptSHA256, "expected terminal V7-r6 receipt file sha256")
	fs.StringVar(&cfg.CanonicalV8ReceiptPath, "canonical-v8-receipt-json", "", "bound terminal V8 receipt")
	fs.StringVar(&cfg.CanonicalV8ReceiptFileSHA256, "canonical-v8-receipt-sha256", eosruntime.AOQTResidualLowRankCanonicalV8ReceiptSHA256, "expected terminal V8 receipt file sha256")
	fs.StringVar(&cfg.CanonicalV9ReceiptPath, "canonical-v9-receipt-json", "", "bound terminal V9 receipt")
	fs.StringVar(&cfg.CanonicalV9ReceiptFileSHA256, "canonical-v9-receipt-sha256", eosruntime.AOQTResidualLowRankCanonicalV9ReceiptSHA256, "expected terminal V9 receipt file sha256")
	fs.StringVar(&cfg.ExpectedAnchorArtifactSHA256, "expected-anchor-artifact-sha256", "", "expected immutable anchor artifact sha256")
	fs.StringVar(&cfg.ExpectedAnchorPackageManifestSHA256, "expected-anchor-package-manifest-sha256", "", "expected immutable anchor package manifest sha256")
	fs.StringVar(&cfg.ExpectedAnchorEmbeddingSpaceID, "anchor-embedding-space-id", "", "expected anchor embedding space id")
	fs.StringVar(&cfg.ExpectedCompatibilityDigest, "compatibility-digest", "", "expected AOQT compatibility digest")
	fs.StringVar(&cfg.ExpectedWorkloadSHA256, "workload-sha256", "", "expected workload sha256")
	fs.StringVar(&cfg.ExpectedQrelsSHA256, "qrels-sha256", "", "expected qrels sha256")
	fs.StringVar(&bits, "turboquant-bits", "3,5", "required TurboQuant bit widths; V10 requires 3,5")
	fs.Int64Var(&cfg.ExpectedTurboQuantSeed, "turboquant-seed", eosruntime.AOQTResidualLowRankTurboQuantSeed, "required TurboQuant quantizer seed")
	fs.IntVar(&cfg.ExpectedTopK, "top-k", eosruntime.AOQTResidualLowRankTopK, "required topK/perQueryTopK binding")
	fs.IntVar(&cfg.ExpectedPerQueryTopK, "per-query-top-k", eosruntime.AOQTResidualLowRankTopK, "required perQueryTopK binding")
	fs.StringVar(&cfg.ConstructionSeed, "construction-seed", eosruntime.AOQTResidualLowRankSelectionSeed, "deterministic low-rank construction seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos screen-aoqt-v10-residual-lowrank [flags]")
	}
	if !allowResearchOnly || !allowDevOnly || !probeOnly || !screenOnly || !dryRunOnly {
		return fmt.Errorf("V10 residual low-rank screen requires --allow-research-only-aoqt, --allow-dev-aoqt-v10, --residual-lowrank-screen-only, --probe-only, and --dry-run-only")
	}
	if optimizerMode != eosruntime.AOQTSidecarOptimizerModeResidualLowRankScreen {
		return fmt.Errorf("V10 residual low-rank screen requires --optimizer-mode=%s", eosruntime.AOQTSidecarOptimizerModeResidualLowRankScreen)
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) == "" {
		return fmt.Errorf("V10 residual low-rank screen requires --residual-lowrank-screen-receipt-json")
	}
	parsedBits, err := parseAOQTResidualLowRankBits(bits)
	if err != nil {
		return err
	}
	cfg.ExpectedTurboQuantBits = parsedBits
	cfg.Mode = eosruntime.AOQTSidecarOptimizerModeResidualLowRankScreen
	cfg.ProbeOnly = true
	cfg.DryRunOnly = true
	result, err := eosruntime.RunAOQTResidualLowRankScreen(cfg)
	if err != nil {
		if result.ReceiptJSONPath != "" {
			fmt.Printf("AOQT V10 residual low-rank screen receipt: %s\n", result.ReceiptJSONPath)
		}
		return err
	}
	fmt.Printf("AOQT V10 residual low-rank screen receipt: %s\n", result.ReceiptJSONPath)
	fmt.Printf("screen: sha256=%s candidate_cap=%d objective_evaluations_max=%d dry_run_only=true\n", result.ReceiptSHA256, eosruntime.AOQTResidualLowRankCandidateCap, eosruntime.AOQTResidualLowRankObjectiveEvalMax)
	return nil
}

func parseAOQTResidualLowRankBits(value string) ([]int, error) {
	parts := strings.Split(value, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		bit, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid TurboQuant bit width %q", part)
		}
		out = append(out, bit)
	}
	return out, nil
}
