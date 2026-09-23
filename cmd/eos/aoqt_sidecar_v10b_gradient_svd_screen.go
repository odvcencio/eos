package main

import (
	"flag"
	"fmt"
	"strings"

	eosruntime "m31labs.dev/eos/runtime"
)

func runScreenAOQTV10bGradientSVD(args []string) error {
	fs := flag.NewFlagSet("screen-aoqt-v10b-gradient-svd", flag.ContinueOnError)
	var cfg eosruntime.AOQTV10bGradientSVDRunConfig
	var allowResearchOnly, allowDevOnly, probeOnly, screenOnly, dryRunOnly bool
	fs.BoolVar(&allowResearchOnly, "allow-research-only-aoqt", false, "explicitly acknowledge research-only AOQT rows")
	fs.BoolVar(&allowDevOnly, "allow-dev-aoqt-v10b", false, "explicitly acknowledge V10b dev-only residual screen")
	fs.BoolVar(&probeOnly, "probe-only", false, "required: derive train-only V10b parameters without evaluator/promotion output")
	fs.BoolVar(&screenOnly, "gradient-svd-screen-only", false, "explicitly select the V10b gradient/SVD residual screen")
	fs.BoolVar(&dryRunOnly, "dry-run-only", false, "required: write only a fail-closed receipt")
	fs.StringVar(&cfg.SourcePath, "gradient-svd-source-json", "", "bound D384 train-gradient/covariance source JSON")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "gradient-svd-screen-receipt-json", "", "write the bound V10b gradient/SVD receipt")
	fs.StringVar(&cfg.OutputReceiptJSONPath, "receipt", "", "alias for --gradient-svd-screen-receipt-json")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptPath, "canonical-v7-r6-receipt-json", "", "bound terminal V7-r6 receipt")
	fs.StringVar(&cfg.CanonicalV7R6ReceiptFileSHA256, "canonical-v7-r6-receipt-sha256", "", "expected terminal V7-r6 receipt file sha256")
	fs.StringVar(&cfg.CanonicalV8ReceiptPath, "canonical-v8-receipt-json", "", "bound terminal V8 receipt")
	fs.StringVar(&cfg.CanonicalV8ReceiptFileSHA256, "canonical-v8-receipt-sha256", "", "expected terminal V8 receipt file sha256")
	fs.StringVar(&cfg.CanonicalV9ReceiptPath, "canonical-v9-receipt-json", "", "bound terminal V9 receipt")
	fs.StringVar(&cfg.CanonicalV9ReceiptFileSHA256, "canonical-v9-receipt-sha256", "", "expected terminal V9 receipt file sha256")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos screen-aoqt-v10b-gradient-svd [flags]")
	}
	if !allowResearchOnly || !allowDevOnly || !probeOnly || !screenOnly || !dryRunOnly {
		return fmt.Errorf("V10b gradient/SVD screen requires --allow-research-only-aoqt, --allow-dev-aoqt-v10b, --gradient-svd-screen-only, --probe-only, and --dry-run-only")
	}
	if strings.TrimSpace(cfg.SourcePath) == "" {
		return fmt.Errorf("V10b gradient/SVD screen requires --gradient-svd-source-json")
	}
	if strings.TrimSpace(cfg.OutputReceiptJSONPath) == "" {
		return fmt.Errorf("V10b gradient/SVD screen requires --gradient-svd-screen-receipt-json")
	}
	result, err := eosruntime.RunAOQTV10bGradientSVDScreen(cfg)
	if err != nil {
		return err
	}
	fmt.Printf("AOQT V10b gradient/SVD screen receipt: %s\n", result.ReceiptJSONPath)
	fmt.Printf("screen: sha256=%s candidate_cap=%d dry_run_only=true evaluator_replay=blocked\n", result.ReceiptSHA256, eosruntime.AOQTV10bGradientSVDCandidateCap)
	return nil
}
