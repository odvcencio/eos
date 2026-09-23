package main

import (
	"strings"
	"testing"
)

func TestRunScreenAOQTV8ConstrainedBasisRequiresExplicitProbeContract(t *testing.T) {
	err := runScreenAOQTV8ConstrainedBasis([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v8",
		"--constrained-basis-screen-only",
		"--probe-only",
		"--optimizer-mode=wrong-mode",
	})
	if err == nil || !strings.Contains(err.Error(), "requires --optimizer-mode=aoqt-v8-constrained-basis-v1") {
		t.Fatalf("invalid V8 mode error = %v", err)
	}
}

func TestRunScreenAOQTV8ConstrainedBasisRequiresScreenOnly(t *testing.T) {
	err := runScreenAOQTV8ConstrainedBasis([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v8",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--constrained-basis-screen-only") {
		t.Fatalf("missing constrained-basis-screen-only error = %v", err)
	}
}

func TestRunScreenAOQTV8ConstrainedBasisRequiresReceiptOutput(t *testing.T) {
	err := runScreenAOQTV8ConstrainedBasis([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v8",
		"--constrained-basis-screen-only",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--constrained-basis-screen-receipt-json") {
		t.Fatalf("missing receipt output error = %v", err)
	}
}

func TestRunScreenAOQTV8ConstrainedBasisRequiresV7R6Receipt(t *testing.T) {
	err := runScreenAOQTV8ConstrainedBasis([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v8",
		"--constrained-basis-screen-only",
		"--probe-only",
		"--constrained-basis-screen-receipt-json=/tmp/v8-receipt.json",
	})
	if err == nil || !strings.Contains(err.Error(), "--canonical-v7-r6-receipt-json") {
		t.Fatalf("missing V7-r6 receipt error = %v", err)
	}
}
