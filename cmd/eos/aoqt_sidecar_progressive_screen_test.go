package main

import (
	"strings"
	"testing"
)

func TestRunScreenAOQTV7R6RequiresExplicitProbeContract(t *testing.T) {
	err := runScreenAOQTV7R6([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v7",
		"--progressive-screen-only",
		"--probe-only",
		"--optimizer-mode=wrong-mode",
	})
	if err == nil || !strings.Contains(err.Error(), "requires --optimizer-mode=aoqt-v7-r6-progressive-screen-v1") {
		t.Fatalf("invalid V7-r6 mode error = %v", err)
	}
}

func TestRunScreenAOQTV7R6RequiresProgressiveScreenOnly(t *testing.T) {
	err := runScreenAOQTV7R6([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v7",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--progressive-screen-only") {
		t.Fatalf("missing progressive-screen-only error = %v", err)
	}
}

func TestRunTrainAOQTSidecarRoutesV7R6ToScreen(t *testing.T) {
	err := runTrainAOQTSidecar([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v7",
		"--progressive-screen-only",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--progressive-screen-receipt-json") {
		t.Fatalf("train command V7-r6 route error = %v", err)
	}
}

func TestRunScreenAOQTV7R6RequiresReceiptOutput(t *testing.T) {
	err := runScreenAOQTV7R6([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v7",
		"--progressive-screen-only",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--progressive-screen-receipt-json") {
		t.Fatalf("missing receipt output error = %v", err)
	}
}
