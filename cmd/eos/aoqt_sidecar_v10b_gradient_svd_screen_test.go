package main

import (
	"strings"
	"testing"
)

func TestRunScreenAOQTV10bGradientSVDRequiresExplicitDryRunContract(t *testing.T) {
	err := runScreenAOQTV10bGradientSVD([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10b",
		"--gradient-svd-screen-only",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--dry-run-only") {
		t.Fatalf("missing dry-run contract error = %v", err)
	}
}

func TestRunScreenAOQTV10bGradientSVDRequiresSource(t *testing.T) {
	err := runScreenAOQTV10bGradientSVD([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10b",
		"--gradient-svd-screen-only",
		"--probe-only",
		"--dry-run-only",
		"--gradient-svd-screen-receipt-json=/tmp/receipt.json",
	})
	if err == nil || !strings.Contains(err.Error(), "--gradient-svd-source-json") {
		t.Fatalf("missing source error = %v", err)
	}
}

func TestRunScreenAOQTV10bGradientSVDRequiresReceipt(t *testing.T) {
	err := runScreenAOQTV10bGradientSVD([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10b",
		"--gradient-svd-screen-only",
		"--probe-only",
		"--dry-run-only",
		"--gradient-svd-source-json=/tmp/source.json",
	})
	if err == nil || !strings.Contains(err.Error(), "--gradient-svd-screen-receipt-json") {
		t.Fatalf("missing receipt error = %v", err)
	}
}
