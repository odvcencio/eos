package main

import (
	"strings"
	"testing"
)

func TestRunScreenAOQTV9ActiveSetNullspaceRequiresExplicitProbeContract(t *testing.T) {
	err := runScreenAOQTV9ActiveSetNullspace([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v9",
		"--active-set-nullspace-screen-only",
		"--probe-only",
		"--optimizer-mode=wrong-mode",
	})
	if err == nil || !strings.Contains(err.Error(), "requires --optimizer-mode=aoqt-v9-active-set-nullspace-v1") {
		t.Fatalf("invalid V9 mode error = %v", err)
	}
}

func TestRunScreenAOQTV9ActiveSetNullspaceRequiresScreenOnly(t *testing.T) {
	err := runScreenAOQTV9ActiveSetNullspace([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v9",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--active-set-nullspace-screen-only") {
		t.Fatalf("missing active-set-nullspace-screen-only error = %v", err)
	}
}

func TestRunScreenAOQTV9ActiveSetNullspaceRequiresReceiptOutput(t *testing.T) {
	err := runScreenAOQTV9ActiveSetNullspace([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v9",
		"--active-set-nullspace-screen-only",
		"--probe-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--active-set-nullspace-screen-receipt-json") {
		t.Fatalf("missing receipt output error = %v", err)
	}
}

func TestRunScreenAOQTV9ActiveSetNullspaceRequiresV8Receipt(t *testing.T) {
	err := runScreenAOQTV9ActiveSetNullspace([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v9",
		"--active-set-nullspace-screen-only",
		"--probe-only",
		"--active-set-nullspace-screen-receipt-json=/tmp/v9-receipt.json",
	})
	if err == nil || !strings.Contains(err.Error(), "--canonical-v8-receipt-json") {
		t.Fatalf("missing V8 receipt error = %v", err)
	}
}
