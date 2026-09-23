package main

import (
	"strings"
	"testing"
)

func TestRunScreenAOQTResidualLowRankRequiresExplicitDryRunContract(t *testing.T) {
	err := runScreenAOQTResidualLowRank([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10",
		"--residual-lowrank-screen-only",
		"--probe-only",
		"--optimizer-mode=wrong",
	})
	if err == nil || !strings.Contains(err.Error(), "--dry-run-only") {
		t.Fatalf("missing dry-run contract error = %v", err)
	}
}

func TestRunScreenAOQTResidualLowRankRequiresCanonicalMode(t *testing.T) {
	err := runScreenAOQTResidualLowRank([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10",
		"--residual-lowrank-screen-only",
		"--probe-only",
		"--dry-run-only",
		"--optimizer-mode=wrong",
	})
	if err == nil || !strings.Contains(err.Error(), "requires --optimizer-mode=eos-d384-residual-lowrank-screen-v1") {
		t.Fatalf("invalid mode error = %v", err)
	}
}

func TestRunScreenAOQTResidualLowRankRequiresReceiptOutput(t *testing.T) {
	err := runScreenAOQTResidualLowRank([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10",
		"--residual-lowrank-screen-only",
		"--probe-only",
		"--dry-run-only",
	})
	if err == nil || !strings.Contains(err.Error(), "--residual-lowrank-screen-receipt-json") {
		t.Fatalf("missing receipt output error = %v", err)
	}
}

func TestRunScreenAOQTResidualLowRankRejectsNonCanonicalConstructionSeed(t *testing.T) {
	err := runScreenAOQTResidualLowRank([]string{
		"--allow-research-only-aoqt",
		"--allow-dev-aoqt-v10",
		"--residual-lowrank-screen-only",
		"--probe-only",
		"--dry-run-only",
		"--residual-lowrank-screen-receipt-json=/tmp/v10-receipt.json",
		"--construction-seed=other",
	})
	if err == nil || !strings.Contains(err.Error(), "construction seed") {
		t.Fatalf("noncanonical construction seed error = %v, want seed rejection", err)
	}
}

func TestParseAOQTResidualLowRankBits(t *testing.T) {
	bits, err := parseAOQTResidualLowRankBits("3,5")
	if err != nil {
		t.Fatalf("parse bits: %v", err)
	}
	if len(bits) != 2 || bits[0] != 3 || bits[1] != 5 {
		t.Fatalf("bits = %+v, want [3 5]", bits)
	}
	if _, err := parseAOQTResidualLowRankBits("3,nope"); err == nil {
		t.Fatalf("invalid bit width should fail")
	}
}
