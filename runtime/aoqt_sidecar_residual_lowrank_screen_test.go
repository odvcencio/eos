package eosruntime

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTResidualLowRankDryRunReceiptValidatesCanonicalSchedule(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("validate dry-run receipt: %v", err)
	}
	if receipt.Passed || receipt.ObjectiveEvaluationCount != 0 || len(receipt.Endpoints) != AOQTResidualLowRankCandidateCap {
		t.Fatalf("dry-run accounting invalid: passed=%v evals=%d endpoints=%d", receipt.Passed, receipt.ObjectiveEvaluationCount, len(receipt.Endpoints))
	}
}

func TestAOQTResidualLowRankReceiptRejectsScheduleTamper(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	receipt.Endpoints[0].Alpha = 0.01
	receipt.ResidualScheduleSHA256 = aoqtResidualLowRankTestScheduleSHA(t, receipt.Endpoints)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtResidualLowRankTestChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "canonical schedule") {
		t.Fatalf("schedule tamper error = %v, want canonical schedule rejection", err)
	}
}

func TestAOQTResidualLowRankReceiptRejectsParameterHashTamper(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	receipt.Endpoints[0].ParameterSHA256 = strings.Repeat("1", 64)
	receipt.ResidualScheduleSHA256 = aoqtResidualLowRankTestScheduleSHA(t, receipt.Endpoints)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtResidualLowRankTestChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "parameter hash mismatch") {
		t.Fatalf("parameter tamper error = %v, want parameter hash rejection", err)
	}
}

func TestAOQTResidualLowRankReceiptRejectsClaimsAndFakeMetrics(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	receipt.QualityClaim = true
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "claims") {
		t.Fatalf("claim tamper error = %v, want claims rejection", err)
	}
	receipt = aoqtResidualLowRankTestReceipt(t)
	receipt.Endpoints[0].BaselineReproduced = true
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtResidualLowRankTestChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "dry-run") {
		t.Fatalf("fake baseline error = %v, want dry-run rejection", err)
	}
}

func TestAOQTResidualLowRankValidateBoundRejectsNonCanonicalReceiptFiles(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	if err := receipt.Validate(); err != nil {
		t.Fatalf("synthetic receipt should validate structurally: %v", err)
	}
	if err := receipt.ValidateBound(); err == nil || !strings.Contains(err.Error(), "canonical V7-r6 receipt hash mismatch") {
		t.Fatalf("bound fake receipt error = %v, want canonical file hash rejection", err)
	}
}

func TestAOQTResidualLowRankReceiptRejectsEvaluatorBindingTamper(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	receipt.Binding.TurboQuantBits = []int{3}
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "bits [3,5]") {
		t.Fatalf("bits tamper error = %v, want exact q3/q5 rejection", err)
	}
	receipt = aoqtResidualLowRankTestReceipt(t)
	receipt.Binding.PerQueryTopK = 100
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "topK/perQueryTopK") {
		t.Fatalf("perQueryTopK tamper error = %v, want topK/perQueryTopK rejection", err)
	}
}

func TestAOQTResidualLowRankReceiptRejectsSeedTamper(t *testing.T) {
	receipt := aoqtResidualLowRankTestReceipt(t)
	receipt.Endpoints[0].Seed = "other"
	receipt.ResidualScheduleSHA256 = aoqtResidualLowRankTestScheduleSHA(t, receipt.Endpoints)
	receipt.EndpointHashChain, receipt.EndpointHashChainTailSHA256 = aoqtResidualLowRankTestChain(t, receipt.Endpoints)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "seed must be canonical") {
		t.Fatalf("endpoint seed tamper error = %v, want canonical seed rejection", err)
	}
	receipt = aoqtResidualLowRankTestReceipt(t)
	receipt.SelectionSeedSHA256 = strings.Repeat("9", 64)
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "selection seed") {
		t.Fatalf("selection seed hash tamper error = %v, want replay rejection", err)
	}
	if _, err := BuildAOQTResidualLowRankDryRunReceipt(receipt.Binding, "other"); err == nil || !strings.Contains(err.Error(), "construction seed") {
		t.Fatalf("noncanonical build seed error = %v, want construction seed rejection", err)
	}
}

func TestRunAOQTResidualLowRankRequiresDryRunBeforeOutput(t *testing.T) {
	dir := t.TempDir()
	_, err := RunAOQTResidualLowRankScreen(AOQTSidecarResidualLowRankScreenConfig{
		Mode: AOQTSidecarOptimizerModeResidualLowRankScreen, ProbeOnly: true, DryRunOnly: false,
		OutputReceiptJSONPath: filepath.Join(dir, "receipt.json"),
	})
	if err == nil || !strings.Contains(err.Error(), "dry-run-only=true") {
		t.Fatalf("dry-run guard error = %v, want dry-run-only rejection", err)
	}
	if _, statErr := os.Stat(filepath.Join(dir, "receipt.json")); !os.IsNotExist(statErr) {
		t.Fatalf("receipt output should not exist after guard failure: %v", statErr)
	}
}

func TestRunAOQTResidualLowRankValidatesBaseCachesBeforeReceiptsOrOutput(t *testing.T) {
	dir := t.TempDir()
	doc := filepath.Join(dir, "bad-doc.jsonl")
	query := filepath.Join(dir, "query.jsonl")
	v7 := filepath.Join(dir, "v7.json")
	v8 := filepath.Join(dir, "v8.json")
	v9 := filepath.Join(dir, "v9.json")
	for _, item := range []struct {
		path string
		data string
	}{
		{doc, `{"id":"d1","embedding":[1,0]}` + "\n"},
		{query, aoqtResidualLowRankVectorRecordForTest(t, "q1") + "\n"},
		{v7, "v7\n"}, {v8, "v8\n"}, {v9, "v9\n"},
	} {
		if err := os.WriteFile(item.path, []byte(item.data), 0o644); err != nil {
			t.Fatalf("write %s: %v", item.path, err)
		}
	}
	out := filepath.Join(dir, "receipt.json")
	_, err := RunAOQTResidualLowRankScreen(AOQTSidecarResidualLowRankScreenConfig{
		Mode: AOQTSidecarOptimizerModeResidualLowRankScreen, ProbeOnly: true, DryRunOnly: true,
		OutputReceiptJSONPath: out, BaseDocVectorPath: doc, BaseQueryVectorPath: query,
		CanonicalV7R6ReceiptPath: v7, CanonicalV8ReceiptPath: v8, CanonicalV9ReceiptPath: v9,
		ExpectedAnchorArtifactSHA256: strings.Repeat("a", 64), ExpectedAnchorPackageManifestSHA256: strings.Repeat("b", 64),
		ExpectedAnchorEmbeddingSpaceID: "eos-d384-test", ExpectedCompatibilityDigest: strings.Repeat("c", 64),
		ExpectedWorkloadSHA256: strings.Repeat("d", 64), ExpectedQrelsSHA256: strings.Repeat("e", 64),
		ExpectedTurboQuantBits: []int{3, 5}, ExpectedTurboQuantSeed: AOQTResidualLowRankTurboQuantSeed,
		ExpectedTopK: AOQTResidualLowRankTopK, ExpectedPerQueryTopK: AOQTResidualLowRankTopK,
	})
	if err == nil || !strings.Contains(err.Error(), "base doc") || !strings.Contains(err.Error(), "dim") {
		t.Fatalf("bad base cache error = %v, want D384 cache rejection", err)
	}
	if _, statErr := os.Stat(out); !os.IsNotExist(statErr) {
		t.Fatalf("receipt output should not exist after cache guard failure: %v", statErr)
	}
}

func aoqtResidualLowRankTestReceipt(t *testing.T) AOQTSidecarResidualLowRankScreenReceipt {
	t.Helper()
	dir := t.TempDir()
	doc := filepath.Join(dir, "doc.jsonl")
	query := filepath.Join(dir, "query.jsonl")
	v7 := filepath.Join(dir, "v7.json")
	v8 := filepath.Join(dir, "v8.json")
	v9 := filepath.Join(dir, "v9.json")
	for _, item := range []struct {
		path string
		data string
	}{
		{doc, aoqtResidualLowRankVectorRecordForTest(t, "d1") + "\n"},
		{query, aoqtResidualLowRankVectorRecordForTest(t, "q1") + "\n"},
		{v7, "v7\n"}, {v8, "v8\n"}, {v9, "v9\n"},
	} {
		if err := os.WriteFile(item.path, []byte(item.data), 0o644); err != nil {
			t.Fatalf("write %s: %v", item.path, err)
		}
	}
	binding := AOQTResidualLowRankScreenBinding{
		BaseDocVectorPath: doc, BaseDocVectorSHA256: shaFileForTest(t, doc),
		BaseQueryVectorPath: query, BaseQueryVectorSHA256: shaFileForTest(t, query),
		CanonicalV7R6ReceiptPath: v7, CanonicalV7R6ReceiptFileSHA256: AOQTResidualLowRankCanonicalV7R6ReceiptSHA256,
		CanonicalV8ReceiptPath: v8, CanonicalV8ReceiptFileSHA256: AOQTResidualLowRankCanonicalV8ReceiptSHA256,
		CanonicalV9ReceiptPath: v9, CanonicalV9ReceiptFileSHA256: AOQTResidualLowRankCanonicalV9ReceiptSHA256,
		AnchorArtifactSHA256: strings.Repeat("a", 64), AnchorPackageManifestSHA256: strings.Repeat("b", 64),
		AnchorEmbeddingSpaceID: "eos-d384-test", CompatibilityDigest: strings.Repeat("c", 64),
		WorkloadSHA256: strings.Repeat("d", 64), QrelsSHA256: strings.Repeat("e", 64),
		TurboQuantBits: []int{3, 5}, TurboQuantSeed: 5581486560434873699, TopK: 120, PerQueryTopK: 120,
	}
	receipt, err := BuildAOQTResidualLowRankDryRunReceipt(binding, AOQTResidualLowRankSelectionSeed)
	if err != nil {
		t.Fatalf("build dry-run receipt: %v", err)
	}
	return receipt
}

func shaFileForTest(t *testing.T, path string) string {
	t.Helper()
	sha, err := sha256FileHexAOQTVectorCache(path)
	if err != nil {
		t.Fatalf("sha %s: %v", path, err)
	}
	return sha
}

func aoqtResidualLowRankTestScheduleSHA(t *testing.T, endpoints []AOQTResidualLowRankEndpoint) string {
	t.Helper()
	sha, err := aoqtResidualLowRankScheduleSHA256(endpoints)
	if err != nil {
		t.Fatalf("schedule sha: %v", err)
	}
	return sha
}

func aoqtResidualLowRankTestChain(t *testing.T, endpoints []AOQTResidualLowRankEndpoint) (string, string) {
	t.Helper()
	chain, tail, err := aoqtResidualLowRankEndpointHashChain(endpoints)
	if err != nil {
		t.Fatalf("chain: %v", err)
	}
	return chain, tail
}
