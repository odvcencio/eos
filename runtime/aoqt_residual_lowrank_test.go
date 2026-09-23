package eosruntime

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAOQTResidualLowRankParametersDeterministicAndValidated(t *testing.T) {
	a, err := NewAOQTResidualLowRankParameters(4, AOQTResidualLowRankRoleSymmetric, 0.0025, "seed")
	if err != nil {
		t.Fatalf("new params: %v", err)
	}
	b, err := NewAOQTResidualLowRankParameters(4, AOQTResidualLowRankRoleSymmetric, 0.0025, "seed")
	if err != nil {
		t.Fatalf("new params b: %v", err)
	}
	shaA, err := a.SHA256()
	if err != nil {
		t.Fatalf("sha a: %v", err)
	}
	shaB, err := b.SHA256()
	if err != nil {
		t.Fatalf("sha b: %v", err)
	}
	if shaA != shaB {
		t.Fatalf("deterministic parameter hash mismatch: %s != %s", shaA, shaB)
	}
	if len(a.A) != 4 || len(a.A[0]) != AOQTResidualLowRankDim || len(a.B) != AOQTResidualLowRankDim || len(a.B[0]) != 4 {
		t.Fatalf("unexpected parameter shape")
	}
	b.Rank = 7
	if err := b.Validate(); err == nil || !strings.Contains(err.Error(), "rank") {
		t.Fatalf("rank tamper error = %v, want rejection", err)
	}
	if _, err := NewAOQTResidualLowRankParameters(-1, AOQTResidualLowRankRoleSymmetric, 0.0025, "seed"); err == nil || !strings.Contains(err.Error(), "rank") {
		t.Fatalf("negative rank error = %v, want rank rejection before allocation", err)
	}
}

func TestAOQTResidualLowRankApplyPreservesUnitNormAndRejectsBadInput(t *testing.T) {
	params, err := NewAOQTResidualLowRankParameters(8, AOQTResidualLowRankRoleQueryOnly, 0.01, "apply")
	if err != nil {
		t.Fatalf("new params: %v", err)
	}
	vector := make([]float32, AOQTResidualLowRankDim)
	vector[0] = 1
	out, err := params.Apply(vector)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if len(out) != AOQTResidualLowRankDim || math.Abs(aoqtResidualNorm(out)-1) > 2e-5 {
		t.Fatalf("output norm/dim invalid: len=%d norm=%g", len(out), aoqtResidualNorm(out))
	}
	if math.Abs(float64(out[0]-1)) < 1e-7 {
		t.Fatalf("expected nonzero residual movement")
	}
	_, err = params.Apply([]float32{1, 0})
	if err == nil || !strings.Contains(err.Error(), "dim") {
		t.Fatalf("short vector error = %v, want dim rejection", err)
	}
	bad := append([]float32(nil), vector...)
	bad[1] = float32(math.NaN())
	_, err = params.Apply(bad)
	if err == nil || !strings.Contains(err.Error(), "must be finite") {
		t.Fatalf("nonfinite error = %v, want finite rejection", err)
	}
}

func TestValidateAOQTResidualLowRankCacheRejectsMalformedRows(t *testing.T) {
	dir := t.TempDir()
	valid := filepath.Join(dir, "valid.jsonl")
	if err := os.WriteFile(valid, []byte(aoqtResidualLowRankVectorRecordForTest(t, "d1")+"\n"), 0o644); err != nil {
		t.Fatalf("write valid cache: %v", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(valid); err != nil {
		t.Fatalf("valid cache should pass: %v", err)
	}
	duplicate := filepath.Join(dir, "duplicate.jsonl")
	if err := os.WriteFile(duplicate, []byte(aoqtResidualLowRankVectorRecordForTest(t, "d1")+"\n"+aoqtResidualLowRankVectorRecordForTest(t, "d1")+"\n"), 0o644); err != nil {
		t.Fatalf("write duplicate cache: %v", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(duplicate); err == nil || !strings.Contains(err.Error(), "duplicate vector id") {
		t.Fatalf("duplicate cache error = %v, want duplicate rejection", err)
	}
	missing := filepath.Join(dir, "missing.jsonl")
	if err := os.WriteFile(missing, []byte(`{"id":"d1"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write missing cache: %v", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(missing); err == nil || !strings.Contains(err.Error(), "missing vector") {
		t.Fatalf("missing vector error = %v, want missing vector rejection", err)
	}
	short := filepath.Join(dir, "short.jsonl")
	if err := os.WriteFile(short, []byte(`{"id":"d1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write short cache: %v", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(short); err == nil || !strings.Contains(err.Error(), "dim") {
		t.Fatalf("short vector error = %v, want dimension rejection", err)
	}
	nonunit := filepath.Join(dir, "nonunit.jsonl")
	vec := make([]float32, AOQTResidualLowRankDim)
	vec[0] = 2
	data, err := json.Marshal(vec)
	if err != nil {
		t.Fatalf("marshal nonunit vector: %v", err)
	}
	if err := os.WriteFile(nonunit, append([]byte(`{"id":"d1","embedding":`), append(data, []byte("}\n")...)...), 0o644); err != nil {
		t.Fatalf("write nonunit cache: %v", err)
	}
	if _, err := validateAOQTResidualLowRankCacheFile(nonunit); err == nil || !strings.Contains(err.Error(), "unit-normalized") {
		t.Fatalf("nonunit vector error = %v, want norm rejection", err)
	}
}

func TestTransformAOQTResidualLowRankVectorCachesHonorsRoleAndBindsHashes(t *testing.T) {
	dir := t.TempDir()
	docPath := filepath.Join(dir, "docs.jsonl")
	queryPath := filepath.Join(dir, "queries.jsonl")
	unit := make([]float32, AOQTResidualLowRankDim)
	unit[0] = 1
	data, err := json.Marshal(unit)
	if err != nil {
		t.Fatalf("marshal vector: %v", err)
	}
	if err := os.WriteFile(docPath, append([]byte(`{"id":"d1","role":"document","embedding":`), append(data, []byte("}\n")...)...), 0o644); err != nil {
		t.Fatalf("write docs: %v", err)
	}
	if err := os.WriteFile(queryPath, append([]byte(`{"id":"q1","role":"query","vector":`), append(data, []byte("}\n")...)...), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	params, err := NewAOQTResidualLowRankParameters(4, AOQTResidualLowRankRoleQueryOnly, 0.005, "cache")
	if err != nil {
		t.Fatalf("new params: %v", err)
	}
	binding, err := TransformAOQTResidualLowRankVectorCaches(AOQTResidualLowRankCacheTransformConfig{
		Parameters: params, DocVectorPath: docPath, QueryVectorPath: queryPath,
		OutputDocPath: filepath.Join(dir, "out-docs.jsonl"), OutputQueryPath: filepath.Join(dir, "out-queries.jsonl"),
		SidecarPath: filepath.Join(dir, "binding.json"), Dataset: "toy", Artifact: "anchor",
	})
	if err != nil {
		t.Fatalf("transform caches: %v", err)
	}
	if binding.Schema != AOQTResidualLowRankTransformSchema || binding.Inputs.DocVectorSHA256 == "" || binding.Outputs.QueryVectorSHA256 == "" {
		t.Fatalf("binding incomplete: %+v", binding)
	}
	var doc, query map[string]any
	readOneJSONL(t, binding.Outputs.DocVectorPath, &doc)
	readOneJSONL(t, binding.Outputs.QueryVectorPath, &query)
	docVec := doc["embedding"].([]any)
	queryVec := query["vector"].([]any)
	if docVec[0].(float64) != 1 {
		t.Fatalf("query-only role changed document vector: %v", docVec[0])
	}
	if math.Abs(queryVec[0].(float64)-1) < 1e-7 {
		t.Fatalf("query vector did not move")
	}
	if _, err := TransformAOQTResidualLowRankVectorCaches(AOQTResidualLowRankCacheTransformConfig{
		Parameters: params, DocVectorPath: docPath, QueryVectorPath: queryPath,
		OutputDocPath: binding.Outputs.DocVectorPath, OutputQueryPath: filepath.Join(dir, "again-query.jsonl"),
		SidecarPath: filepath.Join(dir, "again-binding.json"),
	}); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("overwrite error = %v, want atomic no-overwrite rejection", err)
	}
}

func aoqtResidualLowRankVectorRecordForTest(t *testing.T, id string) string {
	t.Helper()
	unit := make([]float32, AOQTResidualLowRankDim)
	unit[0] = 1
	data, err := json.Marshal(unit)
	if err != nil {
		t.Fatalf("marshal vector: %v", err)
	}
	return `{"id":"` + id + `","embedding":` + string(data) + `}`
}
