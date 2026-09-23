package eosruntime

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTransformAOQTRetrievalVectorCachesIdentityPreservesIDsRolesAndBindsHashes(t *testing.T) {
	dir := t.TempDir()
	transform := testAOQTVectorCacheTransform(t, 2, 0)
	transformPath := filepath.Join(dir, "transform.json")
	if err := transform.WriteFile(transformPath); err != nil {
		t.Fatalf("write transform: %v", err)
	}
	docPath := filepath.Join(dir, "docs.jsonl")
	queryPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(docPath, []byte(`{"_id":"d1","role":"document","embedding":[1,0],"metadata":{"keep":true}}`+"\n"), 0o644); err != nil {
		t.Fatalf("write docs: %v", err)
	}
	if err := os.WriteFile(queryPath, []byte(`{"id":"q1","role":"query","vector":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	outDoc := filepath.Join(dir, "out-docs.jsonl")
	outQuery := filepath.Join(dir, "out-queries.jsonl")
	sidecar := filepath.Join(dir, "binding.json")
	binding, err := TransformAOQTRetrievalVectorCaches(AOQTVectorCacheTransformConfig{
		TransformPath:      transformPath,
		DocVectorPath:      docPath,
		QueryVectorPath:    queryPath,
		OutputDocPath:      outDoc,
		OutputQueryPath:    outQuery,
		SidecarPath:        sidecar,
		ExpectedDim:        2,
		ExpectedStages:     1,
		ExpectedPairsStage: 1,
		ExpectedAngleCount: 1,
		Dataset:            "toy",
		Artifact:           "candidate",
	})
	if err != nil {
		t.Fatalf("transform caches: %v", err)
	}
	if binding.Schema != AOQTVectorCacheTransformBindingSchema || binding.Inputs.DocVectorSHA256 == "" || binding.Outputs.QueryVectorSHA256 == "" {
		t.Fatalf("binding incomplete: %+v", binding)
	}
	var doc map[string]any
	readOneJSONL(t, outDoc, &doc)
	if doc["_id"] != "d1" || doc["role"] != "document" {
		t.Fatalf("identity fields not preserved: %+v", doc)
	}
	embedding := doc["embedding"].([]any)
	if embedding[0].(float64) != 1 || embedding[1].(float64) != 0 {
		t.Fatalf("identity vector changed: %+v", embedding)
	}
	var persisted AOQTVectorCacheTransformBinding
	data, err := os.ReadFile(sidecar)
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("decode sidecar: %v", err)
	}
	if persisted.Outputs.DocVectorSHA256 != binding.Outputs.DocVectorSHA256 || persisted.Topology.PairingsSHA256 != binding.Topology.PairingsSHA256 {
		t.Fatalf("persisted binding mismatch: %+v vs %+v", persisted, binding)
	}
}

func TestTransformAOQTRetrievalVectorCachesAppliesRotation(t *testing.T) {
	dir := t.TempDir()
	transform := testAOQTVectorCacheTransform(t, 2, math.Pi/2)
	transformPath := filepath.Join(dir, "transform.json")
	if err := transform.WriteFile(transformPath); err != nil {
		t.Fatalf("write transform: %v", err)
	}
	docPath := filepath.Join(dir, "docs.jsonl")
	queryPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(docPath, []byte(`{"id":"d1","values":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write docs: %v", err)
	}
	if err := os.WriteFile(queryPath, []byte(`{"id":"q1","embedding":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	_, err := TransformAOQTRetrievalVectorCaches(AOQTVectorCacheTransformConfig{
		TransformPath:      transformPath,
		DocVectorPath:      docPath,
		QueryVectorPath:    queryPath,
		OutputDocPath:      filepath.Join(dir, "out-docs.jsonl"),
		OutputQueryPath:    filepath.Join(dir, "out-queries.jsonl"),
		SidecarPath:        filepath.Join(dir, "binding.json"),
		ExpectedDim:        2,
		ExpectedStages:     1,
		ExpectedPairsStage: 1,
		ExpectedAngleCount: 1,
	})
	if err != nil {
		t.Fatalf("transform caches: %v", err)
	}
	var doc map[string]any
	readOneJSONL(t, filepath.Join(dir, "out-docs.jsonl"), &doc)
	values := doc["values"].([]any)
	if math.Abs(values[0].(float64)) > 1e-6 || math.Abs(values[1].(float64)-1) > 1e-6 {
		t.Fatalf("rotated values = %+v, want [0,1]", values)
	}
}

func TestTransformAOQTRetrievalVectorCachesRejectsMissingAuditAndVectorField(t *testing.T) {
	dir := t.TempDir()
	transform := testAOQTVectorCacheTransform(t, 2, 0)
	transform.Audit = AOQTAuditInfo{}
	transformPath := filepath.Join(dir, "transform.json")
	if err := transform.WriteFile(transformPath); err != nil {
		t.Fatalf("write transform: %v", err)
	}
	docPath := filepath.Join(dir, "docs.jsonl")
	queryPath := filepath.Join(dir, "queries.jsonl")
	if err := os.WriteFile(docPath, []byte(`{"id":"d1","embedding":[1,0]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write docs: %v", err)
	}
	if err := os.WriteFile(queryPath, []byte(`{"id":"q1","embedding":[0,1]}`+"\n"), 0o644); err != nil {
		t.Fatalf("write queries: %v", err)
	}
	_, err := TransformAOQTRetrievalVectorCaches(AOQTVectorCacheTransformConfig{
		TransformPath: transformPath,
		DocVectorPath: docPath, QueryVectorPath: queryPath,
		OutputDocPath: filepath.Join(dir, "out-docs.jsonl"), OutputQueryPath: filepath.Join(dir, "out-queries.jsonl"), SidecarPath: filepath.Join(dir, "binding.json"),
		ExpectedDim: 2, ExpectedStages: 1, ExpectedPairsStage: 1, ExpectedAngleCount: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "audit must bind") {
		t.Fatalf("err = %v, want audit binding failure", err)
	}

	transform = testAOQTVectorCacheTransform(t, 2, 0)
	if err := transform.WriteFile(transformPath + ".ok"); err != nil {
		t.Fatalf("write transform ok: %v", err)
	}
	if err := os.WriteFile(docPath+".bad", []byte(`{"id":"d1","role":"document"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write bad docs: %v", err)
	}
	_, err = TransformAOQTRetrievalVectorCaches(AOQTVectorCacheTransformConfig{
		TransformPath: transformPath + ".ok",
		DocVectorPath: docPath + ".bad", QueryVectorPath: queryPath,
		OutputDocPath: filepath.Join(dir, "bad-out-docs.jsonl"), OutputQueryPath: filepath.Join(dir, "bad-out-queries.jsonl"), SidecarPath: filepath.Join(dir, "bad-binding.json"),
		ExpectedDim: 2, ExpectedStages: 1, ExpectedPairsStage: 1, ExpectedAngleCount: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "missing vector") {
		t.Fatalf("err = %v, want missing vector failure", err)
	}
}

func testAOQTVectorCacheTransform(t *testing.T, dim int, angle float64) AOQTGivensTransform {
	t.Helper()
	transform := AOQTGivensTransform{
		Version: AOQTTransformVersion,
		Kind:    EmbeddingPostPoolTransformAOQTGivens,
		Dim:     dim,
		Seed:    191,
		Stages: []AOQTStage{{
			Pairs:  [][2]int{{0, 1}},
			Angles: []float32{float32(angle)},
		}},
	}
	pairings, err := transform.PairingsSHA256()
	if err != nil {
		t.Fatalf("pairings: %v", err)
	}
	angles, err := transform.AnglesSHA256()
	if err != nil {
		t.Fatalf("angles: %v", err)
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		t.Fatalf("orth: %v", err)
	}
	transform.Audit = AOQTAuditInfo{PairingsSHA256: pairings, AnglesSHA256: angles, Orthogonality: &orth}
	return transform
}

func readOneJSONL(t *testing.T, path string, out any) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		t.Fatalf("%s: no rows", path)
	}
	if err := json.Unmarshal(scanner.Bytes(), out); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}
