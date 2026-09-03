package eosruntime

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"m31labs.dev/turboquant"
)

func TestAOQTMaterializerRecomputesPreparedIPAndDenseScores(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	preflight, err := MaterializeAOQTSidecarCalibration(cfg)
	if err != nil {
		t.Fatalf("materialize: %v", err)
	}
	if preflight.QualityClaim || preflight.ActualTrainingRan || preflight.ActualEvalRan || preflight.ActualVectorExportRan {
		t.Fatalf("preflight claims unsafe work: %+v", preflight)
	}
	row := readOneAOQTRow(t, cfg.RowJSONLPath)
	q3 := turboquant.NewIPWithSeed(AOQTSidecarDim, 3, AOQTSidecarMaterializerQuantSeed)
	q5 := turboquant.NewIPWithSeed(AOQTSidecarDim, 5, AOQTSidecarMaterializerQuantSeed)
	pq3 := q3.PrepareQuery(row.QueryVector)
	pq5 := q5.PrepareQuery(row.QueryVector)
	for i, candidate := range row.CandidateVectors {
		if got, want := row.AnchorScores.Dense[i], dotAOQT(row.QueryVector, candidate); math.Abs(float64(got-want)) > 1e-7 {
			t.Fatalf("dense[%d] = %.9g, want recomputed %.9g", i, got, want)
		}
		if got, want := row.AnchorScores.Q3[i], q3.InnerProductPrepared(q3.Quantize(candidate), pq3); math.Abs(float64(got-want)) > 1e-7 {
			t.Fatalf("q3[%d] = %.9g, want prepared-IP %.9g", i, got, want)
		}
		if got, want := row.AnchorScores.Q5[i], q5.InnerProductPrepared(q5.Quantize(candidate), pq5); math.Abs(float64(got-want)) > 1e-7 {
			t.Fatalf("q5[%d] = %.9g, want prepared-IP %.9g", i, got, want)
		}
		if row.AnchorScores.Q3[i] == -999 || row.AnchorScores.Q5[i] == -999 {
			t.Fatalf("materializer trusted raw evidence score at candidate %d", i)
		}
	}
}

func TestAOQTMaterializerRanksAreDeterministicPermutations(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err != nil {
		t.Fatalf("materialize: %v", err)
	}
	row := readOneAOQTRow(t, cfg.RowJSONLPath)
	for name, ranks := range map[string][]int{"dense": row.AnchorRanks.Dense, "q3": row.AnchorRanks.Q3, "q5": row.AnchorRanks.Q5} {
		seen := map[int]bool{}
		for _, rank := range ranks {
			seen[rank] = true
		}
		if len(seen) != len(row.CandidateDocIDs) {
			t.Fatalf("%s ranks are not a permutation: %v", name, ranks)
		}
	}
	if err := row.Validate(AOQTSidecarCalibrationManifest{
		Schema:               AOQTSidecarManifestSchema,
		Dim:                  AOQTSidecarDim,
		TurboQuantSeed:       AOQTSidecarMaterializerQuantSeed,
		QuantSurfaces:        []AOQTSidecarQuantSurface{{BitWidth: 3, Seed: AOQTSidecarMaterializerQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true}, {BitWidth: 5, Seed: AOQTSidecarMaterializerQuantSeed, ScoreSurface: AOQTSidecarPreparedIPScoreSurface, PreparedQuery: true}},
		QrelsSHA256ByDataset: map[string]string{row.Dataset: row.QrelsSHA256},
		SplitProof:           row.SplitProof,
		SourceArtifactHashes: []string{row.SourceArtifactHash},
		VectorCacheHashes:    []string{hex64("cache")},
		RowCount:             1,
		RowIDSHA256:          aoqtRowIDSHA256([]string{row.RowID}),
		CompatibilityDigest:  row.CompatibilityDigest,
		LegalGates:           aoqtResearchOnlyGates(),
		ObjectiveContract: AOQTSidecarPreparedIPObjectiveConfig{
			TurboQuantSeed:   AOQTSidecarMaterializerQuantSeed,
			NFBoundarySource: "nf_boundary80_120",
		}.ObjectiveContract(row.Weights),
		Topology: mustAOQTTopologyBinding(t),
	}); err != nil {
		t.Fatalf("materialized row validation: %v", err)
	}
}

func TestAOQTMaterializerRejectsExcludedCandidateOutsidePlan(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	data, err := os.ReadFile(cfg.PlanPath)
	if err != nil {
		t.Fatal(err)
	}
	updated := strings.Replace(string(data), `"d080"`, `"outside-plan"`, 1)
	if err := os.WriteFile(cfg.PlanPath, []byte(updated), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "does not match Stage3 plan doc") {
		t.Fatalf("error = %v, want plan candidate exclusion rejection", err)
	}
}

func TestAOQTMaterializerRejectsMissingVector(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	data, err := os.ReadFile(cfg.VectorPaths[0])
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	filtered := lines[:0]
	for _, line := range lines {
		if !strings.Contains(line, `"id":"d080"`) {
			filtered = append(filtered, line)
		}
	}
	if err := os.WriteFile(cfg.VectorPaths[0], []byte(strings.Join(filtered, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "missing doc vector") {
		t.Fatalf("error = %v, want missing vector rejection", err)
	}
}

func TestAOQTMaterializerRejectsQuantSeedMismatch(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	data, err := os.ReadFile(cfg.ScoreEvidencePaths[0])
	if err != nil {
		t.Fatal(err)
	}
	var evidence aoqtMaterializeScoreEvidenceFile
	if err := json.Unmarshal(data, &evidence); err != nil {
		t.Fatal(err)
	}
	evidence.TurboQuant.Seed = 191
	if err := writeJSONFileAOQT(cfg.ScoreEvidencePaths[0], evidence); err != nil {
		t.Fatal(err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "TurboQuant prepared-IP config binding mismatch") {
		t.Fatalf("error = %v, want seed mismatch rejection", err)
	}
}

func TestAOQTMaterializerRejectsQrelsCoverageMismatch(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	provenance := plan["provenance"].(map[string]any)
	provenance["qrels_sha256_by_dataset"] = map[string]any{"nfcorpus": hex64("wrong-qrels")}
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "AOQT qrels sha256 mismatch") {
		t.Fatalf("error = %v, want qrels mismatch rejection", err)
	}
}

func TestAOQTMaterializerRejectsMissingOfficialTestExclusion(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	exclusions["sets"] = exclusions["sets"].([]any)[:2]
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), `missing required qid-only exclusion "official-test"`) {
		t.Fatalf("error = %v, want missing official-test exclusion rejection", err)
	}
}

func TestAOQTMaterializerRejectsPlanLegalScopeClaim(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	legalScope := plan["legal_scope"].(map[string]any)
	legalScope["commercial_use_allowed"] = true
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "legal_scope must be research-only") {
		t.Fatalf("error = %v, want legal scope claim rejection", err)
	}
}

func TestAOQTMaterializerRejectsPlanSelectionPolicyMismatch(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	selectionPolicy := plan["selection_policy"].(map[string]any)
	selectionPolicy["top10_guard_limit"] = 20
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "selection_policy binding mismatch") {
		t.Fatalf("error = %v, want selection policy rejection", err)
	}
}

func TestAOQTMaterializerRejectsPlanTopologyMismatch(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	topology := plan["topology"].(map[string]any)
	topology["angle_count"] = float64(AOQTSidecarAngleCount - 1)
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "topology binding mismatch") {
		t.Fatalf("error = %v, want topology rejection", err)
	}
}

func TestAOQTMaterializerRejectsLoosePlanRowBucket(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	rows := plan["rows"].([]any)
	row := rows[0].(map[string]any)
	row["bucket"] = "stable_candidate_union"
	row["rank_window"] = []any{float64(1), float64(120)}
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "unsupported bucket/bits/window binding") {
		t.Fatalf("error = %v, want loose bucket rejection", err)
	}
}

func TestAOQTMaterializerRejectsDuplicateScoreEvidenceQID(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	data, err := os.ReadFile(cfg.ScoreEvidencePaths[0])
	if err != nil {
		t.Fatal(err)
	}
	var evidence aoqtMaterializeScoreEvidenceFile
	if err := json.Unmarshal(data, &evidence); err != nil {
		t.Fatal(err)
	}
	evidence.Rows = append(evidence.Rows, evidence.Rows[0])
	if err := writeJSONFileAOQT(cfg.ScoreEvidencePaths[0], evidence); err != nil {
		t.Fatal(err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "duplicate score evidence qid") {
		t.Fatalf("error = %v, want duplicate qid rejection", err)
	}
}

func TestAOQTMaterializerRejectsDuplicateScoreEvidenceDoc(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	data, err := os.ReadFile(cfg.ScoreEvidencePaths[0])
	if err != nil {
		t.Fatal(err)
	}
	var evidence aoqtMaterializeScoreEvidenceFile
	if err := json.Unmarshal(data, &evidence); err != nil {
		t.Fatal(err)
	}
	evidence.Rows[0].Docs[1].DocID = evidence.Rows[0].Docs[0].DocID
	if err := writeJSONFileAOQT(cfg.ScoreEvidencePaths[0], evidence); err != nil {
		t.Fatal(err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "duplicate score evidence doc_id") {
		t.Fatalf("error = %v, want duplicate doc rejection", err)
	}
}

func TestAOQTMaterializerBindsRowSourceHashToPlanAndScoreEvidence(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err != nil {
		t.Fatalf("materialize: %v", err)
	}
	row := readOneAOQTRow(t, cfg.RowJSONLPath)
	planData, err := os.ReadFile(cfg.PlanPath)
	if err != nil {
		t.Fatal(err)
	}
	var plan aoqtMaterializePlan
	if err := json.Unmarshal(planData, &plan); err != nil {
		t.Fatal(err)
	}
	planSHA, err := sha256FileAOQT(cfg.PlanPath)
	if err != nil {
		t.Fatal(err)
	}
	q3SHA, err := sha256FileAOQT(cfg.ScoreEvidencePaths[0])
	if err != nil {
		t.Fatal(err)
	}
	if got, want := row.SourceArtifactHash, aoqtMaterializeSourceArtifactHash(planSHA, q3SHA, plan.Rows[0]); got != want {
		t.Fatalf("source_artifact_hash = %s, want plan+score binding %s", got, want)
	}
	if row.SourceArtifactHash == sortedMapValuesAOQT(preflightInputHashesForAOQTTest(t, cfg))[0] {
		t.Fatalf("source_artifact_hash still equals lexicographic raw input hash")
	}
}

func TestAOQTMaterializerIsDeterministic(t *testing.T) {
	cfgA := writeTinyAOQTMaterializerFixture(t)
	cfgB := writeTinyAOQTMaterializerFixture(t)
	cfgB.CreatedAtUTC = cfgA.CreatedAtUTC
	if _, err := MaterializeAOQTSidecarCalibration(cfgA); err != nil {
		t.Fatalf("materialize A: %v", err)
	}
	if _, err := MaterializeAOQTSidecarCalibration(cfgB); err != nil {
		t.Fatalf("materialize B: %v", err)
	}
	for _, pair := range [][2]string{{cfgA.RowJSONLPath, cfgB.RowJSONLPath}, {cfgA.ManifestJSONPath, cfgB.ManifestJSONPath}, {cfgA.PreflightJSONPath, cfgB.PreflightJSONPath}} {
		a, err := os.ReadFile(pair[0])
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(pair[1])
		if err != nil {
			t.Fatal(err)
		}
		if string(a) != string(b) {
			if pair[0] != cfgA.PreflightJSONPath {
				t.Fatalf("%s and %s differ", pair[0], pair[1])
			}
			assertEquivalentAOQTMaterializerPreflights(t, a, b)
		}
	}
}

func assertEquivalentAOQTMaterializerPreflights(t *testing.T, a, b []byte) {
	t.Helper()
	var left AOQTSidecarMaterializePreflight
	var right AOQTSidecarMaterializePreflight
	if err := json.Unmarshal(a, &left); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &right); err != nil {
		t.Fatal(err)
	}
	leftOutputs := left.Outputs
	rightOutputs := right.Outputs
	left.InputSHA256 = nil
	right.InputSHA256 = nil
	left.Outputs = nil
	right.Outputs = nil
	leftData, err := json.Marshal(left)
	if err != nil {
		t.Fatal(err)
	}
	rightData, err := json.Marshal(right)
	if err != nil {
		t.Fatal(err)
	}
	if string(leftData) != string(rightData) {
		t.Fatalf("preflight stable fields differ\nleft: %s\nright: %s", leftData, rightData)
	}
	if strings.Join(sortedStringMapValuesForAOQTTest(mustInputSHA256ForAOQTTest(t, a)), "\n") != strings.Join(sortedStringMapValuesForAOQTTest(mustInputSHA256ForAOQTTest(t, b)), "\n") {
		t.Fatalf("preflight input hashes differ across identical fixtures")
	}
	for _, key := range []string{"rows_jsonl", "manifest_json", "preflight_json"} {
		if leftOutputs[key] == "" || rightOutputs[key] == "" {
			t.Fatalf("preflight outputs missing %q: left=%v right=%v", key, leftOutputs, rightOutputs)
		}
	}
}

func mustInputSHA256ForAOQTTest(t *testing.T, data []byte) map[string]string {
	t.Helper()
	var preflight AOQTSidecarMaterializePreflight
	if err := json.Unmarshal(data, &preflight); err != nil {
		t.Fatal(err)
	}
	return preflight.InputSHA256
}

func sortedStringMapValuesForAOQTTest(values map[string]string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func preflightInputHashesForAOQTTest(t *testing.T, cfg AOQTSidecarMaterializeConfig) map[string]string {
	t.Helper()
	paths := append(append([]string{cfg.PlanPath}, cfg.VectorPaths...), append(cfg.ScoreEvidencePaths, cfg.QrelsPaths...)...)
	out := map[string]string{}
	for _, path := range paths {
		sum, err := sha256FileAOQT(path)
		if err != nil {
			t.Fatal(err)
		}
		out[path] = sum
	}
	return out
}

func readAOQTPlanMap(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	if err := decoder.Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func writeAOQTPlanMap(t *testing.T, path string, plan map[string]any) {
	t.Helper()
	if err := writeJSONFileAOQT(path, plan); err != nil {
		t.Fatal(err)
	}
}

func writeTinyAOQTMaterializerFixture(t *testing.T) AOQTSidecarMaterializeConfig {
	t.Helper()
	dir := t.TempDir()
	space := "eos-d384-anchor-test"
	planPath := filepath.Join(dir, "plan.json")
	vectorsPath := filepath.Join(dir, "vectors.jsonl")
	qrelsPath := filepath.Join(dir, "qrels.jsonl")
	q3Path := filepath.Join(dir, "q3.json")
	q5Path := filepath.Join(dir, "q5.json")
	query := unitVecAtAOQT(0)
	vectorFile, err := os.Create(vectorsPath)
	if err != nil {
		t.Fatal(err)
	}
	writeJSONLineAOQT(t, vectorFile, aoqtMaterializeVectorRecord{Dataset: "nfcorpus", Role: "query", ID: "q1", VectorID: "qv1", EmbeddingSpaceID: space, Vector: query})
	for i := 1; i <= 120; i++ {
		id := fmtDocIDAOQT(i)
		vec := unitVecAtAOQT(i % AOQTSidecarDim)
		if i == 80 {
			vec = unitVecAtAOQT(0)
		}
		writeJSONLineAOQT(t, vectorFile, aoqtMaterializeVectorRecord{Dataset: "nfcorpus", Role: "doc", ID: id, VectorID: "vec-" + id, EmbeddingSpaceID: space, Vector: vec})
	}
	if err := vectorFile.Close(); err != nil {
		t.Fatal(err)
	}
	qrelsFile, err := os.Create(qrelsPath)
	if err != nil {
		t.Fatal(err)
	}
	writeJSONLineAOQT(t, qrelsFile, map[string]any{"dataset": "nfcorpus", "qid": "q1", "doc_id": "d080", "gain": 1})
	if err := qrelsFile.Close(); err != nil {
		t.Fatal(err)
	}
	writeEvidenceAOQT(t, q3Path, 3)
	writeEvidenceAOQT(t, q5Path, 5)
	candidates := make([]string, 0, 61)
	for i := 80; i <= 120; i++ {
		candidates = append(candidates, fmtDocIDAOQT(i))
	}
	qrelsSHA, err := sha256FileAOQT(qrelsPath)
	if err != nil {
		t.Fatal(err)
	}
	plan := map[string]any{
		"schema":                      "eos.aoqt_stage2_calibration_plan.v1",
		"mode":                        "plan_only",
		"actual_cache_generation_ran": false,
		"actual_training_ran":         false,
		"actual_eval_ran":             false,
		"seed":                        AOQTSidecarMaterializerTopologySeed,
		"topology_seed":               AOQTSidecarMaterializerTopologySeed,
		"turboquant":                  map[string]any{"score_mode": "prepared_ip", "seed": AOQTSidecarMaterializerQuantSeed, "required_bits": []int{3, 5}, "top_k": 120},
		"topology":                    map[string]any{"id": "aoqt_givens_v1", "dim": AOQTSidecarDim, "stages": AOQTSidecarStages, "pairs_per_stage": AOQTSidecarPairsPerStage, "angle_count": AOQTSidecarAngleCount, "angle_cap_default": AOQTSidecarDefaultAngleCap, "angle_cap_hard": AOQTSidecarHardMaxAngleCap},
		"legal_scope":                 map[string]any{"train_allowed_for_research": true, "release_train_allowed": false, "commercial_use_allowed": false, "free_open_release_allowed": false, "quality_claim": false},
		"anchor": map[string]any{
			"path":            "anchor.mll",
			"manifest_sha256": hex64("pkg-manifest"),
			"package_sha256":  hex64("anchor"),
		},
		"exclusions": map[string]any{"sets": []map[string]any{
			{"name": "dev4", "path": "dev4.exclusion.json", "qid_count": 4, "manifest_sha256": hex64("dev4")},
			{"name": "reserve4", "path": "reserve4.exclusion.json", "qid_count": 4, "manifest_sha256": hex64("reserve4")},
			{"name": "official-test", "path": "official-test.exclusion.json", "qid_count": 4, "manifest_sha256": hex64("official-test")},
		}},
		"selection_policy": map[string]any{
			"split":                   "train",
			"datasets":                []string{"nfcorpus"},
			"required_score_bits":     []int{3, 5},
			"score_mode":              "prepared_ip",
			"turboquant_seed":         AOQTSidecarMaterializerQuantSeed,
			"top10_guard_limit":       10,
			"nf_boundary_window":      []int{80, 120},
			"official_exclusion_mode": "qid_only",
			"forbidden_splits":        []string{"dev", "reserve", "official", "test", "eval", "heldout"},
		},
		"rows": []map[string]any{{
			"row_id":            "nfcorpus.q3.nf80_120.q1",
			"dataset":           "nfcorpus",
			"bits":              3,
			"bucket":            "nf_boundary80_120_guard",
			"qid":               "q1",
			"rank_window":       []int{80, 120},
			"candidate_doc_ids": candidates,
		}},
		"provenance": map[string]any{"qrels_sha256_by_dataset": map[string]string{"nfcorpus": qrelsSHA}},
	}
	if err := writeJSONFileAOQT(planPath, plan); err != nil {
		t.Fatal(err)
	}
	return AOQTSidecarMaterializeConfig{
		PlanPath:               planPath,
		VectorPaths:            []string{vectorsPath},
		ScoreEvidencePaths:     []string{q3Path, q5Path},
		QrelsPaths:             []string{qrelsPath},
		RowJSONLPath:           filepath.Join(dir, "rows.jsonl"),
		ManifestJSONPath:       filepath.Join(dir, "manifest.json"),
		PreflightJSONPath:      filepath.Join(dir, "preflight.json"),
		AnchorEmbeddingSpaceID: space,
		CreatedAtUTC:           "2026-09-03T00:00:00Z",
		AllowResearchOnly:      true,
	}
}

func writeEvidenceAOQT(t *testing.T, path string, bits int) {
	t.Helper()
	docs := make([]map[string]any, 0, 120)
	for i := 1; i <= 120; i++ {
		docs = append(docs, map[string]any{"doc_id": fmtDocIDAOQT(i), "rank": i, "gain": 0, "score": -999})
	}
	payload := map[string]any{
		"schema":     "eos.aoqt_stage2.score_cache_manifest.v1",
		"dataset":    "nfcorpus",
		"bits":       bits,
		"split":      "train",
		"score_mode": "prepared_ip",
		"top_k":      120,
		"turboquant": map[string]any{"score_mode": "prepared_ip", "bits": bits, "seed": AOQTSidecarMaterializerQuantSeed},
		"rows":       []map[string]any{{"qid": "q1", "docs": docs}},
	}
	if err := writeJSONFileAOQT(path, payload); err != nil {
		t.Fatal(err)
	}
}

func readOneAOQTRow(t *testing.T, path string) AOQTSidecarCalibrationRow {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		t.Fatalf("missing row in %s", path)
	}
	var row AOQTSidecarCalibrationRow
	if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
		t.Fatal(err)
	}
	if scanner.Scan() {
		t.Fatalf("expected one row")
	}
	return row
}

func unitVecAtAOQT(index int) []float32 {
	out := make([]float32, AOQTSidecarDim)
	out[index] = 1
	return out
}

func writeJSONLineAOQT(t *testing.T, file *os.File, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(append(data, '\n')); err != nil {
		t.Fatal(err)
	}
}

func fmtDocIDAOQT(i int) string {
	return "d" + strings.Repeat("0", 3-len(strconv.Itoa(i))) + strconv.Itoa(i)
}

func mustAOQTTopologyBinding(t *testing.T) AOQTSidecarTopologyBinding {
	t.Helper()
	topology, err := NewAOQTGivensIdentityTopology(AOQTSidecarDim, AOQTSidecarStages, AOQTSidecarMaterializerTopologySeed, AOQTSidecarDefaultAngleCap)
	if err != nil {
		t.Fatal(err)
	}
	pairings, err := topology.PairingsSHA256()
	if err != nil {
		t.Fatal(err)
	}
	return AOQTSidecarTopologyBinding{Kind: AOQTTopologyKindGivensV1, Dim: AOQTSidecarDim, Stages: AOQTSidecarStages, PairsPerStage: AOQTSidecarPairsPerStage, AngleCount: AOQTSidecarAngleCount, Seed: AOQTSidecarMaterializerTopologySeed, PairingsSHA256: pairings}
}
