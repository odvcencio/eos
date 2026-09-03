package eosruntime

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"os/exec"
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
		VectorCacheHashes:    []string{hex64AOQTMaterializerTest("cache")},
		RowCount:             1,
		RowIDSHA256:          aoqtRowIDSHA256([]string{row.RowID}),
		CompatibilityDigest:  row.CompatibilityDigest,
		LegalGates:           AOQTSidecarLegalGates{ResearchTrainAllowed: true},
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
	updated := strings.Replace(string(data), `"fiqa-d001"`, `"outside-plan"`, 1)
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
		if !strings.Contains(line, `"id":"fiqa-d001"`) {
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
	provenance["qrels_sha256_by_dataset"] = map[string]any{"nfcorpus": hex64AOQTMaterializerTest("wrong-qrels")}
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "qrels_sha256_by_dataset must be exactly") {
		t.Fatalf("error = %v, want qrels coverage rejection", err)
	}
}

func TestAOQTMaterializerRejectsMissingOfficialTestExclusion(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	sets := exclusions["sets"].([]any)
	filtered := sets[:0]
	for _, item := range sets {
		if item.(map[string]any)["name"] != "official-test" {
			filtered = append(filtered, item)
		}
	}
	exclusions["sets"] = filtered
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), `missing required qid-only exclusion "official-test"`) {
		t.Fatalf("error = %v, want missing official-test exclusion rejection", err)
	}
}

func TestAOQTMaterializerAllowsSameQIDAcrossDifferentExcludedDataset(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixtureWithExclusionMutator(t, func(qids map[string]map[string][]string) {
		qids["dev4"]["fiqa"] = []string{"nfcorpus-q1"}
	})
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err != nil {
		t.Fatalf("materialize with nfcorpus-q1 excluded for different dataset: %v", err)
	}
}

func TestAOQTMaterializerRejectsSameDatasetExcludedQID(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	rewriteStrictExclusionManifestAndPlanAuditAOQT(t, cfg, "dev4", "nfcorpus", []string{"nfcorpus-q1"})
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), `qid "nfcorpus-q1" is excluded for dataset "nfcorpus"`) {
		t.Fatalf("error = %v, want same-dataset qid exclusion rejection", err)
	}
}

func TestAOQTMaterializerRejectsWhitespaceExclusionQIDs(t *testing.T) {
	for name, qid := range map[string]string{
		"ascii-space":   "dev4-fiqa qid",
		"ascii-tab":     "dev4-fiqa\tqid",
		"unicode-space": "dev4-fiqa\u00a0qid",
	} {
		t.Run(name, func(t *testing.T) {
			cfg := writeTinyAOQTMaterializerFixture(t)
			rewriteStrictExclusionManifestAndPlanAuditAOQT(t, cfg, "dev4", "fiqa", []string{qid})
			if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "qids must be non-empty strings without whitespace") {
				t.Fatalf("error = %v, want whitespace qid rejection", err)
			}
		})
	}
}

func TestAOQTMaterializerRejectsMissingExclusionDatasetCoverage(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	sets := exclusions["sets"].([]any)
	official := sets[2].(map[string]any)
	counts := official["qid_count_by_dataset"].(map[string]any)
	delete(counts, "scifact")
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), `must cover exactly fiqa,nfcorpus,scifact`) {
		t.Fatalf("error = %v, want missing dataset-scoped qid coverage rejection", err)
	}
}

func TestAOQTMaterializerRejectsStalePerSetExclusionQIDsByDatasetHash(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	sets := exclusions["sets"].([]any)
	dev4 := sets[0].(map[string]any)
	dev4["qids_by_dataset_sha256"] = hex64AOQTMaterializerTest("stale-per-set-qids-by-dataset")
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), `exclusion manifest "dev4" qids_by_dataset_sha256 mismatch`) {
		t.Fatalf("error = %v, want stale per-set qids_by_dataset hash rejection", err)
	}
}

func TestAOQTMaterializerRejectsStaleAggregateExclusionQIDsByDatasetHash(t *testing.T) {
	cfg := writeTinyAOQTMaterializerFixture(t)
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	exclusions["excluded_qids_by_dataset_sha256"] = hex64AOQTMaterializerTest("stale-aggregate-qids-by-dataset")
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
	if _, err := MaterializeAOQTSidecarCalibration(cfg); err == nil || !strings.Contains(err.Error(), "aggregate excluded_qids_by_dataset_sha256 mismatch") {
		t.Fatalf("error = %v, want stale aggregate qids_by_dataset hash rejection", err)
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
	dir := filepath.Dir(cfgA.RowJSONLPath)
	cfgB := cfgA
	cfgB.RowJSONLPath = filepath.Join(dir, "rows-b.jsonl")
	cfgB.ManifestJSONPath = filepath.Join(dir, "manifest-b.json")
	cfgB.PreflightJSONPath = filepath.Join(dir, "preflight-b.json")
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
	return writeTinyAOQTMaterializerFixtureWithExclusionMutator(t, nil)
}

func writeTinyAOQTMaterializerFixtureWithExclusionMutator(t *testing.T, mutate func(map[string]map[string][]string)) AOQTSidecarMaterializeConfig {
	t.Helper()
	dir := t.TempDir()
	space := "eos-d384-anchor-test"
	planPath := filepath.Join(dir, "plan.json")
	vectorsPath := filepath.Join(dir, "vectors.jsonl")
	anchorPath := filepath.Join(dir, "anchor.json")
	exclusionDir := filepath.Join(dir, "exclusions")
	vectorManifestDir := filepath.Join(dir, "vector-manifests")
	scoreDir := filepath.Join(dir, "scores")
	qrelsDir := filepath.Join(dir, "qrels")
	for _, path := range []string{exclusionDir, vectorManifestDir, scoreDir, qrelsDir} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	anchor := map[string]any{
		"schema":         "eos.aoqt_stage2.anchor_manifest.v1",
		"package_path":   "anchor.mll",
		"package_sha256": hex64AOQTMaterializerTest("anchor"),
		"embedding_dim":  AOQTSidecarDim,
		"topology":       map[string]any{"id": "aoqt_givens_v1", "dim": AOQTSidecarDim, "stages": AOQTSidecarStages, "pairs_per_stage": AOQTSidecarPairsPerStage, "angle_count": AOQTSidecarAngleCount, "angle_cap_default": AOQTSidecarDefaultAngleCap, "angle_cap_hard": AOQTSidecarHardMaxAngleCap},
		"legal_scope":    map[string]any{"train_allowed_for_research": true, "release_train_allowed": false, "commercial_use_allowed": false, "free_open_release_allowed": false, "quality_claim": false},
	}
	if err := writeJSONFileAOQT(anchorPath, anchor); err != nil {
		t.Fatal(err)
	}
	anchorManifestSHA, err := sha256FileAOQT(anchorPath)
	if err != nil {
		t.Fatal(err)
	}
	vectorFile, err := os.Create(vectorsPath)
	if err != nil {
		t.Fatal(err)
	}
	var qrelsPaths []string
	var vectorManifestPaths []string
	var scorePaths []string
	for datasetIndex, dataset := range requiredAOQTMaterializeExclusionDatasets() {
		queryIDs := []string{dataset + "-q1", dataset + "-q2"}
		docIDs := make([]string, 0, 120)
		qrelsPath := filepath.Join(qrelsDir, dataset+".qrels.jsonl")
		qrelsFile, err := os.Create(qrelsPath)
		if err != nil {
			t.Fatal(err)
		}
		for _, qid := range queryIDs {
			writeJSONLineAOQT(t, vectorFile, aoqtMaterializeVectorRecord{Dataset: dataset, Role: "query", ID: qid, VectorID: "vec-" + qid, EmbeddingSpaceID: space, Vector: unitVecAtAOQT(datasetIndex)})
			writeJSONLineAOQT(t, qrelsFile, map[string]any{"dataset": dataset, "qid": qid, "doc_id": dataset + "-d001", "gain": 1})
			if dataset == "nfcorpus" {
				writeJSONLineAOQT(t, qrelsFile, map[string]any{"dataset": dataset, "qid": qid, "doc_id": dataset + "-d080", "gain": 1})
			}
		}
		if err := qrelsFile.Close(); err != nil {
			t.Fatal(err)
		}
		qrelsSHA, err := sha256FileAOQT(qrelsPath)
		if err != nil {
			t.Fatal(err)
		}
		qrelsPaths = append(qrelsPaths, qrelsPath)
		for i := 1; i <= 120; i++ {
			id := dataset + "-" + fmtDocIDAOQT(i)
			docIDs = append(docIDs, id)
			vec := unitVecAtAOQT((datasetIndex + i) % AOQTSidecarDim)
			if i == 1 || (dataset == "nfcorpus" && i == 80) {
				vec = unitVecAtAOQT(datasetIndex)
			}
			writeJSONLineAOQT(t, vectorFile, aoqtMaterializeVectorRecord{Dataset: dataset, Role: "doc", ID: id, VectorID: "vec-" + id, EmbeddingSpaceID: space, Vector: vec})
		}
		for _, role := range []string{"query", "doc"} {
			ids := queryIDs
			component := "query_vector"
			if role == "doc" {
				ids = docIDs
				component = "doc_vector"
			}
			manifestPath := filepath.Join(vectorManifestDir, dataset+"."+role+".json")
			payload := aoqtBuilderCacheManifestAOQT(dataset, role, anchorManifestSHA, anchor, qrelsSHA, hex64AOQTMaterializerTest(dataset+"-"+component+"-source"))
			payload["ids"] = ids
			payload["cache_sha256"] = hex64AOQTMaterializerTest(dataset + "-" + role + "-cache")
			if err := writeJSONFileAOQT(manifestPath, payload); err != nil {
				t.Fatal(err)
			}
			vectorManifestPaths = append(vectorManifestPaths, manifestPath)
		}
		for _, bits := range []int{3, 5} {
			scorePath := filepath.Join(scoreDir, dataset+".q"+strconv.Itoa(bits)+".json")
			writeEvidenceAOQT(t, scorePath, dataset, bits, queryIDs, docIDs, anchorManifestSHA, anchor, qrelsSHA, vectorManifestDir)
			scorePaths = append(scorePaths, scorePath)
		}
	}
	if err := vectorFile.Close(); err != nil {
		t.Fatal(err)
	}
	exclusionQIDs := map[string]map[string][]string{}
	for _, name := range []string{"dev4", "reserve4", "official-test"} {
		exclusionQIDs[name] = map[string][]string{}
		for _, dataset := range requiredAOQTMaterializeExclusionDatasets() {
			exclusionQIDs[name][dataset] = []string{name + "-" + dataset + "-qid"}
		}
	}
	if mutate != nil {
		mutate(exclusionQIDs)
	}
	var exclusionPaths []string
	for _, name := range []string{"dev4", "reserve4", "official-test"} {
		path := filepath.Join(exclusionDir, name+".json")
		if err := writeJSONFileAOQT(path, map[string]any{
			"schema":          "eos.aoqt_stage2.exclusion_qids.v1",
			"name":            name,
			"qids_by_dataset": exclusionQIDs[name],
			"source_sha256":   hex64AOQTMaterializerTest(name + "-source"),
		}); err != nil {
			t.Fatal(err)
		}
		exclusionPaths = append(exclusionPaths, path)
	}
	args := []string{"scripts/build_aoqt_stage2_calibration.py", "--anchor-manifest", anchorPath, "--output-plan", planPath}
	for _, path := range exclusionPaths {
		args = append(args, "--exclusion-qids", path)
	}
	for _, path := range vectorManifestPaths {
		args = append(args, "--vector-cache", path)
	}
	for _, path := range scorePaths {
		args = append(args, "--score-cache", path)
	}
	cmd := exec.Command("python3", args...)
	cmd.Dir = repoRootAOQTMaterializerTest(t)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build AOQT calibration plan: %v\n%s", err, out)
	}
	return AOQTSidecarMaterializeConfig{
		PlanPath:               planPath,
		ExclusionQIDPaths:      exclusionPaths,
		VectorPaths:            []string{vectorsPath},
		ScoreEvidencePaths:     scorePaths,
		QrelsPaths:             qrelsPaths,
		RowJSONLPath:           filepath.Join(dir, "rows.jsonl"),
		ManifestJSONPath:       filepath.Join(dir, "manifest.json"),
		PreflightJSONPath:      filepath.Join(dir, "preflight.json"),
		AnchorEmbeddingSpaceID: space,
		CreatedAtUTC:           "2026-09-03T00:00:00Z",
		AllowResearchOnly:      true,
	}
}

func repoRootAOQTMaterializerTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		next := filepath.Dir(dir)
		if next == dir {
			t.Fatal("repo root not found")
		}
		dir = next
	}
}

func hex64AOQTMaterializerTest(label string) string {
	sum := sha256.Sum256([]byte(label))
	return hex.EncodeToString(sum[:])
}

func aoqtBuilderCacheManifestAOQT(dataset, role, anchorManifestSHA string, anchor map[string]any, qrelsSHA, sourceSHA string) map[string]any {
	return map[string]any{
		"schema":        "eos.aoqt_stage2.vector_cache_manifest.v1",
		"dataset":       dataset,
		"split":         "train",
		"role":          role,
		"anchor":        map[string]any{"package_sha256": anchor["package_sha256"], "manifest_sha256": anchorManifestSHA},
		"topology":      map[string]any{"id": "aoqt_givens_v1", "sha256": mustSHA256JSONAOQT(anchor["topology"])},
		"legal_scope":   anchor["legal_scope"],
		"source_sha256": sourceSHA,
		"qrels_sha256":  qrelsSHA,
		"dataset_source_provenance": map[string]string{
			role + "_vector": sourceSHA,
		},
		"dataset_source_provenance_sha256": mustSHA256JSONAOQT(map[string]string{role + "_vector": sourceSHA}),
	}
}

func writeEvidenceAOQT(t *testing.T, path, dataset string, bits int, queryIDs, docIDs []string, anchorManifestSHA string, anchor map[string]any, qrelsSHA, vectorManifestDir string) {
	t.Helper()
	docs := make([]map[string]any, 0, 120)
	for i, docID := range docIDs {
		rank := i + 1
		gain := 0
		if rank == 1 || (dataset == "nfcorpus" && rank == 80) {
			gain = 1
		}
		docs = append(docs, map[string]any{"doc_id": docID, "rank": rank, "gain": gain})
	}
	queryManifestSHA, err := sha256FileAOQT(filepath.Join(vectorManifestDir, dataset+".query.json"))
	if err != nil {
		t.Fatal(err)
	}
	docManifestSHA, err := sha256FileAOQT(filepath.Join(vectorManifestDir, dataset+".doc.json"))
	if err != nil {
		t.Fatal(err)
	}
	querySource := hex64AOQTMaterializerTest(dataset + "-query_vector-source")
	docSource := hex64AOQTMaterializerTest(dataset + "-doc_vector-source")
	scoreSource := hex64AOQTMaterializerTest(dataset + "-q" + strconv.Itoa(bits) + "_score-source")
	datasetSource := map[string]string{
		"query_vector":                      querySource,
		"doc_vector":                        docSource,
		"q" + strconv.Itoa(bits) + "_score": scoreSource,
	}
	rows := make([]map[string]any, 0, len(queryIDs))
	for _, qid := range queryIDs {
		rows = append(rows, map[string]any{"qid": qid, "docs": docs})
	}
	payload := map[string]any{
		"schema":                           "eos.aoqt_stage2.score_cache_manifest.v1",
		"dataset":                          dataset,
		"bits":                             bits,
		"split":                            "train",
		"score_mode":                       "prepared_ip",
		"top_k":                            120,
		"turboquant":                       map[string]any{"score_mode": "prepared_ip", "bits": bits, "seed": AOQTSidecarMaterializerQuantSeed},
		"anchor":                           map[string]any{"package_sha256": anchor["package_sha256"], "manifest_sha256": anchorManifestSHA},
		"topology":                         map[string]any{"id": "aoqt_givens_v1", "sha256": mustSHA256JSONAOQT(anchor["topology"])},
		"legal_scope":                      anchor["legal_scope"],
		"source_sha256":                    scoreSource,
		"qrels_sha256":                     qrelsSHA,
		"dataset_source_provenance":        datasetSource,
		"dataset_source_provenance_sha256": mustSHA256JSONAOQT(datasetSource),
		"vector_cache": map[string]any{
			"query_manifest_sha256": queryManifestSHA,
			"query_cache_sha256":    hex64AOQTMaterializerTest(dataset + "-query-cache"),
			"doc_manifest_sha256":   docManifestSHA,
			"doc_cache_sha256":      hex64AOQTMaterializerTest(dataset + "-doc-cache"),
		},
		"rows": rows,
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
	return row
}

func rewriteStrictExclusionManifestAndPlanAuditAOQT(t *testing.T, cfg AOQTSidecarMaterializeConfig, name, dataset string, qids []string) {
	t.Helper()
	var targetPath string
	for _, path := range cfg.ExclusionQIDPaths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var manifest map[string]any
		if err := json.Unmarshal(data, &manifest); err != nil {
			t.Fatal(err)
		}
		if manifest["name"] == name {
			targetPath = path
			qidsByDataset := manifest["qids_by_dataset"].(map[string]any)
			qidsByDataset[dataset] = stringsToAnyAOQTMaterializerTest(qids)
			if err := writeJSONFileAOQT(path, manifest); err != nil {
				t.Fatal(err)
			}
			break
		}
	}
	if targetPath == "" {
		t.Fatalf("missing exclusion manifest %q", name)
	}
	manifestSHA, err := sha256FileAOQT(targetPath)
	if err != nil {
		t.Fatal(err)
	}
	plan := readAOQTPlanMap(t, cfg.PlanPath)
	exclusions := plan["exclusions"].(map[string]any)
	sets := exclusions["sets"].([]any)
	for _, item := range sets {
		set := item.(map[string]any)
		if set["name"] != name {
			continue
		}
		qidsByDataset, counts := readStrictExclusionQIDsByDatasetAOQT(t, targetPath)
		set["manifest_sha256"] = manifestSHA
		set["qid_count_by_dataset"] = intMapToAnyAOQTMaterializerTest(counts)
		set["qids_by_dataset_sha256"] = mustSHA256JSONAOQT(qidsByDataset)
	}
	aggregate := map[string]map[string]bool{}
	for _, path := range cfg.ExclusionQIDPaths {
		qidsByDataset, _ := readStrictExclusionQIDsByDatasetAOQT(t, path)
		for dataset, qids := range qidsByDataset {
			if aggregate[dataset] == nil {
				aggregate[dataset] = map[string]bool{}
			}
			for _, qid := range qids {
				aggregate[dataset][qid] = true
			}
		}
	}
	aggregateQIDs := boolExclusionMapToSortedQIDsAOQT(aggregate)
	exclusions["excluded_qid_count_by_dataset"] = intMapToAnyAOQTMaterializerTest(countAOQTMaterializeQIDsByDataset(aggregateQIDs))
	exclusions["excluded_qids_by_dataset_sha256"] = mustSHA256JSONAOQT(aggregateQIDs)
	writeAOQTPlanMap(t, cfg.PlanPath, plan)
}

func readStrictExclusionQIDsByDatasetAOQT(t *testing.T, path string) (map[string][]string, map[string]int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		QIDsByDataset map[string][]string `json:"qids_by_dataset"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	qidsByDataset := canonicalAOQTMaterializeQIDsByDataset(manifest.QIDsByDataset)
	return qidsByDataset, countAOQTMaterializeQIDsByDataset(qidsByDataset)
}

func stringsToAnyAOQTMaterializerTest(values []string) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	return out
}

func intMapToAnyAOQTMaterializerTest(values map[string]int) map[string]any {
	out := map[string]any{}
	for key, value := range values {
		out[key] = value
	}
	return out
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
