package eosruntime

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/turboquant"
)

func TestTurboQuantGatePerQueryEvidenceIncludesDenseTopKAndBinding(t *testing.T) {
	dense := make([]retrievalScoredDoc, 120)
	compact := make([]retrievalScoredDoc, 120)
	for i := range dense {
		dense[i] = retrievalScoredDoc{ID: "dense-" + strconv.Itoa(i), Score: float32(120 - i)}
		compact[i] = retrievalScoredDoc{ID: "compact-" + strconv.Itoa(i), Score: float32(120 - i)}
	}
	rels := map[string]float64{"dense-0": 1, "compact-0": 1}
	binding := json.RawMessage(`{"schema":"eos.aoqt.native_eval_gate_binding.v1","nonce":"test"}`)
	row := buildTurboQuantPerQueryRowWithEvidence(
		"fiqa", "q1", "turboquant_ip_b3", 3, 0, 0, "", "turboquant_ip_prepared", 5581486560434873699,
		compact, turboQuantRetrievalScoreStats{}, rels, nil, nil, nil, nil, dense, binding,
	)
	if len(row.TopK) != 120 || len(row.DenseTopK) != 120 {
		t.Fatalf("top-k lengths = (%d, %d), want (120, 120)", len(row.TopK), len(row.DenseTopK))
	}
	if row.DenseTopK[0].Rank != 1 || row.DenseTopK[119].Rank != 120 || row.DenseTopK[119].DocID != "dense-119" {
		t.Fatalf("dense rank evidence is not complete/ordered: first=%+v last=%+v", row.DenseTopK[0], row.DenseTopK[119])
	}
	if string(row.GateBinding) != string(binding) {
		t.Fatalf("gate binding = %s, want %s", row.GateBinding, binding)
	}
	var encoded map[string]any
	data, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := encoded["dense_top_k"]; !ok {
		t.Fatal("marshaled per-query row omitted dense_top_k")
	}
	if _, ok := encoded["gate_binding"]; !ok {
		t.Fatal("marshaled per-query row omitted gate_binding")
	}
}

func TestTurboQuantMetricsOmitResearchOnlyAOQTForOrdinaryJSON(t *testing.T) {
	data, err := json.Marshal(TurboQuantRetrievalEvalMetrics{
		Schema:  TurboQuantRetrievalEvalMetricsSchema,
		Dataset: "ordinary",
		Config: TurboQuantRetrievalEvalConfigMetrics{
			BatchSize:     64,
			TopK:          100,
			Bits:          []int{8},
			QuantizerSeed: 17,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatal(err)
	}
	var config map[string]json.RawMessage
	if err := json.Unmarshal(payload["config"], &config); err != nil {
		t.Fatal(err)
	}
	if _, ok := config["allow_research_only_aoqt"]; ok {
		t.Fatalf("ordinary metrics emitted allow_research_only_aoqt=false: %s", data)
	}
}

func TestTurboQuantMetricsEmitResearchOnlyAOQTForGateJSON(t *testing.T) {
	for _, allow := range []bool{false, true} {
		data, err := json.Marshal(TurboQuantRetrievalEvalMetrics{
			Schema:  TurboQuantRetrievalEvalMetricsSchema,
			Dataset: "gate",
			Config: TurboQuantRetrievalEvalConfigMetrics{
				BatchSize:             64,
				TopK:                  120,
				Bits:                  []int{3, 5},
				QuantizerSeed:         5581486560434873699,
				AllowResearchOnlyAOQT: allow,
				EmitResearchOnlyAOQT:  true,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(data, &payload); err != nil {
			t.Fatal(err)
		}
		var config map[string]json.RawMessage
		if err := json.Unmarshal(payload["config"], &config); err != nil {
			t.Fatal(err)
		}
		var got bool
		if err := json.Unmarshal(config["allow_research_only_aoqt"], &got); err != nil {
			t.Fatalf("gate metrics omitted allow_research_only_aoqt for allow=%v: %s", allow, data)
		}
		if got != allow {
			t.Fatalf("allow_research_only_aoqt = %v, want %v", got, allow)
		}
	}
}

func TestCurrentTurboQuantRuntimeArgvRejectsProgramOnly(t *testing.T) {
	oldArgs := os.Args
	t.Cleanup(func() { os.Args = oldArgs })
	os.Args = []string{"/tmp/eos"}

	argv, err := currentTurboQuantRuntimeArgv()
	if err == nil || !strings.Contains(err.Error(), "cannot resolve process argv") {
		t.Fatalf("argv=%v err=%v, want deterministic argv error", argv, err)
	}
}

func TestTurboQuantGateQualityUsesEstablishedLinearGradedGain(t *testing.T) {
	scores := []retrievalScoredDoc{{ID: "relevance-1"}, {ID: "relevance-2"}}
	rels := map[string]float64{"relevance-2": 2, "relevance-1": 1}
	quality := turboQuantGateQualityForQuery(scores, rels)
	ordinary := retrievalQualityForQuery(scores, rels)
	if quality != ordinary {
		t.Fatalf("gate quality = %#v, ordinary quality = %#v; gate changed the established metric", quality, ordinary)
	}
	want := (1.0 + 2.0/math.Log2(3)) / (2.0 + 1.0/math.Log2(3))
	if quality.NDCGAt10 != want {
		t.Fatalf("graded nDCG@10 = %v, want %v", quality.NDCGAt10, want)
	}
	if quality.RecallAt100 != 1 {
		t.Fatalf("graded recall@100 = %v, want 1", quality.RecallAt100)
	}
}

func TestTurboQuantRuntimeQrelsIdentitySeparatesQueriesAndPairs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qrels.tsv")
	data := "query-id\titeration\tcorpus-id\tscore\n" +
		"q1\t0\td1\t2\n" +
		"q1\t0\td2\t0\n" +
		"q2\t0\td3\t1\n"
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	identity, err := readTurboQuantRuntimeQrelsIdentity(path)
	if err != nil {
		t.Fatalf("read runtime qrels identity: %v", err)
	}
	if identity.QueryCount != 2 {
		t.Fatalf("query count = %d, want 2", identity.QueryCount)
	}
	if identity.QrelsPairCount != 3 {
		t.Fatalf("qrels pair count = %d, want 3", identity.QrelsPairCount)
	}
	if identity.RelevantPairCnt != 2 {
		t.Fatalf("relevant pair count = %d, want 2", identity.RelevantPairCnt)
	}
	if got := len(identity.Rels["q1"]); got != 2 {
		t.Fatalf("q1 judgment count = %d, want 2", got)
	}
	if got := identity.QIDs; len(got) != 2 || got[0] != "q1" || got[1] != "q2" {
		t.Fatalf("qids = %#v, want [q1 q2]", got)
	}
}

func TestTurboQuantRuntimeQrelsMergeKeepsPositiveJudgmentsOnly(t *testing.T) {
	dst := retrievalQrels{
		"q1": {"d-positive": 2},
	}
	src := retrievalQrels{
		"q1": {"d-positive": 2, "d-zero": 0},
		"q2": {"d-zero-only": 0},
		"q3": {"d-new-positive": 1, "d-new-zero": 0},
	}
	mergeTurboQuantRuntimeQrels(dst, src)

	if got := dst["q1"]["d-positive"]; got != 2 {
		t.Fatalf("q1 positive relevance = %v, want 2", got)
	}
	if _, ok := dst["q1"]["d-zero"]; ok {
		t.Fatalf("zero-relevance q1 judgment was merged: %+v", dst["q1"])
	}
	if _, ok := dst["q2"]; ok {
		t.Fatalf("zero-only qid was merged: %+v", dst["q2"])
	}
	if got := dst["q3"]["d-new-positive"]; got != 1 {
		t.Fatalf("q3 positive relevance = %v, want 1", got)
	}
	if _, ok := dst["q3"]["d-new-zero"]; ok {
		t.Fatalf("zero-relevance q3 judgment was merged: %+v", dst["q3"])
	}
}

func TestTurboQuantRuntimeBoundaryEvidenceBindsBothRankings(t *testing.T) {
	compact := make([]retrievalScoredDoc, 120)
	dense := make([]retrievalScoredDoc, 120)
	for i := range compact {
		compact[i] = retrievalScoredDoc{ID: "compact-" + strconv.Itoa(i), Score: float32(120 - i)}
		dense[i] = retrievalScoredDoc{ID: "dense-" + strconv.Itoa(i), Score: float32(120 - i)}
	}
	info := &turboQuantRuntimeBindingInfo{
		CandidateCount: 123,
		BoundaryQIDs:   map[string]struct{}{"q-boundary": {}},
	}
	evidence := boundaryEvidenceForQuery(info, "q-boundary", compact, dense)
	if evidence == nil {
		t.Fatal("boundary evidence is nil")
	}
	if got := evidence["candidate_count"]; got != 123 {
		t.Fatalf("candidate count = %#v, want 123", got)
	}
	if got := evidence["exit_rank"]; got != 120 {
		t.Fatalf("exit rank = %#v, want 120", got)
	}
	if got := evidence["substitution_count"]; got != 0 {
		t.Fatalf("substitution count = %#v, want 0", got)
	}
	window, ok := evidence["window"].([]map[string]any)
	if !ok || len(window) != 2 || window[0]["rank"] != 80 || window[1]["rank"] != 120 {
		t.Fatalf("boundary window = %#v, want ranks 80 and 120", evidence["window"])
	}
	if got := evidence["compact_top_k_sha256"]; got != turboQuantRankingFingerprint(retrievalTopDocsForEvidence(compact)) {
		t.Fatalf("compact fingerprint = %#v, does not bind compact ranking", got)
	}
	if got := evidence["dense_top_k_sha256"]; got != turboQuantRankingFingerprint(retrievalTopDocsForEvidence(dense)) {
		t.Fatalf("dense fingerprint = %#v, does not bind dense ranking", got)
	}
}

func TestWriteTurboQuantGatePerQueryRowsEmitsCompleteDenseEvidence(t *testing.T) {
	const dim = 2
	q := turboquant.NewIPWithSeed(dim, 3, 1)
	denseDocs := make([]retrievalVectorRecord, 120)
	qdocs := make([]turboQuantRetrievalDoc, 120)
	for i := range denseDocs {
		vector := []float32{float32(i + 1), 1}
		denseDocs[i] = retrievalVectorRecord{ID: "d" + strconv.Itoa(i), Vector: vector}
		qdocs[i] = turboQuantRetrievalDoc{ID: denseDocs[i].ID, Vector: q.Quantize(vector)}
	}
	path := t.TempDir() + "/rows.jsonl"
	binding := json.RawMessage(`{"schema":"eos.aoqt.native_eval_gate_binding.v1","nonce":"test"}`)
	err := writeTurboQuantRetrievalPerQueryRowsWithGateBinding(
		context.Background(), "fiqa", path, q, nil, 3, 120, 120, 1, nil, "", 0,
		denseDocs, []retrievalVectorRecord{{ID: "q1", Vector: []float32{1, 0}}}, qdocs, nil,
		retrievalQrels{"q1": {"d0": 1}}, binding,
	)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var row TurboQuantRetrievalPerQueryRow
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &row); err != nil {
		t.Fatal(err)
	}
	if len(row.TopK) != 120 || len(row.DenseTopK) != 120 {
		t.Fatalf("written top-k lengths = (%d, %d), want (120, 120)", len(row.TopK), len(row.DenseTopK))
	}
	if row.DenseTopK[0].Rank != 1 || row.DenseTopK[119].Rank != 120 {
		t.Fatalf("written dense ranking is not complete/ordered")
	}
	if string(row.GateBinding) != string(binding) {
		t.Fatalf("written gate binding = %s, want %s", row.GateBinding, binding)
	}
}

func TestTurboQuantGatePerQueryAggregateMatchesDirectQualityPass(t *testing.T) {
	const dim = 2
	q := turboquant.NewIPWithSeed(dim, 3, 1)
	denseDocs := make([]retrievalVectorRecord, 120)
	qdocs := make([]turboQuantRetrievalDoc, 120)
	for i := range denseDocs {
		vector := []float32{float32(i + 1), 1}
		denseDocs[i] = retrievalVectorRecord{ID: "d" + strconv.Itoa(i), Vector: vector}
		qdocs[i] = turboQuantRetrievalDoc{ID: denseDocs[i].ID, Vector: q.Quantize(vector)}
	}
	queries := []retrievalVectorRecord{{ID: "q1", Vector: []float32{1, 0}}}
	qrels := retrievalQrels{"q1": {"d0": 2, "d1": 1}}
	path := filepath.Join(t.TempDir(), "rows.jsonl")
	aggregate := &turboQuantPerQueryAggregate{}
	if err := writeTurboQuantRetrievalPerQueryRowsWithBindingsAndAggregate(
		context.Background(), "nfcorpus", path, q, nil, 3, 120, 120, 1, nil, "", 0,
		denseDocs, queries, qdocs, nil, qrels, nil, json.RawMessage(`{"schema":"eos.aoqt.native_eval_runtime_binding.v1"}`), nil, aggregate,
	); err != nil {
		t.Fatal(err)
	}
	expectedQuality, expectedQueries, _, _, _, _, expectedStats := computeTurboQuantRetrievalQuality(context.Background(), q, queries, qdocs, qrels, 120)
	if aggregate.Quality != expectedQuality {
		t.Fatalf("gate aggregate quality = %#v, ordinary quality = %#v", aggregate.Quality, expectedQuality)
	}
	if aggregate.EvaluatedQueries != expectedQueries {
		t.Fatalf("gate aggregate query count = %d, ordinary query count = %d", aggregate.EvaluatedQueries, expectedQueries)
	}
	if aggregate.ScoreStats != expectedStats {
		t.Fatalf("gate aggregate score stats = %#v, ordinary score stats = %#v", aggregate.ScoreStats, expectedStats)
	}
	if len(aggregate.Latencies) != expectedQueries || aggregate.Duration <= 0 {
		t.Fatalf("gate aggregate latency evidence = count %d duration %s, want count %d and positive duration", len(aggregate.Latencies), aggregate.Duration, expectedQueries)
	}
}

func TestLoadTurboQuantGateBindingRejectsPartialEnvironment(t *testing.T) {
	t.Setenv(turboQuantGateBindingJSONEnv, "{}")
	t.Setenv(turboQuantGateBindingSHAEnv, "")
	t.Setenv(turboQuantGateNonceEnv, "nonce")
	t.Setenv(turboQuantFrozenManifestSHAEnv, "frozen")
	if _, err := LoadTurboQuantGateBindingFromEnvironment(); err == nil {
		t.Fatal("partial AOQT gate binding environment unexpectedly succeeded")
	}
}

func TestRequireTurboQuantGateBindingKeysSortsDiagnostics(t *testing.T) {
	fields := map[string]json.RawMessage{
		"zzz_extra": []byte("null"),
		"aaa_extra": []byte("null"),
	}
	for _, key := range turboQuantGateBindingKeys {
		switch key {
		case "schema", "nonce":
			continue
		default:
			fields[key] = []byte("null")
		}
	}
	err := requireTurboQuantGateBindingKeys(fields)
	if err == nil {
		t.Fatal("key-set mismatch unexpectedly succeeded")
	}
	wantMissing := "missing=[nonce schema]"
	wantExtra := "extra=[aaa_extra zzz_extra]"
	if !strings.Contains(err.Error(), wantMissing) || !strings.Contains(err.Error(), wantExtra) {
		t.Fatalf("diagnostic = %q, want sorted %s and %s", err.Error(), wantMissing, wantExtra)
	}
}

func TestValidateTurboQuantGateBindingSyntheticIdentity(t *testing.T) {
	root := t.TempDir()
	datasetDir := filepath.Join(root, "fiqa")
	if err := os.MkdirAll(datasetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	artifactPath := filepath.Join(root, "model.mll")
	packageManifestPath := ResolvePackageManifestPath(artifactPath)
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(datasetDir, "qrels.tsv")
	datasetManifestPath := filepath.Join(datasetDir, "manifest.json")
	for path, data := range map[string][]byte{
		artifactPath:        []byte("artifact"),
		packageManifestPath: []byte("package-manifest"),
		corpusPath:          []byte("corpus"),
		queriesPath:         []byte("queries"),
		qrelsPath:           []byte("qrels"),
		datasetManifestPath: []byte("dataset-manifest"),
	} {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	outputs := map[string]string{
		"metrics":     filepath.Join(root, "metrics.json"),
		"metrics_tsv": filepath.Join(root, "metrics.tsv"),
		"per_query":   filepath.Join(root, "per-query.jsonl"),
	}
	cfg := RetrievalEvalConfig{
		DatasetName:                "fiqa",
		ArtifactPath:               artifactPath,
		CorpusPath:                 corpusPath,
		QueriesPath:                queriesPath,
		QrelsPath:                  qrelsPath,
		BatchSize:                  64,
		TopK:                       120,
		PerQueryTopK:               120,
		QuantizerSeed:              5581486560434873699,
		GateBindingMetricsJSONPath: outputs["metrics"],
		GateBindingMetricsTSVPath:  outputs["metrics_tsv"],
		PerQueryJSONLPath:          outputs["per_query"],
	}
	bits := []int{3, 5}
	argv := []string{"/bin/eos", "eval-retrieval-turboquant", artifactPath, datasetDir}
	hash := func(value string) string { return gateBindingSHA256(value) }
	fields := map[string]any{
		"schema":                            TurboQuantGateBindingSchema,
		"gate_id":                           "synthetic-gate",
		"role":                              "anchor",
		"domain":                            "fiqa",
		"nonce":                             "0123456789abcdef0123456789abcdef",
		"frozen_manifest_sha256":            hash("frozen"),
		"frozen_manifest_digest":            hash("frozen-digest"),
		"package_sha256":                    "",
		"package_manifest_sha256":           "",
		"sibling_rollup_sha256":             hash("siblings"),
		"sidecar_sha256":                    nil,
		"embedding_space_id":                hash("embedding-space"),
		"source_manifest_sha256":            hash("source-manifest"),
		"gate_script_sha256":                hash("gate-script"),
		"binary_sha256":                     hash("binary"),
		"argv":                              argv,
		"argv_sha256":                       "",
		"cwd":                               root,
		"dataset_id":                        "fiqa",
		"dataset_manifest_sha256":           "",
		"corpus_sha256":                     "",
		"queries_sha256":                    "",
		"qrels_sha256":                      "",
		"compatibility_digest":              hash("compatibility"),
		"workload_sha256":                   hash("workload"),
		"approved_workload_sha256":          hash("approved-workload"),
		"workload_qid_set_sha256_by_domain": map[string]string{"fiqa": hash("qids")},
		"workload_query_count_by_domain":    map[string]int{"fiqa": 1},
		"workload_qrels_sha256_by_domain":   map[string]string{"fiqa": hash("qrels-by-domain")},
		"config":                            map[string]any{"dimension": 384, "bits": bits, "seed": cfg.QuantizerSeed, "top_k": 120, "batch_size": 64, "max_docs": 0, "max_queries": 0, "per_query_top_k": 120, "rerank_overfetch": []int{}, "package_mode": "native_mll_sibling", "score_mode": "turboquant_ip_prepared", "split": "test", "allow_research_only_aoqt": false},
		"outputs":                           outputs,
		"nfcorpus_boundary_qids_sha256":     hash("boundary-qids"),
		"nfcorpus_boundary_rank_window":     []int{80, 120},
	}
	packageSHA, err := sha256FileHex(artifactPath)
	if err != nil {
		t.Fatal(err)
	}
	packageManifestSHA, err := sha256FileHex(packageManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	datasetManifestSHA, err := sha256FileHex(datasetManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	corpusSHA, err := sha256FileHex(corpusPath)
	if err != nil {
		t.Fatal(err)
	}
	queriesSHA, err := sha256FileHex(queriesPath)
	if err != nil {
		t.Fatal(err)
	}
	qrelsSHA, err := sha256FileHex(qrelsPath)
	if err != nil {
		t.Fatal(err)
	}
	fields["package_sha256"] = packageSHA
	fields["package_manifest_sha256"] = packageManifestSHA
	fields["dataset_manifest_sha256"] = datasetManifestSHA
	fields["corpus_sha256"] = corpusSHA
	fields["queries_sha256"] = queriesSHA
	fields["qrels_sha256"] = qrelsSHA
	argvJSON, err := json.Marshal(argv)
	if err != nil {
		t.Fatal(err)
	}
	fields["argv_sha256"] = hash(string(argvJSON))
	withoutDigest := make(map[string]any, len(fields))
	for key, value := range fields {
		withoutDigest[key] = value
	}
	withoutDigestJSON, err := json.Marshal(withoutDigest)
	if err != nil {
		t.Fatal(err)
	}
	fields["binding_sha256"] = hash(string(withoutDigestJSON))
	bindingJSON, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(turboQuantGateBindingJSONEnv, string(bindingJSON))
	t.Setenv(turboQuantGateBindingSHAEnv, fields["binding_sha256"].(string))
	t.Setenv(turboQuantGateNonceEnv, fields["nonce"].(string))
	t.Setenv(turboQuantFrozenManifestSHAEnv, fields["frozen_manifest_sha256"].(string))
	loadedBinding, err := LoadTurboQuantGateBindingFromEnvironment()
	if err != nil {
		t.Fatalf("synthetic environment binding load failed: %v", err)
	}
	if loadedBinding != string(bindingJSON) {
		t.Fatal("synthetic environment binding was not preserved canonically")
	}
	if _, err := validateTurboQuantGateBindingJSON(string(bindingJSON), cfg, bits, []int{}, TurboQuantRerankStorageDense, qrelsSHA, 384); err != nil {
		t.Fatalf("synthetic binding validation failed: %v", err)
	}
}

func TestTurboQuantGateCorpusReaderIncludesEmptyRelevantDocuments(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, false)
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		t.Fatal(err)
	}
	gateCorpus, err := readTurboQuantRetrievalCorpus(cfg, qrels)
	if err != nil {
		t.Fatalf("read gate corpus: %v", err)
	}
	if !retrievalTextRecordsContainID(gateCorpus, "d-empty") {
		t.Fatalf("gate-bound corpus dropped qrels-relevant empty document: %+v", gateCorpus)
	}

	cfg.GateBindingJSON = ""
	ordinaryCorpus, err := readTurboQuantRetrievalCorpus(cfg, qrels)
	if err != nil {
		t.Fatalf("read ordinary corpus: %v", err)
	}
	if retrievalTextRecordsContainID(ordinaryCorpus, "d-empty") {
		t.Fatalf("ordinary corpus unexpectedly kept empty document: %+v", ordinaryCorpus)
	}
}

func TestEvaluateTurboQuantVectorCacheGateBoundCountsEmptyRelevantDocument(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, true)
	metrics, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), cfg, []int{3, 5})
	if err != nil {
		t.Fatalf("gate-bound vector-cache evaluation: %v", err)
	}
	if metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 1 || metrics.Inputs.RelevantPairs != 2 {
		t.Fatalf("input accounting = %+v, want 3 docs, 1 query, 2 relevant pairs", metrics.Inputs)
	}
	if metrics.Rows[0].SkippedRelevantDocs != 0 || metrics.Rows[0].SkippedQueries != 0 {
		t.Fatalf("gate-bound skipped counts = row:%+v skipped:%+v, want no skipped relevant docs", metrics.Rows[0], metrics.SkippedCounts)
	}
	data, err := os.ReadFile(cfg.PerQueryJSONLPath)
	if err != nil {
		t.Fatalf("read per-query rows: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query rows = %d, want one row per bit width\n%s", len(lines), data)
	}
	for _, line := range lines {
		var row TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode per-query row: %v", err)
		}
		if row.RelevantCount != 2 {
			t.Fatalf("per-query relevant_count = %d, want 2 for empty relevant doc preservation: %+v", row.RelevantCount, row)
		}
	}
}

func TestEvaluateTurboQuantVectorCacheGateBoundFailsMissingRelevantVector(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, false)
	_, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), cfg, []int{3, 5})
	if err == nil || !strings.Contains(err.Error(), "AOQT gate-bound vector-cache evaluation missing 1 frozen corpus document vectors") {
		t.Fatalf("gate-bound missing relevant vector error = %v", err)
	}

	cfg.GateBindingJSON = ""
	metrics, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), cfg, []int{3})
	if err != nil {
		t.Fatalf("ordinary vector-cache evaluation should preserve skip behavior: %v", err)
	}
	if metrics.Rows[0].SkippedRelevantDocs != 1 || metrics.Rows[0].SkippedQueries != 0 || metrics.Inputs.RelevantPairs != 1 {
		t.Fatalf("ordinary skipped/accounting = inputs:%+v row:%+v, want one skipped relevant doc", metrics.Inputs, metrics.Rows[0])
	}
}

func TestEvaluateTurboQuantVectorCacheGateBoundFailsMissingNonRelevantCorpusVector(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, true)
	if err := os.WriteFile(cfg.DocVectorPath, []byte(
		turboQuantGateTestVectorJSONL("d1", 0)+
			turboQuantGateTestVectorJSONL("d-empty", 2),
	), 0o644); err != nil {
		t.Fatalf("rewrite doc vectors: %v", err)
	}

	_, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), cfg, []int{3, 5})
	if err == nil || !strings.Contains(err.Error(), "AOQT gate-bound vector-cache evaluation missing 1 frozen corpus document vectors") {
		t.Fatalf("gate-bound missing non-relevant corpus vector error = %v", err)
	}

	cfg.GateBindingJSON = ""
	metrics, err := EvaluateTurboQuantVectorCacheRetrieval(context.Background(), cfg, []int{3})
	if err != nil {
		t.Fatalf("ordinary vector-cache evaluation should preserve missing non-relevant behavior: %v", err)
	}
	if metrics.SkippedCounts.DocumentsWithoutVector != 1 {
		t.Fatalf("ordinary missing non-relevant accounting = skipped:%+v row:%+v", metrics.SkippedCounts, metrics.Rows[0])
	}
}

func TestEvaluateTurboQuantRawGateBoundPreservesEmptyRelevantDocumentWithSpecialTokens(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, false)
	model := loadTinyD384QrelsBindingRetrievalModelWithEmptySpecialTokens(t)

	metrics, err := EvaluateTurboQuantRetrieval(context.Background(), model, cfg, []int{3, 5})
	if err != nil {
		t.Fatalf("raw gate-bound empty relevant document evaluation: %v", err)
	}
	if metrics.Inputs.Documents != 3 || metrics.Inputs.Queries != 1 || metrics.Inputs.RelevantPairs != 2 {
		t.Fatalf("input accounting = %+v, want 3 docs, 1 query, 2 relevant pairs", metrics.Inputs)
	}
	if metrics.Rows[0].SkippedRelevantDocs != 0 || metrics.Rows[0].SkippedQueries != 0 {
		t.Fatalf("gate-bound raw skipped counts = row:%+v skipped:%+v, want no skipped relevant docs", metrics.Rows[0], metrics.SkippedCounts)
	}
	data, err := os.ReadFile(cfg.PerQueryJSONLPath)
	if err != nil {
		t.Fatalf("read per-query rows: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("per-query rows = %d, want one row per bit width\n%s", len(lines), data)
	}
	for _, line := range lines {
		var row TurboQuantRetrievalPerQueryRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("decode per-query row: %v", err)
		}
		if row.RelevantCount != 2 {
			t.Fatalf("per-query relevant_count = %d, want 2 for empty relevant doc preservation: %+v", row.RelevantCount, row)
		}
	}
}

func TestEvaluateTurboQuantRawGateBoundFailsClosedWhenTokenizerCannotRepresentEmptyText(t *testing.T) {
	cfg, _ := writeTurboQuantGateEmptyRelevantVectorFixture(t, false)
	model := loadTinyQrelsBindingRetrievalModel(t)

	_, err := EvaluateTurboQuantRetrieval(context.Background(), model, cfg, []int{3, 5})
	if err == nil || !strings.Contains(err.Error(), "embed corpus") || !strings.Contains(err.Error(), "tokenized text is empty") {
		t.Fatalf("raw gate-bound empty relevant document error = %v", err)
	}
}

func loadTinyD384QrelsBindingRetrievalModelWithEmptySpecialTokens(t *testing.T) *EmbeddingModel {
	t.Helper()
	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: "tiny_embed_masked_pooled", Preset: compiler.PresetTinyEmbedMaskedPooled})
	if err != nil {
		t.Fatalf("build tiny embedding model: %v", err)
	}
	manifest := tinyMaskedEmbeddingManifest()
	manifest.Tokenizer.VocabSize = 3
	manifest.Tokenizer.MaxSequence = 2
	rt := New(cuda.New(), metal.New())
	model, err := rt.LoadEmbedding(context.Background(), bundle.Artifact, manifest, tinyD384EmbedWeights()...)
	if err != nil {
		t.Fatalf("load tiny D384 embedding model: %v", err)
	}
	tokenizer := tinyEmbeddingTokenizerFile()
	tokenizer.PadToken = "[PAD]"
	tokenizer.UnknownToken = "[UNK]"
	tokenizer.BOSToken = "[PAD]"
	tokenizer.EOSToken = "[UNK]"
	if err := model.attachTokenizer(tokenizer); err != nil {
		t.Fatalf("attach tiny BOS/EOS tokenizer: %v", err)
	}
	tokens, _, err := model.TokenizeText("")
	if err != nil {
		t.Fatalf("BOS/EOS tokenizer should encode empty text: %v", err)
	}
	if len(tokens) != 2 {
		t.Fatalf("empty text tokens = %v, want BOS/EOS pair", tokens)
	}
	return model
}

func tinyD384EmbedWeights() []LoadOption {
	const dim = 384
	tokenEmbedding := make([]float32, 3*dim)
	tokenEmbedding[0] = 1
	tokenEmbedding[dim+1] = 1
	tokenEmbedding[2*dim] = 1
	tokenEmbedding[2*dim+1] = 1
	projection := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		projection[i*dim+i] = 1
	}
	return []LoadOption{
		WithWeight("token_embedding", backend.NewTensorF16([]int{3, dim}, tokenEmbedding)),
		WithWeight("projection", backend.NewTensorF16([]int{dim, dim}, projection)),
	}
}

func writeTurboQuantGateEmptyRelevantVectorFixture(t *testing.T, includeEmptyDocVector bool) (RetrievalEvalConfig, string) {
	t.Helper()
	root := t.TempDir()
	datasetDir := filepath.Join(root, "fiqa")
	qrelsDir := filepath.Join(datasetDir, "qrels")
	if err := os.MkdirAll(qrelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	artifactPath := filepath.Join(root, "model.mll")
	packageManifestPath := ResolvePackageManifestPath(artifactPath)
	corpusPath := filepath.Join(datasetDir, "corpus.jsonl")
	queriesPath := filepath.Join(datasetDir, "queries.jsonl")
	qrelsPath := filepath.Join(qrelsDir, "test.tsv")
	datasetManifestPath := filepath.Join(datasetDir, "manifest.json")
	docVectorsPath := filepath.Join(root, "doc-vectors.jsonl")
	queryVectorsPath := filepath.Join(root, "query-vectors.jsonl")
	for path, data := range map[string][]byte{
		artifactPath:        []byte("artifact"),
		packageManifestPath: []byte("package-manifest"),
		datasetManifestPath: []byte(`{"schema":"test"}`),
		corpusPath: []byte(
			`{"_id":"d1","text":"alpha"}` + "\n" +
				`{"_id":"d-empty","title":"","text":""}` + "\n" +
				`{"_id":"d2","text":"beta"}` + "\n"),
		queriesPath: []byte(`{"_id":"q1","text":"alpha query"}` + "\n"),
		qrelsPath:   []byte("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td-empty\t1\n"),
	} {
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	docVectorData := turboQuantGateTestVectorJSONL("d1", 0) + turboQuantGateTestVectorJSONL("d2", 1)
	if includeEmptyDocVector {
		docVectorData += turboQuantGateTestVectorJSONL("d-empty", 2)
	}
	if err := os.WriteFile(docVectorsPath, []byte(docVectorData), 0o644); err != nil {
		t.Fatalf("write doc vectors: %v", err)
	}
	if err := os.WriteFile(queryVectorsPath, []byte(turboQuantGateTestVectorJSONL("q1", 0)), 0o644); err != nil {
		t.Fatalf("write query vectors: %v", err)
	}
	cfg := RetrievalEvalConfig{
		DatasetName:                "fiqa",
		ArtifactPath:               artifactPath,
		CorpusPath:                 corpusPath,
		QueriesPath:                queriesPath,
		QrelsPath:                  qrelsPath,
		DocVectorPath:              docVectorsPath,
		QueryVectorPath:            queryVectorsPath,
		BackendName:                "synthetic",
		BatchSize:                  64,
		TopK:                       120,
		PerQueryTopK:               120,
		QuantizerSeed:              5581486560434873699,
		GateBindingMetricsJSONPath: filepath.Join(root, "metrics.json"),
		GateBindingMetricsTSVPath:  filepath.Join(root, "metrics.tsv"),
		PerQueryJSONLPath:          filepath.Join(root, "per-query.jsonl"),
	}
	binding := buildSyntheticTurboQuantGateBindingJSON(t, cfg, datasetDir, []int{3, 5})
	cfg.GateBindingJSON = binding
	return cfg, binding
}

func buildSyntheticTurboQuantGateBindingJSON(t *testing.T, cfg RetrievalEvalConfig, datasetDir string, bits []int) string {
	t.Helper()
	hash := func(value string) string { return gateBindingSHA256(value) }
	outputs := map[string]string{
		"metrics":     cfg.GateBindingMetricsJSONPath,
		"metrics_tsv": cfg.GateBindingMetricsTSVPath,
		"per_query":   cfg.PerQueryJSONLPath,
	}
	argv := []string{"/bin/eos", "eval-retrieval-turboquant", cfg.ArtifactPath, datasetDir}
	fields := map[string]any{
		"schema":                            TurboQuantGateBindingSchema,
		"gate_id":                           "synthetic-gate",
		"role":                              "anchor",
		"domain":                            cfg.DatasetName,
		"nonce":                             "0123456789abcdef0123456789abcdef",
		"frozen_manifest_sha256":            hash("frozen"),
		"frozen_manifest_digest":            hash("frozen-digest"),
		"package_sha256":                    sha256FileHexForTest(t, cfg.ArtifactPath),
		"package_manifest_sha256":           sha256FileHexForTest(t, ResolvePackageManifestPath(cfg.ArtifactPath)),
		"sibling_rollup_sha256":             hash("siblings"),
		"sidecar_sha256":                    nil,
		"embedding_space_id":                hash("embedding-space"),
		"source_manifest_sha256":            hash("source-manifest"),
		"gate_script_sha256":                hash("gate-script"),
		"binary_sha256":                     hash("binary"),
		"argv":                              argv,
		"argv_sha256":                       "",
		"cwd":                               filepath.Dir(datasetDir),
		"dataset_id":                        cfg.DatasetName,
		"dataset_manifest_sha256":           sha256FileHexForTest(t, filepath.Join(filepath.Dir(cfg.CorpusPath), "manifest.json")),
		"corpus_sha256":                     sha256FileHexForTest(t, cfg.CorpusPath),
		"queries_sha256":                    sha256FileHexForTest(t, cfg.QueriesPath),
		"qrels_sha256":                      sha256FileHexForTest(t, cfg.QrelsPath),
		"compatibility_digest":              hash("compatibility"),
		"workload_sha256":                   hash("workload"),
		"approved_workload_sha256":          hash("approved-workload"),
		"workload_qid_set_sha256_by_domain": map[string]string{cfg.DatasetName: hash("qids")},
		"workload_query_count_by_domain":    map[string]int{cfg.DatasetName: 1},
		"workload_qrels_sha256_by_domain":   map[string]string{cfg.DatasetName: hash("qrels-by-domain")},
		"config":                            map[string]any{"dimension": 384, "bits": bits, "seed": cfg.QuantizerSeed, "top_k": cfg.TopK, "batch_size": cfg.BatchSize, "max_docs": cfg.MaxDocs, "max_queries": cfg.MaxQueries, "per_query_top_k": cfg.PerQueryTopK, "rerank_overfetch": []int{}, "package_mode": "native_mll_sibling", "score_mode": "turboquant_ip_prepared", "split": "test", "allow_research_only_aoqt": cfg.AllowResearchOnlyAOQT},
		"outputs":                           outputs,
		"nfcorpus_boundary_qids_sha256":     hash("boundary-qids"),
		"nfcorpus_boundary_rank_window":     []int{80, 120},
	}
	argvJSON, err := json.Marshal(argv)
	if err != nil {
		t.Fatal(err)
	}
	fields["argv_sha256"] = hash(string(argvJSON))
	withoutDigest := make(map[string]any, len(fields))
	for key, value := range fields {
		withoutDigest[key] = value
	}
	withoutDigestJSON, err := json.Marshal(withoutDigest)
	if err != nil {
		t.Fatal(err)
	}
	fields["binding_sha256"] = hash(string(withoutDigestJSON))
	bindingJSON, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	return string(bindingJSON)
}

func sha256FileHexForTest(t *testing.T, path string) string {
	t.Helper()
	sum, err := sha256FileHex(path)
	if err != nil {
		t.Fatalf("hash %s: %v", path, err)
	}
	return sum
}

func turboQuantGateTestVectorJSONL(id string, hotIndex int) string {
	values := make([]string, 384)
	for i := range values {
		values[i] = "0"
	}
	values[hotIndex] = "1"
	return `{"id":` + strconv.Quote(id) + `,"vector":[` + strings.Join(values, ",") + "]}" + "\n"
}

func retrievalTextRecordsContainID(records []retrievalTextRecord, id string) bool {
	for _, record := range records {
		if record.ID == id {
			return true
		}
	}
	return false
}
