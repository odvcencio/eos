package eosruntime

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// TurboQuantGateBindingSchema identifies the nonce-bound native evaluator
// extension consumed by the AOQT heldout gate. The binding is deliberately
// carried as JSON rather than a map so large integer values remain exact.
const TurboQuantGateBindingSchema = "eos.aoqt.native_eval_gate_binding.v1"

// TurboQuantRuntimeBindingSchema identifies the binding produced by the
// native evaluator from the process argv and the files it actually loaded.
// It is intentionally distinct from TurboQuantGateBindingSchema: the latter
// is gate-owned freshness/provenance metadata, while this record is evidence
// that the executable resolved the expected package and workload itself.
const TurboQuantRuntimeBindingSchema = "eos.aoqt.native_eval_runtime_binding.v1"

const turboQuantRuntimeBindingProducer = "eos-native-runtime-resolved-inputs"

const (
	turboQuantGateBindingJSONEnv   = "EOS_AOQT_GATE_BINDING_JSON"
	turboQuantGateBindingSHAEnv    = "EOS_AOQT_GATE_BINDING_SHA256"
	turboQuantGateNonceEnv         = "EOS_AOQT_GATE_NONCE"
	turboQuantFrozenManifestSHAEnv = "EOS_AOQT_FROZEN_MANIFEST_SHA256"
	// The frozen manifest path is deliberately a separate freshness input. A
	// SHA alone cannot tell the executable which immutable workload descriptor
	// to resolve, and accepting a caller-supplied binding would reduce this
	// evidence to an echo channel.
	turboQuantFrozenManifestPathEnv = "EOS_AOQT_FROZEN_MANIFEST_PATH"
)

var turboQuantGateBindingKeys = []string{
	"schema",
	"gate_id",
	"role",
	"domain",
	"nonce",
	"frozen_manifest_sha256",
	"frozen_manifest_digest",
	"package_sha256",
	"package_manifest_sha256",
	"sibling_rollup_sha256",
	"sidecar_sha256",
	"embedding_space_id",
	"source_manifest_sha256",
	"gate_script_sha256",
	"binary_sha256",
	"argv",
	"argv_sha256",
	"cwd",
	"dataset_id",
	"dataset_manifest_sha256",
	"corpus_sha256",
	"queries_sha256",
	"qrels_sha256",
	"compatibility_digest",
	"workload_sha256",
	"approved_workload_sha256",
	"workload_qid_set_sha256_by_domain",
	"workload_query_count_by_domain",
	"workload_qrels_sha256_by_domain",
	"config",
	"outputs",
	"nfcorpus_boundary_qids_sha256",
	"nfcorpus_boundary_rank_window",
	"binding_sha256",
}

// LoadTurboQuantGateBindingFromEnvironment reads the binding channel used by
// the heldout harness. An absent binding preserves ordinary evaluator use; a
// partially populated channel fails closed instead of silently emitting
// unverifiable native evidence.
func LoadTurboQuantGateBindingFromEnvironment() (string, error) {
	raw, rawSet := os.LookupEnv(turboQuantGateBindingJSONEnv)
	bindingSHA, bindingSHASet := os.LookupEnv(turboQuantGateBindingSHAEnv)
	nonce, nonceSet := os.LookupEnv(turboQuantGateNonceEnv)
	frozenSHA, frozenSHASet := os.LookupEnv(turboQuantFrozenManifestSHAEnv)
	if !rawSet {
		if bindingSHASet || nonceSet || frozenSHASet {
			return "", fmt.Errorf("AOQT gate binding environment is incomplete")
		}
		return "", nil
	}
	if strings.TrimSpace(raw) == "" {
		return "", fmt.Errorf("AOQT gate binding environment contains an empty binding")
	}
	if !bindingSHASet || !nonceSet || !frozenSHASet || bindingSHA == "" || nonce == "" || frozenSHA == "" {
		return "", fmt.Errorf("AOQT gate binding environment requires %s, %s, and %s", turboQuantGateBindingSHAEnv, turboQuantGateNonceEnv, turboQuantFrozenManifestSHAEnv)
	}
	canonical, fields, err := parseTurboQuantGateBindingJSON(raw)
	if err != nil {
		return "", err
	}
	if got := gateBindingString(fields, "binding_sha256"); got != bindingSHA {
		return "", fmt.Errorf("AOQT gate binding sha256 environment mismatch")
	}
	if got := gateBindingString(fields, "nonce"); got != nonce {
		return "", fmt.Errorf("AOQT gate nonce environment mismatch")
	}
	if got := gateBindingString(fields, "frozen_manifest_sha256"); got != frozenSHA {
		return "", fmt.Errorf("AOQT frozen manifest sha256 environment mismatch")
	}
	return string(canonical), nil
}

// parseTurboQuantGateBindingJSON accepts only the canonical object emitted by
// the gate. Comparing the canonical re-encoding also rejects duplicate keys,
// trailing data, and whitespace/order substitutions that would otherwise
// make the binding digest ambiguous.
func parseTurboQuantGateBindingJSON(raw string) (json.RawMessage, map[string]json.RawMessage, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil, nil
	}
	decoder := json.NewDecoder(strings.NewReader(raw))
	var fields map[string]json.RawMessage
	if err := decoder.Decode(&fields); err != nil {
		return nil, nil, fmt.Errorf("invalid AOQT gate binding JSON: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return nil, nil, fmt.Errorf("invalid AOQT gate binding JSON: trailing value")
		}
		return nil, nil, fmt.Errorf("invalid AOQT gate binding JSON: %w", err)
	}
	if fields == nil {
		return nil, nil, fmt.Errorf("invalid AOQT gate binding JSON: root must be an object")
	}
	canonical, err := json.Marshal(fields)
	if err != nil {
		return nil, nil, fmt.Errorf("canonicalize AOQT gate binding: %w", err)
	}
	if !bytes.Equal(bytes.TrimSpace([]byte(raw)), canonical) {
		return nil, nil, fmt.Errorf("AOQT gate binding must be canonical JSON without duplicate keys")
	}
	if err := requireTurboQuantGateBindingKeys(fields); err != nil {
		return nil, nil, err
	}
	return append(json.RawMessage(nil), canonical...), fields, nil
}

func requireTurboQuantGateBindingKeys(fields map[string]json.RawMessage) error {
	want := make(map[string]struct{}, len(turboQuantGateBindingKeys))
	for _, key := range turboQuantGateBindingKeys {
		want[key] = struct{}{}
	}
	var missing, extra []string
	for key := range want {
		if _, ok := fields[key]; !ok {
			missing = append(missing, key)
		}
	}
	for key := range fields {
		if _, ok := want[key]; !ok {
			extra = append(extra, key)
		}
	}
	if len(missing) != 0 || len(extra) != 0 {
		sort.Strings(missing)
		sort.Strings(extra)
		return fmt.Errorf("AOQT gate binding key set mismatch (missing=%v extra=%v)", missing, extra)
	}
	return nil
}

func gateBindingString(fields map[string]json.RawMessage, key string) string {
	var value string
	if err := json.Unmarshal(fields[key], &value); err != nil {
		return ""
	}
	return value
}

func gateBindingSHA256(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func requireGateBindingSHA256(fields map[string]json.RawMessage, key string) error {
	value := gateBindingString(fields, key)
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return fmt.Errorf("AOQT gate binding %s must be a lowercase sha256", key)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("AOQT gate binding %s must be a lowercase sha256: %w", key, err)
	}
	return nil
}

func requireGateBindingString(fields map[string]json.RawMessage, key string) error {
	value := gateBindingString(fields, key)
	if value == "" || strings.IndexFunc(value, func(r rune) bool { return r < 0x20 }) >= 0 {
		return fmt.Errorf("AOQT gate binding %s must be a non-empty string", key)
	}
	return nil
}

func requireGateBindingArray[T any](fields map[string]json.RawMessage, key string, value *T) error {
	if err := json.Unmarshal(fields[key], value); err != nil {
		return fmt.Errorf("AOQT gate binding %s has invalid type: %w", key, err)
	}
	return nil
}

func requireGateBindingCanonicalValue(fields map[string]json.RawMessage, key string, expected any) error {
	actual := bytes.TrimSpace(fields[key])
	want, err := json.Marshal(expected)
	if err != nil {
		return fmt.Errorf("AOQT gate binding %s expected value cannot be encoded: %w", key, err)
	}
	if !bytes.Equal(actual, want) {
		return fmt.Errorf("AOQT gate binding %s does not match evaluator configuration", key)
	}
	return nil
}

func validateTurboQuantGateBindingJSON(raw string, cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string, qrelsSHA256 string, dimension int) (json.RawMessage, error) {
	if raw == "" {
		if cfg.AllowResearchOnlyAOQT {
			return nil, fmt.Errorf("AOQT research-only loader requires a gate binding")
		}
		return nil, nil
	}
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("AOQT gate binding must not be whitespace-only")
	}
	canonical, fields, err := parseTurboQuantGateBindingJSON(raw)
	if err != nil || len(canonical) == 0 {
		return nil, err
	}
	if gateBindingString(fields, "schema") != TurboQuantGateBindingSchema {
		return nil, fmt.Errorf("AOQT gate binding schema mismatch")
	}
	for _, key := range []string{"gate_id", "role", "domain", "nonce", "cwd", "dataset_id"} {
		if err := requireGateBindingString(fields, key); err != nil {
			return nil, err
		}
	}
	for _, key := range []string{
		"frozen_manifest_sha256", "frozen_manifest_digest", "package_sha256", "package_manifest_sha256", "sibling_rollup_sha256", "embedding_space_id", "source_manifest_sha256", "gate_script_sha256", "binary_sha256", "argv_sha256", "dataset_manifest_sha256", "corpus_sha256", "queries_sha256", "qrels_sha256", "compatibility_digest", "workload_sha256", "approved_workload_sha256", "nfcorpus_boundary_qids_sha256",
	} {
		if err := requireGateBindingSHA256(fields, key); err != nil {
			return nil, err
		}
	}
	if sidecar, ok := fields["sidecar_sha256"]; !ok || !bytes.Equal(sidecar, []byte("null")) {
		if err := requireGateBindingSHA256(fields, "sidecar_sha256"); err != nil {
			return nil, err
		}
	}
	if err := requireGateBindingSHA256(fields, "binding_sha256"); err != nil {
		return nil, err
	}
	withoutDigest := make(map[string]json.RawMessage, len(fields)-1)
	for key, value := range fields {
		if key != "binding_sha256" {
			withoutDigest[key] = value
		}
	}
	digestBytes, err := json.Marshal(withoutDigest)
	if err != nil {
		return nil, fmt.Errorf("AOQT gate binding digest: %w", err)
	}
	if got, want := gateBindingString(fields, "binding_sha256"), gateBindingSHA256(string(digestBytes)); got != want {
		return nil, fmt.Errorf("AOQT gate binding self-digest mismatch")
	}
	if cfg.DatasetName == "" || gateBindingString(fields, "domain") != cfg.DatasetName || gateBindingString(fields, "dataset_id") != cfg.DatasetName {
		return nil, fmt.Errorf("AOQT gate binding dataset mismatch")
	}
	if qrelsSHA256 != "" && gateBindingString(fields, "qrels_sha256") != qrelsSHA256 {
		return nil, fmt.Errorf("AOQT gate binding qrels sha256 mismatch")
	}
	if cfg.ArtifactPath != "" {
		artifactSHA, err := sha256FileHex(cfg.ArtifactPath)
		if err != nil {
			return nil, fmt.Errorf("AOQT gate binding package hash: %w", err)
		}
		if gateBindingString(fields, "package_sha256") != artifactSHA {
			return nil, fmt.Errorf("AOQT gate binding package sha256 mismatch")
		}
		manifestPath := ResolvePackageManifestPath(cfg.ArtifactPath)
		manifestSHA, err := sha256FileHex(manifestPath)
		if err != nil {
			return nil, fmt.Errorf("AOQT gate binding package manifest hash: %w", err)
		}
		if gateBindingString(fields, "package_manifest_sha256") != manifestSHA {
			return nil, fmt.Errorf("AOQT gate binding package manifest sha256 mismatch")
		}
	}
	for key, path := range map[string]string{
		"corpus_sha256":  cfg.CorpusPath,
		"queries_sha256": cfg.QueriesPath,
		"qrels_sha256":   cfg.QrelsPath,
	} {
		if path == "" {
			return nil, fmt.Errorf("AOQT gate binding %s input path is required", key)
		}
		fileSHA, err := sha256FileHex(path)
		if err != nil {
			return nil, fmt.Errorf("AOQT gate binding %s input hash: %w", key, err)
		}
		if gateBindingString(fields, key) != fileSHA {
			return nil, fmt.Errorf("AOQT gate binding %s input hash mismatch", key)
		}
	}
	datasetManifestPath := filepath.Join(filepath.Dir(cfg.CorpusPath), "manifest.json")
	datasetManifestSHA, err := sha256FileHex(datasetManifestPath)
	if err != nil {
		return nil, fmt.Errorf("AOQT gate binding dataset manifest hash: %w", err)
	}
	if gateBindingString(fields, "dataset_manifest_sha256") != datasetManifestSHA {
		return nil, fmt.Errorf("AOQT gate binding dataset manifest sha256 mismatch")
	}
	if dimension != 0 && dimension != 384 {
		return nil, fmt.Errorf("AOQT gate binding dimension requires D384, got %d", dimension)
	}
	if err := requireGateBindingCanonicalValue(fields, "config", map[string]any{
		"dimension":                384,
		"bits":                     bits,
		"seed":                     cfg.QuantizerSeed,
		"top_k":                    cfg.TopK,
		"batch_size":               cfg.BatchSize,
		"max_docs":                 cfg.MaxDocs,
		"max_queries":              cfg.MaxQueries,
		"per_query_top_k":          cfg.PerQueryTopK,
		"rerank_overfetch":         rerankOverfetch,
		"package_mode":             "native_mll_sibling",
		"score_mode":               "turboquant_ip_prepared",
		"split":                    "test",
		"allow_research_only_aoqt": cfg.AllowResearchOnlyAOQT,
	}); err != nil {
		return nil, err
	}
	if cfg.AllowResearchOnlyAOQT && gateBindingString(fields, "role") != "candidate" {
		return nil, fmt.Errorf("AOQT research-only loader is candidate-only")
	}
	if rerankStorage != "" && len(rerankOverfetch) != 0 {
		return nil, fmt.Errorf("AOQT gate binding cannot enable reranking")
	}
	if err := requireGateBindingCanonicalValue(fields, "outputs", map[string]string{
		"metrics":     cfg.GateBindingMetricsJSONPath,
		"metrics_tsv": cfg.GateBindingMetricsTSVPath,
		"per_query":   cfg.PerQueryJSONLPath,
	}); err != nil {
		return nil, err
	}
	for key, path := range map[string]string{
		"metrics":     cfg.GateBindingMetricsJSONPath,
		"metrics_tsv": cfg.GateBindingMetricsTSVPath,
		"per_query":   cfg.PerQueryJSONLPath,
	} {
		if path == "" || !filepath.IsAbs(path) {
			return nil, fmt.Errorf("AOQT gate binding output %s must be an absolute path", key)
		}
	}
	var argv []string
	if err := requireGateBindingArray(fields, "argv", &argv); err != nil {
		return nil, err
	}
	if len(argv) < 2 || argv[1] != "eval-retrieval-turboquant" || filepath.Clean(argv[len(argv)-2]) != filepath.Clean(cfg.ArtifactPath) || filepath.Clean(argv[len(argv)-1]) != filepath.Clean(filepath.Dir(cfg.CorpusPath)) {
		return nil, fmt.Errorf("AOQT gate binding evaluator argv mismatch")
	}
	argvJSON, err := json.Marshal(argv)
	if err != nil || gateBindingString(fields, "argv_sha256") != gateBindingSHA256(string(argvJSON)) {
		return nil, fmt.Errorf("AOQT gate binding argv sha256 mismatch")
	}
	if len(bits) != 2 || bits[0] != 3 || bits[1] != 5 || cfg.BatchSize != 64 || cfg.TopK != 120 || cfg.PerQueryTopK != 120 || cfg.MaxDocs != 0 || cfg.MaxQueries != 0 || cfg.QuantizerSeed != 5581486560434873699 || len(rerankOverfetch) != 0 {
		return nil, fmt.Errorf("AOQT gate binding evaluator configuration mismatch")
	}
	return canonical, nil
}

func validateTurboQuantGateBindingInputFiles(cfg RetrievalEvalConfig) error {
	for label, path := range map[string]string{
		"corpus":  cfg.CorpusPath,
		"queries": cfg.QueriesPath,
		"qrels":   cfg.QrelsPath,
	} {
		if path == "" {
			return fmt.Errorf("AOQT gate binding %s path is required", label)
		}
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("AOQT gate binding %s path: %w", label, err)
		}
	}
	return nil
}

// turboQuantRuntimeQrelsIdentity is derived from the complete qrels file,
// including zero-relevance rows. The ordinary evaluator intentionally keeps
// only positive qrels; the heldout contract needs the complete workload
// identity and per-query relevant-count evidence.
type turboQuantRuntimeQrelsIdentity struct {
	SHA256          string
	QIDs            []string
	QIDSetSHA256    string
	QueryCount      int
	QrelsPairCount  int
	RelevantPairCnt int
	Rels            retrievalQrels
}

type turboQuantRuntimeBindingInfo struct {
	Binding                json.RawMessage
	DatasetManifestSHA256  string
	CorpusSHA256           string
	QueriesSHA256          string
	WorkloadSHA256         string
	ApprovedWorkloadSHA256 string
	FullQrels              retrievalQrels
	QrelsPairCount         int
	RelevantPairCount      int
	QueryCount             int
	CandidateCount         int
	BoundaryQIDs           map[string]struct{}
}

// buildTurboQuantRuntimeBinding returns nil when the caller is using the
// ordinary evaluator (or the legacy gate-only compatibility path). A frozen
// manifest path opts into the independently measured runtime contract. The
// path and all identities are checked here rather than accepting a binding
// supplied by the launcher.
func buildTurboQuantRuntimeBinding(cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string, qrelsSHA256 string, dimension int) (json.RawMessage, *turboQuantRuntimeBindingInfo, error) {
	frozenPath, ok := os.LookupEnv(turboQuantFrozenManifestPathEnv)
	if !ok || strings.TrimSpace(frozenPath) == "" {
		return nil, nil, nil
	}
	nonce := os.Getenv(turboQuantGateNonceEnv)
	frozenSHA := os.Getenv(turboQuantFrozenManifestSHAEnv)
	if nonce == "" || frozenSHA == "" {
		return nil, nil, fmt.Errorf("AOQT runtime binding requires %s and %s", turboQuantGateNonceEnv, turboQuantFrozenManifestSHAEnv)
	}
	if err := requireRuntimeBindingSHA256(frozenSHA, "frozen manifest sha256"); err != nil {
		return nil, nil, err
	}
	frozenPath, err := runtimeBindingCanonicalPath(frozenPath, "frozen manifest")
	if err != nil {
		return nil, nil, err
	}
	frozenData, err := os.ReadFile(frozenPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read AOQT frozen manifest: %w", err)
	}
	if got := gateBindingSHA256(string(frozenData)); got != frozenSHA {
		return nil, nil, fmt.Errorf("AOQT frozen manifest sha256 mismatch: got %s want %s", got, frozenSHA)
	}
	var frozen map[string]json.RawMessage
	if err := json.Unmarshal(frozenData, &frozen); err != nil {
		return nil, nil, fmt.Errorf("parse AOQT frozen manifest: %w", err)
	}
	if frozen == nil {
		return nil, nil, fmt.Errorf("AOQT frozen manifest root must be an object")
	}
	if schema := runtimeBindingString(frozen["schema"]); schema != "eos.aoqt.heldout_gate_frozen.v4" {
		return nil, nil, fmt.Errorf("AOQT frozen manifest schema mismatch")
	}
	if declared := runtimeBindingString(frozen["manifest_sha256"]); declared == "" {
		return nil, nil, fmt.Errorf("AOQT frozen manifest self-digest is missing")
	} else {
		withoutDigest := cloneRuntimeRawMap(frozen)
		delete(withoutDigest, "manifest_sha256")
		if got := gateBindingSHA256(string(marshalRuntimeRawMap(withoutDigest))); got != declared {
			return nil, nil, fmt.Errorf("AOQT frozen manifest self-digest mismatch")
		}
	}

	source, err := runtimeBindingObject(frozen, "source")
	if err != nil {
		return nil, nil, err
	}
	gateID := runtimeBindingString(frozen["gate_id"])
	if gateID == "" {
		return nil, nil, fmt.Errorf("AOQT frozen manifest gate_id is missing")
	}
	argv, err := currentTurboQuantRuntimeArgv()
	if err != nil {
		return nil, nil, err
	}
	binaryRecord, err := runtimeBindingObject(source, "binary")
	if err != nil {
		return nil, nil, err
	}
	binaryPath, binarySHA, _, err := verifyRuntimeBindingFileRecord(binaryRecord, "source.binary")
	if err != nil {
		return nil, nil, err
	}
	argvBinary, err := runtimeBindingCanonicalPath(argv[0], "argv[0]")
	if err != nil {
		return nil, nil, err
	}
	if argvBinary != binaryPath {
		return nil, nil, fmt.Errorf("AOQT runtime binding binary path mismatch: argv[0]=%s source.binary=%s", argvBinary, binaryPath)
	}
	if got, err := sha256FileHex(argvBinary); err != nil {
		return nil, nil, fmt.Errorf("hash AOQT runtime binary: %w", err)
	} else if got != binarySHA {
		return nil, nil, fmt.Errorf("AOQT runtime binary sha256 mismatch")
	}
	workingDirectory := runtimeBindingString(source["cwd"])
	if workingDirectory == "" {
		return nil, nil, fmt.Errorf("AOQT frozen source cwd is missing")
	}
	workingDirectory, err = runtimeBindingCanonicalPath(workingDirectory, "source.cwd")
	if err != nil {
		return nil, nil, err
	}
	if cwd, err := os.Getwd(); err != nil || cwd != workingDirectory {
		if err != nil {
			return nil, nil, fmt.Errorf("resolve AOQT runtime cwd: %w", err)
		}
		return nil, nil, fmt.Errorf("AOQT runtime cwd mismatch: got %s want %s", cwd, workingDirectory)
	}
	if argv[1] != "eval-retrieval-turboquant" {
		return nil, nil, fmt.Errorf("AOQT runtime binding evaluator command mismatch")
	}

	// Use the actual gate argv for a second independent check when available.
	// This does not supply any runtime identity; it merely rejects a process
	// whose argv differs from the command the gate launched.
	if gateRaw, present := os.LookupEnv(turboQuantGateBindingJSONEnv); present && strings.TrimSpace(gateRaw) != "" {
		_, gateFields, err := parseTurboQuantGateBindingJSON(gateRaw)
		if err != nil {
			return nil, nil, err
		}
		var expectedArgv []string
		if err := json.Unmarshal(gateFields["argv"], &expectedArgv); err != nil || !equalRuntimeStringSlices(argv, expectedArgv) {
			return nil, nil, fmt.Errorf("AOQT runtime binding argv differs from gate command")
		}
	}

	role, expectedPackage, err := runtimeBindingPackageRole(frozen, cfg.ArtifactPath)
	if err != nil {
		return nil, nil, err
	}
	packageIdentity, err := buildRuntimePackageIdentity(cfg.ArtifactPath, expectedPackage, role)
	if err != nil {
		return nil, nil, err
	}

	datasetMap, err := runtimeBindingObject(source, "dataset_by_domain")
	if err != nil {
		return nil, nil, err
	}
	datasetRecord, err := runtimeBindingObject(datasetMap, cfg.DatasetName)
	if err != nil {
		return nil, nil, err
	}
	datasetID := runtimeBindingString(datasetRecord["dataset_id"])
	if datasetID == "" || datasetID != cfg.DatasetName {
		return nil, nil, fmt.Errorf("AOQT runtime binding dataset identity mismatch")
	}
	datasetDir, err := runtimeBindingCanonicalPath(runtimeBindingString(datasetRecord["dataset_dir"]), "dataset.dataset_dir")
	if err != nil {
		return nil, nil, err
	}
	corpusPath, err := runtimeBindingCanonicalPath(cfg.CorpusPath, "corpus path")
	if err != nil {
		return nil, nil, err
	}
	queriesPath, err := runtimeBindingCanonicalPath(cfg.QueriesPath, "queries path")
	if err != nil {
		return nil, nil, err
	}
	qrelsPath, err := runtimeBindingCanonicalPath(cfg.QrelsPath, "qrels path")
	if err != nil {
		return nil, nil, err
	}
	if filepath.Dir(corpusPath) != datasetDir {
		return nil, nil, fmt.Errorf("AOQT runtime binding dataset directory mismatch")
	}
	if err := verifyRuntimeBindingDatasetRecord(datasetRecord, corpusPath, queriesPath); err != nil {
		return nil, nil, err
	}
	datasetManifestRecord, err := runtimeBindingObject(datasetRecord, "manifest")
	if err != nil {
		return nil, nil, err
	}
	datasetManifestPath, datasetManifestSHA, _, err := verifyRuntimeBindingFileRecord(datasetManifestRecord, "dataset.manifest")
	if err != nil {
		return nil, nil, err
	}
	corpusSHA, err := runtimeBindingVerifiedSHA(corpusPath, datasetRecord, "corpus")
	if err != nil {
		return nil, nil, err
	}
	queriesSHA, err := runtimeBindingVerifiedSHA(queriesPath, datasetRecord, "queries")
	if err != nil {
		return nil, nil, err
	}
	if qrelsPath == "" || qrelsSHA256 == "" {
		return nil, nil, fmt.Errorf("AOQT runtime binding qrels path and hash are required")
	}
	qrelsIdentity, err := readTurboQuantRuntimeQrelsIdentity(qrelsPath)
	if err != nil {
		return nil, nil, err
	}
	if qrelsIdentity.SHA256 != qrelsSHA256 {
		return nil, nil, fmt.Errorf("AOQT runtime binding qrels sha256 mismatch")
	}
	if err := verifyRuntimeBindingQrelsRecord(frozen, cfg.DatasetName, qrelsPath, qrelsIdentity); err != nil {
		return nil, nil, err
	}

	workloadRecord, err := runtimeBindingObject(frozen, "workload")
	if err != nil {
		return nil, nil, err
	}
	workloadPath, workloadSHA, _, err := verifyRuntimeBindingFileRecord(workloadRecord, "workload")
	if err != nil {
		return nil, nil, err
	}
	approvedRecord, err := runtimeBindingObject(frozen, "approved_workload")
	if err != nil {
		return nil, nil, err
	}
	_, approvedSHA, _, err := verifyRuntimeBindingFileRecord(approvedRecord, "approved_workload")
	if err != nil {
		return nil, nil, err
	}
	approvedDescriptor, err := runtimeBindingObject(approvedRecord, "descriptor")
	if err != nil {
		return nil, nil, err
	}
	if err := verifyRuntimeBindingDescriptor(approvedDescriptor, gateID, cfg.DatasetName, qrelsIdentity); err != nil {
		return nil, nil, err
	}
	boundaryQIDs, err := runtimeBindingBoundaryQIDs(approvedDescriptor)
	if err != nil {
		return nil, nil, err
	}

	if err := validateRuntimeBindingConfig(cfg, bits, rerankOverfetch, rerankStorage, dimension, argv); err != nil {
		return nil, nil, err
	}
	outputs, err := runtimeBindingOutputs(cfg)
	if err != nil {
		return nil, nil, err
	}
	binding := map[string]any{
		"schema":                 TurboQuantRuntimeBindingSchema,
		"producer":               turboQuantRuntimeBindingProducer,
		"gate_id":                gateID,
		"role":                   role,
		"domain":                 cfg.DatasetName,
		"split":                  "test",
		"nonce":                  nonce,
		"frozen_manifest_sha256": frozenSHA,
		"binary_path":            binaryPath,
		"binary_sha256":          binarySHA,
		"argv":                   argv,
		"argv_sha256":            gateBindingSHA256(string(mustMarshalRuntimeValue(argv))),
		"cwd":                    workingDirectory,
		"dimension":              dimension,
		"score_mode":             "turboquant_ip_prepared",
		"package_mode":           "native_mll_sibling",
		"package":                packageIdentity,
		"dataset": map[string]any{
			"dataset_id":      datasetID,
			"dataset_dir":     datasetDir,
			"manifest_path":   datasetManifestPath,
			"manifest_sha256": datasetManifestSHA,
			"corpus_path":     corpusPath,
			"corpus_sha256":   corpusSHA,
			"queries_path":    queriesPath,
			"queries_sha256":  queriesSHA,
		},
		"qrels": map[string]any{
			"path":                qrelsPath,
			"sha256":              qrelsIdentity.SHA256,
			"qid_set_sha256":      qrelsIdentity.QIDSetSHA256,
			"query_count":         qrelsIdentity.QueryCount,
			"qrels_pair_count":    qrelsIdentity.QrelsPairCount,
			"relevant_pair_count": qrelsIdentity.RelevantPairCnt,
		},
		"workload": map[string]any{
			"path":                     workloadPath,
			"sha256":                   workloadSHA,
			"descriptor_sha256":        approvedSHA,
			"qid_set_sha256_by_domain": approvedDescriptor["qid_set_sha256_by_domain"],
			"query_count_by_domain":    approvedDescriptor["query_count_by_domain"],
		},
		"config": map[string]any{
			"dimension":                384,
			"bits":                     []int{3, 5},
			"seed":                     int64(5581486560434873699),
			"top_k":                    120,
			"per_query_top_k":          120,
			"batch_size":               64,
			"max_docs":                 0,
			"max_queries":              0,
			"split":                    "test",
			"score_mode":               "turboquant_ip_prepared",
			"package_mode":             "native_mll_sibling",
			"rerank_overfetch":         []int{},
			"rerank_bits":              0,
			"allow_research_only_aoqt": cfg.AllowResearchOnlyAOQT,
		},
		"outputs": outputs,
	}
	binding["binding_sha256"] = gateBindingSHA256(string(mustMarshalRuntimeValue(binding)))
	bindingJSON := mustMarshalRuntimeValue(binding)
	return json.RawMessage(bindingJSON), &turboQuantRuntimeBindingInfo{
		Binding:                append(json.RawMessage(nil), bindingJSON...),
		DatasetManifestSHA256:  datasetManifestSHA,
		CorpusSHA256:           corpusSHA,
		QueriesSHA256:          queriesSHA,
		WorkloadSHA256:         workloadSHA,
		ApprovedWorkloadSHA256: approvedSHA,
		FullQrels:              qrelsIdentity.Rels,
		QrelsPairCount:         qrelsIdentity.QrelsPairCount,
		RelevantPairCount:      qrelsIdentity.RelevantPairCnt,
		QueryCount:             qrelsIdentity.QueryCount,
		BoundaryQIDs:           boundaryQIDs,
	}, nil
}

func currentTurboQuantRuntimeArgv() ([]string, error) {
	if len(os.Args) < 2 || os.Args[0] == "" {
		return nil, fmt.Errorf("AOQT runtime binding cannot resolve process argv")
	}
	return append([]string(nil), os.Args...), nil
}

func requireRuntimeBindingSHA256(value, label string) error {
	if len(value) != sha256.Size*2 || strings.ToLower(value) != value {
		return fmt.Errorf("AOQT runtime binding %s must be a lowercase sha256", label)
	}
	if _, err := hex.DecodeString(value); err != nil {
		return fmt.Errorf("AOQT runtime binding %s must be a lowercase sha256: %w", label, err)
	}
	return nil
}

func runtimeBindingString(raw json.RawMessage) string {
	var value string
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return value
}

func runtimeBindingObject(fields map[string]json.RawMessage, key string) (map[string]json.RawMessage, error) {
	var value map[string]json.RawMessage
	raw, ok := fields[key]
	if !ok || json.Unmarshal(raw, &value) != nil || value == nil {
		return nil, fmt.Errorf("AOQT runtime binding %s must be an object", key)
	}
	return value, nil
}

func cloneRuntimeRawMap(fields map[string]json.RawMessage) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(fields))
	for key, value := range fields {
		out[key] = append(json.RawMessage(nil), value...)
	}
	return out
}

func marshalRuntimeRawMap(fields map[string]json.RawMessage) []byte {
	data, _ := json.Marshal(fields)
	return data
}

func mustMarshalRuntimeValue(value any) []byte {
	data, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
}

func runtimeBindingCanonicalPath(raw, label string) (string, error) {
	if raw == "" || !filepath.IsAbs(raw) {
		return "", fmt.Errorf("AOQT runtime binding %s must be an absolute path", label)
	}
	path := filepath.Clean(raw)
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("AOQT runtime binding %s: %w", label, err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("AOQT runtime binding %s must not be a symlink", label)
	}
	return path, nil
}

func runtimeBindingFileRecord(fields map[string]json.RawMessage, label string) (string, string, int64, error) {
	path := runtimeBindingString(fields["path"])
	sha := runtimeBindingString(fields["sha256"])
	var bytesCount int64
	if path == "" || sha == "" || json.Unmarshal(fields["bytes"], &bytesCount) != nil {
		return "", "", 0, fmt.Errorf("AOQT runtime binding %s file record is incomplete", label)
	}
	if err := requireRuntimeBindingSHA256(sha, label+" sha256"); err != nil {
		return "", "", 0, err
	}
	canonical, err := runtimeBindingCanonicalPath(path, label+" path")
	if err != nil {
		return "", "", 0, err
	}
	if info, err := os.Stat(canonical); err != nil || !info.Mode().IsRegular() {
		if err != nil {
			return "", "", 0, fmt.Errorf("AOQT runtime binding %s: %w", label, err)
		}
		return "", "", 0, fmt.Errorf("AOQT runtime binding %s must be a regular file", label)
	}
	return canonical, sha, bytesCount, nil
}

func verifyRuntimeBindingFileRecord(fields map[string]json.RawMessage, label string) (string, string, int64, error) {
	path, expectedSHA, expectedBytes, err := runtimeBindingFileRecord(fields, label)
	if err != nil {
		return "", "", 0, err
	}
	actualSHA, err := sha256FileHex(path)
	if err != nil {
		return "", "", 0, fmt.Errorf("hash AOQT runtime binding %s: %w", label, err)
	}
	if actualSHA != expectedSHA {
		return "", "", 0, fmt.Errorf("AOQT runtime binding %s sha256 mismatch", label)
	}
	if actualBytes := runtimeBindingFileSize(path); actualBytes != expectedBytes {
		return "", "", 0, fmt.Errorf("AOQT runtime binding %s byte count mismatch", label)
	}
	return path, actualSHA, expectedBytes, nil
}

func runtimeBindingFileSize(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return -1
	}
	return info.Size()
}

func runtimeBindingPackageRole(frozen map[string]json.RawMessage, artifactPath string) (string, map[string]json.RawMessage, error) {
	artifactPath, err := runtimeBindingCanonicalPath(artifactPath, "artifact path")
	if err != nil {
		return "", nil, err
	}
	actualSHA, err := sha256FileHex(artifactPath)
	if err != nil {
		return "", nil, fmt.Errorf("hash AOQT runtime artifact: %w", err)
	}
	var match string
	var packageRecord map[string]json.RawMessage
	for _, role := range []string{"anchor", "candidate"} {
		record, err := runtimeBindingObject(frozen, role)
		if err != nil {
			return "", nil, err
		}
		canonical, expectedSHA, _, err := verifyRuntimeBindingFileRecord(record, role+" package")
		if err != nil {
			return "", nil, err
		}
		if canonical == artifactPath && expectedSHA == actualSHA {
			if match != "" {
				return "", nil, fmt.Errorf("AOQT runtime artifact matches both anchor and candidate")
			}
			match = role
			packageRecord = record
		}
	}
	if match == "" {
		return "", nil, fmt.Errorf("AOQT runtime artifact is not the frozen anchor or candidate package")
	}
	return match, packageRecord, nil
}

func buildRuntimePackageIdentity(artifactPath string, expectedPackage map[string]json.RawMessage, role string) (map[string]any, error) {
	artifactPath, err := runtimeBindingCanonicalPath(artifactPath, "package artifact")
	if err != nil {
		return nil, err
	}
	artifactSHA, err := sha256FileHex(artifactPath)
	if err != nil {
		return nil, fmt.Errorf("hash AOQT package artifact: %w", err)
	}
	if runtimeBindingString(expectedPackage["path"]) != artifactPath || runtimeBindingString(expectedPackage["sha256"]) != artifactSHA {
		return nil, fmt.Errorf("AOQT runtime package artifact identity mismatch")
	}
	manifestRaw, hasExpectedManifest := expectedPackage["package_manifest"]
	manifestPath := ResolvePackageManifestPath(artifactPath)
	_, statErr := os.Stat(manifestPath)
	hasActualManifest := statErr == nil
	if hasExpectedManifest != hasActualManifest {
		return nil, fmt.Errorf("AOQT runtime package manifest presence mismatch")
	}
	var manifestSHA any
	roles := []map[string]any{}
	if hasActualManifest {
		expectedManifest, err := runtimeRawObject(manifestRaw, "package.package_manifest")
		if err != nil {
			return nil, err
		}
		expectedPath, expectedSHA, _, err := verifyRuntimeBindingFileRecord(expectedManifest, "package.package_manifest")
		if err != nil {
			return nil, err
		}
		canonicalManifestPath, err := runtimeBindingCanonicalPath(manifestPath, "resolved package manifest")
		if err != nil {
			return nil, err
		}
		if expectedPath != canonicalManifestPath {
			return nil, fmt.Errorf("AOQT runtime package manifest path mismatch")
		}
		actualSHA, err := sha256FileHex(canonicalManifestPath)
		if err != nil || actualSHA != expectedSHA {
			return nil, fmt.Errorf("AOQT runtime package manifest sha256 mismatch")
		}
		manifest, err := ReadPackageManifestFile(canonicalManifestPath)
		if err != nil {
			return nil, fmt.Errorf("read AOQT runtime package manifest: %w", err)
		}
		if err := manifest.Validate(); err != nil {
			return nil, fmt.Errorf("validate AOQT runtime package manifest: %w", err)
		}
		seenPaths := map[string]struct{}{}
		seenRoles := map[string]struct{}{}
		for _, item := range manifest.Files {
			if item.Path == "" || filepath.Base(item.Path) != item.Path {
				return nil, fmt.Errorf("AOQT runtime package role %q has an invalid sibling path", item.Role)
			}
			if _, ok := seenPaths[item.Path]; ok {
				return nil, fmt.Errorf("AOQT runtime package has duplicate sibling path %q", item.Path)
			}
			if _, ok := seenRoles[item.Role]; ok {
				return nil, fmt.Errorf("AOQT runtime package has duplicate sibling role %q", item.Role)
			}
			seenPaths[item.Path] = struct{}{}
			seenRoles[item.Role] = struct{}{}
			path := filepath.Join(filepath.Dir(artifactPath), item.Path)
			canonical, err := runtimeBindingCanonicalPath(path, "package."+item.Role)
			if err != nil {
				return nil, err
			}
			sha, err := sha256FileHex(canonical)
			if err != nil {
				return nil, fmt.Errorf("hash AOQT runtime package role %q: %w", item.Role, err)
			}
			if sha != item.SHA256 || runtimeBindingFileSize(canonical) != item.Bytes {
				return nil, fmt.Errorf("AOQT runtime package role %q hash/size mismatch", item.Role)
			}
			roles = append(roles, map[string]any{"role": item.Role, "name": item.Path, "path": canonical, "sha256": sha, "bytes": item.Bytes})
		}
		sort.Slice(roles, func(i, j int) bool { return roles[i]["role"].(string) < roles[j]["role"].(string) })
		if role == "candidate" && !manifest.HasFileRole(EmbeddingPostPoolTransformRole) {
			return nil, fmt.Errorf("AOQT candidate package is missing post_pool_transform")
		}
		if role == "anchor" && manifest.HasFileRole(EmbeddingPostPoolTransformRole) {
			return nil, fmt.Errorf("AOQT anchor package unexpectedly has post_pool_transform")
		}
		manifestSHA = expectedSHA
	}
	return map[string]any{
		"artifact_path":   artifactPath,
		"artifact_sha256": artifactSHA,
		"package_manifest_path": func() any {
			if hasActualManifest {
				return manifestPath
			}
			return nil
		}(),
		"package_manifest_sha256": manifestSHA,
		"roles":                   roles,
		"roles_sha256":            gateBindingSHA256(string(mustMarshalRuntimeValue(roles))),
	}, nil
}

func runtimeRawObject(raw json.RawMessage, label string) (map[string]json.RawMessage, error) {
	var value map[string]json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil || value == nil {
		return nil, fmt.Errorf("AOQT runtime binding %s must be an object", label)
	}
	return value, nil
}

func verifyRuntimeBindingDatasetRecord(dataset map[string]json.RawMessage, corpusPath, queriesPath string) error {
	for key, actualPath := range map[string]string{"corpus": corpusPath, "queries": queriesPath} {
		record, err := runtimeBindingObject(dataset, key)
		if err != nil {
			return err
		}
		path, _, _, err := verifyRuntimeBindingFileRecord(record, "dataset."+key)
		if err != nil {
			return err
		}
		if path != actualPath {
			return fmt.Errorf("AOQT runtime binding dataset.%s path mismatch", key)
		}
	}
	return nil
}

func runtimeBindingVerifiedSHA(path string, dataset map[string]json.RawMessage, key string) (string, error) {
	record, err := runtimeBindingObject(dataset, key)
	if err != nil {
		return "", err
	}
	recordedPath, sha, _, err := verifyRuntimeBindingFileRecord(record, "dataset."+key)
	if err != nil {
		return "", err
	}
	if recordedPath != path {
		return "", fmt.Errorf("AOQT runtime binding dataset.%s path mismatch", key)
	}
	return sha, nil
}

func verifyRuntimeBindingQrelsRecord(frozen map[string]json.RawMessage, domain, path string, identity turboQuantRuntimeQrelsIdentity) error {
	qrelsMap, err := runtimeBindingObject(frozen, "qrels")
	if err != nil {
		return err
	}
	record, err := runtimeBindingObject(qrelsMap, domain)
	if err != nil {
		return err
	}
	recordedPath, expectedSHA, _, err := verifyRuntimeBindingFileRecord(record, "qrels."+domain)
	if err != nil {
		return err
	}
	if recordedPath != path || expectedSHA != identity.SHA256 {
		return fmt.Errorf("AOQT runtime binding qrels file identity mismatch")
	}
	if runtimeBindingInt(record["query_count"]) != int64(identity.QueryCount) || runtimeBindingInt(record["qrels_pair_count"]) != int64(identity.QrelsPairCount) || runtimeBindingInt(record["relevant_pair_count"]) != int64(identity.RelevantPairCnt) || runtimeBindingString(record["qid_set_sha256"]) != identity.QIDSetSHA256 {
		return fmt.Errorf("AOQT runtime binding qrels counts/qid hash mismatch")
	}
	return nil
}

func runtimeBindingInt(raw json.RawMessage) int64 {
	var value int64
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return -1
	}
	return value
}

func verifyRuntimeBindingDescriptor(descriptor map[string]json.RawMessage, gateID, domain string, identity turboQuantRuntimeQrelsIdentity) error {
	if runtimeBindingString(descriptor["gate_id"]) != gateID || runtimeBindingString(descriptor["split"]) != "test" || runtimeBindingInt(descriptor["dimension"]) != 384 {
		return fmt.Errorf("AOQT runtime binding approved workload identity mismatch")
	}
	qidsByDomain, err := runtimeBindingObject(descriptor, "query_ids_by_domain")
	if err != nil {
		return err
	}
	qidsRaw := qidsByDomain[domain]
	var qids []string
	if json.Unmarshal(qidsRaw, &qids) != nil || !equalRuntimeStringSlices(sortedRuntimeStrings(qids), identity.QIDs) {
		return fmt.Errorf("AOQT runtime binding approved qid set mismatch")
	}
	if runtimeBindingInt(runtimeBindingObjectValue(descriptor, "query_count_by_domain", domain)) != int64(identity.QueryCount) {
		return fmt.Errorf("AOQT runtime binding approved query count mismatch")
	}
	return nil
}

func runtimeBindingObjectValue(fields map[string]json.RawMessage, outer, key string) json.RawMessage {
	var nested map[string]json.RawMessage
	if json.Unmarshal(fields[outer], &nested) != nil {
		return nil
	}
	return nested[key]
}

func sortedRuntimeStrings(values []string) []string {
	copyValues := append([]string(nil), values...)
	sort.Strings(copyValues)
	return copyValues
}

func runtimeBindingBoundaryQIDs(descriptor map[string]json.RawMessage) (map[string]struct{}, error) {
	var qids []string
	if json.Unmarshal(descriptor["nfcorpus_boundary_qids"], &qids) != nil || len(qids) == 0 {
		return nil, fmt.Errorf("AOQT runtime binding NFCorpus boundary qid descriptor is missing")
	}
	if runtimeBindingString(descriptor["nfcorpus_boundary_qids_sha256"]) != gateBindingSHA256(string(mustMarshalRuntimeValue(sortedRuntimeStrings(qids)))) {
		return nil, fmt.Errorf("AOQT runtime binding NFCorpus boundary qid digest mismatch")
	}
	var rankWindow []int
	if json.Unmarshal(descriptor["nfcorpus_boundary_rank_window"], &rankWindow) != nil || len(rankWindow) != 2 || rankWindow[0] != 80 || rankWindow[1] != 120 {
		return nil, fmt.Errorf("AOQT runtime binding NFCorpus boundary rank window mismatch")
	}
	result := make(map[string]struct{}, len(qids))
	for _, qid := range qids {
		if qid == "" || strings.IndexFunc(qid, func(r rune) bool { return r <= ' ' }) >= 0 {
			return nil, fmt.Errorf("AOQT runtime binding NFCorpus boundary qid is invalid")
		}
		if _, exists := result[qid]; exists {
			return nil, fmt.Errorf("AOQT runtime binding NFCorpus boundary qid is duplicated")
		}
		result[qid] = struct{}{}
	}
	return result, nil
}

func validateRuntimeBindingConfig(cfg RetrievalEvalConfig, bits, rerankOverfetch []int, rerankStorage string, dimension int, argv []string) error {
	if dimension != 384 || cfg.BatchSize != 64 || cfg.TopK != 120 || cfg.PerQueryTopK != 120 || cfg.MaxDocs != 0 || cfg.MaxQueries != 0 || cfg.QuantizerSeed != 5581486560434873699 || len(bits) != 2 || bits[0] != 3 || bits[1] != 5 || len(rerankOverfetch) != 0 || cfg.RerankBits != 0 || rerankStorage != TurboQuantRerankStorageDense {
		return fmt.Errorf("AOQT runtime binding evaluator configuration mismatch")
	}
	for key, want := range map[string]string{
		"--dataset":         cfg.DatasetName,
		"--split":           "test",
		"--qrels":           cfg.QrelsPath,
		"--batch-size":      "64",
		"--top-k":           "120",
		"--bits":            "3,5",
		"--quantizer-seed":  "5581486560434873699",
		"--max-docs":        "0",
		"--max-queries":     "0",
		"--per-query-top-k": "120",
	} {
		value, ok := runtimeBindingFlagValue(argv, key)
		if !ok || value != want {
			return fmt.Errorf("AOQT runtime binding argv %s mismatch", key)
		}
	}
	allowResearchFlag := false
	for _, token := range argv {
		if token == "--allow-research-only-aoqt" {
			allowResearchFlag = true
			break
		}
	}
	if allowResearchFlag != cfg.AllowResearchOnlyAOQT {
		return fmt.Errorf("AOQT runtime binding argv research-only loader flag mismatch")
	}
	return nil
}

func runtimeBindingFlagValue(argv []string, flag string) (string, bool) {
	for index := 0; index+1 < len(argv); index++ {
		if argv[index] == flag {
			return argv[index+1], true
		}
	}
	return "", false
}

func runtimeBindingOutputs(cfg RetrievalEvalConfig) (map[string]string, error) {
	paths := map[string]string{"metrics": cfg.GateBindingMetricsJSONPath, "metrics_tsv": cfg.GateBindingMetricsTSVPath, "per_query": cfg.PerQueryJSONLPath}
	for key, value := range paths {
		if value == "" || !filepath.IsAbs(value) {
			return nil, fmt.Errorf("AOQT runtime binding output %s must be an absolute path", key)
		}
		paths[key] = filepath.Clean(value)
	}
	return paths, nil
}

func equalRuntimeStringSlices(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func readTurboQuantRuntimeQrelsIdentity(path string) (turboQuantRuntimeQrelsIdentity, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("read AOQT runtime qrels: %w", err)
	}
	identity := turboQuantRuntimeQrelsIdentity{Rels: make(retrievalQrels)}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	lineNumber := 0
	sawData := false
	for scanner.Scan() {
		lineNumber++
		rawLine := strings.TrimRight(scanner.Text(), "\r\n")
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		fields := strings.Split(rawLine, "\t")
		if len(fields) == 1 {
			fields = strings.Fields(line)
		}
		if !sawData && len(fields) > 0 && (strings.EqualFold(strings.TrimSpace(fields[0]), "query-id") || strings.EqualFold(strings.TrimSpace(fields[0]), "qid") || strings.EqualFold(strings.TrimSpace(fields[0]), "query_id")) {
			sawData = true
			continue
		}
		sawData = true
		var qid, doc, scoreRaw string
		switch len(fields) {
		case 3:
			qid, doc, scoreRaw = strings.TrimSpace(fields[0]), strings.TrimSpace(fields[1]), strings.TrimSpace(fields[2])
		case 4:
			if strings.TrimSpace(fields[1]) != "0" {
				return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels %s:%d iteration field must be zero", path, lineNumber)
			}
			qid, doc, scoreRaw = strings.TrimSpace(fields[0]), strings.TrimSpace(fields[2]), strings.TrimSpace(fields[3])
		default:
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels %s:%d expected 3 or 4 columns", path, lineNumber)
		}
		if qid == "" || doc == "" || strings.IndexFunc(qid, func(r rune) bool { return r <= ' ' }) >= 0 || strings.IndexFunc(doc, func(r rune) bool { return r <= ' ' }) >= 0 {
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels %s:%d invalid query/document id", path, lineNumber)
		}
		relevance, err := strconv.ParseFloat(scoreRaw, 64)
		if err != nil || math.IsNaN(relevance) || math.IsInf(relevance, 0) || relevance < 0 {
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels %s:%d relevance must be finite and non-negative", path, lineNumber)
		}
		if identity.Rels[qid] == nil {
			identity.Rels[qid] = make(map[string]float64)
		}
		if _, exists := identity.Rels[qid][doc]; exists {
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels %s:%d duplicate qid/doc pair", path, lineNumber)
		}
		identity.Rels[qid][doc] = relevance
		identity.QrelsPairCount++
		if relevance > 0 {
			identity.RelevantPairCnt++
		}
	}
	if err := scanner.Err(); err != nil {
		return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("scan AOQT runtime qrels: %w", err)
	}
	if len(identity.Rels) == 0 {
		return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels are empty")
	}
	for qid, rels := range identity.Rels {
		if len(rels) == 0 {
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels qid %q has no pairs", qid)
		}
		positive := false
		for _, relevance := range rels {
			positive = positive || relevance > 0
		}
		if !positive {
			return turboQuantRuntimeQrelsIdentity{}, fmt.Errorf("AOQT runtime qrels qid %q has no positive relevance", qid)
		}
		identity.QIDs = append(identity.QIDs, qid)
	}
	sort.Strings(identity.QIDs)
	identity.QueryCount = len(identity.QIDs)
	identity.QIDSetSHA256 = gateBindingSHA256(string(mustMarshalRuntimeValue(identity.QIDs)))
	identity.SHA256 = gateBindingSHA256(string(data))
	return identity, nil
}

// turboQuantGateQualityForQuery is retained as a compatibility helper for
// callers of the interrupted gate implementation. Gate-bound output uses the
// same linear graded relevance convention as ordinary EOS retrieval.
func turboQuantGateQualityForQuery(scores []retrievalScoredDoc, rels map[string]float64) RetrievalEvalQualityMetrics {
	return retrievalQualityForQuery(scores, rels)
}
