package eosruntime

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"m31labs.dev/eos/runtime/backend"
)

const RetrievalEvalMetricsSchema = "manta.embedding_retrieval_metrics.v1"
const RetrievalEvalPerQuerySchema = "manta.embedding_retrieval_per_query.v1"

const (
	EmbeddingRoleModeAuto          = "auto"
	EmbeddingRoleModeRaw           = "raw"
	EmbeddingRoleModeQueryDocument = "query-document"
)

// RetrievalEvalConfig describes a BEIR-style retrieval eval.
type RetrievalEvalConfig struct {
	DatasetName          string
	ArtifactPath         string
	CorpusPath           string
	QueriesPath          string
	QrelsPath            string
	DocVectorPath        string
	QueryVectorPath      string
	BackendName          string
	BatchSize            int
	TopK                 int
	PerQueryTopK         int
	MaxDocs              int
	MaxQueries           int
	PerQueryJSONLPath    string
	AllowMissingRelevant bool
	QuantizerSeed        int64
	// RerankBits requests an independent TurboQuant bit width for
	// TurboQuantRerankStorageCompactReconstruct reranking: a second
	// IPQuantizer (same seed, same dim, RerankBits wide) quantizes the
	// corpus into a sidecar code set, and overfetched candidates are
	// reranked by dequantizing from that sidecar instead of the primary
	// retrieval codes. Zero disables mixed-width reranking (the rerank
	// dequantizes the primary codes, unchanged from prior behavior).
	RerankBits                   int
	BaselineDim                  int
	MultiVectorAggregation       string
	MultiVectorChildCountPenalty float64
	Hybrid                       RetrievalEvalHybridConfig
	RoleMode                     string
}

type RetrievalEvalHybridConfig struct {
	Method              string
	Alpha               float64
	AlphaSet            bool
	RRFK                float64
	RRFLambda           float64
	DenseProtectTopK    int
	DenseCandidatesOnly bool
}

// RetrievalEvalMetrics records standard retrieval metrics for one dataset split.
type RetrievalEvalMetrics struct {
	Schema        string                       `json:"schema"`
	Dataset       string                       `json:"dataset"`
	Artifact      string                       `json:"artifact,omitempty"`
	Backend       string                       `json:"backend,omitempty"`
	Inputs        RetrievalEvalInputMetrics    `json:"inputs"`
	Config        RetrievalEvalConfigMetrics   `json:"config"`
	Quality       RetrievalEvalQualityMetrics  `json:"quality"`
	Throughput    RetrievalEvalThroughput      `json:"throughput"`
	SkippedCounts RetrievalEvalSkippedCounts   `json:"skipped_counts,omitempty"`
	SparseLexical *SparseLexicalLabelEvalStats `json:"sparse_lexical,omitempty"`
}

type RetrievalEvalInputMetrics struct {
	CorpusPath      string `json:"corpus_path,omitempty"`
	QueriesPath     string `json:"queries_path,omitempty"`
	QrelsPath       string `json:"qrels_path,omitempty"`
	LabelPath       string `json:"label_path,omitempty"`
	HeadPath        string `json:"head_path,omitempty"`
	DocVectorPath   string `json:"doc_vector_path,omitempty"`
	QueryVectorPath string `json:"query_vector_path,omitempty"`
	Documents       int    `json:"documents"`
	Queries         int    `json:"queries"`
	RelevantPairs   int    `json:"relevant_pairs"`
	ScoredPairs     int64  `json:"scored_pairs"`
}

type RetrievalEvalConfigMetrics struct {
	BatchSize  int                         `json:"batch_size"`
	TopK       int                         `json:"top_k"`
	MaxDocs    int                         `json:"max_docs,omitempty"`
	MaxQueries int                         `json:"max_queries,omitempty"`
	Hybrid     *RetrievalEvalHybridMetrics `json:"hybrid,omitempty"`
	RoleMode   string                      `json:"role_mode,omitempty"`
}

type RetrievalEvalHybridMetrics struct {
	Method              string  `json:"method"`
	Alpha               float64 `json:"alpha,omitempty"`
	RRFK                float64 `json:"rrf_k,omitempty"`
	RRFLambda           float64 `json:"rrf_lambda,omitempty"`
	DenseProtectTopK    int     `json:"dense_protect_top_k"`
	DenseCandidatesOnly bool    `json:"dense_candidates_only"`
}

type RetrievalEvalQualityMetrics struct {
	NDCGAt10      float64 `json:"ndcg_at_10"`
	NDCGAt100     float64 `json:"ndcg_at_100"`
	MRRAt10       float64 `json:"mrr_at_10"`
	PrecisionAt1  float64 `json:"precision_at_1"`
	PrecisionAt5  float64 `json:"precision_at_5"`
	PrecisionAt10 float64 `json:"precision_at_10"`
	HitAt1        float64 `json:"hit_at_1"`
	HitAt5        float64 `json:"hit_at_5"`
	HitAt10       float64 `json:"hit_at_10"`
	MAPAt10       float64 `json:"map_at_10"`
	MAPAt100      float64 `json:"map_at_100"`
	RecallAt10    float64 `json:"recall_at_10"`
	RecallAt100   float64 `json:"recall_at_100"`
}

type RetrievalEvalThroughput struct {
	ElapsedSeconds       float64 `json:"elapsed_seconds"`
	DocumentEmbedSeconds float64 `json:"document_embed_seconds"`
	QueryEmbedSeconds    float64 `json:"query_embed_seconds"`
	ScoreSeconds         float64 `json:"score_seconds"`
	DocumentsPerSecond   float64 `json:"documents_per_second"`
	QueriesPerSecond     float64 `json:"queries_per_second"`
	ScoresPerSecond      float64 `json:"scores_per_second"`
}

type RetrievalEvalSkippedCounts struct {
	QueriesWithoutText         int `json:"queries_without_text,omitempty"`
	RelevantDocsWithoutText    int `json:"relevant_docs_without_text,omitempty"`
	QueriesWithoutRelevantDocs int `json:"queries_without_relevant_docs,omitempty"`
	QueriesWithoutVector       int `json:"queries_without_vector,omitempty"`
	DocumentsWithoutVector     int `json:"documents_without_vector,omitempty"`
}

type RetrievalEvalPerQueryRow struct {
	Schema            string                        `json:"schema"`
	Dataset           string                        `json:"dataset"`
	QueryID           string                        `json:"query_id"`
	RelevantCount     int                           `json:"relevant_count"`
	FirstRelevantRank int                           `json:"first_relevant_rank"`
	Quality           RetrievalEvalQualityMetrics   `json:"quality"`
	TopK              []RetrievalEvalPerQueryTopDoc `json:"top_k"`
}

type RetrievalEvalPerQueryTopDoc struct {
	Rank                 int      `json:"rank"`
	DocID                string   `json:"doc_id"`
	Score                float32  `json:"score"`
	Relevance            float64  `json:"relevance"`
	ChildID              string   `json:"child_id,omitempty"`
	ChildScore           *float64 `json:"child_score,omitempty"`
	DenseRank            *int     `json:"dense_rank,omitempty"`
	BM25Rank             *int     `json:"bm25_rank,omitempty"`
	DenseScore           *float64 `json:"dense_score,omitempty"`
	DenseChildID         string   `json:"dense_child_id,omitempty"`
	DenseChildScore      *float64 `json:"dense_child_score,omitempty"`
	CompactRank          *int     `json:"compact_rank,omitempty"`
	CompactScore         *float64 `json:"compact_score,omitempty"`
	CompactChildID       string   `json:"compact_child_id,omitempty"`
	CompactChildScore    *float64 `json:"compact_child_score,omitempty"`
	BM25Score            *float64 `json:"bm25_score,omitempty"`
	DenseNormalizedScore *float64 `json:"dense_normalized_score,omitempty"`
	BM25NormalizedScore  *float64 `json:"bm25_normalized_score,omitempty"`
}

type retrievalTextRecord struct {
	ID   string
	Text string
}

type tokenizedRetrievalTextRecord struct {
	Index  int
	ID     string
	Tokens []int32
}

type retrievalVectorRecord struct {
	ID     string
	Vector []float32
}

type retrievalChildVectorRecord struct {
	ParentID string
	ChildID  string
	Vector   []float32
}

type retrievalQrels map[string]map[string]float64

type retrievalVectorJSONRecord struct {
	ID        string    `json:"id"`
	BEIRID    string    `json:"_id"`
	ParentID  string    `json:"parent_id"`
	ChildID   string    `json:"child_id"`
	Vector    []float32 `json:"vector"`
	Embedding []float32 `json:"embedding"`
	Values    []float32 `json:"values"`
}

// BEIRRetrievalPaths resolves the conventional corpus/query/qrels files under a dataset directory.
func BEIRRetrievalPaths(datasetDir, split string) (corpusPath, queriesPath, qrelsPath string) {
	if split == "" {
		split = "test"
	}
	return filepath.Join(datasetDir, "corpus.jsonl"), filepath.Join(datasetDir, "queries.jsonl"), filepath.Join(datasetDir, "qrels", split+".tsv")
}

// EvaluateEmbeddingRetrieval evaluates an embedding model against BEIR-style corpus, query, and qrels files.
func EvaluateEmbeddingRetrieval(ctx context.Context, model *EmbeddingModel, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	if model == nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embedding model is not loaded")
	}
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docStart := time.Now()
	docRole, queryRole, effectiveRoleMode, err := resolveEmbeddingRetrievalRoles(model, cfg.RoleMode)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	docVectors, err := embedRetrievalTexts(ctx, model, corpus, cfg.BatchSize, docRole)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed corpus: %w", err)
	}
	docDuration := time.Since(docStart)

	queryStart := time.Now()
	queryVectors, err := embedRetrievalTexts(ctx, model, queries, cfg.BatchSize, queryRole)
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("embed queries: %w", err)
	}
	queryDuration := time.Since(queryStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeRetrievalQualityWithPerQuery(queryVectors, docVectors, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated corpus")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  string(model.Backend()),
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:    cfg.CorpusPath,
			QueriesPath:   cfg.QueriesPath,
			QrelsPath:     cfg.QrelsPath,
			Documents:     len(docVectors),
			Queries:       evaluatedQueries,
			RelevantPairs: relevantPairs,
			ScoredPairs:   scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
			RoleMode:   effectiveRoleMode,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: docDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), docDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
		},
	}, nil
}

// EvaluateVectorCacheRetrieval evaluates precomputed document/query vectors against BEIR-style qrels.
func EvaluateVectorCacheRetrieval(ctx context.Context, cfg RetrievalEvalConfig) (RetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return RetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.BackendName == "" {
		cfg.BackendName = "vectors"
	}
	start := time.Now()
	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	docStart := time.Now()
	docVectors, missingDocVectors, docDim, err := readRetrievalVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read document vectors: %w", err)
	}
	docDuration := time.Since(docStart)
	if len(docVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no vectors for the evaluated corpus")
	}

	queryStart := time.Now()
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return RetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	queryDuration := time.Since(queryStart)
	if len(queryVectors) == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return RetrievalEvalMetrics{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}

	scoreStart := time.Now()
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, err := computeRetrievalQualityWithPerQuery(queryVectors, docVectors, qrels, cfg.TopK, cfg.DatasetName, cfg.PerQueryJSONLPath)
	if err != nil {
		return RetrievalEvalMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return RetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant documents in the evaluated vector cache")
	}

	elapsed := time.Since(start)
	scoredPairs := int64(evaluatedQueries) * int64(len(docVectors))
	return RetrievalEvalMetrics{
		Schema:   RetrievalEvalMetricsSchema,
		Dataset:  cfg.DatasetName,
		Artifact: cfg.ArtifactPath,
		Backend:  cfg.BackendName,
		Inputs: RetrievalEvalInputMetrics{
			CorpusPath:      cfg.CorpusPath,
			QueriesPath:     cfg.QueriesPath,
			QrelsPath:       cfg.QrelsPath,
			DocVectorPath:   cfg.DocVectorPath,
			QueryVectorPath: cfg.QueryVectorPath,
			Documents:       len(docVectors),
			Queries:         evaluatedQueries,
			RelevantPairs:   relevantPairs,
			ScoredPairs:     scoredPairs,
		},
		Config: RetrievalEvalConfigMetrics{
			BatchSize:  cfg.BatchSize,
			TopK:       cfg.TopK,
			MaxDocs:    cfg.MaxDocs,
			MaxQueries: cfg.MaxQueries,
		},
		Quality: quality,
		Throughput: RetrievalEvalThroughput{
			ElapsedSeconds:       elapsed.Seconds(),
			DocumentEmbedSeconds: docDuration.Seconds(),
			QueryEmbedSeconds:    queryDuration.Seconds(),
			ScoreSeconds:         scoreDuration.Seconds(),
			DocumentsPerSecond:   ratePerSecond(float64(len(docVectors)), docDuration),
			QueriesPerSecond:     ratePerSecond(float64(len(queryVectors)), queryDuration),
			ScoresPerSecond:      ratePerSecond(float64(scoredPairs), scoreDuration),
		},
		SkippedCounts: RetrievalEvalSkippedCounts{
			QueriesWithoutText:         skippedQueries,
			RelevantDocsWithoutText:    skippedRelevantDocs,
			QueriesWithoutRelevantDocs: skippedNoRelevant,
			QueriesWithoutVector:       missingQueryVectors,
			DocumentsWithoutVector:     missingDocVectors,
		},
	}, nil
}

func normalizeRetrievalEvalConfig(cfg RetrievalEvalConfig) RetrievalEvalConfig {
	if cfg.DatasetName == "" {
		cfg.DatasetName = "retrieval"
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.TopK <= 0 {
		cfg.TopK = 100
	}
	if cfg.TopK < 100 {
		cfg.TopK = 100
	}
	if cfg.MultiVectorAggregation == "" {
		cfg.MultiVectorAggregation = TurboQuantMultiVectorAggregationMax
	}
	if cfg.RoleMode == "" {
		cfg.RoleMode = EmbeddingRoleModeAuto
	}
	return cfg
}

func resolveEmbeddingRetrievalRoles(model *EmbeddingModel, mode string) (docRole, queryRole, effectiveMode string, err error) {
	if mode == "" {
		mode = EmbeddingRoleModeAuto
	}
	roleConditioned := model != nil && model.Manifest().normalized().roleConditioned()
	switch mode {
	case EmbeddingRoleModeAuto:
		if roleConditioned {
			return EmbeddingRoleDocument, EmbeddingRoleQuery, EmbeddingRoleModeQueryDocument, nil
		}
		return EmbeddingRoleRaw, EmbeddingRoleRaw, EmbeddingRoleModeRaw, nil
	case EmbeddingRoleModeRaw:
		return EmbeddingRoleRaw, EmbeddingRoleRaw, EmbeddingRoleModeRaw, nil
	case EmbeddingRoleModeQueryDocument:
		if !roleConditioned {
			return "", "", "", fmt.Errorf("role-mode query-document requires a role-conditioned embedding model")
		}
		return EmbeddingRoleDocument, EmbeddingRoleQuery, EmbeddingRoleModeQueryDocument, nil
	default:
		return "", "", "", fmt.Errorf("unsupported role-mode %q", mode)
	}
}

type beirJSONRecord struct {
	ID    string         `json:"_id"`
	Title string         `json:"title,omitempty"`
	Text  string         `json:"text"`
	Meta  map[string]any `json:"metadata,omitempty"`
}

func readBEIRCorpus(path string, limit int) ([]retrievalTextRecord, error) {
	return readBEIRTextFile(path, nil, limit)
}

func readBEIRCorpusWithRelevantIncludingEmpty(path string, limit int, qrels retrievalQrels) ([]retrievalTextRecord, error) {
	return readBEIRCorpusWithRelevantOptions(path, limit, qrels, true)
}

// readBEIRCorpusWithRelevant reads the corpus capped to `limit` documents, but
// guarantees every qrels-relevant document is included regardless of its
// position in the file. A naive first-N cap can drop all relevant docs (their
// IDs are often late in file order), which silently produces nDCG=0 and makes a
// capped retrieval eval meaningless. Relevant docs come first; the remainder is
// filled with non-relevant distractors up to `limit`. With limit<=0 the full
// corpus is read.
func readBEIRCorpusWithRelevant(path string, limit int, qrels retrievalQrels) ([]retrievalTextRecord, error) {
	return readBEIRCorpusWithRelevantOptions(path, limit, qrels, false)
}

func readBEIRCorpusWithRelevantOptions(path string, limit int, qrels retrievalQrels, includeEmpty bool) ([]retrievalTextRecord, error) {
	if limit <= 0 {
		return readBEIRTextFileOptions(path, nil, 0, includeEmpty)
	}
	relevant := make(map[string]bool)
	for _, docs := range qrels {
		for docID := range docs {
			relevant[docID] = true
		}
	}
	if len(relevant) == 0 {
		return readBEIRTextFileOptions(path, nil, limit, includeEmpty)
	}
	all, err := readBEIRTextFileOptions(path, nil, 0, includeEmpty)
	if err != nil {
		return nil, err
	}
	out := make([]retrievalTextRecord, 0, limit)
	seen := make(map[string]bool, limit)
	for _, rec := range all { // all relevant docs first (kept even if they exceed limit)
		if relevant[rec.ID] {
			out = append(out, rec)
			seen[rec.ID] = true
		}
	}
	for _, rec := range all { // fill with distractors up to the cap
		if len(out) >= limit {
			break
		}
		if !seen[rec.ID] {
			out = append(out, rec)
			seen[rec.ID] = true
		}
	}
	return out, nil
}

func readBEIRQueries(path string, qrels retrievalQrels, limit int) ([]retrievalTextRecord, int, error) {
	needed := make(map[string]bool, len(qrels))
	for id := range qrels {
		needed[id] = true
	}
	records, err := readBEIRTextFile(path, needed, limit)
	if err != nil {
		return nil, 0, err
	}
	return records, len(needed) - len(records), nil
}

func readBEIRTextFile(path string, ids map[string]bool, limit int) ([]retrievalTextRecord, error) {
	return readBEIRTextFileOptions(path, ids, limit, false)
}

func readBEIRTextFileOptions(path string, ids map[string]bool, limit int, includeEmpty bool) ([]retrievalTextRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out := []retrievalTextRecord{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		if limit > 0 && len(out) >= limit {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record beirJSONRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if record.ID == "" {
			continue
		}
		if ids != nil && !ids[record.ID] {
			continue
		}
		text := strings.TrimSpace(strings.Join([]string{record.Title, record.Text}, "\n"))
		if text == "" && !includeEmpty {
			continue
		}
		out = append(out, retrievalTextRecord{ID: record.ID, Text: text})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func retrievalIDs(records []retrievalTextRecord) []string {
	out := make([]string, len(records))
	for i, record := range records {
		out[i] = record.ID
	}
	return out
}

func readRetrievalVectorCache(path string, orderedIDs []string) ([]retrievalVectorRecord, int, int, error) {
	needed := make(map[string]int, len(orderedIDs))
	for i, id := range orderedIDs {
		if id != "" {
			needed[id] = i
		}
	}
	vectors := make(map[string][]float32, len(needed))
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	dim := 0
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record retrievalVectorJSONRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, 0, 0, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := strings.TrimSpace(record.ID)
		if id == "" {
			id = strings.TrimSpace(record.BEIRID)
		}
		if id == "" {
			continue
		}
		if _, ok := needed[id]; !ok {
			continue
		}
		vector := firstRetrievalVector(record)
		if len(vector) == 0 {
			return nil, 0, 0, fmt.Errorf("%s:%d: vector for %q is empty", path, lineNo, id)
		}
		if dim == 0 {
			dim = len(vector)
		} else if len(vector) != dim {
			return nil, 0, 0, fmt.Errorf("%s:%d: vector for %q has dimension %d, want %d", path, lineNo, id, len(vector), dim)
		}
		if _, exists := vectors[id]; !exists {
			vectors[id] = normalizeRetrievalVector(vector)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, 0, err
	}
	out := make([]retrievalVectorRecord, 0, len(orderedIDs))
	missing := 0
	for _, id := range orderedIDs {
		vector, ok := vectors[id]
		if !ok {
			missing++
			continue
		}
		out = append(out, retrievalVectorRecord{ID: id, Vector: vector})
	}
	return out, missing, dim, nil
}

func readRetrievalChildVectorCache(path string, orderedParentIDs []string) ([]retrievalChildVectorRecord, int, int, error) {
	needed := make(map[string]int, len(orderedParentIDs))
	for i, id := range orderedParentIDs {
		if id != "" {
			needed[id] = i
		}
	}
	type childRecord struct {
		childID string
		vector  []float32
	}
	childrenByParent := make(map[string][]childRecord, len(needed))
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	dim := 0
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record retrievalVectorJSONRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, 0, 0, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := firstNonEmptyString(record.ID, record.BEIRID)
		parentID := firstNonEmptyString(record.ParentID, id)
		childID := firstNonEmptyString(record.ChildID, id, parentID)
		if parentID == "" {
			continue
		}
		if _, ok := needed[parentID]; !ok {
			continue
		}
		vector := firstRetrievalVector(record)
		if len(vector) == 0 {
			return nil, 0, 0, fmt.Errorf("%s:%d: vector for parent %q child %q is empty", path, lineNo, parentID, childID)
		}
		if dim == 0 {
			dim = len(vector)
		} else if len(vector) != dim {
			return nil, 0, 0, fmt.Errorf("%s:%d: vector for parent %q child %q has dimension %d, want %d", path, lineNo, parentID, childID, len(vector), dim)
		}
		childrenByParent[parentID] = append(childrenByParent[parentID], childRecord{
			childID: childID,
			vector:  normalizeRetrievalVector(vector),
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, 0, err
	}
	out := make([]retrievalChildVectorRecord, 0)
	missing := 0
	for _, parentID := range orderedParentIDs {
		children := childrenByParent[parentID]
		if len(children) == 0 {
			missing++
			continue
		}
		for _, child := range children {
			out = append(out, retrievalChildVectorRecord{
				ParentID: parentID,
				ChildID:  child.childID,
				Vector:   child.vector,
			})
		}
	}
	return out, missing, dim, nil
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func firstRetrievalVector(record retrievalVectorJSONRecord) []float32 {
	switch {
	case len(record.Vector) > 0:
		return record.Vector
	case len(record.Embedding) > 0:
		return record.Embedding
	default:
		return record.Values
	}
}

func readBEIRQrels(path string) (retrievalQrels, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	qrels := retrievalQrels{}
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if lineNo == 1 && len(parts) >= 2 && strings.Contains(strings.ToLower(parts[0]), "query") {
			continue
		}
		if len(parts) < 3 {
			return nil, fmt.Errorf("%s:%d: expected query-id, corpus-id, score", path, lineNo)
		}
		docField, scoreField := 1, 2
		if len(parts) >= 4 {
			docField, scoreField = 2, 3
		}
		score, err := strconv.ParseFloat(parts[scoreField], 64)
		if err != nil {
			return nil, fmt.Errorf("%s:%d: score: %w", path, lineNo, err)
		}
		if score <= 0 {
			continue
		}
		queryID := strings.TrimSpace(parts[0])
		docID := strings.TrimSpace(parts[docField])
		if queryID == "" || docID == "" {
			continue
		}
		if qrels[queryID] == nil {
			qrels[queryID] = map[string]float64{}
		}
		qrels[queryID][docID] = score
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(qrels) == 0 {
		return nil, fmt.Errorf("qrels file has no positive relevance rows: %s", path)
	}
	return qrels, nil
}

func embedRetrievalTexts(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord, batchSize int, role string) ([]retrievalVectorRecord, error) {
	if batchSize <= 0 {
		batchSize = 64
	}
	tokenized, lengths, err := tokenizeRetrievalTexts(ctx, model, records)
	if err != nil {
		return nil, err
	}
	out := make([]retrievalVectorRecord, len(records))
	for _, length := range lengths {
		group := tokenized[length]
		for start := 0; start < len(group); start += batchSize {
			end := start + batchSize
			if end > len(group) {
				end = len(group)
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			chunk := group[start:end]
			batches := make([][]int32, len(chunk))
			for i, record := range chunk {
				batches[i] = record.Tokens
			}
			result, err := model.EmbedBatchWithRole(ctx, batches, role)
			if err != nil {
				return nil, err
			}
			rows, err := embeddingRowViews(result.Embeddings, len(chunk))
			if err != nil {
				return nil, err
			}
			for i, row := range rows {
				record := chunk[i]
				out[record.Index] = retrievalVectorRecord{
					ID:     record.ID,
					Vector: normalizeRetrievalVector(row),
				}
			}
		}
	}
	return out, nil
}

func embeddingRowViews(t *backend.Tensor, wantRows int) ([][]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("embedding tensor is nil")
	}
	if len(t.F32) == 0 {
		return nil, fmt.Errorf("embedding tensor has no float data")
	}
	switch len(t.Shape) {
	case 1:
		if wantRows != 1 {
			return nil, fmt.Errorf("embedding tensor shape %v cannot provide %d rows", t.Shape, wantRows)
		}
		return [][]float32{t.F32}, nil
	case 2:
		rows, cols := t.Shape[0], t.Shape[1]
		if rows != wantRows {
			return nil, fmt.Errorf("embedding tensor rows = %d, want %d", rows, wantRows)
		}
		if len(t.F32) < rows*cols {
			return nil, fmt.Errorf("embedding tensor has %d values, want at least %d", len(t.F32), rows*cols)
		}
		out := make([][]float32, rows)
		for i := 0; i < rows; i++ {
			out[i] = t.F32[i*cols : (i+1)*cols]
		}
		return out, nil
	default:
		return nil, fmt.Errorf("embedding tensor shape %v is not rank 1 or 2", t.Shape)
	}
}

func tokenizeRetrievalTexts(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) (map[int][]tokenizedRetrievalTextRecord, []int, error) {
	tokenized, err := tokenizeRetrievalTextRecords(ctx, model, records)
	if err != nil {
		return nil, nil, err
	}
	groups := make(map[int][]tokenizedRetrievalTextRecord)
	lengths := []int{}
	for _, record := range tokenized {
		length := len(record.Tokens)
		if len(groups[length]) == 0 {
			lengths = append(lengths, length)
		}
		groups[length] = append(groups[length], record)
	}
	slices.Sort(lengths)
	return groups, lengths, nil
}

func tokenizeRetrievalTextRecords(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) ([]tokenizedRetrievalTextRecord, error) {
	if len(records) == 0 {
		return nil, nil
	}
	workers := min(runtime.GOMAXPROCS(0), len(records))
	if workers <= 1 || len(records) < 128 {
		return tokenizeRetrievalTextRecordsSerial(ctx, model, records)
	}
	out := make([]tokenizedRetrievalTextRecord, len(records))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var firstErr error
	setErr := func(err error) {
		if err == nil {
			return
		}
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	hasErr := func() bool {
		errMu.Lock()
		ok := firstErr != nil
		errMu.Unlock()
		return ok
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				if err := ctx.Err(); err != nil {
					setErr(err)
					continue
				}
				tokens, _, err := model.TokenizeText(records[i].Text)
				if err != nil {
					setErr(fmt.Errorf("text %d: %w", i, err))
					continue
				}
				out[i] = tokenizedRetrievalTextRecord{
					Index:  i,
					ID:     records[i].ID,
					Tokens: tokens,
				}
			}
		}()
	}
	for i := range records {
		if err := ctx.Err(); err != nil {
			setErr(err)
			break
		}
		if hasErr() {
			break
		}
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	errMu.Lock()
	err := firstErr
	errMu.Unlock()
	if err != nil {
		return nil, err
	}
	return out, nil
}

func tokenizeRetrievalTextRecordsSerial(ctx context.Context, model *EmbeddingModel, records []retrievalTextRecord) ([]tokenizedRetrievalTextRecord, error) {
	out := make([]tokenizedRetrievalTextRecord, len(records))
	for i, record := range records {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tokens, _, err := model.TokenizeText(record.Text)
		if err != nil {
			return nil, fmt.Errorf("text %d: %w", i, err)
		}
		out[i] = tokenizedRetrievalTextRecord{
			Index:  i,
			ID:     record.ID,
			Tokens: tokens,
		}
	}
	return out, nil
}

func normalizeRetrievalVector(in []float32) []float32 {
	out := append([]float32(nil), in...)
	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return out
	}
	scale := float32(1 / math.Sqrt(sum))
	for i := range out {
		out[i] *= scale
	}
	return out
}

type retrievalScoredDoc struct {
	ID                   string
	Score                float32
	ChildID              string
	ChildScore           *float64
	DenseRank            int
	BM25Rank             int
	DenseScore           *float64
	DenseChildID         string
	DenseChildScore      *float64
	BM25Score            *float64
	DenseNormalizedScore *float64
	BM25NormalizedScore  *float64
}

func computeRetrievalQuality(queries, docs []retrievalVectorRecord, qrels retrievalQrels, topK int) (RetrievalEvalQualityMetrics, int, int, int, int) {
	quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, _ := computeRetrievalQualityWithPerQuery(queries, docs, qrels, topK, "", "")
	return quality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant
}

func computeRetrievalQualityWithPerQuery(queries, docs []retrievalVectorRecord, qrels retrievalQrels, topK int, datasetName, perQueryJSONLPath string) (RetrievalEvalQualityMetrics, int, int, int, int, error) {
	docIDSet := make(map[string]bool, len(docs))
	for _, doc := range docs {
		docIDSet[doc.ID] = true
	}
	if topK < 100 {
		topK = 100
	}
	writer, err := newRetrievalPerQueryWriter(perQueryJSONLPath)
	if err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
	}
	defer writer.Close()
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	for _, query := range queries {
		rels := qrels[query.ID]
		filteredRels := make(map[string]float64, len(rels))
		for docID, rel := range rels {
			if docIDSet[docID] {
				filteredRels[docID] = rel
			} else {
				skippedRelevantDocs++
			}
		}
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		scores := topRetrievalScores(query.Vector, docs, topK)
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		row := buildRetrievalPerQueryRow(datasetName, query.ID, scores, filteredRels)
		addRetrievalPerQueryQuality(&totals, row.Quality)
		if err := writer.Write(row); err != nil {
			return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
		}
	}
	if err := writer.Close(); err != nil {
		return RetrievalEvalQualityMetrics{}, 0, 0, 0, 0, err
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, nil
}

func topRetrievalScores(query []float32, docs []retrievalVectorRecord, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, doc := range docs {
		score := retrievalScoredDoc{ID: doc.ID, Score: dotRetrievalVectors(query, doc.Vector)}
		if len(h) < topK {
			heap.Push(&h, score)
			continue
		}
		if retrievalScoreBetter(score, h[0]) {
			h[0] = score
			heap.Fix(&h, 0)
		}
	}
	scores := []retrievalScoredDoc(h)
	slices.SortFunc(scores, func(a, b retrievalScoredDoc) int {
		if retrievalScoreBetter(a, b) {
			return -1
		}
		if retrievalScoreBetter(b, a) {
			return 1
		}
		return 0
	})
	return scores
}

func retrievalScoreBetter(a, b retrievalScoredDoc) bool {
	if a.Score > b.Score {
		return true
	}
	if a.Score < b.Score {
		return false
	}
	return a.ID < b.ID
}

type retrievalScoreHeap []retrievalScoredDoc

func (h retrievalScoreHeap) Len() int {
	return len(h)
}

func (h retrievalScoreHeap) Less(i, j int) bool {
	return retrievalScoreBetter(h[j], h[i])
}

func (h retrievalScoreHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *retrievalScoreHeap) Push(x any) {
	*h = append(*h, x.(retrievalScoredDoc))
}

func (h *retrievalScoreHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func dotRetrievalVectors(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

type retrievalPerQueryWriter struct {
	file   *os.File
	writer *bufio.Writer
	closed bool
}

func newRetrievalPerQueryWriter(path string) (*retrievalPerQueryWriter, error) {
	if path == "" {
		return &retrievalPerQueryWriter{}, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create per-query retrieval JSONL: %w", err)
	}
	return &retrievalPerQueryWriter{file: file, writer: bufio.NewWriter(file)}, nil
}

func (w *retrievalPerQueryWriter) Write(row RetrievalEvalPerQueryRow) error {
	if w == nil || w.writer == nil {
		return nil
	}
	data, err := json.Marshal(row)
	if err != nil {
		return err
	}
	if _, err := w.writer.Write(data); err != nil {
		return fmt.Errorf("write per-query retrieval JSONL: %w", err)
	}
	if err := w.writer.WriteByte('\n'); err != nil {
		return fmt.Errorf("write per-query retrieval JSONL: %w", err)
	}
	return nil
}

func (w *retrievalPerQueryWriter) Close() error {
	if w == nil || w.closed {
		return nil
	}
	w.closed = true
	var flushErr, closeErr error
	if w.writer != nil {
		flushErr = w.writer.Flush()
	}
	if w.file != nil {
		closeErr = w.file.Close()
	}
	if flushErr != nil {
		return fmt.Errorf("flush per-query retrieval JSONL: %w", flushErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close per-query retrieval JSONL: %w", closeErr)
	}
	return nil
}

func buildRetrievalPerQueryRow(datasetName, queryID string, scores []retrievalScoredDoc, rels map[string]float64) RetrievalEvalPerQueryRow {
	row := RetrievalEvalPerQueryRow{
		Schema:            RetrievalEvalPerQuerySchema,
		Dataset:           datasetName,
		QueryID:           queryID,
		RelevantCount:     len(rels),
		FirstRelevantRank: firstRelevantRank(scores, rels),
		Quality:           retrievalQualityForQuery(scores, rels),
		TopK:              make([]RetrievalEvalPerQueryTopDoc, 0, len(scores)),
	}
	for i, score := range scores {
		row.TopK = append(row.TopK, RetrievalEvalPerQueryTopDoc{
			Rank:                 i + 1,
			DocID:                score.ID,
			Score:                score.Score,
			Relevance:            rels[score.ID],
			ChildID:              score.ChildID,
			ChildScore:           score.ChildScore,
			DenseRank:            optionalPositiveInt(score.DenseRank),
			BM25Rank:             optionalPositiveInt(score.BM25Rank),
			DenseScore:           score.DenseScore,
			DenseChildID:         score.DenseChildID,
			DenseChildScore:      score.DenseChildScore,
			BM25Score:            score.BM25Score,
			DenseNormalizedScore: score.DenseNormalizedScore,
			BM25NormalizedScore:  score.BM25NormalizedScore,
		})
	}
	return row
}

func optionalPositiveInt(value int) *int {
	if value <= 0 {
		return nil
	}
	return &value
}

func firstRelevantRank(scores []retrievalScoredDoc, rels map[string]float64) int {
	for i, score := range scores {
		if rels[score.ID] > 0 {
			return i + 1
		}
	}
	return 0
}

func retrievalQualityForQuery(scores []retrievalScoredDoc, rels map[string]float64) RetrievalEvalQualityMetrics {
	return RetrievalEvalQualityMetrics{
		NDCGAt10:      ndcgAt(scores, rels, 10),
		NDCGAt100:     ndcgAt(scores, rels, 100),
		MRRAt10:       mrrAt(scores, rels, 10),
		PrecisionAt1:  precisionAt(scores, rels, 1),
		PrecisionAt5:  precisionAt(scores, rels, 5),
		PrecisionAt10: precisionAt(scores, rels, 10),
		HitAt1:        hitAt(scores, rels, 1),
		HitAt5:        hitAt(scores, rels, 5),
		HitAt10:       hitAt(scores, rels, 10),
		MAPAt10:       averagePrecisionAt(scores, rels, 10),
		MAPAt100:      averagePrecisionAt(scores, rels, 100),
		RecallAt10:    recallAt(scores, rels, 10),
		RecallAt100:   recallAt(scores, rels, 100),
	}
}

func ndcgAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	dcg := 0.0
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		rel := rels[scores[i].ID]
		if rel <= 0 {
			continue
		}
		dcg += rel / math.Log2(float64(i)+2)
	}
	idcg := idealDCGAt(rels, k)
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func idealDCGAt(rels map[string]float64, k int) float64 {
	values := make([]float64, 0, len(rels))
	for _, rel := range rels {
		if rel > 0 {
			values = append(values, rel)
		}
	}
	slices.SortFunc(values, func(a, b float64) int {
		if a > b {
			return -1
		}
		if a < b {
			return 1
		}
		return 0
	})
	dcg := 0.0
	limit := min(k, len(values))
	for i := 0; i < limit; i++ {
		dcg += values[i] / math.Log2(float64(i)+2)
	}
	return dcg
}

func mrrAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		if rels[scores[i].ID] > 0 {
			return 1 / float64(i+1)
		}
	}
	return 0
}

func addRetrievalQuality(totals *RetrievalEvalQualityMetrics, scores []retrievalScoredDoc, rels map[string]float64) {
	addRetrievalPerQueryQuality(totals, retrievalQualityForQuery(scores, rels))
}

func addRetrievalPerQueryQuality(totals *RetrievalEvalQualityMetrics, query RetrievalEvalQualityMetrics) {
	totals.NDCGAt10 += query.NDCGAt10
	totals.NDCGAt100 += query.NDCGAt100
	totals.MRRAt10 += query.MRRAt10
	totals.PrecisionAt1 += query.PrecisionAt1
	totals.PrecisionAt5 += query.PrecisionAt5
	totals.PrecisionAt10 += query.PrecisionAt10
	totals.HitAt1 += query.HitAt1
	totals.HitAt5 += query.HitAt5
	totals.HitAt10 += query.HitAt10
	totals.MAPAt10 += query.MAPAt10
	totals.MAPAt100 += query.MAPAt100
	totals.RecallAt10 += query.RecallAt10
	totals.RecallAt100 += query.RecallAt100
}

func averageRetrievalQuality(totals *RetrievalEvalQualityMetrics, evaluatedQueries int) {
	if evaluatedQueries <= 0 {
		return
	}
	denom := float64(evaluatedQueries)
	totals.NDCGAt10 /= denom
	totals.NDCGAt100 /= denom
	totals.MRRAt10 /= denom
	totals.PrecisionAt1 /= denom
	totals.PrecisionAt5 /= denom
	totals.PrecisionAt10 /= denom
	totals.HitAt1 /= denom
	totals.HitAt5 /= denom
	totals.HitAt10 /= denom
	totals.MAPAt10 /= denom
	totals.MAPAt100 /= denom
	totals.RecallAt10 /= denom
	totals.RecallAt100 /= denom
}

func precisionAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	if k <= 0 {
		return 0
	}
	hits := hitsAt(scores, rels, k)
	return float64(hits) / float64(k)
}

func hitAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	if hitsAt(scores, rels, k) > 0 {
		return 1
	}
	return 0
}

func averagePrecisionAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	if len(rels) == 0 || k <= 0 {
		return 0
	}
	limit := min(k, len(scores))
	hits := 0
	sumPrecision := 0.0
	for i := 0; i < limit; i++ {
		if rels[scores[i].ID] > 0 {
			hits++
			sumPrecision += float64(hits) / float64(i+1)
		}
	}
	denom := min(k, len(rels))
	if denom == 0 {
		return 0
	}
	return sumPrecision / float64(denom)
}

func recallAt(scores []retrievalScoredDoc, rels map[string]float64, k int) float64 {
	if len(rels) == 0 {
		return 0
	}
	return float64(hitsAt(scores, rels, k)) / float64(len(rels))
}

func hitsAt(scores []retrievalScoredDoc, rels map[string]float64, k int) int {
	hits := 0
	limit := min(k, len(scores))
	for i := 0; i < limit; i++ {
		if rels[scores[i].ID] > 0 {
			hits++
		}
	}
	return hits
}

func ratePerSecond(count float64, duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return count / duration.Seconds()
}
