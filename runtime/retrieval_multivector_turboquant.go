package eosruntime

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"m31labs.dev/turboquant"
)

const TurboQuantMultiVectorRetrievalEvalMetricsSchema = "manta.embedding_turboquant_multivector_retrieval_metrics.v1"
const TurboQuantMultiVectorRetrievalPerQuerySchema = "manta.embedding_turboquant_multivector_retrieval_per_query.v1"
const DefaultTurboQuantMultiVectorQuantizerSeed int64 = 0x4d756c7469566563
const TurboQuantMultiVectorAggregationMax = "max"

// TurboQuantMultiVectorRetrievalEvalMetrics compares parent-child dense
// max-child retrieval with direct TurboQuant child-vector scoring.
type TurboQuantMultiVectorRetrievalEvalMetrics struct {
	Schema        string                                      `json:"schema"`
	Dataset       string                                      `json:"dataset"`
	Artifact      string                                      `json:"artifact,omitempty"`
	Backend       string                                      `json:"backend,omitempty"`
	Inputs        TurboQuantMultiVectorRetrievalInputMetrics  `json:"inputs"`
	Config        TurboQuantMultiVectorRetrievalConfigMetrics `json:"config"`
	Dense         TurboQuantMultiVectorDenseMetrics           `json:"dense"`
	Rows          []TurboQuantMultiVectorBitMetrics           `json:"rows"`
	SkippedCounts RetrievalEvalSkippedCounts                  `json:"skipped_counts,omitempty"`
}

type TurboQuantMultiVectorRetrievalInputMetrics struct {
	CorpusPath               string  `json:"corpus_path,omitempty"`
	QueriesPath              string  `json:"queries_path,omitempty"`
	QrelsPath                string  `json:"qrels_path,omitempty"`
	DocVectorPath            string  `json:"doc_vector_path,omitempty"`
	QueryVectorPath          string  `json:"query_vector_path,omitempty"`
	Parents                  int     `json:"parents"`
	ParentCount              int     `json:"parent_count"`
	ChildVectors             int     `json:"child_vectors"`
	ChildCount               int     `json:"child_count"`
	AverageChildrenPerParent float64 `json:"average_children_per_parent"`
	AvgChildrenPerParent     float64 `json:"avg_children_per_parent"`
	MaxChildrenPerParent     int     `json:"max_children_per_parent"`
	Queries                  int     `json:"queries"`
	RelevantPairs            int     `json:"relevant_pairs"`
	ScoredChildPairs         int64   `json:"scored_child_pairs"`
}

type TurboQuantMultiVectorRetrievalConfigMetrics struct {
	TopK                    int     `json:"top_k"`
	MaxDocs                 int     `json:"max_docs,omitempty"`
	MaxQueries              int     `json:"max_queries,omitempty"`
	Bits                    []int   `json:"bits"`
	AllowMissingRelevant    bool    `json:"allow_missing_relevant"`
	QuantizerSeed           int64   `json:"quantizer_seed"`
	BaselineDim             int     `json:"baseline_dim"`
	Aggregation             string  `json:"aggregation"`
	ChildCountPenalty       float64 `json:"child_count_penalty"`
	ParentAggregation       string  `json:"parent_aggregation"`
	ParentChildCountPenalty float64 `json:"parent_child_count_penalty"`
}

type TurboQuantMultiVectorDenseMetrics struct {
	Quality                          RetrievalEvalQualityMetrics `json:"quality"`
	Aggregation                      string                      `json:"aggregation"`
	ChildCountPenalty                float64                     `json:"child_count_penalty"`
	BaselineDim                      int                         `json:"baseline_dim"`
	ParentCount                      int                         `json:"parent_count"`
	ChildCount                       int                         `json:"child_count"`
	AvgChildrenPerParent             float64                     `json:"avg_children_per_parent"`
	MaxChildrenPerParent             int                         `json:"max_children_per_parent"`
	DenseBaselineBytes               int64                       `json:"dense_baseline_bytes"`
	DenseBaselineTotalBytes          int64                       `json:"dense_baseline_total_bytes"`
	ChildVectorBytes                 int64                       `json:"child_vector_bytes"`
	QuantizedVectorBytes             int64                       `json:"quantized_vector_bytes"`
	DenseParentBytes                 int64                       `json:"dense_parent_bytes"`
	DenseChildBytes                  int64                       `json:"dense_child_bytes"`
	VectorsThatFitInOneDenseBaseline int64                       `json:"vectors_that_fit_in_one_dense_baseline"`
	StorageMultipleOfDenseBaseline   float64                     `json:"storage_multiple_of_dense_baseline"`
	ParentBudgetStorageMultiple      float64                     `json:"parent_budget_storage_multiple"`
	ScoreSeconds                     float64                     `json:"score_seconds"`
	ScoresPerSecond                  float64                     `json:"scores_per_second"`
	QueryLatency                     RetrievalEvalLatencyMetrics `json:"query_latency"`
}

type TurboQuantMultiVectorBitMetrics struct {
	Bits                             int                         `json:"bits"`
	Method                           string                      `json:"method"`
	Aggregation                      string                      `json:"aggregation"`
	ChildCountPenalty                float64                     `json:"child_count_penalty"`
	QuantizerSeed                    int64                       `json:"quantizer_seed"`
	Quality                          RetrievalEvalQualityMetrics `json:"quality"`
	NDCGAt10Delta                    float64                     `json:"ndcg_at_10_delta"`
	RecallAt100Delta                 float64                     `json:"recall_at_100_delta"`
	BaselineDim                      int                         `json:"baseline_dim"`
	ParentCount                      int                         `json:"parent_count"`
	ChildCount                       int                         `json:"child_count"`
	AvgChildrenPerParent             float64                     `json:"avg_children_per_parent"`
	MaxChildrenPerParent             int                         `json:"max_children_per_parent"`
	QuantizedVectorBytes             int64                       `json:"quantized_vector_bytes"`
	QuantizedChildBytes              int64                       `json:"quantized_child_bytes"`
	DenseBaselineBytes               int64                       `json:"dense_baseline_bytes"`
	DenseBaselineTotalBytes          int64                       `json:"dense_baseline_total_bytes"`
	DenseParentBytes                 int64                       `json:"dense_parent_bytes"`
	DenseChildBytes                  int64                       `json:"dense_child_bytes"`
	DenseChildCompression            float64                     `json:"dense_child_compression_ratio"`
	VectorsThatFitInOneDenseBaseline int64                       `json:"vectors_that_fit_in_one_dense_baseline"`
	StorageMultipleOfDenseBaseline   float64                     `json:"storage_multiple_of_dense_baseline"`
	ParentBudgetStorageMultiple      float64                     `json:"parent_budget_storage_multiple"`
	QuantizeSeconds                  float64                     `json:"quantize_seconds"`
	ScoreSeconds                     float64                     `json:"score_seconds"`
	ChildrenPerSecond                float64                     `json:"children_per_second"`
	ScoresPerSecond                  float64                     `json:"scores_per_second"`
	QueryLatency                     RetrievalEvalLatencyMetrics `json:"query_latency"`
	SkippedRelevantDocs              int                         `json:"skipped_relevant_docs,omitempty"`
	SkippedQueries                   int                         `json:"skipped_queries_without_relevant_docs,omitempty"`
}

type TurboQuantMultiVectorRetrievalPerQueryRow struct {
	Schema            string                        `json:"schema"`
	Dataset           string                        `json:"dataset"`
	QueryID           string                        `json:"query_id"`
	Method            string                        `json:"method"`
	Bits              int                           `json:"bits,omitempty"`
	ScoringSurface    string                        `json:"scoring_surface"`
	Aggregation       string                        `json:"aggregation"`
	ChildCountPenalty float64                       `json:"child_count_penalty"`
	QuantizerSeed     int64                         `json:"quantizer_seed,omitempty"`
	TopKLimit         int                           `json:"top_k_limit"`
	RelevantCount     int                           `json:"relevant_count"`
	FirstRelevantRank int                           `json:"first_relevant_rank"`
	Quality           RetrievalEvalQualityMetrics   `json:"quality"`
	TopK              []RetrievalEvalPerQueryTopDoc `json:"top_k"`
}

// EvaluateTurboQuantMultiVectorCacheRetrieval evaluates precomputed child
// document vectors and query vectors against parent-document qrels. Every child
// is scored, scores are aggregated by max child score per parent, and parent IDs
// are evaluated against BEIR-style qrels.
func EvaluateTurboQuantMultiVectorCacheRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int) (TurboQuantMultiVectorRetrievalEvalMetrics, error) {
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" || cfg.QrelsPath == "" {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("corpus, queries, and qrels paths are required")
	}
	if cfg.DocVectorPath == "" || cfg.QueryVectorPath == "" {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document and query vector paths are required")
	}
	if cfg.BackendName == "" {
		cfg.BackendName = "vectors"
	}
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}

	qrels, err := readBEIRQrels(cfg.QrelsPath)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	corpus, err := readBEIRCorpusWithRelevant(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	queries, skippedQueries, err := readBEIRQueries(cfg.QueriesPath, qrels, cfg.MaxQueries)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if len(corpus) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("no qrels queries found in queries file")
	}

	childVectors, missingParents, docDim, err := readRetrievalChildVectorCache(cfg.DocVectorPath, retrievalIDs(corpus))
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("read document child vectors: %w", err)
	}
	if len(childVectors) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document vector cache has no child vectors for the evaluated corpus")
	}
	queryVectors, missingQueryVectors, queryDim, err := readRetrievalVectorCache(cfg.QueryVectorPath, retrievalIDs(queries))
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("read query vectors: %w", err)
	}
	if len(queryVectors) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query vector cache has no vectors for qrels queries")
	}
	if docDim != queryDim {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("document child vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}

	metrics, err := evaluateTurboQuantMultiVectorRetrieval(ctx, cfg, bits, childVectors, queryVectors, qrels)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	metrics.Artifact = cfg.ArtifactPath
	metrics.Backend = cfg.BackendName
	metrics.Inputs.CorpusPath = cfg.CorpusPath
	metrics.Inputs.QueriesPath = cfg.QueriesPath
	metrics.Inputs.QrelsPath = cfg.QrelsPath
	metrics.Inputs.DocVectorPath = cfg.DocVectorPath
	metrics.Inputs.QueryVectorPath = cfg.QueryVectorPath
	metrics.SkippedCounts.QueriesWithoutText = skippedQueries
	metrics.SkippedCounts.QueriesWithoutVector = missingQueryVectors
	metrics.SkippedCounts.DocumentsWithoutVector = missingParents
	return metrics, nil
}

func evaluateTurboQuantMultiVectorRetrieval(ctx context.Context, cfg RetrievalEvalConfig, bits []int, children []retrievalChildVectorRecord, queries []retrievalVectorRecord, qrels retrievalQrels) (TurboQuantMultiVectorRetrievalEvalMetrics, error) {
	perQueryTopK := cfg.TopK
	cfg = normalizeRetrievalEvalConfig(cfg)
	if cfg.PerQueryTopK > 0 {
		perQueryTopK = cfg.PerQueryTopK
	}
	if perQueryTopK <= 0 {
		perQueryTopK = cfg.TopK
	}
	bits = normalizeTurboQuantRetrievalBits(bits)
	if err := validateTurboQuantRetrievalBits(bits); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if err := validateTurboQuantMultiVectorAggregationConfig(cfg.MultiVectorAggregation, cfg.MultiVectorChildCountPenalty); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if len(children) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vectors are empty")
	}
	if len(queries) == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query vectors are empty")
	}
	if cfg.QuantizerSeed == 0 {
		cfg.QuantizerSeed = DefaultTurboQuantMultiVectorQuantizerSeed
	}
	dim := len(children[0].Vector)
	if dim == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vector dimension is zero")
	}
	baselineDim := cfg.BaselineDim
	if baselineDim == 0 {
		baselineDim = dim
	}
	if baselineDim < 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("baseline dim must be positive or zero to use child vector dimension")
	}
	for _, child := range children {
		if child.ParentID == "" {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child vector has empty parent id")
		}
		if len(child.Vector) != dim {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("parent %q child %q vector dimension = %d, want %d", child.ParentID, child.ChildID, len(child.Vector), dim)
		}
	}
	for _, query := range queries {
		if len(query.Vector) != dim {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("query %q vector dimension = %d, want %d", query.ID, len(query.Vector), dim)
		}
	}

	parentCount := countRetrievalParents(children)
	maxChildrenPerParent := maxRetrievalChildrenPerParent(children)
	avgChildrenPerParent := ratioFloat64(float64(len(children)), float64(parentCount))
	if !cfg.AllowMissingRelevant {
		missingRelevantDocs, missingRelevantQueries := countMissingRelevantParents(queries, children, qrels)
		if missingRelevantDocs > 0 {
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("child-vector cache is missing %d qrels-relevant parent documents across %d queries; rerun with --allow-missing-relevant only for diagnostic smoke metrics", missingRelevantDocs, missingRelevantQueries)
		}
	}
	scoreStart := time.Now()
	denseQuality, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, denseLatency := computeDenseMultiVectorRetrievalQuality(ctx, queries, children, qrels, cfg.TopK, cfg.MultiVectorAggregation, cfg.MultiVectorChildCountPenalty)
	denseScoreDuration := time.Since(scoreStart)
	if evaluatedQueries == 0 {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, fmt.Errorf("no queries had relevant parent documents in the evaluated vector cache")
	}
	perQueryWriter, err := newTurboQuantMultiVectorPerQueryWriter(cfg.PerQueryJSONLPath)
	if err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	if err := writeDenseMultiVectorPerQueryRows(ctx, perQueryWriter, cfg.DatasetName, perQueryTopK, queries, children, qrels, cfg.MultiVectorAggregation, cfg.MultiVectorChildCountPenalty); err != nil {
		_ = perQueryWriter.Close()
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}

	scoredPairs := int64(evaluatedQueries) * int64(len(children))
	denseBaselineBytes := int64(baselineDim * 4)
	denseParentBytes := int64(parentCount) * denseBaselineBytes
	denseChildBytes := int64(len(children) * dim * 4)
	denseChildVectorBytes := int64(dim * 4)
	denseStorageMultiple := ratioFloat64(float64(denseChildBytes), float64(denseParentBytes))
	out := TurboQuantMultiVectorRetrievalEvalMetrics{
		Schema:  TurboQuantMultiVectorRetrievalEvalMetricsSchema,
		Dataset: cfg.DatasetName,
		Inputs: TurboQuantMultiVectorRetrievalInputMetrics{
			Parents:                  parentCount,
			ParentCount:              parentCount,
			ChildVectors:             len(children),
			ChildCount:               len(children),
			AverageChildrenPerParent: avgChildrenPerParent,
			AvgChildrenPerParent:     avgChildrenPerParent,
			MaxChildrenPerParent:     maxChildrenPerParent,
			Queries:                  evaluatedQueries,
			RelevantPairs:            relevantPairs,
			ScoredChildPairs:         scoredPairs,
		},
		Config: TurboQuantMultiVectorRetrievalConfigMetrics{
			TopK:                    cfg.TopK,
			MaxDocs:                 cfg.MaxDocs,
			MaxQueries:              cfg.MaxQueries,
			Bits:                    append([]int(nil), bits...),
			AllowMissingRelevant:    cfg.AllowMissingRelevant,
			QuantizerSeed:           cfg.QuantizerSeed,
			BaselineDim:             baselineDim,
			Aggregation:             cfg.MultiVectorAggregation,
			ChildCountPenalty:       cfg.MultiVectorChildCountPenalty,
			ParentAggregation:       cfg.MultiVectorAggregation,
			ParentChildCountPenalty: cfg.MultiVectorChildCountPenalty,
		},
		Dense: TurboQuantMultiVectorDenseMetrics{
			Quality:                          denseQuality,
			Aggregation:                      cfg.MultiVectorAggregation,
			ChildCountPenalty:                cfg.MultiVectorChildCountPenalty,
			BaselineDim:                      baselineDim,
			ParentCount:                      parentCount,
			ChildCount:                       len(children),
			AvgChildrenPerParent:             avgChildrenPerParent,
			MaxChildrenPerParent:             maxChildrenPerParent,
			DenseBaselineBytes:               denseBaselineBytes,
			DenseBaselineTotalBytes:          denseParentBytes,
			ChildVectorBytes:                 denseChildVectorBytes,
			QuantizedVectorBytes:             denseChildVectorBytes,
			DenseParentBytes:                 denseParentBytes,
			DenseChildBytes:                  denseChildBytes,
			VectorsThatFitInOneDenseBaseline: denseBaselineBytes / denseChildVectorBytes,
			StorageMultipleOfDenseBaseline:   denseStorageMultiple,
			ParentBudgetStorageMultiple:      denseStorageMultiple,
			ScoreSeconds:                     denseScoreDuration.Seconds(),
			ScoresPerSecond:                  ratePerSecond(float64(scoredPairs), denseScoreDuration),
			QueryLatency:                     denseLatency,
		},
		Rows: make([]TurboQuantMultiVectorBitMetrics, 0, len(bits)),
	}
	for _, bitWidth := range bits {
		row, err := evaluateTurboQuantMultiVectorBits(ctx, dim, baselineDim, bitWidth, cfg.TopK, perQueryTopK, cfg.QuantizerSeed, cfg.MultiVectorAggregation, cfg.MultiVectorChildCountPenalty, children, queries, qrels, denseQuality, parentCount, maxChildrenPerParent, denseBaselineBytes, denseParentBytes, denseChildBytes, scoredPairs, perQueryWriter, cfg.DatasetName)
		if err != nil {
			_ = perQueryWriter.Close()
			return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
		}
		row.SkippedRelevantDocs = skippedRelevantDocs
		row.SkippedQueries = skippedNoRelevant
		out.Rows = append(out.Rows, row)
	}
	if err := perQueryWriter.Close(); err != nil {
		return TurboQuantMultiVectorRetrievalEvalMetrics{}, err
	}
	return out, nil
}

type turboQuantMultiVectorChild struct {
	ParentID string
	ChildID  string
	Vector   turboquant.IPQuantized
}

type turboQuantMultiVectorPerQueryWriter struct {
	file   *os.File
	writer *bufio.Writer
}

func newTurboQuantMultiVectorPerQueryWriter(path string) (*turboQuantMultiVectorPerQueryWriter, error) {
	if path == "" {
		return &turboQuantMultiVectorPerQueryWriter{}, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create multivector TurboQuant per-query JSONL: %w", err)
	}
	return &turboQuantMultiVectorPerQueryWriter{file: file, writer: bufio.NewWriter(file)}, nil
}

func (w *turboQuantMultiVectorPerQueryWriter) Write(row TurboQuantMultiVectorRetrievalPerQueryRow) error {
	if w == nil || w.writer == nil {
		return nil
	}
	data, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("marshal multivector TurboQuant per-query row: %w", err)
	}
	if _, err := w.writer.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write multivector TurboQuant per-query JSONL: %w", err)
	}
	return nil
}

func (w *turboQuantMultiVectorPerQueryWriter) Close() error {
	if w == nil || w.file == nil {
		return nil
	}
	flushErr := w.writer.Flush()
	closeErr := w.file.Close()
	w.file = nil
	w.writer = nil
	if flushErr != nil {
		return fmt.Errorf("flush multivector TurboQuant per-query JSONL: %w", flushErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close multivector TurboQuant per-query JSONL: %w", closeErr)
	}
	return nil
}

func writeDenseMultiVectorPerQueryRows(ctx context.Context, writer *turboQuantMultiVectorPerQueryWriter, datasetName string, topK int, queries []retrievalVectorRecord, children []retrievalChildVectorRecord, qrels retrievalQrels, aggregation string, childCountPenalty float64) error {
	if writer == nil || writer.writer == nil {
		return nil
	}
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
	}
	evalTopK, outputTopK := multiVectorEvalAndOutputTopK(topK, len(parentIDSet))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return err
		}
		filteredRels := filterParentRels(qrels[query.ID], parentIDSet, new(int))
		if len(filteredRels) == 0 {
			continue
		}
		scores := topDenseMultiVectorParentScores(query.Vector, children, evalTopK, aggregation, childCountPenalty)
		method := multiVectorDenseMethod(aggregation, childCountPenalty)
		row := buildTurboQuantMultiVectorPerQueryRow(datasetName, query.ID, method, 0, multiVectorDenseScoringSurface(aggregation, childCountPenalty), aggregation, childCountPenalty, 0, outputTopK, scores, filteredRels, nil)
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func writeTurboQuantMultiVectorPerQueryRows(ctx context.Context, writer *turboQuantMultiVectorPerQueryWriter, datasetName string, bitWidth, topK int, quantizerSeed int64, q *turboquant.IPQuantizer, denseChildren []retrievalChildVectorRecord, compactChildren []turboQuantMultiVectorChild, queries []retrievalVectorRecord, qrels retrievalQrels, aggregation string, childCountPenalty float64) error {
	if writer == nil || writer.writer == nil {
		return nil
	}
	parentIDSet := make(map[string]bool, len(compactChildren))
	for _, child := range compactChildren {
		parentIDSet[child.ParentID] = true
	}
	evalTopK, outputTopK := multiVectorEvalAndOutputTopK(topK, len(parentIDSet))
	method := multiVectorTurboQuantMethod(bitWidth, aggregation, childCountPenalty)
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return err
		}
		filteredRels := filterParentRels(qrels[query.ID], parentIDSet, new(int))
		if len(filteredRels) == 0 {
			continue
		}
		denseScores := topDenseMultiVectorParentScores(query.Vector, denseChildren, evalTopK, aggregation, childCountPenalty)
		denseRanks := multiVectorRankEvidence(denseScores)
		prepared := q.PrepareQuery(query.Vector)
		compactScores := topTurboQuantMultiVectorParentScores(q, prepared, compactChildren, evalTopK, aggregation, childCountPenalty)
		row := buildTurboQuantMultiVectorPerQueryRow(datasetName, query.ID, method, bitWidth, multiVectorTurboQuantScoringSurface(aggregation, childCountPenalty), aggregation, childCountPenalty, quantizerSeed, outputTopK, compactScores, filteredRels, denseRanks)
		if err := writer.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func buildTurboQuantMultiVectorPerQueryRow(datasetName, queryID, method string, bitWidth int, surface, aggregation string, childCountPenalty float64, quantizerSeed int64, outputTopK int, scores []retrievalScoredDoc, rels map[string]float64, denseRanks map[string]retrievalScoredDoc) TurboQuantMultiVectorRetrievalPerQueryRow {
	row := TurboQuantMultiVectorRetrievalPerQueryRow{
		Schema:            TurboQuantMultiVectorRetrievalPerQuerySchema,
		Dataset:           datasetName,
		QueryID:           queryID,
		Method:            method,
		Bits:              bitWidth,
		ScoringSurface:    surface,
		Aggregation:       aggregation,
		ChildCountPenalty: childCountPenalty,
		QuantizerSeed:     quantizerSeed,
		TopKLimit:         outputTopK,
		RelevantCount:     len(rels),
		FirstRelevantRank: firstRelevantRank(scores, rels),
		Quality:           retrievalQualityForQuery(scores, rels),
		TopK:              make([]RetrievalEvalPerQueryTopDoc, 0, min(outputTopK, len(scores))),
	}
	if outputTopK <= 0 || outputTopK > len(scores) {
		outputTopK = len(scores)
	}
	compactRanks := multiVectorRankEvidence(scores)
	for i := 0; i < outputTopK; i++ {
		score := scores[i]
		topDoc := RetrievalEvalPerQueryTopDoc{
			Rank:       i + 1,
			DocID:      score.ID,
			Score:      score.Score,
			Relevance:  rels[score.ID],
			ChildID:    score.ChildID,
			ChildScore: score.ChildScore,
		}
		if denseRanks != nil {
			if dense, ok := denseRanks[score.ID]; ok {
				topDoc.DenseRank = optionalPositiveInt(dense.DenseRank)
				topDoc.DenseScore = dense.DenseScore
				topDoc.DenseChildID = dense.ChildID
				topDoc.DenseChildScore = dense.ChildScore
			}
			if compact, ok := compactRanks[score.ID]; ok {
				topDoc.CompactRank = optionalPositiveInt(compact.DenseRank)
				topDoc.CompactScore = compact.DenseScore
				topDoc.CompactChildID = compact.ChildID
				topDoc.CompactChildScore = compact.ChildScore
			}
		}
		row.TopK = append(row.TopK, topDoc)
	}
	return row
}

func multiVectorEvalAndOutputTopK(requestedTopK, parentCount int) (int, int) {
	outputTopK := requestedTopK
	if outputTopK <= 0 || outputTopK > parentCount {
		outputTopK = parentCount
	}
	evalTopK := outputTopK
	if evalTopK < 100 {
		evalTopK = 100
	}
	if evalTopK > parentCount {
		evalTopK = parentCount
	}
	return evalTopK, outputTopK
}

func multiVectorRankEvidence(scores []retrievalScoredDoc) map[string]retrievalScoredDoc {
	out := make(map[string]retrievalScoredDoc, len(scores))
	for i, score := range scores {
		score.DenseRank = i + 1
		value := float64(score.Score)
		score.DenseScore = &value
		out[score.ID] = score
	}
	return out
}

func evaluateTurboQuantMultiVectorBits(ctx context.Context, dim, baselineDim, bitWidth, topK, perQueryTopK int, quantizerSeed int64, aggregation string, childCountPenalty float64, children []retrievalChildVectorRecord, queries []retrievalVectorRecord, qrels retrievalQrels, denseQuality RetrievalEvalQualityMetrics, parentCount, maxChildrenPerParent int, denseBaselineBytes, denseParentBytes, denseChildBytes, scoredPairs int64, perQueryWriter *turboQuantMultiVectorPerQueryWriter, datasetName string) (TurboQuantMultiVectorBitMetrics, error) {
	q := turboquant.NewIPWithSeed(dim, bitWidth, quantizerSeed)
	quantizeStart := time.Now()
	qchildren := make([]turboQuantMultiVectorChild, len(children))
	quantizedVectorBytes := turboquantVectorBytes(dim, bitWidth)
	var quantizedBytes int64
	for i, child := range children {
		if err := ctx.Err(); err != nil {
			return TurboQuantMultiVectorBitMetrics{}, err
		}
		qx := q.Quantize(child.Vector)
		qchildren[i] = turboQuantMultiVectorChild{ParentID: child.ParentID, ChildID: child.ChildID, Vector: qx}
		quantizedBytes += quantizedVectorBytes
	}
	quantizeDuration := time.Since(quantizeStart)

	scoreStart := time.Now()
	quality, evaluatedQueries, _, skippedRelevantDocs, skippedNoRelevant, queryLatency := computeTurboQuantMultiVectorRetrievalQuality(ctx, q, queries, qchildren, qrels, topK, aggregation, childCountPenalty)
	if evaluatedQueries == 0 {
		return TurboQuantMultiVectorBitMetrics{}, fmt.Errorf("no queries had relevant parent documents in the evaluated vector cache")
	}
	if err := writeTurboQuantMultiVectorPerQueryRows(ctx, perQueryWriter, datasetName, bitWidth, perQueryTopK, q.Seed(), q, children, qchildren, queries, qrels, aggregation, childCountPenalty); err != nil {
		return TurboQuantMultiVectorBitMetrics{}, err
	}
	scoreDuration := time.Since(scoreStart)
	return TurboQuantMultiVectorBitMetrics{
		Bits:                             bitWidth,
		Method:                           multiVectorTurboQuantMethod(bitWidth, aggregation, childCountPenalty),
		Aggregation:                      aggregation,
		ChildCountPenalty:                childCountPenalty,
		QuantizerSeed:                    q.Seed(),
		Quality:                          quality,
		NDCGAt10Delta:                    quality.NDCGAt10 - denseQuality.NDCGAt10,
		RecallAt100Delta:                 quality.RecallAt100 - denseQuality.RecallAt100,
		BaselineDim:                      baselineDim,
		ParentCount:                      parentCount,
		ChildCount:                       len(children),
		AvgChildrenPerParent:             ratioFloat64(float64(len(children)), float64(parentCount)),
		MaxChildrenPerParent:             maxChildrenPerParent,
		QuantizedVectorBytes:             quantizedVectorBytes,
		QuantizedChildBytes:              quantizedBytes,
		DenseBaselineBytes:               denseBaselineBytes,
		DenseBaselineTotalBytes:          denseParentBytes,
		DenseParentBytes:                 denseParentBytes,
		DenseChildBytes:                  denseChildBytes,
		DenseChildCompression:            ratioFloat64(float64(denseChildBytes), float64(quantizedBytes)),
		VectorsThatFitInOneDenseBaseline: denseBaselineBytes / quantizedVectorBytes,
		StorageMultipleOfDenseBaseline:   ratioFloat64(float64(quantizedBytes), float64(denseParentBytes)),
		ParentBudgetStorageMultiple:      ratioFloat64(float64(quantizedBytes), float64(denseParentBytes)),
		QuantizeSeconds:                  quantizeDuration.Seconds(),
		ScoreSeconds:                     scoreDuration.Seconds(),
		ChildrenPerSecond:                ratePerSecond(float64(len(children)), quantizeDuration),
		ScoresPerSecond:                  ratePerSecond(float64(scoredPairs), scoreDuration),
		QueryLatency:                     queryLatency,
		SkippedRelevantDocs:              skippedRelevantDocs,
		SkippedQueries:                   skippedNoRelevant,
	}, nil
}

func countMissingRelevantParents(queries []retrievalVectorRecord, children []retrievalChildVectorRecord, qrels retrievalQrels) (int, int) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
	}
	missingDocs := 0
	missingQueries := 0
	for _, query := range queries {
		queryMissing := 0
		for parentID := range qrels[query.ID] {
			if !parentIDSet[parentID] {
				queryMissing++
			}
		}
		if queryMissing > 0 {
			missingQueries++
			missingDocs += queryMissing
		}
	}
	return missingDocs, missingQueries
}

func countRetrievalParents(children []retrievalChildVectorRecord) int {
	seen := make(map[string]bool, len(children))
	for _, child := range children {
		seen[child.ParentID] = true
	}
	return len(seen)
}

func maxRetrievalChildrenPerParent(children []retrievalChildVectorRecord) int {
	counts := make(map[string]int, len(children))
	maxCount := 0
	for _, child := range children {
		counts[child.ParentID]++
		if counts[child.ParentID] > maxCount {
			maxCount = counts[child.ParentID]
		}
	}
	return maxCount
}

func computeDenseMultiVectorRetrievalQuality(ctx context.Context, queries []retrievalVectorRecord, children []retrievalChildVectorRecord, qrels retrievalQrels, topK int, aggregation string, childCountPenalty float64) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
	}
	if topK < 100 {
		topK = 100
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := filterParentRels(rels, parentIDSet, &skippedRelevantDocs)
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		scores := topDenseMultiVectorParentScores(query.Vector, children, topK, aggregation, childCountPenalty)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func computeTurboQuantMultiVectorRetrievalQuality(ctx context.Context, q *turboquant.IPQuantizer, queries []retrievalVectorRecord, children []turboQuantMultiVectorChild, qrels retrievalQrels, topK int, aggregation string, childCountPenalty float64) (RetrievalEvalQualityMetrics, int, int, int, int, RetrievalEvalLatencyMetrics) {
	parentIDSet := make(map[string]bool, len(children))
	for _, child := range children {
		parentIDSet[child.ParentID] = true
	}
	if topK < 100 {
		topK = 100
	}
	var totals RetrievalEvalQualityMetrics
	evaluatedQueries := 0
	relevantPairs := 0
	skippedRelevantDocs := 0
	skippedNoRelevant := 0
	latencies := make([]time.Duration, 0, len(queries))
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			break
		}
		rels := qrels[query.ID]
		filteredRels := filterParentRels(rels, parentIDSet, &skippedRelevantDocs)
		if len(filteredRels) == 0 {
			skippedNoRelevant++
			continue
		}
		queryStart := time.Now()
		prepared := q.PrepareQuery(query.Vector)
		scores := topTurboQuantMultiVectorParentScores(q, prepared, children, topK, aggregation, childCountPenalty)
		latencies = append(latencies, time.Since(queryStart))
		evaluatedQueries++
		relevantPairs += len(filteredRels)
		addRetrievalQuality(&totals, scores, filteredRels)
	}
	averageRetrievalQuality(&totals, evaluatedQueries)
	return totals, evaluatedQueries, relevantPairs, skippedRelevantDocs, skippedNoRelevant, summarizeRetrievalEvalLatencies(latencies)
}

func filterParentRels(rels map[string]float64, parentIDSet map[string]bool, skipped *int) map[string]float64 {
	filtered := make(map[string]float64, len(rels))
	for parentID, rel := range rels {
		if parentIDSet[parentID] {
			filtered[parentID] = rel
		} else {
			(*skipped)++
		}
	}
	return filtered
}

func topDenseMultiVectorParentScores(query []float32, children []retrievalChildVectorRecord, topK int, aggregation string, childCountPenalty float64) []retrievalScoredDoc {
	parents := make(map[string]*multiVectorParentScoreAccumulator, len(children))
	for _, child := range children {
		score := dotRetrievalVectors(query, child.Vector)
		acc := parents[child.ParentID]
		if acc == nil {
			acc = &multiVectorParentScoreAccumulator{parentID: child.ParentID}
			parents[child.ParentID] = acc
		}
		acc.add(child.ChildID, score)
	}
	return topParentScoresFromMap(aggregateMultiVectorParentScores(parents, aggregation, childCountPenalty), topK)
}

func topTurboQuantMultiVectorParentScores(q *turboquant.IPQuantizer, prepared turboquant.PreparedQuery, children []turboQuantMultiVectorChild, topK int, aggregation string, childCountPenalty float64) []retrievalScoredDoc {
	parents := make(map[string]*multiVectorParentScoreAccumulator, len(children))
	for _, child := range children {
		score := q.InnerProductPrepared(child.Vector, prepared)
		acc := parents[child.ParentID]
		if acc == nil {
			acc = &multiVectorParentScoreAccumulator{parentID: child.ParentID}
			parents[child.ParentID] = acc
		}
		acc.add(child.ChildID, score)
	}
	return topParentScoresFromMap(aggregateMultiVectorParentScores(parents, aggregation, childCountPenalty), topK)
}

type multiVectorParentScoreAccumulator struct {
	parentID       string
	count          int
	topScores      []float32
	bestChildID    string
	bestChildScore float32
	hasBest        bool
}

func (a *multiVectorParentScoreAccumulator) add(childID string, score float32) {
	a.count++
	if !a.hasBest || score > a.bestChildScore || (score == a.bestChildScore && childID < a.bestChildID) {
		a.bestChildID = childID
		a.bestChildScore = score
		a.hasBest = true
	}
	a.topScores = append(a.topScores, score)
	for i := len(a.topScores) - 1; i > 0 && a.topScores[i] > a.topScores[i-1]; i-- {
		a.topScores[i], a.topScores[i-1] = a.topScores[i-1], a.topScores[i]
	}
	if len(a.topScores) > 5 {
		a.topScores = a.topScores[:5]
	}
}

func aggregateMultiVectorParentScores(parents map[string]*multiVectorParentScoreAccumulator, aggregation string, childCountPenalty float64) map[string]retrievalScoredDoc {
	best := make(map[string]retrievalScoredDoc, len(parents))
	n := multiVectorAggregationTopN(aggregation)
	for parentID, acc := range parents {
		available := n
		if available <= 0 || available > len(acc.topScores) {
			available = len(acc.topScores)
		}
		sum := float64(0)
		for i := 0; i < available; i++ {
			sum += float64(acc.topScores[i])
		}
		score := sum / float64(available)
		if childCountPenalty != 0 {
			score -= childCountPenalty * math.Log1p(float64(acc.count))
		}
		childScore := float64(acc.bestChildScore)
		best[parentID] = retrievalScoredDoc{
			ID:         parentID,
			Score:      float32(score),
			ChildID:    acc.bestChildID,
			ChildScore: &childScore,
		}
	}
	return best
}

func multiVectorChildScoreBetter(a, b retrievalScoredDoc) bool {
	if a.Score > b.Score {
		return true
	}
	if a.Score < b.Score {
		return false
	}
	if a.ChildID != b.ChildID {
		return a.ChildID < b.ChildID
	}
	return a.ID < b.ID
}

func validateTurboQuantMultiVectorAggregationConfig(aggregation string, childCountPenalty float64) error {
	if childCountPenalty < 0 {
		return fmt.Errorf("child-count-penalty must be non-negative")
	}
	switch aggregation {
	case TurboQuantMultiVectorAggregationMax, "top2-mean", "top3-mean", "top5-mean":
		return nil
	default:
		return fmt.Errorf("aggregation must be one of max, top2-mean, top3-mean, top5-mean")
	}
}

func multiVectorAggregationTopN(aggregation string) int {
	switch aggregation {
	case "top2-mean":
		return 2
	case "top3-mean":
		return 3
	case "top5-mean":
		return 5
	default:
		return 1
	}
}

func multiVectorUsesDefaultAggregation(aggregation string, childCountPenalty float64) bool {
	return aggregation == TurboQuantMultiVectorAggregationMax && childCountPenalty == 0
}

func multiVectorDenseMethod(aggregation string, childCountPenalty float64) string {
	if multiVectorUsesDefaultAggregation(aggregation, childCountPenalty) {
		return "float32_child_max"
	}
	return "float32_child_" + multiVectorAggregationLabel(aggregation, childCountPenalty)
}

func multiVectorDenseScoringSurface(aggregation string, childCountPenalty float64) string {
	if multiVectorUsesDefaultAggregation(aggregation, childCountPenalty) {
		return "dense_child_max"
	}
	return "dense_child_" + multiVectorAggregationLabel(aggregation, childCountPenalty)
}

func multiVectorTurboQuantMethod(bitWidth int, aggregation string, childCountPenalty float64) string {
	if multiVectorUsesDefaultAggregation(aggregation, childCountPenalty) {
		return fmt.Sprintf("turboquant_ip_b%d_child_max", bitWidth)
	}
	return fmt.Sprintf("turboquant_ip_b%d_child_%s", bitWidth, multiVectorAggregationLabel(aggregation, childCountPenalty))
}

func multiVectorTurboQuantScoringSurface(aggregation string, childCountPenalty float64) string {
	if multiVectorUsesDefaultAggregation(aggregation, childCountPenalty) {
		return "turboquant_ip_prepared_child_max"
	}
	return "turboquant_ip_prepared_child_" + multiVectorAggregationLabel(aggregation, childCountPenalty)
}

func multiVectorAggregationLabel(aggregation string, childCountPenalty float64) string {
	label := aggregation
	if childCountPenalty != 0 {
		label += fmt.Sprintf("_penalty_%g", childCountPenalty)
	}
	return label
}

func topParentScoresFromMap(best map[string]retrievalScoredDoc, topK int) []retrievalScoredDoc {
	if topK <= 0 || topK > len(best) {
		topK = len(best)
	}
	h := make(retrievalScoreHeap, 0, topK)
	for _, score := range best {
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
	slicesSortRetrievalScores(scores)
	return scores
}
