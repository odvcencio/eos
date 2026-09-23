package eosruntime

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
)

// EmbeddingScoreSpectrumSource is a random-access, closeable source of
// score-spectrum rows. File-backed implementations keep only row locations
// and compact metadata in memory; the row payload is decoded when Example is
// called.
//
// CandidateCount is deliberately a cheap metadata lookup. An out-of-range
// index returns zero; Example should be used when the caller needs a detailed
// error for an invalid index.
type EmbeddingScoreSpectrumSource interface {
	Len() int
	CandidateCount(index int) int
	MaxCandidateCount() int
	SourcePolicy() EmbeddingScoreSpectrumPolicy
	Example(index int) (EmbeddingScoreSpectrumExample, error)
	Close() error
}

type EmbeddingScoreSpectrumTopKMetadataSource interface {
	TopKEligiblePairCount(index int, negativeMask string) int
	TopKRecallEligiblePairCount(index int, negativeMask string) int
}

type EmbeddingScoreSpectrumObjectiveMetadataSource interface {
	BaseLossDisabled(index int) bool
}

// EmbeddingScoreSpectrumExampleSource is a descriptive alias for
// EmbeddingScoreSpectrumSource used by callers that want to emphasize that
// Example returns a row rather than a stream iterator.
type EmbeddingScoreSpectrumExampleSource = EmbeddingScoreSpectrumSource

// ScoreSpectrumSource is a short alias for EmbeddingScoreSpectrumSource.
type ScoreSpectrumSource = EmbeddingScoreSpectrumSource

// EmbeddingScoreSpectrumSourceOptions is an alias for the existing read
// options so source callers can use a source-specific name without creating a
// second, subtly different gate contract.
type EmbeddingScoreSpectrumSourceOptions = EmbeddingScoreSpectrumReadOptions

// OpenEmbeddingTextScoreSpectrumSource opens and indexes a text JSONL
// score-spectrum dataset. Text rows are tokenized one at a time by Example;
// tokenizer results are cached only for the duration of that row.
func OpenEmbeddingTextScoreSpectrumSource(path string, tokenizer *BPETokenizer, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("nil tokenizer")
	}
	allowResearch := scoreSpectrumAllowResearch(opts)
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	source := &EmbeddingTextScoreSpectrumSource{
		file:          f,
		path:          path,
		tokenizer:     tokenizer,
		allowResearch: allowResearch,
	}
	if err := source.index(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return source, nil
}

// NewEmbeddingTextScoreSpectrumSource is an alias constructor for callers
// that use New-style naming for resource-backed sources.
func NewEmbeddingTextScoreSpectrumSource(path string, tokenizer *BPETokenizer, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	return OpenEmbeddingTextScoreSpectrumSource(path, tokenizer, opts...)
}

// OpenEmbeddingScoreSpectrumSource opens and indexes a tokenized JSONL
// score-spectrum dataset. Each tokenized row is decoded only when Example is
// called.
func OpenEmbeddingScoreSpectrumSource(path string, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	allowResearch := scoreSpectrumAllowResearch(opts)
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	source := &EmbeddingTokenizedScoreSpectrumSource{
		file:          f,
		path:          path,
		allowResearch: allowResearch,
	}
	if err := source.index(); err != nil {
		_ = f.Close()
		return nil, err
	}
	return source, nil
}

// NewEmbeddingScoreSpectrumSource is an alias constructor for tokenized
// score-spectrum JSONL sources.
func NewEmbeddingScoreSpectrumSource(path string, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	return OpenEmbeddingScoreSpectrumSource(path, opts...)
}

// NewEmbeddingScoreSpectrumSliceSource adapts an in-memory tokenized dataset
// to EmbeddingScoreSpectrumSource. It validates and deep-copies rows so that
// callers may mutate their input after construction without changing the
// source or rows returned by Example.
func NewEmbeddingScoreSpectrumSliceSource(examples []EmbeddingScoreSpectrumExample, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	if len(examples) == 0 {
		return nil, fmt.Errorf("score-spectrum dataset is empty")
	}
	allowResearch := scoreSpectrumAllowResearch(opts)
	source := &EmbeddingScoreSpectrumSliceSource{
		examples: make([]EmbeddingScoreSpectrumExample, 0, len(examples)),
		counts:   make([]int, 0, len(examples)),
	}
	acc := newScoreSpectrumPolicyAccumulator()
	for i, example := range examples {
		record, err := newEmbeddingScoreSpectrumRecord(example, allowResearch)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", i, err)
		}
		clean, err := record.example(allowResearch)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", i, err)
		}
		source.examples = append(source.examples, clean)
		source.counts = append(source.counts, len(clean.CandidateTokens))
		acc.add(clean.ReleaseTrainAllowed, clean.CommercialUseAllowed, clean.TrainAllowedForResearch, clean.SourceArtifactHash)
	}
	source.policy = acc.policy()
	source.maxCandidates = maxScoreSpectrumCandidateCount(source.counts)
	return source, nil
}

// NewEmbeddingScoreSpectrumExamplesSource is a compatibility alias for the
// slice-backed adapter.
func NewEmbeddingScoreSpectrumExamplesSource(examples []EmbeddingScoreSpectrumExample, opts ...EmbeddingScoreSpectrumReadOptions) (EmbeddingScoreSpectrumSource, error) {
	return NewEmbeddingScoreSpectrumSliceSource(examples, opts...)
}

// EmbeddingTextScoreSpectrumSource is a lazily decoded text JSONL source.
type EmbeddingTextScoreSpectrumSource struct {
	mu            sync.RWMutex
	file          *os.File
	path          string
	tokenizer     *BPETokenizer
	allowResearch bool
	rows          []embeddingScoreSpectrumSourceRow
	policy        EmbeddingScoreSpectrumPolicy
	maxCandidates int
	closed        bool
}

// EmbeddingTokenizedScoreSpectrumSource is a lazily decoded tokenized JSONL
// source.
type EmbeddingTokenizedScoreSpectrumSource struct {
	mu            sync.RWMutex
	file          *os.File
	path          string
	allowResearch bool
	rows          []embeddingScoreSpectrumSourceRow
	policy        EmbeddingScoreSpectrumPolicy
	maxCandidates int
	closed        bool
}

// EmbeddingScoreSpectrumSliceSource is an in-memory source useful for tests
// and for trainer call sites that already own tokenized rows.
type EmbeddingScoreSpectrumSliceSource struct {
	mu            sync.RWMutex
	examples      []EmbeddingScoreSpectrumExample
	counts        []int
	policy        EmbeddingScoreSpectrumPolicy
	maxCandidates int
	closed        bool
}

var (
	_ EmbeddingScoreSpectrumSource = (*EmbeddingTextScoreSpectrumSource)(nil)
	_ EmbeddingScoreSpectrumSource = (*EmbeddingTokenizedScoreSpectrumSource)(nil)
	_ EmbeddingScoreSpectrumSource = (*EmbeddingScoreSpectrumSliceSource)(nil)
)

var embeddingScoreSpectrumSourceKnownFields = map[string]struct{}{
	"row_id":                             {},
	"source":                             {},
	"query_tokens":                       {},
	"query_mask":                         {},
	"candidate_ids":                      {},
	"candidate_sources":                  {},
	"qrel_gains":                         {},
	"candidate_tokens":                   {},
	"candidate_masks":                    {},
	"positive_indexes":                   {},
	"selected_positive_index":            {},
	"hard_negative_eligible":             {},
	"target_probabilities":               {},
	"base_loss_weight":                   {},
	"hard_loss_weight":                   {},
	"soft_loss_weight":                   {},
	"recovery_loss_weight":               {},
	"turboquant_topk_loss_weight":        {},
	"turboquant_topk_recall_loss_weight": {},
	"train_policy":                       {},
	"legal_gates":                        {},
	"release_train_allowed":              {},
	"commercial_use_allowed":             {},
	"train_allowed_for_research":         {},
	"source_artifact_hash":               {},
	"ExtraFields":                        {},
}

var errEmbeddingScoreSpectrumSourceRecordTooLarge = errors.New("score-spectrum JSONL record is too large")

// embeddingScoreSpectrumSourceRow is intentionally compact. It contains no
// JSON payload, text, or token arrays: only enough information for random row
// access, workload planning, and source-policy accounting.
type embeddingScoreSpectrumSourceRow struct {
	offset                  int64
	length                  int
	candidateCount          int
	topKPairCountHard       int
	topKPairCountAll        int
	topKPairCountQ3         int
	topKPairCountBM25       int
	topKRecallPairCountHard int
	topKRecallPairCountAll  int
	topKRecallPairCountQ3   int
	topKRecallPairCountBM25 int
	baseLossDisabled        bool
	releaseTrainAllowed     bool
	commercialUseAllowed    bool
	trainAllowedForResearch bool
	sourceArtifactHash      string
	lineNo                  int
}

// Len returns the number of indexed rows.
func (s *EmbeddingTextScoreSpectrumSource) Len() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.rows)
}

// CandidateCount returns the canonical candidate count for a row, or zero
// for an out-of-range index.
func (s *EmbeddingTextScoreSpectrumSource) CandidateCount(index int) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].candidateCount
}

func (s *EmbeddingTextScoreSpectrumSource) TopKEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].topKEligiblePairCount(negativeMask)
}

func (s *EmbeddingTextScoreSpectrumSource) TopKRecallEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].topKRecallEligiblePairCount(negativeMask)
}

func (s *EmbeddingTextScoreSpectrumSource) BaseLossDisabled(index int) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return false
	}
	return s.rows[index].baseLossDisabled
}

// MaxCandidateCount returns the largest canonical candidate count in the
// source, or zero for an empty source.
func (s *EmbeddingTextScoreSpectrumSource) MaxCandidateCount() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxCandidates
}

// SourcePolicy returns a defensive copy of the indexed source policy.
func (s *EmbeddingTextScoreSpectrumSource) SourcePolicy() EmbeddingScoreSpectrumPolicy {
	if s == nil {
		return EmbeddingScoreSpectrumPolicy{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneScoreSpectrumSourcePolicy(s.policy)
}

// Example reads, canonicalizes, and tokenizes one text row. The source lock
// also makes tokenizer use and file lifetime safe when callers request rows
// concurrently or close the source from another goroutine.
func (s *EmbeddingTextScoreSpectrumSource) Example(index int) (EmbeddingScoreSpectrumExample, error) {
	if s == nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("nil score-spectrum source")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("score-spectrum source is closed")
	}
	line, lineNo, err := readEmbeddingScoreSpectrumSourceRow(s.file, s.rows, index)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, err
	}
	var record embeddingTextScoreSpectrumRecord
	if err := json.Unmarshal(line, &record); err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: %w", lineNo, err)
	}
	clean, err := record.example(s.allowResearch)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: %w", lineNo, err)
	}
	return tokenizeOneEmbeddingTextScoreSpectrumExample(clean, s.tokenizer, s.allowResearch, lineNo)
}

// Close releases the source file. Close is idempotent.
func (s *EmbeddingTextScoreSpectrumSource) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.file == nil {
		return nil
	}
	return s.file.Close()
}

func (s *EmbeddingTextScoreSpectrumSource) index() error {
	rows, policy, maxCandidates, err := indexTextScoreSpectrumFile(s.file, s.allowResearch)
	if err != nil {
		return err
	}
	s.rows = rows
	s.policy = policy
	s.maxCandidates = maxCandidates
	return nil
}

// Len returns the number of indexed rows.
func (s *EmbeddingTokenizedScoreSpectrumSource) Len() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.rows)
}

// CandidateCount returns the canonical candidate count for a row, or zero
// for an out-of-range index.
func (s *EmbeddingTokenizedScoreSpectrumSource) CandidateCount(index int) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].candidateCount
}

func (s *EmbeddingTokenizedScoreSpectrumSource) TopKEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].topKEligiblePairCount(negativeMask)
}

func (s *EmbeddingTokenizedScoreSpectrumSource) TopKRecallEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return 0
	}
	return s.rows[index].topKRecallEligiblePairCount(negativeMask)
}

func (s *EmbeddingTokenizedScoreSpectrumSource) BaseLossDisabled(index int) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.rows) {
		return false
	}
	return s.rows[index].baseLossDisabled
}

// MaxCandidateCount returns the largest canonical candidate count in the
// source, or zero for an empty source.
func (s *EmbeddingTokenizedScoreSpectrumSource) MaxCandidateCount() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxCandidates
}

// SourcePolicy returns a defensive copy of the indexed source policy.
func (s *EmbeddingTokenizedScoreSpectrumSource) SourcePolicy() EmbeddingScoreSpectrumPolicy {
	if s == nil {
		return EmbeddingScoreSpectrumPolicy{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneScoreSpectrumSourcePolicy(s.policy)
}

// Example reads and canonicalizes one tokenized row. Returned arrays and maps
// are fresh on every call.
func (s *EmbeddingTokenizedScoreSpectrumSource) Example(index int) (EmbeddingScoreSpectrumExample, error) {
	if s == nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("nil score-spectrum source")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("score-spectrum source is closed")
	}
	line, lineNo, err := readEmbeddingScoreSpectrumSourceRow(s.file, s.rows, index)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, err
	}
	var record embeddingScoreSpectrumRecord
	if err := unmarshalEmbeddingScoreSpectrumSourceRecord(line, &record); err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: %w", lineNo, err)
	}
	example, err := record.example(s.allowResearch)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: %w", lineNo, err)
	}
	return example, nil
}

// Close releases the source file. Close is idempotent.
func (s *EmbeddingTokenizedScoreSpectrumSource) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.file == nil {
		return nil
	}
	return s.file.Close()
}

func (s *EmbeddingTokenizedScoreSpectrumSource) index() error {
	rows, policy, maxCandidates, err := indexTokenizedScoreSpectrumFile(s.file, s.allowResearch)
	if err != nil {
		return err
	}
	s.rows = rows
	s.policy = policy
	s.maxCandidates = maxCandidates
	return nil
}

// Len returns the number of rows in the slice source.
func (s *EmbeddingScoreSpectrumSliceSource) Len() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.examples)
}

// CandidateCount returns the canonical candidate count for a row, or zero
// for an out-of-range index.
func (s *EmbeddingScoreSpectrumSliceSource) CandidateCount(index int) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.counts) {
		return 0
	}
	return s.counts[index]
}

func (s *EmbeddingScoreSpectrumSliceSource) TopKEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.examples) {
		return 0
	}
	return scoreSpectrumTopKStaticEligiblePairCount(s.examples[index], negativeMask)
}

func (s *EmbeddingScoreSpectrumSliceSource) TopKRecallEligiblePairCount(index int, negativeMask string) int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.examples) {
		return 0
	}
	return scoreSpectrumTopKRecallStaticEligiblePairCount(s.examples[index], negativeMask)
}

func (s *EmbeddingScoreSpectrumSliceSource) BaseLossDisabled(index int) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if index < 0 || index >= len(s.examples) {
		return false
	}
	return scoreSpectrumEffectiveBaseLossWeight(s.examples[index].BaseLossWeight) == 0
}

// MaxCandidateCount returns the largest canonical candidate count in the
// slice source.
func (s *EmbeddingScoreSpectrumSliceSource) MaxCandidateCount() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.maxCandidates
}

// SourcePolicy returns a defensive copy of the source policy.
func (s *EmbeddingScoreSpectrumSliceSource) SourcePolicy() EmbeddingScoreSpectrumPolicy {
	if s == nil {
		return EmbeddingScoreSpectrumPolicy{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneScoreSpectrumSourcePolicy(s.policy)
}

// Example returns a fresh deep copy of one in-memory row.
func (s *EmbeddingScoreSpectrumSliceSource) Example(index int) (EmbeddingScoreSpectrumExample, error) {
	if s == nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("nil score-spectrum source")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("score-spectrum source is closed")
	}
	if index < 0 || index >= len(s.examples) {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("score-spectrum row index %d out of range [0,%d)", index, len(s.examples))
	}
	return cloneEmbeddingScoreSpectrumExample(s.examples[index]), nil
}

// Close marks the slice source closed. It has no external resources and is
// therefore always nil and idempotent.
func (s *EmbeddingScoreSpectrumSliceSource) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	return nil
}

func indexTextScoreSpectrumFile(file *os.File, allowResearch bool) ([]embeddingScoreSpectrumSourceRow, EmbeddingScoreSpectrumPolicy, int, error) {
	if file == nil {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("nil score-spectrum source file")
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, err
	}
	reader := bufio.NewReaderSize(file, embeddingJSONLScannerInitialBuffer)
	rows := make([]embeddingScoreSpectrumSourceRow, 0)
	acc := newScoreSpectrumPolicyAccumulator()
	var offset int64
	lineNo := 0
	for {
		raw, hasData, err := readEmbeddingScoreSpectrumSourceLine(reader)
		if hasData {
			lineNo++
			rowOffset := offset
			offset += int64(len(raw))
			if errors.Is(err, errEmbeddingScoreSpectrumSourceRecordTooLarge) {
				return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: record exceeds maximum size %d bytes", lineNo, embeddingJSONLMaxRecordBytes)
			}
			if err != nil && err != io.EOF {
				return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, err)
			}
			line := bytes.TrimSpace(raw)
			if len(line) > 0 {
				var record embeddingTextScoreSpectrumRecord
				if unmarshalErr := json.Unmarshal(line, &record); unmarshalErr != nil {
					return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, unmarshalErr)
				}
				clean, exampleErr := record.example(allowResearch)
				if exampleErr != nil {
					return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, exampleErr)
				}
				rows = append(rows, embeddingScoreSpectrumSourceRow{
					offset:                  rowOffset,
					length:                  len(raw),
					candidateCount:          len(clean.Candidates),
					topKPairCountHard:       textScoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskHard),
					topKPairCountAll:        textScoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskAll),
					topKPairCountQ3:         textScoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskQ3),
					topKPairCountBM25:       textScoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskBM25),
					topKRecallPairCountHard: textScoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskHard),
					topKRecallPairCountAll:  textScoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskAll),
					topKRecallPairCountQ3:   textScoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskQ3),
					topKRecallPairCountBM25: textScoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskBM25),
					baseLossDisabled:        scoreSpectrumEffectiveBaseLossWeight(clean.BaseLossWeight) == 0,
					releaseTrainAllowed:     clean.ReleaseTrainAllowed,
					commercialUseAllowed:    clean.CommercialUseAllowed,
					trainAllowedForResearch: clean.TrainAllowedForResearch,
					sourceArtifactHash:      clean.SourceArtifactHash,
					lineNo:                  lineNo,
				})
				acc.add(clean.ReleaseTrainAllowed, clean.CommercialUseAllowed, clean.TrainAllowedForResearch, clean.SourceArtifactHash)
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, EmbeddingScoreSpectrumPolicy{}, 0, err
		}
	}
	if len(rows) == 0 {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("text score-spectrum dataset is empty")
	}
	return rows, acc.policy(), maxScoreSpectrumSourceRowCandidates(rows), nil
}

func indexTokenizedScoreSpectrumFile(file *os.File, allowResearch bool) ([]embeddingScoreSpectrumSourceRow, EmbeddingScoreSpectrumPolicy, int, error) {
	if file == nil {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("nil score-spectrum source file")
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, err
	}
	reader := bufio.NewReaderSize(file, embeddingJSONLScannerInitialBuffer)
	rows := make([]embeddingScoreSpectrumSourceRow, 0)
	acc := newScoreSpectrumPolicyAccumulator()
	var offset int64
	lineNo := 0
	for {
		raw, hasData, err := readEmbeddingScoreSpectrumSourceLine(reader)
		if hasData {
			lineNo++
			rowOffset := offset
			offset += int64(len(raw))
			if errors.Is(err, errEmbeddingScoreSpectrumSourceRecordTooLarge) {
				return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: record exceeds maximum size %d bytes", lineNo, embeddingJSONLMaxRecordBytes)
			}
			if err != nil && err != io.EOF {
				return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, err)
			}
			line := bytes.TrimSpace(raw)
			if len(line) > 0 {
				var record embeddingScoreSpectrumRecord
				if unmarshalErr := unmarshalEmbeddingScoreSpectrumSourceRecord(line, &record); unmarshalErr != nil {
					return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, unmarshalErr)
				}
				clean, exampleErr := record.example(allowResearch)
				if exampleErr != nil {
					return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("line %d: %w", lineNo, exampleErr)
				}
				rows = append(rows, embeddingScoreSpectrumSourceRow{
					offset:                  rowOffset,
					length:                  len(raw),
					candidateCount:          len(clean.CandidateTokens),
					topKPairCountHard:       scoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskHard),
					topKPairCountAll:        scoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskAll),
					topKPairCountQ3:         scoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskQ3),
					topKPairCountBM25:       scoreSpectrumTopKStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskBM25),
					topKRecallPairCountHard: scoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskHard),
					topKRecallPairCountAll:  scoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskAll),
					topKRecallPairCountQ3:   scoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskQ3),
					topKRecallPairCountBM25: scoreSpectrumTopKRecallStaticEligiblePairCount(clean, TurboQuantTopKNegativeMaskBM25),
					baseLossDisabled:        scoreSpectrumEffectiveBaseLossWeight(clean.BaseLossWeight) == 0,
					releaseTrainAllowed:     clean.ReleaseTrainAllowed,
					commercialUseAllowed:    clean.CommercialUseAllowed,
					trainAllowedForResearch: clean.TrainAllowedForResearch,
					sourceArtifactHash:      clean.SourceArtifactHash,
					lineNo:                  lineNo,
				})
				acc.add(clean.ReleaseTrainAllowed, clean.CommercialUseAllowed, clean.TrainAllowedForResearch, clean.SourceArtifactHash)
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, EmbeddingScoreSpectrumPolicy{}, 0, err
		}
	}
	if len(rows) == 0 {
		return nil, EmbeddingScoreSpectrumPolicy{}, 0, fmt.Errorf("score-spectrum dataset is empty")
	}
	return rows, acc.policy(), maxScoreSpectrumSourceRowCandidates(rows), nil
}

// readEmbeddingScoreSpectrumSourceLine reads one physical JSONL record while
// keeping the accumulated record at or below the repository's maximum. The
// bufio reader's fixed-size chunks are copied into a growing result only after
// checking the new total, so a giant or unterminated line cannot force an
// allocation proportional to its full size before being rejected.
//
// hasData distinguishes an empty reader at EOF from an unterminated final
// record. A final record with data and io.EOF is valid and is returned to the
// caller for normal parsing.
func readEmbeddingScoreSpectrumSourceLine(reader *bufio.Reader) (line []byte, hasData bool, err error) {
	if reader == nil {
		return nil, false, fmt.Errorf("nil score-spectrum JSONL reader")
	}
	var out []byte
	for {
		fragment, readErr := reader.ReadSlice('\n')
		if len(fragment) > 0 {
			hasData = true
			if len(out) > embeddingJSONLMaxRecordBytes-len(fragment) {
				return nil, true, errEmbeddingScoreSpectrumSourceRecordTooLarge
			}
			out = append(out, fragment...)
		}
		if readErr == nil {
			return out, true, nil
		}
		if readErr == bufio.ErrBufferFull {
			continue
		}
		if readErr == io.EOF {
			if !hasData {
				return nil, false, io.EOF
			}
			return out, true, io.EOF
		}
		return out, hasData, readErr
	}
}

func readEmbeddingScoreSpectrumSourceRow(file *os.File, rows []embeddingScoreSpectrumSourceRow, index int) ([]byte, int, error) {
	if index < 0 || index >= len(rows) {
		return nil, 0, fmt.Errorf("score-spectrum row index %d out of range [0,%d)", index, len(rows))
	}
	row := rows[index]
	if row.length <= 0 {
		return nil, row.lineNo, fmt.Errorf("line %d: empty indexed record", row.lineNo)
	}
	data := make([]byte, row.length)
	n, err := file.ReadAt(data, row.offset)
	if err != nil && !(err == io.EOF && n == len(data)) {
		return nil, row.lineNo, fmt.Errorf("line %d: read record at offset %d: %w", row.lineNo, row.offset, err)
	}
	if n != len(data) {
		return nil, row.lineNo, fmt.Errorf("line %d: short record read: got %d bytes, want %d", row.lineNo, n, len(data))
	}
	return bytes.TrimSpace(data), row.lineNo, nil
}

// unmarshalEmbeddingScoreSpectrumSourceRecord preserves both the historical
// capitalized ExtraFields member emitted by WriteEmbeddingScoreSpectrum...
// and arbitrary lower-case JSON fields from newer producers. The legacy
// tokenized reader only decoded the known members; the source contract keeps
// provenance just as the text reader does.
func unmarshalEmbeddingScoreSpectrumSourceRecord(data []byte, record *embeddingScoreSpectrumRecord) error {
	if record == nil {
		return fmt.Errorf("nil score-spectrum record")
	}
	if err := json.Unmarshal(data, record); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	extra := map[string]json.RawMessage{}
	if raw, ok := fields["ExtraFields"]; ok {
		var explicit map[string]json.RawMessage
		if err := json.Unmarshal(raw, &explicit); err != nil {
			return fmt.Errorf("ExtraFields: %w", err)
		}
		for key, value := range explicit {
			extra[key] = append(json.RawMessage(nil), value...)
		}
	}
	for key, value := range fields {
		if _, known := embeddingScoreSpectrumSourceKnownFields[key]; known {
			continue
		}
		extra[key] = append(json.RawMessage(nil), value...)
	}
	if len(extra) == 0 {
		record.ExtraFields = nil
	} else {
		record.ExtraFields = extra
	}
	return nil
}

func tokenizeOneEmbeddingTextScoreSpectrumExample(example EmbeddingTextScoreSpectrumExample, tokenizer *BPETokenizer, allowResearch bool, lineNo int) (EmbeddingScoreSpectrumExample, error) {
	if tokenizer == nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: nil tokenizer", lineNo)
	}
	clean, err := validateAndCanonicalizeTextScoreSpectrum(example, allowResearch)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d: %w", lineNo, err)
	}
	// Keep this cache local to one Example call. A global cache would retain
	// every candidate string and token array from the full JSONL corpus.
	cache := embeddingTextTokenCache{}
	query, err := cache.encode(clean.Query, tokenizer)
	if err != nil {
		return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d query: %w", lineNo, err)
	}
	query = cloneTokenizedText(query)
	candidateTokens := make([][]int32, 0, len(clean.Candidates))
	candidateMasks := make([][]int32, 0, len(clean.Candidates))
	for i, text := range clean.Candidates {
		candidate, err := cache.encode(text, tokenizer)
		if err != nil {
			return EmbeddingScoreSpectrumExample{}, fmt.Errorf("line %d candidate %d: %w", lineNo, i, err)
		}
		candidate = cloneTokenizedText(candidate)
		candidateTokens = append(candidateTokens, candidate.tokens)
		candidateMasks = append(candidateMasks, candidate.mask)
	}
	return EmbeddingScoreSpectrumExample{
		RowID:                          clean.RowID,
		Source:                         clean.Source,
		QueryTokens:                    query.tokens,
		QueryMask:                      query.mask,
		CandidateIDs:                   append([]string(nil), clean.CandidateIDs...),
		CandidateSources:               append([]string(nil), clean.CandidateSources...),
		QrelGains:                      append([]float32(nil), clean.QrelGains...),
		CandidateTokens:                candidateTokens,
		CandidateMasks:                 candidateMasks,
		PositiveIndexes:                append([]int(nil), clean.PositiveIndexes...),
		SelectedPositiveIndex:          cloneIntPtr(clean.SelectedPositiveIndex),
		HardNegativeEligible:           append([]bool(nil), clean.HardNegativeEligible...),
		TargetProbabilities:            append([]float32(nil), clean.TargetProbabilities...),
		BaseLossWeight:                 cloneFloat32Ptr(clean.BaseLossWeight),
		HardLossWeight:                 clean.HardLossWeight,
		SoftLossWeight:                 clean.SoftLossWeight,
		RecoveryLossWeight:             clean.RecoveryLossWeight,
		TurboQuantTopKLossWeight:       cloneFloat32Ptr(clean.TurboQuantTopKLossWeight),
		TurboQuantTopKRecallLossWeight: cloneFloat32Ptr(clean.TurboQuantTopKRecallLossWeight),
		TrainPolicy:                    clean.TrainPolicy,
		ReleaseTrainAllowed:            clean.ReleaseTrainAllowed,
		CommercialUseAllowed:           clean.CommercialUseAllowed,
		TrainAllowedForResearch:        clean.TrainAllowedForResearch,
		SourceArtifactHash:             clean.SourceArtifactHash,
		ExtraFields:                    cloneRawMessageMap(clean.ExtraFields),
	}, nil
}

type scoreSpectrumPolicyAccumulator struct {
	rowCount       int
	releaseAllowed bool
	commercial     bool
	research       bool
	researchOnly   bool
	hashes         map[string]struct{}
}

func newScoreSpectrumPolicyAccumulator() *scoreSpectrumPolicyAccumulator {
	return &scoreSpectrumPolicyAccumulator{
		releaseAllowed: true,
		commercial:     true,
		hashes:         map[string]struct{}{},
	}
}

func (a *scoreSpectrumPolicyAccumulator) add(releaseAllowed, commercial, research bool, sourceHash string) {
	if a == nil {
		return
	}
	a.rowCount++
	a.releaseAllowed = a.releaseAllowed && releaseAllowed
	a.commercial = a.commercial && commercial
	a.research = a.research || research
	if research && (!releaseAllowed || !commercial) {
		a.researchOnly = true
	}
	if sourceHash = strings.TrimSpace(sourceHash); sourceHash != "" {
		a.hashes[sourceHash] = struct{}{}
	}
}

func (a *scoreSpectrumPolicyAccumulator) policy() EmbeddingScoreSpectrumPolicy {
	if a == nil || a.rowCount == 0 {
		return EmbeddingScoreSpectrumPolicy{}
	}
	hashes := make([]string, 0, len(a.hashes))
	for hash := range a.hashes {
		hashes = append(hashes, hash)
	}
	sort.Strings(hashes)
	return EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:        true,
		ScoreSpectrumResearchOnly: a.researchOnly,
		TrainAllowedForResearch:   a.research,
		ReleaseTrainAllowed:       a.releaseAllowed,
		CommercialUseAllowed:      a.commercial,
		SourceArtifactHashes:      hashes,
		ScoreSpectrumRowCount:     a.rowCount,
	}
}

func maxScoreSpectrumSourceRowCandidates(rows []embeddingScoreSpectrumSourceRow) int {
	max := 0
	for _, row := range rows {
		if row.candidateCount > max {
			max = row.candidateCount
		}
	}
	return max
}

func (row embeddingScoreSpectrumSourceRow) topKEligiblePairCount(negativeMask string) int {
	mask, err := normalizeTurboQuantTopKNegativeMask(negativeMask)
	if err != nil {
		return 0
	}
	switch mask {
	case TurboQuantTopKNegativeMaskAll:
		return row.topKPairCountAll
	case TurboQuantTopKNegativeMaskQ3:
		return row.topKPairCountQ3
	case TurboQuantTopKNegativeMaskBM25:
		return row.topKPairCountBM25
	default:
		return row.topKPairCountHard
	}
}

func (row embeddingScoreSpectrumSourceRow) topKRecallEligiblePairCount(negativeMask string) int {
	mask, err := normalizeTurboQuantTopKNegativeMask(negativeMask)
	if err != nil {
		return 0
	}
	switch mask {
	case TurboQuantTopKNegativeMaskAll:
		return row.topKRecallPairCountAll
	case TurboQuantTopKNegativeMaskQ3:
		return row.topKRecallPairCountQ3
	case TurboQuantTopKNegativeMaskBM25:
		return row.topKRecallPairCountBM25
	default:
		return row.topKRecallPairCountHard
	}
}

func textScoreSpectrumTopKStaticEligiblePairCount(example EmbeddingTextScoreSpectrumExample, negativeMask string) int {
	tokenized := EmbeddingScoreSpectrumExample{
		CandidateIDs:                   append([]string(nil), example.CandidateIDs...),
		CandidateSources:               append([]string(nil), example.CandidateSources...),
		QrelGains:                      append([]float32(nil), example.QrelGains...),
		PositiveIndexes:                append([]int(nil), example.PositiveIndexes...),
		HardNegativeEligible:           append([]bool(nil), example.HardNegativeEligible...),
		TurboQuantTopKLossWeight:       cloneFloat32Ptr(example.TurboQuantTopKLossWeight),
		TurboQuantTopKRecallLossWeight: cloneFloat32Ptr(example.TurboQuantTopKRecallLossWeight),
	}
	tokenized.CandidateTokens = make([][]int32, len(example.Candidates))
	return scoreSpectrumTopKStaticEligiblePairCount(tokenized, negativeMask)
}

func textScoreSpectrumTopKRecallStaticEligiblePairCount(example EmbeddingTextScoreSpectrumExample, negativeMask string) int {
	tokenized := EmbeddingScoreSpectrumExample{
		CandidateIDs:                   append([]string(nil), example.CandidateIDs...),
		CandidateSources:               append([]string(nil), example.CandidateSources...),
		QrelGains:                      append([]float32(nil), example.QrelGains...),
		PositiveIndexes:                append([]int(nil), example.PositiveIndexes...),
		HardNegativeEligible:           append([]bool(nil), example.HardNegativeEligible...),
		TurboQuantTopKRecallLossWeight: cloneFloat32Ptr(example.TurboQuantTopKRecallLossWeight),
	}
	tokenized.CandidateTokens = make([][]int32, len(example.Candidates))
	return scoreSpectrumTopKRecallStaticEligiblePairCount(tokenized, negativeMask)
}

func maxScoreSpectrumCandidateCount(counts []int) int {
	max := 0
	for _, count := range counts {
		if count > max {
			max = count
		}
	}
	return max
}

func cloneScoreSpectrumSourcePolicy(policy EmbeddingScoreSpectrumPolicy) EmbeddingScoreSpectrumPolicy {
	policy.SourceArtifactHashes = append([]string(nil), policy.SourceArtifactHashes...)
	policy.AutoClearedObjectives = append([]string(nil), policy.AutoClearedObjectives...)
	policy.IsolatedInheritedObjectives = append([]string(nil), policy.IsolatedInheritedObjectives...)
	return policy
}

func cloneEmbeddingScoreSpectrumExample(example EmbeddingScoreSpectrumExample) EmbeddingScoreSpectrumExample {
	return EmbeddingScoreSpectrumExample{
		RowID:                          example.RowID,
		Source:                         example.Source,
		QueryTokens:                    append([]int32(nil), example.QueryTokens...),
		QueryMask:                      append([]int32(nil), example.QueryMask...),
		CandidateIDs:                   append([]string(nil), example.CandidateIDs...),
		CandidateSources:               append([]string(nil), example.CandidateSources...),
		QrelGains:                      append([]float32(nil), example.QrelGains...),
		CandidateTokens:                cloneInt32Matrix(example.CandidateTokens),
		CandidateMasks:                 cloneInt32Matrix(example.CandidateMasks),
		PositiveIndexes:                append([]int(nil), example.PositiveIndexes...),
		SelectedPositiveIndex:          cloneIntPtr(example.SelectedPositiveIndex),
		HardNegativeEligible:           append([]bool(nil), example.HardNegativeEligible...),
		TargetProbabilities:            append([]float32(nil), example.TargetProbabilities...),
		BaseLossWeight:                 cloneFloat32Ptr(example.BaseLossWeight),
		HardLossWeight:                 example.HardLossWeight,
		SoftLossWeight:                 example.SoftLossWeight,
		RecoveryLossWeight:             example.RecoveryLossWeight,
		TurboQuantTopKLossWeight:       cloneFloat32Ptr(example.TurboQuantTopKLossWeight),
		TurboQuantTopKRecallLossWeight: cloneFloat32Ptr(example.TurboQuantTopKRecallLossWeight),
		TrainPolicy:                    example.TrainPolicy,
		ReleaseTrainAllowed:            example.ReleaseTrainAllowed,
		CommercialUseAllowed:           example.CommercialUseAllowed,
		TrainAllowedForResearch:        example.TrainAllowedForResearch,
		SourceArtifactHash:             example.SourceArtifactHash,
		ExtraFields:                    cloneRawMessageMap(example.ExtraFields),
	}
}
