package eosruntime

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unicode"
)

type TokenizerTrainConfig struct {
	CorpusPath string
	VocabSize  int
	MinFreq    int
	// Workers sets how many goroutines shard pair counting and merge
	// application during each BPE training round. <= 0 means "auto": see
	// ResolveTokenizerTrainWorkers. Merge-selection ties always break on a
	// fixed total order (frequency descending, then pair lexicographic
	// ascending; see selectBestPair), so the trained tokenizer is
	// byte-identical for any Workers value.
	Workers int
}

// tokenizerTrainWorkersEnv is the environment variable fallback used by
// ResolveTokenizerTrainWorkers when TokenizerTrainConfig.Workers is <= 0.
const tokenizerTrainWorkersEnv = "EOS_TOKENIZER_TRAIN_WORKERS"

// ResolveTokenizerTrainWorkers returns the effective worker count for
// parallel tokenizer training. It only changes how many goroutines shard
// the per-round pair-counting and merge-application work; it never changes
// the trained tokenizer's content. requested wins when positive; otherwise
// the EOS_TOKENIZER_TRAIN_WORKERS environment variable is used when it
// parses as a positive integer; otherwise runtime.GOMAXPROCS(0). The
// result is always >= 1.
func ResolveTokenizerTrainWorkers(requested int) int {
	if requested > 0 {
		return requested
	}
	if v := strings.TrimSpace(os.Getenv(tokenizerTrainWorkersEnv)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	if n := runtime.GOMAXPROCS(0); n > 0 {
		return n
	}
	return 1
}

// trainTokenizerFromCorpusCallCount counts TrainTokenizerFromCorpus
// invocations. Production code never reads it; it exists so tests can
// assert that TrainTokenizerFromCorpusCached skips training entirely on a
// cache hit.
var trainTokenizerFromCorpusCallCount atomic.Int64

// wordEntry is one unique corpus word: its current BPE token sequence
// (interned token IDs; see tokenInterner) and how many times the word
// occurred in the corpus. TrainTokenizerFromCorpus counts pairs and
// applies merges over the unique-word table (wordMap), not the raw
// per-occurrence corpus stream, so a word's cost is paid once regardless
// of how many times it appears in the corpus.
type wordEntry struct {
	ids  []int32
	freq int
}

// tokenInterner assigns small, stable, never-reused integer IDs to token
// strings as they are first discovered (initial base characters, then one
// new ID per accepted merge). Pair counting and merge application key on
// these IDs instead of raw strings.
//
// This matters a great deal: profiling BPE training on a real ~58MB, ~159k
// unique-word corpus showed Go's string-keyed map operations (hashing and
// comparing the token strings themselves) accounting for roughly half of
// total CPU time in the per-round pair-counting hot loop, because it runs
// once per adjacent token pair per unique word per round. Integer-keyed
// maps use a single-instruction-class hash/compare instead of walking
// string bytes, which is the standard technique production BPE trainers
// use for this reason. Two interned IDs are equal if and only if their
// underlying strings are equal (intern is idempotent), so ID equality is a
// correct, cheaper substitute for string equality everywhere below.
type tokenInterner struct {
	ids     map[string]int32
	strings []string
}

func newTokenInterner() *tokenInterner {
	return &tokenInterner{ids: make(map[string]int32)}
}

// intern returns s's stable ID, assigning a new one the first time s is
// seen. Repeated calls with the same string always return the same ID.
func (t *tokenInterner) intern(s string) int32 {
	if id, ok := t.ids[s]; ok {
		return id
	}
	id := int32(len(t.strings))
	t.strings = append(t.strings, s)
	t.ids[s] = id
	return id
}

// String resolves an ID back to its string. It is an O(1) slice index,
// not a map lookup, so tie-break comparisons that need real string content
// (see pairLessID) stay cheap.
func (t *tokenInterner) String(id int32) string {
	return t.strings[id]
}

func MinimumTokenizerVocabSizeForCorpus(path string) (int, error) {
	words, err := loadCorpusWords(path)
	if err != nil {
		return 0, err
	}
	if len(words) == 0 {
		return 0, fmt.Errorf("empty corpus")
	}
	charSet := make(map[string]bool)
	for _, word := range words {
		for _, r := range word {
			charSet[string(r)] = true
		}
	}
	return 5 + len(charSet), nil
}

// TrainTokenizerFromCorpus builds a lightweight BPE tokenizer file from a raw text corpus.
func TrainTokenizerFromCorpus(cfg TokenizerTrainConfig) (TokenizerFile, error) {
	trainTokenizerFromCorpusCallCount.Add(1)
	if cfg.CorpusPath == "" {
		return TokenizerFile{}, fmt.Errorf("tokenizer corpus path is required")
	}
	if cfg.VocabSize <= 0 {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size must be positive")
	}
	if cfg.VocabSize < 5 {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size %d is too small for Eos special tokens", cfg.VocabSize)
	}
	if cfg.MinFreq <= 0 {
		cfg.MinFreq = 2
	}
	words, err := loadCorpusWords(cfg.CorpusPath)
	if err != nil {
		return TokenizerFile{}, err
	}
	if len(words) == 0 {
		return TokenizerFile{}, fmt.Errorf("empty corpus")
	}

	wordMap := make(map[string]*wordEntry)
	interner := newTokenInterner()
	for _, word := range words {
		if entry, ok := wordMap[word]; ok {
			entry.freq++
			continue
		}
		ids := make([]int32, 0, len(word))
		for _, r := range word {
			ids = append(ids, interner.intern(string(r)))
		}
		wordMap[word] = &wordEntry{ids: ids, freq: 1}
	}
	// baseCharCount snapshots how many IDs are initial base characters
	// (interned above, in corpus-scan order) versus merge-created tokens
	// (interned below, in round/selection order). The final vocabulary
	// keeps the original ordering contract: base characters sorted
	// lexicographically, followed by merge tokens in creation order.
	baseCharCount := len(interner.strings)

	// entries is a stable partition unit for parallel pair counting and
	// merge application. Materializing it once (instead of ranging over
	// wordMap every round) also avoids re-walking the map up to VocabSize
	// times. Which entries land in which shard does not affect the
	// trained tokenizer: pair counting sums per-shard maps together (a
	// commutative, associative reduction) and merge application mutates
	// each entry independently of every other entry.
	entries := make([]*wordEntry, 0, len(wordMap))
	for _, entry := range wordMap {
		entries = append(entries, entry)
	}
	workers := ResolveTokenizerTrainWorkers(cfg.Workers)
	if workers > len(entries) {
		workers = len(entries)
	}
	if workers < 1 {
		workers = 1
	}
	shards := shardWordEntries(entries, workers)

	specials := []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}
	if len(specials)+baseCharCount > cfg.VocabSize {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size %d is too small for %d special+base tokens", cfg.VocabSize, len(specials)+baseCharCount)
	}

	merges := make([]TokenizerMerge, 0)
	for len(specials)+len(interner.strings)+len(merges) < cfg.VocabSize {
		pairFreqs := countPairFreqsParallel(shards)
		if len(pairFreqs) == 0 {
			break
		}

		bestPair, _, found := selectBestPairID(pairFreqs, cfg.MinFreq, interner)
		if !found {
			break
		}

		leftStr, rightStr := interner.String(bestPair.left), interner.String(bestPair.right)
		mergedID := interner.intern(leftStr + rightStr)
		applyMergeParallel(shards, bestPair.left, bestPair.right, mergedID)
		merges = append(merges, TokenizerMerge{Left: leftStr, Right: rightStr})
	}

	baseTokens := make([]string, 0, len(interner.strings))
	baseChars := append([]string(nil), interner.strings[:baseCharCount]...)
	sort.Strings(baseChars)
	baseTokens = append(baseTokens, baseChars...)
	baseTokens = append(baseTokens, interner.strings[baseCharCount:]...)

	tokens := make([]string, 0, len(specials)+len(baseTokens))
	tokens = append(tokens, specials...)
	tokens = append(tokens, baseTokens...)
	file := TokenizerFile{
		Version:      TokenizerFileVersion,
		Tokens:       tokens,
		Merges:       merges,
		PadToken:     "[PAD]",
		UnknownToken: "[UNK]",
		BOSToken:     "[CLS]",
		EOSToken:     "[SEP]",
	}
	return file, file.Validate()
}

func PadTokenizerFileVocab(file TokenizerFile, vocabSize int) (TokenizerFile, error) {
	if vocabSize <= 0 {
		return TokenizerFile{}, fmt.Errorf("tokenizer vocab size must be positive")
	}
	if err := file.Validate(); err != nil {
		return TokenizerFile{}, err
	}
	if len(file.Tokens) > vocabSize {
		return TokenizerFile{}, fmt.Errorf("tokenizer has %d tokens, cannot pad down to vocab size %d", len(file.Tokens), vocabSize)
	}
	if len(file.Tokens) == vocabSize {
		return file, nil
	}
	seen := make(map[string]bool, len(file.Tokens))
	for _, token := range file.Tokens {
		seen[token] = true
	}
	for i := 0; len(file.Tokens) < vocabSize; i++ {
		token := fmt.Sprintf("[UNUSED_%06d]", i)
		if seen[token] {
			continue
		}
		file.Tokens = append(file.Tokens, token)
		seen[token] = true
	}
	return file, file.Validate()
}

func loadCorpusWords(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var words []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		for _, word := range strings.Fields(line) {
			cleaned := normalizeWord(word)
			if cleaned != "" {
				words = append(words, cleaned)
			}
		}
	}
	return words, scanner.Err()
}

func normalizeWord(word string) string {
	var b strings.Builder
	for _, r := range word {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// tokenizerBPEPair is a candidate BPE merge: two adjacent tokens.
//
// selectBestPair/pairLess (below) are the plain-string reference
// implementation of the BPE tie-break total order: easy to read and
// directly unit tested as the specification of "which pair wins." Actual
// training uses the ID-keyed, string-hashing-free selectBestPairID/
// pairLessID (further below), which TestSelectBestPairIDMatchesReference
// checks against this reference on randomized inputs.
type tokenizerBPEPair struct{ left, right string }

func pairLess(a, b tokenizerBPEPair) bool {
	if a.left != b.left {
		return a.left < b.left
	}
	return a.right < b.right
}

// selectBestPair scans combined per-pair frequencies and returns the single
// best merge candidate under the fixed BPE tie-break total order: highest
// frequency wins; frequency ties break on the lexicographically smaller
// pair (left, then right; see pairLess). That relation is a strict total
// order over pairFreqs' keys (every two distinct pairs compare unequal),
// so a running-max fold over it always converges to the same answer
// regardless of Go's randomized map iteration order and regardless of
// whether pairFreqs was built by one goroutine or merged from many
// parallel shards.
func selectBestPair(pairFreqs map[tokenizerBPEPair]int, minFreq int) (tokenizerBPEPair, int, bool) {
	var (
		bestPair tokenizerBPEPair
		bestFreq int
		found    bool
	)
	for candidate, freq := range pairFreqs {
		if freq < minFreq {
			continue
		}
		if !found || freq > bestFreq || (freq == bestFreq && pairLess(candidate, bestPair)) {
			bestPair = candidate
			bestFreq = freq
			found = true
		}
	}
	return bestPair, bestFreq, found
}

// tokenizerBPEPairID is the ID-keyed counterpart of tokenizerBPEPair used
// by the actual training hot path; see tokenInterner.
type tokenizerBPEPairID struct{ left, right int32 }

// pairLessID mirrors pairLess exactly (same left-then-right lexicographic
// rule on the underlying strings), but takes the fast path of comparing
// IDs for equality first: two IDs from the same interner are equal if and
// only if their strings are equal, so interner.String (an O(1) slice
// index) is only needed to determine order, not to test equality.
func pairLessID(a, b tokenizerBPEPairID, interner *tokenInterner) bool {
	if a.left != b.left {
		return interner.String(a.left) < interner.String(b.left)
	}
	if a.right != b.right {
		return interner.String(a.right) < interner.String(b.right)
	}
	return false
}

// selectBestPairID is selectBestPair's ID-keyed production twin: identical
// tie-break total order (see selectBestPair), verified equivalent to it by
// TestSelectBestPairIDMatchesReference.
func selectBestPairID(pairFreqs map[tokenizerBPEPairID]int, minFreq int, interner *tokenInterner) (tokenizerBPEPairID, int, bool) {
	var (
		bestPair tokenizerBPEPairID
		bestFreq int
		found    bool
	)
	for candidate, freq := range pairFreqs {
		if freq < minFreq {
			continue
		}
		if !found || freq > bestFreq || (freq == bestFreq && pairLessID(candidate, bestPair, interner)) {
			bestPair = candidate
			bestFreq = freq
			found = true
		}
	}
	return bestPair, bestFreq, found
}

// shardWordEntries splits entries into up to workers contiguous,
// roughly-equal shards (the last entries.len%workers shards get one extra
// element). The partition only distributes work across goroutines: pair
// counting sums per-shard results together and merge application mutates
// each entry independently, so the trained tokenizer does not depend on
// how entries are partitioned or on shard count.
func shardWordEntries(entries []*wordEntry, workers int) [][]*wordEntry {
	if workers < 1 {
		workers = 1
	}
	if workers > len(entries) {
		workers = len(entries)
	}
	if workers < 1 {
		workers = 1
	}
	shards := make([][]*wordEntry, workers)
	n := len(entries)
	base := n / workers
	rem := n % workers
	idx := 0
	for w := 0; w < workers; w++ {
		size := base
		if w < rem {
			size++
		}
		shards[w] = entries[idx : idx+size]
		idx += size
	}
	return shards
}

// countPairFreqsShard counts adjacent-token-ID pair frequencies, weighted
// by each word's corpus frequency, across one shard of word entries.
// Keying on interned int32 IDs instead of strings is deliberate: this loop
// runs once per adjacent pair per unique word per round, and profiling a
// real ~58MB/~159k-unique-word corpus showed Go's string-keyed map
// hash/compare dominating (roughly half of total CPU time) at that volume.
func countPairFreqsShard(entries []*wordEntry) map[tokenizerBPEPairID]int {
	freqs := make(map[tokenizerBPEPairID]int, 2*len(entries))
	for _, entry := range entries {
		for i := 0; i < len(entry.ids)-1; i++ {
			freqs[tokenizerBPEPairID{left: entry.ids[i], right: entry.ids[i+1]}] += entry.freq
		}
	}
	return freqs
}

// countPairFreqsParallel counts pair frequencies across all shards
// concurrently, then combines the per-shard maps into one map by summing
// counts per pair. Integer addition is commutative and associative, so
// the combined frequencies are identical regardless of shard count or
// goroutine scheduling order.
func countPairFreqsParallel(shards [][]*wordEntry) map[tokenizerBPEPairID]int {
	if len(shards) == 0 {
		return map[tokenizerBPEPairID]int{}
	}
	if len(shards) == 1 {
		return countPairFreqsShard(shards[0])
	}
	partials := make([]map[tokenizerBPEPairID]int, len(shards))
	var wg sync.WaitGroup
	wg.Add(len(shards))
	for i, shard := range shards {
		go func(i int, shard []*wordEntry) {
			defer wg.Done()
			partials[i] = countPairFreqsShard(shard)
		}(i, shard)
	}
	wg.Wait()
	total := make(map[tokenizerBPEPairID]int, len(partials[0]))
	for _, partial := range partials {
		for p, f := range partial {
			total[p] += f
		}
	}
	return total
}

// applyMergeIDs combines every non-overlapping adjacent (left, right) ID
// pair in ids into the single merged ID. It returns the input slice
// unchanged (no allocation) when the pair does not occur at all, which
// matters in aggregate for the same reason described on countPairFreqsShard.
func applyMergeIDs(ids []int32, left, right, merged int32) []int32 {
	if len(ids) < 2 {
		return ids
	}
	matchAt := -1
	for i := 0; i < len(ids)-1; i++ {
		if ids[i] == left && ids[i+1] == right {
			matchAt = i
			break
		}
	}
	if matchAt < 0 {
		return ids
	}
	out := make([]int32, 0, len(ids))
	out = append(out, ids[:matchAt]...)
	for i := matchAt; i < len(ids); {
		if i < len(ids)-1 && ids[i] == left && ids[i+1] == right {
			out = append(out, merged)
			i += 2
			continue
		}
		out = append(out, ids[i])
		i++
	}
	return out
}

// applyMergeShard rewrites token IDs in place for one shard of word entries.
func applyMergeShard(entries []*wordEntry, left, right, merged int32) {
	for _, entry := range entries {
		entry.ids = applyMergeIDs(entry.ids, left, right, merged)
	}
}

// applyMergeParallel applies one merge to every word entry across all
// shards concurrently. Each entry belongs to exactly one shard and is
// rewritten only by that shard's own goroutine, so there is no shared
// mutable state between goroutines and therefore no data race. The result
// does not depend on shard count: every entry's new tokens depend only on
// that entry's own prior tokens, never on any other entry.
func applyMergeParallel(shards [][]*wordEntry, left, right, merged int32) {
	if len(shards) == 1 {
		applyMergeShard(shards[0], left, right, merged)
		return
	}
	var wg sync.WaitGroup
	wg.Add(len(shards))
	for _, shard := range shards {
		go func(shard []*wordEntry) {
			defer wg.Done()
			applyMergeShard(shard, left, right, merged)
		}(shard)
	}
	wg.Wait()
}
