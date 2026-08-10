package eosruntime

import (
	"bufio"
	"container/heap"
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
	// Workers sets how many goroutines shard word entries for the initial
	// pair-frequency scan and for applying each selected merge to the word
	// entries it affects. <= 0 means "auto": see
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
// the initial pair-counting scan and each round's merge application over
// the word entries a merge affects; it never changes the trained
// tokenizer's content. requested wins when positive; otherwise the
// EOS_TOKENIZER_TRAIN_WORKERS environment variable is used when it parses
// as a positive integer; otherwise runtime.GOMAXPROCS(0). The result is
// always >= 1.
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
// of how many times it appears in the corpus. A wordEntry's pointer
// identity is also used as a stable key by the incremental pair index
// (pairIndex in TrainTokenizerFromCorpus; see applyIncrementalMergeShard).
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

	// entries is a stable, one-time partition unit for the initial parallel
	// pair-frequency/index scan (scanWordEntryPairsParallel). Materializing
	// it once (instead of ranging over wordMap) also avoids walking the map
	// itself with its randomized iteration order. Which entries land in
	// which shard does not affect the trained tokenizer: the scan sums
	// per-shard maps together (a commutative, associative reduction), and
	// every later merge round re-shards only the word entries the chosen
	// pair actually affects (see applyIncrementalMergeParallel), never all
	// of entries again.
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

	// pairFreqs and pairIndex (pair -> the word entries currently
	// containing it) are seeded once, in parallel, over the whole corpus.
	// Every merge round after that updates both incrementally (see
	// applyIncrementalMergeParallel/applyMergeDelta) instead of rescanning:
	// per-round cost tracks how many words the chosen merge actually
	// touches, not the corpus size.
	pairFreqs, pairIndex := scanWordEntryPairsParallel(shards)
	pq := newPairHeap(pairFreqs, interner)

	merges := make([]TokenizerMerge, 0)
	for len(specials)+len(interner.strings)+len(merges) < cfg.VocabSize {
		bestPair, _, found := popBestPair(pq, pairFreqs, cfg.MinFreq)
		if !found {
			break
		}
		affectedSet := pairIndex[bestPair]
		if len(affectedSet) == 0 {
			panic(fmt.Sprintf("tokenizer training: pair %+v has live frequency %d but no indexed word entries; pairFreqs and pairIndex are out of sync", bestPair, pairFreqs[bestPair]))
		}
		affected := make([]*wordEntry, 0, len(affectedSet))
		for entry := range affectedSet {
			affected = append(affected, entry)
		}

		leftStr, rightStr := interner.String(bestPair.left), interner.String(bestPair.right)
		mergedID := interner.intern(leftStr + rightStr)

		delta := applyIncrementalMergeParallel(affected, workers, bestPair.left, bestPair.right, mergedID)
		applyMergeDelta(pairFreqs, pairIndex, pq, delta)

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
// TestSelectBestPairIDMatchesReference. The actual training loop no longer
// calls this directly (see popBestPair), but it stays as the readable,
// directly tested specification popBestPair's heap-based selection must
// match.
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
// element). The partition only distributes work across goroutines: callers
// combine per-shard results with commutative, associative reductions and
// mutate each entry independently of every other entry, so the trained
// tokenizer does not depend on how entries are partitioned or on shard
// count. Used both for the one-time initial pair scan (over every word
// entry) and, every merge round, to re-partition just the word entries the
// chosen pair currently affects (see applyIncrementalMergeParallel).
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

// applyMergeIDs combines every non-overlapping adjacent (left, right) ID
// pair in ids into the single merged ID. It returns the input slice
// unchanged (no allocation) when the pair does not occur at all, which
// matters in aggregate for the same reason described on wordPairCounts.
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

// --- incremental pair-frequency bookkeeping ---------------------------
//
// The original trainer recounted every adjacent pair over the whole corpus
// every merge round (see git history for countPairFreqsParallel /
// applyMergeParallel). That makes each round's cost proportional to the
// corpus size, so total training cost is roughly (rounds * corpus size).
// The functions below replace that with the standard production-BPE
// technique: maintain pairFreqs and an inverted pairIndex (pair -> the
// word entries currently containing it) across rounds, and when a merge is
// applied, touch only the word entries pairIndex says the merged pair
// occurs in -- updating just the pairs whose counts actually change
// (decrementing the merged pair and its old neighbors, incrementing its
// new neighbors; see applyIncrementalMergeShard) instead of every pair in
// every word. Selection uses a lazily-validated max-heap (pairHeap) over
// pairFreqs instead of scanning it, so a round's cost tracks the number of
// affected words, not the corpus or vocabulary size.

// wordPairEntry pairs a distinct adjacent-ID pair occurring within one
// word's token sequence with how many times it occurs there, counting
// overlapping occurrences (for example pair (X,X) occurs twice in the
// 3-token sequence [X,X,X]).
type wordPairEntry struct {
	pair  tokenizerBPEPairID
	count int
}

// wordPairCounts returns the raw adjacent-pair counts within one word's
// token sequence: a plain sliding-window scan (i from 0 to len(ids)-2),
// the same counting rule the original whole-corpus recount applied per
// word. It returns a small slice rather than a map: BPE word lengths stay
// short (typically well under a few dozen tokens, and only shrink as
// merges combine tokens), so a linear scan-and-accumulate avoids
// map-allocation overhead in a function called twice per affected word on
// every merge round (see applyIncrementalMergeShard).
func wordPairCounts(ids []int32) []wordPairEntry {
	if len(ids) < 2 {
		return nil
	}
	counts := make([]wordPairEntry, 0, len(ids)-1)
	for i := 0; i < len(ids)-1; i++ {
		pair := tokenizerBPEPairID{left: ids[i], right: ids[i+1]}
		found := false
		for j := range counts {
			if counts[j].pair == pair {
				counts[j].count++
				found = true
				break
			}
		}
		if !found {
			counts = append(counts, wordPairEntry{pair: pair, count: 1})
		}
	}
	return counts
}

// wordPairCountsContain reports whether counts (as returned by
// wordPairCounts) includes pair at all, regardless of its count.
func wordPairCountsContain(counts []wordPairEntry, pair tokenizerBPEPairID) bool {
	for _, c := range counts {
		if c.pair == pair {
			return true
		}
	}
	return false
}

// scanWordEntryPairsShard computes one shard's contribution to the initial
// pair-frequency table and inverted pair index: for every word entry, it
// weights that word's wordPairCounts by the word's corpus frequency (see
// wordEntry) and records the word as a member of every distinct pair it
// contains.
func scanWordEntryPairsShard(entries []*wordEntry) (freqs map[tokenizerBPEPairID]int, index map[tokenizerBPEPairID][]*wordEntry) {
	freqs = make(map[tokenizerBPEPairID]int, 2*len(entries))
	index = make(map[tokenizerBPEPairID][]*wordEntry)
	for _, entry := range entries {
		for _, c := range wordPairCounts(entry.ids) {
			freqs[c.pair] += c.count * entry.freq
			index[c.pair] = append(index[c.pair], entry)
		}
	}
	return freqs, index
}

// scanWordEntryPairsParallel builds the starting pairFreqs table and
// pairIndex (pair -> the set of word entries containing it) across all
// shards concurrently, then combines the per-shard results: frequencies
// sum (commutative, associative integer addition), and index membership
// lists merge into per-pair sets (shards partition entries, so no entry
// can appear in two shards' lists for the same pair). It runs exactly
// once, before the merge-selection loop begins; every later round updates
// both structures incrementally instead of rescanning (see
// applyIncrementalMergeParallel and applyMergeDelta).
func scanWordEntryPairsParallel(shards [][]*wordEntry) (map[tokenizerBPEPairID]int, map[tokenizerBPEPairID]map[*wordEntry]struct{}) {
	type partial struct {
		freqs map[tokenizerBPEPairID]int
		index map[tokenizerBPEPairID][]*wordEntry
	}
	partials := make([]partial, len(shards))
	if len(shards) == 1 {
		freqs, index := scanWordEntryPairsShard(shards[0])
		partials[0] = partial{freqs: freqs, index: index}
	} else {
		var wg sync.WaitGroup
		wg.Add(len(shards))
		for i, shard := range shards {
			go func(i int, shard []*wordEntry) {
				defer wg.Done()
				freqs, index := scanWordEntryPairsShard(shard)
				partials[i] = partial{freqs: freqs, index: index}
			}(i, shard)
		}
		wg.Wait()
	}

	freqs := make(map[tokenizerBPEPairID]int, len(partials[0].freqs))
	index := make(map[tokenizerBPEPairID]map[*wordEntry]struct{}, len(partials[0].index))
	for _, p := range partials {
		for pair, f := range p.freqs {
			freqs[pair] += f
		}
		for pair, members := range p.index {
			set := index[pair]
			if set == nil {
				set = make(map[*wordEntry]struct{}, len(members))
				index[pair] = set
			}
			for _, entry := range members {
				set[entry] = struct{}{}
			}
		}
	}
	return freqs, index
}

// pairHeapItem is one snapshot of a pair's frequency at the moment it was
// pushed onto a pairHeap. Because pairFreqs changes incrementally as
// merges are applied (see applyMergeDelta), an item can go stale: a later
// push for the same pair records its new frequency without removing the
// earlier item. popBestPair treats a popped item as valid only when its
// freq still matches pairFreqs' live value for that pair.
type pairHeapItem struct {
	pair tokenizerBPEPairID
	freq int
}

// pairHeap is a max-heap over pairHeapItem, ordered by exactly the same
// total order as selectBestPairID/pairLessID: highest frequency first,
// ties broken by the lexicographically smaller pair. It supports lazy
// deletion (see pairHeapItem, popBestPair): callers never remove or
// decrease-key an existing item, they just push a fresh one whenever a
// pair's live frequency changes, and popBestPair discards stale items it
// encounters at pop time.
type pairHeap struct {
	items    []pairHeapItem
	interner *tokenInterner
}

func (h *pairHeap) Len() int { return len(h.items) }

// Less reports whether items[i] must be popped before items[j] under the
// fixed BPE tie-break total order (see selectBestPairID): higher frequency
// wins; frequency ties break on the lexicographically smaller pair via
// pairLessID, mirrored here exactly so a pairHeap always agrees with
// selectBestPairID about which pair wins.
func (h *pairHeap) Less(i, j int) bool {
	a, b := h.items[i], h.items[j]
	if a.freq != b.freq {
		return a.freq > b.freq
	}
	return pairLessID(a.pair, b.pair, h.interner)
}

func (h *pairHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }

// Push and Pop satisfy container/heap.Interface. Callers use the package
// functions heap.Push/heap.Pop (see newPairHeap, popBestPair, and
// applyMergeDelta), never these methods directly.
func (h *pairHeap) Push(x any) { h.items = append(h.items, x.(pairHeapItem)) }

func (h *pairHeap) Pop() any {
	old := h.items
	n := len(old)
	item := old[n-1]
	h.items = old[:n-1]
	return item
}

// newPairHeap builds a pairHeap already holding one item per entry in
// freqs, using heap.Init for an O(n) heapify instead of n individual
// O(log n) pushes.
func newPairHeap(freqs map[tokenizerBPEPairID]int, interner *tokenInterner) *pairHeap {
	pq := &pairHeap{items: make([]pairHeapItem, 0, len(freqs)), interner: interner}
	for pair, freq := range freqs {
		pq.items = append(pq.items, pairHeapItem{pair: pair, freq: freq})
	}
	heap.Init(pq)
	return pq
}

// popBestPair returns the same winner selectBestPairID would compute by
// scanning the live pairFreqs table, without scanning it: it pops pq (see
// pairHeap) until it finds an item whose recorded frequency still matches
// pairFreqs' current value for that pair, discarding stale items along the
// way (see pairHeapItem).
//
// That first live item is guaranteed to be the global best. applyMergeDelta
// always pushes a fresh item whenever a pair's live frequency changes, so
// every pair with a positive live count has at least one live item in pq
// at all times, and pq pops in non-increasing frequency order (ties broken
// exactly like selectBestPairID). So if the first live item's frequency is
// below minFreq, every other pair's live frequency is too, and there is no
// winner -- matching selectBestPairID(pairFreqs, minFreq, ...)'s
// found=false case exactly.
func popBestPair(pq *pairHeap, pairFreqs map[tokenizerBPEPairID]int, minFreq int) (tokenizerBPEPairID, int, bool) {
	for pq.Len() > 0 {
		top := heap.Pop(pq).(pairHeapItem)
		live, ok := pairFreqs[top.pair]
		if !ok || live != top.freq {
			continue
		}
		if live < minFreq {
			return tokenizerBPEPairID{}, 0, false
		}
		return top.pair, live, true
	}
	return tokenizerBPEPairID{}, 0, false
}

// shardMergeDelta accumulates one shard's contribution to a merge round:
// how much each pair's live frequency should change (freq, which can be
// negative), and which word entries newly started or stopped containing
// each pair (add/remove), ready to fold into the global
// pairFreqs/pairIndex (see applyMergeDelta).
type shardMergeDelta struct {
	freq   map[tokenizerBPEPairID]int
	add    map[tokenizerBPEPairID][]*wordEntry
	remove map[tokenizerBPEPairID][]*wordEntry
}

func newShardMergeDelta() *shardMergeDelta {
	return &shardMergeDelta{
		freq:   make(map[tokenizerBPEPairID]int),
		add:    make(map[tokenizerBPEPairID][]*wordEntry),
		remove: make(map[tokenizerBPEPairID][]*wordEntry),
	}
}

// applyIncrementalMergeShard applies one merge to every word entry in one
// shard of the affected-entries list (see applyIncrementalMergeParallel)
// and records the exact resulting pair-frequency/index delta.
//
// For each entry it takes the word's raw pair counts before the merge
// (oldCounts) and after (newCounts, computed from the same applyMergeIDs
// rewrite the pre-incremental trainer applied to every word every round),
// both via wordPairCounts, and folds their weighted difference into freq.
//
// Comparing full before/after pair counts -- rather than reasoning
// pair-by-pair about which specific neighbors changed -- handles every
// case correctly with one rule, including words where the merge pair
// occurs more than once (possibly overlapping, so not every occurrence
// becomes a merge site: see applyMergeIDs) or where the same pair type
// also occurs elsewhere in the word, untouched by this merge. The diff is
// exactly what a from-scratch recount of this one word would find, and
// words outside the affected list are provably unchanged (applyMergeIDs is
// a no-op when its pair does not occur in a word), so summing this diff
// over every affected entry reproduces precisely the pairFreqs a full
// recount would.
func applyIncrementalMergeShard(entries []*wordEntry, left, right, merged int32) *shardMergeDelta {
	d := newShardMergeDelta()
	for _, entry := range entries {
		oldCounts := wordPairCounts(entry.ids)
		entry.ids = applyMergeIDs(entry.ids, left, right, merged)
		newCounts := wordPairCounts(entry.ids)
		freqWeight := entry.freq

		for _, oc := range oldCounts {
			d.freq[oc.pair] -= oc.count * freqWeight
			if !wordPairCountsContain(newCounts, oc.pair) {
				d.remove[oc.pair] = append(d.remove[oc.pair], entry)
			}
		}
		for _, nc := range newCounts {
			d.freq[nc.pair] += nc.count * freqWeight
			if !wordPairCountsContain(oldCounts, nc.pair) {
				d.add[nc.pair] = append(d.add[nc.pair], entry)
			}
		}
	}
	return d
}

// applyIncrementalMergeParallel applies one merge across every entry in
// affected (the current live members of pairIndex[{left,right}]),
// re-sharding just that subset across workers goroutines (see
// shardWordEntries) and combining each shard's delta (see
// applyIncrementalMergeShard): pair-frequency deltas sum, and add/remove
// membership lists concatenate (shards partition affected, so no entry is
// processed twice). Every entry belongs to exactly one shard and is
// rewritten only by that shard's own goroutine, so there is no shared
// mutable state between goroutines and therefore no data race; the result
// does not depend on shard count for the same reason the original
// whole-corpus parallel scan did not.
func applyIncrementalMergeParallel(affected []*wordEntry, workers int, left, right, merged int32) *shardMergeDelta {
	shards := shardWordEntries(affected, workers)
	if len(shards) == 1 {
		return applyIncrementalMergeShard(shards[0], left, right, merged)
	}

	partials := make([]*shardMergeDelta, len(shards))
	var wg sync.WaitGroup
	wg.Add(len(shards))
	for i, shard := range shards {
		go func(i int, shard []*wordEntry) {
			defer wg.Done()
			partials[i] = applyIncrementalMergeShard(shard, left, right, merged)
		}(i, shard)
	}
	wg.Wait()

	combined := newShardMergeDelta()
	for _, p := range partials {
		for pair, f := range p.freq {
			combined.freq[pair] += f
		}
		for pair, es := range p.add {
			combined.add[pair] = append(combined.add[pair], es...)
		}
		for pair, es := range p.remove {
			combined.remove[pair] = append(combined.remove[pair], es...)
		}
	}
	return combined
}

// applyMergeDelta folds one merge round's shardMergeDelta into the global
// pairFreqs table, pairIndex, and pq (see pairHeap), so the next
// popBestPair call sees an up-to-date view without rescanning anything.
//
// Index removals are applied before additions, though the two operations
// never actually interleave for the same (pair, entry): a single word
// entry can only be added to or removed from one pair's index membership
// per round (wordPairCountsContain(oldCounts, pair) and
// wordPairCountsContain(newCounts, pair) cannot both be false for the pair
// that put entry in d.add or d.remove in the first place), so add and
// remove operations touching different entries commute freely regardless
// of the order they are applied in.
func applyMergeDelta(pairFreqs map[tokenizerBPEPairID]int, pairIndex map[tokenizerBPEPairID]map[*wordEntry]struct{}, pq *pairHeap, d *shardMergeDelta) {
	for pair, entries := range d.remove {
		set := pairIndex[pair]
		for _, entry := range entries {
			delete(set, entry)
		}
		if len(set) == 0 {
			delete(pairIndex, pair)
		}
	}
	for pair, entries := range d.add {
		set := pairIndex[pair]
		if set == nil {
			set = make(map[*wordEntry]struct{}, len(entries))
			pairIndex[pair] = set
		}
		for _, entry := range entries {
			set[entry] = struct{}{}
		}
	}
	for pair, delta := range d.freq {
		newFreq := pairFreqs[pair] + delta
		if newFreq <= 0 {
			delete(pairFreqs, pair)
			continue
		}
		pairFreqs[pair] = newFreq
		heap.Push(pq, pairHeapItem{pair: pair, freq: newFreq})
	}
}
