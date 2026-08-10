package eosruntime

import (
	"bufio"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestTrainTokenizerFromCorpus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("banana bandana banana band\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	file, err := TrainTokenizerFromCorpus(TokenizerTrainConfig{
		CorpusPath: path,
		VocabSize:  16,
		MinFreq:    2,
	})
	if err != nil {
		t.Fatalf("train tokenizer: %v", err)
	}
	if len(file.Tokens) == 0 {
		t.Fatal("expected non-empty tokenizer tokens")
	}
	if len(file.Tokens) > 16 {
		t.Fatalf("token count = %d, want <= 16", len(file.Tokens))
	}
	if len(file.Merges) == 0 {
		t.Fatal("expected at least one merge from repeated corpus")
	}
	if file.Tokens[0] != "[PAD]" || file.Tokens[1] != "[CLS]" || file.Tokens[2] != "[SEP]" || file.Tokens[3] != "[UNK]" {
		t.Fatalf("unexpected special token prefix: %v", file.Tokens[:4])
	}
}

// --- parallel training: shared synthetic corpus fixture --------------------

// syntheticTokenizerCorpusWords returns a deterministic vocabulary built by
// combining prefixes, roots, and suffixes. The combinatorial construction
// guarantees heavy shared-substring overlap (and therefore genuine pair-
// frequency ties during BPE training), which a purely random letter
// generator would rarely produce.
func syntheticTokenizerCorpusWords() []string {
	prefixes := []string{"", "un", "re", "pre", "dis", "mis", "over", "under"}
	roots := []string{"run", "jump", "walk", "code", "test", "train", "merge", "token", "parallel", "cache", "shard", "count", "select", "build", "write", "read"}
	suffixes := []string{"", "ing", "ed", "er", "s", "tion", "able", "ly", "ness"}
	words := make([]string, 0, len(prefixes)*len(roots)*len(suffixes))
	for _, p := range prefixes {
		for _, r := range roots {
			for _, s := range suffixes {
				words = append(words, p+r+s)
			}
		}
	}
	return words
}

// writeSyntheticTokenizerCorpus writes a deterministic corpus file built
// from syntheticTokenizerCorpusWords, repeating each word a deterministic,
// index-derived number of times so the corpus has a realistic skewed
// (Zipf-like) frequency shape. It returns the corpus file path.
func writeSyntheticTokenizerCorpus(t *testing.T, dir string) string {
	t.Helper()
	words := syntheticTokenizerCorpusWords()
	path := filepath.Join(dir, "synthetic-corpus.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create synthetic corpus: %v", err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	perLine := 0
	for i, word := range words {
		freq := 1 + (i*37+11)%40
		for n := 0; n < freq; n++ {
			if _, err := w.WriteString(word); err != nil {
				t.Fatalf("write synthetic corpus word: %v", err)
			}
			perLine++
			if perLine%20 == 0 {
				if _, err := w.WriteString("\n"); err != nil {
					t.Fatalf("write synthetic corpus newline: %v", err)
				}
			} else if _, err := w.WriteString(" "); err != nil {
				t.Fatalf("write synthetic corpus space: %v", err)
			}
		}
	}
	if err := w.Flush(); err != nil {
		t.Fatalf("flush synthetic corpus: %v", err)
	}
	return path
}

func trainSyntheticTokenizer(t *testing.T, corpusPath string, workers int) TokenizerFile {
	t.Helper()
	file, err := TrainTokenizerFromCorpus(TokenizerTrainConfig{
		CorpusPath: corpusPath,
		VocabSize:  400,
		MinFreq:    2,
		Workers:    workers,
	})
	if err != nil {
		t.Fatalf("train tokenizer (workers=%d): %v", workers, err)
	}
	return file
}

func tokenizerArtifactHash(t *testing.T, file TokenizerFile) string {
	t.Helper()
	data, err := encodeTokenizerMLL(file)
	if err != nil {
		t.Fatalf("encode tokenizer artifact: %v", err)
	}
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum)
}

// --- determinism: worker count and repeated runs must not change the artifact

func TestTrainTokenizerFromCorpus_ParallelMatchesSerial(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeSyntheticTokenizerCorpus(t, dir)

	serial := trainSyntheticTokenizer(t, corpusPath, 1)
	if len(serial.Merges) < 50 {
		t.Fatalf("expected a substantial merge count to exercise sharding, got %d", len(serial.Merges))
	}
	serialHash := tokenizerArtifactHash(t, serial)

	for _, workers := range []int{2, 3, 8, 5000} {
		parallel := trainSyntheticTokenizer(t, corpusPath, workers)
		if !reflect.DeepEqual(serial.Tokens, parallel.Tokens) {
			t.Fatalf("workers=%d: tokens differ from serial baseline", workers)
		}
		if !reflect.DeepEqual(serial.Merges, parallel.Merges) {
			t.Fatalf("workers=%d: merges differ from serial baseline", workers)
		}
		if got := tokenizerArtifactHash(t, parallel); got != serialHash {
			t.Fatalf("workers=%d: artifact hash %s != serial hash %s", workers, got, serialHash)
		}
	}
}

func TestTrainTokenizerFromCorpus_DeterministicAcrossRepeatedRuns(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeSyntheticTokenizerCorpus(t, dir)

	first := trainSyntheticTokenizer(t, corpusPath, 0)
	second := trainSyntheticTokenizer(t, corpusPath, 0)
	if !reflect.DeepEqual(first.Tokens, second.Tokens) {
		t.Fatal("tokens differ between two runs with identical config")
	}
	if !reflect.DeepEqual(first.Merges, second.Merges) {
		t.Fatal("merges differ between two runs with identical config")
	}
	if h1, h2 := tokenizerArtifactHash(t, first), tokenizerArtifactHash(t, second); h1 != h2 {
		t.Fatalf("artifact hash differs between two runs: %s != %s", h1, h2)
	}
}

func TestTrainTokenizerFromCorpus_WorkersExceedingUniqueWordsDoesNotPanic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.txt")
	// Only 3 unique words; request far more workers than that.
	if err := os.WriteFile(path, []byte("banana bandana banana band banana bandana\n"), 0o644); err != nil {
		t.Fatalf("write corpus: %v", err)
	}
	file, err := TrainTokenizerFromCorpus(TokenizerTrainConfig{
		CorpusPath: path,
		VocabSize:  16,
		MinFreq:    2,
		Workers:    10000,
	})
	if err != nil {
		t.Fatalf("train tokenizer with excess workers: %v", err)
	}
	if len(file.Merges) == 0 {
		t.Fatal("expected at least one merge")
	}
}

// --- selectBestPair: total order is independent of map iteration order -----

func TestSelectBestPairTotalOrderIsStable(t *testing.T) {
	freqs := map[tokenizerBPEPair]int{
		{left: "z", right: "z"}: 10,
		{left: "a", right: "b"}: 10, // ties with the above; "a" < "z" must win
		{left: "a", right: "a"}: 10, // ties with the above; "a"=="a" vs "a"=="b": right "a" < "b" must win
		{left: "m", right: "n"}: 9,
		{left: "x", right: "y"}: 1,
	}
	want := tokenizerBPEPair{left: "a", right: "a"}
	for i := 0; i < 500; i++ {
		got, freq, found := selectBestPair(freqs, 1)
		if !found {
			t.Fatalf("iteration %d: expected a winner", i)
		}
		if got != want {
			t.Fatalf("iteration %d: selectBestPair = %+v, want %+v", i, got, want)
		}
		if freq != 10 {
			t.Fatalf("iteration %d: freq = %d, want 10", i, freq)
		}
	}
}

func TestSelectBestPairRespectsMinFreq(t *testing.T) {
	freqs := map[tokenizerBPEPair]int{
		{left: "a", right: "b"}: 1,
		{left: "c", right: "d"}: 1,
	}
	if _, _, found := selectBestPair(freqs, 2); found {
		t.Fatal("expected no winner when every candidate is below minFreq")
	}
}

// --- selectBestPairID / applyMergeIDs: equivalence to the string reference -

// TestSelectBestPairIDMatchesReference builds the same pair-frequency data
// as both a string-keyed map (fed to the reference selectBestPair) and an
// ID-keyed map through a tokenInterner (fed to the production
// selectBestPairID), and asserts they pick the identical winner across
// many distinct pair/frequency shapes, including ties.
func TestSelectBestPairIDMatchesReference(t *testing.T) {
	cases := []map[tokenizerBPEPair]int{
		{{left: "a", right: "b"}: 5, {left: "c", right: "d"}: 5, {left: "a", right: "a"}: 5},
		{{left: "z", right: "y"}: 3, {left: "z", right: "x"}: 3, {left: "a", right: "z"}: 3},
		{{left: "the", right: "re"}: 100, {left: "th", right: "ere"}: 100, {left: "t", right: "here"}: 99},
		{{left: "un", right: "do"}: 1},
	}
	for ci, freqs := range cases {
		interner := newTokenInterner()
		idFreqs := make(map[tokenizerBPEPairID]int, len(freqs))
		for pair, freq := range freqs {
			idFreqs[tokenizerBPEPairID{left: interner.intern(pair.left), right: interner.intern(pair.right)}] = freq
		}
		wantPair, wantFreq, wantFound := selectBestPair(freqs, 1)
		gotPairID, gotFreq, gotFound := selectBestPairID(idFreqs, 1, interner)
		if gotFound != wantFound {
			t.Fatalf("case %d: found = %t, want %t", ci, gotFound, wantFound)
		}
		if !wantFound {
			continue
		}
		got := tokenizerBPEPair{left: interner.String(gotPairID.left), right: interner.String(gotPairID.right)}
		if got != wantPair {
			t.Fatalf("case %d: selectBestPairID = %+v, want %+v", ci, got, wantPair)
		}
		if gotFreq != wantFreq {
			t.Fatalf("case %d: freq = %d, want %d", ci, gotFreq, wantFreq)
		}
	}
}

// TestApplyMergeIDsMatchesStringApplyMerge asserts applyMergeIDs (used by
// the parallel training hot path) rewrites token sequences identically to
// the retained string-based applyMerge, across sequences exercising no
// match, a single match, adjacent/overlapping matches, and a match at the
// start/end of the sequence.
func TestApplyMergeIDsMatchesStringApplyMerge(t *testing.T) {
	cases := []struct {
		tokens      []string
		left, right string
	}{
		{[]string{"a", "b", "c"}, "a", "b"},
		{[]string{"a", "b", "c"}, "b", "c"},
		{[]string{"x", "y", "z"}, "a", "b"},
		{[]string{"a", "a", "a"}, "a", "a"},
		{[]string{"a", "a", "a", "a"}, "a", "a"},
		{[]string{"a"}, "a", "a"},
		{[]string{}, "a", "a"},
		{[]string{"a", "b", "a", "b", "a", "b"}, "a", "b"},
	}
	for ci, tc := range cases {
		interner := newTokenInterner()
		ids := make([]int32, len(tc.tokens))
		for i, tok := range tc.tokens {
			ids[i] = interner.intern(tok)
		}
		leftID, rightID := interner.intern(tc.left), interner.intern(tc.right)
		mergedID := interner.intern(tc.left + tc.right)

		wantTokens := applyMerge(append([]string(nil), tc.tokens...), tc.left, tc.right)
		gotIDs := applyMergeIDs(append([]int32(nil), ids...), leftID, rightID, mergedID)

		// Built with a plain range+append (not make+index) so a nil/empty
		// gotIDs naturally yields a nil gotTokens, matching applyMerge's
		// own nil-preserving contract; reflect.DeepEqual treats a nil
		// slice and a non-nil empty slice as unequal.
		var gotTokens []string
		for _, id := range gotIDs {
			gotTokens = append(gotTokens, interner.String(id))
		}
		if !reflect.DeepEqual(gotTokens, wantTokens) {
			t.Fatalf("case %d: applyMergeIDs = %v, want %v", ci, gotTokens, wantTokens)
		}
	}
}

// --- full-loop regression: ID-interned production path vs. a from-scratch
// string-based reference implementation of the pre-optimization algorithm --

// referenceTrainTokenizer is a plain, unoptimized, string-keyed BPE
// trainer built directly from the retained selectBestPair/pairLess/
// applyMerge reference functions, mirroring the original (pre-interning)
// TrainTokenizerFromCorpus algorithm exactly, including its vocab-budget
// accounting. It exists only so tests can prove the ID-interning
// production path (see tokenInterner) is a pure performance optimization:
// byte-for-byte identical output to the obviously-correct string-based
// algorithm it replaced.
func referenceTrainTokenizer(t *testing.T, corpusPath string, vocabSize, minFreq int) TokenizerFile {
	t.Helper()
	words, err := loadCorpusWords(corpusPath)
	if err != nil {
		t.Fatalf("load corpus words: %v", err)
	}
	type refWordEntry struct {
		tokens []string
		freq   int
	}
	wordMap := make(map[string]*refWordEntry)
	for _, word := range words {
		if entry, ok := wordMap[word]; ok {
			entry.freq++
			continue
		}
		chars := make([]string, 0, len(word))
		for _, r := range word {
			chars = append(chars, string(r))
		}
		wordMap[word] = &refWordEntry{tokens: chars, freq: 1}
	}
	charSet := make(map[string]bool)
	for _, entry := range wordMap {
		for _, tok := range entry.tokens {
			charSet[tok] = true
		}
	}
	baseTokens := make([]string, 0, len(charSet))
	for tok := range charSet {
		baseTokens = append(baseTokens, tok)
	}
	sort.Strings(baseTokens)

	merges := make([]TokenizerMerge, 0)
	specials := []string{"[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"}
	if len(specials)+len(baseTokens) > vocabSize {
		t.Fatalf("reference: vocab size %d too small for base tokens", vocabSize)
	}
	for len(specials)+len(baseTokens)+len(merges) < vocabSize {
		pairFreqs := make(map[tokenizerBPEPair]int)
		for _, entry := range wordMap {
			for i := 0; i < len(entry.tokens)-1; i++ {
				pairFreqs[tokenizerBPEPair{left: entry.tokens[i], right: entry.tokens[i+1]}] += entry.freq
			}
		}
		if len(pairFreqs) == 0 {
			break
		}
		bestPair, _, found := selectBestPair(pairFreqs, minFreq)
		if !found {
			break
		}
		merged := bestPair.left + bestPair.right
		for _, entry := range wordMap {
			entry.tokens = applyMerge(entry.tokens, bestPair.left, bestPair.right)
		}
		merges = append(merges, TokenizerMerge{Left: bestPair.left, Right: bestPair.right})
		already := false
		for _, tok := range baseTokens {
			if tok == merged {
				already = true
				break
			}
		}
		if !already {
			baseTokens = append(baseTokens, merged)
		}
	}
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
	if err := file.Validate(); err != nil {
		t.Fatalf("reference tokenizer invalid: %v", err)
	}
	return file
}

func TestTrainTokenizerFromCorpus_MatchesStringReferenceImplementation(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeSyntheticTokenizerCorpus(t, dir)

	production := trainSyntheticTokenizer(t, corpusPath, 0)
	reference := referenceTrainTokenizer(t, corpusPath, 400, 2)

	if len(reference.Merges) < 50 {
		t.Fatalf("expected a substantial reference merge count, got %d", len(reference.Merges))
	}
	if !reflect.DeepEqual(production.Tokens, reference.Tokens) {
		t.Fatal("ID-interned production tokens differ from the string reference implementation")
	}
	if !reflect.DeepEqual(production.Merges, reference.Merges) {
		t.Fatal("ID-interned production merges differ from the string reference implementation")
	}
}

// --- ResolveTokenizerTrainWorkers -------------------------------------------

func TestResolveTokenizerTrainWorkers(t *testing.T) {
	t.Run("requested wins", func(t *testing.T) {
		t.Setenv(tokenizerTrainWorkersEnv, "7")
		if got := ResolveTokenizerTrainWorkers(4); got != 4 {
			t.Fatalf("got %d, want 4", got)
		}
	})
	t.Run("env wins over GOMAXPROCS when requested <= 0", func(t *testing.T) {
		t.Setenv(tokenizerTrainWorkersEnv, "6")
		if got := ResolveTokenizerTrainWorkers(0); got != 6 {
			t.Fatalf("got %d, want 6", got)
		}
	})
	t.Run("invalid env falls back to GOMAXPROCS", func(t *testing.T) {
		t.Setenv(tokenizerTrainWorkersEnv, "not-a-number")
		got := ResolveTokenizerTrainWorkers(0)
		if got < 1 {
			t.Fatalf("got %d, want >= 1", got)
		}
	})
	t.Run("non-positive env ignored", func(t *testing.T) {
		t.Setenv(tokenizerTrainWorkersEnv, "0")
		got := ResolveTokenizerTrainWorkers(0)
		if got < 1 {
			t.Fatalf("got %d, want >= 1", got)
		}
	})
	t.Run("unset env falls back to GOMAXPROCS", func(t *testing.T) {
		t.Setenv(tokenizerTrainWorkersEnv, "")
		got := ResolveTokenizerTrainWorkers(0)
		if got < 1 {
			t.Fatalf("got %d, want >= 1", got)
		}
	})
	t.Run("result always positive", func(t *testing.T) {
		if got := ResolveTokenizerTrainWorkers(-5); got < 1 {
			t.Fatalf("got %d, want >= 1", got)
		}
	})
}
