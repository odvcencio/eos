package eosruntime

import (
	"container/heap"
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

// This file tests the incremental pair-frequency bookkeeping added on top
// of the S7 parallel trainer (see tokenizer_train.go: wordPairCounts,
// pairHeap/popBestPair, applyIncrementalMergeShard/Parallel,
// applyMergeDelta). TestTrainTokenizerFromCorpus_MatchesStringReferenceImplementation
// and the parallel/serial hash-equality tests in tokenizer_train_test.go
// already prove the trained tokenizer stays byte-identical end to end;
// these tests instead isolate the new machinery so a bug shows up as a
// small, specific failure instead of only a final-hash mismatch.

// --- wordPairCounts / wordPairCountsContain ---------------------------

func TestWordPairCounts(t *testing.T) {
	interner := newTokenInterner()
	id := func(s string) int32 { return interner.intern(s) }
	a, b, c := id("a"), id("b"), id("c")

	cases := []struct {
		name string
		ids  []int32
		want []wordPairEntry
	}{
		{"empty", nil, nil},
		{"single token", []int32{a}, nil},
		{"two tokens", []int32{a, b}, []wordPairEntry{{pair: tokenizerBPEPairID{a, b}, count: 1}}},
		{
			"no repeats", []int32{a, b, c},
			[]wordPairEntry{{pair: tokenizerBPEPairID{a, b}, count: 1}, {pair: tokenizerBPEPairID{b, c}, count: 1}},
		},
		{
			// [X,X,X]: a plain sliding-window scan counts the overlapping
			// (X,X) pair twice (i=0 and i=1), even though only one of the
			// two occurrences can actually become a merge site (see
			// applyMergeIDs). wordPairCounts intentionally counts every
			// raw adjacency, matching what a from-scratch recount over the
			// same sequence would find.
			"overlapping repeats", []int32{a, a, a},
			[]wordPairEntry{{pair: tokenizerBPEPairID{a, a}, count: 2}},
		},
		{
			"repeated non-adjacent pair", []int32{a, b, a, b},
			[]wordPairEntry{{pair: tokenizerBPEPairID{a, b}, count: 2}, {pair: tokenizerBPEPairID{b, a}, count: 1}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := wordPairCounts(tc.ids)
			if len(got) != len(tc.want) {
				t.Fatalf("wordPairCounts(%v) = %v, want %v", tc.ids, got, tc.want)
			}
			for _, w := range tc.want {
				if !containsWordPairEntry(got, w) {
					t.Fatalf("wordPairCounts(%v) = %v, missing %+v", tc.ids, got, w)
				}
			}
			for _, pair := range []tokenizerBPEPairID{{a, b}, {b, a}, {a, a}, {b, c}} {
				want := wordPairCountsContainsPair(tc.want, pair)
				got := wordPairCountsContain(got, pair)
				if got != want {
					t.Fatalf("wordPairCountsContain(%v, %+v) = %t, want %t", tc.ids, pair, got, want)
				}
			}
		})
	}
}

func containsWordPairEntry(counts []wordPairEntry, want wordPairEntry) bool {
	for _, c := range counts {
		if c == want {
			return true
		}
	}
	return false
}

func wordPairCountsContainsPair(counts []wordPairEntry, pair tokenizerBPEPairID) bool {
	for _, c := range counts {
		if c.pair == pair {
			return true
		}
	}
	return false
}

// --- pairHeap / popBestPair: lazy staleness and tie-break -------------

// TestPopBestPairMatchesSelectBestPairIDAcrossStaleHeapMutations directly
// exercises pairHeap's lazy-deletion contract. It replays a long,
// randomized sequence of pairFreqs increments/decrements/removals over a
// handful of pairs chosen so many of them share a right-hand token (and so
// collide on frequency, forcing the pair-lexicographic tie-break) --
// mirroring what applyMergeDelta does across many merge rounds, but
// without needing real word entries. After every mutation it pushes a
// fresh pairHeap item, exactly as applyMergeDelta does, then asserts
// popBestPair's answer matches selectBestPairID computed fresh over the
// live map, including while many stale, superseded items are still
// sitting in the heap.
func TestPopBestPairMatchesSelectBestPairIDAcrossStaleHeapMutations(t *testing.T) {
	interner := newTokenInterner()
	const pairCount = 12
	pairs := make([]tokenizerBPEPairID, pairCount)
	for i := 0; i < pairCount; i++ {
		left := interner.intern(fmt.Sprintf("L%02d", i))
		// Sharing right-hand tokens across left IDs deliberately produces
		// frequency ties whose winner depends on comparing real interned
		// strings, not just IDs.
		right := interner.intern(fmt.Sprintf("R%02d", i%3))
		pairs[i] = tokenizerBPEPairID{left: left, right: right}
	}

	pairFreqs := make(map[tokenizerBPEPairID]int)
	pq := newPairHeap(pairFreqs, interner)

	rng := rand.New(rand.NewSource(12345))
	for step := 0; step < 3000; step++ {
		pair := pairs[rng.Intn(len(pairs))]
		delta := rng.Intn(7) - 3 // -3..3: frequent small moves, including exact ties
		newFreq := pairFreqs[pair] + delta
		if newFreq <= 0 {
			delete(pairFreqs, pair)
		} else {
			pairFreqs[pair] = newFreq
			heap.Push(pq, pairHeapItem{pair: pair, freq: newFreq})
		}

		wantPair, wantFreq, wantFound := selectBestPairID(pairFreqs, 1, interner)
		gotPair, gotFreq, gotFound := popBestPair(pq, pairFreqs, 1)
		if gotFound {
			// popBestPair pops; restore the one valid winner so pairFreqs'
			// invariant ("every live pair has a live heap item") holds for
			// the next step. Stale items popped along the way stay
			// discarded, which is exactly what real usage relies on.
			heap.Push(pq, pairHeapItem{pair: gotPair, freq: gotFreq})
		}

		if gotFound != wantFound {
			t.Fatalf("step %d: found = %t, want %t (pairFreqs=%v)", step, gotFound, wantFound, pairFreqs)
		}
		if !wantFound {
			continue
		}
		if gotPair != wantPair || gotFreq != wantFreq {
			t.Fatalf("step %d: popBestPair = (%+v, %d), want (%+v, %d)", step, gotPair, gotFreq, wantPair, wantFreq)
		}
	}
}

// TestPopBestPairBreaksTiesPairLexicographicAscending pins down the exact
// tie-break rule popBestPair must preserve (matching selectBestPair's
// documented total order): among pairs sharing the maximum frequency, the
// lexicographically smallest (left, then right) wins, regardless of push
// order.
func TestPopBestPairBreaksTiesPairLexicographicAscending(t *testing.T) {
	interner := newTokenInterner()
	mk := func(l, r string) tokenizerBPEPairID {
		return tokenizerBPEPairID{left: interner.intern(l), right: interner.intern(r)}
	}
	zz := mk("z", "z")
	ab := mk("a", "b")
	aa := mk("a", "a")
	mn := mk("m", "n")

	pairFreqs := map[tokenizerBPEPairID]int{
		zz: 10,
		ab: 10, // ties with zz; "a" < "z" must win
		aa: 10, // ties with ab; "a"=="a" vs "a"=="b": right "a" < "b" must win
		mn: 9,
	}
	// Push in several different orders; the winner must not depend on it.
	orders := [][]tokenizerBPEPairID{
		{zz, ab, aa, mn},
		{mn, aa, ab, zz},
		{aa, zz, mn, ab},
	}
	for oi, order := range orders {
		pq := &pairHeap{interner: interner}
		for _, p := range order {
			heap.Push(pq, pairHeapItem{pair: p, freq: pairFreqs[p]})
		}
		gotPair, gotFreq, found := popBestPair(pq, pairFreqs, 1)
		if !found {
			t.Fatalf("order %d: expected a winner", oi)
		}
		if gotPair != aa || gotFreq != 10 {
			t.Fatalf("order %d: popBestPair = (%+v, %d), want (%+v, 10)", oi, gotPair, gotFreq, aa)
		}
	}
}

// --- round-by-round consistency: incremental state vs. from-scratch ----

// buildIncrementalTestWordEntries interns words into a fresh wordEntry set
// (mirroring TrainTokenizerFromCorpus's own setup) so each subtest starts
// from an independent interner and word table.
func buildIncrementalTestWordEntries(words []string) ([]*wordEntry, *tokenInterner) {
	interner := newTokenInterner()
	wordMap := make(map[string]*wordEntry)
	for _, w := range words {
		if e, ok := wordMap[w]; ok {
			e.freq++
			continue
		}
		ids := make([]int32, 0, len(w))
		for _, r := range w {
			ids = append(ids, interner.intern(string(r)))
		}
		wordMap[w] = &wordEntry{ids: ids, freq: 1}
	}
	entries := make([]*wordEntry, 0, len(wordMap))
	for _, e := range wordMap {
		entries = append(entries, e)
	}
	return entries, interner
}

// TestIncrementalPairBookkeepingMatchesFromScratchRescanEveryRound drives
// the exact sequence TrainTokenizerFromCorpus uses -- popBestPair,
// applyIncrementalMergeParallel, applyMergeDelta -- directly over a small,
// adversarial word set chosen to exercise repeated/overlapping occurrences
// within a single word (see wordPairCounts's "overlapping repeats" case)
// and shared substrings across words. After every single round (not just
// at the end) it rescans the current, now-mutated word entries completely
// from scratch via scanWordEntryPairsParallel and asserts the
// incrementally maintained pairFreqs and pairIndex are exactly identical
// to that from-scratch answer. This is a stronger check than matching the
// final trained tokenizer: it proves the incremental bookkeeping never
// drifts from ground truth at any intermediate point, for several worker
// counts.
func TestIncrementalPairBookkeepingMatchesFromScratchRescanEveryRound(t *testing.T) {
	words := []string{
		"aaaa", "aaaa", "aaaa",
		"aaaaaa",
		"banana", "banana", "banana",
		"bandana", "bandana",
		"band",
		"aabaa", "aabaa", "aabaa",
		"mississippi", "mississippi",
		"abababab",
	}

	for _, workers := range []int{1, 2, 4, 7} {
		entries, interner := buildIncrementalTestWordEntries(words)
		shards := shardWordEntries(entries, workers)
		pairFreqs, pairIndex := scanWordEntryPairsParallel(shards)
		pq := newPairHeap(pairFreqs, interner)

		round := 0
		for {
			bestPair, _, found := popBestPair(pq, pairFreqs, 1)
			if !found {
				break
			}
			affectedSet := pairIndex[bestPair]
			if len(affectedSet) == 0 {
				t.Fatalf("workers=%d round=%d: pair %+v has live frequency but no indexed entries", workers, round, bestPair)
			}
			affected := make([]*wordEntry, 0, len(affectedSet))
			for e := range affectedSet {
				affected = append(affected, e)
			}
			mergedID := interner.intern(interner.String(bestPair.left) + interner.String(bestPair.right))

			delta := applyIncrementalMergeParallel(affected, workers, bestPair.left, bestPair.right, mergedID)
			applyMergeDelta(pairFreqs, pairIndex, pq, delta)

			freshFreqs, freshIndex := scanWordEntryPairsParallel(shardWordEntries(entries, workers))
			if !reflect.DeepEqual(pairFreqs, freshFreqs) {
				t.Fatalf("workers=%d round=%d: incremental pairFreqs diverged from from-scratch rescan\nincremental=%v\nfresh=%v",
					workers, round, pairFreqs, freshFreqs)
			}
			if len(pairIndex) != len(freshIndex) {
				t.Fatalf("workers=%d round=%d: pairIndex has %d pairs, from-scratch rescan has %d", workers, round, len(pairIndex), len(freshIndex))
			}
			for pair, freshSet := range freshIndex {
				gotSet, ok := pairIndex[pair]
				if !ok {
					t.Fatalf("workers=%d round=%d: pairIndex missing pair %+v present in from-scratch rescan", workers, round, pair)
				}
				if len(gotSet) != len(freshSet) {
					t.Fatalf("workers=%d round=%d: pairIndex[%+v] has %d members, from-scratch rescan has %d", workers, round, pair, len(gotSet), len(freshSet))
				}
				for e := range freshSet {
					if _, ok := gotSet[e]; !ok {
						t.Fatalf("workers=%d round=%d: pairIndex[%+v] missing a word entry present in from-scratch rescan", workers, round, pair)
					}
				}
			}

			round++
			if round > 500 {
				t.Fatalf("workers=%d: exceeded 500 rounds without exhausting pairs; likely an infinite loop", workers)
			}
		}
		if round == 0 {
			t.Fatalf("workers=%d: expected at least one merge round", workers)
		}
	}
}
