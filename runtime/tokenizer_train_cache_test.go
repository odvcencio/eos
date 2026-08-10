package eosruntime

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func writeTinyTokenizerCorpus(t *testing.T, dir, name, contents string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("write corpus %s: %v", name, err)
	}
	return path
}

// --- TokenizerTrainCacheKey --------------------------------------------

func TestTokenizerTrainCacheKey_StableForSameInputs(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}

	k1, err := TokenizerTrainCacheKey(cfg)
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	k2, err := TokenizerTrainCacheKey(cfg)
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	if k1 != k2 {
		t.Fatalf("cache key changed for identical inputs: %s != %s", k1, k2)
	}
	if k1 == "" {
		t.Fatal("expected a non-empty cache key")
	}
}

func TestTokenizerTrainCacheKey_IgnoresWorkers(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")

	k1, err := TokenizerTrainCacheKey(TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2, Workers: 1})
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	k2, err := TokenizerTrainCacheKey(TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2, Workers: 64})
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	if k1 != k2 {
		t.Fatalf("cache key must not depend on Workers: %s != %s", k1, k2)
	}
}

func TestTokenizerTrainCacheKey_ChangesWithCorpusVocabMinFreq(t *testing.T) {
	dir := t.TempDir()
	corpusA := writeTinyTokenizerCorpus(t, dir, "a.txt", "banana bandana banana band\n")
	corpusB := writeTinyTokenizerCorpus(t, dir, "b.txt", "orange orangutan orange or\n")
	base := TokenizerTrainConfig{CorpusPath: corpusA, VocabSize: 16, MinFreq: 2}

	baseKey, err := TokenizerTrainCacheKey(base)
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}

	variants := []TokenizerTrainConfig{
		{CorpusPath: corpusB, VocabSize: base.VocabSize, MinFreq: base.MinFreq},
		{CorpusPath: base.CorpusPath, VocabSize: 32, MinFreq: base.MinFreq},
		{CorpusPath: base.CorpusPath, VocabSize: base.VocabSize, MinFreq: 3},
	}
	seen := map[string]bool{baseKey: true}
	for i, variant := range variants {
		key, err := TokenizerTrainCacheKey(variant)
		if err != nil {
			t.Fatalf("variant %d cache key: %v", i, err)
		}
		if seen[key] {
			t.Fatalf("variant %d produced a colliding cache key %s", i, key)
		}
		seen[key] = true
	}
}

func TestTokenizerTrainCacheKey_RequiresCorpusPath(t *testing.T) {
	if _, err := TokenizerTrainCacheKey(TokenizerTrainConfig{VocabSize: 16}); err == nil {
		t.Fatal("expected an error for an empty corpus path")
	}
}

// --- TrainTokenizerFromCorpusCached: hit/miss --------------------------

func TestTrainTokenizerFromCorpusCached_MissThenHit(t *testing.T) {
	dir := t.TempDir()
	cacheDir := filepath.Join(dir, "cache")
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}
	cacheCfg := TokenizerTrainCacheConfig{CacheDir: cacheDir}

	before := trainTokenizerFromCorpusCallCount.Load()
	file1, result1, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if result1.Hit {
		t.Fatal("first call should be a cache miss")
	}
	afterFirst := trainTokenizerFromCorpusCallCount.Load()
	if afterFirst != before+1 {
		t.Fatalf("expected exactly one training invocation on miss, delta=%d", afterFirst-before)
	}
	if result1.Path == "" {
		t.Fatal("expected a non-empty cache entry path")
	}
	if _, err := os.Stat(result1.Path); err != nil {
		t.Fatalf("expected cache entry to exist on disk: %v", err)
	}

	file2, result2, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if !result2.Hit {
		t.Fatal("second call should be a cache hit")
	}
	afterSecond := trainTokenizerFromCorpusCallCount.Load()
	if afterSecond != afterFirst {
		t.Fatalf("cache hit must skip training entirely, but call count advanced by %d", afterSecond-afterFirst)
	}
	if result1.Key != result2.Key {
		t.Fatalf("cache key changed between calls: %s != %s", result1.Key, result2.Key)
	}
	if !reflect.DeepEqual(file1.Tokens, file2.Tokens) || !reflect.DeepEqual(file1.Merges, file2.Merges) {
		t.Fatal("cache hit returned a different tokenizer than the original training run")
	}
}

func TestTrainTokenizerFromCorpusCached_Disabled(t *testing.T) {
	dir := t.TempDir()
	cacheDir := filepath.Join(dir, "cache")
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}
	cacheCfg := TokenizerTrainCacheConfig{CacheDir: cacheDir, Disable: true}

	before := trainTokenizerFromCorpusCallCount.Load()
	_, result1, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if !result1.Disabled {
		t.Fatal("expected result to report the cache as disabled")
	}
	_, result2, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if result2.Hit {
		t.Fatal("a disabled cache must never report a hit")
	}
	after := trainTokenizerFromCorpusCallCount.Load()
	if after != before+2 {
		t.Fatalf("expected two training invocations with the cache disabled, delta=%d", after-before)
	}
	entries, err := os.ReadDir(cacheDir)
	if err == nil && len(entries) != 0 {
		t.Fatalf("disabled cache must not write cache entries, found %d", len(entries))
	}
}

func TestTrainTokenizerFromCorpusCached_RequiresCacheDirUnlessDisabled(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}
	if _, _, err := TrainTokenizerFromCorpusCached(cfg, TokenizerTrainCacheConfig{}); err == nil {
		t.Fatal("expected an error when CacheDir is empty and the cache is not disabled")
	}
}

func TestTrainTokenizerFromCorpusCached_CorruptEntrySelfHeals(t *testing.T) {
	dir := t.TempDir()
	cacheDir := filepath.Join(dir, "cache")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatalf("mkdir cache dir: %v", err)
	}
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}
	cacheCfg := TokenizerTrainCacheConfig{CacheDir: cacheDir}

	key, err := TokenizerTrainCacheKey(cfg)
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	corruptPath := tokenizerCacheEntryPath(cacheDir, key)
	if err := os.WriteFile(corruptPath, []byte("not a valid tokenizer artifact"), 0o644); err != nil {
		t.Fatalf("write corrupt cache entry: %v", err)
	}

	file, result, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
	if err != nil {
		t.Fatalf("expected corrupt cache entry to self-heal, got error: %v", err)
	}
	if result.Hit {
		t.Fatal("a corrupt cache entry must be treated as a miss")
	}
	if len(file.Tokens) == 0 {
		t.Fatal("expected a freshly trained tokenizer")
	}
	healed, err := ReadTokenizerFile(corruptPath)
	if err != nil {
		t.Fatalf("expected the corrupt entry to be overwritten with a valid one: %v", err)
	}
	if !reflect.DeepEqual(healed.Tokens, file.Tokens) {
		t.Fatal("healed cache entry does not match the freshly trained tokenizer")
	}
}

func TestTrainTokenizerFromCorpusCached_ConcurrentWritersConverge(t *testing.T) {
	dir := t.TempDir()
	cacheDir := filepath.Join(dir, "cache")
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", strings30Words())
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 60, MinFreq: 1}
	cacheCfg := TokenizerTrainCacheConfig{CacheDir: cacheDir}

	const concurrency = 8
	results := make([]TokenizerFile, concurrency)
	errs := make([]error, concurrency)
	var wg sync.WaitGroup
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func(i int) {
			defer wg.Done()
			file, _, err := TrainTokenizerFromCorpusCached(cfg, cacheCfg)
			results[i] = file
			errs[i] = err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: %v", i, err)
		}
	}
	for i := 1; i < concurrency; i++ {
		if !reflect.DeepEqual(results[0].Tokens, results[i].Tokens) {
			t.Fatalf("goroutine %d tokens differ from goroutine 0", i)
		}
		if !reflect.DeepEqual(results[0].Merges, results[i].Merges) {
			t.Fatalf("goroutine %d merges differ from goroutine 0", i)
		}
	}

	key, err := TokenizerTrainCacheKey(cfg)
	if err != nil {
		t.Fatalf("cache key: %v", err)
	}
	final, err := ReadTokenizerFile(tokenizerCacheEntryPath(cacheDir, key))
	if err != nil {
		t.Fatalf("expected a valid final cache entry: %v", err)
	}
	if !reflect.DeepEqual(final.Tokens, results[0].Tokens) {
		t.Fatal("final cache entry does not match the converged in-memory result")
	}
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		t.Fatalf("read cache dir: %v", err)
	}
	tmpLeftovers := 0
	finalEntries := 0
	for _, e := range entries {
		name := e.Name()
		if name == key+".tokenizer.mll" {
			finalEntries++
			continue
		}
		tmpLeftovers++
	}
	if finalEntries != 1 {
		t.Fatalf("expected exactly one final cache entry, found %d among %v", finalEntries, entries)
	}
	_ = tmpLeftovers // temp files may or may not linger depending on scheduling; only the final entry name is load-bearing.
}

func strings30Words() string {
	words := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
	out := ""
	for i := 0; i < 200; i++ {
		out += words[i%len(words)] + " "
		if i%15 == 14 {
			out += "\n"
		}
	}
	return out
}

// --- env var resolution --------------------------------------------------

func TestTokenizerCacheConfig_EnvResolution(t *testing.T) {
	dir := t.TempDir()
	corpusPath := writeTinyTokenizerCorpus(t, dir, "corpus.txt", "banana bandana banana band\n")
	cfg := TokenizerTrainConfig{CorpusPath: corpusPath, VocabSize: 16, MinFreq: 2}

	t.Run("env cache dir used when flag is empty", func(t *testing.T) {
		envDir := filepath.Join(dir, "env-cache")
		t.Setenv(tokenizerCacheDirEnv, envDir)
		t.Setenv(tokenizerCacheDisableEnv, "")
		_, result, err := TrainTokenizerFromCorpusCached(cfg, TokenizerTrainCacheConfig{})
		if err != nil {
			t.Fatalf("train with env cache dir: %v", err)
		}
		if result.Disabled {
			t.Fatal("did not expect the cache to be disabled")
		}
		if _, err := os.Stat(envDir); err != nil {
			t.Fatalf("expected env-configured cache dir to be created: %v", err)
		}
	})

	t.Run("env disable bypasses even with a cache dir set", func(t *testing.T) {
		cacheDir := filepath.Join(dir, "cache-disabled-by-env")
		t.Setenv(tokenizerCacheDirEnv, "")
		t.Setenv(tokenizerCacheDisableEnv, "1")
		_, result, err := TrainTokenizerFromCorpusCached(cfg, TokenizerTrainCacheConfig{CacheDir: cacheDir})
		if err != nil {
			t.Fatalf("train with env disable: %v", err)
		}
		if !result.Disabled {
			t.Fatal("expected EOS_TOKENIZER_CACHE_DISABLE=1 to disable the cache")
		}
	})
}
