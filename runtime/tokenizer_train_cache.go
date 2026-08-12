package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// tokenizerTrainAlgorithmVersion identifies the BPE training algorithm
// (pair-counting and merge-selection rules), independent of
// TokenizerFileVersion (the on-disk artifact format). Bump this if the
// algorithm ever changes in a way that could change output for the same
// corpus/vocab/min-freq, so stale cache entries stop being served.
const tokenizerTrainAlgorithmVersion = "bpe-unique-word-total-order-v1"

// tokenizerCacheDirEnv and tokenizerCacheDisableEnv are read by
// TrainTokenizerFromCorpusCached when the corresponding
// TokenizerTrainCacheConfig field is left at its zero value.
const (
	tokenizerCacheDirEnv     = "EOS_TOKENIZER_CACHE_DIR"
	tokenizerCacheDisableEnv = "EOS_TOKENIZER_CACHE_DISABLE"
)

// TokenizerTrainCacheConfig configures the read-through, content-addressed
// cache used by TrainTokenizerFromCorpusCached.
type TokenizerTrainCacheConfig struct {
	// CacheDir is the directory holding cached tokenizer artifacts, one
	// file per cache key. It is created if missing. Empty means "use the
	// EOS_TOKENIZER_CACHE_DIR environment variable"; if that is also
	// empty and Disable is false, TrainTokenizerFromCorpusCached returns
	// an error rather than silently choosing a default location.
	CacheDir string
	// Disable bypasses the cache entirely: no lookup, no write, and
	// TrainTokenizerFromCorpus always runs. Empty CacheDir is allowed
	// when Disable is true. Also settable via EOS_TOKENIZER_CACHE_DISABLE
	// (1/true) when this field is false.
	Disable bool
}

// TokenizerTrainCacheResult reports what TrainTokenizerFromCorpusCached did.
type TokenizerTrainCacheResult struct {
	// Key is the content-addressed cache key (empty when Disabled).
	Key string
	// Hit is true when an existing cache entry was loaded instead of
	// training.
	Hit bool
	// Path is the cache entry file path (populated on both hit and a
	// fresh write; empty when Disabled).
	Path string
	// Disabled is true when the cache was bypassed for this call.
	Disabled bool
}

// TokenizerTrainCacheKey returns the content-addressed cache key for cfg:
// the SHA256 of the training algorithm version, the tokenizer artifact
// format version, the SHA256 of the corpus file's bytes, the vocab size,
// and the effective min-freq. It deliberately excludes cfg.Workers: worker
// count only changes how training is sharded across goroutines, never the
// trained tokenizer's content (see TokenizerTrainConfig.Workers), so
// varying it must not fragment the cache.
func TokenizerTrainCacheKey(cfg TokenizerTrainConfig) (string, error) {
	if cfg.CorpusPath == "" {
		return "", fmt.Errorf("tokenizer corpus path is required")
	}
	f, err := os.Open(cfg.CorpusPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	corpusHash := sha256.New()
	if _, err := io.Copy(corpusHash, f); err != nil {
		return "", err
	}
	minFreq := cfg.MinFreq
	if minFreq <= 0 {
		minFreq = 2
	}
	keyHash := sha256.New()
	fmt.Fprintf(keyHash, "eos-tokenizer-cache-key/v1\n")
	fmt.Fprintf(keyHash, "algorithm=%s\n", tokenizerTrainAlgorithmVersion)
	fmt.Fprintf(keyHash, "format=%s\n", TokenizerFileVersion)
	fmt.Fprintf(keyHash, "corpus_sha256=%x\n", corpusHash.Sum(nil))
	fmt.Fprintf(keyHash, "vocab_size=%d\n", cfg.VocabSize)
	fmt.Fprintf(keyHash, "min_freq=%d\n", minFreq)
	return hex.EncodeToString(keyHash.Sum(nil)), nil
}

// resolveTokenizerTrainCacheConfig applies environment-variable fallbacks
// for fields left at their zero value.
func resolveTokenizerTrainCacheConfig(cache TokenizerTrainCacheConfig) TokenizerTrainCacheConfig {
	if !cache.Disable {
		if v := strings.TrimSpace(os.Getenv(tokenizerCacheDisableEnv)); v == "1" || strings.EqualFold(v, "true") {
			cache.Disable = true
		}
	}
	if cache.CacheDir == "" {
		cache.CacheDir = strings.TrimSpace(os.Getenv(tokenizerCacheDirEnv))
	}
	return cache
}

// TokenizerCacheConfigOrDisabled resolves cache (applying the
// EOS_TOKENIZER_CACHE_DIR / EOS_TOKENIZER_CACHE_DISABLE environment
// fallbacks; see TokenizerTrainCacheConfig) and, when no cache directory
// ends up configured anywhere, returns a disabled config instead of the
// ambiguous empty-directory configuration that TrainTokenizerFromCorpusCached
// otherwise rejects with an error.
//
// Use this to make tokenizer-training caching strictly opt-in for a call
// site: the cache activates only when the caller passes a non-empty
// CacheDir or EOS_TOKENIZER_CACHE_DIR is set in the environment, and stays
// silently off (identical to calling TrainTokenizerFromCorpus directly)
// otherwise. Callers that want a misconfigured cache to fail loudly
// instead should pass their TokenizerTrainCacheConfig to
// TrainTokenizerFromCorpusCached directly.
func TokenizerCacheConfigOrDisabled(cache TokenizerTrainCacheConfig) TokenizerTrainCacheConfig {
	resolved := resolveTokenizerTrainCacheConfig(cache)
	if resolved.CacheDir == "" {
		resolved.Disable = true
	}
	return resolved
}

// TrainTokenizerFromCorpusCached is a read-through, content-addressed
// wrapper around TrainTokenizerFromCorpus. On a cache hit it loads and
// returns the existing artifact without invoking training at all. On a
// miss it trains normally, then writes the artifact into the cache via a
// write-to-temp-then-rename so concurrent callers (goroutines or separate
// processes) racing on the same key never observe a partially written
// entry; since training is deterministic for a given cache key (see
// TokenizerTrainCacheKey), whichever writer's rename wins is fine, and
// every racer's in-memory result is byte-identical regardless.
//
// A cache entry that fails to parse or validate (for example, a leftover
// partial write from an older, non-atomic writer, or disk corruption) is
// treated as a miss: training proceeds and the entry is overwritten with a
// fresh, valid one.
func TrainTokenizerFromCorpusCached(cfg TokenizerTrainConfig, cache TokenizerTrainCacheConfig) (TokenizerFile, TokenizerTrainCacheResult, error) {
	cache = resolveTokenizerTrainCacheConfig(cache)
	if cache.Disable {
		file, err := TrainTokenizerFromCorpus(cfg)
		return file, TokenizerTrainCacheResult{Disabled: true}, err
	}
	if cache.CacheDir == "" {
		return TokenizerFile{}, TokenizerTrainCacheResult{}, fmt.Errorf("tokenizer cache dir is required unless the cache is disabled (set TokenizerTrainCacheConfig.CacheDir, %s, or Disable)", tokenizerCacheDirEnv)
	}
	key, err := TokenizerTrainCacheKey(cfg)
	if err != nil {
		return TokenizerFile{}, TokenizerTrainCacheResult{}, err
	}
	if err := os.MkdirAll(cache.CacheDir, 0o755); err != nil {
		return TokenizerFile{}, TokenizerTrainCacheResult{Key: key}, err
	}
	entryPath := tokenizerCacheEntryPath(cache.CacheDir, key)
	if file, err := ReadTokenizerFile(entryPath); err == nil {
		return file, TokenizerTrainCacheResult{Key: key, Hit: true, Path: entryPath}, nil
	}

	file, err := TrainTokenizerFromCorpus(cfg)
	if err != nil {
		return TokenizerFile{}, TokenizerTrainCacheResult{Key: key}, err
	}
	if err := writeTokenizerCacheEntry(cache.CacheDir, entryPath, file); err != nil {
		return TokenizerFile{}, TokenizerTrainCacheResult{Key: key}, err
	}
	return file, TokenizerTrainCacheResult{Key: key, Hit: false, Path: entryPath}, nil
}

func tokenizerCacheEntryPath(cacheDir, key string) string {
	return filepath.Join(cacheDir, key+".tokenizer.mll")
}

// writeTokenizerCacheEntry encodes file and publishes it at finalPath by
// writing to a temp file in the same directory (so the following rename is
// on one filesystem and therefore atomic) and renaming it into place.
// Concurrent writers for the same key each produce byte-identical content
// (training is deterministic), so whichever rename lands last simply wins
// harmlessly; no reader ever observes a partial file.
func writeTokenizerCacheEntry(cacheDir, finalPath string, file TokenizerFile) error {
	if err := file.Validate(); err != nil {
		return err
	}
	data, err := encodeTokenizerMLL(file)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(cacheDir, ".tokenizer-cache-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	_, writeErr := tmp.Write(data)
	if writeErr == nil {
		writeErr = tmp.Sync()
	}
	closeErr := tmp.Close()
	if writeErr != nil {
		os.Remove(tmpPath)
		return writeErr
	}
	if closeErr != nil {
		os.Remove(tmpPath)
		return closeErr
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		os.Remove(tmpPath)
		return err
	}
	return nil
}
