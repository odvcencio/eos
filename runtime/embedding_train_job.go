package eosruntime

import (
	"fmt"
	"sort"
	"strings"
)

type EmbeddingCorpusTrainConfig struct {
	TokenizerPath      string
	TokenizerVocabSize int
	TokenizerMinFreq   int
	// TokenizerWorkers sets the parallel worker count for BPE tokenizer
	// training. <= 0 means "auto"; see ResolveTokenizerTrainWorkers. It
	// never changes the trained tokenizer's content.
	TokenizerWorkers int
	// TokenizerCacheDir and TokenizerCacheDisable configure the
	// content-addressed tokenizer training cache; see
	// TrainTokenizerFromCorpusCached and TokenizerTrainCacheConfig. Empty
	// TokenizerCacheDir falls back to EOS_TOKENIZER_CACHE_DIR.
	TokenizerCacheDir     string
	TokenizerCacheDisable bool
	TrainPairsPath        string
	EvalPairsPath         string
	Mining                EmbeddingTextMiningConfig
	Run                   EmbeddingTrainRunConfig
}

type EmbeddingCorpusTrainPaths struct {
	TokenizerPath  string
	TrainPairsPath string
	EvalPairsPath  string
	Package        EmbeddingTrainPackagePaths
}

func defaultEmbeddingTrainPackagePaths(artifactPath string) EmbeddingTrainPackagePaths {
	return EmbeddingTrainPackagePaths{
		ArtifactPath:          artifactPath,
		EmbeddingManifestPath: ResolveEmbeddingManifestPath(artifactPath),
		TokenizerPath:         DefaultTokenizerPath(artifactPath),
		WeightFilePath:        DefaultWeightFilePath(artifactPath),
		MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
		TrainManifestPath:     ResolveEmbeddingTrainManifestPath(artifactPath),
		CheckpointPath:        DefaultEmbeddingCheckpointPath(artifactPath),
		TrainProfilePath:      DefaultEmbeddingTrainProfilePath(artifactPath),
		PackageManifestPath:   ResolvePackageManifestPath(artifactPath),
	}
}

// TrainEmbeddingPackageFromContrastiveFiles reloads a packaged trainer, fits it on a JSONL contrastive dataset, and writes the updated package back.
func TrainEmbeddingPackageFromContrastiveFiles(artifactPath, trainPath, evalPath string, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, EmbeddingTrainPackagePaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	defer trainer.Close()
	if cfg.EvalOnly && evalPath == "" && cfg.ScoreSpectrumEvalPath == "" && !cfg.ListwiseGeometryTrain {
		evalPath = trainPath
		trainPath = ""
	}
	if cfg.VectorDistillTrain {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("vector distillation training requires text tokenization; remove --no-tokenizer or set --tokenizer")
	}
	if cfg.ScoreSpectrumTrain {
		var trainSet []EmbeddingScoreSpectrumExample
		if !cfg.EvalOnly {
			trainSet, err = ReadEmbeddingScoreSpectrumExamplesFile(trainPath, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train score-spectrum dataset: %w", err)
			}
			trainer.SetScoreSpectrumLineage(ScoreSpectrumPolicyFromExamples(trainSet))
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalPairs, err = ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval pair dataset: %w", err)
			}
		}
		if cfg.ScoreSpectrumEvalPath != "" {
			cfg.ScoreSpectrumEval, err = ReadEmbeddingScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read score-spectrum eval dataset: %w", err)
			}
		}
		summary, err := trainer.FitScoreSpectrum(trainSet, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.ListwiseGeometryTrain {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("listwise geometry training requires text tokenization; remove --no-tokenizer or set --tokenizer")
	}
	if cfg.HardNegativeTrain {
		var trainSet []EmbeddingHardNegativeExample
		if !cfg.EvalOnly {
			trainSet, err = ReadEmbeddingHardNegativeExamplesFile(trainPath)
			if err != nil {
				trainPairs, pairErr := ReadEmbeddingPairExamplesFile(trainPath)
				if pairErr != nil {
					return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train hard-negative dataset: %w", err)
				}
				trainSet, err = BuildEmbeddingHardNegativeExamplesFromPairs(trainPairs, cfg.HardNegativesPerQuery)
				if err != nil {
					return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("build train hard-negative dataset: %w", err)
				}
			}
			trainSet = limitHardNegativeExamples(trainSet, cfg.HardNegativesPerQuery)
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalPairs, err = ReadEmbeddingHardNegativeEvalPairsFile(evalPath, cfg.HardNegativesPerQuery)
			if err != nil {
				evalSet, contrastiveErr := ReadEmbeddingContrastiveExamplesFile(evalPath)
				if contrastiveErr != nil {
					return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval pair dataset: %w", err)
				}
				evalPairs = expandContrastiveExamples(evalSet)
			}
		}
		summary, err := trainer.FitHardNegatives(trainSet, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.PairwiseTrain {
		var trainPairs []EmbeddingPairExample
		if !cfg.EvalOnly {
			trainPairs, err = ReadEmbeddingPairExamplesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train pair dataset: %w", err)
			}
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalPairs, err = ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval pair dataset: %w", err)
			}
		}
		summary, err := trainer.Fit(trainPairs, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	var trainSet []EmbeddingContrastiveExample
	if !cfg.EvalOnly {
		trainSet, err = ReadEmbeddingContrastiveExamplesFile(trainPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train dataset: %w", err)
		}
	}
	var evalSet []EmbeddingContrastiveExample
	var evalPairs []EmbeddingPairExample
	if evalPath != "" {
		evalSet, err = ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, err = ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval dataset: %w", err)
			}
		}
	}
	var summary EmbeddingTrainRunSummary
	if len(evalPairs) > 0 && len(evalSet) == 0 {
		if cfg.EvalOnly {
			summary, err = trainer.Fit(nil, evalPairs, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		} else {
			// Pairwise eval data drives per-epoch selection, early stopping,
			// and best restore through cfg.EvalPairs; an empty contrastive
			// eval set would otherwise silently disable all of them.
			cfg.EvalPairs = evalPairs
			summary, err = trainer.FitContrastive(trainSet, nil, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		}
	} else {
		summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
	}
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	return summary, paths, nil
}

// TrainEmbeddingPackageFromTextContrastiveFiles reloads a packaged trainer, tokenizes text-pair JSONL with a Eos tokenizer file,
// fits it, and writes the updated package back.
func TrainEmbeddingPackageFromTextContrastiveFiles(artifactPath, tokenizerPath, trainPath, evalPath string, cfg EmbeddingTrainRunConfig) (EmbeddingTrainRunSummary, EmbeddingTrainPackagePaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	defer trainer.Close()
	if cfg.EvalOnly && evalPath == "" && cfg.ScoreSpectrumEvalPath == "" && !cfg.ListwiseGeometryTrain {
		evalPath = trainPath
		trainPath = ""
	}
	tokenizerFile, err := ReadTokenizerFile(tokenizerPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read tokenizer: %w", err)
	}
	tokenizer, err := NewBPETokenizer(tokenizerFile, trainer.manifest.Tokenizer)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("build tokenizer: %w", err)
	}
	tokenCache := embeddingTextTokenCache{}
	if cfg.ScoreSpectrumTrain {
		var trainSet []EmbeddingScoreSpectrumExample
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingTextScoreSpectrumExamplesFile(trainPath, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text score-spectrum dataset: %w", err)
			}
			trainSet, err = TokenizeEmbeddingTextScoreSpectrumExamples(trainText, tokenizer, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train score-spectrum dataset: %w", err)
			}
			trainer.SetScoreSpectrumLineage(ScoreSpectrumPolicyFromExamples(trainSet))
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalText, err := ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text pair dataset: %w", err)
			}
			evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval pair dataset: %w", err)
			}
		}
		if cfg.ScoreSpectrumEvalPath != "" {
			evalText, err := ReadEmbeddingTextScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read score-spectrum eval text dataset: %w", err)
			}
			cfg.ScoreSpectrumEval, err = TokenizeEmbeddingTextScoreSpectrumExamples(evalText, tokenizer, EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize score-spectrum eval dataset: %w", err)
			}
		}
		summary, err := trainer.FitScoreSpectrum(trainSet, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.ListwiseGeometryTrain {
		var trainSet []EmbeddingTokenizedListwiseGeometryBatch
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingListwiseGeometryBatchesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train listwise geometry dataset: %w", err)
			}
			trainSet, err = TokenizeEmbeddingListwiseGeometryBatches(trainText, tokenizer)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train listwise geometry dataset: %w", err)
			}
			trainer.SetListwiseGeometryLineage(ListwiseGeometryPolicyFromBatches(trainSet))
		}
		if cfg.EvalOnly && evalPath == "" && len(cfg.ListwiseGeometryEval) == 0 {
			evalText, err := ReadEmbeddingListwiseGeometryBatchesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read listwise geometry eval dataset: %w", err)
			}
			cfg.ListwiseGeometryEval, err = TokenizeEmbeddingListwiseGeometryBatches(evalText, tokenizer)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize listwise geometry eval dataset: %w", err)
			}
			trainPath = ""
		}
		if evalPath != "" {
			evalText, err := ReadEmbeddingListwiseGeometryBatchesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read listwise geometry eval dataset: %w", err)
			}
			cfg.ListwiseGeometryEval, err = TokenizeEmbeddingListwiseGeometryBatches(evalText, tokenizer)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize listwise geometry eval dataset: %w", err)
			}
		}
		var evalPairs []EmbeddingPairExample
		summary, err := trainer.FitListwiseGeometry(trainSet, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		if cfg.EvalOnly && len(trainSet) == 0 {
			return summary, defaultEmbeddingTrainPackagePaths(artifactPath), nil
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.VectorDistillTrain {
		var trainSet []EmbeddingTokenizedVectorDistillExample
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingVectorDistillExamplesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train vector-distill dataset: %w", err)
			}
			trainSet, err = TokenizeEmbeddingVectorDistillExamples(trainText, tokenizer)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train vector-distill dataset: %w", err)
			}
		}
		summary, err := trainer.FitVectorDistill(trainSet, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		if cfg.EvalOnly && len(trainSet) == 0 {
			return summary, defaultEmbeddingTrainPackagePaths(artifactPath), nil
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.HardNegativeTrain {
		var trainSet []EmbeddingHardNegativeExample
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingTextHardNegativeExamplesFile(trainPath)
			if err != nil {
				trainTextPairs, pairErr := ReadEmbeddingTextPairExamplesFile(trainPath)
				if pairErr != nil {
					return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text hard-negative dataset: %w", err)
				}
				trainText, err = BuildEmbeddingTextHardNegativeExamplesFromPairs(trainTextPairs, cfg.HardNegativesPerQuery)
				if err != nil {
					return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("build train text hard-negative dataset: %w", err)
				}
			}
			trainSet, err = tokenizeEmbeddingTextHardNegativeExamples(trainText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train hard-negative dataset: %w", err)
			}
			trainSet = limitHardNegativeExamples(trainSet, cfg.HardNegativesPerQuery)
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalText, err := ReadEmbeddingTextHardNegativeEvalPairsFile(evalPath, cfg.HardNegativesPerQuery)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text pair dataset: %w", err)
			}
			evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval pair dataset: %w", err)
			}
		}
		summary, err := trainer.FitHardNegatives(trainSet, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	if cfg.PairwiseTrain {
		var trainPairs []EmbeddingPairExample
		if !cfg.EvalOnly {
			trainText, err := ReadEmbeddingTextPairExamplesFile(trainPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text pair dataset: %w", err)
			}
			trainPairs, err = tokenizeEmbeddingTextPairExamples(trainText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train pair dataset: %w", err)
			}
		}
		var evalPairs []EmbeddingPairExample
		if evalPath != "" {
			evalText, err := ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text pair dataset: %w", err)
			}
			evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval pair dataset: %w", err)
			}
		}
		summary, err := trainer.Fit(trainPairs, evalPairs, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		paths, err := trainer.WriteTrainingPackage(artifactPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
		return summary, paths, nil
	}
	var trainSet []EmbeddingContrastiveExample
	if !cfg.EvalOnly {
		trainText, err := ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read train text dataset: %w", err)
		}
		trainSet, err = tokenizeEmbeddingTextContrastiveExamples(trainText, tokenizer, tokenCache, false)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize train dataset: %w", err)
		}
	}
	var (
		evalSet   []EmbeddingContrastiveExample
		evalPairs []EmbeddingPairExample
	)
	if evalPath != "" {
		evalText, err := ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("read eval text dataset: %w", err)
		}
		evalPairs, err = tokenizeEmbeddingTextPairExamples(evalText, tokenizer, tokenCache, false)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval dataset: %w", err)
		}
		allPositive := true
		for _, example := range evalText {
			if example.Target <= 0 {
				allPositive = false
				break
			}
		}
		if allPositive {
			evalSet, err = tokenizeEmbeddingTextContrastiveExamples(toTextContrastiveExamples(evalText), tokenizer, tokenCache, false)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, fmt.Errorf("tokenize eval contrastive dataset: %w", err)
			}
		}
	}
	var summary EmbeddingTrainRunSummary
	if len(evalPairs) > 0 && len(evalSet) == 0 {
		if cfg.EvalOnly {
			summary, err = trainer.Fit(nil, evalPairs, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		} else {
			// Pairwise eval data drives per-epoch selection, early stopping,
			// and best restore through cfg.EvalPairs; an empty contrastive
			// eval set would otherwise silently disable all of them.
			cfg.EvalPairs = evalPairs
			summary, err = trainer.FitContrastive(trainSet, nil, cfg)
			if err != nil {
				return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
			}
		}
	} else {
		summary, err = trainer.FitContrastive(trainSet, evalSet, cfg)
		if err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
		}
	}
	paths, err := trainer.WriteTrainingPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingTrainPackagePaths{}, err
	}
	return summary, paths, nil
}

func TrainEmbeddingPackageFromCorpusFile(artifactPath, corpusPath string, cfg EmbeddingCorpusTrainConfig) (EmbeddingTrainRunSummary, EmbeddingCorpusTrainPaths, error) {
	trainer, err := LoadEmbeddingTrainerPackage(artifactPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	defer trainer.Close()

	tokenizerPath := cfg.TokenizerPath
	if tokenizerPath == "" {
		tokenizerPath = DefaultTokenizerPath(artifactPath)
	}
	vocabSize := cfg.TokenizerVocabSize
	if vocabSize == 0 {
		vocabSize = trainer.manifest.Tokenizer.VocabSize
	}
	minVocabSize, err := MinimumTokenizerVocabSizeForCorpus(corpusPath)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if vocabSize < minVocabSize {
		vocabSize = minVocabSize
	}
	if vocabSize <= 0 {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, fmt.Errorf("tokenizer vocab size must be set via config or embedding manifest")
	}
	if manifestVocabSize := trainer.manifest.Tokenizer.VocabSize; manifestVocabSize > 0 && vocabSize < manifestVocabSize {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, fmt.Errorf("tokenizer vocab size %d would shrink package vocab contract %d; checkpoint tensor resize is not supported by corpus tokenizer training", vocabSize, manifestVocabSize)
	}
	minFreq := cfg.TokenizerMinFreq
	if minFreq <= 0 {
		minFreq = 2
	}
	tokenizer, _, err := TrainTokenizerFromCorpusCached(
		TokenizerTrainConfig{
			CorpusPath: corpusPath,
			VocabSize:  vocabSize,
			MinFreq:    minFreq,
			Workers:    cfg.TokenizerWorkers,
		},
		TokenizerCacheConfigOrDisabled(TokenizerTrainCacheConfig{
			CacheDir: cfg.TokenizerCacheDir,
			Disable:  cfg.TokenizerCacheDisable,
		}),
	)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	tokenizer, err = PadTokenizerFileVocab(tokenizer, vocabSize)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if err := tokenizer.WriteFile(tokenizerPath); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	if err := SyncEmbeddingTokenizerVocab(artifactPath, vocabSize); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	trainPairs, evalPairs, err := MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, cfg.Mining)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	trainPairsPath := cfg.TrainPairsPath
	if trainPairsPath == "" {
		trainPairsPath = DefaultMinedTrainPairsPath(artifactPath)
	}
	evalPairsPath := cfg.EvalPairsPath
	if evalPairsPath == "" {
		evalPairsPath = DefaultMinedEvalPairsPath(artifactPath)
	}
	if err := WriteEmbeddingTextContrastiveExamplesFile(trainPairsPath, trainPairs); err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	effectiveEvalPath := ""
	if len(evalPairs) > 0 {
		if err := WriteEmbeddingTextPairExamplesFile(evalPairsPath, evalPairs); err != nil {
			return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
		}
		effectiveEvalPath = evalPairsPath
	}
	summary, paths, err := TrainEmbeddingPackageFromTextContrastiveFiles(artifactPath, tokenizerPath, trainPairsPath, effectiveEvalPath, cfg.Run)
	if err != nil {
		return EmbeddingTrainRunSummary{}, EmbeddingCorpusTrainPaths{}, err
	}
	return summary, EmbeddingCorpusTrainPaths{
		TokenizerPath:  tokenizerPath,
		TrainPairsPath: trainPairsPath,
		EvalPairsPath:  effectiveEvalPath,
		Package:        paths,
	}, nil
}

func toTextContrastiveExamples(examples []EmbeddingTextPairExample) []EmbeddingTextContrastiveExample {
	out := make([]EmbeddingTextContrastiveExample, 0, len(examples))
	for _, example := range examples {
		if example.Target <= 0 {
			continue
		}
		out = append(out, EmbeddingTextContrastiveExample{
			Query:    example.Query,
			Positive: example.Right,
		})
	}
	return out
}

func ScoreSpectrumPolicyFromExamples(examples []EmbeddingScoreSpectrumExample) EmbeddingScoreSpectrumPolicy {
	policy := EmbeddingScoreSpectrumPolicy{
		ScoreSpectrumTrain:    len(examples) > 0,
		ReleaseTrainAllowed:   len(examples) > 0,
		CommercialUseAllowed:  len(examples) > 0,
		ScoreSpectrumRowCount: len(examples),
	}
	hashes := map[string]bool{}
	for _, example := range examples {
		if strings.TrimSpace(example.SourceArtifactHash) != "" {
			hashes[strings.TrimSpace(example.SourceArtifactHash)] = true
		}
		if example.TrainAllowedForResearch {
			policy.TrainAllowedForResearch = true
		}
		if !example.ReleaseTrainAllowed {
			policy.ReleaseTrainAllowed = false
		}
		if !example.CommercialUseAllowed {
			policy.CommercialUseAllowed = false
		}
		if example.TrainAllowedForResearch && (!example.ReleaseTrainAllowed || !example.CommercialUseAllowed) {
			policy.ScoreSpectrumResearchOnly = true
		}
	}
	policy.SourceArtifactHashes = make([]string, 0, len(hashes))
	for hash := range hashes {
		policy.SourceArtifactHashes = append(policy.SourceArtifactHashes, hash)
	}
	sort.Strings(policy.SourceArtifactHashes)
	return policy
}

func ListwiseGeometryPolicyFromBatches(batches []EmbeddingTokenizedListwiseGeometryBatch) EmbeddingListwiseGeometryPolicy {
	policy := EmbeddingListwiseGeometryPolicy{
		ListwiseGeometryTrain:      len(batches) > 0,
		ReleaseTrainAllowed:        len(batches) > 0,
		CommercialUseAllowed:       len(batches) > 0,
		ListwiseGeometryBatchCount: len(batches),
	}
	hashes := map[string]bool{}
	for _, batch := range batches {
		if strings.TrimSpace(batch.SourceArtifactHash) != "" {
			hashes[strings.TrimSpace(batch.SourceArtifactHash)] = true
		}
		if batch.TrainAllowedForResearch {
			policy.TrainAllowedForResearch = true
		}
		if !batch.ReleaseTrainAllowed {
			policy.ReleaseTrainAllowed = false
		}
		if !batch.CommercialUseAllowed {
			policy.CommercialUseAllowed = false
		}
		if batch.TrainAllowedForResearch && (!batch.ReleaseTrainAllowed || !batch.CommercialUseAllowed) {
			policy.ListwiseGeometryResearchOnly = true
		}
	}
	policy.SourceArtifactHashes = make([]string, 0, len(hashes))
	for hash := range hashes {
		policy.SourceArtifactHashes = append(policy.SourceArtifactHashes, hash)
	}
	sort.Strings(policy.SourceArtifactHashes)
	return policy
}
