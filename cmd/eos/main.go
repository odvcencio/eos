package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/compiler"
	"m31labs.dev/eos/models"
	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backend"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
)

func main() {
	stopProfile, err := startOptionalProfiles()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer stopProfile()
	if err := run(os.Args[1:]); err != nil {
		printCommandError(os.Stderr, err)
		os.Exit(1)
	}
}

func startOptionalProfiles() (func(), error) {
	cpuPath := eosEnv("EOS_CPU_PROFILE")
	memPath := eosEnv("EOS_MEM_PROFILE")
	var cpuFile *os.File
	if cpuPath != "" {
		file, err := os.Create(cpuPath)
		if err != nil {
			return nil, fmt.Errorf("create CPU profile %q: %w", cpuPath, err)
		}
		if err := pprof.StartCPUProfile(file); err != nil {
			_ = file.Close()
			return nil, fmt.Errorf("start CPU profile %q: %w", cpuPath, err)
		}
		cpuFile = file
	}
	return func() {
		if cpuFile != nil {
			pprof.StopCPUProfile()
			_ = cpuFile.Close()
			fmt.Fprintf(os.Stderr, "cpu profile: %s\n", cpuPath)
		}
		if memPath != "" {
			runtime.GC()
			file, err := os.Create(memPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
				return
			}
			if err := pprof.WriteHeapProfile(file); err != nil {
				fmt.Fprintf(os.Stderr, "write memory profile %q: %v\n", memPath, err)
			}
			_ = file.Close()
			fmt.Fprintf(os.Stderr, "memory profile: %s\n", memPath)
		}
	}, nil
}

func eosEnv(name string) string {
	if value, ok := os.LookupEnv(name); ok {
		return value
	}
	return ""
}

func run(args []string) error {
	if len(args) == 0 {
		printUsage()
		return nil
	}

	switch args[0] {
	case "version":
		fmt.Println("eos dev")
		return nil
	case "compile":
		return runCompile(args[1:])
	case "graph":
		return runGraph(args[1:])
	case "kernels":
		return runKernels(args[1:])
	case "doctor":
		return runDoctor(args[1:])
	case "run":
		return runArtifact(args[1:])
	case "embed-text":
		return runEmbedText(args[1:])
	case "embed-pretrained-bert-package":
		return runEmbedPretrainedBERTPackage(args[1:])
	case "default-embedder":
		return runDefaultEmbedder(args[1:])
	case "imported-embedder-candidate":
		return runImportedEmbedderCandidate(args[1:])
	case "export-retrieval-vectors":
		return runExportRetrievalVectors(args[1:])
	case "export-pretrained-bert-retrieval-vectors":
		return runExportPretrainedBERTRetrievalVectors(args[1:])
	case "fit-pretrained-bert-projection-head":
		return runFitPretrainedBERTProjectionHead(args[1:])
	case "export-sparse-token-pool-vectors":
		return runExportSparseTokenPoolVectors(args[1:])
	case "export-sparse-encoder-vectors":
		return runExportSparseEncoderVectors(args[1:])
	case "export-sparse-lexical-labels":
		return runExportSparseLexicalLabels(args[1:])
	case "eval-sparse-lexical-labels":
		return runEvalSparseLexicalLabels(args[1:])
	case "fit-sparse-lexical-head":
		return runFitSparseLexicalHead(args[1:])
	case "eval-sparse-lexical-head":
		return runEvalSparseLexicalHead(args[1:])
	case "eval-sparse-lexical-head-vectors-hybrid":
		return runEvalSparseLexicalHeadVectorsHybrid(args[1:])
	case "fit-sparse-lexical-projection-head":
		return runFitSparseLexicalProjectionHead(args[1:])
	case "eval-sparse-lexical-projection-head-vectors-hybrid":
		return runEvalSparseLexicalProjectionHeadVectorsHybrid(args[1:])
	case "fit-sparse-lexical-linear-head":
		return runFitSparseLexicalLinearHead(args[1:])
	case "eval-sparse-lexical-linear-head-vectors-hybrid":
		return runEvalSparseLexicalLinearHeadVectorsHybrid(args[1:])
	case "export-timeseries-vectors":
		return runExportTimeSeriesVectors(args[1:])
	case "export-event-trace-vectors":
		return runExportEventTraceVectors(args[1:])
	case "eval-retrieval":
		return runEvalRetrieval(args[1:])
	case "eval-retrieval-hybrid":
		return runEvalRetrievalHybrid(args[1:])
	case "eval-retrieval-turboquant":
		return runEvalRetrievalTurboQuant(args[1:])
	case "eval-retrieval-vectors-turboquant":
		return runEvalRetrievalVectorsTurboQuant(args[1:])
	case "eval-retrieval-multivector-turboquant":
		return runEvalRetrievalMultiVectorTurboQuant(args[1:])
	case "eval-retrieval-vectors":
		return runEvalRetrievalVectors(args[1:])
	case "eval-retrieval-vectors-hybrid":
		return runEvalRetrievalVectorsHybrid(args[1:])
	case "eval-retrieval-bm25":
		return runEvalRetrievalBM25(args[1:])
	case "mine-retrieval-hard-negatives":
		return runMineRetrievalHardNegatives(args[1:])
	case "mine-retrieval-model-hard-negatives":
		return runMineRetrievalModelHardNegatives(args[1:])
	case "mine-retrieval-compact-hard-negatives":
		return runMineRetrievalCompactHardNegatives(args[1:])
	case "demo":
		return runDemo(args[1:])
	case "inspect":
		return runInspect(args[1:])
	case "inspect-pretrained-bert-package":
		return runInspectPretrainedBERTPackage(args[1:])
	case "export-mll":
		return runExportMLL(args[1:])
	case "import-pretrained-bert":
		return runImportPretrainedBERT(args[1:])
	case "init-model":
		return runInitModel(args[1:])
	case "init-mirage":
		return runInitMirage(args[1:])
	case "init-train":
		return runInitTrain(args[1:])
	case "rename-embed":
		return runRenameEmbed(args[1:])
	case "train-tokenizer":
		return runTrainTokenizer(args[1:])
	case "tokenize-embed":
		return runTokenizeEmbed(args[1:])
	case "mine-text-pairs":
		return runMineTextPairs(args[1:])
	case "export-teacher-score-requests":
		return runExportTeacherScoreRequests(args[1:])
	case "import-teacher-scores":
		return runImportTeacherScores(args[1:])
	case "score-teacher-hard-negatives":
		return runScoreTeacherHardNegatives(args[1:])
	case "audit-teacher-scores":
		return runAuditTeacherScores(args[1:])
	case "filter-teacher-scores":
		return runFilterTeacherScores(args[1:])
	case "relabel-teacher-negatives":
		return runRelabelTeacherNegatives(args[1:])
	case "sample-corpus-negatives":
		return runSampleCorpusNegatives(args[1:])
	case "plan-sparse-attention":
		return runPlanSparseAttention(args[1:])
	case "calibrate-sparse-routing":
		return runCalibrateSparseRouting(args[1:])
	case "smoke-sparse-embedding-encoder":
		return runSmokeSparseEmbeddingEncoder(args[1:])
	case "plan-multivector-storage":
		return runPlanMultiVectorStorage(args[1:])
	case "train-embed":
		return runTrainEmbed(args[1:])
	case "train-corpus":
		return runTrainCorpus(args[1:])
	case "compare-train-metrics":
		return runCompareTrainMetrics(args[1:])
	case "compare-retrieval-metrics":
		return runCompareRetrievalMetrics(args[1:])
	case "diagnose-train-metrics":
		return runDiagnoseTrainMetrics(args[1:])
	case "gate-train-metrics":
		return runGateTrainMetrics(args[1:])
	case "gate-retrieval-metrics":
		return runGateRetrievalMetrics(args[1:])
	case "gate-scoreboard", "gate-retrieval-scoreboard":
		return runGateScoreboard(args[1:])
	default:
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func runCompile(args []string) error {
	fs := flag.NewFlagSet("compile", flag.ContinueOnError)
	bundleDir := fs.String("bundle", "", "write inspection bundle sidecar directory")
	validateKernels := fs.Bool("validate-kernels", false, "record Prism kernel source validation status in the bundle manifest")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos compile [--bundle dir] [--validate-kernels] <source.eos> [output.mll]")
	}
	srcPath := fs.Arg(0)
	outPath := defaultArtifactPath(srcPath)
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		outPath = fs.Arg(1)
	}

	src, err := os.ReadFile(srcPath)
	if err != nil {
		return err
	}
	moduleName := strings.TrimSuffix(filepath.Base(srcPath), filepath.Ext(srcPath))
	bundle, err := compiler.Build(src, compiler.Options{ModuleName: moduleName})
	if err != nil {
		return attachSource(srcPath, src, err)
	}
	if err := eosartifact.WriteFile(outPath, bundle.Artifact); err != nil {
		return err
	}
	if *bundleDir != "" {
		if err := writeCompileBundle(*bundleDir, srcPath, src, outPath, bundle, *validateKernels); err != nil {
			fmt.Fprintf(os.Stderr, "bundle warning: %v\n", err)
		} else {
			fmt.Printf("bundle: %s\n", *bundleDir)
		}
	}

	fmt.Printf("compiled %q -> %q\n", srcPath, outPath)
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	return nil
}

func runDemo(args []string) error {
	name := "tiny_embed"
	if len(args) > 0 && args[0] != "" {
		name = args[0]
	}
	preset := compiler.PresetTinyEmbed
	switch name {
	case "mirage_v1":
		mod, err := models.DefaultMirageV1Module(models.MirageV1Config{})
		if err != nil {
			return err
		}
		return runDemoModule(mod)
	case "tiny_embed_pooled":
		preset = compiler.PresetTinyEmbedPooled
	case "tiny_embed_masked_pooled":
		preset = compiler.PresetTinyEmbedMaskedPooled
	case "tiny_decode":
		preset = compiler.PresetTinyDecode
	case "tiny_score":
		preset = compiler.PresetTinyScore
	case "tiny_rerank":
		preset = compiler.PresetTinyRerank
	case "tiny_select":
		preset = compiler.PresetTinySelect
	case "tiny_retrieve":
		preset = compiler.PresetTinyRetrieve
	case "tiny_candidates":
		preset = compiler.PresetTinyCandidates
	case "tiny_batch_candidates":
		preset = compiler.PresetTinyBatchCandidates
	case "tiny_packed_candidates":
		preset = compiler.PresetTinyPackedCandidates
	}

	bundle, err := compiler.Build(nil, compiler.Options{ModuleName: name, Preset: preset})
	if err != nil {
		return err
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), bundle.Artifact, stubLoadOptions(bundle.Artifact)...)
	if err != nil {
		return err
	}

	entryName := defaultEntryName(bundle.Artifact)
	entry, err := entryPointByName(bundle.Artifact, entryName)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded module %q for backend %q\n", bundle.Artifact.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n",
		len(bundle.Artifact.EntryPoints), len(bundle.Artifact.Steps), len(bundle.Artifact.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(bundle.Artifact.Kernels))
	fmt.Printf("params: %d\n", len(bundle.Artifact.Params))
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(bundle.Artifact.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runDemoModule(mod *eosartifact.Module) error {
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	entry, err := entryPointByName(mod, entryName)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}
	fmt.Printf("loaded module %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("kernel ops: %d\n", totalKernelOps(mod.Kernels))
	fmt.Printf("params: %d\n", len(mod.Params))
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runArtifact(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: eos run <artifact.mll> [entry]")
	}
	path := args[0]
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		return err
	}
	entryName := defaultEntryName(mod)
	if len(args) > 1 && args[1] != "" {
		entryName = args[1]
	}
	entry, err := entryPointByName(mod, entryName)
	if err != nil {
		return err
	}
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	prog, err := rt.Load(context.Background(), mod, stubLoadOptions(mod)...)
	if err != nil {
		return err
	}
	result, err := prog.Run(context.Background(), backend.Request{
		Entry:  entryName,
		Inputs: stubInputs(entry),
	})
	if err != nil {
		return err
	}

	fmt.Printf("loaded artifact %q for backend %q\n", mod.Name, prog.Backend())
	fmt.Printf("ran entrypoint: %s\n", entryName)
	fmt.Printf("outputs: %s\n", strings.Join(sortedValueKeys(result.Outputs), ", "))
	fmt.Printf("output summary: %s\n", strings.Join(outputSummaries(result.Outputs), "; "))
	fmt.Printf("trace steps: %d\n", len(result.Trace))
	return nil
}

func runEmbedText(args []string) error {
	fs := flag.NewFlagSet("embed-text", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	role := fs.String("role", eosruntime.EmbeddingRoleRaw, "embedding role: raw, query, or document")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos embed-text [--role raw|query|document] <artifact.mll> <text...>")
	}
	path := fs.Arg(0)
	text := strings.Join(fs.Args()[1:], " ")
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), path)
	if err != nil {
		return err
	}
	tokens, _, err := model.TokenizeText(text)
	if err != nil {
		return err
	}
	result, err := model.EmbedTextWithRole(context.Background(), text, *role)
	if err != nil {
		return err
	}
	if result.Embeddings == nil {
		return fmt.Errorf("embedding output tensor is nil")
	}
	manifest := model.Manifest()
	fmt.Printf("loaded embedding %q for backend %q\n", displayManifestName(manifest.Name), model.Backend())
	fmt.Printf("tokens: %d\n", len(tokens))
	fmt.Printf("output: %s\n", displayManifestName(result.OutputName))
	fmt.Printf("embedding: %s%v\n", result.Embeddings.DType, result.Embeddings.Shape)
	return nil
}

func runDefaultEmbedder(args []string) error {
	fs := flag.NewFlagSet("default-embedder", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	root := fs.String("root", "", "repository root containing assets/corkscrewdb-default-embedder")
	pathOnly := fs.Bool("path-only", false, "print only the sealed MLL artifact path")
	verify := fs.Bool("verify", false, "verify artifact and tokenizer SHA256 hashes")
	jsonOut := fs.Bool("json", false, "write JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos default-embedder [--root dir] [--path-only] [--verify] [--json]")
	}

	info, err := models.DefaultEmbedderAssetInfo(*root)
	if err != nil {
		return err
	}
	if *pathOnly && !*jsonOut {
		fmt.Println(info.ArtifactPath)
		return nil
	}

	var verification *models.DefaultEmbedderAssetVerification
	if *verify {
		report, err := models.VerifyDefaultEmbedderAsset(*root)
		verification = &report
		if err != nil {
			if *jsonOut {
				_ = writeJSON(os.Stdout, struct {
					Asset        models.DefaultEmbedderAsset              `json:"asset"`
					Verification *models.DefaultEmbedderAssetVerification `json:"verification,omitempty"`
				}{Asset: info, Verification: verification})
			}
			return err
		}
	}
	if *jsonOut {
		return writeJSON(os.Stdout, struct {
			Asset        models.DefaultEmbedderAsset              `json:"asset"`
			Verification *models.DefaultEmbedderAssetVerification `json:"verification,omitempty"`
		}{Asset: info, Verification: verification})
	}

	fmt.Printf("asset_id: %s\n", info.AssetID)
	fmt.Printf("model: %s\n", info.ModelName)
	fmt.Printf("artifact: %s\n", info.ArtifactPath)
	fmt.Printf("tokenizer: %s\n", info.TokenizerPath)
	fmt.Printf("manifest: %s\n", info.ManifestPath)
	if verification != nil {
		for _, check := range verification.Files {
			status := "FAIL"
			if check.OK {
				status = "OK"
			}
			fmt.Printf("sha256 %s: %s %s\n", check.Role, status, check.SHA256)
		}
	}
	return nil
}

func runImportedEmbedderCandidate(args []string) error {
	fs := flag.NewFlagSet("imported-embedder-candidate", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	root := fs.String("root", "", "repository root containing candidate run artifacts")
	packagePath := fs.String("package", "", "override imported pretrained BERT package path")
	pathOnly := fs.Bool("path-only", false, "print only the imported candidate package path")
	verify := fs.Bool("verify", false, "verify package SHA256 hash and package identity")
	jsonOut := fs.Bool("json", false, "write JSON output")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos imported-embedder-candidate (--package path | --root dir) [--path-only] [--verify] [--json]")
	}

	info, err := models.ImportedEmbedderCandidateAssetInfo(*root, *packagePath)
	if err != nil {
		return err
	}
	if *pathOnly && !*jsonOut {
		fmt.Println(info.PackagePath)
		return nil
	}

	var verification *models.ImportedEmbedderCandidateVerification
	if *verify {
		report, err := models.VerifyImportedEmbedderCandidate(*root, *packagePath)
		verification = &report
		if err != nil {
			if *jsonOut {
				_ = writeJSON(os.Stdout, struct {
					Asset        models.ImportedEmbedderCandidateAsset         `json:"asset"`
					Verification *models.ImportedEmbedderCandidateVerification `json:"verification,omitempty"`
				}{Asset: info, Verification: verification})
			}
			return err
		}
	}
	if *jsonOut {
		return writeJSON(os.Stdout, struct {
			Asset        models.ImportedEmbedderCandidateAsset         `json:"asset"`
			Verification *models.ImportedEmbedderCandidateVerification `json:"verification,omitempty"`
		}{Asset: info, Verification: verification})
	}

	fmt.Printf("model: %s\n", info.ModelName)
	fmt.Printf("public_id: %s\n", info.PublicID)
	fmt.Printf("display_name: %s\n", info.DisplayName)
	fmt.Printf("legacy_model_name: %s\n", info.LegacyModelName)
	fmt.Printf("source_model: %s\n", info.SourceModel)
	fmt.Printf("candidate_id: %s\n", info.CandidateID)
	fmt.Printf("status: %s\n", info.Status)
	fmt.Printf("package: %s\n", info.PackagePath)
	fmt.Printf("package_sha256: %s\n", info.PackageSHA256)
	fmt.Printf("package_identity: %s\n", info.PackageIdentity)
	fmt.Printf("source_snapshot_commit: %s\n", info.SourceSnapshotCommit)
	fmt.Printf("upstream_model_url: %s\n", info.UpstreamModelURL)
	fmt.Printf("license: %s attribution=%q notice_required=%t provenance_notice_required=%t\n", info.LicenseID, info.Attribution, info.LicenseNoticeRequired, info.ProvenanceNoticeRequired)
	fmt.Printf("role_contract: schema=%s query_role=%s query_prefix=%q document_role=%s document_prefix=%q pooling=%s normalization=%s max_length=%d native_dim=%d\n", info.RoleContractSchema, info.QueryRole, info.QueryPrefix, info.DocumentRole, info.DocumentPrefix, info.Pooling, info.Normalization, info.MaxLength, info.NativeDim)
	fmt.Printf("load_path: %s\n", info.LoadPath)
	fmt.Printf("quality_claim: %t\n", info.QualityClaim)
	fmt.Printf("default_alias_changed: %t\n", info.DefaultAliasChanged)
	if verification != nil {
		status := "FAIL"
		if verification.File.OK {
			status = "OK"
		}
		fmt.Printf("sha256 package: %s %s\n", status, verification.File.SHA256)
		identityStatus := "FAIL"
		if verification.Identity.OK {
			identityStatus = "OK"
		}
		fmt.Printf("identity package: %s %s\n", identityStatus, verification.Identity.Identity)
	}
	return nil
}

func runExportRetrievalVectors(args []string) error {
	fs := flag.NewFlagSet("export-retrieval-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path; when present, export keeps qrels-relevant docs/queries under caps")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit queries for smoke exports")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	documentChunkWords := fs.Int("document-chunk-words", 0, "when positive, export parent-child document word chunks")
	documentChunkOverlap := fs.Int("document-chunk-overlap", 0, "word overlap between adjacent document chunks")
	documentChunkMinWords := fs.Int("document-chunk-min-words", 1, "minimum words for a trailing document chunk")
	documentPrefix := fs.String("document-prefix", "", "prefix prepended to document/chunk text before embedding")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before embedding")
	roleMode := fs.String("role-mode", eosruntime.EmbeddingRoleModeAuto, "embedding role mode: auto, raw, or query-document")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos export-retrieval-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	outputDir := fs.Arg(2)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	summary, err := eosruntime.ExportEmbeddingRetrievalVectors(context.Background(), model, eosruntime.RetrievalVectorExportConfig{
		DatasetName:           *datasetName,
		ArtifactPath:          artifactPath,
		CorpusPath:            corpusPath,
		QueriesPath:           queriesPath,
		QrelsPath:             *qrelsPath,
		OutputDir:             outputDir,
		BatchSize:             *batchSize,
		MaxDocs:               *maxDocs,
		MaxQueries:            *maxQueries,
		OutputDim:             *outputDim,
		DocumentChunkWords:    *documentChunkWords,
		DocumentChunkOverlap:  *documentChunkOverlap,
		DocumentChunkMinWords: *documentChunkMinWords,
		DocumentPrefix:        *documentPrefix,
		QueryPrefix:           *queryPrefix,
		RoleMode:              *roleMode,
		ManifestJSONPath:      *manifestPath,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported retrieval vectors: dataset=%s backend=%s docs=%d queries=%d", summary.Dataset, summary.Backend, summary.Documents, summary.Queries)
	if summary.ChildVectors > 0 {
		fmt.Printf(" child_vectors=%d", summary.ChildVectors)
	}
	fmt.Printf(" dim=%d\n", summary.Dimension)
	if summary.ModelDimension != 0 && summary.ModelDimension != summary.Dimension {
		fmt.Printf("model_dim: %d\n", summary.ModelDimension)
	}
	if summary.DocVectorPath != "" {
		fmt.Printf("doc_vectors: %s\n", summary.DocVectorPath)
	}
	if summary.ChildDocVectorPath != "" {
		fmt.Printf("child_doc_vectors: %s\n", summary.ChildDocVectorPath)
	}
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func runExportSparseTokenPoolVectors(args []string) error {
	fs := flag.NewFlagSet("export-sparse-token-pool-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path; when present, export keeps qrels-relevant docs/queries under caps")
	batchSize := fs.Int("batch-size", 1, "reserved batch size metadata for sparse-token prototype exports")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit queries for smoke exports")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	documentChunkWords := fs.Int("document-chunk-words", 0, "when positive, export parent-child document word chunks")
	documentChunkOverlap := fs.Int("document-chunk-overlap", 0, "word overlap between adjacent document chunks")
	documentChunkMinWords := fs.Int("document-chunk-min-words", 1, "minimum words for a trailing document chunk")
	tokenSpanTokens := fs.Int("token-span-tokens", 0, "when positive, export child vectors by pooling one encoded document over token spans of this size")
	tokenSpanOverlap := fs.Int("token-span-overlap", 0, "token overlap between adjacent token-span child vectors")
	tokenSpanMinTokens := fs.Int("token-span-min-tokens", 0, "minimum tokens for a trailing token span; default 1 when token-span mode is enabled")
	documentPrefix := fs.String("document-prefix", "", "prefix prepended to document/chunk text before embedding")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before embedding")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest")
	topK := fs.Int("top-k", 0, "sparse selected keys per query; 0 uses sparse attention reference default")
	routeBlockSize := fs.Int("route-block-size", 0, "route block size; 0 disables routed block-anchor preselection")
	routeTopBlocks := fs.Int("route-top-blocks", 0, "route blocks selected per query; 0 disables routed block-anchor preselection")
	bits := fs.Int("bits", 4, "TurboQuant K/V bits: 2, 4, or 8")
	keyBits := fs.Int("key-bits", 0, "TurboQuant key bits override: 0 inherits --bits; supported: 2, 4, or 8")
	valueBits := fs.Int("value-bits", 0, "TurboQuant value bits override: 0 inherits --bits; supported: 2, 4, or 8")
	seed := fs.Int64("seed", 0x4d697261, "TurboQuant Hadamard seed")
	maxTokens := fs.Int("max-tokens", 0, "truncate tokenized text to at most this many tokens after tokenizer limits; 0 keeps tokenizer output")
	tokenizerMaxSeq := fs.Int("tokenizer-max-seq", 0, "export-time tokenizer max_sequence override for diagnostic sparse retrieval-cache exports; 0 keeps artifact tokenizer contract")
	minObservedDocTokens := fs.Int("min-observed-doc-tokens", 0, "fail when max observed document tokenizer-output tokens consumed by the sparse-token-pool encoder is below this threshold; 0 disables")
	attentionMode := fs.String("attention-mode", eosruntime.SparseTokenPoolAttentionModeTurboQuantSparse, "attention implementation: turboquant_sparse or dense")
	resume := fs.Bool("resume", false, "resume JSONL vector export from per-file progress sidecars, truncating partial output to the last completed record before appending")
	progressEvery := fs.Int("progress-every", 0, "emit sparse token-pool export progress to stderr every N completed records; 0 disables")
	weightPath := fs.String("weights", "", "explicit sibling weight file path; default is <artifact>.weights.mll")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos export-sparse-token-pool-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	outputDir := fs.Arg(2)
	resolvedWeightPath, err := resolveSparseExportWeightPath(artifactPath, outputDir, *weightPath)
	if err != nil {
		return err
	}
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	summary, err := eosruntime.ExportSparseTokenPoolRetrievalVectors(context.Background(), model, eosruntime.SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:                  *datasetName,
		ArtifactPath:                 artifactPath,
		WeightFilePath:               resolvedWeightPath,
		CorpusPath:                   corpusPath,
		QueriesPath:                  queriesPath,
		QrelsPath:                    *qrelsPath,
		OutputDir:                    outputDir,
		BatchSize:                    *batchSize,
		MaxDocs:                      *maxDocs,
		MaxQueries:                   *maxQueries,
		OutputDim:                    *outputDim,
		DocumentChunkWords:           *documentChunkWords,
		DocumentChunkOverlap:         *documentChunkOverlap,
		DocumentChunkMinWords:        *documentChunkMinWords,
		TokenSpanTokens:              *tokenSpanTokens,
		TokenSpanOverlap:             *tokenSpanOverlap,
		TokenSpanMinTokens:           *tokenSpanMinTokens,
		DocumentPrefix:               *documentPrefix,
		QueryPrefix:                  *queryPrefix,
		ManifestJSONPath:             *manifestPath,
		TopK:                         *topK,
		RouteBlockSize:               *routeBlockSize,
		RouteTopBlocks:               *routeTopBlocks,
		Bits:                         *bits,
		KeyBits:                      *keyBits,
		ValueBits:                    *valueBits,
		Seed:                         *seed,
		MaxTokens:                    *maxTokens,
		TokenizerMaxSequenceOverride: *tokenizerMaxSeq,
		MinObservedDocTokens:         *minObservedDocTokens,
		AttentionMode:                *attentionMode,
		Resume:                       *resume,
		ProgressEvery:                *progressEvery,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported experimental sparse-token pool vectors: dataset=%s docs=%d queries=%d", summary.Dataset, summary.Documents, summary.Queries)
	if summary.ChildVectors > 0 {
		fmt.Printf(" child_vectors=%d", summary.ChildVectors)
	}
	fmt.Printf(" dim=%d quality_claim=%t\n", summary.Dimension, summary.QualityClaim)
	fmt.Printf("method: %s\n", summary.Method)
	fmt.Printf("attention: mode=%s turboquant_kv_applied=%t kv_decode=%s top_k=%d route_block_size=%d route_top_blocks=%d bits=%d key_bits=%d value_bits=%d seed=%d\n", summary.AttentionMode, summary.TurboQuantKVApplied, summary.KVDecode, summary.TopK, summary.RouteBlockSize, summary.RouteTopBlocks, summary.Bits, summary.KeyBits, summary.ValueBits, summary.QuantizerSeed)
	if summary.TokenSpanTokens > 0 {
		fmt.Printf("token_span: tokens=%d overlap=%d min_tokens=%d\n", summary.TokenSpanTokens, summary.TokenSpanOverlap, summary.TokenSpanMinTokens)
	}
	if summary.ResumeEnabled || summary.ProgressEvery > 0 {
		fmt.Printf("resume: enabled=%t progress_every=%d resumed_document_records=%d resumed_child_vectors=%d resumed_query_records=%d\n", summary.ResumeEnabled, summary.ProgressEvery, summary.ResumedDocumentRecords, summary.ResumedChildVectors, summary.ResumedQueryRecords)
	}
	if summary.TokenizerMaxSequenceOverride > 0 {
		fmt.Printf("tokenizer_max_sequence: original=%d effective=%d override=%d\n", summary.TokenizerMaxSequenceOriginal, summary.TokenizerMaxSequenceEffective, summary.TokenizerMaxSequenceOverride)
	}
	fmt.Printf("tokenizer_output: doc_records=%d doc_max_tokens=%d doc_mean_tokens=%.2f doc_total_tokens=%d doc_truncated_by_max_tokens=%d query_records=%d query_max_tokens=%d query_mean_tokens=%.2f query_total_tokens=%d query_truncated_by_max_tokens=%d\n", summary.DocumentTokenizerOutput.RecordCount, summary.DocumentTokenizerOutput.MaxObservedTokens, summary.DocumentTokenizerOutput.MeanObservedTokens, summary.DocumentTokenizerOutput.TotalTokens, summary.DocumentTokenizerOutput.TruncatedByMaxTokensCount, summary.QueryTokenizerOutput.RecordCount, summary.QueryTokenizerOutput.MaxObservedTokens, summary.QueryTokenizerOutput.MeanObservedTokens, summary.QueryTokenizerOutput.TotalTokens, summary.QueryTokenizerOutput.TruncatedByMaxTokensCount)
	fmt.Printf("weights: attention=%t attention_output=%t hidden_projection=%t projection=%t dense_kv_materialized=%t\n", summary.AttentionWeightsApplied, summary.AttentionOutputApplied, summary.HiddenProjectionApplied, summary.ProjectionApplied, summary.DenseKVMaterialized)
	if summary.DocVectorPath != "" {
		fmt.Printf("doc_vectors: %s\n", summary.DocVectorPath)
	}
	if summary.ChildDocVectorPath != "" {
		fmt.Printf("child_doc_vectors: %s\n", summary.ChildDocVectorPath)
	}
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func runExportSparseEncoderVectors(args []string) error {
	fs := flag.NewFlagSet("export-sparse-encoder-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path; when present, export keeps qrels-relevant docs/queries under caps")
	batchSize := fs.Int("batch-size", 1, "reserved batch size metadata for sparse encoder prototype exports")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit queries for smoke exports")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	documentPrefix := fs.String("document-prefix", "", "prefix prepended to document text before embedding")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before embedding")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest")
	topK := fs.Int("top-k", 0, "sparse selected keys per query; 0 uses sparse attention reference default")
	routeBlockSize := fs.Int("route-block-size", 0, "route block size; 0 disables routed block-anchor preselection")
	routeTopBlocks := fs.Int("route-top-blocks", 0, "route blocks selected per query; 0 disables routed block-anchor preselection")
	bits := fs.Int("bits", 4, "TurboQuant K/V bits: 2, 4, or 8")
	keyBits := fs.Int("key-bits", 0, "TurboQuant key bits override: 0 inherits --bits; supported: 2, 4, or 8")
	valueBits := fs.Int("value-bits", 0, "TurboQuant value bits override: 0 inherits --bits; supported: 2, 4, or 8")
	seed := fs.Int64("seed", 0x4d697261, "TurboQuant Hadamard seed")
	maxTokens := fs.Int("max-tokens", 0, "truncate tokenized text to at most this many tokens after tokenizer limits; 0 keeps tokenizer output")
	tokenizerMaxSeq := fs.Int("tokenizer-max-seq", 0, "export-time tokenizer max_sequence override for diagnostic sparse retrieval-cache exports; 0 keeps artifact tokenizer contract")
	minObservedDocTokens := fs.Int("min-observed-doc-tokens", 0, "fail when max observed document tokenizer-output tokens consumed by the sparse encoder is below this threshold; 0 disables")
	attentionMode := fs.String("attention-mode", eosruntime.SparseTokenPoolAttentionModeTurboQuantSparse, "attention implementation: turboquant_sparse or dense")
	weightPath := fs.String("weights", "", "explicit sibling weight file path; default is <artifact>.weights.mll")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos export-sparse-encoder-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	outputDir := fs.Arg(2)
	resolvedWeightPath, err := resolveSparseExportWeightPath(artifactPath, outputDir, *weightPath)
	if err != nil {
		return err
	}
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	summary, err := eosruntime.ExportSparseTokenPoolRetrievalVectors(context.Background(), model, eosruntime.SparseTokenPoolRetrievalVectorExportConfig{
		DatasetName:                  *datasetName,
		ArtifactPath:                 artifactPath,
		WeightFilePath:               resolvedWeightPath,
		CorpusPath:                   corpusPath,
		QueriesPath:                  queriesPath,
		QrelsPath:                    *qrelsPath,
		OutputDir:                    outputDir,
		BatchSize:                    *batchSize,
		MaxDocs:                      *maxDocs,
		MaxQueries:                   *maxQueries,
		OutputDim:                    *outputDim,
		DocumentPrefix:               *documentPrefix,
		QueryPrefix:                  *queryPrefix,
		ManifestJSONPath:             *manifestPath,
		TopK:                         *topK,
		RouteBlockSize:               *routeBlockSize,
		RouteTopBlocks:               *routeTopBlocks,
		Bits:                         *bits,
		KeyBits:                      *keyBits,
		ValueBits:                    *valueBits,
		Seed:                         *seed,
		MaxTokens:                    *maxTokens,
		TokenizerMaxSequenceOverride: *tokenizerMaxSeq,
		MinObservedDocTokens:         *minObservedDocTokens,
		AttentionMode:                *attentionMode,
		RequireFullEncoder:           true,
		Method:                       "experimental_sparse_encoder_host_reference",
		EvidenceLevel:                "retrieval_cache_host_reference_sparse_encoder",
		ClaimBoundary:                "Prototype host-reference sparse encoder retrieval-cache evidence only; not a trained sparse/LongEmbed encoder, not sealed runtime inference, and not production quality evidence.",
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported experimental sparse encoder vectors: dataset=%s docs=%d queries=%d dim=%d quality_claim=%t\n", summary.Dataset, summary.Documents, summary.Queries, summary.Dimension, summary.QualityClaim)
	fmt.Printf("method: %s\n", summary.Method)
	fmt.Printf("evidence_level: %s\n", summary.EvidenceLevel)
	fmt.Printf("full_encoder: require=%t applied=%t\n", summary.RequireFullEncoder, summary.FullEncoderApplied)
	fmt.Printf("attention: mode=%s turboquant_kv_applied=%t dense_kv_materialized=%t kv_decode=%s top_k=%d route_block_size=%d route_top_blocks=%d bits=%d key_bits=%d value_bits=%d seed=%d\n", summary.AttentionMode, summary.TurboQuantKVApplied, summary.DenseKVMaterialized, summary.KVDecode, summary.TopK, summary.RouteBlockSize, summary.RouteTopBlocks, summary.Bits, summary.KeyBits, summary.ValueBits, summary.QuantizerSeed)
	if summary.MaxObservedDocPlan != nil {
		fmt.Printf("sparse_plan_max_doc: key_len=%d candidate_key_budget=%d score_fraction=%.6f subquadratic=%t\n", summary.MaxObservedDocPlan.KeyLen, summary.MaxObservedDocPlan.CandidateKeyBudget, summary.MaxObservedDocPlan.ScoreCountFraction, summary.MaxObservedDocPlan.SubquadraticScorePlan)
	}
	if summary.TokenizerMaxSequenceOverride > 0 {
		fmt.Printf("tokenizer_max_sequence: original=%d effective=%d override=%d\n", summary.TokenizerMaxSequenceOriginal, summary.TokenizerMaxSequenceEffective, summary.TokenizerMaxSequenceOverride)
	}
	fmt.Printf("tokenizer_output: doc_records=%d doc_max_tokens=%d doc_mean_tokens=%.2f doc_total_tokens=%d doc_truncated_by_max_tokens=%d query_records=%d query_max_tokens=%d query_mean_tokens=%.2f query_total_tokens=%d query_truncated_by_max_tokens=%d\n", summary.DocumentTokenizerOutput.RecordCount, summary.DocumentTokenizerOutput.MaxObservedTokens, summary.DocumentTokenizerOutput.MeanObservedTokens, summary.DocumentTokenizerOutput.TotalTokens, summary.DocumentTokenizerOutput.TruncatedByMaxTokensCount, summary.QueryTokenizerOutput.RecordCount, summary.QueryTokenizerOutput.MaxObservedTokens, summary.QueryTokenizerOutput.MeanObservedTokens, summary.QueryTokenizerOutput.TotalTokens, summary.QueryTokenizerOutput.TruncatedByMaxTokensCount)
	fmt.Printf("doc_vectors: %s\n", summary.DocVectorPath)
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func resolveSparseExportWeightPath(artifactPath, outputDir, explicitPath string) (string, error) {
	if explicitPath != "" {
		return explicitPath, nil
	}
	defaultPath := eosruntime.DefaultWeightFilePath(artifactPath)
	if _, err := os.Stat(defaultPath); err == nil {
		return defaultPath, nil
	} else if err != nil && !os.IsNotExist(err) {
		return "", err
	}
	pkg, err := eosruntime.ReadSealedEmbeddingPackage(artifactPath)
	if err != nil {
		return defaultPath, nil
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", err
	}
	sealedWeightPath := filepath.Join(outputDir, "sealed-weights.mll")
	if err := pkg.Weights.WriteFile(sealedWeightPath); err != nil {
		return "", err
	}
	return sealedWeightPath, nil
}

func runExportTimeSeriesVectors(args []string) error {
	fs := flag.NewFlagSet("export-timeseries-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxSeries := fs.Int("max-series", 0, "limit series rows for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit query rows for smoke exports")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	windowSize := fs.Int("window-size", 0, "time-series window size in points")
	windowStride := fs.Int("window-stride", 0, "time-series window stride in points; 0 uses --window-size")
	seriesPrefix := fs.String("series-prefix", "", "prefix prepended to rendered series-window text before embedding")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before embedding")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 4 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" || fs.Arg(3) == "" {
		return fmt.Errorf("usage: eos export-timeseries-vectors [flags] <artifact.mll> <series.jsonl> <queries.jsonl> <output-dir>")
	}
	artifactPath := fs.Arg(0)
	seriesPath := fs.Arg(1)
	queriesPath := fs.Arg(2)
	outputDir := fs.Arg(3)
	if *datasetName == "" {
		*datasetName = strings.TrimSuffix(filepath.Base(seriesPath), filepath.Ext(seriesPath))
		if *datasetName == "" {
			*datasetName = "timeseries"
		}
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	summary, err := eosruntime.ExportTimeSeriesWindowVectors(context.Background(), model, eosruntime.TimeSeriesVectorExportConfig{
		DatasetName:      *datasetName,
		ArtifactPath:     artifactPath,
		SeriesPath:       seriesPath,
		QueriesPath:      queriesPath,
		OutputDir:        outputDir,
		BatchSize:        *batchSize,
		MaxSeries:        *maxSeries,
		MaxQueries:       *maxQueries,
		OutputDim:        *outputDim,
		WindowSize:       *windowSize,
		WindowStride:     *windowStride,
		SeriesPrefix:     *seriesPrefix,
		QueryPrefix:      *queryPrefix,
		ManifestJSONPath: *manifestPath,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported time-series vectors: dataset=%s backend=%s series=%d queries=%d child_window_vectors=%d dim=%d\n", summary.Dataset, summary.Backend, summary.Series, summary.Queries, summary.ChildVectors, summary.Dimension)
	if summary.ModelDimension != 0 && summary.ModelDimension != summary.Dimension {
		fmt.Printf("model_dim: %d\n", summary.ModelDimension)
	}
	fmt.Printf("windows: size=%d stride=%d\n", summary.WindowSize, summary.WindowStride)
	fmt.Printf("dataset_dir: %s\n", summary.OutputDir)
	fmt.Printf("corpus: %s\n", summary.CorpusPath)
	fmt.Printf("queries: %s\n", summary.BEIRQueriesPath)
	fmt.Printf("child_doc_vectors: %s\n", summary.ChildDocVectorPath)
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func runExportEventTraceVectors(args []string) error {
	fs := flag.NewFlagSet("export-event-trace-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxTraces := fs.Int("max-traces", 0, "limit trace rows for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit query rows for smoke exports")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	tracePrefix := fs.String("trace-prefix", "", "prefix prepended to rendered trace-event text before embedding")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before embedding")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 4 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" || fs.Arg(3) == "" {
		return fmt.Errorf("usage: eos export-event-trace-vectors [flags] <artifact.mll> <traces.jsonl> <queries.jsonl> <output-dir>")
	}
	artifactPath := fs.Arg(0)
	tracesPath := fs.Arg(1)
	queriesPath := fs.Arg(2)
	outputDir := fs.Arg(3)
	if *datasetName == "" {
		*datasetName = strings.TrimSuffix(filepath.Base(tracesPath), filepath.Ext(tracesPath))
		if *datasetName == "" {
			*datasetName = "event-traces"
		}
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	summary, err := eosruntime.ExportEventTraceVectors(context.Background(), model, eosruntime.EventTraceVectorExportConfig{
		DatasetName:      *datasetName,
		ArtifactPath:     artifactPath,
		TracesPath:       tracesPath,
		QueriesPath:      queriesPath,
		OutputDir:        outputDir,
		BatchSize:        *batchSize,
		MaxTraces:        *maxTraces,
		MaxQueries:       *maxQueries,
		OutputDim:        *outputDim,
		TracePrefix:      *tracePrefix,
		QueryPrefix:      *queryPrefix,
		ManifestJSONPath: *manifestPath,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported event trace vectors: dataset=%s backend=%s traces=%d queries=%d child_event_vectors=%d dim=%d\n", summary.Dataset, summary.Backend, summary.Traces, summary.Queries, summary.ChildVectors, summary.Dimension)
	if summary.ModelDimension != 0 && summary.ModelDimension != summary.Dimension {
		fmt.Printf("model_dim: %d\n", summary.ModelDimension)
	}
	fmt.Printf("dataset_dir: %s\n", summary.OutputDir)
	fmt.Printf("corpus: %s\n", summary.CorpusPath)
	fmt.Printf("queries: %s\n", summary.BEIRQueriesPath)
	fmt.Printf("child_doc_vectors: %s\n", summary.ChildDocVectorPath)
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func runEvalRetrieval(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval", flag.ContinueOnError)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	roleMode := fs.String("role-mode", eosruntime.EmbeddingRoleModeAuto, "embedding role mode: auto, raw, or query-document")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	metrics, err := eosruntime.EvaluateEmbeddingRetrieval(context.Background(), model, eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      artifactPath,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		BatchSize:         *batchSize,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		RoleMode:          *roleMode,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval eval: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalRetrievalVectors(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-vectors", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	backendName := fs.String("backend", "vectors", "backend/provider label for metrics output")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" || *docVectorsPath == "" || *queryVectorsPath == "" {
		return fmt.Errorf("usage: eos eval-retrieval-vectors [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateVectorCacheRetrieval(context.Background(), eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		BackendName:       *backendName,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval vectors: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalRetrievalHybrid(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-hybrid", flag.ContinueOnError)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	method := fs.String("method", "minmax", "hybrid fusion method: minmax, minmax_blend, zscore, zscore_blend, or rrf")
	alpha := fs.Float64("alpha", 0.75, "BM25 weight for minmax/zscore hybrid blending")
	rrfK := fs.Float64("rrf-k", 60, "RRF rank constant")
	rrfLambda := fs.Float64("rrf-lambda", 1.0, "BM25 contribution multiplier for RRF")
	denseProtectTopK := fs.Int("dense-protect-top-k", 0, "preserve the dense top-N prefix before appending fused hybrid tail candidates")
	denseCandidatesOnly := fs.Bool("dense-candidates-only", false, "experimental rerank mode: only dense top-k candidates may appear in hybrid results")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos eval-retrieval-hybrid [flags] <artifact.mll> <beir-dataset-dir>")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	metrics, err := eosruntime.EvaluateHybridRetrieval(context.Background(), model, eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      artifactPath,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		BatchSize:         *batchSize,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		Hybrid: eosruntime.RetrievalEvalHybridConfig{
			Method:              *method,
			Alpha:               *alpha,
			AlphaSet:            true,
			RRFK:                *rrfK,
			RRFLambda:           *rrfLambda,
			DenseProtectTopK:    *denseProtectTopK,
			DenseCandidatesOnly: *denseCandidatesOnly,
		},
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval hybrid: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	printRetrievalHybridConfig(metrics)
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalRetrievalVectorsHybrid(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-vectors-hybrid", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	backendName := fs.String("backend", "vectors-hybrid", "backend/provider label for metrics output")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	method := fs.String("method", "minmax", "hybrid fusion method: minmax, minmax_blend, zscore, zscore_blend, or rrf")
	alpha := fs.Float64("alpha", 0.75, "BM25 weight for minmax/zscore hybrid blending")
	rrfK := fs.Float64("rrf-k", 60, "RRF rank constant")
	rrfLambda := fs.Float64("rrf-lambda", 1.0, "BM25 contribution multiplier for RRF")
	denseProtectTopK := fs.Int("dense-protect-top-k", 0, "preserve the dense top-N prefix before appending fused hybrid tail candidates")
	denseCandidatesOnly := fs.Bool("dense-candidates-only", false, "experimental rerank mode: only dense top-k candidates may appear in hybrid results")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" || *docVectorsPath == "" || *queryVectorsPath == "" {
		return fmt.Errorf("usage: eos eval-retrieval-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateVectorCacheHybridRetrieval(context.Background(), eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		BackendName:       *backendName,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		Hybrid: eosruntime.RetrievalEvalHybridConfig{
			Method:              *method,
			Alpha:               *alpha,
			AlphaSet:            true,
			RRFK:                *rrfK,
			RRFLambda:           *rrfLambda,
			DenseProtectTopK:    *denseProtectTopK,
			DenseCandidatesOnly: *denseCandidatesOnly,
		},
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval vectors hybrid: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	printRetrievalHybridConfig(metrics)
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalSparseLexicalHeadVectorsHybrid(args []string) error {
	fs := flag.NewFlagSet("eval-sparse-lexical-head-vectors-hybrid", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL document/query universe from export-sparse-lexical-labels")
	headPath := fs.String("head-json", "", "experimental sparse lexical hash head JSON")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	method := fs.String("method", "minmax", "hybrid fusion method: minmax, minmax_blend, zscore, zscore_blend, or rrf")
	alpha := fs.Float64("alpha", 0.75, "sparse hash-head weight for minmax/zscore hybrid blending")
	rrfK := fs.Float64("rrf-k", 60, "RRF rank constant")
	rrfLambda := fs.Float64("rrf-lambda", 1.0, "sparse hash-head contribution multiplier for RRF")
	denseProtectTopK := fs.Int("dense-protect-top-k", 0, "preserve the dense top-N prefix before appending fused hybrid tail candidates")
	denseCandidatesOnly := fs.Bool("dense-candidates-only", false, "experimental rerank mode: only dense top-k candidates may appear in hybrid results")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-sparse-lexical-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --labels labels.jsonl --head-json head.json <beir-dataset-dir>")
	}
	if *docVectorsPath == "" {
		return fmt.Errorf("doc-vectors is required")
	}
	if *queryVectorsPath == "" {
		return fmt.Errorf("query-vectors is required")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateSparseLexicalHashHeadVectorHybrid(context.Background(), eosruntime.SparseLexicalHashHeadEvalConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		LabelsPath:        *labelsPath,
		HeadPath:          *headPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		Hybrid: eosruntime.RetrievalEvalHybridConfig{
			Method:              *method,
			Alpha:               *alpha,
			AlphaSet:            true,
			RRFK:                *rrfK,
			RRFLambda:           *rrfLambda,
			DenseProtectTopK:    *denseProtectTopK,
			DenseCandidatesOnly: *denseCandidatesOnly,
		},
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval sparse lexical hash head vectors hybrid: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	printRetrievalHybridConfig(metrics)
	if metrics.SparseLexical != nil {
		fmt.Printf("sparse_hash_head: hash_bins=%d documents=%d queries=%d doc_avg_hash_nonzeros=%.2f doc_max_hash_nonzeros=%d query_avg_hash_nonzeros=%.2f query_max_hash_nonzeros=%d doc_merged_bins=%d query_merged_bins=%d representation=%s\n",
			metrics.SparseLexical.HashBins, metrics.SparseLexical.DocumentLabels, metrics.SparseLexical.QueryLabels, metrics.SparseLexical.DocumentAvgHashNNZ, metrics.SparseLexical.DocumentMaxHashNNZ, metrics.SparseLexical.QueryAvgHashNNZ, metrics.SparseLexical.QueryMaxHashNNZ, metrics.SparseLexical.DocumentMergedBins, metrics.SparseLexical.QueryMergedBins, metrics.SparseLexical.Representation)
	}
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	fmt.Printf("labels: %s\n", *labelsPath)
	fmt.Printf("head: %s\n", *headPath)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runFitSparseLexicalProjectionHead(args []string) error {
	fs := flag.NewFlagSet("fit-sparse-lexical-projection-head", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for head metadata")
	split := fs.String("split", "train", "label split for validation")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL used as fit-time supervision")
	docVectorsPath := fs.String("doc-vectors", "", "train document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "train query vector cache JSONL")
	hashBins := fs.Int("hash-bins", 65536, "deterministic FNV-1a hashed sparse bins")
	maxPrototypes := fs.Int("max-prototypes", 4096, "maximum learned hashed-bin prototypes to store")
	maxTerms := fs.Int("max-terms", 32, "maximum positive predicted sparse bins per vector")
	prototypeRank := fs.String("prototype-rank", "support", "prototype retention rank policy: support, total_weight, or avg_weight")
	headPath := fs.String("head-json", "", "write experimental sparse lexical projection head JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos fit-sparse-lexical-projection-head --labels labels.jsonl --doc-vectors docs.jsonl --query-vectors queries.jsonl --hash-bins 65536 --max-prototypes 4096 --max-terms 32 --head-json head.json")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	if *docVectorsPath == "" {
		return fmt.Errorf("doc-vectors is required")
	}
	if *queryVectorsPath == "" {
		return fmt.Errorf("query-vectors is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	head, err := eosruntime.FitSparseLexicalProjectionHead(eosruntime.SparseLexicalProjectionHeadFitConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		LabelsPath:        *labelsPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		HeadPath:          *headPath,
		HashBins:          *hashBins,
		MaxPrototypes:     *maxPrototypes,
		MaxPredictedTerms: *maxTerms,
		PrototypeRank:     *prototypeRank,
	})
	if err != nil {
		return err
	}
	fmt.Printf("fit sparse lexical projection head: schema=%s experimental=%t dataset=%s split=%s dim=%d hash_bins=%d prototypes=%d max_terms=%d prototype_rank=%s\n",
		head.Schema, head.Experimental, head.Dataset, head.Split, head.Config.Dimension, head.Hashing.Bins, len(head.Prototypes), head.Config.MaxPredictedTerms, head.Config.PrototypeRank)
	fmt.Printf("fit_vectors: documents=%d queries=%d missing_documents=%d missing_queries=%d candidate_prototypes=%d stored_prototypes=%d normalization=%s\n",
		head.Stats.DocumentVectors, head.Stats.QueryVectors, head.Stats.MissingDocVectors, head.Stats.MissingQueryVectors, head.Stats.CandidatePrototypes, head.Stats.StoredPrototypes, head.Config.Normalization)
	fmt.Printf("labels: %s\n", *labelsPath)
	fmt.Printf("doc_vectors: %s\n", *docVectorsPath)
	fmt.Printf("query_vectors: %s\n", *queryVectorsPath)
	fmt.Printf("head: %s\n", *headPath)
	return nil
}

func runEvalSparseLexicalProjectionHeadVectorsHybrid(args []string) error {
	fs := flag.NewFlagSet("eval-sparse-lexical-projection-head-vectors-hybrid", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	headPath := fs.String("head-json", "", "experimental sparse lexical projection head JSON")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	method := fs.String("method", "minmax", "fusion method: sparse_only, minmax, minmax_blend, zscore, zscore_blend, or rrf")
	alpha := fs.Float64("alpha", 0.75, "projected sparse weight for minmax/zscore hybrid blending")
	rrfK := fs.Float64("rrf-k", 60, "RRF rank constant")
	rrfLambda := fs.Float64("rrf-lambda", 1.0, "projected sparse contribution multiplier for RRF")
	denseProtectTopK := fs.Int("dense-protect-top-k", 0, "preserve the dense top-N prefix before appending fused hybrid tail candidates")
	denseCandidatesOnly := fs.Bool("dense-candidates-only", false, "experimental rerank mode: only dense top-k candidates may appear in hybrid results")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-sparse-lexical-projection-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json <beir-dataset-dir>")
	}
	if *docVectorsPath == "" {
		return fmt.Errorf("doc-vectors is required")
	}
	if *queryVectorsPath == "" {
		return fmt.Errorf("query-vectors is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateSparseLexicalProjectionHeadVectorHybrid(context.Background(), eosruntime.SparseLexicalProjectionHeadEvalConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		HeadPath:          *headPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		Hybrid: eosruntime.RetrievalEvalHybridConfig{
			Method:              *method,
			Alpha:               *alpha,
			AlphaSet:            true,
			RRFK:                *rrfK,
			RRFLambda:           *rrfLambda,
			DenseProtectTopK:    *denseProtectTopK,
			DenseCandidatesOnly: *denseCandidatesOnly,
		},
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval sparse lexical projection head vectors hybrid: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	printRetrievalHybridConfig(metrics)
	if metrics.SparseLexical != nil {
		fmt.Printf("sparse_projection_head: hash_bins=%d predicted_doc_terms_max=%d predicted_query_terms_max=%d representation=%s\n",
			metrics.SparseLexical.HashBins, metrics.SparseLexical.DocumentMaxHashNNZ, metrics.SparseLexical.QueryMaxHashNNZ, metrics.SparseLexical.Representation)
	}
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	fmt.Printf("head: %s\n", *headPath)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runFitSparseLexicalLinearHead(args []string) error {
	fs := flag.NewFlagSet("fit-sparse-lexical-linear-head", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for head metadata")
	split := fs.String("split", "train", "label split for validation")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL used as fit-time supervision")
	docVectorsPath := fs.String("doc-vectors", "", "train document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "train query vector cache JSONL")
	hashBins := fs.Int("hash-bins", 65536, "deterministic FNV-1a hashed sparse bins")
	maxBins := fs.Int("max-bins", 4096, "maximum retained hashed output bins to train and store")
	maxTerms := fs.Int("max-terms", 32, "maximum positive predicted sparse bins per vector")
	binRank := fs.String("bin-rank", "support", "retained-bin rank policy: support, total_weight, or avg_weight")
	epochs := fs.Int("epochs", 3, "deterministic SGD epochs")
	learningRate := fs.Float64("learning-rate", 0.05, "deterministic SGD learning rate")
	negativeRatio := fs.Int("negative-ratio", 2, "deterministic zero-target bins per training example")
	targetTransform := fs.String("target-transform", "identity", "fit-time target transform: identity or log1p")
	headPath := fs.String("head-json", "", "write experimental sparse lexical linear head JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos fit-sparse-lexical-linear-head --labels labels.jsonl --doc-vectors docs.jsonl --query-vectors queries.jsonl --hash-bins 65536 --max-bins 4096 --max-terms 32 --head-json head.json")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	if *docVectorsPath == "" {
		return fmt.Errorf("doc-vectors is required")
	}
	if *queryVectorsPath == "" {
		return fmt.Errorf("query-vectors is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	head, err := eosruntime.FitSparseLexicalLinearHead(eosruntime.SparseLexicalLinearHeadFitConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		LabelsPath:        *labelsPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		HeadPath:          *headPath,
		HashBins:          *hashBins,
		MaxBins:           *maxBins,
		MaxPredictedTerms: *maxTerms,
		BinRank:           *binRank,
		Epochs:            *epochs,
		LearningRate:      *learningRate,
		NegativeRatio:     *negativeRatio,
		TargetTransform:   *targetTransform,
	})
	if err != nil {
		return err
	}
	fmt.Printf("fit sparse lexical linear head: schema=%s experimental=%t dataset=%s split=%s dim=%d hash_bins=%d bins=%d max_terms=%d bin_rank=%s epochs=%d learning_rate=%g negative_ratio=%d target_transform=%s\n",
		head.Schema, head.Experimental, head.Dataset, head.Split, head.Config.Dimension, head.Hashing.Bins, len(head.Bins), head.Config.MaxPredictedTerms, head.Config.BinRank, head.Config.Epochs, head.Config.LearningRate, head.Config.NegativeRatio, head.Config.TargetTransform)
	fmt.Printf("fit_vectors: documents=%d queries=%d missing_documents=%d missing_queries=%d candidate_bins=%d stored_bins=%d updates=%d final_mse=%.6f normalization=%s\n",
		head.Stats.DocumentVectors, head.Stats.QueryVectors, head.Stats.MissingDocVectors, head.Stats.MissingQueryVectors, head.Stats.CandidateBins, head.Stats.StoredBins, head.Stats.TrainingUpdates, head.Stats.FinalMSE, head.Config.Normalization)
	fmt.Printf("labels: %s\n", *labelsPath)
	fmt.Printf("doc_vectors: %s\n", *docVectorsPath)
	fmt.Printf("query_vectors: %s\n", *queryVectorsPath)
	fmt.Printf("head: %s\n", *headPath)
	return nil
}

func runEvalSparseLexicalLinearHeadVectorsHybrid(args []string) error {
	fs := flag.NewFlagSet("eval-sparse-lexical-linear-head-vectors-hybrid", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	headPath := fs.String("head-json", "", "experimental sparse lexical linear head JSON")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	docMaxTerms := fs.Int("doc-max-terms", 0, "maximum predicted sparse bins per document vector; 0 uses head artifact max terms")
	queryMaxTerms := fs.Int("query-max-terms", 0, "maximum predicted sparse bins per query vector; 0 uses head artifact max terms")
	scoreThreshold := fs.Float64("score-threshold", 0, "minimum predicted sparse score; predictions require score > threshold")
	method := fs.String("method", "minmax", "fusion method: sparse_only, minmax, minmax_blend, zscore, zscore_blend, or rrf")
	alpha := fs.Float64("alpha", 0.75, "linear-head sparse weight for minmax/zscore hybrid blending")
	rrfK := fs.Float64("rrf-k", 60, "RRF rank constant")
	rrfLambda := fs.Float64("rrf-lambda", 1.0, "linear-head sparse contribution multiplier for RRF")
	denseProtectTopK := fs.Int("dense-protect-top-k", 0, "preserve the dense top-N prefix before appending fused hybrid tail candidates")
	denseCandidatesOnly := fs.Bool("dense-candidates-only", false, "experimental rerank mode: only dense top-k candidates may appear in hybrid results")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-sparse-lexical-linear-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json <beir-dataset-dir>")
	}
	if *docVectorsPath == "" {
		return fmt.Errorf("doc-vectors is required")
	}
	if *queryVectorsPath == "" {
		return fmt.Errorf("query-vectors is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateSparseLexicalLinearHeadVectorHybrid(context.Background(), eosruntime.SparseLexicalLinearHeadEvalConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		HeadPath:          *headPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		DocMaxTerms:       *docMaxTerms,
		QueryMaxTerms:     *queryMaxTerms,
		ScoreThreshold:    *scoreThreshold,
		PerQueryJSONLPath: *perQueryPath,
		Hybrid: eosruntime.RetrievalEvalHybridConfig{
			Method:              *method,
			Alpha:               *alpha,
			AlphaSet:            true,
			RRFK:                *rrfK,
			RRFLambda:           *rrfLambda,
			DenseProtectTopK:    *denseProtectTopK,
			DenseCandidatesOnly: *denseCandidatesOnly,
		},
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval sparse lexical linear head vectors hybrid: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	printRetrievalHybridConfig(metrics)
	if metrics.SparseLexical != nil {
		fmt.Printf("sparse_linear_head: hash_bins=%d predicted_doc_terms_max=%d predicted_query_terms_max=%d score_threshold=%g representation=%s\n",
			metrics.SparseLexical.HashBins, metrics.SparseLexical.DocumentMaxHashNNZ, metrics.SparseLexical.QueryMaxHashNNZ, metrics.SparseLexical.ScoreThreshold, metrics.SparseLexical.Representation)
	}
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	fmt.Printf("head: %s\n", *headPath)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func printRetrievalHybridConfig(metrics eosruntime.RetrievalEvalMetrics) {
	if metrics.Config.Hybrid == nil {
		return
	}
	hybrid := metrics.Config.Hybrid
	fmt.Printf("hybrid: method=%s alpha=%.6g rrf_k=%.6g rrf_lambda=%.6g dense_protect_top_k=%d dense_candidates_only=%t\n", hybrid.Method, hybrid.Alpha, hybrid.RRFK, hybrid.RRFLambda, hybrid.DenseProtectTopK, hybrid.DenseCandidatesOnly)
}

func runEvalRetrievalTurboQuant(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-turboquant", flag.ContinueOnError)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	bitsRaw := fs.String("bits", "2,4,8", "comma-separated TurboQuant IP bit widths; supported: 2..8")
	quantizerSeed := fs.Int64("quantizer-seed", eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed, "TurboQuant IP quantizer seed for deterministic rows")
	rerankOverfetchRaw := fs.String("rerank-overfetch", "", "optional comma-separated TurboQuant candidate depths to rerank, e.g. 200,500")
	rerankStorage := fs.String("rerank-storage", eosruntime.TurboQuantRerankStorageDense, "rerank storage for --rerank-overfetch: dense, compact-reconstruct, or fp16")
	metricsPath := fs.String("metrics-json", "", "write TurboQuant retrieval metrics JSON")
	metricsTSVPath := fs.String("metrics-tsv", "", "write compact dense/quantized metrics TSV")
	perQueryPath := fs.String("per-query-jsonl", "", "write one compact TurboQuant retrieval diagnostics JSONL row per evaluated query and method")
	perQueryTopK := fs.Int("per-query-top-k", 0, "optional retrieval depth for per-query diagnostics only; metrics still use --top-k")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos eval-retrieval-turboquant [flags] <artifact.mll> <beir-dataset-dir>")
	}
	bits, err := parsePositiveIntCSV(*bitsRaw, "bits")
	if err != nil {
		return fmt.Errorf("bits: %w", err)
	}
	rerankOverfetch, err := parseOptionalPositiveIntCSV(*rerankOverfetchRaw, "rerank-overfetch")
	if err != nil {
		return err
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}

	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	metrics, err := eosruntime.EvaluateTurboQuantRetrievalWithRerankStorage(context.Background(), model, eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      artifactPath,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		BatchSize:         *batchSize,
		TopK:              *topK,
		PerQueryTopK:      *perQueryTopK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		QuantizerSeed:     *quantizerSeed,
	}, bits, rerankOverfetch, *rerankStorage)
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	if *metricsTSVPath != "" {
		if err := writeTurboQuantRetrievalMetricsTSV(*metricsTSVPath, metrics); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval turboquant: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("dense: ndcg@10=%.6f ndcg@100=%.6f map@10=%.6f recall@100=%.6f vector_bytes=%d scores/s=%.2f query_p95_ms=%.3f\n",
		metrics.Dense.Quality.NDCGAt10, metrics.Dense.Quality.NDCGAt100, metrics.Dense.Quality.MAPAt10, metrics.Dense.Quality.RecallAt100, metrics.Dense.VectorBytes, metrics.Dense.ScoresPerSecond, metrics.Dense.QueryLatency.P95MS)
	for _, row := range metrics.Rows {
		label := fmt.Sprintf("q%d", row.Bits)
		if row.RerankOverfetch > 0 {
			label = fmt.Sprintf("q%d-rerank%d", row.Bits, row.RerankOverfetch)
		}
		fmt.Printf("%s: ndcg@10=%.6f delta=%+.6f recall@100=%.6f delta=%+.6f vector_bytes=%d total_vector_bytes=%d compression=%.2fx total_compression=%.2fx scores/s=%.2f query_p95_ms=%.3f\n",
			label,
			row.Quality.NDCGAt10,
			row.NDCGAt10Delta,
			row.Quality.RecallAt100,
			row.RecallAt100Delta,
			row.VectorBytes,
			row.TotalVectorBytes,
			row.CompressionRatio,
			row.TotalCompression,
			row.ScoresPerSecond,
			row.QueryLatency.P95MS,
		)
	}
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *metricsTSVPath != "" {
		fmt.Printf("metrics_tsv: %s\n", *metricsTSVPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalRetrievalVectorsTurboQuant(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-vectors-turboquant", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	docVectorsPath := fs.String("doc-vectors", "", "document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	backendName := fs.String("backend", "vectors", "backend/provider label for metrics output")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	bitsRaw := fs.String("bits", "2,4,8", "comma-separated TurboQuant IP bit widths; supported: 2..8")
	quantizerSeed := fs.Int64("quantizer-seed", eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed, "TurboQuant IP quantizer seed for deterministic rows")
	rerankOverfetchRaw := fs.String("rerank-overfetch", "", "optional comma-separated TurboQuant candidate depths to rerank, e.g. 200,500")
	rerankStorage := fs.String("rerank-storage", eosruntime.TurboQuantRerankStorageDense, "rerank storage for --rerank-overfetch: dense, compact-reconstruct, or fp16")
	metricsPath := fs.String("metrics-json", "", "write TurboQuant retrieval metrics JSON")
	metricsTSVPath := fs.String("metrics-tsv", "", "write compact dense/quantized metrics TSV")
	perQueryPath := fs.String("per-query-jsonl", "", "write one compact TurboQuant retrieval diagnostics JSONL row per evaluated query and method")
	perQueryTopK := fs.Int("per-query-top-k", 0, "optional retrieval depth for per-query diagnostics only; metrics still use --top-k")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" || *docVectorsPath == "" || *queryVectorsPath == "" {
		return fmt.Errorf("usage: eos eval-retrieval-vectors-turboquant [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	}
	bits, err := parsePositiveIntCSV(*bitsRaw, "bits")
	if err != nil {
		return fmt.Errorf("bits: %w", err)
	}
	rerankOverfetch, err := parseOptionalPositiveIntCSV(*rerankOverfetchRaw, "rerank-overfetch")
	if err != nil {
		return err
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateTurboQuantVectorCacheRetrievalWithRerankStorage(context.Background(), eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		ArtifactPath:      *artifactLabel,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		DocVectorPath:     *docVectorsPath,
		QueryVectorPath:   *queryVectorsPath,
		BackendName:       *backendName,
		TopK:              *topK,
		PerQueryTopK:      *perQueryTopK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
		QuantizerSeed:     *quantizerSeed,
	}, bits, rerankOverfetch, *rerankStorage)
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	if *metricsTSVPath != "" {
		if err := writeTurboQuantRetrievalMetricsTSV(*metricsTSVPath, metrics); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval vectors turboquant: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("dense: ndcg@10=%.6f ndcg@100=%.6f map@10=%.6f recall@100=%.6f vector_bytes=%d scores/s=%.2f query_p95_ms=%.3f\n",
		metrics.Dense.Quality.NDCGAt10, metrics.Dense.Quality.NDCGAt100, metrics.Dense.Quality.MAPAt10, metrics.Dense.Quality.RecallAt100, metrics.Dense.VectorBytes, metrics.Dense.ScoresPerSecond, metrics.Dense.QueryLatency.P95MS)
	for _, row := range metrics.Rows {
		label := fmt.Sprintf("q%d", row.Bits)
		if row.RerankOverfetch > 0 {
			label = fmt.Sprintf("q%d-rerank%d", row.Bits, row.RerankOverfetch)
		}
		fmt.Printf("%s: ndcg@10=%.6f delta=%+.6f recall@100=%.6f delta=%+.6f vector_bytes=%d total_vector_bytes=%d compression=%.2fx total_compression=%.2fx scores/s=%.2f query_p95_ms=%.3f\n",
			label,
			row.Quality.NDCGAt10,
			row.NDCGAt10Delta,
			row.Quality.RecallAt100,
			row.RecallAt100Delta,
			row.VectorBytes,
			row.TotalVectorBytes,
			row.CompressionRatio,
			row.TotalCompression,
			row.ScoresPerSecond,
			row.QueryLatency.P95MS,
		)
	}
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *metricsTSVPath != "" {
		fmt.Printf("metrics_tsv: %s\n", *metricsTSVPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runEvalRetrievalMultiVectorTurboQuant(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-multivector-turboquant", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	docVectorsPath := fs.String("doc-vectors", "", "parent-child document vector cache JSONL")
	queryVectorsPath := fs.String("query-vectors", "", "query vector cache JSONL")
	backendName := fs.String("backend", "vectors", "backend/provider label for metrics output")
	artifactLabel := fs.String("artifact", "", "external artifact/model label for metrics output")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus parent documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	bitsRaw := fs.String("bits", "2,4,8", "comma-separated TurboQuant IP bit widths; supported: 2..8")
	quantizerSeed := fs.Int64("quantizer-seed", eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed, "TurboQuant IP quantizer seed for deterministic rows")
	baselineDim := fs.Int("baseline-dim", 0, "dense fp32 baseline dimension for one-parent-vector budget; 0 uses child vector dimension")
	aggregation := fs.String("aggregation", eosruntime.TurboQuantMultiVectorAggregationMax, "parent aggregation mode: max, top2-mean, top3-mean, top5-mean")
	childCountPenalty := fs.Float64("child-count-penalty", 0, "subtract FLOAT*log1p(parent child count) from each parent score after aggregation")
	allowMissingRelevant := fs.Bool("allow-missing-relevant", false, "allow qrels-relevant parent IDs missing from the child-vector cache")
	metricsPath := fs.String("metrics-json", "", "write parent-child TurboQuant retrieval metrics JSON")
	metricsTSVPath := fs.String("metrics-tsv", "", "write compact parent-child dense/quantized metrics TSV")
	perQueryPath := fs.String("per-query-jsonl", "", "write one parent-child retrieval diagnostics JSONL row per evaluated query and dense/q-bit method")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" || *docVectorsPath == "" || *queryVectorsPath == "" {
		return fmt.Errorf("usage: eos eval-retrieval-multivector-turboquant [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	}
	bits, err := parsePositiveIntCSV(*bitsRaw, "bits")
	if err != nil {
		return fmt.Errorf("bits: %w", err)
	}
	if *childCountPenalty < 0 {
		return fmt.Errorf("child-count-penalty must be non-negative")
	}
	switch *aggregation {
	case eosruntime.TurboQuantMultiVectorAggregationMax, "top2-mean", "top3-mean", "top5-mean":
	default:
		return fmt.Errorf("aggregation must be one of max, top2-mean, top3-mean, top5-mean")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateTurboQuantMultiVectorCacheRetrieval(context.Background(), eosruntime.RetrievalEvalConfig{
		DatasetName:                  *datasetName,
		ArtifactPath:                 *artifactLabel,
		CorpusPath:                   corpusPath,
		QueriesPath:                  queriesPath,
		QrelsPath:                    *qrelsPath,
		DocVectorPath:                *docVectorsPath,
		QueryVectorPath:              *queryVectorsPath,
		BackendName:                  *backendName,
		TopK:                         *topK,
		PerQueryTopK:                 *topK,
		MaxDocs:                      *maxDocs,
		MaxQueries:                   *maxQueries,
		AllowMissingRelevant:         *allowMissingRelevant,
		QuantizerSeed:                *quantizerSeed,
		BaselineDim:                  *baselineDim,
		MultiVectorAggregation:       *aggregation,
		MultiVectorChildCountPenalty: *childCountPenalty,
		PerQueryJSONLPath:            *perQueryPath,
	}, bits)
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	if *metricsTSVPath != "" {
		if err := writeTurboQuantMultiVectorRetrievalMetricsTSV(*metricsTSVPath, metrics); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval multivector turboquant: dataset=%s backend=%s parents=%d child_vectors=%d avg_children=%.2f queries=%d relevant_pairs=%d scored_child_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Parents, metrics.Inputs.ChildVectors, metrics.Inputs.AverageChildrenPerParent, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredChildPairs)
	fmt.Printf("dense-child: ndcg@10=%.6f ndcg@100=%.6f map@10=%.6f recall@100=%.6f baseline_dim=%d dense_baseline_bytes=%d dense_baseline_total_bytes=%d dense_child_bytes=%d storage_multiple=%.2fx scores/s=%.2f query_p95_ms=%.3f\n",
		metrics.Dense.Quality.NDCGAt10, metrics.Dense.Quality.NDCGAt100, metrics.Dense.Quality.MAPAt10, metrics.Dense.Quality.RecallAt100, metrics.Dense.BaselineDim, metrics.Dense.DenseBaselineBytes, metrics.Dense.DenseBaselineTotalBytes, metrics.Dense.DenseChildBytes, metrics.Dense.StorageMultipleOfDenseBaseline, metrics.Dense.ScoresPerSecond, metrics.Dense.QueryLatency.P95MS)
	for _, row := range metrics.Rows {
		fmt.Printf("q%d: ndcg@10=%.6f delta=%+.6f recall@100=%.6f delta=%+.6f quantized_child_bytes=%d quantized_vector_bytes=%d vectors_per_baseline=%d dense_child_compression=%.2fx storage_multiple=%.2fx scores/s=%.2f query_p95_ms=%.3f\n",
			row.Bits,
			row.Quality.NDCGAt10,
			row.NDCGAt10Delta,
			row.Quality.RecallAt100,
			row.RecallAt100Delta,
			row.QuantizedChildBytes,
			row.QuantizedVectorBytes,
			row.VectorsThatFitInOneDenseBaseline,
			row.DenseChildCompression,
			row.StorageMultipleOfDenseBaseline,
			row.ScoresPerSecond,
			row.QueryLatency.P95MS,
		)
	}
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *metricsTSVPath != "" {
		fmt.Printf("metrics_tsv: %s\n", *metricsTSVPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func writeTurboQuantRetrievalMetricsTSV(path string, metrics eosruntime.TurboQuantRetrievalEvalMetrics) error {
	var b strings.Builder
	b.WriteString("dataset\trow\tbits\tmethod\trerank_overfetch\trerank_storage\tndcg_at_10\tndcg_at_100\tmrr_at_10\tprecision_at_1\tprecision_at_5\tprecision_at_10\thit_at_1\thit_at_5\thit_at_10\tmap_at_10\tmap_at_100\trecall_at_10\trecall_at_100\tndcg_at_10_delta\trecall_at_100_delta\tvector_bytes\tdense_vector_bytes\trerank_sidecar_bytes\ttotal_vector_bytes\tcompression_ratio\ttotal_compression_ratio\tscores_per_second\tquery_latency_p50_ms\tquery_latency_p95_ms\tquery_latency_p99_ms\tquery_latency_max_ms\tdocs_per_second\trerank_scores\n")
	fmt.Fprintf(&b, "%s\tdense\t\tfloat32\t\t\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\t\t%d\t%d\t0\t%d\t%.6f\t%.6f\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t\t\n",
		metrics.Dataset,
		metrics.Dense.Quality.NDCGAt10,
		metrics.Dense.Quality.NDCGAt100,
		metrics.Dense.Quality.MRRAt10,
		metrics.Dense.Quality.PrecisionAt1,
		metrics.Dense.Quality.PrecisionAt5,
		metrics.Dense.Quality.PrecisionAt10,
		metrics.Dense.Quality.HitAt1,
		metrics.Dense.Quality.HitAt5,
		metrics.Dense.Quality.HitAt10,
		metrics.Dense.Quality.MAPAt10,
		metrics.Dense.Quality.MAPAt100,
		metrics.Dense.Quality.RecallAt10,
		metrics.Dense.Quality.RecallAt100,
		metrics.Dense.VectorBytes,
		metrics.Dense.VectorBytes,
		metrics.Dense.VectorBytes,
		1.0,
		1.0,
		metrics.Dense.ScoresPerSecond,
		metrics.Dense.QueryLatency.P50MS,
		metrics.Dense.QueryLatency.P95MS,
		metrics.Dense.QueryLatency.P99MS,
		metrics.Dense.QueryLatency.MaxMS,
	)
	for _, row := range metrics.Rows {
		rowKind := "quantized"
		if row.RerankOverfetch > 0 {
			rowKind = "quantized_rerank"
		}
		fmt.Fprintf(&b, "%s\t%s\t%d\t%s\t%d\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f\t%+.6f\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f\t%d\n",
			metrics.Dataset,
			rowKind,
			row.Bits,
			row.Method,
			row.RerankOverfetch,
			row.RerankStorage,
			row.Quality.NDCGAt10,
			row.Quality.NDCGAt100,
			row.Quality.MRRAt10,
			row.Quality.PrecisionAt1,
			row.Quality.PrecisionAt5,
			row.Quality.PrecisionAt10,
			row.Quality.HitAt1,
			row.Quality.HitAt5,
			row.Quality.HitAt10,
			row.Quality.MAPAt10,
			row.Quality.MAPAt100,
			row.Quality.RecallAt10,
			row.Quality.RecallAt100,
			row.NDCGAt10Delta,
			row.RecallAt100Delta,
			row.VectorBytes,
			row.DenseVectorBytes,
			row.RerankSidecarBytes,
			row.TotalVectorBytes,
			row.CompressionRatio,
			row.TotalCompression,
			row.ScoresPerSecond,
			row.QueryLatency.P50MS,
			row.QueryLatency.P95MS,
			row.QueryLatency.P99MS,
			row.QueryLatency.MaxMS,
			row.DocsPerSecond,
			row.RerankScores,
		)
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeTurboQuantMultiVectorRetrievalMetricsTSV(path string, metrics eosruntime.TurboQuantMultiVectorRetrievalEvalMetrics) error {
	var b strings.Builder
	b.WriteString("dataset\trow\tbits\tmethod\tquantizer_seed\tallow_missing_relevant\tbaseline_dim\tparent_count\tchild_count\tavg_children_per_parent\tmax_children_per_parent\tqueries\trelevant_pairs\tscored_child_pairs\tndcg_at_10\tndcg_at_100\tmrr_at_10\tprecision_at_1\tprecision_at_5\tprecision_at_10\thit_at_1\thit_at_5\thit_at_10\tmap_at_10\tmap_at_100\trecall_at_10\trecall_at_100\tndcg_at_10_delta\trecall_at_100_delta\tdense_baseline_bytes\tdense_baseline_total_bytes\tdense_parent_bytes\tdense_child_bytes\tquantized_vector_bytes\tquantized_child_bytes\tdense_child_compression_ratio\tvectors_that_fit_in_one_dense_baseline\tstorage_multiple_of_dense_baseline\tparent_budget_storage_multiple\tscores_per_second\tquery_latency_p50_ms\tquery_latency_p95_ms\tquery_latency_p99_ms\tquery_latency_max_ms\tchildren_per_second\n")
	fmt.Fprintf(&b, "%s\tdense-child\t\tfloat32_child_max\t%d\t%t\t%d\t%d\t%d\t%.6f\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t\t\t%d\t%d\t%d\t%d\t%d\t%d\t\t%d\t%.6f\t%.6f\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t\n",
		metrics.Dataset,
		metrics.Config.QuantizerSeed,
		metrics.Config.AllowMissingRelevant,
		metrics.Config.BaselineDim,
		metrics.Inputs.ParentCount,
		metrics.Inputs.ChildCount,
		metrics.Inputs.AvgChildrenPerParent,
		metrics.Inputs.MaxChildrenPerParent,
		metrics.Inputs.Queries,
		metrics.Inputs.RelevantPairs,
		metrics.Inputs.ScoredChildPairs,
		metrics.Dense.Quality.NDCGAt10,
		metrics.Dense.Quality.NDCGAt100,
		metrics.Dense.Quality.MRRAt10,
		metrics.Dense.Quality.PrecisionAt1,
		metrics.Dense.Quality.PrecisionAt5,
		metrics.Dense.Quality.PrecisionAt10,
		metrics.Dense.Quality.HitAt1,
		metrics.Dense.Quality.HitAt5,
		metrics.Dense.Quality.HitAt10,
		metrics.Dense.Quality.MAPAt10,
		metrics.Dense.Quality.MAPAt100,
		metrics.Dense.Quality.RecallAt10,
		metrics.Dense.Quality.RecallAt100,
		metrics.Dense.DenseBaselineBytes,
		metrics.Dense.DenseBaselineTotalBytes,
		metrics.Dense.DenseParentBytes,
		metrics.Dense.DenseChildBytes,
		metrics.Dense.QuantizedVectorBytes,
		metrics.Dense.DenseChildBytes,
		metrics.Dense.VectorsThatFitInOneDenseBaseline,
		metrics.Dense.StorageMultipleOfDenseBaseline,
		metrics.Dense.ParentBudgetStorageMultiple,
		metrics.Dense.ScoresPerSecond,
		metrics.Dense.QueryLatency.P50MS,
		metrics.Dense.QueryLatency.P95MS,
		metrics.Dense.QueryLatency.P99MS,
		metrics.Dense.QueryLatency.MaxMS,
	)
	for _, row := range metrics.Rows {
		fmt.Fprintf(&b, "%s\tquantized-child\t%d\t%s\t%d\t%t\t%d\t%d\t%d\t%.6f\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f\t%+.6f\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%d\t%.6f\t%.6f\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%.2f\n",
			metrics.Dataset,
			row.Bits,
			row.Method,
			row.QuantizerSeed,
			metrics.Config.AllowMissingRelevant,
			row.BaselineDim,
			row.ParentCount,
			row.ChildCount,
			row.AvgChildrenPerParent,
			row.MaxChildrenPerParent,
			metrics.Inputs.Queries,
			metrics.Inputs.RelevantPairs,
			metrics.Inputs.ScoredChildPairs,
			row.Quality.NDCGAt10,
			row.Quality.NDCGAt100,
			row.Quality.MRRAt10,
			row.Quality.PrecisionAt1,
			row.Quality.PrecisionAt5,
			row.Quality.PrecisionAt10,
			row.Quality.HitAt1,
			row.Quality.HitAt5,
			row.Quality.HitAt10,
			row.Quality.MAPAt10,
			row.Quality.MAPAt100,
			row.Quality.RecallAt10,
			row.Quality.RecallAt100,
			row.NDCGAt10Delta,
			row.RecallAt100Delta,
			row.DenseBaselineBytes,
			row.DenseBaselineTotalBytes,
			row.DenseParentBytes,
			row.DenseChildBytes,
			row.QuantizedVectorBytes,
			row.QuantizedChildBytes,
			row.DenseChildCompression,
			row.VectorsThatFitInOneDenseBaseline,
			row.StorageMultipleOfDenseBaseline,
			row.ParentBudgetStorageMultiple,
			row.ScoresPerSecond,
			row.QueryLatency.P50MS,
			row.QueryLatency.P95MS,
			row.QueryLatency.P99MS,
			row.QueryLatency.MaxMS,
			row.ChildrenPerSecond,
		)
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func runEvalRetrievalBM25(args []string) error {
	fs := flag.NewFlagSet("eval-retrieval-bm25", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	topK := fs.Int("top-k", 100, "retrieval depth for scoring")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-retrieval-bm25 [flags] <beir-dataset-dir>")
	}
	datasetDir := fs.Arg(0)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateBM25Retrieval(context.Background(), eosruntime.RetrievalEvalConfig{
		DatasetName:       *datasetName,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		TopK:              *topK,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval bm25: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs)
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs docs/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runExportSparseLexicalLabels(args []string) error {
	fs := flag.NewFlagSet("export-sparse-lexical-labels", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for label records and manifest")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	topTerms := fs.Int("top-terms", 128, "maximum nonzero lexical terms per document or query")
	hashBins := fs.Int("hash-bins", 0, "optional deterministic FNV-1a hash bins recorded on each term; 0 disables")
	manifestPath := fs.String("manifest-json", "", "write sparse lexical label manifest and oracle summary JSON")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks while retaining qrels-relevant documents")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	oracleTopK := fs.Int("oracle-top-k", 100, "top-k depth for sparse-only oracle summary")
	oracleMaxQueries := fs.Int("oracle-max-queries", 0, "limit queries used for oracle summary; 0 uses all exported queries")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos export-sparse-lexical-labels [flags] <beir-dataset-dir> <labels.jsonl>")
	}
	datasetDir := fs.Arg(0)
	labelsPath := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	summary, err := eosruntime.ExportSparseLexicalLabels(context.Background(), eosruntime.SparseLexicalLabelExportConfig{
		DatasetName:      *datasetName,
		Split:            *split,
		CorpusPath:       corpusPath,
		QueriesPath:      queriesPath,
		QrelsPath:        *qrelsPath,
		OutputPath:       labelsPath,
		ManifestPath:     *manifestPath,
		TopTerms:         *topTerms,
		HashBins:         *hashBins,
		MaxDocs:          *maxDocs,
		MaxQueries:       *maxQueries,
		OracleTopK:       *oracleTopK,
		OracleMaxQueries: *oracleMaxQueries,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported sparse lexical labels: dataset=%s split=%s docs=%d queries=%d top_terms=%d hash_bins=%d\n",
		summary.Dataset, summary.Split, summary.Stats.Documents, summary.Stats.Queries, summary.Config.TopTerms, summary.Config.HashBins)
	fmt.Printf("density: doc_avg_nonzeros=%.2f doc_max_nonzeros=%d query_avg_nonzeros=%.2f query_max_nonzeros=%d\n",
		summary.Stats.DocumentAvgNNZ, summary.Stats.DocumentMaxNNZ, summary.Stats.QueryAvgNNZ, summary.Stats.QueryMaxNNZ)
	fmt.Printf("truncation: document_records=%d document_terms_omitted=%d query_records=%d query_terms_omitted=%d exported_terms_exact=%t\n",
		summary.Stats.DocumentTruncated, summary.Stats.DocumentOmitted, summary.Stats.QueryTruncated, summary.Stats.QueryOmitted, summary.Oracle.ExportedTermsExact)
	fmt.Printf("oracle: queries=%d max_abs_score_delta=%.12g exact=%t reconstruction_terms=%s ndcg@10=%.6f recall@100=%.6f\n",
		summary.Oracle.Queries, summary.Oracle.MaxAbsScoreDelta, summary.Oracle.ExactScoreReconstruction, summary.Oracle.ReconstructionTerms, summary.Oracle.NDCGAt10, summary.Oracle.RecallAt100)
	fmt.Printf("labels: %s\n", labelsPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

func runEvalSparseLexicalLabels(args []string) error {
	fs := flag.NewFlagSet("eval-sparse-lexical-labels", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL document/query universe from export-sparse-lexical-labels")
	topK := fs.Int("top-k", 100, "retrieval depth over label-file document records")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-sparse-lexical-labels [flags] --labels labels.jsonl <beir-dataset-dir>")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	datasetDir := fs.Arg(0)
	_, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateSparseLexicalLabels(context.Background(), eosruntime.SparseLexicalLabelEvalConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		LabelsPath:        *labelsPath,
		TopK:              *topK,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval sparse lexical labels: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d top_k=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs, metrics.Config.TopK)
	if metrics.SparseLexical != nil {
		fmt.Printf("labels: documents=%d queries=%d doc_avg_nonzeros=%.2f doc_max_nonzeros=%d query_avg_nonzeros=%.2f query_max_nonzeros=%d representation=%s\n",
			metrics.SparseLexical.DocumentLabels, metrics.SparseLexical.QueryLabels, metrics.SparseLexical.DocumentAvgNNZ, metrics.SparseLexical.DocumentMaxNNZ, metrics.SparseLexical.QueryAvgNNZ, metrics.SparseLexical.QueryMaxNNZ, metrics.SparseLexical.Representation)
	}
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs labels/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	fmt.Printf("labels: %s\n", *labelsPath)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runFitSparseLexicalHead(args []string) error {
	fs := flag.NewFlagSet("fit-sparse-lexical-head", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for head metadata")
	split := fs.String("split", "train", "label split for validation")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL from export-sparse-lexical-labels")
	hashBins := fs.Int("hash-bins", 65536, "deterministic FNV-1a hashed sparse head bins")
	headPath := fs.String("head-json", "", "write experimental sparse lexical hash head JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos fit-sparse-lexical-head --labels labels.jsonl --hash-bins 65536 --head-json head.json")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	head, err := eosruntime.FitSparseLexicalHashHead(eosruntime.SparseLexicalHashHeadFitConfig{
		DatasetName: *datasetName,
		Split:       *split,
		LabelsPath:  *labelsPath,
		HeadPath:    *headPath,
		HashBins:    *hashBins,
	})
	if err != nil {
		return err
	}
	fmt.Printf("fit sparse lexical hash head: schema=%s experimental=%t dataset=%s split=%s hash_bins=%d\n",
		head.Schema, head.Experimental, head.Dataset, head.Split, head.Hashing.Bins)
	fmt.Printf("labels: documents=%d queries=%d doc_avg_hash_nonzeros=%.2f doc_max_hash_nonzeros=%d query_avg_hash_nonzeros=%.2f query_max_hash_nonzeros=%d doc_merged_bins=%d query_merged_bins=%d\n",
		head.Stats.DocumentLabels, head.Stats.QueryLabels, head.Stats.DocumentAvgHashNNZ, head.Stats.DocumentMaxHashNNZ, head.Stats.QueryAvgHashNNZ, head.Stats.QueryMaxHashNNZ, head.Stats.DocumentMergedBins, head.Stats.QueryMergedBins)
	fmt.Printf("labels: %s\n", *labelsPath)
	fmt.Printf("head: %s\n", *headPath)
	return nil
}

func runEvalSparseLexicalHead(args []string) error {
	fs := flag.NewFlagSet("eval-sparse-lexical-head", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for metrics output")
	split := fs.String("split", "test", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	labelsPath := fs.String("labels", "", "sparse lexical label JSONL document/query universe from export-sparse-lexical-labels")
	headPath := fs.String("head-json", "", "experimental sparse lexical hash head JSON")
	topK := fs.Int("top-k", 100, "retrieval depth over label-file document records")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	metricsPath := fs.String("metrics-json", "", "write retrieval metrics JSON")
	perQueryPath := fs.String("per-query-jsonl", "", "write one retrieval diagnostics JSONL row per evaluated query")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos eval-sparse-lexical-head [flags] --labels labels.jsonl --head-json head.json <beir-dataset-dir>")
	}
	if *labelsPath == "" {
		return fmt.Errorf("labels is required")
	}
	if *headPath == "" {
		return fmt.Errorf("head-json is required")
	}
	datasetDir := fs.Arg(0)
	_, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	metrics, err := eosruntime.EvaluateSparseLexicalHashHead(context.Background(), eosruntime.SparseLexicalHashHeadEvalConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		LabelsPath:        *labelsPath,
		HeadPath:          *headPath,
		TopK:              *topK,
		MaxQueries:        *maxQueries,
		PerQueryJSONLPath: *perQueryPath,
	})
	if err != nil {
		return err
	}
	if *metricsPath != "" {
		data, err := json.MarshalIndent(metrics, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*metricsPath, data, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("retrieval sparse lexical hash head: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d top_k=%d\n",
		metrics.Dataset, metrics.Backend, metrics.Inputs.Documents, metrics.Inputs.Queries, metrics.Inputs.RelevantPairs, metrics.Inputs.ScoredPairs, metrics.Config.TopK)
	if metrics.SparseLexical != nil {
		fmt.Printf("head: hash_bins=%d documents=%d queries=%d doc_avg_hash_nonzeros=%.2f doc_max_hash_nonzeros=%d query_avg_hash_nonzeros=%.2f query_max_hash_nonzeros=%d doc_merged_bins=%d query_merged_bins=%d representation=%s\n",
			metrics.SparseLexical.HashBins, metrics.SparseLexical.DocumentLabels, metrics.SparseLexical.QueryLabels, metrics.SparseLexical.DocumentAvgHashNNZ, metrics.SparseLexical.DocumentMaxHashNNZ, metrics.SparseLexical.QueryAvgHashNNZ, metrics.SparseLexical.QueryMaxHashNNZ, metrics.SparseLexical.DocumentMergedBins, metrics.SparseLexical.QueryMergedBins, metrics.SparseLexical.Representation)
	}
	fmt.Printf("quality: ndcg@10=%.6f ndcg@100=%.6f mrr@10=%.6f p@1=%.6f p@5=%.6f p@10=%.6f hit@1=%.6f hit@5=%.6f hit@10=%.6f map@10=%.6f map@100=%.6f recall@10=%.6f recall@100=%.6f\n",
		metrics.Quality.NDCGAt10, metrics.Quality.NDCGAt100, metrics.Quality.MRRAt10, metrics.Quality.PrecisionAt1, metrics.Quality.PrecisionAt5, metrics.Quality.PrecisionAt10, metrics.Quality.HitAt1, metrics.Quality.HitAt5, metrics.Quality.HitAt10, metrics.Quality.MAPAt10, metrics.Quality.MAPAt100, metrics.Quality.RecallAt10, metrics.Quality.RecallAt100)
	fmt.Printf("throughput: elapsed=%.3fs labels/s=%.2f queries/s=%.2f scores/s=%.2f\n",
		metrics.Throughput.ElapsedSeconds, metrics.Throughput.DocumentsPerSecond, metrics.Throughput.QueriesPerSecond, metrics.Throughput.ScoresPerSecond)
	fmt.Printf("labels: %s\n", *labelsPath)
	fmt.Printf("head: %s\n", *headPath)
	if *metricsPath != "" {
		fmt.Printf("metrics: %s\n", *metricsPath)
	}
	if *perQueryPath != "" {
		fmt.Printf("per_query: %s\n", *perQueryPath)
	}
	return nil
}

func runMineRetrievalHardNegatives(args []string) error {
	fs := flag.NewFlagSet("mine-retrieval-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for status output")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	negatives := fs.Int("negatives", 1, "BM25 hard negatives per positive qrel")
	candidateTopK := fs.Int("candidate-top-k", 100, "BM25 candidate depth to mine negatives from")
	maxExamples := fs.Int("max-examples", 0, "limit mined hard-negative examples")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	dfPruneThreshold := fs.Float64("df-prune-threshold", eosruntime.DefaultBM25MiningDFPruneThreshold, "skip query terms whose document frequency exceeds this fraction of the corpus during BM25 candidate generation (fraction of corpus documents; <=0 or >=1 disables pruning)")
	miningWorkers := fs.Int("mining-workers", 0, "parallel query workers mining against the shared read-only BM25 index (<=0: auto, min(8, GOMAXPROCS))")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos mine-retrieval-hard-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	}
	if *negatives <= 0 {
		return fmt.Errorf("negatives must be positive")
	}
	if *candidateTopK <= 0 {
		return fmt.Errorf("candidate-top-k must be positive")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	datasetDir := fs.Arg(0)
	outputPath := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	examples, summary, err := eosruntime.MineBM25TextHardNegatives(context.Background(), eosruntime.RetrievalHardNegativeMiningConfig{
		DatasetName:          *datasetName,
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            *qrelsPath,
		NegativesPerPositive: *negatives,
		CandidateTopK:        *candidateTopK,
		MaxExamples:          *maxExamples,
		MaxDocs:              *maxDocs,
		MaxQueries:           *maxQueries,
		DFPruneThreshold:     *dfPruneThreshold,
		MiningWorkers:        *miningWorkers,
	})
	if err != nil {
		return err
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	fmt.Printf("mined retrieval hard negatives: dataset=%s examples=%d positives=%d negatives=%d queries=%d\n",
		summary.DatasetName, summary.Examples, summary.PositivePairs, summary.Negatives, summary.Queries)
	fmt.Printf("skipped: queries_without_text=%d positives_without_text=%d queries_without_negatives=%d duplicate_positive_text_negatives=%d\n",
		summary.SkippedQueriesNoText, summary.SkippedPositiveDocs, summary.SkippedQueriesNoNegative, summary.DuplicatePositiveTextNegativesSkipped)
	fmt.Printf("config: df_prune_threshold=%.4f mining_workers=%d\n", summary.DFPruneThreshold, summary.MiningWorkers)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

func runMineRetrievalModelHardNegatives(args []string) error {
	fs := flag.NewFlagSet("mine-retrieval-model-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for status output")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	negatives := fs.Int("negatives", 1, "model-ranked hard negatives per positive qrel")
	candidateTopK := fs.Int("candidate-top-k", 100, "model-ranked candidate depth to mine negatives from")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	maxExamples := fs.Int("max-examples", 0, "limit mined hard-negative examples")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	roleMode := fs.String("role-mode", eosruntime.EmbeddingRoleModeAuto, "embedding role mode: auto, raw, or query-document")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos mine-retrieval-model-hard-negatives [flags] <artifact.mll> <beir-dataset-dir> <output.jsonl>")
	}
	if *negatives <= 0 {
		return fmt.Errorf("negatives must be positive")
	}
	if *candidateTopK <= 0 {
		return fmt.Errorf("candidate-top-k must be positive")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	artifactPath := fs.Arg(0)
	datasetDir := fs.Arg(1)
	outputPath := fs.Arg(2)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	examples, summary, err := eosruntime.MineModelTextHardNegatives(context.Background(), model, eosruntime.RetrievalHardNegativeMiningConfig{
		DatasetName:          *datasetName,
		CorpusPath:           corpusPath,
		QueriesPath:          queriesPath,
		QrelsPath:            *qrelsPath,
		NegativesPerPositive: *negatives,
		CandidateTopK:        *candidateTopK,
		BatchSize:            *batchSize,
		MaxExamples:          *maxExamples,
		MaxDocs:              *maxDocs,
		MaxQueries:           *maxQueries,
		RoleMode:             *roleMode,
	})
	if err != nil {
		return err
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	fmt.Printf("mined model retrieval hard negatives: dataset=%s backend=%s role_mode=%s examples=%d positives=%d negatives=%d queries=%d\n",
		summary.DatasetName, model.Backend(), summary.RoleMode, summary.Examples, summary.PositivePairs, summary.Negatives, summary.Queries)
	fmt.Printf("skipped: queries_without_text=%d positives_without_text=%d queries_without_negatives=%d duplicate_positive_text_negatives=%d\n",
		summary.SkippedQueriesNoText, summary.SkippedPositiveDocs, summary.SkippedQueriesNoNegative, summary.DuplicatePositiveTextNegativesSkipped)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

func runMineRetrievalCompactHardNegatives(args []string) error {
	fs := flag.NewFlagSet("mine-retrieval-compact-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	datasetName := fs.String("dataset", "", "dataset name for status output")
	split := fs.String("split", "train", "qrels split under <dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path")
	perQueryPath := fs.String("per-query-jsonl", "", "compact per-query diagnostics JSONL from eval-retrieval-turboquant")
	manifestPath := fs.String("manifest-json", "", "write compact hard-negative mining manifest JSON")
	method := fs.String("method", "", "compact diagnostics method to mine, default derives from bits/overfetch/rerank-storage")
	bits := fs.Int("bits", 4, "TurboQuant bit width to mine")
	overfetch := fs.Int("overfetch", 0, "TurboQuant overfetch depth for rerank rows")
	rerankStorage := fs.String("rerank-storage", eosruntime.TurboQuantRerankStorageFP16, "rerank storage for rerank rows")
	quantizerSeed := fs.Int64("quantizer-seed", eosruntime.DefaultTurboQuantMultiVectorQuantizerSeed, "TurboQuant quantizer seed recorded in manifest")
	artifactSHA := fs.String("artifact-sha256", "", "optional source artifact SHA256 for manifest provenance")
	negatives := fs.Int("negatives", 4, "compact hard negatives per emitted row")
	maxExamples := fs.Int("max-examples", 0, "limit mined hard-negative examples")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke checks")
	maxQueries := fs.Int("max-queries", 0, "limit qrels queries for smoke checks")
	trainSelection := fs.Bool("train-selection", true, "allow rows to be used for training selection when split is non-test")
	allowTestSmoke := fs.Bool("allow-test-smoke", false, "permit test split only as no-train validation smoke")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos mine-retrieval-compact-hard-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	}
	if *perQueryPath == "" {
		return fmt.Errorf("per-query-jsonl is required")
	}
	if *bits <= 0 {
		return fmt.Errorf("bits must be positive")
	}
	if *overfetch < 0 {
		return fmt.Errorf("overfetch must be non-negative")
	}
	if *negatives <= 0 {
		return fmt.Errorf("negatives must be positive")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	datasetDir := fs.Arg(0)
	outputPath := fs.Arg(1)
	corpusPath, queriesPath, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	manifest, err := eosruntime.MineCompactTextHardNegatives(context.Background(), eosruntime.CompactHardNegativeMiningConfig{
		DatasetName:       *datasetName,
		Split:             *split,
		CorpusPath:        corpusPath,
		QueriesPath:       queriesPath,
		QrelsPath:         *qrelsPath,
		PerQueryJSONLPath: *perQueryPath,
		OutputPath:        outputPath,
		ManifestPath:      *manifestPath,
		Method:            *method,
		BitWidth:          *bits,
		Overfetch:         *overfetch,
		RerankStorage:     *rerankStorage,
		QuantizerSeed:     *quantizerSeed,
		TrainSelection:    *trainSelection,
		AllowTestSmoke:    *allowTestSmoke,
		NegativesPerRow:   *negatives,
		MaxExamples:       *maxExamples,
		MaxDocs:           *maxDocs,
		MaxQueries:        *maxQueries,
		ArtifactSHA256:    *artifactSHA,
	})
	if err != nil {
		return err
	}
	fmt.Printf("mined compact retrieval hard negatives: dataset=%s split=%s method=%s examples=%d positives=%d negatives=%d train_allowed=%t\n",
		manifest.Dataset, manifest.Split, manifest.Method, manifest.RowsEmitted, manifest.PositivePairs, manifest.Negatives, manifest.TrainAllowed)
	fmt.Printf("leak_guard: %s\n", manifest.LeakGuardStatus)
	fmt.Printf("rows: read=%d matched=%d emitted=%d\n", manifest.RowsRead, manifest.RowsMatched, manifest.RowsEmitted)
	fmt.Printf("output: %s\n", outputPath)
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	}
	return nil
}

type teacherScoreImportRecord struct {
	Source         string    `json:"source,omitempty"`
	Query          string    `json:"query,omitempty"`
	Positive       string    `json:"positive,omitempty"`
	Document       string    `json:"document,omitempty"`
	Candidate      string    `json:"candidate,omitempty"`
	Text           string    `json:"text,omitempty"`
	Score          *float64  `json:"score,omitempty"`
	Scores         []float64 `json:"scores,omitempty"`
	TeacherScores  []float64 `json:"teacher_scores,omitempty"`
	PositiveScore  *float64  `json:"positive_score,omitempty"`
	NegativeScores []float64 `json:"negative_scores,omitempty"`
}

type teacherScoreRequestRecord struct {
	Source         string `json:"source,omitempty"`
	Query          string `json:"query"`
	Candidate      string `json:"candidate"`
	Role           string `json:"role"`
	ExampleIndex   int    `json:"example_index"`
	CandidateIndex int    `json:"candidate_index"`
}

type teacherScoreImportTable struct {
	ExampleScores   map[string][]float32
	CandidateScores map[string]float32
	ExampleRows     int
	CandidateRows   int
}

type teacherScoreRequestSummary struct {
	Schema           string `json:"schema"`
	CreatedUTC       string `json:"created_utc"`
	InputJSONL       string `json:"input_jsonl"`
	OutputJSONL      string `json:"output_jsonl"`
	Examples         int    `json:"examples"`
	ExportedExamples int    `json:"exported_examples"`
	SkippedExisting  int    `json:"skipped_existing"`
	Rows             int    `json:"rows"`
	PositiveRows     int    `json:"positive_rows"`
	NegativeRows     int    `json:"negative_rows"`
}

type teacherScoreImportSummary struct {
	Schema          string `json:"schema"`
	CreatedUTC      string `json:"created_utc"`
	InputJSONL      string `json:"input_jsonl"`
	ScoresJSONL     string `json:"scores_jsonl"`
	OutputJSONL     string `json:"output_jsonl"`
	TeacherModelID  string `json:"teacher_model_id,omitempty"`
	TeacherRevision string `json:"teacher_revision,omitempty"`
	Prompt          string `json:"prompt,omitempty"`
	ScoreScale      string `json:"score_scale,omitempty"`
	Examples        int    `json:"examples"`
	Updated         int    `json:"updated"`
	SkippedExisting int    `json:"skipped_existing"`
	SkippedMissing  int    `json:"skipped_missing"`
	ExampleRows     int    `json:"example_rows"`
	CandidateRows   int    `json:"candidate_rows"`
}

type teacherHardNegativeScoreSummary struct {
	Schema          string `json:"schema"`
	CreatedUTC      string `json:"created_utc"`
	TeacherArtifact string `json:"teacher_artifact"`
	InputJSONL      string `json:"input_jsonl"`
	OutputJSONL     string `json:"output_jsonl"`
	TeacherModelID  string `json:"teacher_model_id,omitempty"`
	TeacherRevision string `json:"teacher_revision,omitempty"`
	Prompt          string `json:"prompt,omitempty"`
	ScoreScale      string `json:"score_scale"`
	Backend         string `json:"backend,omitempty"`
	BatchSize       int    `json:"batch_size"`
	Examples        int    `json:"examples"`
	Updated         int    `json:"updated"`
	SkippedExisting int    `json:"skipped_existing"`
}

type teacherScoreAuditSummary struct {
	Schema      string  `json:"schema"`
	CreatedUTC  string  `json:"created_utc"`
	InputJSONL  string  `json:"input_jsonl"`
	Mode        string  `json:"mode"`
	Temperature float64 `json:"temperature"`
	LabelPolicy string  `json:"label_policy,omitempty"`
	teacherScoreAuditStats
	Sources map[string]teacherScoreAuditStats `json:"sources,omitempty"`
}

type teacherScoreFilterConfig struct {
	RequirePositiveTop1  bool    `json:"require_positive_top1"`
	MinMargin            float64 `json:"min_margin"`
	MaxNormalizedEntropy float64 `json:"max_normalized_entropy,omitempty"`
	Temperature          float64 `json:"temperature"`
	DropFailingExamples  bool    `json:"drop_failing_examples"`
}

type teacherScoreFilterSummary struct {
	Schema                 string                             `json:"schema"`
	CreatedUTC             string                             `json:"created_utc"`
	InputJSONL             string                             `json:"input_jsonl"`
	OutputJSONL            string                             `json:"output_jsonl"`
	Mode                   string                             `json:"mode"`
	Config                 teacherScoreFilterConfig           `json:"filter_config"`
	Examples               int                                `json:"examples"`
	Scored                 int                                `json:"scored"`
	Missing                int                                `json:"missing"`
	KeptTeacherScores      int                                `json:"kept_teacher_scores"`
	ClearedTeacherScores   int                                `json:"cleared_teacher_scores"`
	DroppedExamples        int                                `json:"dropped_examples"`
	Candidates             int                                `json:"candidates"`
	PositiveTop1RateBefore float64                            `json:"positive_top1_rate_before"`
	PositiveTop1RateAfter  float64                            `json:"positive_top1_rate_after"`
	TeacherScoreKeptRate   float64                            `json:"teacher_score_kept_rate"`
	MeanMarginBefore       float64                            `json:"mean_margin_before"`
	MeanMarginAfter        float64                            `json:"mean_margin_after"`
	Sources                map[string]teacherScoreFilterStats `json:"sources,omitempty"`
}

type teacherScoreFilterStats struct {
	Examples               int     `json:"examples"`
	Scored                 int     `json:"scored"`
	Missing                int     `json:"missing"`
	KeptTeacherScores      int     `json:"kept_teacher_scores"`
	ClearedTeacherScores   int     `json:"cleared_teacher_scores"`
	DroppedExamples        int     `json:"dropped_examples"`
	Candidates             int     `json:"candidates"`
	PositiveTop1RateBefore float64 `json:"positive_top1_rate_before"`
	PositiveTop1RateAfter  float64 `json:"positive_top1_rate_after"`
	TeacherScoreKeptRate   float64 `json:"teacher_score_kept_rate"`
	MeanMarginBefore       float64 `json:"mean_margin_before"`
	MeanMarginAfter        float64 `json:"mean_margin_after"`
}

type teacherScoreFilterCounters struct {
	Examples             int
	Scored               int
	Missing              int
	KeptTeacherScores    int
	ClearedTeacherScores int
	DroppedExamples      int
	Candidates           int
	PositiveTop1Before   int
	PositiveTop1After    int
	MarginBeforeSum      float64
	MarginAfterSum       float64
}

type teacherScoreAuditStats struct {
	Examples                               int      `json:"examples"`
	ScoredExamples                         int      `json:"scored_examples"`
	MissingExamples                        int      `json:"missing_examples"`
	Candidates                             int      `json:"candidates"`
	ScoredCandidates                       int      `json:"scored_candidates"`
	PositiveTop1                           int      `json:"positive_top1"`
	PositiveTop1Rate                       float64  `json:"positive_top1_rate"`
	PositiveMeanRank                       float64  `json:"positive_mean_rank"`
	PositiveMeanMargin                     float64  `json:"positive_mean_margin"`
	AnyPositiveExamples                    *int     `json:"any_positive_examples,omitempty"`
	AnyPositiveTop1                        *int     `json:"any_positive_top1,omitempty"`
	AnyPositiveTop1Rate                    *float64 `json:"any_positive_top1_rate,omitempty"`
	AnyPositiveMeanRank                    *float64 `json:"any_positive_mean_rank,omitempty"`
	AnyPositiveMeanMargin                  *float64 `json:"any_positive_mean_margin,omitempty"`
	DuplicatePositiveNegativeCandidates    *int     `json:"duplicate_positive_negative_candidates,omitempty"`
	ExamplesWithDuplicatePositiveNegatives *int     `json:"examples_with_duplicate_positive_negatives,omitempty"`
	MeanScore                              float64  `json:"mean_score"`
	MeanScoreRange                         float64  `json:"mean_score_range"`
	MeanEntropy                            float64  `json:"mean_entropy"`
	MeanNormalizedEntropy                  float64  `json:"mean_normalized_entropy"`
}

type teacherScoreAuditCounters struct {
	Examples                               int
	ScoredExamples                         int
	MissingExamples                        int
	Candidates                             int
	ScoredCandidates                       int
	PositiveTop1                           int
	PositiveRankSum                        float64
	PositiveMarginSum                      float64
	AnyPositiveExamples                    int
	AnyPositiveTop1                        int
	AnyPositiveRankSum                     float64
	AnyPositiveMarginSum                   float64
	DuplicatePositiveNegativeCandidates    int
	ExamplesWithDuplicatePositiveNegatives int
	ScoreSum                               float64
	ScoreRangeSum                          float64
	EntropySum                             float64
	NormalizedEntropySum                   float64
}

type sparseAttentionPlanReport struct {
	Schema     string                    `json:"schema"`
	CreatedUTC string                    `json:"created_utc"`
	Config     sparseAttentionPlanConfig `json:"config"`
	Gate       sparseAttentionPlanGate   `json:"gate"`
	Rows       []sparseAttentionPlanRow  `json:"rows"`
}

type sparseAttentionPlanConfig struct {
	KeyLens          []int   `json:"key_lens"`
	QueryLen         int     `json:"query_len"`
	QueryDim         int     `json:"query_dim"`
	ValueDim         int     `json:"value_dim"`
	TopK             int     `json:"top_k"`
	RouteBlockSize   int     `json:"route_block_size"`
	RouteTopBlocks   int     `json:"route_top_blocks"`
	Exact            bool    `json:"exact"`
	Bits             int     `json:"bits"`
	Batches          int     `json:"batches"`
	MaxScoreFraction float64 `json:"max_score_fraction,omitempty"`
	MaxTurboKVMiB    float64 `json:"max_turbo_kv_mib,omitempty"`
	RequireSubq      bool    `json:"require_subquadratic,omitempty"`
}

type sparseAttentionPlanGate struct {
	Passed                   bool     `json:"passed"`
	FailureReasons           []string `json:"failure_reasons,omitempty"`
	Rows                     int      `json:"rows"`
	SubquadraticRows         int      `json:"subquadratic_rows"`
	MaxScoreFractionObserved float64  `json:"max_score_fraction_observed"`
	MaxTurboKVMiBObserved    float64  `json:"max_turbo_kv_mib_observed"`
	ScoreAlpha               float64  `json:"score_alpha"`
}

type sparseAttentionPlanRow struct {
	KeyLen                      int     `json:"key_len"`
	QueryLen                    int     `json:"query_len"`
	QueryDim                    int     `json:"query_dim"`
	ValueDim                    int     `json:"value_dim"`
	TopK                        int     `json:"top_k"`
	Routing                     string  `json:"routing"`
	RouteBlockSize              int     `json:"route_block_size"`
	RouteTopBlocks              int     `json:"route_top_blocks"`
	RouteBlockCount             int     `json:"route_block_count"`
	SelectedRouteBlocks         int     `json:"selected_route_blocks"`
	SelectedKeyCount            int     `json:"selected_key_count"`
	CandidateKeyBudget          int     `json:"candidate_key_budget"`
	DenseScoreCountPerQuery     int     `json:"dense_score_count_per_query"`
	EstimatedScoreCountPerQuery int     `json:"estimated_score_count_per_query"`
	ScoreCountFraction          float64 `json:"score_count_fraction"`
	CandidateKeyFraction        float64 `json:"candidate_key_fraction"`
	SubquadraticScorePlan       bool    `json:"subquadratic_score_plan"`
	DenseTotalScoreCount        int64   `json:"dense_total_score_count"`
	EstimatedTotalScoreCount    int64   `json:"estimated_total_score_count"`
	Bits                        int     `json:"bits"`
	DenseKVBytes                int64   `json:"dense_kv_bytes"`
	TurboQuantKVBytes           int64   `json:"turboquant_kv_bytes"`
	TurboQuantKVMiB             float64 `json:"turboquant_kv_mib"`
	TurboQuantCompressionRatio  float64 `json:"turboquant_compression_ratio"`
}

func runExportTeacherScoreRequests(args []string) error {
	fs := flag.NewFlagSet("export-teacher-score-requests", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	missingOnly := fs.Bool("missing-only", false, "only export examples that do not already have teacher_scores")
	maxExamples := fs.Int("max-examples", 0, "maximum examples to export; 0 means all")
	manifestPath := fs.String("manifest", "", "request manifest path; default is <output>.teacher-score-requests.manifest.json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos export-teacher-score-requests [flags] <hard-negatives.jsonl> <requests.jsonl>")
	}
	if *maxExamples < 0 {
		return fmt.Errorf("max-examples must be non-negative")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".teacher-score-requests.manifest.json"
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(out)
	summary := teacherScoreRequestSummary{
		Schema:      "manta.teacher_score_requests.v1",
		CreatedUTC:  time.Now().UTC().Format(time.RFC3339),
		InputJSONL:  inputPath,
		OutputJSONL: outputPath,
		Examples:    len(examples),
	}
	for exampleIndex, example := range examples {
		if *maxExamples > 0 && summary.ExportedExamples >= *maxExamples {
			break
		}
		if *missingOnly && len(example.TeacherScores) > 0 {
			summary.SkippedExisting++
			continue
		}
		candidates := append([]string{example.Positive}, example.Negatives...)
		for candidateIndex, candidate := range candidates {
			role := "negative"
			if candidateIndex == 0 {
				role = "positive"
			}
			record := teacherScoreRequestRecord{
				Source:         example.Source,
				Query:          example.Query,
				Candidate:      candidate,
				Role:           role,
				ExampleIndex:   exampleIndex,
				CandidateIndex: candidateIndex,
			}
			if err := enc.Encode(record); err != nil {
				_ = out.Close()
				return err
			}
			summary.Rows++
			if role == "positive" {
				summary.PositiveRows++
			} else {
				summary.NegativeRows++
			}
		}
		summary.ExportedExamples++
	}
	if err := out.Close(); err != nil {
		return err
	}
	if err := writeTeacherScoreRequestManifest(*manifestPath, summary); err != nil {
		return err
	}
	fmt.Printf("exported teacher score requests: examples=%d exported=%d skipped_existing=%d rows=%d positive_rows=%d negative_rows=%d\n",
		summary.Examples, summary.ExportedExamples, summary.SkippedExisting, summary.Rows, summary.PositiveRows, summary.NegativeRows)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

func runImportTeacherScores(args []string) error {
	fs := flag.NewFlagSet("import-teacher-scores", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	overwrite := fs.Bool("overwrite", false, "replace existing teacher_scores")
	allowMissing := fs.Bool("allow-missing", false, "keep examples without complete teacher scores instead of failing")
	manifestPath := fs.String("manifest", "", "provenance manifest path; default is <output>.teacher-scores.manifest.json")
	teacherModelID := fs.String("teacher-model-id", "", "external teacher model id")
	teacherRevision := fs.String("teacher-revision", "", "external teacher model revision")
	prompt := fs.String("prompt", "", "teacher prompt or instruction provenance")
	scoreScale := fs.String("score-scale", "", "teacher score scale, such as cosine, logit, or probability")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos import-teacher-scores [flags] <hard-negatives.jsonl> <scores.jsonl> <output.jsonl>")
	}
	inputPath := fs.Arg(0)
	scoresPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".teacher-scores.manifest.json"
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	table, err := readTeacherScoreImportTable(scoresPath)
	if err != nil {
		return err
	}
	summary := teacherScoreImportSummary{
		Schema:          "manta.teacher_score_import.v1",
		CreatedUTC:      time.Now().UTC().Format(time.RFC3339),
		InputJSONL:      inputPath,
		ScoresJSONL:     scoresPath,
		OutputJSONL:     outputPath,
		TeacherModelID:  *teacherModelID,
		TeacherRevision: *teacherRevision,
		Prompt:          *prompt,
		ScoreScale:      *scoreScale,
		Examples:        len(examples),
		ExampleRows:     table.ExampleRows,
		CandidateRows:   table.CandidateRows,
	}
	for i := range examples {
		example := &examples[i]
		if len(example.TeacherScores) > 0 && !*overwrite {
			summary.SkippedExisting++
			continue
		}
		scores, ok := teacherScoresForExample(*example, table)
		if !ok {
			summary.SkippedMissing++
			if *allowMissing {
				continue
			}
			return fmt.Errorf("missing complete teacher scores for example %d source=%q query=%q", i, example.Source, example.Query)
		}
		example.TeacherScores = scores
		summary.Updated++
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	if err := writeTeacherScoreImportManifest(*manifestPath, summary); err != nil {
		return err
	}
	fmt.Printf("imported teacher scores: examples=%d updated=%d skipped_existing=%d skipped_missing=%d example_rows=%d candidate_rows=%d\n",
		summary.Examples, summary.Updated, summary.SkippedExisting, summary.SkippedMissing, summary.ExampleRows, summary.CandidateRows)
	if summary.TeacherModelID != "" || summary.TeacherRevision != "" || summary.ScoreScale != "" {
		fmt.Printf("teacher: model_id=%s revision=%s score_scale=%s\n", summary.TeacherModelID, summary.TeacherRevision, summary.ScoreScale)
	}
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

func runScoreTeacherHardNegatives(args []string) error {
	fs := flag.NewFlagSet("score-teacher-hard-negatives", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	batchSize := fs.Int("batch-size", 64, "embedding batch size for candidate texts")
	overwrite := fs.Bool("overwrite", false, "replace existing teacher_scores")
	manifestPath := fs.String("manifest", "", "provenance manifest path; default is <output>.teacher-scores.manifest.json")
	teacherModelID := fs.String("teacher-model-id", "", "teacher model id; defaults to embedding manifest name")
	teacherRevision := fs.String("teacher-revision", "", "teacher model revision")
	prompt := fs.String("prompt", "", "teacher prompt or instruction provenance")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos score-teacher-hard-negatives [flags] <teacher.mll> <hard-negatives.jsonl> <output.jsonl>")
	}
	if *batchSize <= 0 {
		return fmt.Errorf("batch-size must be positive")
	}
	artifactPath := fs.Arg(0)
	inputPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if *manifestPath == "" {
		*manifestPath = outputPath + ".teacher-scores.manifest.json"
	}
	examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
	if err != nil {
		return err
	}
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	model, err := rt.LoadEmbeddingPackage(context.Background(), artifactPath)
	if err != nil {
		return err
	}
	manifest := model.Manifest()
	if *teacherModelID == "" {
		*teacherModelID = manifest.Name
	}
	summary := teacherHardNegativeScoreSummary{
		Schema:          "manta.teacher_hard_negative_score.v1",
		CreatedUTC:      time.Now().UTC().Format(time.RFC3339),
		TeacherArtifact: artifactPath,
		InputJSONL:      inputPath,
		OutputJSONL:     outputPath,
		TeacherModelID:  *teacherModelID,
		TeacherRevision: *teacherRevision,
		Prompt:          *prompt,
		ScoreScale:      "cosine",
		Backend:         string(model.Backend()),
		BatchSize:       *batchSize,
		Examples:        len(examples),
	}
	for i := range examples {
		example := &examples[i]
		if len(example.TeacherScores) > 0 && !*overwrite {
			summary.SkippedExisting++
			continue
		}
		scores, err := scoreTeacherHardNegativeExample(context.Background(), model, *example, *batchSize)
		if err != nil {
			return fmt.Errorf("example %d source=%q query=%q: %w", i, example.Source, example.Query, err)
		}
		example.TeacherScores = scores
		summary.Updated++
	}
	if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, examples); err != nil {
		return err
	}
	if err := writeTeacherHardNegativeScoreManifest(*manifestPath, summary); err != nil {
		return err
	}
	fmt.Printf("scored teacher hard negatives: examples=%d updated=%d skipped_existing=%d backend=%s batch_size=%d\n",
		summary.Examples, summary.Updated, summary.SkippedExisting, summary.Backend, summary.BatchSize)
	fmt.Printf("teacher: model_id=%s revision=%s score_scale=%s\n", summary.TeacherModelID, summary.TeacherRevision, summary.ScoreScale)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("manifest: %s\n", *manifestPath)
	return nil
}

type teacherScoreAnyPositivePolicy struct {
	qrels        map[string]map[string]bool
	qrelsTextSet map[string]map[string]bool
	docText      map[string]string
}

func runAuditTeacherScores(args []string) error {
	fs := flag.NewFlagSet("audit-teacher-scores", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	mode := fs.String("mode", "text", "input mode: text or tokenized")
	temperature := fs.Float64("temperature", 1, "softmax temperature used for entropy diagnostics")
	qrelsPath := fs.String("qrels", "", "BEIR qrels TSV path for optional any-qrels-positive diagnostics")
	corpusPath := fs.String("corpus", "", "BEIR corpus JSONL path for optional exact-text qrels-positive diagnostics")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos audit-teacher-scores [flags] <hard-negatives.jsonl> [summary.json]")
	}
	if *temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	if (*qrelsPath == "") != (*corpusPath == "") {
		return fmt.Errorf("--qrels and --corpus must be supplied together")
	}
	inputPath := fs.Arg(0)
	summaryPath := inputPath + ".teacher-score-audit.json"
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		summaryPath = fs.Arg(1)
	}
	normalizedMode := strings.ToLower(strings.TrimSpace(*mode))
	total := teacherScoreAuditCounters{}
	sourceTotals := map[string]*teacherScoreAuditCounters{}
	add := func(source string, candidateCount int, scores []float32, anyPositiveLabels []bool) {
		total.addWithLabels(candidateCount, scores, anyPositiveLabels, *temperature)
		key := strings.TrimSpace(source)
		if key == "" {
			key = "unknown"
		}
		sourceTotal := sourceTotals[key]
		if sourceTotal == nil {
			sourceTotal = &teacherScoreAuditCounters{}
			sourceTotals[key] = sourceTotal
		}
		sourceTotal.addWithLabels(candidateCount, scores, anyPositiveLabels, *temperature)
	}
	var anyPositivePolicy *teacherScoreAnyPositivePolicy
	if *qrelsPath != "" {
		if normalizedMode == "tokenized" || normalizedMode == "tokens" {
			return fmt.Errorf("--qrels/--corpus diagnostics are unsupported in tokenized mode")
		}
		policy, err := loadTeacherScoreAnyPositivePolicy(*qrelsPath, *corpusPath)
		if err != nil {
			return err
		}
		anyPositivePolicy = policy
	}
	switch normalizedMode {
	case "text":
		examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		for _, example := range examples {
			var labels []bool
			if anyPositivePolicy != nil {
				labels = anyPositivePolicy.labelsForExample(example)
			}
			add(example.Source, 1+len(example.Negatives), example.TeacherScores, labels)
		}
	case "tokenized", "tokens":
		normalizedMode = "tokenized"
		examples, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		for _, example := range examples {
			add(example.Source, 1+len(example.NegativeTokens), example.TeacherScores, nil)
		}
	default:
		return fmt.Errorf("unsupported mode %q: want text or tokenized", *mode)
	}
	summary := teacherScoreAuditSummary{
		Schema:                 "manta.teacher_score_audit.v1",
		CreatedUTC:             time.Now().UTC().Format(time.RFC3339),
		InputJSONL:             inputPath,
		Mode:                   normalizedMode,
		Temperature:            *temperature,
		teacherScoreAuditStats: total.summary(),
	}
	if anyPositivePolicy != nil {
		summary.LabelPolicy = "selected_positive_and_any_qrels_positive"
	}
	if len(sourceTotals) > 0 {
		summary.Sources = make(map[string]teacherScoreAuditStats, len(sourceTotals))
		for source, counters := range sourceTotals {
			summary.Sources[source] = counters.summary()
		}
	}
	if err := writeTeacherScoreAuditSummary(summaryPath, summary); err != nil {
		return err
	}
	fmt.Printf("audited teacher scores: examples=%d scored=%d missing=%d positive_top1_rate=%.6f mean_margin=%.6f mean_normalized_entropy=%.6f\n",
		summary.Examples, summary.ScoredExamples, summary.MissingExamples, summary.PositiveTop1Rate, summary.PositiveMeanMargin, summary.MeanNormalizedEntropy)
	fmt.Printf("summary: %s\n", summaryPath)
	return nil
}

func loadTeacherScoreAnyPositivePolicy(qrelsPath, corpusPath string) (*teacherScoreAnyPositivePolicy, error) {
	qrels, err := readTeacherScoreAuditQrels(qrelsPath)
	if err != nil {
		return nil, fmt.Errorf("read qrels: %w", err)
	}
	docText, err := readTeacherScoreAuditCorpusText(corpusPath)
	if err != nil {
		return nil, fmt.Errorf("read corpus: %w", err)
	}
	qrelsTextSet := make(map[string]map[string]bool, len(qrels))
	for queryID, docIDs := range qrels {
		for docID := range docIDs {
			text := docText[docID]
			if text == "" {
				continue
			}
			if qrelsTextSet[queryID] == nil {
				qrelsTextSet[queryID] = map[string]bool{}
			}
			qrelsTextSet[queryID][text] = true
		}
	}
	return &teacherScoreAnyPositivePolicy{qrels: qrels, qrelsTextSet: qrelsTextSet, docText: docText}, nil
}

func (p *teacherScoreAnyPositivePolicy) labelsForExample(example eosruntime.EmbeddingTextHardNegativeExample) []bool {
	if p == nil || len(example.TeacherScores) == 0 {
		return nil
	}
	queryID, ok := teacherScoreAuditExtraString(example.ExtraFields, "query_id")
	if !ok {
		return nil
	}
	positiveDocID, ok := teacherScoreAuditExtraString(example.ExtraFields, "positive_doc_id")
	if !ok {
		return nil
	}
	negativeDocIDs, ok := teacherScoreAuditExtraStringSlice(example.ExtraFields, "negative_doc_ids")
	if !ok || len(negativeDocIDs) != len(example.Negatives) {
		return nil
	}
	candidateCount := 1 + len(example.Negatives)
	if len(example.TeacherScores) != candidateCount {
		return nil
	}
	labels := make([]bool, candidateCount)
	candidates := make([]string, 0, candidateCount)
	candidates = append(candidates, example.Positive)
	candidates = append(candidates, example.Negatives...)
	docIDs := make([]string, 0, candidateCount)
	docIDs = append(docIDs, positiveDocID)
	docIDs = append(docIDs, negativeDocIDs...)
	for i := range labels {
		labels[i] = p.isQrelsPositiveCandidate(queryID, docIDs[i], candidates[i])
	}
	return labels
}

func (p *teacherScoreAnyPositivePolicy) isQrelsPositiveCandidate(queryID, docID, text string) bool {
	if p.qrels[queryID][docID] {
		return true
	}
	normalizedText := teacherScoreAuditNormalizeText(text)
	if normalizedText == "" {
		normalizedText = p.docText[docID]
	}
	return normalizedText != "" && p.qrelsTextSet[queryID][normalizedText]
}

func teacherScoreAuditExtraString(fields map[string]json.RawMessage, key string) (string, bool) {
	raw := fields[key]
	if len(raw) == 0 {
		return "", false
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}
	value = strings.TrimSpace(value)
	return value, value != ""
}

func teacherScoreAuditExtraStringSlice(fields map[string]json.RawMessage, key string) ([]string, bool) {
	raw := fields[key]
	if len(raw) == 0 {
		return nil, false
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, false
	}
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = strings.TrimSpace(value)
	}
	return out, true
}

func readTeacherScoreAuditQrels(path string) (map[string]map[string]bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	qrels := map[string]map[string]bool{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
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
			qrels[queryID] = map[string]bool{}
		}
		qrels[queryID][docID] = true
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(qrels) == 0 {
		return nil, fmt.Errorf("qrels file has no positive relevance rows: %s", path)
	}
	return qrels, nil
}

func readTeacherScoreAuditCorpusText(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	docText := map[string]string{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record struct {
			ID    string `json:"_id"`
			Title string `json:"title,omitempty"`
			Text  string `json:"text"`
		}
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		id := strings.TrimSpace(record.ID)
		if id == "" {
			continue
		}
		docText[id] = teacherScoreAuditNormalizeText(strings.Join([]string{record.Title, record.Text}, "\n"))
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(docText) == 0 {
		return nil, fmt.Errorf("corpus file has no documents: %s", path)
	}
	return docText, nil
}

func teacherScoreAuditNormalizeText(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

func runFilterTeacherScores(args []string) error {
	fs := flag.NewFlagSet("filter-teacher-scores", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	mode := fs.String("mode", "text", "input mode: text or tokenized")
	requirePositiveTop1 := fs.Bool("require-positive-top1", true, "clear teacher_scores unless the labeled positive is teacher top-1")
	minMargin := fs.Float64("min-margin", 0, "minimum positive_score-best_negative_score margin required to keep teacher_scores")
	maxNormalizedEntropy := fs.Float64("max-normalized-entropy", 0, "maximum normalized teacher softmax entropy required to keep teacher_scores; 0 disables")
	temperature := fs.Float64("temperature", 1, "softmax temperature used for entropy diagnostics")
	dropFailingExamples := fs.Bool("drop-failing-examples", false, "drop examples with unsafe teacher_scores instead of clearing teacher_scores")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos filter-teacher-scores [flags] <scored-hard-negatives.jsonl> <output.jsonl> [summary.json]")
	}
	if *temperature <= 0 {
		return fmt.Errorf("temperature must be positive")
	}
	if *maxNormalizedEntropy < 0 {
		return fmt.Errorf("max-normalized-entropy must be non-negative")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	summaryPath := outputPath + ".teacher-score-filter.json"
	if fs.NArg() > 2 && fs.Arg(2) != "" {
		summaryPath = fs.Arg(2)
	}
	normalizedMode := strings.ToLower(strings.TrimSpace(*mode))
	cfg := teacherScoreFilterConfig{
		RequirePositiveTop1:  *requirePositiveTop1,
		MinMargin:            *minMargin,
		MaxNormalizedEntropy: *maxNormalizedEntropy,
		Temperature:          *temperature,
		DropFailingExamples:  *dropFailingExamples,
	}
	summary := teacherScoreFilterSummary{
		Schema:      "manta.teacher_score_filter.v1",
		CreatedUTC:  time.Now().UTC().Format(time.RFC3339),
		InputJSONL:  inputPath,
		OutputJSONL: outputPath,
		Mode:        normalizedMode,
		Config:      cfg,
	}
	total := &teacherScoreFilterCounters{}
	sourceTotals := map[string]*teacherScoreFilterCounters{}
	add := func(source string, candidateCount int, scores []float32) (bool, bool) {
		key := strings.TrimSpace(source)
		if key == "" {
			key = "unknown"
		}
		sourceTotal := sourceTotals[key]
		if sourceTotal == nil {
			sourceTotal = &teacherScoreFilterCounters{}
			sourceTotals[key] = sourceTotal
		}
		keepScores, keepExample := evaluateTeacherScoreFilter(total, candidateCount, scores, cfg)
		evaluateTeacherScoreFilter(sourceTotal, candidateCount, scores, cfg)
		return keepScores, keepExample
	}
	switch normalizedMode {
	case "text":
		examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		out := make([]eosruntime.EmbeddingTextHardNegativeExample, 0, len(examples))
		for _, example := range examples {
			keepScores, keepExample := add(example.Source, 1+len(example.Negatives), example.TeacherScores)
			if !keepExample {
				continue
			}
			if !keepScores {
				example.TeacherScores = nil
			}
			out = append(out, example)
		}
		if err := eosruntime.WriteEmbeddingTextHardNegativeExamplesFile(outputPath, out); err != nil {
			return err
		}
	case "tokenized", "tokens":
		normalizedMode = "tokenized"
		summary.Mode = normalizedMode
		examples, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(inputPath)
		if err != nil {
			return err
		}
		out := make([]eosruntime.EmbeddingHardNegativeExample, 0, len(examples))
		for _, example := range examples {
			keepScores, keepExample := add(example.Source, 1+len(example.NegativeTokens), example.TeacherScores)
			if !keepExample {
				continue
			}
			if !keepScores {
				example.TeacherScores = nil
			}
			out = append(out, example)
		}
		if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(outputPath, out); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported mode %q: want text or tokenized", *mode)
	}
	summary.applyCounters(*total)
	if len(sourceTotals) > 0 {
		summary.Sources = make(map[string]teacherScoreFilterStats, len(sourceTotals))
		for source, counters := range sourceTotals {
			summary.Sources[source] = counters.summary()
		}
	}
	if err := writeTeacherScoreFilterSummary(summaryPath, summary); err != nil {
		return err
	}
	fmt.Printf("filtered teacher scores: examples=%d scored=%d missing=%d kept=%d cleared=%d dropped=%d positive_top1_rate_before=%.6f kept_rate=%.6f mean_margin_before=%.6f mean_margin_after=%.6f\n",
		summary.Examples, summary.Scored, summary.Missing, summary.KeptTeacherScores, summary.ClearedTeacherScores, summary.DroppedExamples,
		summary.PositiveTop1RateBefore, summary.TeacherScoreKeptRate, summary.MeanMarginBefore, summary.MeanMarginAfter)
	fmt.Printf("output: %s\n", outputPath)
	fmt.Printf("summary: %s\n", summaryPath)
	return nil
}

func runPlanSparseAttention(args []string) error {
	fs := flag.NewFlagSet("plan-sparse-attention", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	keyLensCSV := fs.String("key-lens", "4096,8192,16384,32768,65536", "comma-separated key/context lengths to sweep")
	queryLen := fs.Int("query-len", 1, "query rows per batch item")
	queryDim := fs.Int("query-dim", 128, "query/key head dimension")
	valueDim := fs.Int("value-dim", 128, "value head dimension")
	topK := fs.Int("top-k", 64, "selected sparse keys per query; 0 uses ceil(sqrt(key_len))")
	routeBlockSize := fs.Int("route-block-size", 0, "route block size; 0 uses ceil(sqrt(key_len)) when routing is enabled")
	routeTopBlocks := fs.Int("route-top-blocks", 2, "route blocks selected per query; 0 disables routing")
	exact := fs.Bool("exact", false, "disable routing and report exact sparse top-k scoring")
	bits := fs.Int("bits", 4, "TurboQuant K/V bits: 2, 4, or 8")
	batches := fs.Int("batches", 1, "batch items for total score and KV memory estimates")
	denseBytesPerValue := fs.Int("dense-bytes-per-value", 2, "dense K/V bytes per value, usually 2 for f16")
	jsonPath := fs.String("json", "", "write machine-readable plan JSON")
	maxScoreFraction := fs.Float64("max-score-fraction", 1, "fail if any row exceeds this estimated score-work fraction versus dense")
	maxTurboKVMiB := fs.Float64("max-turbo-kv-mib", 0, "fail if any row exceeds this TurboQuant K/V MiB budget; 0 disables")
	requireSubq := fs.Bool("require-subquadratic", false, "fail if any row does not reduce score work versus dense scoring")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos plan-sparse-attention [flags]")
	}
	keyLens, err := parsePositiveIntCSV(*keyLensCSV, "key-lens")
	if err != nil {
		return err
	}
	if *queryLen <= 0 {
		return fmt.Errorf("query-len must be positive")
	}
	if *queryDim <= 0 {
		return fmt.Errorf("query-dim must be positive")
	}
	if *valueDim <= 0 {
		return fmt.Errorf("value-dim must be positive")
	}
	if *topK < 0 {
		return fmt.Errorf("top-k must be non-negative")
	}
	if *routeBlockSize < 0 {
		return fmt.Errorf("route-block-size must be non-negative")
	}
	if *routeTopBlocks < 0 {
		return fmt.Errorf("route-top-blocks must be non-negative")
	}
	if *bits != 2 && *bits != 4 && *bits != 8 {
		return fmt.Errorf("bits must be 2, 4, or 8")
	}
	if *batches <= 0 {
		return fmt.Errorf("batches must be positive")
	}
	if *denseBytesPerValue <= 0 {
		return fmt.Errorf("dense-bytes-per-value must be positive")
	}
	if *maxScoreFraction <= 0 {
		return fmt.Errorf("max-score-fraction must be positive")
	}
	if *maxTurboKVMiB < 0 {
		return fmt.Errorf("max-turbo-kv-mib must be non-negative")
	}
	report := sparseAttentionPlanReport{
		Schema:     "manta.sparse_attention_plan.v1",
		CreatedUTC: time.Now().UTC().Format(time.RFC3339),
		Config: sparseAttentionPlanConfig{
			KeyLens:          keyLens,
			QueryLen:         *queryLen,
			QueryDim:         *queryDim,
			ValueDim:         *valueDim,
			TopK:             *topK,
			RouteBlockSize:   *routeBlockSize,
			RouteTopBlocks:   *routeTopBlocks,
			Exact:            *exact,
			Bits:             *bits,
			Batches:          *batches,
			MaxScoreFraction: *maxScoreFraction,
			MaxTurboKVMiB:    *maxTurboKVMiB,
			RequireSubq:      *requireSubq,
		},
	}
	report.Rows = make([]sparseAttentionPlanRow, 0, len(keyLens))
	for _, keyLen := range keyLens {
		blockSize := *routeBlockSize
		topBlocks := *routeTopBlocks
		if *exact {
			blockSize = 0
			topBlocks = 0
		} else if topBlocks > 0 && blockSize == 0 {
			blockSize = int(math.Ceil(math.Sqrt(float64(keyLen))))
		}
		plan := backend.PlanSparseAttention(backend.SparseAttentionPlanInput{
			QueryLen:       *queryLen,
			KeyLen:         keyLen,
			QueryDim:       *queryDim,
			ValueDim:       *valueDim,
			TopK:           *topK,
			RouteBlockSize: blockSize,
			RouteTopBlocks: topBlocks,
		})
		kv := backend.PlanTurboQuantKVMemory(backend.TurboQuantKVMemoryPlanInput{
			Batches:            *batches,
			KeyLen:             keyLen,
			KeyDim:             *queryDim,
			ValueDim:           *valueDim,
			Bits:               *bits,
			DenseBytesPerValue: *denseBytesPerValue,
		})
		row := sparseAttentionPlanRow{
			KeyLen:                      keyLen,
			QueryLen:                    plan.QueryLen,
			QueryDim:                    plan.QueryDim,
			ValueDim:                    plan.ValueDim,
			TopK:                        plan.TopK,
			Routing:                     plan.Routing,
			RouteBlockSize:              plan.RouteBlockSize,
			RouteTopBlocks:              plan.RouteTopBlocks,
			RouteBlockCount:             plan.RouteBlockCount,
			SelectedRouteBlocks:         plan.SelectedRouteBlocks,
			SelectedKeyCount:            plan.SelectedKeyCount,
			CandidateKeyBudget:          plan.CandidateKeyBudget,
			DenseScoreCountPerQuery:     plan.DenseScoreCountPerQuery,
			EstimatedScoreCountPerQuery: plan.EstimatedScoreCountPerQuery,
			ScoreCountFraction:          plan.ScoreCountFraction,
			CandidateKeyFraction:        plan.CandidateKeyFraction,
			SubquadraticScorePlan:       plan.SubquadraticScorePlan,
			DenseTotalScoreCount:        int64(*batches) * int64(plan.QueryLen) * int64(plan.DenseScoreCountPerQuery),
			EstimatedTotalScoreCount:    int64(*batches) * int64(plan.QueryLen) * int64(plan.EstimatedScoreCountPerQuery),
			Bits:                        kv.Bits,
			DenseKVBytes:                kv.DenseKVBytes,
			TurboQuantKVBytes:           kv.TurboQuantKVBytes,
			TurboQuantKVMiB:             float64(kv.TurboQuantKVBytes) / (1024 * 1024),
			TurboQuantCompressionRatio:  kv.CompressionRatio,
		}
		report.Rows = append(report.Rows, row)
	}
	report.Gate = evaluateSparseAttentionPlanGate(report.Rows, *maxScoreFraction, *maxTurboKVMiB, *requireSubq)
	printSparseAttentionPlanTSV(report)
	if *jsonPath != "" {
		if err := writeSparseAttentionPlanReport(*jsonPath, report); err != nil {
			return err
		}
		fmt.Printf("json: %s\n", *jsonPath)
	}
	fmt.Printf("summary: rows=%d subq_rows=%d score_alpha=%.3f max_score_fraction=%.6f max_turbo_kv_mib=%.3f gate=%s\n",
		report.Gate.Rows, report.Gate.SubquadraticRows, report.Gate.ScoreAlpha, report.Gate.MaxScoreFractionObserved, report.Gate.MaxTurboKVMiBObserved, passFail(report.Gate.Passed))
	if !report.Gate.Passed {
		return fmt.Errorf("sparse attention plan gate failed: %s", strings.Join(report.Gate.FailureReasons, "; "))
	}
	return nil
}

func runPlanMultiVectorStorage(args []string) error {
	fs := flag.NewFlagSet("plan-multivector-storage", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	dim := fs.Int("dim", 128, "embedding dimension")
	baselineDim := fs.Int("baseline-dim", 0, "dense fp32 baseline dimension for one-vector budget; 0 uses --dim")
	bitsRaw := fs.String("bits", "2,4,8", "comma-separated TurboQuant IP bit widths; supported: 2..8")
	objects := fs.Int("objects", 1, "parent object count")
	vectorsPerObjectRaw := fs.String("vectors-per-object", "1,16,64,128", "comma-separated quantized child vectors per parent object")
	seriesLengthsRaw := fs.String("series-lengths", "", "comma-separated time-series point counts; derives vectors-per-object from covering windows")
	windowSize := fs.Int("window-size", 0, "time-series window size in points; required with --series-lengths")
	windowStride := fs.Int("window-stride", 0, "time-series window stride in points; 0 uses --window-size")
	sidecarStorage := fs.String("sidecar-storage", eosruntime.MultiVectorSidecarNone, "optional per-child sidecar storage: none, fp16, or dense")
	vectorOverheadBytes := fs.Int64("vector-overhead-bytes", 0, "per stored vector/index-entry overhead bytes")
	packedObjectOverheadBytes := fs.Int64("packed-object-overhead-bytes", 0, "per parent packed-object overhead bytes for packed layout planning")
	jsonPath := fs.String("json", "", "write machine-readable plan JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("usage: eos plan-multivector-storage [flags]")
	}
	if *vectorOverheadBytes < 0 {
		return fmt.Errorf("vector-overhead-bytes must be non-negative")
	}
	if *packedObjectOverheadBytes < 0 {
		return fmt.Errorf("packed-object-overhead-bytes must be non-negative")
	}
	vectorsPerObjectExplicit := false
	fs.Visit(func(flag *flag.Flag) {
		if flag.Name == "vectors-per-object" {
			vectorsPerObjectExplicit = true
		}
	})
	bits, err := parsePositiveIntCSV(*bitsRaw, "bits")
	if err != nil {
		return fmt.Errorf("bits: %w", err)
	}
	seriesLengths, err := parseOptionalPositiveIntCSV(*seriesLengthsRaw, "series-lengths")
	if err != nil {
		return err
	}
	var vectorsPerObject []int
	if len(seriesLengths) > 0 {
		if vectorsPerObjectExplicit {
			return fmt.Errorf("use either --series-lengths with --window-size/--window-stride or explicit --vectors-per-object, not both")
		}
	} else {
		vectorsPerObject, err = parsePositiveIntCSV(*vectorsPerObjectRaw, "vectors-per-object")
		if err != nil {
			return err
		}
	}
	plan, err := eosruntime.PlanMultiVectorStorage(eosruntime.MultiVectorStoragePlanInput{
		Dim:                       *dim,
		BaselineDim:               *baselineDim,
		Bits:                      bits,
		Objects:                   *objects,
		VectorsPerObject:          vectorsPerObject,
		SeriesLengths:             seriesLengths,
		WindowSize:                *windowSize,
		WindowStride:              *windowStride,
		SidecarStorage:            *sidecarStorage,
		VectorOverheadBytes:       *vectorOverheadBytes,
		PackedObjectOverheadBytes: *packedObjectOverheadBytes,
	})
	if err != nil {
		return err
	}
	printMultiVectorStoragePlanTSV(plan)
	if *jsonPath != "" {
		data, err := json.MarshalIndent(plan, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		if err := os.WriteFile(*jsonPath, data, 0o644); err != nil {
			return err
		}
		fmt.Printf("json: %s\n", *jsonPath)
	}
	fmt.Printf("summary: rows=%d dim=%d baseline_dim=%d objects=%d sidecar_storage=%s vector_overhead_bytes=%d packed_object_overhead_bytes=%d\n",
		len(plan.Rows), plan.Config.Dim, plan.Config.BaselineDim, plan.Config.Objects, plan.Config.SidecarStorage, plan.Config.VectorOverheadBytes, plan.Config.PackedObjectOverheadBytes)
	return nil
}

func printMultiVectorStoragePlanTSV(plan eosruntime.MultiVectorStoragePlan) {
	fmt.Println("dim\tbaseline_dim\tbits\tobjects\tvectors_per_object\tdense_parent_bytes\tdense_parent_total_bytes\tdense_baseline_bytes\tdense_baseline_total_bytes\tquantized_payload_bytes\tsidecar_storage\tsidecar_bytes_per_vector\tquantized_vector_bytes\tvector_overhead_bytes\tdense_vector_storage_bytes\tquantized_vector_storage_bytes\ttotal_quantized_bytes\tpacked_object_overhead_bytes\tpacked_quantized_storage_bytes\tpacked_total_quantized_bytes\tdense_to_quantized_vector_ratio\ttotal_compression_ratio\tpacked_total_compression_ratio\tvectors_that_fit_in_one_dense_vector\tfits_in_one_dense_vector_storage\tstorage_multiple_of_dense_parent_cost\tpacked_vectors_that_fit_in_one_dense_vector\tpacked_fits_in_one_dense_vector_storage\tpacked_storage_multiple_of_dense_parent_cost\tseries_length\twindow_size\twindow_stride\tderived_window_count")
	for _, row := range plan.Rows {
		fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%d\t%t\t%.6f\t%d\t%t\t%.6f\t%d\t%d\t%d\t%d\n",
			row.Dim,
			row.BaselineDim,
			row.Bits,
			row.Objects,
			row.VectorsPerObject,
			row.DenseParentBytes,
			row.DenseParentTotalBytes,
			row.DenseBaselineBytes,
			row.DenseBaselineTotalBytes,
			row.QuantizedPayloadBytes,
			row.SidecarStorage,
			row.SidecarBytesPerVector,
			row.QuantizedVectorBytes,
			row.VectorOverheadBytes,
			row.DenseVectorStorageBytes,
			row.QuantizedVectorStorageBytes,
			row.TotalQuantizedBytes,
			row.PackedObjectOverheadBytes,
			row.PackedQuantizedStorageBytes,
			row.PackedTotalQuantizedBytes,
			row.DenseToQuantizedVectorRatio,
			row.TotalCompressionRatio,
			row.PackedTotalCompressionRatio,
			row.VectorsThatFitInOneDenseVector,
			row.FitsInOneDenseVectorStorage,
			row.StorageMultipleOfDenseParentCost,
			row.PackedVectorsThatFitInOneDenseVector,
			row.PackedFitsInOneDenseVectorStorage,
			row.PackedStorageMultipleOfDenseParentCost,
			row.SeriesLength,
			row.WindowSize,
			row.WindowStride,
			row.DerivedWindowCount,
		)
	}
}

func parsePositiveIntCSV(raw, label string) ([]int, error) {
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for i, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("%s[%d] %q is not an integer", label, i, part)
		}
		if value <= 0 {
			return nil, fmt.Errorf("%s[%d] must be positive", label, i)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s must contain at least one positive integer", label)
	}
	return out, nil
}

func parseOptionalPositiveIntCSV(raw, label string) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	return parsePositiveIntCSV(raw, label)
}

func evaluateSparseAttentionPlanGate(rows []sparseAttentionPlanRow, maxScoreFraction, maxTurboKVMiB float64, requireSubq bool) sparseAttentionPlanGate {
	gate := sparseAttentionPlanGate{
		Passed:     true,
		Rows:       len(rows),
		ScoreAlpha: fitSparseAttentionPlanAlpha(rows),
	}
	for _, row := range rows {
		if row.SubquadraticScorePlan {
			gate.SubquadraticRows++
		} else if requireSubq {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d is not subquadratic", row.KeyLen))
		}
		if row.ScoreCountFraction > gate.MaxScoreFractionObserved {
			gate.MaxScoreFractionObserved = row.ScoreCountFraction
		}
		if row.TurboQuantKVMiB > gate.MaxTurboKVMiBObserved {
			gate.MaxTurboKVMiBObserved = row.TurboQuantKVMiB
		}
		if maxScoreFraction > 0 && row.ScoreCountFraction > maxScoreFraction {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d score_fraction %.6f exceeds %.6f", row.KeyLen, row.ScoreCountFraction, maxScoreFraction))
		}
		if maxTurboKVMiB > 0 && row.TurboQuantKVMiB > maxTurboKVMiB {
			gate.Passed = false
			gate.FailureReasons = append(gate.FailureReasons, fmt.Sprintf("key_len=%d turbo_kv_mib %.3f exceeds %.3f", row.KeyLen, row.TurboQuantKVMiB, maxTurboKVMiB))
		}
	}
	return gate
}

func fitSparseAttentionPlanAlpha(rows []sparseAttentionPlanRow) float64 {
	if len(rows) < 2 {
		return 0
	}
	var sumX, sumY, sumXX, sumXY float64
	n := 0
	for _, row := range rows {
		if row.KeyLen <= 0 || row.EstimatedScoreCountPerQuery <= 0 {
			continue
		}
		x := math.Log(float64(row.KeyLen))
		y := math.Log(float64(row.EstimatedScoreCountPerQuery))
		sumX += x
		sumY += y
		sumXX += x * x
		sumXY += x * y
		n++
	}
	if n < 2 {
		return 0
	}
	denom := float64(n)*sumXX - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (float64(n)*sumXY - sumX*sumY) / denom
}

func printSparseAttentionPlanTSV(report sparseAttentionPlanReport) {
	fmt.Println("key_len\trouting\troute_block_size\troute_top_blocks\ttop_k\tcandidate_key_budget\testimated_scores_per_query\tscore_fraction\tturbo_kv_mib\tcompression_ratio\tsubquadratic")
	for _, row := range report.Rows {
		fmt.Printf("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.3f\t%.3f\t%t\n",
			row.KeyLen,
			row.Routing,
			row.RouteBlockSize,
			row.RouteTopBlocks,
			row.TopK,
			row.CandidateKeyBudget,
			row.EstimatedScoreCountPerQuery,
			row.ScoreCountFraction,
			row.TurboQuantKVMiB,
			row.TurboQuantCompressionRatio,
			row.SubquadraticScorePlan,
		)
	}
}

func writeSparseAttentionPlanReport(path string, report sparseAttentionPlanReport) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func passFail(pass bool) string {
	if pass {
		return "pass"
	}
	return "fail"
}

func scoreTeacherHardNegativeExample(ctx context.Context, model *eosruntime.EmbeddingModel, example eosruntime.EmbeddingTextHardNegativeExample, batchSize int) ([]float32, error) {
	queryVector, err := embedTeacherText(ctx, model, example.Query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	candidates := append([]string{example.Positive}, example.Negatives...)
	scores := make([]float32, 0, len(candidates))
	for start := 0; start < len(candidates); start += batchSize {
		end := min(start+batchSize, len(candidates))
		result, err := model.EmbedTextBatch(ctx, candidates[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed candidates %d-%d: %w", start, end, err)
		}
		rows, err := teacherEmbeddingRows(result.Embeddings, end-start)
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			scores = append(scores, dotTeacherVectors(queryVector, normalizeTeacherVector(row)))
		}
	}
	return scores, nil
}

func embedTeacherText(ctx context.Context, model *eosruntime.EmbeddingModel, text string) ([]float32, error) {
	result, err := model.EmbedText(ctx, text)
	if err != nil {
		return nil, err
	}
	rows, err := teacherEmbeddingRows(result.Embeddings, 1)
	if err != nil {
		return nil, err
	}
	return normalizeTeacherVector(rows[0]), nil
}

func teacherEmbeddingRows(t *backend.Tensor, wantRows int) ([][]float32, error) {
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

func normalizeTeacherVector(in []float32) []float32 {
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

func dotTeacherVectors(a, b []float32) float32 {
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

func (c *teacherScoreAuditCounters) add(candidateCount int, scores []float32, temperature float64) {
	c.addWithLabels(candidateCount, scores, nil, temperature)
}

func (c *teacherScoreAuditCounters) addWithLabels(candidateCount int, scores []float32, anyPositiveLabels []bool, temperature float64) {
	c.Examples++
	if candidateCount < 0 {
		candidateCount = 0
	}
	c.Candidates += candidateCount
	if len(scores) == 0 {
		c.MissingExamples++
		return
	}
	c.ScoredExamples++
	c.ScoredCandidates += len(scores)
	positive := float64(scores[0])
	rank := 1
	minScore := positive
	maxScore := positive
	bestNegative := math.Inf(-1)
	for i, raw := range scores {
		score := float64(raw)
		c.ScoreSum += score
		if score < minScore {
			minScore = score
		}
		if score > maxScore {
			maxScore = score
		}
		if i > 0 {
			if score > positive {
				rank++
			}
			if score > bestNegative {
				bestNegative = score
			}
		}
	}
	if math.IsInf(bestNegative, -1) {
		bestNegative = positive
	}
	if rank == 1 {
		c.PositiveTop1++
	}
	entropy, normalizedEntropy := teacherScoreEntropy(scores, temperature)
	c.PositiveRankSum += float64(rank)
	c.PositiveMarginSum += positive - bestNegative
	c.ScoreRangeSum += maxScore - minScore
	c.EntropySum += entropy
	c.NormalizedEntropySum += normalizedEntropy
	if len(anyPositiveLabels) == len(scores) {
		c.addAnyPositiveScores(scores, anyPositiveLabels)
	}
}

func (c *teacherScoreAuditCounters) addAnyPositiveScores(scores []float32, labels []bool) {
	bestPositive := math.Inf(-1)
	bestNonPositive := math.Inf(-1)
	hasPositive := false
	duplicatePositiveNegatives := 0
	for i, raw := range scores {
		score := float64(raw)
		if labels[i] {
			hasPositive = true
			if i > 0 {
				duplicatePositiveNegatives++
			}
			if score > bestPositive {
				bestPositive = score
			}
			continue
		}
		if score > bestNonPositive {
			bestNonPositive = score
		}
	}
	if !hasPositive {
		return
	}
	rank := 1
	for i, raw := range scores {
		if labels[i] {
			continue
		}
		if float64(raw) > bestPositive {
			rank++
		}
	}
	if math.IsInf(bestNonPositive, -1) {
		bestNonPositive = bestPositive
	}
	c.AnyPositiveExamples++
	if rank == 1 {
		c.AnyPositiveTop1++
	}
	c.AnyPositiveRankSum += float64(rank)
	c.AnyPositiveMarginSum += bestPositive - bestNonPositive
	c.DuplicatePositiveNegativeCandidates += duplicatePositiveNegatives
	if duplicatePositiveNegatives > 0 {
		c.ExamplesWithDuplicatePositiveNegatives++
	}
}

func (c teacherScoreAuditCounters) summary() teacherScoreAuditStats {
	summary := teacherScoreAuditStats{
		Examples:         c.Examples,
		ScoredExamples:   c.ScoredExamples,
		MissingExamples:  c.MissingExamples,
		Candidates:       c.Candidates,
		ScoredCandidates: c.ScoredCandidates,
		PositiveTop1:     c.PositiveTop1,
	}
	if c.ScoredExamples > 0 {
		inv := 1 / float64(c.ScoredExamples)
		summary.PositiveTop1Rate = float64(c.PositiveTop1) * inv
		summary.PositiveMeanRank = c.PositiveRankSum * inv
		summary.PositiveMeanMargin = c.PositiveMarginSum * inv
		summary.MeanScoreRange = c.ScoreRangeSum * inv
		summary.MeanEntropy = c.EntropySum * inv
		summary.MeanNormalizedEntropy = c.NormalizedEntropySum * inv
	}
	if c.ScoredCandidates > 0 {
		summary.MeanScore = c.ScoreSum / float64(c.ScoredCandidates)
	}
	if c.AnyPositiveExamples > 0 {
		anyExamples := c.AnyPositiveExamples
		anyTop1 := c.AnyPositiveTop1
		duplicateCandidates := c.DuplicatePositiveNegativeCandidates
		examplesWithDuplicates := c.ExamplesWithDuplicatePositiveNegatives
		anyInv := 1 / float64(c.AnyPositiveExamples)
		anyRate := float64(c.AnyPositiveTop1) * anyInv
		anyRank := c.AnyPositiveRankSum * anyInv
		anyMargin := c.AnyPositiveMarginSum * anyInv
		summary.AnyPositiveExamples = &anyExamples
		summary.AnyPositiveTop1 = &anyTop1
		summary.AnyPositiveTop1Rate = &anyRate
		summary.AnyPositiveMeanRank = &anyRank
		summary.AnyPositiveMeanMargin = &anyMargin
		summary.DuplicatePositiveNegativeCandidates = &duplicateCandidates
		summary.ExamplesWithDuplicatePositiveNegatives = &examplesWithDuplicates
	}
	return summary
}

func evaluateTeacherScoreFilter(c *teacherScoreFilterCounters, candidateCount int, scores []float32, cfg teacherScoreFilterConfig) (keepScores bool, keepExample bool) {
	c.Examples++
	if candidateCount < 0 {
		candidateCount = 0
	}
	c.Candidates += candidateCount
	if len(scores) == 0 {
		c.Missing++
		return true, true
	}
	c.Scored++
	metrics := teacherScoreFilterMetricsForScores(scores, cfg.Temperature)
	if metrics.PositiveTop1 {
		c.PositiveTop1Before++
	}
	c.MarginBeforeSum += metrics.Margin
	pass := true
	if cfg.RequirePositiveTop1 && !metrics.PositiveTop1 {
		pass = false
	}
	if metrics.Margin < cfg.MinMargin {
		pass = false
	}
	if cfg.MaxNormalizedEntropy > 0 && metrics.NormalizedEntropy > cfg.MaxNormalizedEntropy {
		pass = false
	}
	if pass {
		c.KeptTeacherScores++
		c.PositiveTop1After++
		c.MarginAfterSum += metrics.Margin
		return true, true
	}
	if cfg.DropFailingExamples {
		c.DroppedExamples++
		return false, false
	}
	c.ClearedTeacherScores++
	return false, true
}

type teacherScoreFilterMetrics struct {
	PositiveTop1      bool
	Margin            float64
	NormalizedEntropy float64
}

func teacherScoreFilterMetricsForScores(scores []float32, temperature float64) teacherScoreFilterMetrics {
	if len(scores) == 0 {
		return teacherScoreFilterMetrics{}
	}
	positive := float64(scores[0])
	bestNegative := math.Inf(-1)
	positiveTop1 := true
	for _, raw := range scores[1:] {
		score := float64(raw)
		if score > positive {
			positiveTop1 = false
		}
		if score > bestNegative {
			bestNegative = score
		}
	}
	if math.IsInf(bestNegative, -1) {
		bestNegative = positive
	}
	_, normalizedEntropy := teacherScoreEntropy(scores, temperature)
	return teacherScoreFilterMetrics{
		PositiveTop1:      positiveTop1,
		Margin:            positive - bestNegative,
		NormalizedEntropy: normalizedEntropy,
	}
}

func (c teacherScoreFilterCounters) summary() teacherScoreFilterStats {
	summary := teacherScoreFilterStats{
		Examples:             c.Examples,
		Scored:               c.Scored,
		Missing:              c.Missing,
		KeptTeacherScores:    c.KeptTeacherScores,
		ClearedTeacherScores: c.ClearedTeacherScores,
		DroppedExamples:      c.DroppedExamples,
		Candidates:           c.Candidates,
	}
	if c.Scored > 0 {
		inv := 1 / float64(c.Scored)
		summary.PositiveTop1RateBefore = float64(c.PositiveTop1Before) * inv
		summary.TeacherScoreKeptRate = float64(c.KeptTeacherScores) * inv
		summary.MeanMarginBefore = c.MarginBeforeSum * inv
	}
	if c.KeptTeacherScores > 0 {
		inv := 1 / float64(c.KeptTeacherScores)
		summary.PositiveTop1RateAfter = float64(c.PositiveTop1After) * inv
		summary.MeanMarginAfter = c.MarginAfterSum * inv
	}
	return summary
}

func (s *teacherScoreFilterSummary) applyCounters(c teacherScoreFilterCounters) {
	stats := c.summary()
	s.Examples = stats.Examples
	s.Scored = stats.Scored
	s.Missing = stats.Missing
	s.KeptTeacherScores = stats.KeptTeacherScores
	s.ClearedTeacherScores = stats.ClearedTeacherScores
	s.DroppedExamples = stats.DroppedExamples
	s.Candidates = stats.Candidates
	s.PositiveTop1RateBefore = stats.PositiveTop1RateBefore
	s.PositiveTop1RateAfter = stats.PositiveTop1RateAfter
	s.TeacherScoreKeptRate = stats.TeacherScoreKeptRate
	s.MeanMarginBefore = stats.MeanMarginBefore
	s.MeanMarginAfter = stats.MeanMarginAfter
}

func teacherScoreEntropy(scores []float32, temperature float64) (float64, float64) {
	if len(scores) == 0 {
		return 0, 0
	}
	if temperature <= 0 {
		temperature = 1
	}
	maxScore := float64(scores[0])
	for _, raw := range scores[1:] {
		score := float64(raw)
		if score > maxScore {
			maxScore = score
		}
	}
	probs := make([]float64, len(scores))
	denom := 0.0
	for i, raw := range scores {
		v := math.Exp((float64(raw) - maxScore) / temperature)
		probs[i] = v
		denom += v
	}
	if denom == 0 {
		return 0, 0
	}
	entropy := 0.0
	for _, p := range probs {
		p /= denom
		if p > 0 {
			entropy -= p * math.Log(p)
		}
	}
	normalized := 0.0
	if len(scores) > 1 {
		normalized = entropy / math.Log(float64(len(scores)))
	}
	return entropy, normalized
}

func readTeacherScoreImportTable(path string) (teacherScoreImportTable, error) {
	f, err := os.Open(path)
	if err != nil {
		return teacherScoreImportTable{}, err
	}
	defer f.Close()
	table := teacherScoreImportTable{
		ExampleScores:   map[string][]float32{},
		CandidateScores: map[string]float32{},
	}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var record teacherScoreImportRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if scores, ok, err := record.teacherScoreVector(); err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		} else if ok {
			if strings.TrimSpace(record.Query) == "" {
				return teacherScoreImportTable{}, fmt.Errorf("%s:%d: query is required for score vectors", path, lineNo)
			}
			key := teacherScoreExampleKey(record.Source, record.Query)
			if _, exists := table.ExampleScores[key]; exists {
				return teacherScoreImportTable{}, fmt.Errorf("%s:%d: duplicate score vector for source=%q query=%q", path, lineNo, record.Source, record.Query)
			}
			table.ExampleScores[key] = scores
			table.ExampleRows++
			continue
		}
		if record.Score == nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: expected score, scores, teacher_scores, or positive_score/negative_scores", path, lineNo)
		}
		score, err := finiteFloat32(*record.Score, "score")
		if err != nil {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		candidate := firstNonEmptyString(record.Candidate, record.Document, record.Text, record.Positive)
		if strings.TrimSpace(record.Query) == "" || strings.TrimSpace(candidate) == "" {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: query and candidate/document/text are required for candidate score rows", path, lineNo)
		}
		key := teacherScoreCandidateKey(record.Source, record.Query, candidate)
		if _, exists := table.CandidateScores[key]; exists {
			return teacherScoreImportTable{}, fmt.Errorf("%s:%d: duplicate candidate score for source=%q query=%q candidate=%q", path, lineNo, record.Source, record.Query, candidate)
		}
		table.CandidateScores[key] = score
		table.CandidateRows++
	}
	if err := scanner.Err(); err != nil {
		return teacherScoreImportTable{}, err
	}
	if table.ExampleRows == 0 && table.CandidateRows == 0 {
		return teacherScoreImportTable{}, fmt.Errorf("teacher score file is empty: %s", path)
	}
	return table, nil
}

func (r teacherScoreImportRecord) teacherScoreVector() ([]float32, bool, error) {
	values := r.TeacherScores
	if len(values) == 0 {
		values = r.Scores
	}
	if len(values) == 0 && r.PositiveScore != nil {
		values = append([]float64{*r.PositiveScore}, r.NegativeScores...)
	}
	if len(values) == 0 {
		return nil, false, nil
	}
	out := make([]float32, len(values))
	for i, value := range values {
		score, err := finiteFloat32(value, fmt.Sprintf("score[%d]", i))
		if err != nil {
			return nil, true, err
		}
		out[i] = score
	}
	return out, true, nil
}

func teacherScoresForExample(example eosruntime.EmbeddingTextHardNegativeExample, table teacherScoreImportTable) ([]float32, bool) {
	if scores, ok := lookupTeacherScoreVector(table, example.Source, example.Query); ok {
		if len(scores) != 1+len(example.Negatives) {
			return nil, false
		}
		return append([]float32(nil), scores...), true
	}
	candidates := append([]string{example.Positive}, example.Negatives...)
	out := make([]float32, len(candidates))
	for i, candidate := range candidates {
		score, ok := lookupTeacherCandidateScore(table, example.Source, example.Query, candidate)
		if !ok {
			return nil, false
		}
		out[i] = score
	}
	return out, true
}

func lookupTeacherScoreVector(table teacherScoreImportTable, source, query string) ([]float32, bool) {
	if scores, ok := table.ExampleScores[teacherScoreExampleKey(source, query)]; ok {
		return scores, true
	}
	if strings.TrimSpace(source) != "" {
		if scores, ok := table.ExampleScores[teacherScoreExampleKey("", query)]; ok {
			return scores, true
		}
	}
	return nil, false
}

func lookupTeacherCandidateScore(table teacherScoreImportTable, source, query, candidate string) (float32, bool) {
	if score, ok := table.CandidateScores[teacherScoreCandidateKey(source, query, candidate)]; ok {
		return score, true
	}
	if strings.TrimSpace(source) != "" {
		if score, ok := table.CandidateScores[teacherScoreCandidateKey("", query, candidate)]; ok {
			return score, true
		}
	}
	return 0, false
}

func teacherScoreExampleKey(source, query string) string {
	return strings.TrimSpace(source) + "\x00" + strings.TrimSpace(query)
}

func teacherScoreCandidateKey(source, query, candidate string) string {
	return teacherScoreExampleKey(source, query) + "\x00" + strings.TrimSpace(candidate)
}

func finiteFloat32(value float64, label string) (float32, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("%s must be finite", label)
	}
	return float32(value), nil
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func writeTeacherScoreRequestManifest(path string, summary teacherScoreRequestSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherScoreImportManifest(path string, summary teacherScoreImportSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherHardNegativeScoreManifest(path string, summary teacherHardNegativeScoreSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherScoreAuditSummary(path string, summary teacherScoreAuditSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeTeacherScoreFilterSummary(path string, summary teacherScoreFilterSummary) error {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func runInspect(args []string) error {
	if len(args) == 0 || args[0] == "" {
		return fmt.Errorf("usage: eos inspect <artifact.mll>")
	}
	path := args[0]
	mod, err := eosartifact.ReadFile(path)
	if err != nil {
		return err
	}
	fmt.Printf("module: %s\n", mod.Name)
	fmt.Printf("artifact version: %s\n", displayArtifactVersion(mod.Version))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	fmt.Printf("backends: %s\n", joinBackendKinds(mod.Requirements.SupportedBackends))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	embeddedPackage := false
	embeddingManifestPath := eosruntime.ResolveEmbeddingManifestPath(path)
	if _, err := os.Stat(embeddingManifestPath); err == nil {
		manifest, err := eosruntime.ReadEmbeddingManifestFile(embeddingManifestPath)
		if err != nil {
			return err
		}
		fmt.Printf("embedding manifest: %s\n", embeddingManifestPath)
		printEmbeddingManifestSummary(manifest)
	} else if sealed, err := eosruntime.ReadSealedEmbeddingPackage(path); err == nil {
		fmt.Println("embedding manifest: embedded")
		printEmbeddingManifestSummary(sealed.Manifest)
		fmt.Println("package: embedded sealed MLL")
		fmt.Println("package verify: OK")
		embeddedPackage = true
	}
	packagePath := eosruntime.ResolvePackageManifestPath(path)
	if !embeddedPackage {
		if _, err := os.Stat(packagePath); err == nil {
			pkg, err := eosruntime.ReadPackageManifestFile(packagePath)
			if err != nil {
				return err
			}
			verifyPaths := map[string]string{
				"artifact":           path,
				"embedding_manifest": eosruntime.DefaultEmbeddingManifestPath(path),
				"tokenizer":          eosruntime.DefaultTokenizerPath(path),
				"weights":            eosruntime.DefaultWeightFilePath(path),
				"memory_plan":        eosruntime.DefaultMemoryPlanPath(path),
				"train_manifest":     eosruntime.DefaultEmbeddingTrainManifestPath(path),
				"checkpoint":         eosruntime.DefaultEmbeddingCheckpointPath(path),
				"train_profile":      eosruntime.DefaultEmbeddingTrainProfilePath(path),
			}
			if pkg.Kind == eosruntime.PackageEmbedding {
				delete(verifyPaths, "train_manifest")
				delete(verifyPaths, "checkpoint")
				delete(verifyPaths, "train_profile")
			}
			verifyErr := pkg.VerifyFiles(verifyPaths)
			fmt.Printf("package: %s (%s)\n", packagePath, pkg.Kind)
			if verifyErr != nil {
				fmt.Printf("package verify: FAIL (%v)\n", verifyErr)
			} else {
				fmt.Println("package verify: OK")
			}
		}
	}
	profilePath := eosruntime.DefaultEmbeddingTrainProfilePath(path)
	if _, err := os.Stat(profilePath); err == nil {
		profile, err := eosruntime.ReadEmbeddingTrainProfileFile(profilePath)
		if err != nil {
			return err
		}
		fmt.Printf("train profile: step=%d forward=%s optimizer=%s activation=%s contrastive=%s\n",
			profile.Step,
			displayTrainBackend(profile.ForwardBackend),
			displayTrainBackend(profile.OptimizerBackend),
			displayTrainBackend(profile.ActivationBackend),
			displayTrainBackend(profile.ContrastiveBackend),
		)
	}
	return nil
}

func printEmbeddingManifestSummary(manifest eosruntime.EmbeddingManifest) {
	fmt.Printf("embedding model: %s pooled=%s batch=%s output=%s/%s\n",
		displayManifestName(manifest.Name),
		manifest.PooledEntry,
		displayManifestName(manifest.BatchEntry),
		manifest.OutputName,
		manifest.OutputDType,
	)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	if manifest.Tokenizer.VocabSize > 0 || manifest.Tokenizer.MaxSequence > 0 {
		fmt.Printf("tokenizer: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	}
}

func runExportMLL(args []string) error {
	fs := flag.NewFlagSet("export-mll", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	packQuantized := fs.Bool("pack-quantized", true, "store q8/q4 fake-quantized weights as packed payloads with per-tensor scales; false widens them to float32")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos export-mll [flags] <artifact.mll> [output.mll]")
	}
	artifactPath := fs.Arg(0)
	outputPath := ""
	if fs.NArg() > 1 {
		outputPath = fs.Arg(1)
	}
	writtenPath, err := eosruntime.ExportPackageToMLLWithOptions(artifactPath, outputPath, eosruntime.MLLExportOptions{
		PackQuantizedWeights: *packQuantized,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported %q -> %q\n", artifactPath, writtenPath)
	return nil
}

func runInitModel(args []string) error {
	fs := flag.NewFlagSet("init-model", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var vocabSize int
	var maxSequence int
	var architecture string
	var modelDim int
	var outputDim int
	var embeddingDim int
	var hiddenDim int
	var attentionHeads int
	var encoderRepeats int
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var weightDType string
	var optimizer string
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var bootstrapFrom string
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size")
	fs.IntVar(&maxSequence, "max-seq", 0, "maximum token sequence length")
	fs.StringVar(&architecture, "architecture", "", "embedding architecture: legacy_tied_v1 or compact_transformer_v1")
	fs.IntVar(&modelDim, "model-dim", 0, "internal model dimension")
	fs.IntVar(&outputDim, "output-dim", 0, "embedding output dimension")
	fs.IntVar(&embeddingDim, "embedding-dim", 0, "embedding/model dimension")
	fs.IntVar(&hiddenDim, "hidden-dim", 0, "FFN hidden dimension")
	fs.IntVar(&attentionHeads, "attention-heads", 0, "attention head count")
	fs.IntVar(&encoderRepeats, "encoder-repeats", 0, "number of encoder layer repeats (weights are tied; default 2)")
	fs.Int64Var(&seed, "seed", 0, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&weightDType, "weight-dtype", "q8", "trainable weight dtype: q8 or q4")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.StringVar(&bootstrapFrom, "bootstrap-from", "", "initialize overlapping model weights from an existing embedding artifact/package; uses its sibling .embed-train.mll checkpoint")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos init-model [flags] <artifact.mll>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if weightDecay < 0 {
		return fmt.Errorf("weight-decay must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	if embeddingDim > 0 && modelDim > 0 && embeddingDim != modelDim {
		return fmt.Errorf("embedding-dim %d conflicts with model-dim %d", embeddingDim, modelDim)
	}
	path := fs.Arg(0)
	paths, err := models.InitDefaultEmbeddingPackage(path, models.DefaultEmbeddingPackageConfig{
		Name:               name,
		VocabSize:          vocabSize,
		MaxSequence:        maxSequence,
		Architecture:       architecture,
		ModelDim:           modelDim,
		OutputDim:          outputDim,
		EmbeddingDim:       embeddingDim,
		HiddenDim:          hiddenDim,
		AttentionHeads:     attentionHeads,
		EncoderRepeats:     encoderRepeats,
		Seed:               seed,
		LearningRate:       float32(learningRate),
		WeightDecay:        float32(weightDecay),
		WeightBits:         weightBits,
		WeightDType:        weightDType,
		Optimizer:          optimizer,
		ContrastiveLoss:    contrastiveLoss,
		Temperature:        float32(temperature),
		GroupedLossWeight:  float32(groupedLossWeight),
		TeacherLossWeight:  float32(teacherLossWeight),
		TeacherTemperature: float32(teacherTemperature),
		BootstrapFrom:      bootstrapFrom,
	})
	if err != nil {
		return err
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(paths.EmbeddingManifestPath)
	if err != nil {
		return err
	}
	checkpoint, err := eosruntime.ReadEmbeddingTrainCheckpointFile(paths.CheckpointPath)
	if err != nil {
		return err
	}
	fmt.Printf("initialized default embedding model %q\n", manifest.Name)
	fmt.Printf("artifact: %s\n", paths.ArtifactPath)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("tokenizer contract: vocab=%d max_sequence=%d\n", manifest.Tokenizer.VocabSize, manifest.Tokenizer.MaxSequence)
	fmt.Printf("encoder repeats: %d\n", manifest.EncoderRepeats)
	fmt.Printf("training: optimizer=%s loss=%s lr=%.6f temperature=%.6f weight_bits=%d\n",
		checkpoint.Config.Optimizer,
		checkpoint.Config.ContrastiveLoss,
		checkpoint.Config.LearningRate,
		checkpoint.Config.Temperature,
		checkpoint.Config.WeightBits,
	)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	if bootstrapFrom != "" {
		fmt.Printf("bootstrap: %s\n", bootstrapFrom)
	}
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runInitMirage(args []string) error {
	fs := flag.NewFlagSet("init-mirage", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var imageHeight int
	var imageWidth int
	var latentChannels int
	var hyperChannels int
	var bits int
	var factorization string
	var lambda float64
	fs.StringVar(&name, "name", "", "model name")
	fs.IntVar(&imageHeight, "height", 0, "image height")
	fs.IntVar(&imageWidth, "width", 0, "image width")
	fs.IntVar(&latentChannels, "latent-channels", 0, "latent channels")
	fs.IntVar(&hyperChannels, "hyper-channels", 0, "hyperprior channels")
	fs.IntVar(&bits, "bits", 0, "TurboQuant bits: 2, 4, or 8")
	fs.StringVar(&factorization, "factorization", "", "coordinate entropy factorization: categorical or bit-plane")
	fs.Float64Var(&lambda, "lambda", 0, "rate-distortion lambda")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos init-mirage [flags] <artifact.mll>")
	}
	cfg := models.MirageV1Config{
		Name:           name,
		ImageHeight:    imageHeight,
		ImageWidth:     imageWidth,
		LatentChannels: latentChannels,
		HyperChannels:  hyperChannels,
		BitWidth:       bits,
		Factorization:  factorization,
		Lambda:         lambda,
	}
	if err := models.InitMirageV1Artifact(fs.Arg(0), cfg); err != nil {
		return err
	}
	mod, err := eosartifact.ReadFile(fs.Arg(0))
	if err != nil {
		return err
	}
	fmt.Printf("initialized Mirage Image v1 module %q\n", mod.Name)
	fmt.Printf("artifact: %s\n", fs.Arg(0))
	fmt.Printf("entrypoints: %d, steps: %d, kernels: %d\n", len(mod.EntryPoints), len(mod.Steps), len(mod.Kernels))
	if len(mod.Requirements.Capabilities) > 0 {
		fmt.Printf("capabilities: %s\n", strings.Join(mod.Requirements.Capabilities, ", "))
	}
	return nil
}

func runInitTrain(args []string) error {
	fs := flag.NewFlagSet("init-train", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var manifestPath string
	var seed int64
	var learningRate float64
	var weightDecay float64
	var weightBits int
	var optimizer string
	var contrastiveLoss string
	var beta1 float64
	var beta2 float64
	var epsilon float64
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var dims dimFlag
	fs.StringVar(&manifestPath, "manifest", "", "path to embedding manifest (defaults to sibling .embedding.mll)")
	fs.Int64Var(&seed, "seed", 1, "initialization seed")
	fs.Float64Var(&learningRate, "lr", 0, "trainer learning rate")
	fs.Float64Var(&weightDecay, "weight-decay", 0, "trainer weight decay")
	fs.IntVar(&weightBits, "weight-bits", 0, "forward fake-quant bits")
	fs.StringVar(&optimizer, "optimizer", "", "optimizer name")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&beta1, "beta1", 0, "optimizer beta1")
	fs.Float64Var(&beta2, "beta2", 0, "optimizer beta2")
	fs.Float64Var(&epsilon, "epsilon", 0, "optimizer epsilon")
	fs.Float64Var(&temperature, "temperature", 0, "contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.Var(&dims, "dim", "symbolic dimension binding in NAME=VALUE form; repeatable")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos init-train [flags] <artifact.mll>")
	}
	path := fs.Arg(0)
	cfg := eosruntime.EmbeddingTrainConfig{
		LearningRate:       float32(learningRate),
		WeightDecay:        float32(weightDecay),
		WeightBits:         weightBits,
		Optimizer:          optimizer,
		Beta1:              float32(beta1),
		Beta2:              float32(beta2),
		Epsilon:            float32(epsilon),
		ContrastiveLoss:    contrastiveLoss,
		Temperature:        float32(temperature),
		GroupedLossWeight:  float32(groupedLossWeight),
		TeacherLossWeight:  float32(teacherLossWeight),
		TeacherTemperature: float32(teacherTemperature),
	}
	opts := eosruntime.EmbeddingTrainInitOptions{
		Seed:       seed,
		ShapeSizes: dims.values(),
	}
	var (
		paths eosruntime.EmbeddingTrainPackagePaths
		err   error
	)
	if manifestPath == "" {
		manifestPath = eosruntime.ResolveEmbeddingManifestPath(path)
	}
	manifest, readErr := eosruntime.ReadEmbeddingManifestFile(manifestPath)
	if readErr != nil {
		return readErr
	}
	paths, err = eosruntime.InitializeEmbeddingTrainerPackageWithManifest(path, manifest, cfg, opts)
	if err != nil {
		return err
	}
	fmt.Printf("initialized training package %q\n", path)
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("train manifest: %s\n", paths.TrainManifestPath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func runRenameEmbed(args []string) error {
	fs := flag.NewFlagSet("rename-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var name string
	var maxSeq int
	fs.StringVar(&name, "name", "", "new embedding model name")
	fs.IntVar(&maxSeq, "max-seq", 0, "new tokenizer maximum sequence length; 0 leaves unchanged")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos rename-embed [--name <model-name>] [--max-seq N] <input.mll> <output.mll>")
	}
	if maxSeq < 0 {
		return fmt.Errorf("rename-embed --max-seq must be non-negative")
	}
	name = strings.TrimSpace(name)
	if name == "" && maxSeq == 0 {
		return fmt.Errorf("rename-embed requires --name or a positive --max-seq")
	}
	inputPath := fs.Arg(0)
	outputPath := fs.Arg(1)
	trainer, err := eosruntime.LoadEmbeddingTrainerPackage(inputPath)
	if err != nil {
		return err
	}
	if name != "" {
		if err := trainer.RenameEmbeddingModel(name); err != nil {
			return err
		}
	}
	if maxSeq > 0 {
		if err := trainer.SetEmbeddingTokenizerMaxSequence(maxSeq); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	if err := copyTokenizerIfPresent(inputPath, outputPath); err != nil {
		return err
	}
	paths, err := trainer.WriteTrainingPackage(outputPath)
	if err != nil {
		return err
	}
	fmt.Printf("rewrote embedding package %q -> %q\n", inputPath, outputPath)
	if name != "" {
		fmt.Printf("model: %s\n", name)
	}
	if maxSeq > 0 {
		fmt.Printf("tokenizer max_sequence: %d\n", maxSeq)
	}
	fmt.Printf("embedding manifest: %s\n", paths.EmbeddingManifestPath)
	fmt.Printf("weights: %s\n", paths.WeightFilePath)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	return nil
}

func copyTokenizerIfPresent(inputPath, outputPath string) error {
	sourcePath := eosruntime.DefaultTokenizerPath(inputPath)
	if _, err := os.Stat(sourcePath); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		return err
	}
	return os.WriteFile(eosruntime.DefaultTokenizerPath(outputPath), data, 0o644)
}

func runTrainEmbed(args []string) error {
	fs := flag.NewFlagSet("train-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var evalEverySteps int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var maxListwiseTrainPairs int64
	var maxListwiseEvalPairs int64
	var tokenizerPath string
	var noTokenizer bool
	var planOnly bool
	var evalOnly bool
	var pairwiseTrain bool
	var hardNegativeTrain bool
	var scoreSpectrumTrain bool
	var listwiseGeometryTrain bool
	var vectorDistillTrain bool
	var vectorDistillDefaultRole string
	var vectorDistillRelationalWeight float64
	var movementDiagnostics bool
	var allowResearchOnlyScoreSpectrum bool
	var allowResearchOnlyListwiseGeometry bool
	var allowResearchOnlyVectorDistill bool
	var scoreSpectrumEvalPath string
	var scoreSpectrumLossMode string
	var scoreSpectrumRecoveryWeight float64
	var scoreSpectrumRecoveryMargin float64
	var scoreSpectrumRecoveryTopK int
	var scoreSpectrumRecoveryTau float64
	var hardNegativesPerQuery int
	var hardNegativeSourceWeights string
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var teacherSourceTemperatures string
	var teacherSourceWeights string
	var teacherScoreNormalization string
	var matryoshkaDims string
	var matryoshkaWeights string
	var clearTurboQuantPrefix bool
	var turboQuantPrefixBits string
	var turboQuantPrefixObjectives string
	var turboQuantPrefixWeight float64
	var turboQuantPrefixSeed int64
	var turboQuantPrefixScoreMode string
	var turboQuantCompactObjectives string
	var clearTurboQuantRankMargin bool
	var turboQuantRankMarginObjectives string
	var turboQuantRankMargin float64
	var retrievalEvalDir string
	var retrievalEvalSplit string
	var retrievalEvalMaxDocs int
	var retrievalEvalMaxQueries int
	var retrievalEvalBatchSize int
	var retrievalEvalTopK int
	var retrievalEvalTokenizerPath string
	var retrievalEvalRoleMode string
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, loss, retrieval_ndcg(_at_10), retrieval_map(_at_100), or retrieval_recall(_at_100)")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.Int64Var(&maxListwiseTrainPairs, "max-listwise-train-pairs", 100000, "maximum listwise geometry train query-document pairs per epoch (0 disables)")
	fs.Int64Var(&maxListwiseEvalPairs, "max-listwise-eval-pairs", 100000, "maximum listwise geometry eval query-document pairs per pass (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer JSON for text-pair datasets")
	fs.BoolVar(&noTokenizer, "no-tokenizer", false, "disable sibling tokenizer discovery and treat JSONL as tokenized")
	fs.BoolVar(&planOnly, "plan-only", false, "print planned workload and exit without training")
	fs.BoolVar(&evalOnly, "eval-only", false, "evaluate the package without running optimizer steps")
	fs.BoolVar(&pairwiseTrain, "pairwise-train", false, "treat the training JSONL as labeled pair examples instead of contrastive positives")
	fs.BoolVar(&hardNegativeTrain, "hard-negative-train", false, "group labeled pair JSONL into query-positive-hard-negative contrastive batches")
	fs.BoolVar(&scoreSpectrumTrain, "score-spectrum-train", false, "treat the training JSONL as grouped score-spectrum examples with row-local candidates")
	fs.BoolVar(&listwiseGeometryTrain, "listwise-geometry-train", false, "treat the training JSONL as listwise query/document teacher geometry batches")
	fs.BoolVar(&vectorDistillTrain, "vector-distill-train", false, "treat the training JSONL as (text, teacher_vector) vector-distillation examples")
	fs.StringVar(&vectorDistillDefaultRole, "vector-distill-default-role", eosruntime.EmbeddingRoleQuery, "role (query, document, or raw) applied to vector-distill JSONL rows that omit an explicit \"role\" field")
	fs.Float64Var(&vectorDistillRelationalWeight, "vector-distill-relational-weight", 0, "weight for an opt-in in-batch relational (similarity-matrix) distillation term over raw student vectors (0 disables); requires --vector-distill-train")
	fs.BoolVar(&movementDiagnostics, "movement-diagnostics", false, "record aggregate gradient and parameter-delta movement diagnostics for listwise geometry training")
	fs.BoolVar(&allowResearchOnlyScoreSpectrum, "allow-research-only-score-spectrum", false, "allow research-only score-spectrum training rows")
	fs.BoolVar(&allowResearchOnlyListwiseGeometry, "allow-research-only-listwise-geometry", false, "allow research-only listwise geometry training rows")
	fs.BoolVar(&allowResearchOnlyVectorDistill, "allow-research-only-vector-distill", false, "allow research-only vector-distillation training rows")
	fs.StringVar(&scoreSpectrumEvalPath, "score-spectrum-eval", "", "path to grouped score-spectrum JSONL for native score-spectrum eval")
	fs.StringVar(&scoreSpectrumLossMode, "score-spectrum-loss-mode", "", "score-spectrum loss mode: hard_soft, recovery, or hard_soft_recovery")
	fs.Float64Var(&scoreSpectrumRecoveryWeight, "score-spectrum-recovery-weight", 0, "global recovery loss weight for score-spectrum training")
	fs.Float64Var(&scoreSpectrumRecoveryMargin, "score-spectrum-recovery-margin", 0, "non-negative recovery margin for score-spectrum training")
	fs.IntVar(&scoreSpectrumRecoveryTopK, "score-spectrum-recovery-top-k", 0, "top-k hardest eligible negatives for score-spectrum recovery loss (default 4)")
	fs.Float64Var(&scoreSpectrumRecoveryTau, "score-spectrum-recovery-tau", 0, "positive temperature for score-spectrum recovery loss")
	fs.IntVar(&hardNegativesPerQuery, "hard-negatives-per-query", 1, "maximum explicit negatives to attach to each query-positive example")
	fs.StringVar(&hardNegativeSourceWeights, "hard-negative-source-weights", "", "comma-separated source=weight hard-negative batch mix, for example scifact=2,nfcorpus=1,fiqa=2")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.StringVar(&teacherSourceTemperatures, "teacher-source-temperatures", "", "comma-separated source=temperature overrides for teacher distillation, for example scifact=10,nfcorpus:model=1.5")
	fs.StringVar(&teacherSourceWeights, "teacher-source-weights", "", "comma-separated source=weight overrides for teacher distillation influence, for example scifact=1,nfcorpus=0,fiqa=0.25")
	fs.StringVar(&teacherScoreNormalization, "teacher-score-normalization", "", "normalize hard-negative teacher_scores before distillation: none, source_zscore, family_zscore, or example_zscore")
	fs.StringVar(&matryoshkaDims, "matryoshka-dims", "", "comma-separated compact prefix dimensions to train with InfoNCE, for example 64,128")
	fs.StringVar(&matryoshkaWeights, "matryoshka-weights", "", "optional comma-separated positive weights matching --matryoshka-dims")
	fs.BoolVar(&clearTurboQuantPrefix, "clear-turboquant-prefix", false, "clear inherited TurboQuant compact-prefix objectives for continuation training")
	fs.StringVar(&turboQuantPrefixBits, "turboquant-prefix-bits", "", "comma-separated TurboQuant bit widths for quantized compact-prefix InfoNCE, supported: 2..8")
	fs.StringVar(&turboQuantPrefixObjectives, "turboquant-prefix-objectives", "", "comma-separated TurboQuant compact-prefix objectives as dim:bit=weight, for example 128:4=0.5")
	fs.Float64Var(&turboQuantPrefixWeight, "turboquant-prefix-weight", 0, "optional weight for each TurboQuant compact-prefix objective (default 1 when bits are set)")
	fs.Int64Var(&turboQuantPrefixSeed, "turboquant-prefix-seed", 0, "TurboQuant compact-prefix quantizer seed (default matches multivector retrieval)")
	fs.StringVar(&turboQuantPrefixScoreMode, "turboquant-prefix-score-mode", "", "TurboQuant compact-prefix score mode: reconstruct_cosine (default) or prepared_ip")
	fs.StringVar(&turboQuantCompactObjectives, "turboquant-compact-objectives", "", "comma-separated hard-negative prepared-IP TurboQuant compact objectives as dim:bit=weight, for example 256:4=0.05")
	fs.BoolVar(&clearTurboQuantRankMargin, "clear-turboquant-rank-margin", false, "clear inherited TurboQuant rank-margin objectives for continuation hard-negative training")
	fs.StringVar(&turboQuantRankMarginObjectives, "turboquant-rank-margin-objectives", "", "comma-separated TurboQuant hard-negative rank-margin objectives as dim:bit=weight, for example 128:4=0.1")
	fs.Float64Var(&turboQuantRankMargin, "turboquant-rank-margin", 0, "TurboQuant hard-negative rank-margin target (default 0.02 when objectives are enabled)")
	fs.StringVar(&retrievalEvalDir, "retrieval-eval-dir", "", "BEIR-style dataset dir for per-epoch retrieval eval with current weights (nDCG@10, MAP@100, recall@100); enables -select-metric retrieval_ndcg(_at_10), retrieval_map, or retrieval_recall")
	fs.StringVar(&retrievalEvalSplit, "retrieval-eval-split", "test", "qrels split for retrieval eval (test/dev/train)")
	fs.IntVar(&retrievalEvalMaxDocs, "retrieval-eval-max-docs", 5000, "cap corpus docs embedded per retrieval eval (0 = all); smaller is faster per-epoch")
	fs.IntVar(&retrievalEvalMaxQueries, "retrieval-eval-max-queries", 500, "cap queries per retrieval eval (0 = all)")
	fs.IntVar(&retrievalEvalBatchSize, "retrieval-eval-batch-size", 0, "batch size for retrieval eval embedding (0 = use --batch-size)")
	fs.IntVar(&retrievalEvalTopK, "retrieval-eval-top-k", 100, "top-k for retrieval eval ranking")
	fs.StringVar(&retrievalEvalTokenizerPath, "retrieval-eval-tokenizer", "", "tokenizer for raw-text retrieval eval when training uses --no-tokenizer (defaults to --tokenizer or sibling tokenizer)")
	fs.StringVar(&retrievalEvalRoleMode, "retrieval-eval-role-mode", eosruntime.EmbeddingRoleModeAuto, "embedding role mode for retrieval eval: auto, raw, or query-document")
	if err := fs.Parse(args); err != nil {
		return err
	}
	progressEveryProvided := flagWasProvided(fs, "progress-every")
	teacherLossWeightSet := flagWasProvided(fs, "teacher-loss-weight")
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]\n       eos train-embed --eval-only [flags] <artifact.mll> <eval.jsonl>")
	}
	// Retrieval-gated selection default: when --retrieval-eval-dir is set and
	// the caller did not explicitly pass --select-metric, upgrade the
	// selection metric to the retrieval nDCG@10 gate instead of the legacy
	// pairwise default. Explicit --select-metric always wins. This guards the
	// recorded failure mode where --restore-best silently restored an
	// untrained step-0 checkpoint because selection ran on a pairwise metric
	// uncorrelated with retrieval (see commit 2861ac9).
	retrievalEvalDirSet := strings.TrimSpace(retrievalEvalDir) != ""
	selectMetricProvided := flagWasProvided(fs, "select-metric")
	if retrievalEvalDirSet && !selectMetricProvided {
		selectMetric = trainEmbedRetrievalGatedSelectMetric
		fmt.Fprintf(os.Stderr, "select-metric: auto-selected %q because --retrieval-eval-dir was set without an explicit --select-metric (retrieval-gated checkpoint selection)\n", selectMetric)
	}
	if restoreBest && !evalOnly && !retrievalEvalDirSet && !isRetrievalSelectMetric(selectMetric) {
		if mode := activeRetrievalOrientedTrainMode(hardNegativeTrain, scoreSpectrumTrain, listwiseGeometryTrain, vectorDistillTrain); mode != "" {
			fmt.Fprint(os.Stderr, restoreBestPairwiseSelectionWarning(mode, selectMetric))
		}
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	if turboQuantPrefixWeight < 0 {
		return fmt.Errorf("turboquant-prefix-weight must be non-negative")
	}
	if turboQuantRankMargin < 0 || math.IsNaN(turboQuantRankMargin) || math.IsInf(turboQuantRankMargin, 0) {
		return fmt.Errorf("turboquant-rank-margin must be finite and non-negative")
	}
	if vectorDistillRelationalWeight < 0 || math.IsNaN(vectorDistillRelationalWeight) || math.IsInf(vectorDistillRelationalWeight, 0) {
		return fmt.Errorf("vector-distill-relational-weight must be finite and non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	if maxListwiseTrainPairs < 0 {
		return fmt.Errorf("max-listwise-train-pairs must be non-negative")
	}
	if maxListwiseEvalPairs < 0 {
		return fmt.Errorf("max-listwise-eval-pairs must be non-negative")
	}
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	parsedRetrievalEvalRoleMode, parseErr := normalizeRetrievalEvalRoleModeForCLI(retrievalEvalRoleMode)
	if parseErr != nil {
		return parseErr
	}
	if hardNegativesPerQuery < 0 {
		return fmt.Errorf("hard-negatives-per-query must be non-negative")
	}
	if scoreSpectrumRecoveryWeight < 0 || math.IsNaN(scoreSpectrumRecoveryWeight) || math.IsInf(scoreSpectrumRecoveryWeight, 0) {
		return fmt.Errorf("score-spectrum-recovery-weight must be finite and non-negative")
	}
	if scoreSpectrumRecoveryMargin < 0 || math.IsNaN(scoreSpectrumRecoveryMargin) || math.IsInf(scoreSpectrumRecoveryMargin, 0) {
		return fmt.Errorf("score-spectrum-recovery-margin must be finite and non-negative")
	}
	if scoreSpectrumRecoveryTopK < 0 {
		return fmt.Errorf("score-spectrum-recovery-top-k must be non-negative")
	}
	if scoreSpectrumRecoveryTau < 0 || math.IsNaN(scoreSpectrumRecoveryTau) || math.IsInf(scoreSpectrumRecoveryTau, 0) {
		return fmt.Errorf("score-spectrum-recovery-tau must be finite and non-negative")
	}
	trainModeCount := 0
	for _, enabled := range []bool{pairwiseTrain, hardNegativeTrain, scoreSpectrumTrain, listwiseGeometryTrain, vectorDistillTrain} {
		if enabled {
			trainModeCount++
		}
	}
	if trainModeCount > 1 {
		return fmt.Errorf("set only one of --pairwise-train, --hard-negative-train, --score-spectrum-train, --listwise-geometry-train, or --vector-distill-train")
	}
	parsedSourceWeights, parseErr := parsePositiveIntWeightMap(hardNegativeSourceWeights)
	if parseErr != nil {
		return fmt.Errorf("hard-negative-source-weights: %w", parseErr)
	}
	parsedTeacherSourceTemperatures, parseErr := parsePositiveFloatMap(teacherSourceTemperatures)
	if parseErr != nil {
		return fmt.Errorf("teacher-source-temperatures: %w", parseErr)
	}
	parsedTeacherSourceWeights, parseErr := parseNonNegativeFloatMap(teacherSourceWeights)
	if parseErr != nil {
		return fmt.Errorf("teacher-source-weights: %w", parseErr)
	}
	parsedMatryoshkaDims, parseErr := parsePositiveIntList(matryoshkaDims)
	if parseErr != nil {
		return fmt.Errorf("matryoshka-dims: %w", parseErr)
	}
	parsedMatryoshkaWeights, parseErr := parsePositiveFloatList(matryoshkaWeights)
	if parseErr != nil {
		return fmt.Errorf("matryoshka-weights: %w", parseErr)
	}
	parsedTurboQuantPrefixBits, parseErr := parsePositiveIntList(turboQuantPrefixBits)
	if parseErr != nil {
		return fmt.Errorf("turboquant-prefix-bits: %w", parseErr)
	}
	if err := validateTurboQuantPrefixBitsFlag(parsedTurboQuantPrefixBits); err != nil {
		return err
	}
	parsedTurboQuantPrefixObjectives, parseErr := eosruntime.ParseTurboQuantPrefixObjectives(turboQuantPrefixObjectives)
	if parseErr != nil {
		return fmt.Errorf("turboquant-prefix-objectives: %w", parseErr)
	}
	if len(parsedTurboQuantPrefixObjectives) > 0 && len(parsedTurboQuantPrefixBits) > 0 {
		return fmt.Errorf("--turboquant-prefix-objectives is mutually exclusive with --turboquant-prefix-bits")
	}
	if clearTurboQuantPrefix && len(parsedTurboQuantPrefixBits) > 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-bits")
	}
	if clearTurboQuantPrefix && len(parsedTurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-objectives")
	}
	if clearTurboQuantPrefix && turboQuantPrefixWeight != 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-weight")
	}
	if len(parsedTurboQuantPrefixObjectives) > 0 && turboQuantPrefixWeight != 0 {
		return fmt.Errorf("--turboquant-prefix-weight must not be set with --turboquant-prefix-objectives")
	}
	parsedTurboQuantCompactObjectives, parseErr := eosruntime.ParseTurboQuantPrefixObjectives(turboQuantCompactObjectives)
	if parseErr != nil {
		return fmt.Errorf("turboquant-compact-objectives: %w", parseErr)
	}
	parsedTurboQuantRankMarginObjectives, parseErr := eosruntime.ParseTurboQuantPrefixObjectives(turboQuantRankMarginObjectives)
	if parseErr != nil {
		return fmt.Errorf("turboquant-rank-margin-objectives: %w", parseErr)
	}
	if clearTurboQuantPrefix && turboQuantPrefixSeed != 0 && len(parsedTurboQuantCompactObjectives) == 0 && len(parsedTurboQuantRankMarginObjectives) == 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-seed")
	}
	if clearTurboQuantPrefix && strings.TrimSpace(turboQuantPrefixScoreMode) != "" && len(parsedTurboQuantRankMarginObjectives) == 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-score-mode")
	}
	if clearTurboQuantRankMargin && len(parsedTurboQuantRankMarginObjectives) > 0 {
		return fmt.Errorf("--clear-turboquant-rank-margin is mutually exclusive with --turboquant-rank-margin-objectives")
	}
	if clearTurboQuantRankMargin && turboQuantRankMargin != 0 {
		return fmt.Errorf("--clear-turboquant-rank-margin is mutually exclusive with --turboquant-rank-margin")
	}
	parsedTurboQuantPrefixScoreMode := ""
	if strings.TrimSpace(turboQuantPrefixScoreMode) != "" {
		mode, parseErr := eosruntime.NormalizeTurboQuantPrefixScoreModeForCLI(turboQuantPrefixScoreMode)
		if parseErr != nil {
			return parseErr
		}
		parsedTurboQuantPrefixScoreMode = mode
	}
	if len(parsedSourceWeights) > 0 && !hardNegativeTrain {
		return fmt.Errorf("--hard-negative-source-weights requires --hard-negative-train")
	}
	if len(parsedTeacherSourceTemperatures) > 0 && !hardNegativeTrain {
		return fmt.Errorf("--teacher-source-temperatures requires --hard-negative-train")
	}
	if len(parsedTeacherSourceWeights) > 0 && !hardNegativeTrain {
		return fmt.Errorf("--teacher-source-weights requires --hard-negative-train")
	}
	if strings.TrimSpace(teacherScoreNormalization) != "" && !hardNegativeTrain {
		return fmt.Errorf("--teacher-score-normalization requires --hard-negative-train")
	}
	if len(parsedTurboQuantRankMarginObjectives) > 0 && !hardNegativeTrain && !scoreSpectrumTrain && !listwiseGeometryTrain {
		return fmt.Errorf("--turboquant-rank-margin-objectives requires --hard-negative-train")
	}
	if len(parsedTurboQuantCompactObjectives) > 0 && !hardNegativeTrain && !scoreSpectrumTrain && !listwiseGeometryTrain {
		return fmt.Errorf("--turboquant-compact-objectives requires --hard-negative-train")
	}
	if allowResearchOnlyScoreSpectrum && !scoreSpectrumTrain {
		return fmt.Errorf("--allow-research-only-score-spectrum requires --score-spectrum-train")
	}
	if allowResearchOnlyListwiseGeometry && !listwiseGeometryTrain {
		return fmt.Errorf("--allow-research-only-listwise-geometry requires --listwise-geometry-train")
	}
	if allowResearchOnlyVectorDistill && !vectorDistillTrain {
		return fmt.Errorf("--allow-research-only-vector-distill requires --vector-distill-train")
	}
	if flagWasProvided(fs, "vector-distill-default-role") && !vectorDistillTrain {
		return fmt.Errorf("--vector-distill-default-role requires --vector-distill-train")
	}
	if vectorDistillRelationalWeight != 0 && !vectorDistillTrain {
		return fmt.Errorf("--vector-distill-relational-weight requires --vector-distill-train")
	}
	parsedVectorDistillDefaultRole, parseErr := normalizeVectorDistillRoleForCLI(vectorDistillDefaultRole)
	if parseErr != nil {
		return fmt.Errorf("vector-distill-default-role: %w", parseErr)
	}
	parsedScoreSpectrumLossMode := ""
	if strings.TrimSpace(scoreSpectrumLossMode) != "" {
		var parseErr error
		parsedScoreSpectrumLossMode, parseErr = eosruntime.NormalizeScoreSpectrumLossModeForCLI(scoreSpectrumLossMode)
		if parseErr != nil {
			return parseErr
		}
	}
	if strings.TrimSpace(scoreSpectrumEvalPath) != "" && !scoreSpectrumTrain {
		return fmt.Errorf("--score-spectrum-eval requires --score-spectrum-train")
	}
	if (parsedScoreSpectrumLossMode != "" || scoreSpectrumRecoveryWeight != 0 || scoreSpectrumRecoveryMargin != 0 || scoreSpectrumRecoveryTopK != 0 || scoreSpectrumRecoveryTau != 0) && !scoreSpectrumTrain {
		return fmt.Errorf("score-spectrum recovery options require --score-spectrum-train")
	}
	if scoreSpectrumTrain {
		if len(parsedMatryoshkaDims) > 0 {
			return fmt.Errorf("--score-spectrum-train does not support --matryoshka-dims")
		}
		if len(parsedTurboQuantPrefixBits) > 0 || len(parsedTurboQuantPrefixObjectives) > 0 {
			return fmt.Errorf("--score-spectrum-train does not support TurboQuant prefix objectives")
		}
		if len(parsedTurboQuantCompactObjectives) > 0 {
			return fmt.Errorf("--score-spectrum-train does not support TurboQuant compact objectives")
		}
		if len(parsedTurboQuantRankMarginObjectives) > 0 {
			return fmt.Errorf("--score-spectrum-train does not support TurboQuant rank-margin objectives")
		}
	}
	if listwiseGeometryTrain {
		if !progressEveryProvided && progressEvery == 0 {
			progressEvery = 1
		}
		if len(parsedMatryoshkaDims) > 0 {
			return fmt.Errorf("--listwise-geometry-train does not support --matryoshka-dims")
		}
		if len(parsedTurboQuantPrefixBits) > 0 || len(parsedTurboQuantPrefixObjectives) > 0 {
			return fmt.Errorf("--listwise-geometry-train does not support TurboQuant prefix objectives")
		}
		if len(parsedTurboQuantCompactObjectives) > 0 {
			return fmt.Errorf("--listwise-geometry-train does not support TurboQuant compact objectives")
		}
		if len(parsedTurboQuantRankMarginObjectives) > 0 {
			return fmt.Errorf("--listwise-geometry-train does not support TurboQuant rank-margin objectives")
		}
		if noTokenizer {
			return fmt.Errorf("--listwise-geometry-train requires text tokenization; remove --no-tokenizer or set --tokenizer")
		}
	} else {
		maxListwiseTrainPairs = 0
		maxListwiseEvalPairs = 0
	}
	if vectorDistillTrain {
		if !progressEveryProvided && progressEvery == 0 {
			progressEvery = 1
		}
		if len(parsedMatryoshkaDims) > 0 {
			return fmt.Errorf("--vector-distill-train does not support --matryoshka-dims")
		}
		if len(parsedTurboQuantPrefixBits) > 0 || len(parsedTurboQuantPrefixObjectives) > 0 {
			return fmt.Errorf("--vector-distill-train does not support TurboQuant prefix objectives")
		}
		if len(parsedTurboQuantCompactObjectives) > 0 {
			return fmt.Errorf("--vector-distill-train does not support TurboQuant compact objectives")
		}
		if len(parsedTurboQuantRankMarginObjectives) > 0 {
			return fmt.Errorf("--vector-distill-train does not support TurboQuant rank-margin objectives")
		}
		if noTokenizer {
			return fmt.Errorf("--vector-distill-train requires text tokenization; remove --no-tokenizer or set --tokenizer")
		}
	}
	path := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	if evalOnly && fs.NArg() == 2 && strings.TrimSpace(scoreSpectrumEvalPath) == "" && !listwiseGeometryTrain {
		evalPath = trainPath
		trainPath = ""
	}
	if noTokenizer && tokenizerPath != "" {
		return fmt.Errorf("set either --tokenizer or --no-tokenizer, not both")
	}
	if tokenizerPath == "" && !noTokenizer {
		defaultTokenizerPath := eosruntime.DefaultTokenizerPath(path)
		if _, err := os.Stat(defaultTokenizerPath); err == nil {
			tokenizerPath = defaultTokenizerPath
		}
	}
	runConfig := eosruntime.EmbeddingTrainRunConfig{
		Epochs:                            epochs,
		BatchSize:                         batchSize,
		Shuffle:                           shuffle,
		Seed:                              seed,
		EvalEveryEpoch:                    evalEvery,
		EvalEverySteps:                    evalEverySteps,
		EarlyStoppingPatience:             patience,
		SelectMetric:                      selectMetric,
		MinDelta:                          float32(minDelta),
		RestoreBest:                       restoreBest,
		LengthBucketBatches:               lengthBucketBatches,
		LearningRate:                      float32(learningRate),
		ContrastiveLoss:                   contrastiveLoss,
		Temperature:                       float32(temperature),
		GroupedLossWeight:                 float32(groupedLossWeight),
		TeacherLossWeight:                 float32(teacherLossWeight),
		TeacherLossWeightSet:              teacherLossWeightSet,
		TeacherTemperature:                float32(teacherTemperature),
		TeacherSourceTemperatures:         parsedTeacherSourceTemperatures,
		TeacherSourceWeights:              parsedTeacherSourceWeights,
		TeacherScoreNormalization:         teacherScoreNormalization,
		MatryoshkaDims:                    parsedMatryoshkaDims,
		MatryoshkaWeights:                 parsedMatryoshkaWeights,
		ClearTurboQuantPrefix:             clearTurboQuantPrefix,
		TurboQuantPrefixBits:              parsedTurboQuantPrefixBits,
		TurboQuantPrefixObjectives:        parsedTurboQuantPrefixObjectives,
		TurboQuantPrefixWeight:            float32(turboQuantPrefixWeight),
		TurboQuantPrefixSeed:              turboQuantPrefixSeed,
		TurboQuantPrefixScoreMode:         parsedTurboQuantPrefixScoreMode,
		TurboQuantCompactObjectives:       parsedTurboQuantCompactObjectives,
		ClearTurboQuantRankMargin:         clearTurboQuantRankMargin,
		TurboQuantRankMarginObjectives:    parsedTurboQuantRankMarginObjectives,
		TurboQuantRankMargin:              float32(turboQuantRankMargin),
		ProgressEverySteps:                progressEvery,
		EvalOnly:                          evalOnly,
		PairwiseTrain:                     pairwiseTrain,
		HardNegativeTrain:                 hardNegativeTrain,
		ScoreSpectrumTrain:                scoreSpectrumTrain,
		ListwiseGeometryTrain:             listwiseGeometryTrain,
		VectorDistillTrain:                vectorDistillTrain,
		VectorDistillDefaultRole:          parsedVectorDistillDefaultRole,
		VectorDistillRelationalWeight:     float32(vectorDistillRelationalWeight),
		MovementDiagnostics:               movementDiagnostics,
		AllowResearchOnlyScoreSpectrum:    allowResearchOnlyScoreSpectrum,
		AllowResearchOnlyListwiseGeometry: allowResearchOnlyListwiseGeometry,
		AllowResearchOnlyVectorDistill:    allowResearchOnlyVectorDistill,
		ScoreSpectrumEvalPath:             scoreSpectrumEvalPath,
		MaxListwiseGeometryTrainPairs:     maxListwiseTrainPairs,
		MaxListwiseGeometryEvalPairs:      maxListwiseEvalPairs,
		ScoreSpectrumLossMode:             parsedScoreSpectrumLossMode,
		ScoreSpectrumRecoveryWeight:       float32(scoreSpectrumRecoveryWeight),
		ScoreSpectrumRecoveryMargin:       float32(scoreSpectrumRecoveryMargin),
		ScoreSpectrumRecoveryTopK:         scoreSpectrumRecoveryTopK,
		ScoreSpectrumRecoveryTau:          float32(scoreSpectrumRecoveryTau),
		HardNegativesPerQuery:             hardNegativesPerQuery,
		HardNegativeSourceWeights:         parsedSourceWeights,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	if strings.TrimSpace(retrievalEvalDir) != "" {
		retrievalTokenizerPath := strings.TrimSpace(retrievalEvalTokenizerPath)
		if retrievalTokenizerPath == "" {
			retrievalTokenizerPath = tokenizerPath
		}
		if retrievalTokenizerPath == "" {
			return fmt.Errorf("--retrieval-eval-dir needs a tokenizer to embed corpus/query text (set --retrieval-eval-tokenizer, set --tokenizer, or keep the sibling tokenizer)")
		}
		tokFile, terr := eosruntime.ReadTokenizerFile(retrievalTokenizerPath)
		if terr != nil {
			return fmt.Errorf("retrieval eval tokenizer: %w", terr)
		}
		corpusPath, queriesPath, qrelsPath := eosruntime.BEIRRetrievalPaths(retrievalEvalDir, retrievalEvalSplit)
		retrBatch := retrievalEvalBatchSize
		if retrBatch <= 0 {
			retrBatch = batchSize
		}
		runConfig.RetrievalEvalRuntime = eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
		runConfig.RetrievalEvalTokenizer = &tokFile
		runConfig.RetrievalEval = eosruntime.RetrievalEvalConfig{
			DatasetName: filepath.Base(retrievalEvalDir),
			CorpusPath:  corpusPath,
			QueriesPath: queriesPath,
			QrelsPath:   qrelsPath,
			BatchSize:   retrBatch,
			TopK:        retrievalEvalTopK,
			MaxDocs:     retrievalEvalMaxDocs,
			MaxQueries:  retrievalEvalMaxQueries,
			RoleMode:    parsedRetrievalEvalRoleMode,
		}
	}
	workload, workloadErr := estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath, runConfig)
	if workloadErr == nil {
		fmt.Printf("planned workload: %s\n", formatTrainWorkload(workload))
		if err := validateTrainEmbedListwiseGeometryWorkload(workload, runConfig); err != nil {
			return err
		}
	}
	if planOnly {
		if workloadErr != nil {
			return workloadErr
		}
		return nil
	}
	if progressEvery > 0 {
		mode := "train"
		if evalOnly {
			mode = "eval"
		}
		fmt.Printf("progress: phase=fit_start mode=%s artifact=%q train=%q eval=%q tokenizer=%q progress_every=%d\n", mode, path, trainPath, evalPath, tokenizerPath, progressEvery)
	}
	var (
		summary eosruntime.EmbeddingTrainRunSummary
		paths   eosruntime.EmbeddingTrainPackagePaths
		err     error
	)
	if tokenizerPath != "" {
		summary, paths, err = eosruntime.TrainEmbeddingPackageFromTextContrastiveFiles(path, tokenizerPath, trainPath, evalPath, runConfig)
	} else {
		summary, paths, err = eosruntime.TrainEmbeddingPackageFromContrastiveFiles(path, trainPath, evalPath, runConfig)
	}
	if err != nil {
		return err
	}
	if evalOnly {
		fmt.Printf("evaluated package %q\n", path)
	} else {
		fmt.Printf("trained package %q\n", path)
	}
	if tokenizerPath != "" {
		fmt.Printf("tokenizer: %s\n", tokenizerPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f retrieval_ndcg=%.6f retrieval_map=%.6f retrieval_recall=%.6f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.RetrievalNDCGAt10, summary.FinalEval.RetrievalMAPAt100, summary.FinalEval.RetrievalRecallAt100, summary.FinalEval.PairCount)
		printFinalRetrievalEval(summary.FinalEval)
	}
	if summary.FinalScoreSpectrumEval != nil {
		fmt.Printf("final score-spectrum eval: loss=%.6f any_positive_top1=%.6f original_positive_top1=%.6f alternate_recovery=%.6f best_positive_hardest_negative_margin=%.6f rows=%d candidates=%d\n", summary.FinalScoreSpectrumEval.Loss, summary.FinalScoreSpectrumEval.AnyPositiveTop1, summary.FinalScoreSpectrumEval.OriginalPositiveTop1, summary.FinalScoreSpectrumEval.AlternateRelevantRecovery, summary.FinalScoreSpectrumEval.BestPositiveHardestNegativeMargin, summary.FinalScoreSpectrumEval.RowCount, summary.FinalScoreSpectrumEval.CandidateCount)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.TrainProfilePath)
	if metricsJSONPath != "" {
		mode := "train"
		if evalOnly {
			mode = "eval"
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-embed", mode, path, tokenizerPath, summary, paths, nil); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

func normalizeRetrievalEvalRoleModeForCLI(mode string) (string, error) {
	switch strings.TrimSpace(mode) {
	case "", eosruntime.EmbeddingRoleModeAuto:
		return eosruntime.EmbeddingRoleModeAuto, nil
	case eosruntime.EmbeddingRoleModeRaw:
		return eosruntime.EmbeddingRoleModeRaw, nil
	case eosruntime.EmbeddingRoleModeQueryDocument:
		return eosruntime.EmbeddingRoleModeQueryDocument, nil
	default:
		return "", fmt.Errorf("unsupported retrieval-eval-role-mode %q (supported: %s, %s, %s)", mode, eosruntime.EmbeddingRoleModeAuto, eosruntime.EmbeddingRoleModeRaw, eosruntime.EmbeddingRoleModeQueryDocument)
	}
}

func normalizeVectorDistillRoleForCLI(role string) (string, error) {
	switch strings.TrimSpace(role) {
	case "", eosruntime.EmbeddingRoleQuery:
		return eosruntime.EmbeddingRoleQuery, nil
	case eosruntime.EmbeddingRoleDocument:
		return eosruntime.EmbeddingRoleDocument, nil
	case eosruntime.EmbeddingRoleRaw:
		return eosruntime.EmbeddingRoleRaw, nil
	default:
		return "", fmt.Errorf("unsupported vector-distill role %q (supported: %s, %s, %s)", role, eosruntime.EmbeddingRoleQuery, eosruntime.EmbeddingRoleDocument, eosruntime.EmbeddingRoleRaw)
	}
}

func parsePositiveIntWeightMap(raw string) (map[string]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := map[string]int{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("entry %q must be source=weight", item)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key == "" {
			return nil, fmt.Errorf("entry %q has an empty source", item)
		}
		weight, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil || weight <= 0 {
			return nil, fmt.Errorf("entry %q must use a positive integer weight", item)
		}
		out[key] = weight
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func parsePositiveFloatMap(raw string) (map[string]float32, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := map[string]float32{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("entry %q must be source=value", item)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key == "" {
			return nil, fmt.Errorf("entry %q has an empty source", item)
		}
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 32)
		if err != nil || parsed <= 0 {
			return nil, fmt.Errorf("entry %q must use a positive float value", item)
		}
		out[key] = float32(parsed)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func parsePositiveIntList(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := []int{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		value, err := strconv.Atoi(item)
		if err != nil || value <= 0 {
			return nil, fmt.Errorf("entry %q must be a positive integer", item)
		}
		out = append(out, value)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func validateTurboQuantPrefixBitsFlag(bits []int) error {
	for _, bitWidth := range bits {
		if bitWidth < 2 || bitWidth > 8 {
			return fmt.Errorf("turboquant-prefix-bits must be in supported range 2..8")
		}
	}
	return nil
}

func flagWasProvided(fs *flag.FlagSet, name string) bool {
	provided := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			provided = true
		}
	})
	return provided
}

// trainEmbedRetrievalGatedSelectMetric is the selection metric identifier
// the retrieval eval gate reports (nDCG@10 over the configured BEIR-style
// held-out set; see EmbeddingEvalMetrics.RetrievalNDCGAt10 and
// evalRankMetric's "retrieval_ndcg" case in runtime/embedding_train_runner.go).
// train-embed auto-selects this metric when --retrieval-eval-dir is set
// without an explicit --select-metric.
const trainEmbedRetrievalGatedSelectMetric = "retrieval_ndcg"

// isRetrievalSelectMetric reports whether metric names one of the retrieval
// eval gate's selection metrics (nDCG@10, MAP@100, or recall@100 over a
// --retrieval-eval-dir held-out set). It mirrors the accepted spellings in
// runtime's canonicalTrainSelectionMetric/validTrainSelectionMetric.
func isRetrievalSelectMetric(metric string) bool {
	switch strings.TrimSpace(metric) {
	case "retrieval_ndcg", "retrieval_ndcg_at_10", "retrieval_map", "retrieval_map_at_100", "retrieval_recall", "retrieval_recall_at_100":
		return true
	default:
		return false
	}
}

// activeRetrievalOrientedTrainMode returns the flag name of the enabled
// retrieval-oriented training mode, or "" when none are enabled. train-embed
// enforces at most one of these being set.
func activeRetrievalOrientedTrainMode(hardNegativeTrain, scoreSpectrumTrain, listwiseGeometryTrain, vectorDistillTrain bool) string {
	switch {
	case hardNegativeTrain:
		return "--hard-negative-train"
	case scoreSpectrumTrain:
		return "--score-spectrum-train"
	case listwiseGeometryTrain:
		return "--listwise-geometry-train"
	case vectorDistillTrain:
		return "--vector-distill-train"
	default:
		return ""
	}
}

// restoreBestPairwiseSelectionWarning renders a loud, multi-line stderr
// warning for --restore-best runs of retrieval-oriented training modes that
// select checkpoints on a pairwise/non-retrieval metric without a
// --retrieval-eval-dir gate. This guards the recorded failure mode (commit
// 2861ac9's diagnosis): restore-best with pairwise selection has previously
// restored untrained step-0 checkpoints (2026-06-09, 2026-06-29). It does not
// fail the run — existing scripts must keep working.
func restoreBestPairwiseSelectionWarning(mode, selectMetric string) string {
	return fmt.Sprintf("\n"+
		"########################################################################\n"+
		"# WARNING: %s is training with --restore-best enabled and\n"+
		"# --select-metric=%s (a pairwise/non-retrieval metric), and no\n"+
		"# --retrieval-eval-dir is set.\n"+
		"#\n"+
		"# restore-best with pairwise selection has previously restored untrained\n"+
		"# step-0 checkpoints; pass --retrieval-eval-dir to gate selection on\n"+
		"# retrieval.\n"+
		"########################################################################\n\n",
		mode, selectMetric)
}

func parsePositiveFloatList(raw string) ([]float32, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := []float32{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		value, err := strconv.ParseFloat(item, 32)
		if err != nil || value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("entry %q must be a finite positive float", item)
		}
		out = append(out, float32(value))
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func parseNonNegativeFloatMap(raw string) (map[string]float32, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	out := map[string]float32{}
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		key, value, ok := strings.Cut(item, "=")
		if !ok {
			return nil, fmt.Errorf("entry %q must be source=value", item)
		}
		key = strings.ToLower(strings.TrimSpace(key))
		if key == "" {
			return nil, fmt.Errorf("entry %q has an empty source", item)
		}
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 32)
		if err != nil || parsed < 0 || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
			return nil, fmt.Errorf("entry %q must use a non-negative float value", item)
		}
		out[key] = float32(parsed)
	}
	if len(out) == 0 {
		return nil, nil
	}
	return out, nil
}

func estimateTrainEmbedWorkload(tokenizerPath, trainPath, evalPath string, cfg eosruntime.EmbeddingTrainRunConfig) (eosruntime.EmbeddingTrainWorkload, error) {
	if cfg.EvalOnly && evalPath == "" && cfg.ScoreSpectrumEvalPath == "" && !cfg.ListwiseGeometryTrain && !cfg.VectorDistillTrain {
		evalPath = trainPath
		trainPath = ""
	}
	if cfg.EvalOnly && cfg.VectorDistillTrain {
		if tokenizerPath == "" {
			return eosruntime.EmbeddingTrainWorkload{}, fmt.Errorf("vector distillation training requires text tokenization; remove --no-tokenizer or set --tokenizer")
		}
		return eosruntime.EstimateVectorDistillTrainWorkload(0, 0, cfg), nil
	}
	if cfg.EvalOnly && cfg.ScoreSpectrumTrain {
		evalCount := 0
		if evalPath != "" {
			if tokenizerPath != "" {
				evalPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount += len(evalPairs)
			} else {
				evalPairs, err := eosruntime.ReadEmbeddingPairExamplesFile(evalPath)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount += len(evalPairs)
			}
		}
		if cfg.ScoreSpectrumEvalPath != "" {
			if tokenizerPath != "" {
				scoreEval, err := eosruntime.ReadEmbeddingTextScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				for _, example := range scoreEval {
					evalCount += len(example.Candidates)
				}
			} else {
				scoreEval, err := eosruntime.ReadEmbeddingScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				for _, example := range scoreEval {
					evalCount += len(example.CandidateTokens)
				}
			}
		}
		return eosruntime.EstimateScoreSpectrumTrainWorkload(nil, evalCount, cfg), nil
	}
	if cfg.EvalOnly && cfg.ListwiseGeometryTrain {
		if tokenizerPath == "" {
			return eosruntime.EmbeddingTrainWorkload{}, fmt.Errorf("listwise geometry training requires text tokenization; remove --no-tokenizer or set --tokenizer")
		}
		if evalPath != "" {
			listwiseEval, err := eosruntime.ReadEmbeddingListwiseGeometryBatchesFile(evalPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			cfg.ListwiseGeometryEval = tokenizedListwiseGeometryWorkloadShape(listwiseEval)
		} else if trainPath != "" {
			listwiseEval, err := eosruntime.ReadEmbeddingListwiseGeometryBatchesFile(trainPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			cfg.ListwiseGeometryEval = tokenizedListwiseGeometryWorkloadShape(listwiseEval)
		}
		return eosruntime.EstimateListwiseGeometryTrainWorkload(nil, 0, cfg), nil
	}
	if tokenizerPath != "" {
		if cfg.EvalOnly {
			evalPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			allPositive := true
			positiveCount := 0
			for _, example := range evalPairs {
				if example.Target > 0 {
					positiveCount++
				} else {
					allPositive = false
				}
			}
			if allPositive {
				return eosruntime.EstimateContrastiveTrainWorkload(0, positiveCount, cfg), nil
			}
			return eosruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		if cfg.PairwiseTrain {
			trainPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(trainPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			return eosruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
		}
		if cfg.ScoreSpectrumTrain {
			trainSet, err := eosruntime.ReadEmbeddingTextScoreSpectrumExamplesFile(trainPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			if cfg.ScoreSpectrumEvalPath != "" {
				scoreEval, err := eosruntime.ReadEmbeddingTextScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				for _, example := range scoreEval {
					evalCount += len(example.Candidates)
				}
			}
			tokenized := make([]eosruntime.EmbeddingScoreSpectrumExample, 0, len(trainSet))
			for _, example := range trainSet {
				tokenized = append(tokenized, eosruntime.EmbeddingScoreSpectrumExample{
					CandidateTokens: make([][]int32, len(example.Candidates)),
				})
			}
			for i, example := range trainSet {
				for j := range example.Candidates {
					tokenized[i].CandidateTokens[j] = []int32{1}
				}
			}
			return eosruntime.EstimateScoreSpectrumTrainWorkload(tokenized, evalCount, cfg), nil
		}
		if cfg.ListwiseGeometryTrain {
			trainSet, err := eosruntime.ReadEmbeddingListwiseGeometryBatchesFile(trainPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			if evalPath != "" {
				listwiseEval, err := eosruntime.ReadEmbeddingListwiseGeometryBatchesFile(evalPath)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				cfg.ListwiseGeometryEval = tokenizedListwiseGeometryWorkloadShape(listwiseEval)
			}
			tokenized := tokenizedListwiseGeometryWorkloadShape(trainSet)
			return eosruntime.EstimateListwiseGeometryTrainWorkload(tokenized, 0, cfg), nil
		}
		if cfg.VectorDistillTrain {
			trainSet, err := eosruntime.ReadEmbeddingVectorDistillExamplesFile(trainPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			return eosruntime.EstimateVectorDistillTrainWorkload(len(trainSet), 0, cfg), nil
		}
		if cfg.HardNegativeTrain {
			trainSet, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(trainPath)
			if err != nil {
				trainPairs, pairErr := eosruntime.ReadEmbeddingTextPairExamplesFile(trainPath)
				if pairErr != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				trainSet, err = eosruntime.BuildEmbeddingTextHardNegativeExamplesFromPairs(trainPairs, cfg.HardNegativesPerQuery)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
			}
			evalCount := 0
			if evalPath != "" {
				evalPairs, err := eosruntime.ReadEmbeddingTextHardNegativeEvalPairsFile(evalPath, cfg.HardNegativesPerQuery)
				if err != nil {
					return eosruntime.EmbeddingTrainWorkload{}, err
				}
				evalCount = len(evalPairs)
			}
			return eosruntime.EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, evalCount, cfg), nil
		}
		trainSet, err := eosruntime.ReadEmbeddingTextContrastiveExamplesFile(trainPath)
		if err != nil {
			return eosruntime.EmbeddingTrainWorkload{}, err
		}
		if evalPath == "" {
			return eosruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg), nil
		}
		evalPairs, err := eosruntime.ReadEmbeddingTextPairExamplesFile(evalPath)
		if err != nil {
			return eosruntime.EmbeddingTrainWorkload{}, err
		}
		allPositive := true
		positiveCount := 0
		for _, example := range evalPairs {
			if example.Target > 0 {
				positiveCount++
			} else {
				allPositive = false
			}
		}
		if allPositive {
			return eosruntime.EstimateContrastiveTrainWorkload(len(trainSet), positiveCount, cfg), nil
		}
		workload := eosruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
		workload = eosruntime.RetargetWorkloadToPairwiseEval(workload, len(evalPairs), cfg)
		return workload, nil
	}

	if cfg.EvalOnly {
		evalSet, err := eosruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := eosruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			return eosruntime.EstimatePairwiseTrainWorkload(0, len(evalPairs), cfg), nil
		}
		return eosruntime.EstimateContrastiveTrainWorkload(0, len(evalSet), cfg), nil
	}
	if cfg.PairwiseTrain {
		trainPairs, err := eosruntime.ReadEmbeddingPairExamplesFile(trainPath)
		if err != nil {
			return eosruntime.EmbeddingTrainWorkload{}, err
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := eosruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		return eosruntime.EstimatePairwiseTrainWorkload(len(trainPairs), evalCount, cfg), nil
	}
	if cfg.ScoreSpectrumTrain {
		trainSet, err := eosruntime.ReadEmbeddingScoreSpectrumExamplesFile(trainPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
		if err != nil {
			return eosruntime.EmbeddingTrainWorkload{}, err
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := eosruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		if cfg.ScoreSpectrumEvalPath != "" {
			scoreEval, err := eosruntime.ReadEmbeddingScoreSpectrumExamplesFile(cfg.ScoreSpectrumEvalPath, eosruntime.EmbeddingScoreSpectrumReadOptions{AllowResearchOnly: cfg.AllowResearchOnlyScoreSpectrum})
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			for _, example := range scoreEval {
				evalCount += len(example.CandidateTokens)
			}
		}
		return eosruntime.EstimateScoreSpectrumTrainWorkload(trainSet, evalCount, cfg), nil
	}
	if cfg.ListwiseGeometryTrain {
		return eosruntime.EmbeddingTrainWorkload{}, fmt.Errorf("listwise geometry training requires text tokenization; remove --no-tokenizer or set --tokenizer")
	}
	if cfg.VectorDistillTrain {
		return eosruntime.EmbeddingTrainWorkload{}, fmt.Errorf("vector distillation training requires text tokenization; remove --no-tokenizer or set --tokenizer")
	}
	if cfg.HardNegativeTrain {
		trainSet, err := eosruntime.ReadEmbeddingHardNegativeExamplesFile(trainPath)
		if err != nil {
			trainPairs, pairErr := eosruntime.ReadEmbeddingPairExamplesFile(trainPath)
			if pairErr != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			trainSet, err = eosruntime.BuildEmbeddingHardNegativeExamplesFromPairs(trainPairs, cfg.HardNegativesPerQuery)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
		}
		evalCount := 0
		if evalPath != "" {
			evalPairs, err := eosruntime.ReadEmbeddingHardNegativeEvalPairsFile(evalPath, cfg.HardNegativesPerQuery)
			if err != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			evalCount = len(evalPairs)
		}
		return eosruntime.EstimateHardNegativeTrainWorkload(len(trainSet), cfg.HardNegativesPerQuery, evalCount, cfg), nil
	}
	trainSet, err := eosruntime.ReadEmbeddingContrastiveExamplesFile(trainPath)
	if err != nil {
		return eosruntime.EmbeddingTrainWorkload{}, err
	}
	evalCount := 0
	if evalPath != "" {
		evalSet, err := eosruntime.ReadEmbeddingContrastiveExamplesFile(evalPath)
		if err != nil {
			evalPairs, pairErr := eosruntime.ReadEmbeddingPairExamplesFile(evalPath)
			if pairErr != nil {
				return eosruntime.EmbeddingTrainWorkload{}, err
			}
			workload := eosruntime.EstimateContrastiveTrainWorkload(len(trainSet), 0, cfg)
			workload.EvalMode = "pairwise"
			workload.EvalExamples = len(evalPairs)
			workload.EvalPairsPerPass = int64(len(evalPairs))
			workload.PlannedEvalPasses = 1
			workload.PlannedEvalPairs = int64(len(evalPairs))
			workload.PlannedTotalPairs = workload.PlannedTrainPairs + workload.PlannedEvalPairs
			return workload, nil
		}
		evalCount = len(evalSet)
	}
	return eosruntime.EstimateContrastiveTrainWorkload(len(trainSet), evalCount, cfg), nil
}

func tokenizedListwiseGeometryWorkloadShape(batches []eosruntime.EmbeddingListwiseGeometryBatch) []eosruntime.EmbeddingTokenizedListwiseGeometryBatch {
	tokenized := make([]eosruntime.EmbeddingTokenizedListwiseGeometryBatch, 0, len(batches))
	for _, batch := range batches {
		tokenized = append(tokenized, eosruntime.EmbeddingTokenizedListwiseGeometryBatch{
			QueryTokens:       make([][]int32, len(batch.Queries)),
			DocumentTokens:    make([][]int32, len(batch.Documents)),
			TeacherSimilarity: batch.TeacherSimilarity,
		})
	}
	for i := range tokenized {
		for j := range tokenized[i].QueryTokens {
			tokenized[i].QueryTokens[j] = []int32{1}
		}
		for j := range tokenized[i].DocumentTokens {
			tokenized[i].DocumentTokens[j] = []int32{1}
		}
	}
	return tokenized
}

func formatTrainWorkload(workload eosruntime.EmbeddingTrainWorkload) string {
	parts := []string{
		fmt.Sprintf("train=%d %s examples", workload.TrainExamples, workload.TrainMode),
		fmt.Sprintf("batch=%d", workload.BatchSize),
		fmt.Sprintf("steps/epoch=%d", workload.TrainBatchesPerEpoch),
		fmt.Sprintf("train_pairs/epoch=%d", workload.TrainPairsPerEpoch),
	}
	if workload.EvalMode != "" {
		parts = append(parts,
			fmt.Sprintf("eval=%d %s examples", workload.EvalExamples, workload.EvalMode),
			fmt.Sprintf("eval_pairs/pass=%d", workload.EvalPairsPerPass),
			fmt.Sprintf("eval_passes(planned=%d actual=%d)", workload.PlannedEvalPasses, workload.ActualEvalPasses),
		)
	}
	parts = append(parts,
		fmt.Sprintf("pairs(planned=%d actual=%d)", workload.PlannedTotalPairs, workload.ActualTotalPairs),
	)
	return strings.Join(parts, " ")
}

func validateTrainEmbedListwiseGeometryWorkload(workload eosruntime.EmbeddingTrainWorkload, cfg eosruntime.EmbeddingTrainRunConfig) error {
	if workload.TrainMode != "listwise_geometry" && workload.EvalMode != "listwise_geometry" && workload.EvalMode != "mixed" {
		return nil
	}
	if cfg.MaxListwiseGeometryTrainPairs > 0 && workload.TrainPairsPerEpoch > cfg.MaxListwiseGeometryTrainPairs {
		return fmt.Errorf("planned listwise geometry train_pairs/epoch %d exceeds --max-listwise-train-pairs %d; pass a smaller split or set --max-listwise-train-pairs=0 to allow this research workload", workload.TrainPairsPerEpoch, cfg.MaxListwiseGeometryTrainPairs)
	}
	if cfg.MaxListwiseGeometryEvalPairs > 0 && workload.EvalPairsPerPass > cfg.MaxListwiseGeometryEvalPairs {
		return fmt.Errorf("planned listwise geometry eval_pairs/pass %d exceeds --max-listwise-eval-pairs %d; pass a smaller eval split or set --max-listwise-eval-pairs=0 to allow this research workload", workload.EvalPairsPerPass, cfg.MaxListwiseGeometryEvalPairs)
	}
	return nil
}

func formatTrainThroughput(summary eosruntime.EmbeddingTrainRunSummary) string {
	parts := []string{fmt.Sprintf("elapsed=%s", summary.Elapsed.Round(time.Millisecond))}
	if rate := itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed); rate > 0 {
		parts = append(parts, fmt.Sprintf("pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("train_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_examples/s=%.2f", rate))
	}
	if rate := pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("eval_pairs/s=%.2f", rate))
	}
	if rate := itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration); rate > 0 {
		parts = append(parts, fmt.Sprintf("optimizer_steps/s=%.2f", rate))
	}
	return strings.Join(parts, " ")
}

func printTrainProgress(progress eosruntime.EmbeddingTrainProgress) {
	phase := progress.Phase
	if phase == "" {
		phase = "train"
	}
	if phase == "eval_start" || phase == "eval_done" {
		fmt.Printf(
			"progress: phase=%s epoch=%d step=%d eval_pass=%d eval_examples=%d eval_pairs=%d elapsed=%s\n",
			phase,
			progress.Epoch,
			progress.Step,
			progress.EvalPass,
			progress.EvalExamples,
			progress.EvalPairs,
			progress.Elapsed.Round(time.Millisecond),
		)
		return
	}
	epochPairs := fmt.Sprintf("%d", progress.EpochTrainPairs)
	if progress.PlannedEpochPairs > 0 {
		epochPairs = fmt.Sprintf("%d/%d", progress.EpochTrainPairs, progress.PlannedEpochPairs)
	}
	fmt.Printf(
		"progress: phase=%s epoch=%d batch=%d/%d step=%d loss=%.6f avg_score=%.6f batch_examples=%d batch_pairs=%d epoch_examples=%d epoch_pairs=%s elapsed=%s\n",
		phase,
		progress.Epoch,
		progress.Batch,
		progress.Batches,
		progress.Step,
		progress.Loss,
		progress.AverageScore,
		progress.BatchExamples,
		progress.BatchPairs,
		progress.EpochTrainExamples,
		epochPairs,
		progress.Elapsed.Round(time.Millisecond),
	)
}

func itemsPerSecond(items int64, elapsed time.Duration) float64 {
	if items <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(items) / elapsed.Seconds()
}

func pairsPerSecond(pairs int64, elapsed time.Duration) float64 {
	if pairs <= 0 || elapsed <= 0 {
		return 0
	}
	return float64(pairs) / elapsed.Seconds()
}

func runTrainCorpus(args []string) error {
	fs := flag.NewFlagSet("train-corpus", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var epochs int
	var batchSize int
	var shuffle bool
	var seed int64
	var evalEvery int
	var evalEverySteps int
	var patience int
	var selectMetric string
	var minDelta float64
	var restoreBest bool
	var lengthBucketBatches bool
	var progressEvery int
	var tokenizerPath string
	var vocabSize int
	var minFreq int
	var trainPairsPath string
	var evalPairsPath string
	var minChars int
	var maxPairs int
	var evalPairs int
	var metricsJSONPath string
	var learningRate float64
	var contrastiveLoss string
	var temperature float64
	var groupedLossWeight float64
	var teacherLossWeight float64
	var teacherTemperature float64
	var matryoshkaDims string
	var matryoshkaWeights string
	var clearTurboQuantPrefix bool
	var turboQuantPrefixBits string
	var turboQuantPrefixObjectives string
	var turboQuantPrefixWeight float64
	var turboQuantPrefixSeed int64
	var turboQuantPrefixScoreMode string
	var turboQuantCompactObjectives string
	fs.IntVar(&epochs, "epochs", 10, "number of epochs")
	fs.IntVar(&batchSize, "batch-size", 8, "batch size")
	fs.BoolVar(&shuffle, "shuffle", true, "shuffle training set each epoch")
	fs.Int64Var(&seed, "seed", 1, "shuffle/mining seed")
	fs.IntVar(&evalEvery, "eval-every", 1, "evaluate every N epochs")
	fs.IntVar(&evalEverySteps, "eval-every-steps", 0, "evaluate every N optimizer steps within an epoch (0 disables)")
	fs.IntVar(&patience, "patience", 3, "early stopping patience in evals")
	fs.StringVar(&selectMetric, "select-metric", "top1_accuracy", "selection metric: top1_accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, score_margin, pair_accuracy, threshold_accuracy, auc, loss, retrieval_ndcg(_at_10), retrieval_map(_at_100), or retrieval_recall(_at_100)")
	fs.Float64Var(&minDelta, "min-delta", 0, "minimum eval improvement to count as better")
	fs.BoolVar(&restoreBest, "restore-best", true, "restore best checkpoint at end")
	fs.BoolVar(&lengthBucketBatches, "length-bucket-batches", true, "cluster contrastive batches by token length to improve batched GPU training")
	fs.IntVar(&progressEvery, "progress-every", 0, "print training progress every N optimizer steps (0 disables)")
	fs.StringVar(&tokenizerPath, "tokenizer", "", "output tokenizer path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for tokenizer merges")
	fs.StringVar(&trainPairsPath, "train-pairs", "", "output mined train pair dataset path")
	fs.StringVar(&evalPairsPath, "eval-pairs-path", "", "output mined eval pair dataset path")
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.StringVar(&metricsJSONPath, "metrics-json", "", "write machine-readable run metrics JSON to this path")
	fs.Float64Var(&learningRate, "lr", 0, "override package learning rate for this run")
	fs.StringVar(&contrastiveLoss, "contrastive-loss", "", "override package contrastive loss: pair_mse, infonce, grouped_infonce, or hybrid_infonce")
	fs.Float64Var(&temperature, "temperature", 0, "override package contrastive softmax temperature")
	fs.Float64Var(&groupedLossWeight, "grouped-loss-weight", 0, "grouped hard-negative loss weight for hybrid_infonce")
	fs.Float64Var(&teacherLossWeight, "teacher-loss-weight", 0, "teacher score distillation weight for hard-negative training")
	fs.Float64Var(&teacherTemperature, "teacher-temperature", 0, "teacher score softmax temperature for hard-negative distillation")
	fs.StringVar(&matryoshkaDims, "matryoshka-dims", "", "comma-separated compact prefix dimensions to train with InfoNCE, for example 64,128")
	fs.StringVar(&matryoshkaWeights, "matryoshka-weights", "", "optional comma-separated positive weights matching --matryoshka-dims")
	fs.BoolVar(&clearTurboQuantPrefix, "clear-turboquant-prefix", false, "clear inherited TurboQuant compact-prefix objectives for continuation training")
	fs.StringVar(&turboQuantPrefixBits, "turboquant-prefix-bits", "", "comma-separated TurboQuant bit widths for quantized compact-prefix InfoNCE, supported: 2..8")
	fs.StringVar(&turboQuantPrefixObjectives, "turboquant-prefix-objectives", "", "comma-separated TurboQuant compact-prefix objectives as dim:bit=weight, for example 128:4=0.5")
	fs.Float64Var(&turboQuantPrefixWeight, "turboquant-prefix-weight", 0, "optional weight for each TurboQuant compact-prefix objective (default 1 when bits are set)")
	fs.Int64Var(&turboQuantPrefixSeed, "turboquant-prefix-seed", 0, "TurboQuant compact-prefix quantizer seed (default matches multivector retrieval)")
	fs.StringVar(&turboQuantPrefixScoreMode, "turboquant-prefix-score-mode", "", "TurboQuant compact-prefix score mode: reconstruct_cosine (default) or prepared_ip")
	fs.StringVar(&turboQuantCompactObjectives, "turboquant-compact-objectives", "", "comma-separated hard-negative prepared-IP TurboQuant compact objectives as dim:bit=weight, for example 256:4=0.05")
	if err := fs.Parse(args); err != nil {
		return err
	}
	teacherLossWeightSet := flagWasProvided(fs, "teacher-loss-weight")
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos train-corpus [flags] <artifact.mll> <corpus.txt>")
	}
	if learningRate < 0 {
		return fmt.Errorf("lr must be non-negative")
	}
	if temperature < 0 {
		return fmt.Errorf("temperature must be non-negative")
	}
	if groupedLossWeight < 0 {
		return fmt.Errorf("grouped-loss-weight must be non-negative")
	}
	if teacherLossWeight < 0 {
		return fmt.Errorf("teacher-loss-weight must be non-negative")
	}
	if teacherTemperature < 0 {
		return fmt.Errorf("teacher-temperature must be non-negative")
	}
	if turboQuantPrefixWeight < 0 {
		return fmt.Errorf("turboquant-prefix-weight must be non-negative")
	}
	if progressEvery < 0 {
		return fmt.Errorf("progress-every must be non-negative")
	}
	if evalEverySteps < 0 {
		return fmt.Errorf("eval-every-steps must be non-negative")
	}
	parsedMatryoshkaDims, parseErr := parsePositiveIntList(matryoshkaDims)
	if parseErr != nil {
		return fmt.Errorf("matryoshka-dims: %w", parseErr)
	}
	parsedMatryoshkaWeights, parseErr := parsePositiveFloatList(matryoshkaWeights)
	if parseErr != nil {
		return fmt.Errorf("matryoshka-weights: %w", parseErr)
	}
	parsedTurboQuantPrefixBits, parseErr := parsePositiveIntList(turboQuantPrefixBits)
	if parseErr != nil {
		return fmt.Errorf("turboquant-prefix-bits: %w", parseErr)
	}
	if err := validateTurboQuantPrefixBitsFlag(parsedTurboQuantPrefixBits); err != nil {
		return err
	}
	parsedTurboQuantPrefixObjectives, parseErr := eosruntime.ParseTurboQuantPrefixObjectives(turboQuantPrefixObjectives)
	if parseErr != nil {
		return fmt.Errorf("turboquant-prefix-objectives: %w", parseErr)
	}
	if len(parsedTurboQuantPrefixObjectives) > 0 && len(parsedTurboQuantPrefixBits) > 0 {
		return fmt.Errorf("--turboquant-prefix-objectives is mutually exclusive with --turboquant-prefix-bits")
	}
	if clearTurboQuantPrefix && len(parsedTurboQuantPrefixBits) > 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-bits")
	}
	if clearTurboQuantPrefix && len(parsedTurboQuantPrefixObjectives) > 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-objectives")
	}
	if clearTurboQuantPrefix && turboQuantPrefixWeight != 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-weight")
	}
	if clearTurboQuantPrefix && turboQuantPrefixSeed != 0 {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-seed")
	}
	if clearTurboQuantPrefix && strings.TrimSpace(turboQuantPrefixScoreMode) != "" {
		return fmt.Errorf("--clear-turboquant-prefix is mutually exclusive with --turboquant-prefix-score-mode")
	}
	if len(parsedTurboQuantPrefixObjectives) > 0 && turboQuantPrefixWeight != 0 {
		return fmt.Errorf("--turboquant-prefix-weight must not be set with --turboquant-prefix-objectives")
	}
	parsedTurboQuantCompactObjectives, parseErr := eosruntime.ParseTurboQuantPrefixObjectives(turboQuantCompactObjectives)
	if parseErr != nil {
		return fmt.Errorf("turboquant-compact-objectives: %w", parseErr)
	}
	if len(parsedTurboQuantCompactObjectives) > 0 {
		return fmt.Errorf("--turboquant-compact-objectives requires --hard-negative-train")
	}
	parsedTurboQuantPrefixScoreMode := ""
	if strings.TrimSpace(turboQuantPrefixScoreMode) != "" {
		mode, parseErr := eosruntime.NormalizeTurboQuantPrefixScoreModeForCLI(turboQuantPrefixScoreMode)
		if parseErr != nil {
			return parseErr
		}
		parsedTurboQuantPrefixScoreMode = mode
	}
	path := fs.Arg(0)
	corpusPath := fs.Arg(1)
	runConfig := eosruntime.EmbeddingTrainRunConfig{
		Epochs:                     epochs,
		BatchSize:                  batchSize,
		Shuffle:                    shuffle,
		Seed:                       seed,
		EvalEveryEpoch:             evalEvery,
		EvalEverySteps:             evalEverySteps,
		EarlyStoppingPatience:      patience,
		SelectMetric:               selectMetric,
		MinDelta:                   float32(minDelta),
		RestoreBest:                restoreBest,
		LengthBucketBatches:        lengthBucketBatches,
		LearningRate:               float32(learningRate),
		ContrastiveLoss:            contrastiveLoss,
		Temperature:                float32(temperature),
		GroupedLossWeight:          float32(groupedLossWeight),
		TeacherLossWeight:          float32(teacherLossWeight),
		TeacherLossWeightSet:       teacherLossWeightSet,
		TeacherTemperature:         float32(teacherTemperature),
		MatryoshkaDims:             parsedMatryoshkaDims,
		MatryoshkaWeights:          parsedMatryoshkaWeights,
		ClearTurboQuantPrefix:      clearTurboQuantPrefix,
		TurboQuantPrefixBits:       parsedTurboQuantPrefixBits,
		TurboQuantPrefixObjectives: parsedTurboQuantPrefixObjectives,
		TurboQuantPrefixWeight:     float32(turboQuantPrefixWeight),
		TurboQuantPrefixSeed:       turboQuantPrefixSeed,
		TurboQuantPrefixScoreMode:  parsedTurboQuantPrefixScoreMode,
		ProgressEverySteps:         progressEvery,
	}
	if progressEvery > 0 {
		runConfig.Progress = printTrainProgress
	}
	summary, paths, err := eosruntime.TrainEmbeddingPackageFromCorpusFile(path, corpusPath, eosruntime.EmbeddingCorpusTrainConfig{
		TokenizerPath:      tokenizerPath,
		TokenizerVocabSize: vocabSize,
		TokenizerMinFreq:   minFreq,
		TrainPairsPath:     trainPairsPath,
		EvalPairsPath:      evalPairsPath,
		Mining: eosruntime.EmbeddingTextMiningConfig{
			MinChars:  minChars,
			MaxPairs:  maxPairs,
			EvalPairs: evalPairs,
			Seed:      seed,
		},
		Run: runConfig,
	})
	if err != nil {
		return err
	}
	fmt.Printf("trained package %q from corpus\n", path)
	fmt.Printf("tokenizer: %s\n", paths.TokenizerPath)
	fmt.Printf("train pairs: %s\n", paths.TrainPairsPath)
	if paths.EvalPairsPath != "" {
		fmt.Printf("eval pairs: %s\n", paths.EvalPairsPath)
	}
	fmt.Printf("epochs: %d, steps: %d, run_steps: %d, best_epoch: %d, best_step: %d\n", summary.EpochsCompleted, summary.StepsCompleted, summary.StepsRun, summary.BestEpoch, summary.BestStep)
	fmt.Printf("final train: loss=%.6f avg_score=%.6f batch=%d\n", summary.FinalTrain.Loss, summary.FinalTrain.AverageScore, summary.FinalTrain.BatchSize)
	if summary.FinalEval != nil {
		fmt.Printf("final eval: loss=%.6f margin=%.6f accuracy=%.6f threshold_accuracy=%.6f threshold=%.6f auc=%.6f top1=%.6f top5=%.6f top10=%.6f mrr=%.6f mean_rank=%.3f retrieval_ndcg=%.6f retrieval_map=%.6f retrieval_recall=%.6f pairs=%d\n", summary.FinalEval.Loss, summary.FinalEval.ScoreMargin, summary.FinalEval.PairAccuracy, summary.FinalEval.ThresholdAccuracy, summary.FinalEval.ScoreThreshold, summary.FinalEval.ROCAUC, summary.FinalEval.Top1Accuracy, summary.FinalEval.Top5Accuracy, summary.FinalEval.Top10Accuracy, summary.FinalEval.MeanReciprocalRank, summary.FinalEval.MeanPositiveRank, summary.FinalEval.RetrievalNDCGAt10, summary.FinalEval.RetrievalMAPAt100, summary.FinalEval.RetrievalRecallAt100, summary.FinalEval.PairCount)
		printFinalRetrievalEval(summary.FinalEval)
	}
	fmt.Printf("workload: %s\n", formatTrainWorkload(summary.Workload))
	fmt.Printf("throughput: %s\n", formatTrainThroughput(summary))
	fmt.Printf("accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		displayTrainBackend(summary.EndProfile.ForwardBackend),
		displayTrainBackend(summary.EndProfile.OptimizerBackend),
		displayTrainBackend(summary.EndProfile.ActivationBackend),
		displayTrainBackend(summary.EndProfile.ContrastiveBackend),
	)
	fmt.Printf("profile delta: matmul_bind_calls=%d matmul_runs=%d matmul_run_upload_mb=%.2f matmul_run_download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		summary.DeltaProfile.ForwardResidency.MatMul.BindCalls,
		summary.DeltaProfile.ForwardResidency.MatMul.RunCalls,
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunUploadedBytes)/(1024*1024),
		float64(summary.DeltaProfile.ForwardResidency.MatMul.RunDownloadedBytes)/(1024*1024),
		summary.DeltaProfile.Optimizer.UpdateCalls,
		summary.DeltaProfile.Activation.GELUBackwardCalls+summary.DeltaProfile.Activation.SoftmaxBackwardCalls+summary.DeltaProfile.Activation.LayerNormBackwardCalls,
		summary.DeltaProfile.Contrastive.RunCalls,
	)
	fmt.Printf("checkpoint: %s\n", paths.Package.CheckpointPath)
	fmt.Printf("profile: %s\n", paths.Package.TrainProfilePath)
	if metricsJSONPath != "" {
		extra := map[string]string{
			"tokenizer":   paths.TokenizerPath,
			"train_pairs": paths.TrainPairsPath,
		}
		if paths.EvalPairsPath != "" {
			extra["eval_pairs"] = paths.EvalPairsPath
		}
		if err := writeTrainMetricsJSON(metricsJSONPath, "train-corpus", "train", path, paths.TokenizerPath, summary, paths.Package, extra); err != nil {
			return err
		}
		fmt.Printf("metrics: %s\n", metricsJSONPath)
	}
	return nil
}

type trainMetricsJSON struct {
	Schema                    string                           `json:"schema"`
	Command                   string                           `json:"command"`
	Mode                      string                           `json:"mode"`
	Artifact                  string                           `json:"artifact"`
	Tokenizer                 string                           `json:"tokenizer,omitempty"`
	Summary                   trainRunSummaryJSON              `json:"summary"`
	Config                    trainRunConfigJSON               `json:"config"`
	FinalTrain                trainBatchMetricsJSON            `json:"final_train"`
	LastEval                  *evalMetricsJSON                 `json:"last_eval,omitempty"`
	BestEval                  *evalMetricsJSON                 `json:"best_eval,omitempty"`
	FinalEval                 *evalMetricsJSON                 `json:"final_eval,omitempty"`
	LastScoreSpectrumEval     *scoreSpectrumEvalMetricsJSON    `json:"last_score_spectrum_eval,omitempty"`
	BestScoreSpectrumEval     *scoreSpectrumEvalMetricsJSON    `json:"best_score_spectrum_eval,omitempty"`
	FinalScoreSpectrumEval    *scoreSpectrumEvalMetricsJSON    `json:"final_score_spectrum_eval,omitempty"`
	LastListwiseGeometryEval  *listwiseGeometryEvalMetricsJSON `json:"last_listwise_geometry_eval,omitempty"`
	BestListwiseGeometryEval  *listwiseGeometryEvalMetricsJSON `json:"best_listwise_geometry_eval,omitempty"`
	FinalListwiseGeometryEval *listwiseGeometryEvalMetricsJSON `json:"final_listwise_geometry_eval,omitempty"`
	EvalHistory               []trainEvalSummaryJSON           `json:"eval_history,omitempty"`
	Workload                  trainWorkloadJSON                `json:"workload"`
	Throughput                trainThroughputJSON              `json:"throughput"`
	Accelerators              trainAcceleratorsJSON            `json:"accelerators"`
	ProfileDelta              trainProfileDeltaJSON            `json:"profile_delta"`
	Package                   trainPackagePathsJSON            `json:"package"`
	Artifacts                 map[string]string                `json:"artifacts,omitempty"`
}

type trainRunSummaryJSON struct {
	EpochsCompleted int  `json:"epochs_completed"`
	StepsCompleted  int  `json:"steps_completed"`
	StepsRun        int  `json:"steps_run"`
	BestEpoch       int  `json:"best_epoch"`
	BestStep        int  `json:"best_step"`
	RestoredBest    bool `json:"restored_best"`
	StoppedEarly    bool `json:"stopped_early"`
}

type trainEvalSummaryJSON struct {
	Epoch                int                              `json:"epoch"`
	Step                 int                              `json:"step"`
	EvalPass             int                              `json:"eval_pass"`
	Trigger              string                           `json:"trigger"`
	Improved             bool                             `json:"improved"`
	Eval                 *evalMetricsJSON                 `json:"eval,omitempty"`
	ScoreSpectrumEval    *scoreSpectrumEvalMetricsJSON    `json:"score_spectrum_eval,omitempty"`
	ListwiseGeometryEval *listwiseGeometryEvalMetricsJSON `json:"listwise_geometry_eval,omitempty"`
}

type trainRunConfigJSON struct {
	Epochs                            int                                    `json:"epochs"`
	BatchSize                         int                                    `json:"batch_size"`
	Shuffle                           bool                                   `json:"shuffle"`
	Seed                              int64                                  `json:"seed"`
	EvalEveryEpoch                    int                                    `json:"eval_every_epoch"`
	EvalEverySteps                    int                                    `json:"eval_every_steps"`
	Patience                          int                                    `json:"patience"`
	SelectMetric                      string                                 `json:"select_metric"`
	MinDelta                          float32                                `json:"min_delta"`
	RestoreBest                       bool                                   `json:"restore_best"`
	LengthBucketBatches               bool                                   `json:"length_bucket_batches"`
	LearningRate                      float32                                `json:"learning_rate"`
	EffectiveLearningRate             float32                                `json:"effective_learning_rate"`
	ContrastiveLoss                   string                                 `json:"contrastive_loss,omitempty"`
	Temperature                       float32                                `json:"temperature"`
	GroupedLossWeight                 float32                                `json:"grouped_loss_weight,omitempty"`
	TeacherLossWeight                 float32                                `json:"teacher_loss_weight,omitempty"`
	TeacherTemperature                float32                                `json:"teacher_temperature,omitempty"`
	TeacherSourceTemperatures         map[string]float32                     `json:"teacher_source_temperatures,omitempty"`
	TeacherSourceWeights              map[string]float32                     `json:"teacher_source_weights,omitempty"`
	TeacherScoreNormalization         string                                 `json:"teacher_score_normalization,omitempty"`
	MatryoshkaDims                    []int                                  `json:"matryoshka_dims,omitempty"`
	MatryoshkaWeights                 []float32                              `json:"matryoshka_weights,omitempty"`
	TurboQuantPrefixBits              []int                                  `json:"turboquant_prefix_bits,omitempty"`
	TurboQuantPrefixObjectives        []eosruntime.TurboQuantPrefixObjective `json:"turboquant_prefix_objectives,omitempty"`
	TurboQuantPrefixWeight            float32                                `json:"turboquant_prefix_weight,omitempty"`
	TurboQuantPrefixSeed              int64                                  `json:"turboquant_prefix_seed,omitempty"`
	TurboQuantPrefixScoreMode         string                                 `json:"turboquant_prefix_score_mode,omitempty"`
	TurboQuantCompactObjectives       []eosruntime.TurboQuantPrefixObjective `json:"turboquant_compact_objectives,omitempty"`
	ClearTurboQuantRankMargin         bool                                   `json:"clear_turboquant_rank_margin,omitempty"`
	TurboQuantRankMarginObjectives    []eosruntime.TurboQuantPrefixObjective `json:"turboquant_rank_margin_objectives,omitempty"`
	TurboQuantRankMargin              float32                                `json:"turboquant_rank_margin,omitempty"`
	ProgressEverySteps                int                                    `json:"progress_every_steps"`
	EvalOnly                          bool                                   `json:"eval_only"`
	PairwiseTrain                     bool                                   `json:"pairwise_train"`
	HardNegativeTrain                 bool                                   `json:"hard_negative_train"`
	ScoreSpectrumTrain                bool                                   `json:"score_spectrum_train"`
	ListwiseGeometryTrain             bool                                   `json:"listwise_geometry_train"`
	VectorDistillTrain                bool                                   `json:"vector_distill_train,omitempty"`
	MovementDiagnostics               bool                                   `json:"movement_diagnostics"`
	AllowResearchOnlyListwiseGeometry bool                                   `json:"allow_research_only_listwise_geometry,omitempty"`
	AllowResearchOnlyVectorDistill    bool                                   `json:"allow_research_only_vector_distill,omitempty"`
	MaxListwiseGeometryTrainPairs     int64                                  `json:"max_listwise_geometry_train_pairs,omitempty"`
	MaxListwiseGeometryEvalPairs      int64                                  `json:"max_listwise_geometry_eval_pairs,omitempty"`
	ScoreSpectrumEvalPath             string                                 `json:"score_spectrum_eval_path,omitempty"`
	ScoreSpectrumLossMode             string                                 `json:"score_spectrum_loss_mode,omitempty"`
	ScoreSpectrumRecoveryWeight       float32                                `json:"score_spectrum_recovery_weight,omitempty"`
	ScoreSpectrumRecoveryMargin       float32                                `json:"score_spectrum_recovery_margin,omitempty"`
	ScoreSpectrumRecoveryTopK         int                                    `json:"score_spectrum_recovery_top_k,omitempty"`
	ScoreSpectrumRecoveryTau          float32                                `json:"score_spectrum_recovery_tau,omitempty"`
	HardNegativesPerQuery             int                                    `json:"hard_negatives_per_query"`
	HardNegativeSourceWeights         map[string]int                         `json:"hard_negative_source_weights,omitempty"`
}

type trainBatchMetricsJSON struct {
	Loss         float32                   `json:"loss"`
	AverageScore float32                   `json:"average_score"`
	BatchSize    int                       `json:"batch_size"`
	Movement     *trainMovementMetricsJSON `json:"movement,omitempty"`
}

type trainMovementMetricsJSON struct {
	Gradient       trainStatMetricsJSON `json:"gradient"`
	ParameterDelta trainStatMetricsJSON `json:"parameter_delta"`
}

type trainStatMetricsJSON struct {
	L2Norm       float32 `json:"l2_norm"`
	MaxAbs       float32 `json:"max_abs"`
	NonzeroCount int     `json:"nonzero_count"`
	TotalCount   int     `json:"total_count"`
}

type evalMetricsJSON struct {
	Loss                 float32                          `json:"loss"`
	AverageScore         float32                          `json:"average_score"`
	PositiveMeanScore    float32                          `json:"positive_mean_score"`
	NegativeMeanScore    float32                          `json:"negative_mean_score"`
	PairAccuracy         float32                          `json:"pair_accuracy"`
	ThresholdAccuracy    float32                          `json:"threshold_accuracy"`
	ScoreThreshold       float32                          `json:"score_threshold"`
	ROCAUC               float32                          `json:"roc_auc"`
	ScoreMargin          float32                          `json:"score_margin"`
	Top1Accuracy         float32                          `json:"top1_accuracy"`
	Top5Accuracy         float32                          `json:"top5_accuracy"`
	Top10Accuracy        float32                          `json:"top10_accuracy"`
	MeanReciprocalRank   float32                          `json:"mean_reciprocal_rank"`
	MeanPositiveRank     float32                          `json:"mean_positive_rank"`
	RetrievalNDCGAt10    float32                          `json:"retrieval_ndcg_at_10"`
	RetrievalMAPAt100    float32                          `json:"retrieval_map_at_100"`
	RetrievalRecallAt100 float32                          `json:"retrieval_recall_at_100"`
	RetrievalEval        *eosruntime.RetrievalEvalMetrics `json:"retrieval_eval,omitempty"`
	PairCount            int                              `json:"pair_count"`
	PositiveCount        int                              `json:"positive_count"`
	NegativeCount        int                              `json:"negative_count"`
}

type scoreSpectrumEvalMetricsJSON struct {
	Loss                              float32 `json:"loss"`
	AverageScore                      float32 `json:"average_score"`
	AnyPositiveTop1                   float32 `json:"score_spectrum_any_positive_top1"`
	OriginalPositiveTop1              float32 `json:"score_spectrum_original_positive_top1"`
	AlternateRelevantRecovery         float32 `json:"score_spectrum_alternate_recovery"`
	BestPositiveHardestNegativeMargin float32 `json:"score_spectrum_best_positive_hardest_negative_margin"`
	TargetCrossEntropy                float32 `json:"target_cross_entropy"`
	TargetKL                          float32 `json:"target_kl"`
	RowCount                          int     `json:"row_count"`
	CandidateCount                    int     `json:"candidate_count"`
	AnyPositiveRowCount               int     `json:"any_positive_row_count"`
	OriginalPositiveRowCount          int     `json:"original_positive_row_count"`
	AlternateRecoveryRowCount         int     `json:"alternate_recovery_row_count"`
	MarginRowCount                    int     `json:"margin_row_count"`
	TargetDistributionRowCount        int     `json:"target_distribution_row_count"`
}

type listwiseGeometryEvalMetricsJSON struct {
	Loss                  float32 `json:"loss"`
	AverageScore          float32 `json:"average_score"`
	TeacherCrossEntropy   float32 `json:"teacher_cross_entropy"`
	TeacherKL             float32 `json:"teacher_kl"`
	TeacherTop1Agreement  float32 `json:"teacher_top1_agreement"`
	AnyPositiveTop1       float32 `json:"any_positive_top1"`
	QueryCount            int     `json:"query_count"`
	DocumentCellCount     int     `json:"document_cell_count"`
	BatchCount            int     `json:"batch_count"`
	AnyPositiveQueryCount int     `json:"any_positive_query_count"`
}

type trainWorkloadJSON struct {
	TrainMode            string `json:"train_mode"`
	EvalMode             string `json:"eval_mode,omitempty"`
	TrainExamples        int    `json:"train_examples"`
	EvalExamples         int    `json:"eval_examples"`
	BatchSize            int    `json:"batch_size"`
	PlannedEpochs        int    `json:"planned_epochs"`
	CompletedEpochs      int    `json:"completed_epochs"`
	TrainBatchesPerEpoch int    `json:"train_batches_per_epoch"`
	TrainPairsPerEpoch   int64  `json:"train_pairs_per_epoch"`
	EvalPairsPerPass     int64  `json:"eval_pairs_per_pass"`
	PlannedEvalPasses    int    `json:"planned_eval_passes"`
	ActualEvalPasses     int    `json:"actual_eval_passes"`
	PlannedTrainPairs    int64  `json:"planned_train_pairs"`
	ActualTrainPairs     int64  `json:"actual_train_pairs"`
	ActualTrainExamples  int64  `json:"actual_train_examples"`
	PlannedEvalPairs     int64  `json:"planned_eval_pairs"`
	ActualEvalPairs      int64  `json:"actual_eval_pairs"`
	ActualEvalExamples   int64  `json:"actual_eval_examples"`
	PlannedTotalPairs    int64  `json:"planned_total_pairs"`
	ActualTotalPairs     int64  `json:"actual_total_pairs"`
	ActualTotalExamples  int64  `json:"actual_total_examples"`
}

type trainThroughputJSON struct {
	ElapsedSeconds          float64 `json:"elapsed_seconds"`
	TrainSeconds            float64 `json:"train_seconds"`
	EvalSeconds             float64 `json:"eval_seconds"`
	ExamplesPerSecond       float64 `json:"examples_per_second"`
	PairsPerSecond          float64 `json:"pairs_per_second"`
	TrainExamplesPerSecond  float64 `json:"train_examples_per_second"`
	TrainPairsPerSecond     float64 `json:"train_pairs_per_second"`
	EvalExamplesPerSecond   float64 `json:"eval_examples_per_second"`
	EvalPairsPerSecond      float64 `json:"eval_pairs_per_second"`
	OptimizerStepsPerSecond float64 `json:"optimizer_steps_per_second"`
}

type trainAcceleratorsJSON struct {
	Forward     string `json:"forward"`
	Optimizer   string `json:"optimizer"`
	Activation  string `json:"activation"`
	Contrastive string `json:"contrastive"`
}

type trainProfileDeltaJSON struct {
	MatMulBindCalls     int64   `json:"matmul_bind_calls"`
	MatMulRuns          int64   `json:"matmul_runs"`
	MatMulRunUploadMB   float64 `json:"matmul_run_upload_mb"`
	MatMulRunDownloadMB float64 `json:"matmul_run_download_mb"`
	OptimizerUpdates    int64   `json:"optimizer_updates"`
	OptimizerSyncs      int64   `json:"optimizer_syncs"`
	ActivationCalls     int64   `json:"activation_calls"`
	ContrastiveCalls    int64   `json:"contrastive_calls"`
}

type trainPackagePathsJSON struct {
	Artifact     string `json:"artifact"`
	Checkpoint   string `json:"checkpoint"`
	TrainProfile string `json:"train_profile"`
	Manifest     string `json:"manifest,omitempty"`
	Weights      string `json:"weights,omitempty"`
	MemoryPlan   string `json:"memory_plan,omitempty"`
	Package      string `json:"package,omitempty"`
}

func writeTrainMetricsJSON(outputPath, command, mode, artifactPath, tokenizerPath string, summary eosruntime.EmbeddingTrainRunSummary, paths eosruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) error {
	payload := trainMetricsPayload(command, mode, artifactPath, tokenizerPath, summary, paths, extraArtifacts)
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("encode metrics JSON: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(outputPath, data, 0o644); err != nil {
		return fmt.Errorf("write metrics JSON %q: %w", outputPath, err)
	}
	return nil
}

func trainMetricsPayload(command, mode, artifactPath, tokenizerPath string, summary eosruntime.EmbeddingTrainRunSummary, paths eosruntime.EmbeddingTrainPackagePaths, extraArtifacts map[string]string) trainMetricsJSON {
	return trainMetricsJSON{
		Schema:                    "manta.embedding_train_metrics.v1",
		Command:                   command,
		Mode:                      mode,
		Artifact:                  artifactPath,
		Tokenizer:                 tokenizerPath,
		Summary:                   trainRunSummaryPayload(summary),
		Config:                    trainRunConfigPayload(summary.Config, summary.EffectiveLearningRate),
		FinalTrain:                trainBatchMetricsPayload(summary.FinalTrain),
		LastEval:                  evalMetricsPayload(summary.LastEval),
		BestEval:                  evalMetricsPayload(summary.BestEval),
		FinalEval:                 evalMetricsPayload(summary.FinalEval),
		LastScoreSpectrumEval:     scoreSpectrumEvalMetricsPayload(summary.LastScoreSpectrumEval),
		BestScoreSpectrumEval:     scoreSpectrumEvalMetricsPayload(summary.BestScoreSpectrumEval),
		FinalScoreSpectrumEval:    scoreSpectrumEvalMetricsPayload(summary.FinalScoreSpectrumEval),
		LastListwiseGeometryEval:  listwiseGeometryEvalMetricsPayload(summary.LastListwiseGeometryEval),
		BestListwiseGeometryEval:  listwiseGeometryEvalMetricsPayload(summary.BestListwiseGeometryEval),
		FinalListwiseGeometryEval: listwiseGeometryEvalMetricsPayload(summary.FinalListwiseGeometryEval),
		EvalHistory:               trainEvalHistoryPayload(summary.EvalHistory),
		Workload:                  trainWorkloadPayload(summary.Workload),
		Throughput:                trainThroughputPayload(summary),
		Accelerators:              trainAcceleratorsPayload(summary.EndProfile),
		ProfileDelta:              trainProfileDeltaPayload(summary.DeltaProfile),
		Package:                   trainPackagePathsPayload(paths),
		Artifacts:                 extraArtifacts,
	}
}

func trainRunSummaryPayload(summary eosruntime.EmbeddingTrainRunSummary) trainRunSummaryJSON {
	return trainRunSummaryJSON{
		EpochsCompleted: summary.EpochsCompleted,
		StepsCompleted:  summary.StepsCompleted,
		StepsRun:        summary.StepsRun,
		BestEpoch:       summary.BestEpoch,
		BestStep:        summary.BestStep,
		RestoredBest:    summary.RestoredBest,
		StoppedEarly:    summary.StoppedEarly,
	}
}

func trainEvalHistoryPayload(history []eosruntime.EmbeddingTrainEvalSummary) []trainEvalSummaryJSON {
	if len(history) == 0 {
		return nil
	}
	payload := make([]trainEvalSummaryJSON, 0, len(history))
	for _, record := range history {
		payload = append(payload, trainEvalSummaryJSON{
			Epoch:                record.Epoch,
			Step:                 record.Step,
			EvalPass:             record.EvalPass,
			Trigger:              record.Trigger,
			Improved:             record.Improved,
			Eval:                 evalMetricsPayload(record.Eval),
			ScoreSpectrumEval:    scoreSpectrumEvalMetricsPayload(record.ScoreSpectrumEval),
			ListwiseGeometryEval: listwiseGeometryEvalMetricsPayload(record.ListwiseGeometryEval),
		})
	}
	return payload
}

func trainRunConfigPayload(cfg eosruntime.EmbeddingTrainRunConfig, effectiveLearningRate float32) trainRunConfigJSON {
	return trainRunConfigJSON{
		Epochs:                            cfg.Epochs,
		BatchSize:                         cfg.BatchSize,
		Shuffle:                           cfg.Shuffle,
		Seed:                              cfg.Seed,
		EvalEveryEpoch:                    cfg.EvalEveryEpoch,
		EvalEverySteps:                    cfg.EvalEverySteps,
		Patience:                          cfg.EarlyStoppingPatience,
		SelectMetric:                      cfg.SelectMetric,
		MinDelta:                          cfg.MinDelta,
		RestoreBest:                       cfg.RestoreBest,
		LengthBucketBatches:               cfg.LengthBucketBatches,
		LearningRate:                      cfg.LearningRate,
		EffectiveLearningRate:             effectiveLearningRate,
		ContrastiveLoss:                   cfg.ContrastiveLoss,
		Temperature:                       cfg.Temperature,
		GroupedLossWeight:                 cfg.GroupedLossWeight,
		TeacherLossWeight:                 cfg.TeacherLossWeight,
		TeacherTemperature:                cfg.TeacherTemperature,
		TeacherSourceTemperatures:         cfg.TeacherSourceTemperatures,
		TeacherSourceWeights:              cfg.TeacherSourceWeights,
		TeacherScoreNormalization:         cfg.TeacherScoreNormalization,
		MatryoshkaDims:                    cfg.MatryoshkaDims,
		MatryoshkaWeights:                 cfg.MatryoshkaWeights,
		TurboQuantPrefixBits:              cfg.TurboQuantPrefixBits,
		TurboQuantPrefixObjectives:        cfg.TurboQuantPrefixObjectives,
		TurboQuantPrefixWeight:            cfg.TurboQuantPrefixWeight,
		TurboQuantPrefixSeed:              cfg.TurboQuantPrefixSeed,
		TurboQuantPrefixScoreMode:         cfg.TurboQuantPrefixScoreMode,
		TurboQuantCompactObjectives:       cfg.TurboQuantCompactObjectives,
		ClearTurboQuantRankMargin:         cfg.ClearTurboQuantRankMargin,
		TurboQuantRankMarginObjectives:    cfg.TurboQuantRankMarginObjectives,
		TurboQuantRankMargin:              cfg.TurboQuantRankMargin,
		ProgressEverySteps:                cfg.ProgressEverySteps,
		EvalOnly:                          cfg.EvalOnly,
		PairwiseTrain:                     cfg.PairwiseTrain,
		HardNegativeTrain:                 cfg.HardNegativeTrain,
		ScoreSpectrumTrain:                cfg.ScoreSpectrumTrain,
		ListwiseGeometryTrain:             cfg.ListwiseGeometryTrain,
		MovementDiagnostics:               cfg.MovementDiagnostics,
		AllowResearchOnlyListwiseGeometry: cfg.AllowResearchOnlyListwiseGeometry,
		MaxListwiseGeometryTrainPairs:     cfg.MaxListwiseGeometryTrainPairs,
		MaxListwiseGeometryEvalPairs:      cfg.MaxListwiseGeometryEvalPairs,
		ScoreSpectrumEvalPath:             cfg.ScoreSpectrumEvalPath,
		ScoreSpectrumLossMode:             cfg.ScoreSpectrumLossMode,
		ScoreSpectrumRecoveryWeight:       cfg.ScoreSpectrumRecoveryWeight,
		ScoreSpectrumRecoveryMargin:       cfg.ScoreSpectrumRecoveryMargin,
		ScoreSpectrumRecoveryTopK:         cfg.ScoreSpectrumRecoveryTopK,
		ScoreSpectrumRecoveryTau:          cfg.ScoreSpectrumRecoveryTau,
		HardNegativesPerQuery:             cfg.HardNegativesPerQuery,
		HardNegativeSourceWeights:         cfg.HardNegativeSourceWeights,
	}
}

func trainBatchMetricsPayload(metrics eosruntime.EmbeddingTrainMetrics) trainBatchMetricsJSON {
	payload := trainBatchMetricsJSON{
		Loss:         metrics.Loss,
		AverageScore: metrics.AverageScore,
		BatchSize:    metrics.BatchSize,
	}
	if metrics.Movement != nil {
		payload.Movement = &trainMovementMetricsJSON{
			Gradient:       trainStatMetricsPayload(metrics.Movement.Gradient),
			ParameterDelta: trainStatMetricsPayload(metrics.Movement.ParameterDelta),
		}
	}
	return payload
}

func trainStatMetricsPayload(metrics eosruntime.EmbeddingTrainStatMetrics) trainStatMetricsJSON {
	return trainStatMetricsJSON{
		L2Norm:       metrics.L2Norm,
		MaxAbs:       metrics.MaxAbs,
		NonzeroCount: metrics.NonzeroCount,
		TotalCount:   metrics.TotalCount,
	}
}

func evalMetricsPayload(metrics *eosruntime.EmbeddingEvalMetrics) *evalMetricsJSON {
	if metrics == nil {
		return nil
	}
	return &evalMetricsJSON{
		Loss:                 metrics.Loss,
		AverageScore:         metrics.AverageScore,
		PositiveMeanScore:    metrics.PositiveMeanScore,
		NegativeMeanScore:    metrics.NegativeMeanScore,
		PairAccuracy:         metrics.PairAccuracy,
		ThresholdAccuracy:    metrics.ThresholdAccuracy,
		ScoreThreshold:       metrics.ScoreThreshold,
		ROCAUC:               metrics.ROCAUC,
		ScoreMargin:          metrics.ScoreMargin,
		Top1Accuracy:         metrics.Top1Accuracy,
		Top5Accuracy:         metrics.Top5Accuracy,
		Top10Accuracy:        metrics.Top10Accuracy,
		MeanReciprocalRank:   metrics.MeanReciprocalRank,
		MeanPositiveRank:     metrics.MeanPositiveRank,
		RetrievalNDCGAt10:    metrics.RetrievalNDCGAt10,
		RetrievalMAPAt100:    metrics.RetrievalMAPAt100,
		RetrievalRecallAt100: metrics.RetrievalRecallAt100,
		RetrievalEval:        cloneRetrievalEvalMetrics(metrics.RetrievalEval),
		PairCount:            metrics.PairCount,
		PositiveCount:        metrics.PositiveCount,
		NegativeCount:        metrics.NegativeCount,
	}
}

func cloneRetrievalEvalMetrics(metrics *eosruntime.RetrievalEvalMetrics) *eosruntime.RetrievalEvalMetrics {
	if metrics == nil {
		return nil
	}
	out := *metrics
	if metrics.SparseLexical != nil {
		sparse := *metrics.SparseLexical
		out.SparseLexical = &sparse
	}
	return &out
}

func printFinalRetrievalEval(eval *eosruntime.EmbeddingEvalMetrics) {
	if eval == nil || eval.RetrievalEval == nil {
		return
	}
	retrieval := eval.RetrievalEval
	fmt.Printf("final retrieval eval: dataset=%s backend=%s docs=%d queries=%d relevant_pairs=%d scored_pairs=%d elapsed=%.3fs doc_embed=%.3fs query_embed=%.3fs score=%.3fs docs_per_sec=%.2f queries_per_sec=%.2f scores_per_sec=%.2f\n",
		retrieval.Dataset,
		retrieval.Backend,
		retrieval.Inputs.Documents,
		retrieval.Inputs.Queries,
		retrieval.Inputs.RelevantPairs,
		retrieval.Inputs.ScoredPairs,
		retrieval.Throughput.ElapsedSeconds,
		retrieval.Throughput.DocumentEmbedSeconds,
		retrieval.Throughput.QueryEmbedSeconds,
		retrieval.Throughput.ScoreSeconds,
		retrieval.Throughput.DocumentsPerSecond,
		retrieval.Throughput.QueriesPerSecond,
		retrieval.Throughput.ScoresPerSecond,
	)
}

func scoreSpectrumEvalMetricsPayload(metrics *eosruntime.EmbeddingScoreSpectrumEvalMetrics) *scoreSpectrumEvalMetricsJSON {
	if metrics == nil {
		return nil
	}
	return &scoreSpectrumEvalMetricsJSON{
		Loss:                              metrics.Loss,
		AverageScore:                      metrics.AverageScore,
		AnyPositiveTop1:                   metrics.AnyPositiveTop1,
		OriginalPositiveTop1:              metrics.OriginalPositiveTop1,
		AlternateRelevantRecovery:         metrics.AlternateRelevantRecovery,
		BestPositiveHardestNegativeMargin: metrics.BestPositiveHardestNegativeMargin,
		TargetCrossEntropy:                metrics.TargetCrossEntropy,
		TargetKL:                          metrics.TargetKL,
		RowCount:                          metrics.RowCount,
		CandidateCount:                    metrics.CandidateCount,
		AnyPositiveRowCount:               metrics.AnyPositiveRowCount,
		OriginalPositiveRowCount:          metrics.OriginalPositiveRowCount,
		AlternateRecoveryRowCount:         metrics.AlternateRecoveryRowCount,
		MarginRowCount:                    metrics.MarginRowCount,
		TargetDistributionRowCount:        metrics.TargetDistributionRowCount,
	}
}

func listwiseGeometryEvalMetricsPayload(metrics *eosruntime.EmbeddingListwiseGeometryEvalMetrics) *listwiseGeometryEvalMetricsJSON {
	if metrics == nil {
		return nil
	}
	return &listwiseGeometryEvalMetricsJSON{
		Loss:                  metrics.Loss,
		AverageScore:          metrics.AverageScore,
		TeacherCrossEntropy:   metrics.TeacherCrossEntropy,
		TeacherKL:             metrics.TeacherKL,
		TeacherTop1Agreement:  metrics.TeacherTop1Agreement,
		AnyPositiveTop1:       metrics.AnyPositiveTop1,
		QueryCount:            metrics.QueryCount,
		DocumentCellCount:     metrics.DocumentCellCount,
		BatchCount:            metrics.BatchCount,
		AnyPositiveQueryCount: metrics.AnyPositiveQueryCount,
	}
}

func trainWorkloadPayload(workload eosruntime.EmbeddingTrainWorkload) trainWorkloadJSON {
	return trainWorkloadJSON{
		TrainMode:            workload.TrainMode,
		EvalMode:             workload.EvalMode,
		TrainExamples:        workload.TrainExamples,
		EvalExamples:         workload.EvalExamples,
		BatchSize:            workload.BatchSize,
		PlannedEpochs:        workload.PlannedEpochs,
		CompletedEpochs:      workload.CompletedEpochs,
		TrainBatchesPerEpoch: workload.TrainBatchesPerEpoch,
		TrainPairsPerEpoch:   workload.TrainPairsPerEpoch,
		EvalPairsPerPass:     workload.EvalPairsPerPass,
		PlannedEvalPasses:    workload.PlannedEvalPasses,
		ActualEvalPasses:     workload.ActualEvalPasses,
		PlannedTrainPairs:    workload.PlannedTrainPairs,
		ActualTrainPairs:     workload.ActualTrainPairs,
		ActualTrainExamples:  workload.ActualTrainExamples,
		PlannedEvalPairs:     workload.PlannedEvalPairs,
		ActualEvalPairs:      workload.ActualEvalPairs,
		ActualEvalExamples:   workload.ActualEvalExamples,
		PlannedTotalPairs:    workload.PlannedTotalPairs,
		ActualTotalPairs:     workload.ActualTotalPairs,
		ActualTotalExamples:  workload.ActualTotalExamples,
	}
}

func trainThroughputPayload(summary eosruntime.EmbeddingTrainRunSummary) trainThroughputJSON {
	return trainThroughputJSON{
		ElapsedSeconds:          summary.Elapsed.Seconds(),
		TrainSeconds:            summary.TrainDuration.Seconds(),
		EvalSeconds:             summary.EvalDuration.Seconds(),
		ExamplesPerSecond:       itemsPerSecond(summary.Workload.ActualTotalExamples, summary.Elapsed),
		PairsPerSecond:          pairsPerSecond(summary.Workload.ActualTotalPairs, summary.Elapsed),
		TrainExamplesPerSecond:  itemsPerSecond(summary.Workload.ActualTrainExamples, summary.TrainDuration),
		TrainPairsPerSecond:     pairsPerSecond(summary.Workload.ActualTrainPairs, summary.TrainDuration),
		EvalExamplesPerSecond:   itemsPerSecond(summary.Workload.ActualEvalExamples, summary.EvalDuration),
		EvalPairsPerSecond:      pairsPerSecond(summary.Workload.ActualEvalPairs, summary.EvalDuration),
		OptimizerStepsPerSecond: itemsPerSecond(int64(summary.StepsRun), summary.TrainDuration),
	}
}

func trainAcceleratorsPayload(profile eosruntime.EmbeddingTrainProfile) trainAcceleratorsJSON {
	return trainAcceleratorsJSON{
		Forward:     displayTrainBackend(profile.ForwardBackend),
		Optimizer:   displayTrainBackend(profile.OptimizerBackend),
		Activation:  displayTrainBackend(profile.ActivationBackend),
		Contrastive: displayTrainBackend(profile.ContrastiveBackend),
	}
}

func trainProfileDeltaPayload(profile eosruntime.EmbeddingTrainProfile) trainProfileDeltaJSON {
	return trainProfileDeltaJSON{
		MatMulBindCalls:     profile.ForwardResidency.MatMul.BindCalls,
		MatMulRuns:          profile.ForwardResidency.MatMul.RunCalls,
		MatMulRunUploadMB:   bytesToMiB(profile.ForwardResidency.MatMul.RunUploadedBytes),
		MatMulRunDownloadMB: bytesToMiB(profile.ForwardResidency.MatMul.RunDownloadedBytes),
		OptimizerUpdates:    profile.Optimizer.UpdateCalls,
		OptimizerSyncs:      profile.Optimizer.SyncCalls,
		ActivationCalls:     profile.Activation.GELUBackwardCalls + profile.Activation.SoftmaxBackwardCalls + profile.Activation.LayerNormBackwardCalls,
		ContrastiveCalls:    profile.Contrastive.RunCalls,
	}
}

func trainPackagePathsPayload(paths eosruntime.EmbeddingTrainPackagePaths) trainPackagePathsJSON {
	return trainPackagePathsJSON{
		Artifact:     paths.ArtifactPath,
		Checkpoint:   paths.CheckpointPath,
		TrainProfile: paths.TrainProfilePath,
		Manifest:     paths.EmbeddingManifestPath,
		Weights:      paths.WeightFilePath,
		MemoryPlan:   paths.MemoryPlanPath,
		Package:      paths.PackageManifestPath,
	}
}

func bytesToMiB(bytes int64) float64 {
	return float64(bytes) / (1024 * 1024)
}

func runCompareTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("compare-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	}
	currentPath := fs.Arg(0)
	current, err := readTrainMetricsJSON(currentPath)
	if err != nil {
		return err
	}
	var baseline *trainMetricsJSON
	baselinePath := ""
	if fs.NArg() > 1 && fs.Arg(1) != "" {
		baselinePath = fs.Arg(1)
		loaded, err := readTrainMetricsJSON(baselinePath)
		if err != nil {
			return err
		}
		baseline = &loaded
	}
	fmt.Print(formatTrainMetricsReport(currentPath, current, baselinePath, baseline))
	return nil
}

func runCompareRetrievalMetrics(args []string) error {
	fs := flag.NewFlagSet("compare-retrieval-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	metric := fs.String("metric", "ndcg_at_10", "metric to compare for gate mode")
	minRatio := fs.Float64("min-ratio", 1.0, "required current/baseline ratio when --require-win is set")
	requireWin := fs.Bool("require-win", false, "return non-zero unless current metric meets or beats baseline ratio")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos compare-retrieval-metrics [--metric ndcg_at_10] [--min-ratio 1.0] [--require-win] <current.retrieval.metrics.json> <baseline.retrieval.metrics.json>")
	}
	if *minRatio < 0 {
		return fmt.Errorf("min-ratio must be non-negative")
	}
	currentPath := fs.Arg(0)
	baselinePath := fs.Arg(1)
	current, err := readRetrievalMetricsJSON(currentPath)
	if err != nil {
		return err
	}
	baseline, err := readRetrievalMetricsJSON(baselinePath)
	if err != nil {
		return err
	}
	if current.Dataset != baseline.Dataset {
		return fmt.Errorf("dataset mismatch: current=%q baseline=%q", current.Dataset, baseline.Dataset)
	}
	fmt.Printf("current: %s backend=%s dataset=%s\n", currentPath, current.Backend, current.Dataset)
	fmt.Printf("baseline: %s backend=%s dataset=%s\n", baselinePath, baseline.Backend, baseline.Dataset)
	printRetrievalMetricDelta("ndcg_at_10", current.Quality.NDCGAt10, baseline.Quality.NDCGAt10)
	printRetrievalMetricDelta("mrr_at_10", current.Quality.MRRAt10, baseline.Quality.MRRAt10)
	printRetrievalMetricDelta("recall_at_10", current.Quality.RecallAt10, baseline.Quality.RecallAt10)
	printRetrievalMetricDelta("recall_at_100", current.Quality.RecallAt100, baseline.Quality.RecallAt100)
	currentValue, ok := retrievalMetricValue(current, *metric)
	if !ok {
		return fmt.Errorf("metric %q is unavailable", *metric)
	}
	baselineValue, ok := retrievalMetricValue(baseline, *metric)
	if !ok {
		return fmt.Errorf("metric %q is unavailable", *metric)
	}
	required := baselineValue * *minRatio
	ratio := 0.0
	if baselineValue != 0 {
		ratio = currentValue / baselineValue
	}
	fmt.Printf("target: %s=%.6g baseline=%.6g required=%.6g ratio=%.6g\n", *metric, currentValue, baselineValue, required, ratio)
	if *requireWin && !trainMetricThresholdPassed(currentValue, required, ">=") {
		fmt.Printf("retrieval baseline gate: FAIL metric=%s current=%.6g baseline=%.6g min_ratio=%.6g\n", *metric, currentValue, baselineValue, *minRatio)
		return fmt.Errorf("retrieval baseline gate failed")
	}
	if *requireWin {
		fmt.Printf("retrieval baseline gate: PASS metric=%s current=%.6g baseline=%.6g min_ratio=%.6g\n", *metric, currentValue, baselineValue, *minRatio)
	}
	return nil
}

func printRetrievalMetricDelta(name string, current, baseline float64) {
	ratio := 0.0
	if baseline != 0 {
		ratio = current / baseline
	}
	fmt.Printf("%s: current=%.6f baseline=%.6f delta=%+.6f ratio=%.6f\n", name, current, baseline, current-baseline, ratio)
}

func readTrainMetricsJSON(path string) (trainMetricsJSON, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return trainMetricsJSON{}, err
	}
	var metrics trainMetricsJSON
	if err := json.Unmarshal(data, &metrics); err != nil {
		return trainMetricsJSON{}, fmt.Errorf("parse metrics JSON %q: %w", path, err)
	}
	if metrics.Schema == "" {
		return trainMetricsJSON{}, fmt.Errorf("metrics JSON %q is missing schema", path)
	}
	return metrics, nil
}

func formatTrainMetricsReport(currentPath string, current trainMetricsJSON, baselinePath string, baseline *trainMetricsJSON) string {
	var b strings.Builder
	fmt.Fprintf(&b, "metrics: %s\n", currentPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", current.Schema, current.Command, current.Mode, current.Artifact)
	if current.FinalEval != nil {
		fmt.Fprintf(&b, "quality: top1=%.6f top5=%.6f top10=%.6f mrr=%.6f auc=%.6f margin=%.6f loss=%.6f mean_rank=%.3f pairs=%d\n",
			current.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin,
			current.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank,
			current.FinalEval.PairCount,
		)
	}
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f pairs/s=%.2f elapsed_s=%.3f\n",
		current.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "accelerators: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		current.Accelerators.Forward,
		current.Accelerators.Optimizer,
		current.Accelerators.Activation,
		current.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "profile_delta: matmul_runs=%d upload_mb=%.2f download_mb=%.2f optimizer_updates=%d activation_calls=%d contrastive_calls=%d\n",
		current.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls,
	)
	if baseline == nil {
		return b.String()
	}
	fmt.Fprintf(&b, "baseline: %s\n", baselinePath)
	if current.FinalEval != nil && baseline.FinalEval != nil {
		fmt.Fprintf(&b, "quality_delta: top1=%+.6f top5=%+.6f top10=%+.6f mrr=%+.6f auc=%+.6f margin=%+.6f loss=%+.6f mean_rank=%+.3f\n",
			current.FinalEval.Top1Accuracy-baseline.FinalEval.Top1Accuracy,
			current.FinalEval.Top5Accuracy-baseline.FinalEval.Top5Accuracy,
			current.FinalEval.Top10Accuracy-baseline.FinalEval.Top10Accuracy,
			current.FinalEval.MeanReciprocalRank-baseline.FinalEval.MeanReciprocalRank,
			current.FinalEval.ROCAUC-baseline.FinalEval.ROCAUC,
			current.FinalEval.ScoreMargin-baseline.FinalEval.ScoreMargin,
			current.FinalEval.Loss-baseline.FinalEval.Loss,
			current.FinalEval.MeanPositiveRank-baseline.FinalEval.MeanPositiveRank,
		)
	}
	fmt.Fprintf(&b, "throughput_delta: train_pairs/s=%+.2f eval_pairs/s=%+.2f optimizer_steps/s=%+.2f pairs/s=%+.2f elapsed_s=%+.3f\n",
		current.Throughput.TrainPairsPerSecond-baseline.Throughput.TrainPairsPerSecond,
		current.Throughput.EvalPairsPerSecond-baseline.Throughput.EvalPairsPerSecond,
		current.Throughput.OptimizerStepsPerSecond-baseline.Throughput.OptimizerStepsPerSecond,
		current.Throughput.PairsPerSecond-baseline.Throughput.PairsPerSecond,
		current.Throughput.ElapsedSeconds-baseline.Throughput.ElapsedSeconds,
	)
	fmt.Fprintf(&b, "profile_delta_delta: matmul_runs=%+d upload_mb=%+.2f download_mb=%+.2f optimizer_updates=%+d activation_calls=%+d contrastive_calls=%+d\n",
		current.ProfileDelta.MatMulRuns-baseline.ProfileDelta.MatMulRuns,
		current.ProfileDelta.MatMulRunUploadMB-baseline.ProfileDelta.MatMulRunUploadMB,
		current.ProfileDelta.MatMulRunDownloadMB-baseline.ProfileDelta.MatMulRunDownloadMB,
		current.ProfileDelta.OptimizerUpdates-baseline.ProfileDelta.OptimizerUpdates,
		current.ProfileDelta.ActivationCalls-baseline.ProfileDelta.ActivationCalls,
		current.ProfileDelta.ContrastiveCalls-baseline.ProfileDelta.ContrastiveCalls,
	)
	return b.String()
}

func runDiagnoseTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("diagnose-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos diagnose-train-metrics <metrics.json>")
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	fmt.Print(formatTrainMetricsDiagnosis(metricsPath, metrics))
	return nil
}

func formatTrainMetricsDiagnosis(metricsPath string, metrics trainMetricsJSON) string {
	var b strings.Builder
	warnings := 0
	notes := 0
	fmt.Fprintf(&b, "metrics: %s\n", metricsPath)
	fmt.Fprintf(&b, "identity: schema=%s command=%s mode=%s artifact=%s\n", metrics.Schema, metrics.Command, metrics.Mode, metrics.Artifact)
	fmt.Fprintf(&b, "backend: forward=%s optimizer=%s activation=%s contrastive=%s\n",
		metrics.Accelerators.Forward,
		metrics.Accelerators.Optimizer,
		metrics.Accelerators.Activation,
		metrics.Accelerators.Contrastive,
	)
	fmt.Fprintf(&b, "throughput: train_pairs/s=%.2f eval_pairs/s=%.2f optimizer_steps/s=%.2f elapsed_s=%.3f\n",
		metrics.Throughput.TrainPairsPerSecond,
		metrics.Throughput.EvalPairsPerSecond,
		metrics.Throughput.OptimizerStepsPerSecond,
		metrics.Throughput.ElapsedSeconds,
	)
	pairCount := trainMetricsDiagnosisPairCount(metrics)
	if metrics.ProfileDelta.MatMulRuns > 0 {
		parts := []string{}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("matmul_runs/update=%.2f", float64(metrics.ProfileDelta.MatMulRuns)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		} else {
			parts = append(parts, fmt.Sprintf("matmul_runs=%d", metrics.ProfileDelta.MatMulRuns))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("pairs/matmul_run=%.2f", float64(pairCount)/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if metrics.ProfileDelta.OptimizerUpdates > 0 {
			parts = append(parts, fmt.Sprintf("optimizer_syncs/update=%.2f", float64(metrics.ProfileDelta.OptimizerSyncs)/float64(metrics.ProfileDelta.OptimizerUpdates)))
		}
		fmt.Fprintf(&b, "efficiency: %s\n", strings.Join(parts, " "))
	}
	totalTransferMB := metrics.ProfileDelta.MatMulRunUploadMB + metrics.ProfileDelta.MatMulRunDownloadMB
	if totalTransferMB > 0 {
		parts := []string{fmt.Sprintf("total_mb=%.2f", totalTransferMB)}
		if metrics.ProfileDelta.MatMulRuns > 0 {
			parts = append(parts, fmt.Sprintf("mb/matmul_run=%.4f", totalTransferMB/float64(metrics.ProfileDelta.MatMulRuns)))
		}
		if pairCount > 0 {
			parts = append(parts, fmt.Sprintf("kb/pair=%.4f", totalTransferMB*1024/float64(pairCount)))
		}
		fmt.Fprintf(&b, "transfer: %s\n", strings.Join(parts, " "))
	}
	hostBackends := trainMetricsHostCriticalBackends(metrics)
	if len(hostBackends) == 0 {
		fmt.Fprintf(&b, "finding: ok production-critical accelerators are device-backed\n")
	} else {
		warnings++
		fmt.Fprintf(&b, "finding: warn production-critical accelerators include host fallback: %s\n", strings.Join(hostBackends, " "))
	}
	if trainMetricsEvalOnly(metrics) {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			fmt.Fprintf(&b, "finding: ok eval-only run recorded zero optimizer updates\n")
		} else {
			warnings++
			fmt.Fprintf(&b, "finding: warn eval-only run recorded optimizer_updates=%d\n", metrics.ProfileDelta.OptimizerUpdates)
		}
	} else {
		if metrics.ProfileDelta.OptimizerUpdates == 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training run recorded zero optimizer updates\n")
		} else if metrics.Throughput.OptimizerStepsPerSecond == 0 && metrics.Throughput.TrainSeconds > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn optimizer updates were recorded but optimizer_steps/s is zero\n")
		}
		if metrics.Throughput.TrainPairsPerSecond == 0 && metrics.Workload.ActualTrainPairs > 0 {
			warnings++
			fmt.Fprintf(&b, "finding: warn training pairs were processed but train_pairs/s is zero\n")
		}
		if metrics.Throughput.EvalSeconds > metrics.Throughput.TrainSeconds && metrics.Throughput.TrainSeconds > 0 {
			notes++
			fmt.Fprintf(&b, "finding: note eval time exceeds train time; keep production gates in final eval-only passes unless debugging convergence\n")
		}
	}
	if metrics.Accelerators.Activation == "" || metrics.Accelerators.Activation == "host" {
		notes++
		fmt.Fprintf(&b, "finding: note activation accelerator is host; this can be intentional until activation residency avoids extra transfers\n")
	}
	status := "OK"
	if warnings > 0 {
		status = "WARN"
	}
	fmt.Fprintf(&b, "diagnosis: %s warnings=%d notes=%d\n", status, warnings, notes)
	return b.String()
}

func trainMetricsDiagnosisPairCount(metrics trainMetricsJSON) int64 {
	if metrics.Workload.ActualTrainPairs > 0 {
		return metrics.Workload.ActualTrainPairs
	}
	if metrics.Workload.ActualEvalPairs > 0 {
		return metrics.Workload.ActualEvalPairs
	}
	return metrics.Workload.ActualTotalPairs
}

func trainMetricsEvalOnly(metrics trainMetricsJSON) bool {
	return strings.EqualFold(metrics.Mode, "eval") || metrics.Config.EvalOnly
}

func trainMetricsHostCriticalBackends(metrics trainMetricsJSON) []string {
	backends := []string{}
	if trainMetricsHostBackend(metrics.Accelerators.Forward) {
		backends = append(backends, "forward="+displayBackendLabel(metrics.Accelerators.Forward))
	}
	if !trainMetricsEvalOnly(metrics) && trainMetricsHostBackend(metrics.Accelerators.Optimizer) {
		backends = append(backends, "optimizer="+displayBackendLabel(metrics.Accelerators.Optimizer))
	}
	if trainMetricsHostBackend(metrics.Accelerators.Contrastive) {
		backends = append(backends, "contrastive="+displayBackendLabel(metrics.Accelerators.Contrastive))
	}
	return backends
}

func trainMetricsHostBackend(backend string) bool {
	return backend == "" || backend == "host"
}

func displayBackendLabel(backend string) string {
	if backend == "" {
		return "unknown"
	}
	return backend
}

type trainMetricThreshold struct {
	Env    string
	Metric string
	Op     string
	Scope  string
}

var trainMetricThresholds = []trainMetricThreshold{
	{Env: "EOS_MIN_MRR", Metric: "mrr", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_TOP1", Metric: "top1", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_TOP5", Metric: "top5", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_TOP10", Metric: "top10", Op: ">=", Scope: "quality"},
	{Env: "EOS_MAX_MEAN_RANK", Metric: "mean_rank", Op: "<=", Scope: "quality"},
	{Env: "EOS_MIN_AUC", Metric: "auc", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_THRESHOLD_ACCURACY", Metric: "threshold_accuracy", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_SCORE_MARGIN", Metric: "margin", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_PAIR_ACCURACY", Metric: "accuracy", Op: ">=", Scope: "quality"},
	{Env: "EOS_MAX_LOSS", Metric: "loss", Op: "<=", Scope: "quality"},
	{Env: "EOS_MIN_TRAIN_PAIRS_PER_SEC", Metric: "train_pairs/s", Op: ">=", Scope: "efficiency"},
	{Env: "EOS_MIN_OPTIMIZER_STEPS_PER_SEC", Metric: "optimizer_steps/s", Op: ">=", Scope: "efficiency"},
	{Env: "EOS_MAX_MATMUL_RUNS", Metric: "matmul_runs", Op: "<=", Scope: "efficiency"},
	{Env: "EOS_MAX_MATMUL_RUN_UPLOAD_MB", Metric: "matmul_run_upload_mb", Op: "<=", Scope: "efficiency"},
	{Env: "EOS_MAX_MATMUL_RUN_DOWNLOAD_MB", Metric: "matmul_run_download_mb", Op: "<=", Scope: "efficiency"},
	{Env: "EOS_MAX_OPTIMIZER_UPDATES", Metric: "optimizer_updates", Op: "<=", Scope: "eval-only"},
}

func runGateTrainMetrics(args []string) error {
	fs := flag.NewFlagSet("gate-train-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var thresholdsPath string
	var scope string
	fs.StringVar(&thresholdsPath, "thresholds", "", "optional KEY=VALUE threshold env file")
	fs.StringVar(&scope, "scope", "all", "gate scope: all, quality, efficiency, or eval-only")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos gate-train-metrics [--thresholds thresholds.env] [--scope all|quality|efficiency|eval-only] <metrics.json>")
	}
	scope = strings.ToLower(scope)
	if !validTrainMetricsGateScope(scope) {
		return fmt.Errorf("unsupported gate scope %q", scope)
	}
	metricsPath := fs.Arg(0)
	metrics, err := readTrainMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	thresholds, err := trainMetricThresholdValues(thresholdsPath)
	if err != nil {
		return err
	}
	fmt.Printf("metrics: %s\n", metricsPath)
	if thresholdsPath != "" {
		fmt.Printf("thresholds: %s\n", thresholdsPath)
	}
	fmt.Printf("scope: %s\n", scope)
	checked := 0
	failed := 0
	for _, threshold := range trainMetricThresholds {
		if !trainMetricThresholdInScope(threshold.Scope, scope) {
			continue
		}
		limitText := thresholds[threshold.Env]
		if limitText == "" {
			continue
		}
		limit, err := strconv.ParseFloat(limitText, 64)
		if err != nil {
			return fmt.Errorf("%s=%q is not numeric: %w", threshold.Env, limitText, err)
		}
		got, ok := trainMetricValue(metrics, threshold.Metric)
		if !ok {
			return fmt.Errorf("metric %s is unavailable in %s", threshold.Metric, metricsPath)
		}
		checked++
		passed := trainMetricThresholdPassed(got, limit, threshold.Op)
		status := "pass"
		if !passed {
			status = "fail"
			failed++
		}
		fmt.Printf("%s: %s=%.6g %s %.6g (%s)\n", status, threshold.Metric, got, threshold.Op, limit, threshold.Env)
	}
	if scope == "eval-only" {
		got, ok := trainMetricValue(metrics, "optimizer_updates")
		if !ok {
			return fmt.Errorf("metric optimizer_updates is unavailable in %s", metricsPath)
		}
		checked++
		if got == 0 {
			fmt.Printf("pass: optimizer_updates=%.6g == 0 (eval-only)\n", got)
		} else {
			fmt.Printf("fail: optimizer_updates=%.6g == 0 (eval-only)\n", got)
			failed++
		}
	}
	if checked == 0 {
		return fmt.Errorf("no thresholds selected for scope %q", scope)
	}
	if failed > 0 {
		fmt.Printf("gate: FAIL checks=%d failed=%d\n", checked, failed)
		return fmt.Errorf("metrics gate failed")
	}
	fmt.Printf("gate: PASS checks=%d\n", checked)
	return nil
}

func validTrainMetricsGateScope(scope string) bool {
	switch scope {
	case "all", "quality", "efficiency", "eval-only":
		return true
	default:
		return false
	}
}

func trainMetricThresholdInScope(thresholdScope, requestedScope string) bool {
	switch requestedScope {
	case "all":
		return thresholdScope == "quality" || thresholdScope == "efficiency"
	default:
		return thresholdScope == requestedScope
	}
}

func trainMetricThresholdValues(path string) (map[string]string, error) {
	values := map[string]string{}
	for _, env := range os.Environ() {
		key, value, ok := strings.Cut(env, "=")
		if ok && strings.HasPrefix(key, "EOS_") {
			values[key] = value
		}
	}
	if path == "" {
		return values, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	for i, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected KEY=VALUE", path, i+1)
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if !strings.HasPrefix(key, "EOS_") {
			return nil, fmt.Errorf("%s:%d: threshold key must start with EOS_: %s", path, i+1, key)
		}
		if values[key] == "" {
			values[key] = value
		}
	}
	return values, nil
}

func trainMetricValue(metrics trainMetricsJSON, metric string) (float64, bool) {
	switch metric {
	case "train_pairs/s":
		return metrics.Throughput.TrainPairsPerSecond, true
	case "optimizer_steps/s":
		return metrics.Throughput.OptimizerStepsPerSecond, true
	case "matmul_runs":
		return float64(metrics.ProfileDelta.MatMulRuns), true
	case "matmul_run_upload_mb":
		return metrics.ProfileDelta.MatMulRunUploadMB, true
	case "matmul_run_download_mb":
		return metrics.ProfileDelta.MatMulRunDownloadMB, true
	case "optimizer_updates":
		return float64(metrics.ProfileDelta.OptimizerUpdates), true
	}
	if metrics.FinalEval == nil {
		return 0, false
	}
	switch metric {
	case "mrr":
		return float64(metrics.FinalEval.MeanReciprocalRank), true
	case "top1":
		return float64(metrics.FinalEval.Top1Accuracy), true
	case "top5":
		return float64(metrics.FinalEval.Top5Accuracy), true
	case "top10":
		return float64(metrics.FinalEval.Top10Accuracy), true
	case "mean_rank":
		return float64(metrics.FinalEval.MeanPositiveRank), true
	case "auc":
		return float64(metrics.FinalEval.ROCAUC), true
	case "threshold_accuracy":
		return float64(metrics.FinalEval.ThresholdAccuracy), true
	case "margin":
		return float64(metrics.FinalEval.ScoreMargin), true
	case "accuracy":
		return float64(metrics.FinalEval.PairAccuracy), true
	case "loss":
		return float64(metrics.FinalEval.Loss), true
	default:
		return 0, false
	}
}

func trainMetricThresholdPassed(got, limit float64, op string) bool {
	const epsilon = 1e-6
	switch op {
	case ">=":
		return got >= limit-epsilon
	case "<=":
		return got <= limit+epsilon
	default:
		return false
	}
}

type retrievalMetricThreshold struct {
	Env    string
	Metric string
	Op     string
	Scope  string
}

var retrievalMetricThresholds = []retrievalMetricThreshold{
	{Env: "EOS_MIN_RETRIEVAL_NDCG10", Metric: "ndcg_at_10", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_RETRIEVAL_MRR10", Metric: "mrr_at_10", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_RETRIEVAL_RECALL10", Metric: "recall_at_10", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_RETRIEVAL_RECALL100", Metric: "recall_at_100", Op: ">=", Scope: "quality"},
	{Env: "EOS_MIN_RETRIEVAL_DOCUMENTS_PER_SEC", Metric: "documents/s", Op: ">=", Scope: "efficiency"},
	{Env: "EOS_MIN_RETRIEVAL_QUERIES_PER_SEC", Metric: "queries/s", Op: ">=", Scope: "efficiency"},
	{Env: "EOS_MIN_RETRIEVAL_SCORES_PER_SEC", Metric: "scores/s", Op: ">=", Scope: "efficiency"},
}

func runGateRetrievalMetrics(args []string) error {
	fs := flag.NewFlagSet("gate-retrieval-metrics", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var thresholdsPath string
	var scope string
	fs.StringVar(&thresholdsPath, "thresholds", "", "optional KEY=VALUE threshold env file")
	fs.StringVar(&scope, "scope", "all", "gate scope: all, quality, or efficiency")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || fs.Arg(0) == "" {
		return fmt.Errorf("usage: eos gate-retrieval-metrics [--thresholds thresholds.env] [--scope all|quality|efficiency] <retrieval.metrics.json>")
	}
	scope = strings.ToLower(scope)
	if !validRetrievalMetricsGateScope(scope) {
		return fmt.Errorf("unsupported gate scope %q", scope)
	}
	metricsPath := fs.Arg(0)
	metrics, err := readRetrievalMetricsJSON(metricsPath)
	if err != nil {
		return err
	}
	thresholds, err := trainMetricThresholdValues(thresholdsPath)
	if err != nil {
		return err
	}
	fmt.Printf("metrics: %s\n", metricsPath)
	if thresholdsPath != "" {
		fmt.Printf("thresholds: %s\n", thresholdsPath)
	}
	fmt.Printf("dataset: %s\n", metrics.Dataset)
	fmt.Printf("scope: %s\n", scope)
	checked := 0
	failed := 0
	for _, threshold := range retrievalMetricThresholds {
		if !trainMetricThresholdInScope(threshold.Scope, scope) {
			continue
		}
		envName, limitText := retrievalThresholdValue(thresholds, threshold.Env, metrics.Dataset)
		if limitText == "" {
			continue
		}
		limit, err := strconv.ParseFloat(limitText, 64)
		if err != nil {
			return fmt.Errorf("%s=%q is not numeric: %w", envName, limitText, err)
		}
		got, ok := retrievalMetricValue(metrics, threshold.Metric)
		if !ok {
			return fmt.Errorf("metric %s is unavailable in %s", threshold.Metric, metricsPath)
		}
		checked++
		passed := trainMetricThresholdPassed(got, limit, threshold.Op)
		status := "pass"
		if !passed {
			status = "fail"
			failed++
		}
		fmt.Printf("%s: %s=%.6g %s %.6g (%s)\n", status, threshold.Metric, got, threshold.Op, limit, envName)
	}
	if checked == 0 {
		return fmt.Errorf("no retrieval thresholds selected for scope %q", scope)
	}
	if failed > 0 {
		fmt.Printf("retrieval gate: FAIL checks=%d failed=%d\n", checked, failed)
		return fmt.Errorf("retrieval metrics gate failed")
	}
	fmt.Printf("retrieval gate: PASS checks=%d\n", checked)
	return nil
}

type retrievalScoreboard struct {
	Schema string                   `json:"schema"`
	Rows   []retrievalScoreboardRow `json:"rows"`
}

type retrievalScoreboardRow struct {
	Category              string  `json:"category"`
	Dataset               string  `json:"dataset"`
	Baseline              string  `json:"baseline"`
	Status                string  `json:"status"`
	Method                string  `json:"method,omitempty"`
	Bits                  int     `json:"bits,omitempty"`
	QuantizerSeed         int64   `json:"quantizer_seed,omitempty"`
	RerankStorage         string  `json:"rerank_storage,omitempty"`
	NDCGAt10              float64 `json:"ndcg_at_10,omitempty"`
	NDCGAt100             float64 `json:"ndcg_at_100,omitempty"`
	MRRAt10               float64 `json:"mrr_at_10,omitempty"`
	PrecisionAt1          float64 `json:"precision_at_1,omitempty"`
	PrecisionAt5          float64 `json:"precision_at_5,omitempty"`
	PrecisionAt10         float64 `json:"precision_at_10,omitempty"`
	HitAt1                float64 `json:"hit_at_1,omitempty"`
	HitAt5                float64 `json:"hit_at_5,omitempty"`
	HitAt10               float64 `json:"hit_at_10,omitempty"`
	MAPAt10               float64 `json:"map_at_10,omitempty"`
	MAPAt100              float64 `json:"map_at_100,omitempty"`
	RecallAt10            float64 `json:"recall_at_10,omitempty"`
	RecallAt100           float64 `json:"recall_at_100,omitempty"`
	VectorBytes           int64   `json:"vector_bytes,omitempty"`
	DenseVectorBytes      int64   `json:"dense_vector_bytes,omitempty"`
	RerankSidecarBytes    int64   `json:"rerank_sidecar_bytes,omitempty"`
	TotalVectorBytes      int64   `json:"total_vector_bytes,omitempty"`
	CompressionRatio      float64 `json:"compression_ratio,omitempty"`
	TotalCompressionRatio float64 `json:"total_compression_ratio,omitempty"`
	DocumentsPerSecond    float64 `json:"documents_per_second,omitempty"`
	QueriesPerSecond      float64 `json:"queries_per_second,omitempty"`
	ScoresPerSecond       float64 `json:"scores_per_second,omitempty"`
}

type scoreboardGateSelection struct {
	Category string
	Baseline string
	Method   string
	Bits     int
}

func runGateScoreboard(args []string) error {
	fs := flag.NewFlagSet("gate-scoreboard", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	category := fs.String("category", "short_retrieval", "scoreboard category to compare")
	rowBaseline := fs.String("baseline", "eos", "scoreboard row baseline label to compare")
	method := fs.String("method", "", "optional scoreboard row method filter")
	bits := fs.Int("bits", -1, "optional scoreboard row bit-width filter")
	anchorCategory := fs.String("anchor-category", "", "optional anchor scoreboard category override; default uses --category")
	anchorBaseline := fs.String("anchor-baseline", "", "optional anchor scoreboard baseline override; default uses --baseline")
	anchorMethod := fs.String("anchor-method", "", "optional anchor scoreboard method override; default uses --method")
	anchorBits := fs.Int("anchor-bits", -2, "optional anchor scoreboard bit-width override; -2 uses --bits, -1 disables anchor bit filtering")
	datasetsText := fs.String("datasets", "", "comma-separated datasets that must all pass")
	metricsText := fs.String("metrics", "ndcg_at_10,recall_at_100", "comma-separated metrics that must all pass")
	tolerance := fs.Float64("tolerance", 0, "allowed current metric drop below anchor")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos gate-scoreboard [--category short_retrieval] [--baseline eos] [--method label] [--bits n] [--anchor-method label] [--anchor-bits n] --datasets scifact,nfcorpus,fiqa [--metrics ndcg_at_10,recall_at_100] [--tolerance 0] <current.scoreboard.json> <anchor.scoreboard.json>")
	}
	if *tolerance < 0 {
		return fmt.Errorf("tolerance must be non-negative")
	}
	if *bits < -1 {
		return fmt.Errorf("bits must be non-negative when set")
	}
	if *anchorBits < -2 {
		return fmt.Errorf("anchor-bits must be -2, -1, or non-negative")
	}
	datasets := splitCommaList(*datasetsText)
	if len(datasets) == 0 {
		return fmt.Errorf("--datasets is required")
	}
	metrics := splitCommaList(*metricsText)
	if len(metrics) == 0 {
		return fmt.Errorf("--metrics must select at least one metric")
	}
	for _, metric := range metrics {
		if !scoreboardRetrievalMetricSupported(metric) {
			return fmt.Errorf("unsupported scoreboard metric %q", metric)
		}
	}
	currentPath := fs.Arg(0)
	anchorPath := fs.Arg(1)
	current, err := readRetrievalScoreboard(currentPath)
	if err != nil {
		return err
	}
	anchor, err := readRetrievalScoreboard(anchorPath)
	if err != nil {
		return err
	}
	currentSelection := scoreboardGateSelection{
		Category: strings.TrimSpace(*category),
		Baseline: strings.TrimSpace(*rowBaseline),
		Method:   strings.TrimSpace(*method),
		Bits:     *bits,
	}
	anchorSelection := currentSelection
	if override := strings.TrimSpace(*anchorCategory); override != "" {
		anchorSelection.Category = override
	}
	if override := strings.TrimSpace(*anchorBaseline); override != "" {
		anchorSelection.Baseline = override
	}
	if override := strings.TrimSpace(*anchorMethod); override != "" {
		anchorSelection.Method = override
	}
	if *anchorBits != -2 {
		anchorSelection.Bits = *anchorBits
	}
	if currentSelection.Category == "" || currentSelection.Baseline == "" {
		return fmt.Errorf("category and baseline must be non-empty")
	}
	if anchorSelection.Category == "" || anchorSelection.Baseline == "" {
		return fmt.Errorf("anchor category and baseline must be non-empty")
	}
	fmt.Printf("current: %s\n", currentPath)
	fmt.Printf("anchor: %s\n", anchorPath)
	if currentSelection == anchorSelection {
		fmt.Printf("selection: %s\n", formatScoreboardGateSelection(currentSelection))
	} else {
		fmt.Printf("current selection: %s\n", formatScoreboardGateSelection(currentSelection))
		fmt.Printf("anchor selection: %s\n", formatScoreboardGateSelection(anchorSelection))
	}
	fmt.Printf("datasets: %s\n", strings.Join(datasets, ","))
	fmt.Printf("metrics: %s tolerance=%.6g\n", strings.Join(metrics, ","), *tolerance)

	failures := 0
	checked := 0
	macroCurrent := make(map[string]float64, len(metrics))
	macroAnchor := make(map[string]float64, len(metrics))
	for _, dataset := range datasets {
		currentRow, err := selectRetrievalScoreboardRow(current.Rows, currentSelection, dataset, currentPath)
		if err != nil {
			return fmt.Errorf("current scoreboard row selection failed: %w", err)
		}
		anchorRow, err := selectRetrievalScoreboardRow(anchor.Rows, anchorSelection, dataset, anchorPath)
		if err != nil {
			return fmt.Errorf("anchor scoreboard row selection failed: %w", err)
		}
		if err := validateCompactScoreboardProvenance(currentRow, anchorRow, dataset); err != nil {
			return err
		}
		for _, metric := range metrics {
			currentValue, _ := scoreboardRetrievalMetricValue(currentRow, metric)
			anchorValue, _ := scoreboardRetrievalMetricValue(anchorRow, metric)
			limit := anchorValue - *tolerance
			passed := currentValue >= limit-1e-6
			status := "PASS"
			if !passed {
				status = "FAIL"
				failures++
			}
			checked++
			macroCurrent[metric] += currentValue
			macroAnchor[metric] += anchorValue
			fmt.Printf("%s dataset=%s metric=%s current=%.6f anchor=%.6f delta=%+.6f required>=%.6f\n",
				status, dataset, metric, currentValue, anchorValue, currentValue-anchorValue, limit)
		}
	}
	for _, metric := range metrics {
		currentMean := macroCurrent[metric] / float64(len(datasets))
		anchorMean := macroAnchor[metric] / float64(len(datasets))
		fmt.Printf("macro metric=%s current=%.6f anchor=%.6f delta=%+.6f\n", metric, currentMean, anchorMean, currentMean-anchorMean)
	}
	if failures > 0 {
		fmt.Printf("scoreboard gate: FAIL checks=%d failed=%d\n", checked, failures)
		return fmt.Errorf("scoreboard gate failed")
	}
	fmt.Printf("scoreboard gate: PASS checks=%d\n", checked)
	return nil
}

func formatScoreboardGateSelection(selection scoreboardGateSelection) string {
	var b strings.Builder
	fmt.Fprintf(&b, "category=%s baseline=%s", selection.Category, selection.Baseline)
	if selection.Method != "" {
		fmt.Fprintf(&b, " method=%s", selection.Method)
	}
	if selection.Bits >= 0 {
		fmt.Fprintf(&b, " bits=%d", selection.Bits)
	}
	return b.String()
}

func splitCommaList(text string) []string {
	items := []string{}
	for _, raw := range strings.Split(text, ",") {
		item := strings.TrimSpace(raw)
		if item != "" {
			items = append(items, item)
		}
	}
	return items
}

func readRetrievalScoreboard(path string) (retrievalScoreboard, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return retrievalScoreboard{}, err
	}
	var scoreboard retrievalScoreboard
	if err := json.Unmarshal(data, &scoreboard); err != nil {
		return retrievalScoreboard{}, fmt.Errorf("parse scoreboard JSON %q: %w", path, err)
	}
	if scoreboard.Schema == "" {
		return retrievalScoreboard{}, fmt.Errorf("scoreboard JSON %q is missing schema", path)
	}
	if len(scoreboard.Rows) == 0 {
		return retrievalScoreboard{}, fmt.Errorf("scoreboard JSON %q has no rows", path)
	}
	return scoreboard, nil
}

func selectRetrievalScoreboardRow(rows []retrievalScoreboardRow, selection scoreboardGateSelection, dataset, path string) (retrievalScoreboardRow, error) {
	matches := selectRetrievalScoreboardRows(rows, selection, dataset, []string{selection.Baseline})
	if len(matches) == 0 {
		aliases := scoreboardBaselineAliases(selection.Baseline)
		if len(aliases) > 1 {
			matches = selectRetrievalScoreboardRows(rows, selection, dataset, aliases[1:])
		}
	}
	if len(matches) == 0 {
		return retrievalScoreboardRow{}, fmt.Errorf("scoreboard row missing in %s: category=%s dataset=%s baseline=%s%s%s",
			path, selection.Category, dataset, selection.Baseline, scoreboardMethodErrorSuffix(selection.Method), scoreboardBitsErrorSuffix(selection.Bits))
	}
	if len(matches) > 1 {
		return retrievalScoreboardRow{}, fmt.Errorf("scoreboard row is ambiguous in %s: category=%s dataset=%s baseline=%s%s%s matches=%d",
			path, selection.Category, dataset, selection.Baseline, scoreboardMethodErrorSuffix(selection.Method), scoreboardBitsErrorSuffix(selection.Bits), len(matches))
	}
	return matches[0], nil
}

func selectRetrievalScoreboardRows(rows []retrievalScoreboardRow, selection scoreboardGateSelection, dataset string, baselines []string) []retrievalScoreboardRow {
	var matches []retrievalScoreboardRow
	for _, row := range rows {
		if row.Category != selection.Category || row.Dataset != dataset || !slices.Contains(baselines, row.Baseline) {
			continue
		}
		if selection.Method != "" && row.Method != selection.Method {
			continue
		}
		if selection.Bits >= 0 && row.Bits != selection.Bits {
			continue
		}
		matches = append(matches, row)
	}
	return matches
}

func validateCompactScoreboardProvenance(currentRow, anchorRow retrievalScoreboardRow, dataset string) error {
	if !scoreboardRowRequiresQuantizerSeed(currentRow) && !scoreboardRowRequiresQuantizerSeed(anchorRow) {
		return nil
	}
	if currentRow.QuantizerSeed == 0 {
		return fmt.Errorf("compact scoreboard provenance failed: dataset=%s current row baseline=%s method=%s is missing quantizer_seed", dataset, currentRow.Baseline, currentRow.Method)
	}
	if anchorRow.QuantizerSeed == 0 {
		return fmt.Errorf("compact scoreboard provenance failed: dataset=%s anchor row baseline=%s method=%s is missing quantizer_seed", dataset, anchorRow.Baseline, anchorRow.Method)
	}
	if currentRow.QuantizerSeed != anchorRow.QuantizerSeed {
		return fmt.Errorf("compact scoreboard provenance failed: dataset=%s quantizer_seed mismatch current=%d anchor=%d", dataset, currentRow.QuantizerSeed, anchorRow.QuantizerSeed)
	}
	return nil
}

func scoreboardRowRequiresQuantizerSeed(row retrievalScoreboardRow) bool {
	return strings.Contains(row.Baseline, "turboquant") || strings.HasPrefix(row.Method, "turboquant_")
}

func scoreboardBaselineAliases(baseline string) []string {
	switch baseline {
	case "eos":
		return []string{"eos", "manta"}
	case "eos-hybrid":
		return []string{"eos-hybrid", "manta-hybrid"}
	case "eos-turboquant":
		return []string{"eos-turboquant", "manta-turboquant"}
	default:
		return []string{baseline}
	}
}

func scoreboardMethodErrorSuffix(method string) string {
	if method == "" {
		return ""
	}
	return " method=" + method
}

func scoreboardBitsErrorSuffix(bits int) string {
	if bits < 0 {
		return ""
	}
	return fmt.Sprintf(" bits=%d", bits)
}

func scoreboardRetrievalMetricSupported(metric string) bool {
	_, ok := scoreboardRetrievalMetricValue(retrievalScoreboardRow{}, metric)
	return ok
}

func scoreboardRetrievalMetricValue(row retrievalScoreboardRow, metric string) (float64, bool) {
	switch metric {
	case "ndcg_at_10":
		return row.NDCGAt10, true
	case "ndcg_at_100":
		return row.NDCGAt100, true
	case "mrr_at_10":
		return row.MRRAt10, true
	case "precision_at_1":
		return row.PrecisionAt1, true
	case "precision_at_5":
		return row.PrecisionAt5, true
	case "precision_at_10":
		return row.PrecisionAt10, true
	case "hit_at_1":
		return row.HitAt1, true
	case "hit_at_5":
		return row.HitAt5, true
	case "hit_at_10":
		return row.HitAt10, true
	case "map_at_10":
		return row.MAPAt10, true
	case "map_at_100":
		return row.MAPAt100, true
	case "recall_at_10":
		return row.RecallAt10, true
	case "recall_at_100":
		return row.RecallAt100, true
	case "compression_ratio":
		return row.CompressionRatio, true
	case "total_compression_ratio":
		return row.TotalCompressionRatio, true
	case "documents/s", "documents_per_second":
		return row.DocumentsPerSecond, true
	case "queries/s", "queries_per_second":
		return row.QueriesPerSecond, true
	case "scores/s", "scores_per_second":
		return row.ScoresPerSecond, true
	default:
		return 0, false
	}
}

func readRetrievalMetricsJSON(path string) (eosruntime.RetrievalEvalMetrics, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return eosruntime.RetrievalEvalMetrics{}, err
	}
	var metrics eosruntime.RetrievalEvalMetrics
	if err := json.Unmarshal(data, &metrics); err != nil {
		return eosruntime.RetrievalEvalMetrics{}, fmt.Errorf("parse retrieval metrics JSON %q: %w", path, err)
	}
	if metrics.Schema != eosruntime.RetrievalEvalMetricsSchema {
		return eosruntime.RetrievalEvalMetrics{}, fmt.Errorf("unsupported retrieval metrics schema %q", metrics.Schema)
	}
	return metrics, nil
}

func validRetrievalMetricsGateScope(scope string) bool {
	switch scope {
	case "all", "quality", "efficiency":
		return true
	default:
		return false
	}
}

func retrievalThresholdValue(values map[string]string, baseEnv, dataset string) (string, string) {
	if suffix := retrievalDatasetEnvSuffix(dataset); suffix != "" {
		datasetEnv := baseEnv + "_" + suffix
		if value := values[datasetEnv]; value != "" {
			return datasetEnv, value
		}
	}
	return baseEnv, values[baseEnv]
}

func retrievalDatasetEnvSuffix(dataset string) string {
	dataset = strings.ToUpper(strings.TrimSpace(dataset))
	if dataset == "" {
		return ""
	}
	var b strings.Builder
	lastUnderscore := false
	for _, r := range dataset {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(b.String(), "_")
}

func retrievalMetricValue(metrics eosruntime.RetrievalEvalMetrics, metric string) (float64, bool) {
	switch metric {
	case "ndcg_at_10":
		return metrics.Quality.NDCGAt10, true
	case "mrr_at_10":
		return metrics.Quality.MRRAt10, true
	case "recall_at_10":
		return metrics.Quality.RecallAt10, true
	case "recall_at_100":
		return metrics.Quality.RecallAt100, true
	case "documents/s":
		return metrics.Throughput.DocumentsPerSecond, true
	case "queries/s":
		return metrics.Throughput.QueriesPerSecond, true
	case "scores/s":
		return metrics.Throughput.ScoresPerSecond, true
	default:
		return 0, false
	}
}

func runTrainTokenizer(args []string) error {
	fs := flag.NewFlagSet("train-tokenizer", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var outputPath string
	var manifestPath string
	var vocabSize int
	var minFreq int
	fs.StringVar(&outputPath, "output", "", "output tokenizer path")
	fs.StringVar(&manifestPath, "manifest", "", "embedding manifest path")
	fs.IntVar(&vocabSize, "vocab-size", 0, "tokenizer vocab size override")
	fs.IntVar(&minFreq, "min-freq", 2, "minimum pair frequency for merges")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	}
	artifactPath := fs.Arg(0)
	corpusPath := fs.Arg(1)
	if outputPath == "" {
		outputPath = eosruntime.DefaultTokenizerPath(artifactPath)
	}
	if manifestPath == "" {
		manifestPath = eosruntime.DefaultEmbeddingManifestPath(artifactPath)
	}
	manifestVocabSize := 0
	if manifest, err := eosruntime.ReadEmbeddingManifestFile(manifestPath); err == nil {
		manifestVocabSize = manifest.Tokenizer.VocabSize
	} else if vocabSize == 0 {
		return fmt.Errorf("read embedding manifest for vocab size: %w", err)
	}
	if vocabSize == 0 {
		vocabSize = manifestVocabSize
	}
	if vocabSize <= 0 {
		return fmt.Errorf("tokenizer vocab size must be set via --vocab-size or embedding manifest")
	}
	if manifestVocabSize > 0 && vocabSize < manifestVocabSize {
		return fmt.Errorf("tokenizer vocab size %d would shrink package vocab contract %d; checkpoint tensor resize is not supported by train-tokenizer", vocabSize, manifestVocabSize)
	}
	tokenizer, err := eosruntime.TrainTokenizerFromCorpus(eosruntime.TokenizerTrainConfig{
		CorpusPath: corpusPath,
		VocabSize:  vocabSize,
		MinFreq:    minFreq,
	})
	if err != nil {
		return err
	}
	tokenizer, err = eosruntime.PadTokenizerFileVocab(tokenizer, vocabSize)
	if err != nil {
		return err
	}
	if err := tokenizer.WriteFile(outputPath); err != nil {
		return err
	}
	if err := eosruntime.SyncEmbeddingTokenizerVocab(artifactPath, vocabSize); err != nil {
		return fmt.Errorf("sync tokenizer vocab through Eos package: %w", err)
	}
	fmt.Printf("trained tokenizer %q\n", outputPath)
	fmt.Printf("vocab: %d tokens, merges: %d\n", len(tokenizer.Tokens), len(tokenizer.Merges))
	return nil
}

func runTokenizeEmbed(args []string) error {
	fs := flag.NewFlagSet("tokenize-embed", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var tokenizerPath string
	var mode string
	var hardNegativesPerQuery int
	fs.StringVar(&tokenizerPath, "tokenizer", "", "path to tokenizer .mll (default: sibling tokenizer)")
	fs.StringVar(&mode, "mode", "contrastive", "output mode: contrastive, pair, or hard-negative")
	fs.IntVar(&hardNegativesPerQuery, "hard-negatives-per-query", 1, "maximum explicit negatives per query-positive example in hard-negative mode")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 3 || fs.Arg(0) == "" || fs.Arg(1) == "" || fs.Arg(2) == "" {
		return fmt.Errorf("usage: eos tokenize-embed [--mode contrastive|pair|hard-negative] [--tokenizer tokenizer.mll] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	}
	if hardNegativesPerQuery < 0 {
		return fmt.Errorf("hard-negatives-per-query must be non-negative")
	}
	artifactPath := fs.Arg(0)
	inputPath := fs.Arg(1)
	outputPath := fs.Arg(2)
	if tokenizerPath == "" {
		tokenizerPath = eosruntime.DefaultTokenizerPath(artifactPath)
	}
	tokenizerFile, err := eosruntime.ReadTokenizerFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("read tokenizer: %w", err)
	}
	manifest, err := eosruntime.ReadEmbeddingManifestFile(eosruntime.ResolveEmbeddingManifestPath(artifactPath))
	if err != nil {
		return fmt.Errorf("read embedding manifest: %w", err)
	}
	tokenizer, err := eosruntime.NewBPETokenizer(tokenizerFile, manifest.Tokenizer)
	if err != nil {
		return fmt.Errorf("build tokenizer: %w", err)
	}
	switch strings.ToLower(mode) {
	case "contrastive":
		examples, err := eosruntime.ReadEmbeddingTextContrastiveExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text contrastive dataset: %w", err)
		}
		tokenized, err := eosruntime.TokenizeEmbeddingTextContrastiveExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize contrastive dataset: %w", err)
		}
		if err := eosruntime.WriteEmbeddingContrastiveExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized contrastive examples: %d\n", len(tokenized))
	case "pair":
		examples, err := eosruntime.ReadEmbeddingTextPairExamplesFile(inputPath)
		if err != nil {
			return fmt.Errorf("read text pair dataset: %w", err)
		}
		tokenized, err := eosruntime.TokenizeEmbeddingTextPairExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize pair dataset: %w", err)
		}
		if err := eosruntime.WriteEmbeddingPairExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized pair examples: %d\n", len(tokenized))
	case "hard-negative", "hard_negative":
		examples, err := eosruntime.ReadEmbeddingTextHardNegativeExamplesFile(inputPath)
		if err != nil {
			pairs, pairErr := eosruntime.ReadEmbeddingTextPairExamplesFile(inputPath)
			if pairErr != nil {
				return fmt.Errorf("read text hard-negative dataset: %w", err)
			}
			examples, err = eosruntime.BuildEmbeddingTextHardNegativeExamplesFromPairs(pairs, hardNegativesPerQuery)
			if err != nil {
				return fmt.Errorf("build text hard-negative dataset: %w", err)
			}
		}
		tokenized, err := eosruntime.TokenizeEmbeddingTextHardNegativeExamples(examples, tokenizer)
		if err != nil {
			return fmt.Errorf("tokenize hard-negative dataset: %w", err)
		}
		if err := eosruntime.WriteEmbeddingHardNegativeExamplesFile(outputPath, tokenized); err != nil {
			return err
		}
		fmt.Printf("tokenized hard-negative examples: %d\n", len(tokenized))
	default:
		return fmt.Errorf("unsupported tokenize mode %q: want contrastive, pair, or hard-negative", mode)
	}
	fmt.Printf("tokenizer: %s\n", tokenizerPath)
	fmt.Printf("input: %s\n", inputPath)
	fmt.Printf("output: %s\n", outputPath)
	return nil
}

func runMineTextPairs(args []string) error {
	fs := flag.NewFlagSet("mine-text-pairs", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	var minChars int
	var maxPairs int
	var evalPairs int
	var seed int64
	fs.IntVar(&minChars, "min-chars", 8, "minimum normalized text length for mined segments")
	fs.IntVar(&maxPairs, "max-pairs", 0, "maximum number of positive training pairs to keep (0 = all)")
	fs.IntVar(&evalPairs, "eval-pairs", 32, "number of positive pairs to hold out for eval")
	fs.Int64Var(&seed, "seed", 1, "shuffle seed used when truncating mined pairs")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos mine-text-pairs [flags] <corpus.txt> <train.jsonl> [eval.jsonl]")
	}
	corpusPath := fs.Arg(0)
	trainPath := fs.Arg(1)
	evalPath := ""
	if fs.NArg() > 2 {
		evalPath = fs.Arg(2)
	}
	trainSet, evalSet, err := eosruntime.MineEmbeddingTextDatasetsFromCorpusFile(corpusPath, eosruntime.EmbeddingTextMiningConfig{
		MinChars:  minChars,
		MaxPairs:  maxPairs,
		EvalPairs: evalPairs,
		Seed:      seed,
	})
	if err != nil {
		return err
	}
	if err := eosruntime.WriteEmbeddingTextContrastiveExamplesFile(trainPath, trainSet); err != nil {
		return err
	}
	if evalPath != "" && len(evalSet) > 0 {
		if err := eosruntime.WriteEmbeddingTextPairExamplesFile(evalPath, evalSet); err != nil {
			return err
		}
	}
	fmt.Printf("mined train pairs: %d\n", len(trainSet))
	if evalPath != "" {
		fmt.Printf("mined eval pairs: %d\n", len(evalSet))
	}
	fmt.Printf("train: %s\n", trainPath)
	if evalPath != "" {
		fmt.Printf("eval: %s\n", evalPath)
	}
	return nil
}

func printUsage() {
	fmt.Println("usage:")
	fmt.Println("  eos version")
	fmt.Println("  eos compile [--bundle dir] [--validate-kernels] <source.eos> [output.mll]")
	fmt.Println("  eos graph [--format json|dot] <source.eos|artifact.mll>")
	fmt.Println("  eos kernels [--backend backend] [--out dir] <source.eos|artifact.mll>")
	fmt.Println("  eos doctor")
	fmt.Println("  eos inspect <artifact.mll>")
	fmt.Println("  eos inspect-pretrained-bert-package [--json] <package.mll>")
	fmt.Println("  eos export-mll <artifact.mll> [output.mll]")
	fmt.Println("  eos import-pretrained-bert --source <hf-snapshot-dir> [--model-name name] [--plan-json plan.json] [--verify-weights] [--weights-out weights.mll] [--module-out artifact.mll]")
	fmt.Println("  eos fit-pretrained-bert-projection-head --weights-json weights.json --out head.mll [metadata flags]")
	fmt.Println("  eos embed-text <artifact.mll> <text...>")
	fmt.Println("  eos embed-pretrained-bert-package [--role raw|query|document] [--json] [--max-length N] <package.mll> <text...>")
	fmt.Println("  eos default-embedder [--root dir] [--path-only] [--verify] [--json]")
	fmt.Println("  eos imported-embedder-candidate [--root dir] [--package path] [--path-only] [--verify] [--json]")
	fmt.Println("  eos export-retrieval-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	fmt.Println("  eos export-sparse-token-pool-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	fmt.Println("  eos export-sparse-encoder-vectors [flags] <artifact.mll> <beir-dataset-dir> <output-dir>")
	fmt.Println("  eos export-sparse-lexical-labels [flags] <beir-dataset-dir> <labels.jsonl>")
	fmt.Println("  eos fit-sparse-lexical-head --labels labels.jsonl --hash-bins 65536 --head-json head.json")
	fmt.Println("  eos fit-sparse-lexical-projection-head --labels labels.jsonl --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json")
	fmt.Println("  eos fit-sparse-lexical-linear-head --labels labels.jsonl --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json")
	fmt.Println("  eos export-timeseries-vectors [flags] <artifact.mll> <series.jsonl> <queries.jsonl> <output-dir>")
	fmt.Println("  eos export-event-trace-vectors [flags] <artifact.mll> <traces.jsonl> <queries.jsonl> <output-dir>")
	fmt.Println("  eos eval-retrieval [flags] <artifact.mll> <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-hybrid [flags] <artifact.mll> <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-turboquant [flags] <artifact.mll> <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-vectors [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	fmt.Println("  eos eval-sparse-lexical-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --labels labels.jsonl --head-json head.json <beir-dataset-dir>")
	fmt.Println("  eos eval-sparse-lexical-projection-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json <beir-dataset-dir>")
	fmt.Println("  eos eval-sparse-lexical-linear-head-vectors-hybrid [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl --head-json head.json <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-vectors-turboquant [flags] --doc-vectors docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-multivector-turboquant [flags] --doc-vectors child-docs.jsonl --query-vectors queries.jsonl <beir-dataset-dir>")
	fmt.Println("  eos eval-retrieval-bm25 [flags] <beir-dataset-dir>")
	fmt.Println("  eos eval-sparse-lexical-labels [flags] --labels labels.jsonl <beir-dataset-dir>")
	fmt.Println("  eos eval-sparse-lexical-head [flags] --labels labels.jsonl --head-json head.json <beir-dataset-dir>")
	fmt.Println("  eos mine-retrieval-hard-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  eos mine-retrieval-model-hard-negatives [flags] <artifact.mll> <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  eos mine-retrieval-compact-hard-negatives [flags] --per-query-jsonl diagnostics.jsonl <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  eos export-teacher-score-requests [flags] <hard-negatives.jsonl> <requests.jsonl>")
	fmt.Println("  eos import-teacher-scores [flags] <hard-negatives.jsonl> <scores.jsonl> <output.jsonl>")
	fmt.Println("  eos score-teacher-hard-negatives [flags] <teacher.mll> <hard-negatives.jsonl> <output.jsonl>")
	fmt.Println("  eos audit-teacher-scores [flags] <hard-negatives.jsonl> [summary.json]")
	fmt.Println("  eos filter-teacher-scores [flags] <scored-hard-negatives.jsonl> <output.jsonl> [summary.json]")
	fmt.Println("  eos relabel-teacher-negatives [flags] <scored-hard-negatives.jsonl> <output.jsonl>")
	fmt.Println("  eos sample-corpus-negatives [flags] <beir-dataset-dir> <output.jsonl>")
	fmt.Println("  eos plan-sparse-attention [flags]")
	fmt.Println("  eos calibrate-sparse-routing [flags]")
	fmt.Println("  eos smoke-sparse-embedding-encoder [flags]")
	fmt.Println("  eos plan-multivector-storage [flags]")
	fmt.Println("  eos init-model [flags] <artifact.mll>")
	fmt.Println("  eos init-mirage [flags] <artifact.mll>")
	fmt.Println("  eos init-train [flags] <artifact.mll>")
	fmt.Println("  eos rename-embed [--name <model-name>] [--max-seq N] <input.mll> <output.mll>")
	fmt.Println("  eos train-tokenizer [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  eos tokenize-embed [flags] <artifact.mll> <input-text.jsonl> <output-token.jsonl>")
	fmt.Println("  eos train-corpus [flags] <artifact.mll> <corpus.txt>")
	fmt.Println("  eos train-embed [flags] <artifact.mll> <train.jsonl> [eval.jsonl]")
	fmt.Println("  eos compare-train-metrics <current.metrics.json> [baseline.metrics.json]")
	fmt.Println("  eos compare-retrieval-metrics <current.retrieval.metrics.json> <baseline.retrieval.metrics.json>")
	fmt.Println("  eos diagnose-train-metrics <metrics.json>")
	fmt.Println("  eos gate-train-metrics [flags] <metrics.json>")
	fmt.Println("  eos gate-retrieval-metrics [flags] <retrieval.metrics.json>")
	fmt.Println("  eos gate-scoreboard [flags] --datasets dataset[,dataset...] <current.scoreboard.json> <anchor.scoreboard.json>")
	fmt.Println("  eos export-pretrained-bert-retrieval-vectors [flags] <beir-dataset-dir> <output-dir>")
	fmt.Println("  eos run <artifact.mll> [entry]")
	fmt.Println("  eos demo [module-name]")
	fmt.Println()
	fmt.Println("compile lowers a Eos source file into an .mll artifact.")
	fmt.Println("graph prints compiler or artifact graph structure as JSON or DOT.")
	fmt.Println("kernels extracts backend kernel sources and a manifest for inspection.")
	fmt.Println("doctor reports Eos runtime, backend, tool, and relevant environment facts.")
	fmt.Println("inspect summarizes an artifact and verifies its sibling package manifest when present.")
	fmt.Println("export-mll seals an artifact package into a weight-carrying .mll container while preserving Eos metadata in XMTA.")
	fmt.Println("import-pretrained-bert writes a BERT-family HF config/tensor plan and optional WordPiece tokenizer smoke check; --verify-weights validates safetensors headers, --weights-out writes role-named decoded weights, and --module-out writes the host-reference executable BERT embedder module artifact.")
	fmt.Println("fit-pretrained-bert-projection-head packages a learned imported-BERT compact projection matrix as a validated MLL sidecar.")
	fmt.Println("embed-text loads a packaged or sealed embedding .mll and embeds text with its tokenizer.")
	fmt.Println("embed-pretrained-bert-package directly embeds text from a source-free imported BERT package; --role defaults to raw, query/document use the package retrieval role contract, and JSON keeps quality_claim=false.")
	fmt.Println("export-retrieval-vectors writes BEIR document/query vector caches from a packaged or sealed Eos embedding .mll, optionally as parent-child document chunks.")
	fmt.Println("export-pretrained-bert-retrieval-vectors writes BEIR document/query vector caches from an imported BERT embedder module, role-named weights, and a Hugging Face WordPiece tokenizer snapshot; --projection-head applies a learned compact sidecar; quality_claim=false.")
	fmt.Println("export-sparse-token-pool-vectors writes experimental_sparse_token_pool BEIR vector caches from tokenizer ids, token_embedding rows, and host-reference attention (--attention-mode turboquant_sparse|dense); --token-span-tokens emits child vectors from one encoded document pass; quality_claim=false; --min-observed-doc-tokens can fail runs that never consume the requested document token length.")
	fmt.Println("export-sparse-encoder-vectors writes parent doc/query vector caches as experimental_sparse_encoder_host_reference; it requires full manifest encoder weights, records retrieval_cache_host_reference_sparse_encoder evidence, and keeps quality_claim=false.")
	fmt.Println("export-sparse-lexical-labels writes deterministic BM25 sparse lexical document/query labels plus an oracle manifest; it is non-training and does not modify embedding assets.")
	fmt.Println("fit-sparse-lexical-head writes an experimental non-default hashed-bin sparse lexical sidecar from manta.sparse_lexical_labels.v1 labels.")
	fmt.Println("fit-sparse-lexical-projection-head writes an experimental non-default prototype projection sidecar that learns hashed sparse-bin prototypes from labels plus dense vector caches.")
	fmt.Println("fit-sparse-lexical-linear-head writes an experimental non-default trainable linear sidecar that learns retained hashed sparse bins from labels plus dense vector caches.")
	fmt.Println("export-timeseries-vectors writes text-rendered time-series window child-vector caches plus query vectors for the multivector TurboQuant quality harness.")
	fmt.Println("export-event-trace-vectors writes text-rendered event/trace child-vector caches plus query vectors for the multivector TurboQuant quality harness.")
	fmt.Println("eval-retrieval scores a sealed embedding .mll on BEIR-style corpus/query/qrels files with nDCG/MRR/Recall metrics.")
	fmt.Println("eval-retrieval-hybrid fuses sealed embedding dense top-k with BM25 top-k using minmax, zscore, or RRF scoring.")
	fmt.Println("eval-retrieval-turboquant compares dense retrieval quality/cost against TurboQuant IP-preserving quantized document vectors.")
	fmt.Println("eval-retrieval-vectors scores precomputed document/query vector JSONL caches on the same BEIR metrics.")
	fmt.Println("eval-retrieval-vectors-hybrid fuses external dense vector caches with BM25 top-k over the same BEIR files.")
	fmt.Println("eval-sparse-lexical-head-vectors-hybrid fuses external dense vector caches with the experimental hashed-bin sparse lexical sidecar.")
	fmt.Println("eval-sparse-lexical-projection-head-vectors-hybrid predicts sparse bins from dense vectors using the experimental projection sidecar, without eval-time sparse labels.")
	fmt.Println("eval-sparse-lexical-linear-head-vectors-hybrid predicts sparse bins from dense vectors using the experimental linear sidecar, without eval-time sparse labels.")
	fmt.Println("eval-retrieval-vectors-turboquant compares external vector caches against TurboQuant IP-preserving document-vector compression.")
	fmt.Println("eval-retrieval-multivector-turboquant compares dense and direct TurboQuant child-vector scoring aggregated by max child score per parent.")
	fmt.Println("eval-retrieval-bm25 scores the same BEIR files with an in-repo BM25 lexical baseline.")
	fmt.Println("eval-sparse-lexical-labels scores capped exported manta.sparse_lexical_labels.v1 query labels against the label-file document universe and BEIR qrels.")
	fmt.Println("eval-sparse-lexical-head scores the experimental hashed-bin sparse lexical sidecar over the label-file document universe and BEIR qrels.")
	fmt.Println("mine-retrieval-hard-negatives creates text hard-negative training JSONL from BEIR qrels using the BM25 baseline.")
	fmt.Println("mine-retrieval-model-hard-negatives creates text hard-negative training JSONL from BEIR qrels using a Eos embedding model's own misses.")
	fmt.Println("mine-retrieval-compact-hard-negatives creates text hard-negative JSONL from compact TurboQuant per-query diagnostics and rejects test-split train selection by default.")
	fmt.Println("export-teacher-score-requests writes per-candidate JSONL rows for external teachers to score before import-teacher-scores.")
	fmt.Println("import-teacher-scores merges external teacher score JSONL into text hard-negative JSONL and writes a provenance manifest.")
	fmt.Println("score-teacher-hard-negatives uses a Eos embedding teacher to score existing text hard-negative JSONL into teacher_scores.")
	fmt.Println("audit-teacher-scores summarizes teacher score coverage, positive rank, margins, and entropy before distillation runs.")
	fmt.Println("filter-teacher-scores clears unsafe teacher_scores when the teacher does not rank the labeled positive above hard negatives, preserving examples for base hard-negative training.")
	fmt.Println("relabel-teacher-negatives promotes teacher-confirmed-relevant mined negatives to positive rows, keeps teacher-confirmed-irrelevant candidates as negatives, and drops the ambiguous band.")
	fmt.Println("sample-corpus-negatives emits random non-qrel corpus documents per query for teacher scoring into a true-negative pool.")
	fmt.Println("plan-sparse-attention preflights routed sparse attention plus logical TurboQuant K/V memory budgets before GPU runs.")
	fmt.Println("calibrate-sparse-routing sweeps sparse routing policy budgets, including optional calibration-only oracle policies, on synthetic tensors and writes router recall, output delta, and score-work artifacts.")
	fmt.Println("smoke-sparse-embedding-encoder runs a deterministic routed TurboQuant sparse-attention encoder-shaped smoke and writes manifest.json, summary.tsv, scorecard.json, and scorecard.tsv.")
	fmt.Println("plan-multivector-storage estimates how many TurboQuant child vectors per parent fit in one dense fp32 baseline-vector budget; use --baseline-dim to compare compact children against a larger dense baseline, and --series-lengths with --window-size/--window-stride to derive vectors per object from time-series windows.")
	fmt.Println("init-model creates the Eos-owned default quantized embedding training package.")
	fmt.Println("init-mirage creates the Eos-owned Mirage Image v1 host-reference artifact.")
	fmt.Println("init-train creates a native training package next to an artifact.")
	fmt.Println("rename-embed rewrites a training package under a new embedding model identity and/or diagnostic tokenizer max-sequence contract.")
	fmt.Println("train-tokenizer builds a sibling .tokenizer.mll from a raw text corpus, using embedding-manifest vocab_size by default.")
	fmt.Println("tokenize-embed converts text JSONL into reusable token JSONL for contrastive, pair, or hard-negative training and eval.")
	fmt.Println("train-corpus trains tokenizer + mined text pairs + embedder in one Eos job from a raw text corpus.")
	fmt.Println("train-embed reloads a training package, fits or --eval-only evaluates token JSONL or text JSONL (with --tokenizer or a sibling .tokenizer.mll; use --no-tokenizer for token JSONL beside a tokenizer), and writes it back.")
	fmt.Println("compare-train-metrics summarizes metrics JSON and prints deltas against a baseline metrics JSON when provided.")
	fmt.Println("compare-retrieval-metrics summarizes retrieval quality deltas and can gate a candidate against a baseline.")
	fmt.Println("diagnose-train-metrics explains backend use, transfer pressure, and suspicious training/eval counters from metrics JSON.")
	fmt.Println("gate-train-metrics checks metrics JSON against EOS_* thresholds from the environment or a thresholds env file.")
	fmt.Println("gate-retrieval-metrics checks BEIR retrieval metrics against dataset-specific EOS_* thresholds.")
	fmt.Println("gate-scoreboard checks every selected retrieval scoreboard row and metric against an anchor scoreboard.")
	fmt.Println("run loads an artifact, binds stub weights and inputs, and executes one entrypoint.")
	fmt.Println("demo creates a tiny inference-style module and loads it through the runtime.")
}

func totalKernelOps(kernels []eosartifact.Kernel) int {
	total := 0
	for _, kernel := range kernels {
		total += len(kernel.Body)
	}
	return total
}

func defaultArtifactPath(srcPath string) string {
	ext := filepath.Ext(srcPath)
	if ext == "" {
		return srcPath + ".mll"
	}
	return strings.TrimSuffix(srcPath, ext) + ".mll"
}

func stubLoadOptions(mod *eosartifact.Module) []eosruntime.LoadOption {
	sizes := defaultSymbolSizes(mod)
	opts := make([]eosruntime.LoadOption, 0, len(mod.Params))
	for _, param := range mod.Params {
		opts = append(opts, eosruntime.WithWeight(param.Name, stubTensorForParam(param.Name, param.Type, sizes)))
	}
	return opts
}

func defaultEntryName(mod *eosartifact.Module) string {
	if mod != nil && len(mod.EntryPoints) > 0 {
		return mod.EntryPoints[0].Name
	}
	return ""
}

func entryPointByName(mod *eosartifact.Module, name string) (eosartifact.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return eosartifact.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func stubInputs(entry eosartifact.EntryPoint) map[string]any {
	sizes := defaultShapeSizes(entry)
	out := make(map[string]any, len(entry.Inputs))
	for _, input := range entry.Inputs {
		if input.Type.Kind == eosartifact.ValueKVCache {
			out[input.Name] = backend.NewKVCache(backend.NewTensorF16([]int{sizes["T"], sizes["D"]}, make([]float32, sizes["T"]*sizes["D"])))
			continue
		}
		out[input.Name] = stubTensorForInput(input.Name, input.Type, sizes)
	}
	return out
}

func displayTrainBackend(kind eosartifact.BackendKind) string {
	if kind == "" {
		return "host"
	}
	return string(kind)
}

func displayArtifactVersion(version string) string {
	return version
}

func displayManifestName(value string) string {
	if value == "" {
		return "-"
	}
	return value
}

func joinBackendKinds(kinds []eosartifact.BackendKind) string {
	if len(kinds) == 0 {
		return ""
	}
	out := make([]string, 0, len(kinds))
	for _, kind := range kinds {
		out = append(out, string(kind))
	}
	return strings.Join(out, ", ")
}

func sortedValueKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func outputSummaries(outputs map[string]backend.Value) []string {
	keys := sortedValueKeys(outputs)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		value := outputs[key]
		tensor, ok := value.Data.(*backend.Tensor)
		if ok && tensor != nil {
			out = append(out, fmt.Sprintf("%s=%s%v", key, tensor.DType, tensor.Shape))
			continue
		}
		if pack, ok := value.Data.(*backend.CandidatePack); ok && pack != nil {
			out = append(out, fmt.Sprintf("%s=candidate_pack(ids=i64%v,scores=f32%v,docs=%s%v)", key, pack.IDs.Shape, pack.Scores.Shape, pack.Docs.DType, pack.Docs.Shape))
			continue
		}
		out = append(out, key+"=<non-tensor>")
	}
	return out
}

type dimFlag struct {
	pairs []string
}

func (f *dimFlag) String() string {
	return strings.Join(f.pairs, ",")
}

func (f *dimFlag) Set(value string) error {
	if value == "" {
		return fmt.Errorf("empty dimension binding")
	}
	parts := strings.SplitN(value, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf("invalid dimension binding %q, want NAME=VALUE", value)
	}
	if _, err := strconv.Atoi(parts[1]); err != nil {
		return fmt.Errorf("invalid dimension binding %q: %w", value, err)
	}
	f.pairs = append(f.pairs, value)
	return nil
}

func (f *dimFlag) values() map[string]int {
	if len(f.pairs) == 0 {
		return nil
	}
	out := make(map[string]int, len(f.pairs))
	for _, pair := range f.pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			continue
		}
		n, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		out[parts[0]] = n
	}
	return out
}

func defaultSymbolSizes(mod *eosartifact.Module) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	if mod == nil {
		return sizes
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			continue
		}
		for _, dim := range param.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	for _, entry := range mod.EntryPoints {
		for _, input := range entry.Inputs {
			if input.Type.Tensor == nil {
				continue
			}
			for _, dim := range input.Type.Tensor.Shape {
				if _, ok := sizes[dim]; !ok {
					sizes[dim] = 2
				}
			}
		}
	}
	return sizes
}

func defaultShapeSizes(entry eosartifact.EntryPoint) map[string]int {
	sizes := map[string]int{
		"V":  3,
		"D":  2,
		"E":  2,
		"T":  2,
		"N":  2,
		"K":  2,
		"H":  2,
		"HD": 2,
	}
	for _, input := range entry.Inputs {
		if input.Type.Tensor == nil {
			continue
		}
		for _, dim := range input.Type.Tensor.Shape {
			if _, ok := sizes[dim]; !ok {
				sizes[dim] = 2
			}
		}
	}
	return sizes
}

func stubTensorForParam(name string, typ eosartifact.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	switch name {
	case "token_embedding":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
			1, 1,
		})
	case "projection", "wq":
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	case "docs":
		return backend.NewTensorQ4(shape, []float32{
			1, 0,
			0, 1,
		})
	default:
		return fillTensor(typ, shape, 1)
	}
}

func stubTensorForInput(name string, typ eosartifact.ValueType, sizes map[string]int) *backend.Tensor {
	shape := concreteShape(typ, sizes)
	if typ.Tensor != nil && typ.Tensor.DType == "i32" {
		values := make([]int32, product(shape))
		if name == "attention_mask" {
			for i := range values {
				values[i] = 1
			}
			return backend.NewTensorI32(shape, values)
		}
		limit := sizes["V"]
		if limit <= 0 {
			limit = len(values)
		}
		for i := range values {
			values[i] = int32(i % limit)
		}
		return backend.NewTensorI32(shape, values)
	}
	if name == "x" {
		if product(shape) != 4 {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "query" {
		values := make([]float32, product(shape))
		if len(values) > 0 {
			values[0] = 1
		}
		return backend.NewTensorF16(shape, values)
	}
	if name == "queries" && typ.Tensor != nil && typ.Tensor.DType == "f16" && len(shape) == 2 && shape[0] == 2 && shape[1] == 2 {
		return backend.NewTensorF16(shape, []float32{
			1, 0,
			0, 1,
		})
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 2 && shape[1] == 2 {
		values := []float32{1, 0, 0, 1}
		if shape[0] >= 3 {
			values = []float32{1, 0, 0, 1, 1, 1}
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "docs" && typ.Tensor != nil && typ.Tensor.DType == "q4" && len(shape) == 3 && shape[0] == 2 && shape[2] == 2 {
		values := []float32{
			1, 0,
			0, 1,
			1, 1,
			0, 1,
			1, 0,
			1, 1,
		}
		if product(shape) > len(values) {
			return fillTensor(typ, shape, 0)
		}
		return backend.NewTensorQ4(shape, values[:product(shape)])
	}
	if name == "candidate_ids" && typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(1001 + i*1001)
		}
		return backend.NewTensorI64(shape, values)
	}
	if typ.Tensor != nil && typ.Tensor.DType == "i64" {
		values := make([]int64, product(shape))
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	}
	return fillTensor(typ, shape, 0)
}

func fillTensor(typ eosartifact.ValueType, shape []int, offset float32) *backend.Tensor {
	n := product(shape)
	switch typ.Tensor.DType {
	case "i32":
		values := make([]int32, n)
		for i := range values {
			values[i] = int32(i)
		}
		return backend.NewTensorI32(shape, values)
	case "i64":
		values := make([]int64, n)
		for i := range values {
			values[i] = int64(i)
		}
		return backend.NewTensorI64(shape, values)
	case "f16":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF16(shape, values)
	case "q2", "q4", "q_norm":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		if typ.Tensor.DType == "q2" {
			return backend.NewTensorQ2(shape, values)
		}
		if typ.Tensor.DType == "q_norm" {
			return backend.NewTensorQNorm(shape, values)
		}
		return backend.NewTensorQ4(shape, values)
	case "q8":
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorQ8(shape, values)
	default:
		values := make([]float32, n)
		for i := range values {
			values[i] = offset + float32(i+1)/10
		}
		return backend.NewTensorF32(shape, values)
	}
}

func concreteShape(typ eosartifact.ValueType, sizes map[string]int) []int {
	if typ.Tensor == nil {
		return []int{1}
	}
	shape := make([]int, len(typ.Tensor.Shape))
	for i, dim := range typ.Tensor.Shape {
		if literal, err := strconv.Atoi(dim); err == nil {
			shape[i] = literal
			continue
		}
		n, ok := sizes[dim]
		if !ok {
			n = 2
		}
		shape[i] = n
	}
	return shape
}

func product(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	return n
}
