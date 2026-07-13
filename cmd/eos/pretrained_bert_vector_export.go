package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	eosruntime "m31labs.dev/eos/runtime"
	"m31labs.dev/eos/runtime/backends/cuda"
	"m31labs.dev/eos/runtime/backends/directml"
	"m31labs.dev/eos/runtime/backends/metal"
	"m31labs.dev/eos/runtime/backends/vulkan"
	"m31labs.dev/eos/runtime/backends/webgpu"
)

func runExportPretrainedBERTRetrievalVectors(args []string) error {
	fs := flag.NewFlagSet("export-pretrained-bert-retrieval-vectors", flag.ContinueOnError)
	fs.SetOutput(flag.CommandLine.Output())
	sourceDir := fs.String("source", "", "local Hugging Face BERT-family snapshot directory containing config.json and vocab.txt")
	modulePath := fs.String("module", "", "Eos MLL BERT embedder module artifact produced by import-pretrained-bert --module-out")
	weightsPath := fs.String("weights", "", "Eos MLL role-named BERT weights file produced by import-pretrained-bert --weights-out")
	packagePath := fs.String("package", "", "source-free imported BERT package produced by import-pretrained-bert --package-out")
	datasetName := fs.String("dataset", "", "dataset name for manifest/status output")
	split := fs.String("split", "test", "qrels split under <beir-dataset-dir>/qrels")
	qrelsPath := fs.String("qrels", "", "explicit qrels TSV path; when present, export keeps qrels-relevant docs/queries under caps")
	queryPrefix := fs.String("query-prefix", "", "prefix prepended to query text before WordPiece tokenization")
	documentPrefix := fs.String("document-prefix", "", "prefix prepended to document text before WordPiece tokenization")
	docPrefix := fs.String("doc-prefix", "", "compatibility alias for --document-prefix")
	usePackageRoleContract := fs.Bool("use-package-role-contract", false, "when using --package, load query/document prefixes from the package retrieval role contract")
	batchSize := fs.Int("batch-size", 64, "embedding batch size")
	outputDim := fs.Int("output-dim", 0, "when positive, prefix-truncate embeddings to this dimension and L2-renormalize before writing")
	projectionHeadPath := fs.String("projection-head", "", "MLL projection head sidecar to apply to native embeddings before writing")
	requireDeviceEncoder := fs.Bool("require-device-encoder", false, "fail unless validated execution provenance proves the imported BERT encoder runs fully on device")
	maxDocs := fs.Int("max-docs", 0, "limit corpus documents for smoke exports")
	maxQueries := fs.Int("max-queries", 0, "limit queries for smoke exports")
	maxLength := fs.Int("max-length", 0, "WordPiece max sequence length; default uses config max_position_embeddings")
	manifestPath := fs.String("manifest-json", "", "write export summary JSON manifest; default is <output-dir>/manifest.json")
	resume := fs.Bool("resume", false, "reuse a valid prefix of existing vector cache JSONL files and append missing rows")
	progressEvery := fs.Int("progress-every", 0, "emit export progress every N completed records; default off")
	if err := fs.Parse(args); err != nil {
		return err
	}
	resolvedDocumentPrefix, err := resolvePretrainedBERTDocumentPrefix(fs, *documentPrefix, *docPrefix)
	if err != nil {
		return err
	}
	if fs.NArg() != 2 || fs.Arg(0) == "" || fs.Arg(1) == "" {
		return fmt.Errorf("usage: eos export-pretrained-bert-retrieval-vectors (--package <imported.mll> | --source <hf-snapshot> --module <bert_embed.mll> --weights <weights.mll>) [flags] <beir-dataset-dir> <output-dir>")
	}
	if *packagePath == "" && (*sourceDir == "" || *modulePath == "" || *weightsPath == "") {
		return fmt.Errorf("--source, --module, and --weights are required unless --package is provided")
	}
	if *packagePath != "" && (*sourceDir != "" || *modulePath != "" || *weightsPath != "") {
		return fmt.Errorf("--package cannot be combined with --source, --module, or --weights")
	}
	datasetDir := fs.Arg(0)
	outputDir := fs.Arg(1)
	if *datasetName == "" {
		*datasetName = filepath.Base(datasetDir)
	}
	_, _, defaultQrelsPath := eosruntime.BEIRRetrievalPaths(datasetDir, *split)
	if *qrelsPath == "" {
		*qrelsPath = defaultQrelsPath
	}
	rt := eosruntime.New(cuda.New(), metal.New(), vulkan.New(), directml.New(), webgpu.New())
	summary, err := eosruntime.ExportPretrainedBERTRetrievalVectors(context.Background(), eosruntime.PretrainedBERTRetrievalVectorExportConfig{
		DatasetName:            *datasetName,
		DatasetDir:             datasetDir,
		QrelsPath:              *qrelsPath,
		OutputDir:              outputDir,
		SourceDir:              *sourceDir,
		ModulePath:             *modulePath,
		WeightsPath:            *weightsPath,
		PackagePath:            *packagePath,
		QueryPrefix:            *queryPrefix,
		DocumentPrefix:         resolvedDocumentPrefix,
		QueryPrefixSet:         flagWasProvided(fs, "query-prefix"),
		DocumentPrefixSet:      flagWasProvided(fs, "document-prefix") || flagWasProvided(fs, "doc-prefix"),
		UsePackageRoleContract: *usePackageRoleContract,
		BatchSize:              *batchSize,
		OutputDim:              *outputDim,
		ProjectionHeadPath:     *projectionHeadPath,
		MaxDocs:                *maxDocs,
		MaxQueries:             *maxQueries,
		MaxLength:              *maxLength,
		Split:                  *split,
		Runtime:                rt,
		ManifestJSONPath:       *manifestPath,
		Resume:                 *resume,
		RequireDeviceEncoder:   *requireDeviceEncoder,
		ProgressEvery:          *progressEvery,
		Progress:               printPretrainedBERTVectorExportProgress,
	})
	if err != nil {
		return err
	}
	fmt.Printf("exported pretrained BERT retrieval vectors: dataset=%s mode=%s docs=%d queries=%d dim=%d\n", summary.Dataset, summary.ExecutionMode, summary.Documents, summary.Queries, summary.OutputDim)
	if summary.ProjectionHeadPath != "" {
		fmt.Printf("projection_head: %s identity=%s sha256=%s\n", summary.ProjectionHeadPath, summary.ProjectionHeadIdentity, summary.ProjectionHeadSHA256)
	}
	fmt.Printf("doc_vectors: %s\n", summary.DocVectorPath)
	fmt.Printf("query_vectors: %s\n", summary.QueryVectorPath)
	if summary.Resume || summary.ProgressEvery > 0 {
		fmt.Printf("resume: enabled=%t progress_every=%d reused_documents=%d reused_queries=%d written_documents=%d written_queries=%d\n", summary.Resume, summary.ProgressEvery, summary.ReusedDocuments, summary.ReusedQueries, summary.WrittenDocuments, summary.WrittenQueries)
	}
	if *manifestPath != "" {
		fmt.Printf("manifest: %s\n", *manifestPath)
	} else {
		fmt.Printf("manifest: %s\n", filepath.Join(outputDir, "manifest.json"))
	}
	return nil
}

func resolvePretrainedBERTDocumentPrefix(fs *flag.FlagSet, documentPrefix, docPrefix string) (string, error) {
	documentPrefixSet := flagWasProvided(fs, "document-prefix")
	docPrefixSet := flagWasProvided(fs, "doc-prefix")
	if documentPrefixSet && docPrefixSet && documentPrefix != docPrefix {
		return "", fmt.Errorf("--document-prefix and --doc-prefix differ")
	}
	if documentPrefixSet {
		return documentPrefix, nil
	}
	return docPrefix, nil
}

func printPretrainedBERTVectorExportProgress(progress eosruntime.PretrainedBERTRetrievalVectorExportProgress) {
	fmt.Fprintf(os.Stderr, "export-pretrained-bert-retrieval-vectors progress: kind=%s completed=%d/%d reused=%d written=%d elapsed=%.1fs path=%s\n", progress.Kind, progress.Completed, progress.Total, progress.Reused, progress.Written, progress.ElapsedSeconds, progress.Path)
}
