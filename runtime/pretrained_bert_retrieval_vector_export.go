package eosruntime

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const PretrainedBERTRetrievalVectorExportManifestSchema = "manta.pretrained_bert_retrieval_vector_export.v1"

// PretrainedBERTTextEmbedder runs an imported BERT-family embedder module with
// Hugging Face WordPiece tokenization and role-named weight files.
type PretrainedBERTTextEmbedder struct {
	config                PretrainedBERTConfig
	stMeta                *PretrainedBERTSTMetadata
	tokenizer             *HFWordPieceTokenizer
	program               *Program
	maxLength             int
	maxLengthSource       string
	pooling               string
	normalization         string
	modelName             string
	architecture          string
	dynamicTokens         bool
	packagePath           string
	packageSHA256         string
	packageIdentity       string
	packageFileSHA        map[string]string
	retrievalRoleContract *PretrainedBERTRetrievalRoleContract
	executionMu           sync.Mutex
	executionStats        pretrainedBERTEncoderExecutionStats
}

type PretrainedBERTTextEmbedderConfig struct {
	SourceDir   string
	ModulePath  string
	WeightsPath string
	PackagePath string
	MaxLength   int
	Runtime     *Runtime
}

type PretrainedBERTRetrievalVectorExportConfig struct {
	DatasetName            string
	DatasetDir             string
	CorpusPath             string
	QueriesPath            string
	QrelsPath              string
	OutputDir              string
	SourceDir              string
	ModulePath             string
	WeightsPath            string
	PackagePath            string
	QueryPrefix            string
	DocumentPrefix         string
	QueryPrefixSet         bool
	DocumentPrefixSet      bool
	UsePackageRoleContract bool
	BatchSize              int
	OutputDim              int
	MaxDocs                int
	MaxQueries             int
	MaxLength              int
	Split                  string
	Runtime                *Runtime
	ManifestJSONPath       string
	ProjectionHeadPath     string
	RequireDeviceEncoder   bool
	Resume                 bool
	ProgressEvery          int
	Progress               PretrainedBERTRetrievalVectorExportProgressFunc
}

type PretrainedBERTEncoderExecutionProvenance struct {
	ExecutionMode                      string `json:"execution_mode"`
	SelectedBackend                    string `json:"selected_backend,omitempty"`
	FullDeviceExecution                bool   `json:"full_device_execution"`
	ValidatedDeviceEncoder             bool   `json:"validated_device_encoder"`
	HostReference                      bool   `json:"host_reference"`
	OpportunisticDeviceOpsIgnored      bool   `json:"opportunistic_device_ops_ignored"`
	OpportunisticDeviceOpsUsed         bool   `json:"opportunistic_device_ops_used"`
	DenseAccelerationPolicy            string `json:"dense_acceleration_policy"`
	CUDADenseAccelerationObserved      bool   `json:"cuda_dense_acceleration_observed"`
	CUDADenseMatMulRuns                int64  `json:"cuda_dense_matmul_runs"`
	CUDADenseMatMulUploadedBytes       int64  `json:"cuda_dense_matmul_uploaded_bytes"`
	HostReferenceExecutionStatus       string `json:"host_reference_execution_status,omitempty"`
	DeviceEncoderContract              string `json:"device_encoder_contract"`
	DeviceEncoderContractSatisfied     bool   `json:"device_encoder_contract_satisfied"`
	DeviceEncoderContractMissingReason string `json:"device_encoder_contract_missing_reason,omitempty"`
	DeviceEncoderValidationFailureHint string `json:"device_encoder_validation_failure_hint,omitempty"`
}

type pretrainedBERTEncoderExecutionStats struct {
	cudaDenseMatMulRuns          int64
	cudaDenseMatMulUploadedBytes int64
	hostReferenceExecutionStatus string
}

type PretrainedBERTRetrievalVectorExportSummary struct {
	Schema                            string                                   `json:"schema"`
	Dataset                           string                                   `json:"dataset"`
	SourceDir                         string                                   `json:"source_dir"`
	ModulePath                        string                                   `json:"module_path"`
	WeightsPath                       string                                   `json:"weights_path"`
	PackagePath                       string                                   `json:"package_path,omitempty"`
	PackageSHA256                     string                                   `json:"package_sha256,omitempty"`
	PackageIdentitySHA256             string                                   `json:"package_identity_sha256,omitempty"`
	ExecutionMode                     string                                   `json:"execution_mode"`
	EncoderExecution                  PretrainedBERTEncoderExecutionProvenance `json:"encoder_execution"`
	QualityClaim                      bool                                     `json:"quality_claim"`
	EmbeddingSpaceID                  string                                   `json:"embedding_space_id,omitempty"`
	ProjectionHeadPath                string                                   `json:"projection_head_path,omitempty"`
	ProjectionHeadSHA256              string                                   `json:"projection_head_sha256,omitempty"`
	ProjectionHeadIdentity            string                                   `json:"projection_head_identity,omitempty"`
	ProjectionHeadSchema              string                                   `json:"projection_head_schema,omitempty"`
	ProjectionHeadSourceModel         string                                   `json:"projection_head_source_model,omitempty"`
	ModuleSHA256                      string                                   `json:"module_sha256,omitempty"`
	WeightsSHA256                     string                                   `json:"weights_sha256,omitempty"`
	ConfigSHA256                      string                                   `json:"config_sha256,omitempty"`
	VocabSHA256                       string                                   `json:"vocab_sha256,omitempty"`
	TokenizerJSONSHA256               string                                   `json:"tokenizer_json_sha256,omitempty"`
	TokenizerConfigSHA256             string                                   `json:"tokenizer_config_sha256,omitempty"`
	SpecialTokensMapSHA256            string                                   `json:"special_tokens_map_sha256,omitempty"`
	SentenceTransformersPoolingSHA256 string                                   `json:"sentence_transformers_pooling_sha256,omitempty"`
	SentenceTransformersConfigSHA256  string                                   `json:"sentence_transformers_config_sha256,omitempty"`
	Normalization                     string                                   `json:"normalization,omitempty"`
	Documents                         int                                      `json:"documents"`
	Queries                           int                                      `json:"queries"`
	NativeDim                         int                                      `json:"native_dim"`
	OutputDim                         int                                      `json:"output_dim"`
	DocVectorPath                     string                                   `json:"doc_vector_path"`
	QueryVectorPath                   string                                   `json:"query_vector_path"`
	QueryPrefix                       string                                   `json:"query_prefix"`
	DocumentPrefix                    string                                   `json:"document_prefix"`
	LegacyDocPrefix                   string                                   `json:"doc_prefix"`
	DocumentRoleApplied               bool                                     `json:"document_role_applied"`
	QueryRoleApplied                  bool                                     `json:"query_role_applied"`
	Resume                            bool                                     `json:"resume"`
	ProgressEvery                     int                                      `json:"progress_every,omitempty"`
	ReusedDocuments                   int                                      `json:"reused_documents,omitempty"`
	ReusedQueries                     int                                      `json:"reused_queries,omitempty"`
	WrittenDocuments                  int                                      `json:"written_documents"`
	WrittenQueries                    int                                      `json:"written_queries"`
	DocumentBatching                  PretrainedBERTVectorBatchingSummary      `json:"document_batching"`
	QueryBatching                     PretrainedBERTVectorBatchingSummary      `json:"query_batching"`
	MaxLength                         int                                      `json:"max_length"`
	MaxLengthSource                   string                                   `json:"max_length_source,omitempty"`
	Pooling                           string                                   `json:"pooling,omitempty"`
	BatchSize                         int                                      `json:"batch_size"`
	MaxDocs                           int                                      `json:"max_docs,omitempty"`
	MaxQueries                        int                                      `json:"max_queries,omitempty"`
	CorpusPath                        string                                   `json:"corpus_path,omitempty"`
	QueriesPath                       string                                   `json:"queries_path,omitempty"`
	QrelsPath                         string                                   `json:"qrels_path,omitempty"`
	ElapsedSeconds                    float64                                  `json:"elapsed_seconds"`
	CreatedAt                         time.Time                                `json:"created_at"`
}

type PretrainedBERTVectorBatchingSummary struct {
	Strategy           string `json:"strategy"`
	BatchSize          int    `json:"batch_size"`
	BatchCount         int    `json:"batch_count"`
	Items              int    `json:"items"`
	TotalItems         int    `json:"total_items"`
	ReusedItems        int    `json:"reused_items,omitempty"`
	ComputedItems      int    `json:"computed_items"`
	ActualTokens       int64  `json:"actual_tokens"`
	PaddedTokens       int64  `json:"padded_tokens"`
	FixedMaxTokens     int64  `json:"fixed_max_tokens"`
	MaxEffectiveLength int    `json:"max_effective_length"`
}

func (s PretrainedBERTVectorBatchingSummary) PaddingRatio() float64 {
	if s.FixedMaxTokens <= 0 {
		return 0
	}
	return float64(s.PaddedTokens) / float64(s.FixedMaxTokens)
}

type PretrainedBERTRetrievalVectorExportProgressFunc func(PretrainedBERTRetrievalVectorExportProgress)

type PretrainedBERTRetrievalVectorExportProgress struct {
	Kind           string
	Path           string
	Completed      int
	Total          int
	Reused         int
	Written        int
	ElapsedSeconds float64
}

func LoadPretrainedBERTTextEmbedder(ctx context.Context, cfg PretrainedBERTTextEmbedderConfig) (*PretrainedBERTTextEmbedder, error) {
	if cfg.Runtime == nil {
		return nil, fmt.Errorf("runtime is required")
	}
	if cfg.PackagePath != "" {
		return loadPretrainedBERTTextEmbedderFromPackage(ctx, cfg)
	}
	if cfg.SourceDir == "" || cfg.ModulePath == "" || cfg.WeightsPath == "" {
		return nil, fmt.Errorf("source dir, module path, and weights path are required")
	}
	config, err := LoadPretrainedBERTConfig(filepath.Join(cfg.SourceDir, "config.json"))
	if err != nil {
		return nil, err
	}
	stMeta, err := LoadPretrainedBERTSentenceTransformersMetadata(cfg.SourceDir)
	if err != nil {
		return nil, err
	}
	maxLength, maxLengthSource := resolvePretrainedBERTMaxLength(cfg.MaxLength, config, stMeta)
	if maxLength <= 0 || maxLength > config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("max length must be in [1,%d], got %d", config.MaxPositionEmbeddings, maxLength)
	}
	tokenizer, err := LoadHFWordPieceTokenizerFromDir(cfg.SourceDir)
	if err != nil {
		return nil, err
	}
	module, err := eosartifact.ReadFile(cfg.ModulePath)
	if err != nil {
		return nil, err
	}
	pooling := ""
	if value, ok := module.Metadata["pooling"].(string); ok {
		pooling = value
	}
	normalization := ""
	if value, ok := module.Metadata["normalization"].(string); ok {
		normalization = value
	}
	modelName := ""
	if value, ok := module.Metadata["model_name"].(string); ok {
		modelName = value
	}
	architecture := ""
	if value, ok := module.Metadata["architecture"].(string); ok {
		architecture = value
	}
	weights, err := ReadWeightFile(cfg.WeightsPath)
	if err != nil {
		return nil, err
	}
	program, err := cfg.Runtime.Load(ctx, module, weights.LoadOptions()...)
	if err != nil {
		return nil, err
	}
	return &PretrainedBERTTextEmbedder{
		config:          config,
		stMeta:          stMeta,
		tokenizer:       tokenizer,
		program:         program,
		maxLength:       maxLength,
		maxLengthSource: maxLengthSource,
		pooling:         pooling,
		normalization:   normalization,
		modelName:       modelName,
		architecture:    architecture,
		dynamicTokens:   pretrainedBERTProgramSupportsDynamicTokens(program, "bert_embed"),
	}, nil
}

func loadPretrainedBERTTextEmbedderFromPackage(ctx context.Context, cfg PretrainedBERTTextEmbedderConfig) (*PretrainedBERTTextEmbedder, error) {
	pkg, packageSHA, err := readPretrainedBERTPackageFileWithSHA256(cfg.PackagePath)
	if err != nil {
		return nil, err
	}
	return loadPretrainedBERTTextEmbedderFromValidatedPackage(ctx, cfg, pkg, packageSHA)
}

func loadPretrainedBERTTextEmbedderFromValidatedPackage(ctx context.Context, cfg PretrainedBERTTextEmbedderConfig, pkg PretrainedBERTPackage, packageSHA string) (*PretrainedBERTTextEmbedder, error) {
	config := pkg.Config
	stMeta := pkg.STMetadata
	maxLength, maxLengthSource := resolvePretrainedBERTMaxLength(cfg.MaxLength, config, stMeta)
	if maxLength <= 0 || maxLength > config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("max length must be in [1,%d], got %d", config.MaxPositionEmbeddings, maxLength)
	}
	tokenizer, err := pkg.Tokenizer()
	if err != nil {
		return nil, err
	}
	module, _, err := pkg.moduleWithRuntimeProvenance(packageSHA)
	if err != nil {
		return nil, err
	}
	pooling := pkg.Pooling
	if value, ok := module.Metadata["pooling"].(string); ok && value != "" {
		pooling = value
	}
	normalization := pkg.Normalization
	if value, ok := module.Metadata["normalization"].(string); ok && value != "" {
		normalization = value
	}
	modelName := pkg.ModelName
	if value, ok := module.Metadata["model_name"].(string); ok && value != "" {
		modelName = value
	}
	architecture := pkg.Architecture
	if value, ok := module.Metadata["architecture"].(string); ok && value != "" {
		architecture = value
	}
	weights, err := pkg.Weights()
	if err != nil {
		return nil, err
	}
	program, err := cfg.Runtime.Load(ctx, module, weights.LoadOptions()...)
	if err != nil {
		return nil, err
	}
	fileHashes := make(map[string]string, len(pkg.Files))
	for _, file := range pkg.Files {
		fileHashes[file.Role] = file.SHA256
	}
	return &PretrainedBERTTextEmbedder{
		config:                config,
		stMeta:                stMeta,
		tokenizer:             tokenizer,
		program:               program,
		maxLength:             maxLength,
		maxLengthSource:       maxLengthSource,
		pooling:               pooling,
		normalization:         normalization,
		modelName:             modelName,
		architecture:          architecture,
		dynamicTokens:         pretrainedBERTProgramSupportsDynamicTokens(program, "bert_embed"),
		packagePath:           cfg.PackagePath,
		packageSHA256:         packageSHA,
		packageIdentity:       pkg.IdentitySHA256,
		packageFileSHA:        fileHashes,
		retrievalRoleContract: clonePretrainedBERTRetrievalRoleContract(pkg.RetrievalRoleContract),
	}, nil
}

func (e *PretrainedBERTTextEmbedder) MaxLength() int {
	if e == nil {
		return 0
	}
	return e.maxLength
}

func (e *PretrainedBERTTextEmbedder) MaxLengthSource() string {
	if e == nil {
		return ""
	}
	return e.maxLengthSource
}

func (e *PretrainedBERTTextEmbedder) Pooling() string {
	if e == nil {
		return ""
	}
	return e.pooling
}

func (e *PretrainedBERTTextEmbedder) Normalization() string {
	if e == nil {
		return ""
	}
	return e.normalization
}

func (e *PretrainedBERTTextEmbedder) ModelName() string {
	if e == nil {
		return ""
	}
	return e.modelName
}

func (e *PretrainedBERTTextEmbedder) PackagePath() string {
	if e == nil {
		return ""
	}
	return e.packagePath
}

func (e *PretrainedBERTTextEmbedder) PackageSHA256() string {
	if e == nil {
		return ""
	}
	return e.packageSHA256
}

func (e *PretrainedBERTTextEmbedder) PackageIdentitySHA256() string {
	if e == nil {
		return ""
	}
	return e.packageIdentity
}

func (e *PretrainedBERTTextEmbedder) EncoderExecutionProvenance() PretrainedBERTEncoderExecutionProvenance {
	policy := pretrainedBERTDenseAccelerationPolicy()
	provenance := PretrainedBERTEncoderExecutionProvenance{
		ExecutionMode:                      "pretrained_bert_host_reference",
		FullDeviceExecution:                false,
		ValidatedDeviceEncoder:             false,
		HostReference:                      true,
		OpportunisticDeviceOpsIgnored:      true,
		DenseAccelerationPolicy:            policy,
		DeviceEncoderContract:              "full_device_pretrained_bert_encoder",
		DeviceEncoderContractSatisfied:     false,
		DeviceEncoderContractMissingReason: "no validated full-device StepBERTEmbedder/pretrained BERT encoder contract is present; imported BERT embedding uses the host-reference full stack",
		DeviceEncoderValidationFailureHint: "do not treat requested backends or observed opportunistic CUDA matmul/dense helper use as proof of full device encoder execution",
	}
	if e != nil && e.program != nil {
		provenance.SelectedBackend = string(e.program.Backend())
	}
	if e != nil {
		e.executionMu.Lock()
		stats := e.executionStats
		e.executionMu.Unlock()
		provenance.CUDADenseMatMulRuns = stats.cudaDenseMatMulRuns
		provenance.CUDADenseMatMulUploadedBytes = stats.cudaDenseMatMulUploadedBytes
		provenance.CUDADenseAccelerationObserved = stats.cudaDenseMatMulRuns > 0 || stats.cudaDenseMatMulUploadedBytes > 0
		provenance.OpportunisticDeviceOpsUsed = provenance.CUDADenseAccelerationObserved
		provenance.HostReferenceExecutionStatus = stats.hostReferenceExecutionStatus
	}
	return provenance
}

func pretrainedBERTDenseAccelerationPolicy() string {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("EOS_BERT_DENSE_ACCEL")))
	switch value {
	case "0", "false", "off", "disabled":
		return "disabled"
	default:
		return "enabled"
	}
}

func (e *PretrainedBERTTextEmbedder) NativeDim() int {
	if e == nil {
		return 0
	}
	return e.config.HiddenSize
}

func (e *PretrainedBERTTextEmbedder) RetrievalRoleContract() *PretrainedBERTRetrievalRoleContract {
	if e == nil {
		return nil
	}
	return clonePretrainedBERTRetrievalRoleContract(e.retrievalRoleContract)
}

func (e *PretrainedBERTTextEmbedder) PrefixForRole(role string) (string, error) {
	normalized := strings.TrimSpace(strings.ToLower(role))
	if normalized == "" {
		normalized = EmbeddingRoleRaw
	}
	switch normalized {
	case EmbeddingRoleRaw:
		return "", nil
	case EmbeddingRoleQuery, EmbeddingRoleDocument:
	default:
		return "", fmt.Errorf("unsupported pretrained BERT embedding role %q; want raw, query, or document", role)
	}
	if e == nil {
		return "", fmt.Errorf("pretrained BERT text embedder is not loaded")
	}
	if e.retrievalRoleContract == nil {
		if e.packagePath != "" {
			return "", fmt.Errorf("package %s does not declare retrieval_role_contract; role %q requires a package role contract or explicit lower-level prefix", e.packagePath, normalized)
		}
		return "", fmt.Errorf("role %q requires a retrieval role contract or explicit lower-level prefix", normalized)
	}
	switch normalized {
	case EmbeddingRoleQuery:
		return e.retrievalRoleContract.QueryPrefix, nil
	case EmbeddingRoleDocument:
		return e.retrievalRoleContract.DocumentPrefix, nil
	default:
		return "", nil
	}
}

func (e *PretrainedBERTTextEmbedder) EmbedTextBatchWithRole(ctx context.Context, texts []string, role string) ([][]float32, string, error) {
	prefix, err := e.PrefixForRole(role)
	if err != nil {
		return nil, "", err
	}
	embeddings, err := e.EmbedTextBatch(ctx, texts, prefix)
	if err != nil {
		return nil, "", err
	}
	return embeddings, prefix, nil
}

func resolvePretrainedBERTMaxLength(explicit int, config PretrainedBERTConfig, stMeta *PretrainedBERTSTMetadata) (int, string) {
	if explicit > 0 {
		return explicit, "explicit"
	}
	if stMeta != nil && stMeta.MaxSeqLength > 0 {
		return stMeta.MaxSeqLength, stMeta.MaxLengthSource
	}
	return config.MaxPositionEmbeddings, "config.max_position_embeddings"
}

func (e *PretrainedBERTTextEmbedder) EmbedTextBatch(ctx context.Context, texts []string, prefix string) ([][]float32, error) {
	if e == nil || e.program == nil || e.tokenizer == nil {
		return nil, fmt.Errorf("pretrained BERT text embedder is not loaded")
	}
	if len(texts) == 0 {
		return nil, nil
	}
	encodedRows := make([]HFWordPieceEncoding, len(texts))
	effectiveLength := 1
	for i, text := range texts {
		encoded, err := e.tokenizer.Encode(prefix+text, HFWordPieceEncodeOptions{
			MaxLength: e.maxLength,
		})
		if err != nil {
			return nil, err
		}
		encodedRows[i] = encoded
		if len(encoded.IDs) > effectiveLength {
			effectiveLength = len(encoded.IDs)
		}
	}
	if !e.dynamicTokens {
		effectiveLength = e.maxLength
	}
	if effectiveLength > e.maxLength {
		effectiveLength = e.maxLength
	}
	inputIDs, attentionMask, tokenTypeIDs, err := padPretrainedBERTBatchEncodings(e.tokenizer, encodedRows, effectiveLength)
	if err != nil {
		return nil, err
	}
	result, err := e.program.Run(ctx, backend.Request{
		Entry: "bert_embed",
		Inputs: map[string]any{
			"input_ids":      backend.NewTensorI32([]int{len(texts), effectiveLength}, inputIDs),
			"attention_mask": backend.NewTensorI32([]int{len(texts), effectiveLength}, attentionMask),
			"token_type_ids": backend.NewTensorI32([]int{len(texts), effectiveLength}, tokenTypeIDs),
		},
	})
	if err != nil {
		return nil, err
	}
	e.recordEncoderExecutionResult(result)
	value, ok := result.Outputs["embeddings"]
	if !ok {
		return nil, fmt.Errorf("bert_embed output missing embeddings")
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok {
		return nil, fmt.Errorf("bert_embed embeddings output has data type %T, want *backend.Tensor", value.Data)
	}
	if len(tensor.Shape) != 2 || tensor.Shape[0] != len(texts) {
		return nil, fmt.Errorf("bert_embed embeddings shape = %v, want [%d,D]", tensor.Shape, len(texts))
	}
	dim := tensor.Shape[1]
	if dim <= 0 || len(tensor.F32) != len(texts)*dim {
		return nil, fmt.Errorf("bert_embed embeddings backing data length %d does not match shape %v", len(tensor.F32), tensor.Shape)
	}
	out := make([][]float32, len(texts))
	for row := range texts {
		start := row * dim
		vec := append([]float32(nil), tensor.F32[start:start+dim]...)
		for i, value := range vec {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("embedding row %d value %d is not finite: %v", row, i, value)
			}
		}
		out[row] = vec
	}
	return out, nil
}

func (e *PretrainedBERTTextEmbedder) recordEncoderExecutionResult(result backend.Result) {
	if e == nil {
		return
	}
	stats := pretrainedBERTEncoderExecutionStats{
		cudaDenseMatMulRuns:          parsePretrainedBERTMetadataInt64(result.Metadata["cuda_matmul_bound_right_runs"]),
		cudaDenseMatMulUploadedBytes: parsePretrainedBERTMetadataInt64(result.Metadata["cuda_matmul_run_uploaded_bytes"]),
	}
	if value, ok := result.Outputs["embeddings"]; ok && value.Metadata != nil {
		if status, ok := value.Metadata["execution_status"].(string); ok {
			stats.hostReferenceExecutionStatus = status
		}
	}
	e.executionMu.Lock()
	e.executionStats.cudaDenseMatMulRuns += stats.cudaDenseMatMulRuns
	e.executionStats.cudaDenseMatMulUploadedBytes += stats.cudaDenseMatMulUploadedBytes
	if stats.hostReferenceExecutionStatus != "" {
		e.executionStats.hostReferenceExecutionStatus = stats.hostReferenceExecutionStatus
	}
	e.executionMu.Unlock()
}

func parsePretrainedBERTMetadataInt64(value string) int64 {
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func padPretrainedBERTBatchEncodings(tokenizer *HFWordPieceTokenizer, rows []HFWordPieceEncoding, length int) ([]int32, []int32, []int32, error) {
	if tokenizer == nil {
		return nil, nil, nil, fmt.Errorf("nil wordpiece tokenizer")
	}
	if length <= 0 {
		return nil, nil, nil, fmt.Errorf("pretrained BERT batch padded length must be positive, got %d", length)
	}
	padID, ok := tokenizer.TokenID(tokenizer.Config().PadToken)
	if !ok {
		return nil, nil, nil, fmt.Errorf("wordpiece vocab missing pad token %q", tokenizer.Config().PadToken)
	}
	inputIDs := make([]int32, 0, len(rows)*length)
	attentionMask := make([]int32, 0, len(rows)*length)
	tokenTypeIDs := make([]int32, 0, len(rows)*length)
	for row, encoded := range rows {
		if len(encoded.IDs) > length {
			return nil, nil, nil, fmt.Errorf("encoded row %d length %d exceeds padded length %d", row, len(encoded.IDs), length)
		}
		if len(encoded.AttentionMask) != len(encoded.IDs) || len(encoded.TokenTypeIDs) != len(encoded.IDs) {
			return nil, nil, nil, fmt.Errorf("encoded row %d has inconsistent ids/masks/token types lengths: %d/%d/%d", row, len(encoded.IDs), len(encoded.AttentionMask), len(encoded.TokenTypeIDs))
		}
		inputIDs = append(inputIDs, encoded.IDs...)
		attentionMask = append(attentionMask, encoded.AttentionMask...)
		tokenTypeIDs = append(tokenTypeIDs, encoded.TokenTypeIDs...)
		for pad := len(encoded.IDs); pad < length; pad++ {
			inputIDs = append(inputIDs, padID)
			attentionMask = append(attentionMask, 0)
			tokenTypeIDs = append(tokenTypeIDs, 0)
		}
	}
	return inputIDs, attentionMask, tokenTypeIDs, nil
}

func pretrainedBERTProgramSupportsDynamicTokens(program *Program, entryName string) bool {
	if program == nil || program.module == nil {
		return false
	}
	for _, entry := range program.module.EntryPoints {
		if entry.Name != entryName {
			continue
		}
		for _, input := range entry.Inputs {
			if input.Name != "input_ids" || input.Type.Tensor == nil {
				continue
			}
			shape := input.Type.Tensor.Shape
			return len(shape) == 2 && shape[1] == "T"
		}
	}
	return false
}

func ExportPretrainedBERTRetrievalVectors(ctx context.Context, cfg PretrainedBERTRetrievalVectorExportConfig) (PretrainedBERTRetrievalVectorExportSummary, error) {
	cfg = normalizePretrainedBERTRetrievalVectorExportConfig(cfg)
	if cfg.OutputDir == "" {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("output dir is required")
	}
	if cfg.PackagePath == "" && (cfg.SourceDir == "" || cfg.ModulePath == "" || cfg.WeightsPath == "") {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("source dir, module path, and weights path are required unless package path is provided")
	}
	if cfg.PackagePath != "" && (cfg.SourceDir != "" || cfg.ModulePath != "" || cfg.WeightsPath != "") {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("package path cannot be combined with source dir, module path, or weights path")
	}
	if cfg.CorpusPath == "" || cfg.QueriesPath == "" {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("corpus path and queries path are required")
	}
	if cfg.BatchSize <= 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("batch-size must be positive")
	}
	if cfg.OutputDim < 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("output-dim must be non-negative")
	}
	if cfg.MaxDocs < 0 || cfg.MaxQueries < 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("max-docs and max-queries must be non-negative")
	}
	if cfg.ProgressEvery < 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("progress-every must be non-negative")
	}
	start := time.Now()
	embedder, err := LoadPretrainedBERTTextEmbedder(ctx, PretrainedBERTTextEmbedderConfig{
		SourceDir:   cfg.SourceDir,
		ModulePath:  cfg.ModulePath,
		WeightsPath: cfg.WeightsPath,
		PackagePath: cfg.PackagePath,
		MaxLength:   cfg.MaxLength,
		Runtime:     cfg.Runtime,
	})
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	encoderExecution := embedder.EncoderExecutionProvenance()
	if cfg.RequireDeviceEncoder {
		if err := validatePretrainedBERTDeviceEncoderRequired(encoderExecution); err != nil {
			return PretrainedBERTRetrievalVectorExportSummary{}, err
		}
	}
	queryPrefix, documentPrefix, err := resolvePretrainedBERTRetrievalPrefixes(cfg, embedder)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	cfg.QueryPrefix = queryPrefix
	cfg.DocumentPrefix = documentPrefix
	nativeDim := embedder.config.HiddenSize
	var projectionHead *PretrainedBERTProjectionHead
	if cfg.ProjectionHeadPath != "" {
		head, err := ReadPretrainedBERTProjectionHeadFile(cfg.ProjectionHeadPath)
		if err != nil {
			return PretrainedBERTRetrievalVectorExportSummary{}, err
		}
		if head.InputDim != nativeDim {
			return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("projection head input_dim %d does not match pretrained BERT native dimension %d", head.InputDim, nativeDim)
		}
		if cfg.OutputDim > 0 && cfg.OutputDim != head.OutputDim {
			return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("output-dim %d does not match projection head output_dim %d", cfg.OutputDim, head.OutputDim)
		}
		cfg.OutputDim = head.OutputDim
		projectionHead = &head
	}
	effectiveOutputDim := cfg.OutputDim
	if effectiveOutputDim == 0 {
		effectiveOutputDim = nativeDim
	}
	if effectiveOutputDim > nativeDim {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("output-dim %d exceeds pretrained BERT native dimension %d", effectiveOutputDim, nativeDim)
	}
	var qrels retrievalQrels
	if cfg.QrelsPath != "" {
		qrels, err = readBEIRQrels(cfg.QrelsPath)
		if err != nil {
			return PretrainedBERTRetrievalVectorExportSummary{}, err
		}
	}
	corpus, err := readRetrievalExportCorpus(cfg.CorpusPath, cfg.MaxDocs, qrels)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	queries, _, err := readRetrievalExportQueries(cfg.QueriesPath, cfg.MaxQueries, qrels)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	if len(corpus) == 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("corpus is empty")
	}
	if len(queries) == 0 {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("queries are empty")
	}
	identity, err := buildPretrainedBERTRetrievalEmbeddingSpaceIdentity(cfg, embedder)
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	if err := validatePretrainedBERTRetrievalVectorExportResumeManifest(cfg, identity.EmbeddingSpaceID); err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	docVectorPath := filepath.Join(cfg.OutputDir, "doc-vectors.jsonl")
	queryVectorPath := filepath.Join(cfg.OutputDir, "query-vectors.jsonl")
	docDim, docResume, err := writePretrainedBERTVectorCache(ctx, embedder, corpus, docVectorPath, cfg.BatchSize, cfg.DocumentPrefix, pretrainedBERTVectorCacheWriteOptions{
		Kind:           "documents",
		Resume:         cfg.Resume,
		ProgressEvery:  cfg.ProgressEvery,
		Progress:       cfg.Progress,
		OutputDim:      cfg.OutputDim,
		ExpectedDim:    effectiveOutputDim,
		NativeDim:      nativeDim,
		ProjectionHead: projectionHead,
	})
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("write document vectors: %w", err)
	}
	queryDim, queryResume, err := writePretrainedBERTVectorCache(ctx, embedder, queries, queryVectorPath, cfg.BatchSize, cfg.QueryPrefix, pretrainedBERTVectorCacheWriteOptions{
		Kind:           "queries",
		Resume:         cfg.Resume,
		ProgressEvery:  cfg.ProgressEvery,
		Progress:       cfg.Progress,
		OutputDim:      cfg.OutputDim,
		ExpectedDim:    effectiveOutputDim,
		NativeDim:      nativeDim,
		ProjectionHead: projectionHead,
	})
	if err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("write query vectors: %w", err)
	}
	if docDim != queryDim {
		return PretrainedBERTRetrievalVectorExportSummary{}, fmt.Errorf("document vectors have dimension %d but query vectors have dimension %d", docDim, queryDim)
	}
	encoderExecution = embedder.EncoderExecutionProvenance()
	summary := PretrainedBERTRetrievalVectorExportSummary{
		Schema:                            PretrainedBERTRetrievalVectorExportManifestSchema,
		Dataset:                           cfg.DatasetName,
		SourceDir:                         cfg.SourceDir,
		ModulePath:                        cfg.ModulePath,
		WeightsPath:                       cfg.WeightsPath,
		PackagePath:                       cfg.PackagePath,
		PackageSHA256:                     identity.PackageSHA256,
		PackageIdentitySHA256:             identity.PackageIdentitySHA256,
		ExecutionMode:                     encoderExecution.ExecutionMode,
		EncoderExecution:                  encoderExecution,
		QualityClaim:                      false,
		EmbeddingSpaceID:                  identity.EmbeddingSpaceID,
		ProjectionHeadPath:                cfg.ProjectionHeadPath,
		ProjectionHeadSHA256:              identity.ProjectionHeadSHA256,
		ProjectionHeadIdentity:            identity.ProjectionHeadIdentity,
		ProjectionHeadSchema:              identity.ProjectionHeadSchema,
		ProjectionHeadSourceModel:         identity.ProjectionHeadSourceModel,
		ModuleSHA256:                      identity.ModuleSHA256,
		WeightsSHA256:                     identity.WeightsSHA256,
		ConfigSHA256:                      identity.ConfigSHA256,
		VocabSHA256:                       identity.VocabSHA256,
		TokenizerJSONSHA256:               identity.TokenizerJSONSHA256,
		TokenizerConfigSHA256:             identity.TokenizerConfigSHA256,
		SpecialTokensMapSHA256:            identity.SpecialTokensMapSHA256,
		SentenceTransformersPoolingSHA256: identity.SentenceTransformersPoolingSHA256,
		SentenceTransformersConfigSHA256:  identity.SentenceTransformersConfigSHA256,
		Normalization:                     embedder.Normalization(),
		Documents:                         len(corpus),
		Queries:                           len(queries),
		NativeDim:                         nativeDim,
		OutputDim:                         docDim,
		DocVectorPath:                     docVectorPath,
		QueryVectorPath:                   queryVectorPath,
		QueryPrefix:                       cfg.QueryPrefix,
		DocumentPrefix:                    cfg.DocumentPrefix,
		LegacyDocPrefix:                   cfg.DocumentPrefix,
		DocumentRoleApplied:               cfg.DocumentPrefix != "",
		QueryRoleApplied:                  cfg.QueryPrefix != "",
		Resume:                            cfg.Resume,
		ProgressEvery:                     cfg.ProgressEvery,
		ReusedDocuments:                   docResume.Reused,
		ReusedQueries:                     queryResume.Reused,
		WrittenDocuments:                  docResume.Written,
		WrittenQueries:                    queryResume.Written,
		DocumentBatching:                  docResume.Batching,
		QueryBatching:                     queryResume.Batching,
		MaxLength:                         embedder.MaxLength(),
		MaxLengthSource:                   embedder.MaxLengthSource(),
		Pooling:                           embedder.Pooling(),
		BatchSize:                         cfg.BatchSize,
		MaxDocs:                           cfg.MaxDocs,
		MaxQueries:                        cfg.MaxQueries,
		CorpusPath:                        cfg.CorpusPath,
		QueriesPath:                       cfg.QueriesPath,
		QrelsPath:                         cfg.QrelsPath,
		ElapsedSeconds:                    time.Since(start).Seconds(),
		CreatedAt:                         time.Now().UTC(),
	}
	if err := WritePretrainedBERTRetrievalVectorExportSummaryFile(cfg.ManifestJSONPath, summary); err != nil {
		return PretrainedBERTRetrievalVectorExportSummary{}, err
	}
	return summary, nil
}

func validatePretrainedBERTDeviceEncoderRequired(provenance PretrainedBERTEncoderExecutionProvenance) error {
	if provenance.FullDeviceExecution && provenance.ValidatedDeviceEncoder && provenance.DeviceEncoderContractSatisfied {
		return nil
	}
	return fmt.Errorf("--require-device-encoder requested, but pretrained BERT encoder execution is not fully device-backed: execution_mode=%s selected_backend=%s full_device_execution=%t validated_device_encoder=%t contract=%s satisfied=%t; %s; %s",
		provenance.ExecutionMode,
		provenance.SelectedBackend,
		provenance.FullDeviceExecution,
		provenance.ValidatedDeviceEncoder,
		provenance.DeviceEncoderContract,
		provenance.DeviceEncoderContractSatisfied,
		provenance.DeviceEncoderContractMissingReason,
		provenance.DeviceEncoderValidationFailureHint,
	)
}

func resolvePretrainedBERTRetrievalPrefixes(cfg PretrainedBERTRetrievalVectorExportConfig, embedder *PretrainedBERTTextEmbedder) (string, string, error) {
	if !cfg.UsePackageRoleContract {
		return cfg.QueryPrefix, cfg.DocumentPrefix, nil
	}
	if embedder == nil || embedder.packagePath == "" {
		return "", "", fmt.Errorf("use-package-role-contract requires package mode")
	}
	contract := embedder.retrievalRoleContract
	if contract == nil {
		return "", "", fmt.Errorf("package %s does not declare retrieval_role_contract", embedder.packagePath)
	}
	if cfg.QueryPrefixSet && cfg.QueryPrefix != contract.QueryPrefix {
		return "", "", fmt.Errorf("query prefix %q conflicts with package retrieval role contract %q", cfg.QueryPrefix, contract.QueryPrefix)
	}
	if cfg.DocumentPrefixSet && cfg.DocumentPrefix != contract.DocumentPrefix {
		return "", "", fmt.Errorf("document prefix %q conflicts with package retrieval role contract %q", cfg.DocumentPrefix, contract.DocumentPrefix)
	}
	return contract.QueryPrefix, contract.DocumentPrefix, nil
}

func WritePretrainedBERTRetrievalVectorExportSummaryFile(path string, summary PretrainedBERTRetrievalVectorExportSummary) error {
	if path == "" {
		path = filepath.Join(filepath.Dir(summary.DocVectorPath), "manifest.json")
	}
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

type pretrainedBERTRetrievalEmbeddingSpaceIdentity struct {
	EmbeddingSpaceID                  string `json:"-"`
	Schema                            string `json:"schema"`
	ExecutionMode                     string `json:"execution_mode"`
	ModelName                         string `json:"model_name,omitempty"`
	Architecture                      string `json:"architecture,omitempty"`
	ModelType                         string `json:"model_type,omitempty"`
	ModuleSHA256                      string `json:"module_sha256"`
	WeightsSHA256                     string `json:"weights_sha256"`
	PackageSHA256                     string `json:"package_sha256,omitempty"`
	PackageIdentitySHA256             string `json:"package_identity_sha256,omitempty"`
	ConfigSHA256                      string `json:"config_sha256"`
	VocabSHA256                       string `json:"vocab_sha256"`
	TokenizerJSONSHA256               string `json:"tokenizer_json_sha256,omitempty"`
	TokenizerConfigSHA256             string `json:"tokenizer_config_sha256,omitempty"`
	SpecialTokensMapSHA256            string `json:"special_tokens_map_sha256,omitempty"`
	SentenceTransformersPoolingSHA256 string `json:"sentence_transformers_pooling_sha256,omitempty"`
	SentenceTransformersConfigSHA256  string `json:"sentence_transformers_config_sha256,omitempty"`
	Pooling                           string `json:"pooling,omitempty"`
	Normalization                     string `json:"normalization,omitempty"`
	MaxLength                         int    `json:"max_length"`
	OutputDim                         int    `json:"output_dim"`
	QueryPrefix                       string `json:"query_prefix"`
	DocumentPrefix                    string `json:"document_prefix"`
	QueryRoleApplied                  bool   `json:"query_role_applied"`
	DocumentRoleApplied               bool   `json:"document_role_applied"`
	ProjectionHeadPath                string `json:"projection_head_path,omitempty"`
	ProjectionHeadSHA256              string `json:"projection_head_sha256,omitempty"`
	ProjectionHeadIdentity            string `json:"projection_head_identity,omitempty"`
	ProjectionHeadSchema              string `json:"projection_head_schema,omitempty"`
	ProjectionHeadSourceModel         string `json:"projection_head_source_model,omitempty"`
}

func buildPretrainedBERTRetrievalEmbeddingSpaceIdentity(cfg PretrainedBERTRetrievalVectorExportConfig, embedder *PretrainedBERTTextEmbedder) (pretrainedBERTRetrievalEmbeddingSpaceIdentity, error) {
	identity := pretrainedBERTRetrievalEmbeddingSpaceIdentity{
		Schema:              PretrainedBERTRetrievalVectorExportManifestSchema,
		ExecutionMode:       "pretrained_bert_host_reference",
		ModelName:           embedder.modelName,
		Architecture:        embedder.architecture,
		ModelType:           embedder.config.ModelType,
		Pooling:             embedder.Pooling(),
		Normalization:       embedder.Normalization(),
		MaxLength:           embedder.MaxLength(),
		OutputDim:           resolvePretrainedBERTOutputDim(cfg.OutputDim, embedder.config.HiddenSize),
		QueryPrefix:         cfg.QueryPrefix,
		DocumentPrefix:      cfg.DocumentPrefix,
		QueryRoleApplied:    cfg.QueryPrefix != "",
		DocumentRoleApplied: cfg.DocumentPrefix != "",
		ProjectionHeadPath:  cfg.ProjectionHeadPath,
	}
	var err error
	if embedder.packagePath != "" {
		identity.PackageSHA256 = embedder.packageSHA256
		identity.PackageIdentitySHA256 = embedder.packageIdentity
		identity.ModuleSHA256 = embedder.packageFileSHA["module"]
		identity.WeightsSHA256 = embedder.packageFileSHA["weights"]
		identity.ConfigSHA256 = embedder.packageFileSHA["config"]
		identity.VocabSHA256 = embedder.packageFileSHA["vocab"]
		identity.TokenizerJSONSHA256 = embedder.packageFileSHA["tokenizer_json"]
		identity.TokenizerConfigSHA256 = embedder.packageFileSHA["tokenizer_config"]
		identity.SpecialTokensMapSHA256 = embedder.packageFileSHA["special_tokens_map"]
		identity.SentenceTransformersPoolingSHA256 = embedder.packageFileSHA["sentence_transformers_pooling"]
		identity.SentenceTransformersConfigSHA256 = embedder.packageFileSHA["sentence_transformers_config"]
	} else {
		if identity.ModuleSHA256, err = sha256FileHex(cfg.ModulePath); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash module: %w", err)
		}
		if identity.WeightsSHA256, err = sha256FileHex(cfg.WeightsPath); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash weights: %w", err)
		}
		if identity.ConfigSHA256, err = sha256FileHex(filepath.Join(cfg.SourceDir, "config.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash config.json: %w", err)
		}
		if identity.VocabSHA256, err = sha256FileHex(filepath.Join(cfg.SourceDir, "vocab.txt")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash vocab.txt: %w", err)
		}
		if identity.TokenizerJSONSHA256, err = optionalSHA256FileHex(filepath.Join(cfg.SourceDir, "tokenizer.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
		if identity.TokenizerConfigSHA256, err = optionalSHA256FileHex(filepath.Join(cfg.SourceDir, "tokenizer_config.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
		if identity.SpecialTokensMapSHA256, err = optionalSHA256FileHex(filepath.Join(cfg.SourceDir, "special_tokens_map.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
		if identity.SentenceTransformersPoolingSHA256, err = optionalSHA256FileHex(filepath.Join(cfg.SourceDir, "1_Pooling", "config.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
		if identity.SentenceTransformersConfigSHA256, err = optionalSHA256FileHex(filepath.Join(cfg.SourceDir, "sentence_bert_config.json")); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
	}
	if cfg.ProjectionHeadPath != "" {
		head, err := ReadPretrainedBERTProjectionHeadFile(cfg.ProjectionHeadPath)
		if err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
		}
		if head.InputDim != embedder.config.HiddenSize {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("projection head input_dim %d does not match pretrained BERT native dimension %d", head.InputDim, embedder.config.HiddenSize)
		}
		identity.ProjectionHeadSchema = head.Schema
		identity.ProjectionHeadSourceModel = head.SourceModel
		identity.ProjectionHeadIdentity = head.IdentityHash()
		identity.OutputDim = head.OutputDim
		if identity.ProjectionHeadSHA256, err = sha256FileHex(cfg.ProjectionHeadPath); err != nil {
			return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, fmt.Errorf("hash projection head: %w", err)
		}
	}
	canonical := identity
	canonical.ProjectionHeadPath = ""
	canonical.ProjectionHeadSHA256 = ""
	canonical.ProjectionHeadSourceModel = ""
	data, err := json.Marshal(canonical)
	if err != nil {
		return pretrainedBERTRetrievalEmbeddingSpaceIdentity{}, err
	}
	sum := sha256.Sum256(data)
	identity.EmbeddingSpaceID = hex.EncodeToString(sum[:])
	return identity, nil
}

func resolvePretrainedBERTOutputDim(outputDim, nativeDim int) int {
	if outputDim > 0 {
		return outputDim
	}
	return nativeDim
}

func validatePretrainedBERTRetrievalVectorExportResumeManifest(cfg PretrainedBERTRetrievalVectorExportConfig, embeddingSpaceID string) error {
	if !cfg.Resume {
		return nil
	}
	data, err := os.ReadFile(cfg.ManifestJSONPath)
	if os.IsNotExist(err) {
		if cfg.PackagePath != "" {
			return fmt.Errorf("resume with package requires manifest %s with matching embedding_space_id", cfg.ManifestJSONPath)
		}
		if cfg.ProjectionHeadPath != "" {
			return fmt.Errorf("resume with projection head requires manifest %s with matching embedding_space_id", cfg.ManifestJSONPath)
		}
		return nil
	}
	if err != nil {
		return err
	}
	var manifest PretrainedBERTRetrievalVectorExportSummary
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("parse resume manifest %s: %w", cfg.ManifestJSONPath, err)
	}
	// Legacy manifests predate embedding_space_id. Keep compatibility for
	// no-head partial caches and rely on row ID/dimension prefix checks.
	if manifest.EmbeddingSpaceID == "" {
		if cfg.PackagePath != "" {
			return fmt.Errorf("resume manifest %s is missing embedding_space_id required for package resume", cfg.ManifestJSONPath)
		}
		if cfg.ProjectionHeadPath != "" {
			return fmt.Errorf("resume manifest %s is missing embedding_space_id required for projection head resume", cfg.ManifestJSONPath)
		}
		return nil
	}
	if cfg.PackagePath != "" && manifest.PackageIdentitySHA256 == "" {
		return fmt.Errorf("resume manifest %s is missing package_identity_sha256 required for package resume", cfg.ManifestJSONPath)
	}
	if cfg.ProjectionHeadPath != "" && manifest.ProjectionHeadIdentity == "" {
		return fmt.Errorf("resume manifest %s is missing projection_head_identity required for projection head resume", cfg.ManifestJSONPath)
	}
	if manifest.EmbeddingSpaceID != embeddingSpaceID {
		return fmt.Errorf("resume manifest %s embedding_space_id %q does not match current embedding space %q", cfg.ManifestJSONPath, manifest.EmbeddingSpaceID, embeddingSpaceID)
	}
	return nil
}

func sha256FileHex(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func optionalSHA256FileHex(path string) (string, error) {
	value, err := sha256FileHex(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("hash optional file %s: %w", path, err)
	}
	return value, nil
}

func normalizePretrainedBERTRetrievalVectorExportConfig(cfg PretrainedBERTRetrievalVectorExportConfig) PretrainedBERTRetrievalVectorExportConfig {
	if cfg.DatasetName == "" {
		if cfg.DatasetDir != "" {
			cfg.DatasetName = filepath.Base(cfg.DatasetDir)
		} else {
			cfg.DatasetName = "retrieval"
		}
	}
	if cfg.Split == "" {
		cfg.Split = "test"
	}
	if cfg.DatasetDir != "" {
		corpus, queries, qrels := BEIRRetrievalPaths(cfg.DatasetDir, cfg.Split)
		if cfg.CorpusPath == "" {
			cfg.CorpusPath = corpus
		}
		if cfg.QueriesPath == "" {
			cfg.QueriesPath = queries
		}
		if cfg.QrelsPath == "" {
			cfg.QrelsPath = qrels
		}
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 64
	}
	if cfg.ManifestJSONPath == "" && cfg.OutputDir != "" {
		cfg.ManifestJSONPath = filepath.Join(cfg.OutputDir, "manifest.json")
	}
	return cfg
}

type pretrainedBERTVectorCacheWriteOptions struct {
	Kind           string
	Resume         bool
	ProgressEvery  int
	Progress       PretrainedBERTRetrievalVectorExportProgressFunc
	OutputDim      int
	ExpectedDim    int
	NativeDim      int
	ProjectionHead *PretrainedBERTProjectionHead
}

type pretrainedBERTVectorCacheResumeResult struct {
	Reused   int
	Written  int
	Batching PretrainedBERTVectorBatchingSummary
}

func writePretrainedBERTVectorCache(ctx context.Context, embedder *PretrainedBERTTextEmbedder, records []retrievalTextRecord, path string, batchSize int, prefix string, opts pretrainedBERTVectorCacheWriteOptions) (int, pretrainedBERTVectorCacheResumeResult, error) {
	reused, dim, err := validatePretrainedBERTVectorCachePrefix(path, records, opts.Resume, opts.ExpectedDim)
	if err != nil {
		return 0, pretrainedBERTVectorCacheResumeResult{}, err
	}
	flags := os.O_CREATE | os.O_WRONLY
	if opts.Resume {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		return 0, pretrainedBERTVectorCacheResumeResult{}, err
	}
	defer file.Close()
	writer := bufio.NewWriter(file)
	result := pretrainedBERTVectorCacheResumeResult{
		Reused: reused,
		Batching: PretrainedBERTVectorBatchingSummary{
			Strategy:    "stable_length_bucket_window_v1",
			BatchSize:   batchSize,
			TotalItems:  len(records),
			ReusedItems: reused,
		},
	}
	progressStart := time.Now()
	if opts.Resume && reused > 0 && reused < len(records) {
		endsNewline, err := fileEndsWithNewline(path)
		if err != nil {
			return 0, pretrainedBERTVectorCacheResumeResult{}, err
		}
		if !endsNewline {
			if err := writer.WriteByte('\n'); err != nil {
				return 0, pretrainedBERTVectorCacheResumeResult{}, err
			}
		}
	}
	windowSize := pretrainedBERTLengthBucketWindowSize(batchSize)
	for windowStart := reused; windowStart < len(records); windowStart += windowSize {
		windowEnd := windowStart + windowSize
		if windowEnd > len(records) {
			windowEnd = len(records)
		}
		windowVectors, stats, err := embedPretrainedBERTVectorCacheWindow(ctx, embedder, records[windowStart:windowEnd], batchSize, prefix, opts)
		if err != nil {
			return 0, pretrainedBERTVectorCacheResumeResult{}, err
		}
		accumulatePretrainedBERTVectorBatchingSummary(&result.Batching, stats)
		for i, embedding := range windowVectors {
			record := records[windowStart+i]
			nextDim, err := validatePretrainedBERTVectorCacheRow(path, windowStart+i+1, retrievalVectorExportRow{ID: record.ID, Embedding: embedding}, record.ID, dim, opts.ExpectedDim)
			if err != nil {
				return 0, pretrainedBERTVectorCacheResumeResult{}, err
			}
			dim = nextDim
			row := retrievalVectorExportRow{ID: record.ID, Embedding: embedding}
			data, err := json.Marshal(row)
			if err != nil {
				return 0, pretrainedBERTVectorCacheResumeResult{}, err
			}
			if _, err := writer.Write(append(data, '\n')); err != nil {
				return 0, pretrainedBERTVectorCacheResumeResult{}, err
			}
			result.Written++
		}
		if err := writer.Flush(); err != nil {
			return 0, pretrainedBERTVectorCacheResumeResult{}, err
		}
		reportPretrainedBERTVectorExportProgress(opts, path, reused+result.Written, len(records), result.Reused, result.Written, progressStart)
	}
	if err := writer.Flush(); err != nil {
		return 0, pretrainedBERTVectorCacheResumeResult{}, err
	}
	if dim == 0 {
		return 0, pretrainedBERTVectorCacheResumeResult{}, fmt.Errorf("no vector dimension observed for %s", path)
	}
	return dim, result, nil
}

type pretrainedBERTVectorCacheBatchItem struct {
	Ordinal int
	Text    string
	Length  int
}

func pretrainedBERTLengthBucketWindowSize(batchSize int) int {
	if batchSize <= 0 {
		return 1
	}
	windowSize := batchSize * 16
	if windowSize < batchSize {
		return batchSize
	}
	return windowSize
}

func embedPretrainedBERTVectorCacheWindow(ctx context.Context, embedder *PretrainedBERTTextEmbedder, records []retrievalTextRecord, batchSize int, prefix string, opts pretrainedBERTVectorCacheWriteOptions) ([][]float32, PretrainedBERTVectorBatchingSummary, error) {
	items := make([]pretrainedBERTVectorCacheBatchItem, len(records))
	stats := PretrainedBERTVectorBatchingSummary{
		Strategy:      "stable_length_bucket_window_v1",
		BatchSize:     batchSize,
		Items:         len(records),
		TotalItems:    len(records),
		ComputedItems: len(records),
	}
	for i, record := range records {
		length, err := embedder.pretrainedBERTEncodedLength(prefix, record.Text)
		if err != nil {
			return nil, PretrainedBERTVectorBatchingSummary{}, fmt.Errorf("tokenize %q for length bucket: %w", record.ID, err)
		}
		items[i] = pretrainedBERTVectorCacheBatchItem{Ordinal: i, Text: record.Text, Length: length}
		stats.ActualTokens += int64(length)
		stats.FixedMaxTokens += int64(embedder.MaxLength())
	}
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].Length != items[j].Length {
			return items[i].Length < items[j].Length
		}
		return items[i].Ordinal < items[j].Ordinal
	})
	out := make([][]float32, len(records))
	for start := 0; start < len(items); start += batchSize {
		end := start + batchSize
		if end > len(items) {
			end = len(items)
		}
		batch := items[start:end]
		texts := make([]string, len(batch))
		effectiveLength := 1
		for i, item := range batch {
			texts[i] = item.Text
			if item.Length > effectiveLength {
				effectiveLength = item.Length
			}
		}
		stats.BatchCount++
		stats.PaddedTokens += int64(len(batch) * effectiveLength)
		if effectiveLength > stats.MaxEffectiveLength {
			stats.MaxEffectiveLength = effectiveLength
		}
		vectors, err := embedder.EmbedTextBatch(ctx, texts, prefix)
		if err != nil {
			return nil, PretrainedBERTVectorBatchingSummary{}, err
		}
		for i, vector := range vectors {
			item := batch[i]
			if len(vector) == 0 {
				return nil, PretrainedBERTVectorBatchingSummary{}, fmt.Errorf("vector for ordinal %d is empty", item.Ordinal)
			}
			if opts.NativeDim > 0 && len(vector) != opts.NativeDim {
				return nil, PretrainedBERTVectorBatchingSummary{}, fmt.Errorf("vector for ordinal %d has native dimension %d, want %d", item.Ordinal, len(vector), opts.NativeDim)
			}
			var embedding []float32
			if opts.ProjectionHead != nil {
				embedding, err = opts.ProjectionHead.Apply(vector)
				if err != nil {
					return nil, PretrainedBERTVectorBatchingSummary{}, fmt.Errorf("project vector for ordinal %d: %w", item.Ordinal, err)
				}
			} else {
				embedding, err = transformRetrievalExportVector(vector, opts.OutputDim)
				if err != nil {
					return nil, PretrainedBERTVectorBatchingSummary{}, fmt.Errorf("vector for ordinal %d: %w", item.Ordinal, err)
				}
			}
			out[item.Ordinal] = embedding
		}
	}
	return out, stats, nil
}

func (e *PretrainedBERTTextEmbedder) pretrainedBERTEncodedLength(prefix, text string) (int, error) {
	if e == nil || e.tokenizer == nil {
		return 0, fmt.Errorf("pretrained BERT text embedder is not loaded")
	}
	encoded, err := e.tokenizer.Encode(prefix+text, HFWordPieceEncodeOptions{
		MaxLength: e.maxLength,
	})
	if err != nil {
		return 0, err
	}
	if len(encoded.IDs) <= 0 {
		return 1, nil
	}
	return len(encoded.IDs), nil
}

func accumulatePretrainedBERTVectorBatchingSummary(dst *PretrainedBERTVectorBatchingSummary, src PretrainedBERTVectorBatchingSummary) {
	dst.Items += src.Items
	dst.ComputedItems += src.ComputedItems
	dst.BatchCount += src.BatchCount
	dst.ActualTokens += src.ActualTokens
	dst.PaddedTokens += src.PaddedTokens
	dst.FixedMaxTokens += src.FixedMaxTokens
	if src.MaxEffectiveLength > dst.MaxEffectiveLength {
		dst.MaxEffectiveLength = src.MaxEffectiveLength
	}
}

func validatePretrainedBERTVectorCachePrefix(path string, records []retrievalTextRecord, resume bool, expectedDim int) (int, int, error) {
	if !resume {
		return 0, 0, nil
	}
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0, 0, nil
	}
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	count, dim := 0, 0
	for scanner.Scan() {
		count++
		if count > len(records) {
			return 0, 0, fmt.Errorf("resume cache %s has too many rows: got at least %d, want %d", path, count, len(records))
		}
		var row retrievalVectorExportRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return 0, 0, fmt.Errorf("resume cache %s row %d is malformed: %w", path, count, err)
		}
		nextDim, err := validatePretrainedBERTVectorCacheRow(path, count, row, records[count-1].ID, dim, expectedDim)
		if err != nil {
			return 0, 0, err
		}
		dim = nextDim
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, fmt.Errorf("read resume cache %s: %w", path, err)
	}
	return count, dim, nil
}

func fileEndsWithNewline(path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return false, err
	}
	if info.Size() == 0 {
		return true, nil
	}
	buf := []byte{0}
	if _, err := file.ReadAt(buf, info.Size()-1); err != nil {
		return false, err
	}
	return buf[0] == '\n', nil
}

func validatePretrainedBERTVectorCacheRow(path string, rowNumber int, row retrievalVectorExportRow, wantID string, dim, expectedDim int) (int, error) {
	if row.ID != wantID {
		return 0, fmt.Errorf("resume cache %s row %d has id %q, want prefix id %q", path, rowNumber, row.ID, wantID)
	}
	if len(row.Embedding) == 0 {
		return 0, fmt.Errorf("resume cache %s row %d vector for %q is empty", path, rowNumber, wantID)
	}
	if dim == 0 {
		dim = len(row.Embedding)
	} else if len(row.Embedding) != dim {
		return 0, fmt.Errorf("resume cache %s row %d vector for %q has dimension %d, want %d", path, rowNumber, wantID, len(row.Embedding), dim)
	}
	if expectedDim > 0 && len(row.Embedding) != expectedDim {
		return 0, fmt.Errorf("resume cache %s row %d vector for %q has dimension %d, want current model output dimension %d", path, rowNumber, wantID, len(row.Embedding), expectedDim)
	}
	for i, value := range row.Embedding {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("resume cache %s row %d vector for %q value %d is not finite: %v", path, rowNumber, wantID, i, value)
		}
	}
	return dim, nil
}

func reportPretrainedBERTVectorExportProgress(opts pretrainedBERTVectorCacheWriteOptions, path string, completed, total, reused, written int, start time.Time) {
	if opts.Progress == nil || opts.ProgressEvery <= 0 || completed <= 0 {
		return
	}
	if completed%opts.ProgressEvery != 0 && completed != total {
		return
	}
	opts.Progress(PretrainedBERTRetrievalVectorExportProgress{
		Kind:           opts.Kind,
		Path:           path,
		Completed:      completed,
		Total:          total,
		Reused:         reused,
		Written:        written,
		ElapsedSeconds: time.Since(start).Seconds(),
	})
}
