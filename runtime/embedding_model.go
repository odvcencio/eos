package eosruntime

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
)

const EmbeddingManifestVersion = "manta/embedding-manifest/v0alpha1"

const (
	EmbeddingAttentionMaskModeNone            = "none"
	EmbeddingAttentionMaskModeKey             = "key"
	EmbeddingPositionEncodingNone             = "none"
	EmbeddingPositionEncodingRoPE             = "rope"
	EmbeddingAttentionScoreScaleNone          = "none"
	EmbeddingAttentionScoreScaleKeyDimRSQ     = "key_dim_rsqrt"
	EmbeddingArchitectureLegacyV1             = "legacy_tied_v1"
	EmbeddingArchitectureCompactTransformerV1 = "compact_transformer_v1"
	EmbeddingParameterTyingLegacyTied         = "legacy_tied"
	EmbeddingParameterTyingUntied             = "untied"
	EmbeddingRoleConditioningNone             = "none"
	EmbeddingRoleConditioningAdditiveV1       = "additive_token_embedding_v1"
	EmbeddingRoleRaw                          = "raw"
	EmbeddingRoleQuery                        = "query"
	EmbeddingRoleDocument                     = "document"
)

// TokenizerManifest carries embedding-model tokenization limits and ids.
type TokenizerManifest struct {
	VocabSize   int   `json:"vocab_size,omitempty"`
	MaxSequence int   `json:"max_sequence,omitempty"`
	PadID       int32 `json:"pad_id,omitempty"`
	BOSID       int32 `json:"bos_id,omitempty"`
	EOSID       int32 `json:"eos_id,omitempty"`
	UnknownID   int32 `json:"unknown_id,omitempty"`
}

// EmbeddingManifest describes the serving contract for an embedding module.
type EmbeddingManifest struct {
	Name                  string            `json:"name,omitempty"`
	PooledEntry           string            `json:"pooled_entry,omitempty"`
	BatchEntry            string            `json:"batch_entry,omitempty"`
	EncoderRepeats        int               `json:"encoder_repeats,omitempty"`
	TokenInput            string            `json:"token_input,omitempty"`
	MaskInput             string            `json:"mask_input,omitempty"`
	OutputName            string            `json:"output_name,omitempty"`
	OutputDType           string            `json:"output_dtype,omitempty"`
	ArchitectureVersion   string            `json:"architecture_version,omitempty"`
	ModelDim              int               `json:"model_dim,omitempty"`
	OutputDim             int               `json:"output_dim,omitempty"`
	AttentionHeads        int               `json:"attention_heads,omitempty"`
	HeadDim               int               `json:"head_dim,omitempty"`
	FFNDim                int               `json:"ffn_dim,omitempty"`
	ParameterTying        string            `json:"parameter_tying,omitempty"`
	TokenEmbeddingParam   string            `json:"token_embedding_param,omitempty"`
	RoleConditioning      string            `json:"role_conditioning,omitempty"`
	RoleEmbeddingParam    string            `json:"role_embedding_param,omitempty"`
	RoleInput             string            `json:"role_input,omitempty"`
	BatchRoleInput        string            `json:"batch_role_input,omitempty"`
	RawRoleIndex          int32             `json:"raw_role_index,omitempty"`
	QueryRoleIndex        int32             `json:"query_role_index,omitempty"`
	DocumentRoleIndex     int32             `json:"document_role_index,omitempty"`
	AttentionQueryParam   string            `json:"attention_query_param,omitempty"`
	AttentionKeyParam     string            `json:"attention_key_param,omitempty"`
	AttentionValueParam   string            `json:"attention_value_param,omitempty"`
	AttentionOutputParam  string            `json:"attention_output_param,omitempty"`
	AttentionMaskMode     string            `json:"attention_mask_mode,omitempty"`
	AttentionScoreScale   string            `json:"attention_score_scale,omitempty"`
	AttentionResidual     bool              `json:"attention_residual,omitempty"`
	AttentionLayerNorm    bool              `json:"attention_layernorm,omitempty"`
	PositionEncoding      string            `json:"position_encoding,omitempty"`
	HiddenProjectionParam string            `json:"hidden_projection_param,omitempty"`
	FFNResidual           bool              `json:"ffn_residual,omitempty"`
	FFNLayerNorm          bool              `json:"ffn_layernorm,omitempty"`
	ProjectionParam       string            `json:"projection_param,omitempty"`
	OutputProjectionParam string            `json:"output_projection_param,omitempty"`
	PostPoolTransform     string            `json:"post_pool_transform,omitempty"`
	Tokenizer             TokenizerManifest `json:"tokenizer,omitempty"`
}

// EmbeddingModel is a manifest-backed embedding serving handle.
type EmbeddingModel struct {
	program           *Program
	manifest          EmbeddingManifest
	tokenizerFile     *TokenizerFile
	tokenizer         *BPETokenizer
	postPoolTransform *AOQTGivensTransform
}

// ReadEmbeddingManifestFile decodes an authored MLL embedding manifest.
func ReadEmbeddingManifestFile(path string) (EmbeddingManifest, error) {
	doc, err := readAuthoredManifestMLL(path, "embedding_manifest", EmbeddingManifestVersion)
	if err != nil {
		return EmbeddingManifest{}, err
	}
	return embeddingManifestFromDoc(doc)
}

// DefaultEmbeddingManifestPath returns the conventional sibling manifest path for an .mll artifact.
func DefaultEmbeddingManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".embedding.mll")
}

// ResolveEmbeddingManifestPath returns the sibling embedding manifest path.
func ResolveEmbeddingManifestPath(artifactPath string) string {
	return DefaultEmbeddingManifestPath(artifactPath)
}

// WriteFile writes the embedding manifest as an authored MLL container.
func (m EmbeddingManifest) WriteFile(path string) error {
	return writeAuthoredManifestMLL(path, "embedding_manifest", EmbeddingManifestVersion, m.nameOrDefault(), "Eos embedding manifest", m.mllValues())
}

// LoadEmbedding loads an embedding module with a validated serving manifest.
func (rt *Runtime) LoadEmbedding(ctx context.Context, mod *eosartifact.Module, manifest EmbeddingManifest, opts ...LoadOption) (*EmbeddingModel, error) {
	manifest = manifest.normalizedForModule(mod)
	if err := manifest.validateServingGraphSupported(); err != nil {
		return nil, err
	}
	if err := manifest.ValidateModule(mod); err != nil {
		return nil, err
	}
	cfg := loadConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := validatePostPoolTransformForManifest(manifest, cfg.postPoolTransform); err != nil {
		return nil, err
	}
	prog, err := rt.Load(ctx, mod, opts...)
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{program: prog, manifest: manifest, postPoolTransform: cfg.postPoolTransform}, nil
}

// LoadEmbeddingFile reads a .mll artifact and loads it as an embedding model.
func (rt *Runtime) LoadEmbeddingFile(ctx context.Context, artifactPath string, manifest EmbeddingManifest, opts ...LoadOption) (*EmbeddingModel, error) {
	mod, err := eosartifact.ReadFile(artifactPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadEmbedding(ctx, mod, manifest, opts...)
}

// LoadEmbeddingBundle reads a .mll artifact plus its sibling embedding manifest.
func (rt *Runtime) LoadEmbeddingBundle(ctx context.Context, artifactPath string, opts ...LoadOption) (*EmbeddingModel, error) {
	return rt.LoadEmbeddingBundleWithManifest(ctx, artifactPath, ResolveEmbeddingManifestPath(artifactPath), opts...)
}

// LoadEmbeddingBundleWithManifest reads a .mll artifact plus an explicit embedding manifest path.
func (rt *Runtime) LoadEmbeddingBundleWithManifest(ctx context.Context, artifactPath, manifestPath string, opts ...LoadOption) (*EmbeddingModel, error) {
	manifest, err := ReadEmbeddingManifestFile(manifestPath)
	if err != nil {
		return nil, err
	}
	return rt.LoadEmbeddingFile(ctx, artifactPath, manifest, opts...)
}

func (m EmbeddingManifest) validateServingGraphSupported() error {
	return nil
}

func (m EmbeddingManifest) nameOrDefault() string {
	if m.Name != "" {
		return m.Name
	}
	return "embedding_manifest"
}

func (m EmbeddingManifest) mllValues() map[string]authoredManifestValue {
	return map[string]authoredManifestValue{
		"name":                    authoredString(m.Name),
		"pooled_entry":            authoredString(m.PooledEntry),
		"batch_entry":             authoredString(m.BatchEntry),
		"encoder_repeats":         authoredInt(int64(m.EncoderRepeats)),
		"token_input":             authoredString(m.TokenInput),
		"mask_input":              authoredString(m.MaskInput),
		"output_name":             authoredString(m.OutputName),
		"output_dtype":            authoredString(m.OutputDType),
		"architecture_version":    authoredString(m.ArchitectureVersion),
		"model_dim":               authoredInt(int64(m.ModelDim)),
		"output_dim":              authoredInt(int64(m.OutputDim)),
		"attention_heads":         authoredInt(int64(m.AttentionHeads)),
		"head_dim":                authoredInt(int64(m.HeadDim)),
		"ffn_dim":                 authoredInt(int64(m.FFNDim)),
		"parameter_tying":         authoredString(m.ParameterTying),
		"token_embedding_param":   authoredString(m.TokenEmbeddingParam),
		"role_conditioning":       authoredString(m.RoleConditioning),
		"role_embedding_param":    authoredString(m.RoleEmbeddingParam),
		"role_input":              authoredString(m.RoleInput),
		"batch_role_input":        authoredString(m.BatchRoleInput),
		"raw_role_index":          authoredInt(int64(m.RawRoleIndex)),
		"query_role_index":        authoredInt(int64(m.QueryRoleIndex)),
		"document_role_index":     authoredInt(int64(m.DocumentRoleIndex)),
		"attention_query_param":   authoredString(m.AttentionQueryParam),
		"attention_key_param":     authoredString(m.AttentionKeyParam),
		"attention_value_param":   authoredString(m.AttentionValueParam),
		"attention_output_param":  authoredString(m.AttentionOutputParam),
		"attention_mask_mode":     authoredString(m.AttentionMaskMode),
		"attention_score_scale":   authoredString(m.AttentionScoreScale),
		"attention_residual":      authoredBool(m.AttentionResidual),
		"attention_layernorm":     authoredBool(m.AttentionLayerNorm),
		"position_encoding":       authoredString(m.PositionEncoding),
		"hidden_projection_param": authoredString(m.HiddenProjectionParam),
		"ffn_residual":            authoredBool(m.FFNResidual),
		"ffn_layernorm":           authoredBool(m.FFNLayerNorm),
		"projection_param":        authoredString(m.ProjectionParam),
		"output_projection_param": authoredString(m.OutputProjectionParam),
		"post_pool_transform":     authoredString(m.PostPoolTransform),
		"tokenizer.vocab_size":    authoredInt(int64(m.Tokenizer.VocabSize)),
		"tokenizer.max_sequence":  authoredInt(int64(m.Tokenizer.MaxSequence)),
		"tokenizer.pad_id":        authoredInt(int64(m.Tokenizer.PadID)),
		"tokenizer.bos_id":        authoredInt(int64(m.Tokenizer.BOSID)),
		"tokenizer.eos_id":        authoredInt(int64(m.Tokenizer.EOSID)),
		"tokenizer.unknown_id":    authoredInt(int64(m.Tokenizer.UnknownID)),
	}
}

func embeddingManifestFromDoc(doc authoredManifestDoc) (EmbeddingManifest, error) {
	var manifest EmbeddingManifest
	if value, _, err := doc.string("name"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Name = value
	}
	if value, _, err := doc.string("pooled_entry"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.PooledEntry = value
	}
	if value, _, err := doc.string("batch_entry"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.BatchEntry = value
	}
	if value, _, err := doc.int("encoder_repeats"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.EncoderRepeats = int(value)
	}
	if value, _, err := doc.string("token_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.TokenInput = value
	}
	if value, _, err := doc.string("mask_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.MaskInput = value
	}
	if value, _, err := doc.string("output_name"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputName = value
	}
	if value, _, err := doc.string("output_dtype"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputDType = value
	}
	if value, _, err := doc.string("architecture_version"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.ArchitectureVersion = value
	}
	if value, _, err := doc.int("model_dim"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.ModelDim = int(value)
	}
	if value, _, err := doc.int("output_dim"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputDim = int(value)
	}
	if value, _, err := doc.int("attention_heads"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionHeads = int(value)
	}
	if value, _, err := doc.int("head_dim"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.HeadDim = int(value)
	}
	if value, _, err := doc.int("ffn_dim"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.FFNDim = int(value)
	}
	if value, _, err := doc.string("parameter_tying"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.ParameterTying = value
	}
	if value, _, err := doc.string("token_embedding_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.TokenEmbeddingParam = value
	}
	if value, _, err := doc.string("role_conditioning"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.RoleConditioning = value
	}
	if value, _, err := doc.string("role_embedding_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.RoleEmbeddingParam = value
	}
	if value, _, err := doc.string("role_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.RoleInput = value
	}
	if value, _, err := doc.string("batch_role_input"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.BatchRoleInput = value
	}
	if value, _, err := doc.int("raw_role_index"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.RawRoleIndex = int32(value)
	}
	if value, _, err := doc.int("query_role_index"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.QueryRoleIndex = int32(value)
	}
	if value, _, err := doc.int("document_role_index"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.DocumentRoleIndex = int32(value)
	}
	if value, _, err := doc.string("attention_query_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionQueryParam = value
	}
	if value, _, err := doc.string("attention_key_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionKeyParam = value
	}
	if value, _, err := doc.string("attention_value_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionValueParam = value
	}
	if value, _, err := doc.string("attention_output_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionOutputParam = value
	}
	if value, _, err := doc.string("attention_mask_mode"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionMaskMode = value
	}
	if value, _, err := doc.string("attention_score_scale"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionScoreScale = value
	}
	if value, _, err := doc.bool("attention_residual"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionResidual = value
	}
	if value, _, err := doc.bool("attention_layernorm"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.AttentionLayerNorm = value
	}
	if value, _, err := doc.string("position_encoding"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.PositionEncoding = value
	}
	if value, _, err := doc.string("hidden_projection_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.HiddenProjectionParam = value
	}
	if value, _, err := doc.bool("ffn_residual"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.FFNResidual = value
	}
	if value, _, err := doc.bool("ffn_layernorm"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.FFNLayerNorm = value
	}
	if value, _, err := doc.string("projection_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.ProjectionParam = value
	}
	if value, _, err := doc.string("output_projection_param"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.OutputProjectionParam = value
	}
	if value, _, err := doc.string("post_pool_transform"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.PostPoolTransform = value
	}
	if value, _, err := doc.int("tokenizer.vocab_size"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.VocabSize = int(value)
	}
	if value, _, err := doc.int("tokenizer.max_sequence"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.MaxSequence = int(value)
	}
	if value, _, err := doc.int("tokenizer.pad_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.PadID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.bos_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.BOSID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.eos_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.EOSID = int32(value)
	}
	if value, _, err := doc.int("tokenizer.unknown_id"); err != nil {
		return EmbeddingManifest{}, err
	} else {
		manifest.Tokenizer.UnknownID = int32(value)
	}
	return manifest, nil
}

// Manifest reports the normalized embedding manifest.
func (m *EmbeddingModel) Manifest() EmbeddingManifest {
	if m == nil {
		return EmbeddingManifest{}
	}
	return m.manifest
}

// Backend reports the selected backend.
func (m *EmbeddingModel) Backend() eosartifact.BackendKind {
	if m == nil || m.program == nil {
		return ""
	}
	return m.program.Backend()
}

// Program exposes the underlying loaded program.
func (m *EmbeddingModel) Program() *Program {
	if m == nil {
		return nil
	}
	return m.program
}

func (m *EmbeddingModel) MemoryPlan() *MemoryPlan {
	if m == nil || m.program == nil {
		return nil
	}
	return m.program.MemoryPlan()
}

// HasTokenizer reports whether this model can tokenize text directly.
func (m *EmbeddingModel) HasTokenizer() bool {
	return m != nil && m.tokenizer != nil
}

// TokenizerFile returns the tokenizer metadata attached to this model.
func (m *EmbeddingModel) TokenizerFile() (TokenizerFile, bool) {
	if m == nil || m.tokenizerFile == nil {
		return TokenizerFile{}, false
	}
	return cloneTokenizerFile(*m.tokenizerFile), true
}

// WithTokenizerMaxSequenceOverride returns a cloned model handle whose
// tokenizer contract is raised for diagnostic export paths.
func (m *EmbeddingModel) WithTokenizerMaxSequenceOverride(maxSeq int) (*EmbeddingModel, error) {
	if maxSeq < 0 {
		return nil, fmt.Errorf("tokenizer max sequence override must be non-negative")
	}
	if m == nil {
		return nil, fmt.Errorf("embedding model is not loaded")
	}
	if maxSeq == 0 {
		return m, nil
	}
	current := m.manifest.Tokenizer.MaxSequence
	if current == 0 || maxSeq <= current {
		return m, nil
	}
	cloned := *m
	cloned.manifest = m.manifest
	cloned.manifest.Tokenizer.MaxSequence = maxSeq
	cloned.tokenizer = nil
	cloned.tokenizerFile = nil
	if m.tokenizerFile != nil {
		if err := cloned.attachTokenizer(*m.tokenizerFile); err != nil {
			return nil, err
		}
	}
	return &cloned, nil
}

// TokenizeText tokenizes text with the tokenizer packaged beside or inside the model.
func (m *EmbeddingModel) TokenizeText(text string) ([]int32, []int32, error) {
	if m == nil {
		return nil, nil, fmt.Errorf("embedding model is not loaded")
	}
	if m.tokenizer == nil {
		return nil, nil, fmt.Errorf("embedding model has no tokenizer")
	}
	return m.tokenizer.Encode(text)
}

// EmbedText tokenizes text and executes the pooled embedding entrypoint.
func (m *EmbeddingModel) EmbedText(ctx context.Context, text string) (EmbeddingResult, error) {
	return m.EmbedTextWithRole(ctx, text, EmbeddingRoleRaw)
}

// EmbedTextWithRole tokenizes text and executes the pooled embedding entrypoint with an explicit semantic role.
func (m *EmbeddingModel) EmbedTextWithRole(ctx context.Context, text, role string) (EmbeddingResult, error) {
	tokens, _, err := m.TokenizeText(text)
	if err != nil {
		return EmbeddingResult{}, err
	}
	return m.EmbedWithRole(ctx, tokens, role)
}

// EmbedTextBatch tokenizes text rows and executes the batched pooled embedding entrypoint.
func (m *EmbeddingModel) EmbedTextBatch(ctx context.Context, texts []string) (EmbeddingResult, error) {
	return m.EmbedTextBatchWithRole(ctx, texts, EmbeddingRoleRaw)
}

// EmbedTextBatchWithRole tokenizes text rows and executes the batched pooled embedding entrypoint with an explicit role.
func (m *EmbeddingModel) EmbedTextBatchWithRole(ctx context.Context, texts []string, role string) (EmbeddingResult, error) {
	if m == nil {
		return EmbeddingResult{}, fmt.Errorf("embedding model is not loaded")
	}
	if len(texts) == 0 {
		return EmbeddingResult{}, fmt.Errorf("texts are empty")
	}
	batches := make([][]int32, 0, len(texts))
	for i, text := range texts {
		tokens, _, err := m.TokenizeText(text)
		if err != nil {
			return EmbeddingResult{}, fmt.Errorf("text %d: %w", i, err)
		}
		batches = append(batches, tokens)
	}
	return m.EmbedBatchWithRole(ctx, batches, role)
}

func (m *EmbeddingModel) EmbedQueryText(ctx context.Context, text string) (EmbeddingResult, error) {
	return m.EmbedTextWithRole(ctx, text, EmbeddingRoleQuery)
}

func (m *EmbeddingModel) EmbedDocumentText(ctx context.Context, text string) (EmbeddingResult, error) {
	return m.EmbedTextWithRole(ctx, text, EmbeddingRoleDocument)
}

// Embed executes the pooled embedding entrypoint for one token sequence.
func (m *EmbeddingModel) Embed(ctx context.Context, tokens []int32) (EmbeddingResult, error) {
	return m.EmbedWithRole(ctx, tokens, EmbeddingRoleRaw)
}

// EmbedWithRole executes the pooled embedding entrypoint for one token sequence with an explicit semantic role.
func (m *EmbeddingModel) EmbedWithRole(ctx context.Context, tokens []int32, role string) (EmbeddingResult, error) {
	if m == nil || m.program == nil {
		return EmbeddingResult{}, fmt.Errorf("embedding model is not loaded")
	}
	if err := m.validateTokenSequence(tokens); err != nil {
		return EmbeddingResult{}, err
	}
	roleIndex, err := m.manifest.roleIndex(role)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if m.manifest.MaskInput == "" && !m.manifest.roleConditioned() {
		result, err := m.program.RunEmbed(ctx, m.manifest.PooledEntry, tokens)
		if err != nil {
			return EmbeddingResult{}, err
		}
		if err := m.validateEmbeddingResult(result, false); err != nil {
			return EmbeddingResult{}, err
		}
		return m.applyPostPoolTransform(result)
	}
	entry, err := findEntryPoint(m.program.module, m.manifest.PooledEntry)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenInput, err := requireEntryInput(entry, m.manifest.TokenInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	var tokenTensor *backend.Tensor
	inputs := map[string]any{
		tokenInput.Name: tokenTensor,
	}
	if m.manifest.MaskInput != "" {
		maskInput, err := requireEntryInput(entry, m.manifest.MaskInput)
		if err != nil {
			return EmbeddingResult{}, err
		}
		var maskTensor *backend.Tensor
		tokenTensor, maskTensor, err = buildMaskedTokenInputs([][]int32{tokens}, m.manifest.Tokenizer.PadID, false)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[tokenInput.Name] = tokenTensor
		inputs[maskInput.Name] = maskTensor
	} else {
		tokenTensor, err = buildTokenInputTensor([][]int32{tokens}, m.manifest.Tokenizer.PadID, false)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[tokenInput.Name] = tokenTensor
	}
	if m.manifest.roleConditioned() {
		roleInput, err := requireEntryInput(entry, m.manifest.RoleInput)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[roleInput.Name] = buildRoleInputTensor([][]int32{tokens}, roleIndex, false)
	}
	raw, err := m.program.Run(ctx, backend.Request{
		Entry:  m.manifest.PooledEntry,
		Inputs: inputs,
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	result, err := decodeEmbeddingResult(raw)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if err := m.validateEmbeddingResult(result, false); err != nil {
		return EmbeddingResult{}, err
	}
	return m.applyPostPoolTransform(result)
}

// EmbedBatch executes the batched pooled embedding entrypoint.
func (m *EmbeddingModel) EmbedBatch(ctx context.Context, batches [][]int32) (EmbeddingResult, error) {
	return m.EmbedBatchWithRole(ctx, batches, EmbeddingRoleRaw)
}

// EmbedBatchWithRole executes the batched pooled embedding entrypoint with an explicit semantic role.
func (m *EmbeddingModel) EmbedBatchWithRole(ctx context.Context, batches [][]int32, role string) (EmbeddingResult, error) {
	return m.embedBatchWithRole(ctx, batches, role, true)
}

func (m *EmbeddingModel) embedBatchWithRole(ctx context.Context, batches [][]int32, role string, applyTransform bool) (EmbeddingResult, error) {
	if m == nil || m.program == nil {
		return EmbeddingResult{}, fmt.Errorf("embedding model is not loaded")
	}
	if len(batches) == 0 {
		return EmbeddingResult{}, fmt.Errorf("token batches are empty")
	}
	for i, batch := range batches {
		if err := m.validateTokenSequence(batch); err != nil {
			return EmbeddingResult{}, fmt.Errorf("batch %d: %w", i, err)
		}
	}
	roleIndex, err := m.manifest.roleIndex(role)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if m.manifest.MaskInput != "" && raggedTokenBatches(batches) {
		return m.embedBatchByTokenLength(ctx, batches, role)
	}
	if m.manifest.MaskInput == "" && !m.manifest.roleConditioned() {
		result, err := m.program.RunEmbedBatch(ctx, m.manifest.BatchEntry, batches)
		if err != nil {
			return EmbeddingResult{}, err
		}
		if err := m.validateEmbeddingResult(result, true); err != nil {
			return EmbeddingResult{}, err
		}
		if !applyTransform {
			return result, nil
		}
		return m.applyPostPoolTransform(result)
	}
	entry, err := findEntryPoint(m.program.module, m.manifest.BatchEntry)
	if err != nil {
		return EmbeddingResult{}, err
	}
	tokenInput, err := requireEntryInput(entry, m.manifest.TokenInput)
	if err != nil {
		return EmbeddingResult{}, err
	}
	var tokenTensor *backend.Tensor
	inputs := map[string]any{
		tokenInput.Name: tokenTensor,
	}
	if m.manifest.MaskInput != "" {
		maskInput, err := requireEntryInput(entry, m.manifest.MaskInput)
		if err != nil {
			return EmbeddingResult{}, err
		}
		var maskTensor *backend.Tensor
		tokenTensor, maskTensor, err = buildMaskedTokenInputs(batches, m.manifest.Tokenizer.PadID, true)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[tokenInput.Name] = tokenTensor
		inputs[maskInput.Name] = maskTensor
	} else {
		tokenTensor, err = buildTokenInputTensor(batches, m.manifest.Tokenizer.PadID, true)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[tokenInput.Name] = tokenTensor
	}
	if m.manifest.roleConditioned() {
		roleInput, err := requireEntryInput(entry, m.manifest.BatchRoleInput)
		if err != nil {
			return EmbeddingResult{}, err
		}
		inputs[roleInput.Name] = buildRoleInputTensor(batches, roleIndex, true)
	}
	raw, err := m.program.Run(ctx, backend.Request{
		Entry:  m.manifest.BatchEntry,
		Inputs: inputs,
	})
	if err != nil {
		return EmbeddingResult{}, err
	}
	result, err := decodeEmbeddingResult(raw)
	if err != nil {
		return EmbeddingResult{}, err
	}
	if err := m.validateEmbeddingResult(result, true); err != nil {
		return EmbeddingResult{}, err
	}
	if !applyTransform {
		return result, nil
	}
	return m.applyPostPoolTransform(result)
}

func (m *EmbeddingModel) applyPostPoolTransform(result EmbeddingResult) (EmbeddingResult, error) {
	if m == nil || m.postPoolTransform == nil {
		return result, nil
	}
	transformed, err := m.postPoolTransform.ApplyTensor(result.Embeddings)
	if err != nil {
		return EmbeddingResult{}, err
	}
	result.Embeddings = transformed
	if result.Raw.Outputs != nil && result.OutputName != "" {
		value := result.Raw.Outputs[result.OutputName]
		value.Data = transformed
		result.Raw.Outputs[result.OutputName] = value
	}
	return result, nil
}

type embeddingBatchSlot struct {
	index  int
	tokens []int32
}

func raggedTokenBatches(batches [][]int32) bool {
	if len(batches) < 2 {
		return false
	}
	n := len(batches[0])
	for _, batch := range batches[1:] {
		if len(batch) != n {
			return true
		}
	}
	return false
}

func (m *EmbeddingModel) embedBatchByTokenLength(ctx context.Context, batches [][]int32, role string) (EmbeddingResult, error) {
	groups := map[int][]embeddingBatchSlot{}
	lengths := []int{}
	for i, batch := range batches {
		n := len(batch)
		if len(groups[n]) == 0 {
			lengths = append(lengths, n)
		}
		groups[n] = append(groups[n], embeddingBatchSlot{index: i, tokens: batch})
	}
	slices.Sort(lengths)
	var outputName string
	var dtype string
	var width int
	rows := make([][]float32, len(batches))
	trace := []backend.TraceStep{}
	for _, n := range lengths {
		slots := groups[n]
		groupBatches := make([][]int32, len(slots))
		for i, slot := range slots {
			groupBatches[i] = slot.tokens
		}
		result, err := m.embedBatchWithRole(ctx, groupBatches, role, false)
		if err != nil {
			return EmbeddingResult{}, err
		}
		if result.Embeddings == nil {
			return EmbeddingResult{}, fmt.Errorf("embedding output tensor is nil")
		}
		if outputName == "" {
			outputName = result.OutputName
			dtype = result.Embeddings.DType
			if len(result.Embeddings.Shape) != 2 {
				return EmbeddingResult{}, fmt.Errorf("grouped embedding shape = %v, want rank 2", result.Embeddings.Shape)
			}
			width = result.Embeddings.Shape[1]
		} else if result.OutputName != outputName || result.Embeddings.DType != dtype || len(result.Embeddings.Shape) != 2 || result.Embeddings.Shape[1] != width {
			return EmbeddingResult{}, fmt.Errorf("grouped embedding output changed from %s/%s/%d to %s/%s/%v", outputName, dtype, width, result.OutputName, result.Embeddings.DType, result.Embeddings.Shape)
		}
		groupRows, err := embeddingRows(result.Embeddings, len(slots))
		if err != nil {
			return EmbeddingResult{}, err
		}
		for i, slot := range slots {
			rows[slot.index] = groupRows[i]
		}
		trace = append(trace, result.Raw.Trace...)
	}
	data := make([]float32, 0, len(batches)*width)
	for i, row := range rows {
		if len(row) != width {
			return EmbeddingResult{}, fmt.Errorf("missing grouped embedding row %d", i)
		}
		data = append(data, row...)
	}
	tensor := &backend.Tensor{
		DType: dtype,
		Shape: []int{len(batches), width},
		F32:   data,
	}
	result := EmbeddingResult{
		OutputName: outputName,
		Embeddings: tensor,
		Raw: backend.Result{
			Outputs: map[string]backend.Value{
				outputName: {
					Type: eosartifact.ValueType{
						Kind: eosartifact.ValueTensor,
						Tensor: &eosartifact.TensorType{
							DType: dtype,
							Shape: []string{strconv.Itoa(len(batches)), strconv.Itoa(width)},
						},
					},
					Data: tensor,
				},
			},
			Trace: trace,
		},
	}
	if err := m.validateEmbeddingResult(result, true); err != nil {
		return EmbeddingResult{}, err
	}
	return m.applyPostPoolTransform(result)
}

func embeddingRows(t *backend.Tensor, wantRows int) ([][]float32, error) {
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
		return [][]float32{append([]float32(nil), t.F32...)}, nil
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
			out[i] = append([]float32(nil), t.F32[i*cols:(i+1)*cols]...)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("embedding tensor shape %v is not rank 1 or 2", t.Shape)
	}
}

func (m EmbeddingManifest) normalized() EmbeddingManifest {
	if m.PooledEntry == "" {
		m.PooledEntry = "embed_pooled"
	}
	if m.BatchEntry == "" {
		m.BatchEntry = "embed_pooled_batch"
	}
	if m.EncoderRepeats <= 0 {
		m.EncoderRepeats = 1
	}
	if m.TokenInput == "" {
		m.TokenInput = "tokens"
	}
	if m.OutputDType == "" {
		m.OutputDType = "f16"
	}
	if m.TokenEmbeddingParam == "" {
		m.TokenEmbeddingParam = "token_embedding"
	}
	if m.RoleConditioning == "" {
		m.RoleConditioning = EmbeddingRoleConditioningNone
	}
	if m.roleConditioned() {
		if m.RoleEmbeddingParam == "" {
			m.RoleEmbeddingParam = "role_embedding"
		}
		if m.RoleInput == "" {
			m.RoleInput = "role_ids"
		}
		if m.BatchRoleInput == "" {
			m.BatchRoleInput = "role_ids"
		}
		if m.QueryRoleIndex == 0 && m.DocumentRoleIndex == 0 {
			m.RawRoleIndex = 0
			m.QueryRoleIndex = 1
			m.DocumentRoleIndex = 2
		}
	}
	if m.ProjectionParam == "" {
		m.ProjectionParam = "projection"
	}
	if m.AttentionMaskMode == "" {
		m.AttentionMaskMode = EmbeddingAttentionMaskModeNone
	}
	if m.AttentionScoreScale == "" {
		m.AttentionScoreScale = EmbeddingAttentionScoreScaleNone
	}
	if m.PositionEncoding == "" {
		m.PositionEncoding = EmbeddingPositionEncodingNone
	}
	if m.ArchitectureVersion == "" && m.hasArchitectureMetadata() {
		m.ArchitectureVersion = EmbeddingArchitectureLegacyV1
	}
	if m.ParameterTying == "" && m.hasArchitectureMetadata() {
		m.ParameterTying = EmbeddingParameterTyingLegacyTied
	}
	if m.AttentionHeads == 0 && m.hasArchitectureMetadata() {
		m.AttentionHeads = 1
	}
	if m.HeadDim == 0 && m.ModelDim > 0 && m.AttentionHeads > 0 && m.ModelDim%m.AttentionHeads == 0 {
		m.HeadDim = m.ModelDim / m.AttentionHeads
	}
	return m
}

func (m EmbeddingManifest) normalizedForModule(mod *eosartifact.Module) EmbeddingManifest {
	m = m.normalized()
	if m.ModelDim == 0 {
		if dim, ok := moduleParamDim(mod, m.TokenEmbeddingParam, 1); ok {
			m.ModelDim = dim
		}
	}
	if m.OutputDim == 0 {
		if dim, ok := moduleParamDim(mod, m.ProjectionParam, 1); ok {
			m.OutputDim = dim
		} else if m.ModelDim > 0 {
			m.OutputDim = m.ModelDim
		}
	}
	if m.FFNDim == 0 {
		if dim, ok := moduleParamDim(mod, m.HiddenProjectionParam, 1); ok {
			m.FFNDim = dim
		}
	}
	if m.AttentionHeads == 0 && (m.ModelDim > 0 || m.OutputDim > 0 || m.FFNDim > 0) {
		m.AttentionHeads = 1
	}
	if m.HeadDim == 0 && m.ModelDim > 0 && m.AttentionHeads > 0 && m.ModelDim%m.AttentionHeads == 0 {
		m.HeadDim = m.ModelDim / m.AttentionHeads
	}
	if m.ArchitectureVersion == "" && (m.ModelDim > 0 || m.OutputDim > 0 || m.FFNDim > 0 || m.AttentionHeads > 0) {
		m.ArchitectureVersion = EmbeddingArchitectureLegacyV1
	}
	if m.ParameterTying == "" && (m.ModelDim > 0 || m.OutputDim > 0 || m.FFNDim > 0 || m.AttentionHeads > 0) {
		m.ParameterTying = EmbeddingParameterTyingLegacyTied
	}
	return m
}

func (m EmbeddingManifest) hasArchitectureMetadata() bool {
	return m.ArchitectureVersion != "" ||
		m.ModelDim != 0 ||
		m.OutputDim != 0 ||
		m.AttentionHeads != 0 ||
		m.HeadDim != 0 ||
		m.FFNDim != 0 ||
		m.ParameterTying != ""
}

func (m EmbeddingManifest) roleConditioned() bool {
	return m.RoleConditioning == EmbeddingRoleConditioningAdditiveV1
}

func (m EmbeddingManifest) roleIndex(role string) (int32, error) {
	role = strings.TrimSpace(strings.ToLower(role))
	if role == "" {
		role = EmbeddingRoleRaw
	}
	switch role {
	case EmbeddingRoleRaw:
		return m.RawRoleIndex, nil
	case EmbeddingRoleQuery:
		if !m.roleConditioned() {
			return 0, fmt.Errorf("embedding model does not support query role")
		}
		return m.QueryRoleIndex, nil
	case EmbeddingRoleDocument:
		if !m.roleConditioned() {
			return 0, fmt.Errorf("embedding model does not support document role")
		}
		return m.DocumentRoleIndex, nil
	default:
		return 0, fmt.Errorf("unsupported embedding role %q", role)
	}
}

// ValidateModule checks that a module satisfies the embedding serving contract.
func (m EmbeddingManifest) ValidateModule(mod *eosartifact.Module) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	m = m.normalizedForModule(mod)
	if err := m.validateArchitectureMetadata(); err != nil {
		return err
	}
	if err := m.validatePostPoolTransformDeclaration(); err != nil {
		return err
	}
	if (m.AttentionResidual || m.AttentionLayerNorm) && m.AttentionQueryParam == "" {
		return fmt.Errorf("attention residual/layernorm requires attention params")
	}
	if (m.FFNResidual || m.FFNLayerNorm) && m.HiddenProjectionParam == "" {
		return fmt.Errorf("ffn residual/layernorm requires hidden_projection_param")
	}
	switch m.AttentionMaskMode {
	case "", EmbeddingAttentionMaskModeNone:
	case EmbeddingAttentionMaskModeKey:
		if m.MaskInput == "" {
			return fmt.Errorf("attention_mask_mode=%q requires mask_input", m.AttentionMaskMode)
		}
		if !moduleHasKernelOp(mod, "masked_softmax") && !moduleHasKernelOp(mod, "compact_multihead_attention") {
			return fmt.Errorf("attention_mask_mode=%q requires masked_softmax or compact_multihead_attention in serving graph", m.AttentionMaskMode)
		}
	default:
		return fmt.Errorf("unsupported attention_mask_mode %q", m.AttentionMaskMode)
	}
	switch m.AttentionScoreScale {
	case "", EmbeddingAttentionScoreScaleNone:
	case EmbeddingAttentionScoreScaleKeyDimRSQ:
		if !moduleHasScaledAttentionMatMul(mod) && !moduleHasKernelOp(mod, "compact_multihead_attention") {
			return fmt.Errorf("attention_score_scale=%q requires scaled attention score matmul or compact_multihead_attention in serving graph", m.AttentionScoreScale)
		}
	default:
		return fmt.Errorf("unsupported attention_score_scale %q", m.AttentionScoreScale)
	}
	switch m.PositionEncoding {
	case "", EmbeddingPositionEncodingNone:
	case EmbeddingPositionEncodingRoPE:
		if !moduleHasKernelOp(mod, "rope") {
			return fmt.Errorf("position_encoding=%q requires rope in serving graph", m.PositionEncoding)
		}
	default:
		return fmt.Errorf("unsupported position_encoding %q", m.PositionEncoding)
	}
	switch m.RoleConditioning {
	case "", EmbeddingRoleConditioningNone:
	case EmbeddingRoleConditioningAdditiveV1:
		if err := validateEmbeddingParam(mod, m.RoleEmbeddingParam); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported role_conditioning %q", m.RoleConditioning)
	}
	if err := validateEmbeddingEntry(mod, m.PooledEntry, m.TokenInput, m.MaskInput, 1, 1, m.OutputDType); err != nil {
		return err
	}
	if err := validateEmbeddingEntry(mod, m.BatchEntry, m.TokenInput, m.MaskInput, 2, 2, m.OutputDType); err != nil {
		return err
	}
	if m.roleConditioned() {
		if err := validateEmbeddingRoleEntry(mod, m.PooledEntry, m.RoleInput, 1); err != nil {
			return err
		}
		if err := validateEmbeddingRoleEntry(mod, m.BatchEntry, m.BatchRoleInput, 2); err != nil {
			return err
		}
	}
	if err := validateEmbeddingParam(mod, m.TokenEmbeddingParam); err != nil {
		return err
	}
	if err := validateAttentionParams(mod, m); err != nil {
		return err
	}
	if m.HiddenProjectionParam != "" {
		if err := validateEmbeddingParam(mod, m.HiddenProjectionParam); err != nil {
			return err
		}
	}
	if m.OutputProjectionParam != "" {
		if err := validateEmbeddingParam(mod, m.OutputProjectionParam); err != nil {
			return err
		}
	}
	if err := validateEmbeddingParam(mod, m.ProjectionParam); err != nil {
		return err
	}
	return nil
}

func (m EmbeddingManifest) validateArchitectureMetadata() error {
	if !m.hasArchitectureMetadata() {
		return nil
	}
	switch m.ArchitectureVersion {
	case "", EmbeddingArchitectureLegacyV1, EmbeddingArchitectureCompactTransformerV1:
	default:
		return fmt.Errorf("unsupported architecture_version %q", m.ArchitectureVersion)
	}
	if m.ModelDim <= 0 {
		return fmt.Errorf("model_dim must be positive when architecture metadata is declared")
	}
	if m.OutputDim <= 0 {
		return fmt.Errorf("output_dim must be positive when architecture metadata is declared")
	}
	if m.AttentionHeads <= 0 {
		return fmt.Errorf("attention_heads must be positive when architecture metadata is declared")
	}
	if m.ModelDim%m.AttentionHeads != 0 {
		return fmt.Errorf("model_dim %d must be divisible by attention_heads %d", m.ModelDim, m.AttentionHeads)
	}
	if m.HeadDim <= 0 {
		return fmt.Errorf("head_dim must be positive when architecture metadata is declared")
	}
	if want := m.ModelDim / m.AttentionHeads; m.HeadDim != want {
		return fmt.Errorf("head_dim = %d, want model_dim/attention_heads = %d", m.HeadDim, want)
	}
	if m.FFNDim < 0 {
		return fmt.Errorf("ffn_dim must be non-negative")
	}
	switch m.ParameterTying {
	case "", EmbeddingParameterTyingLegacyTied, EmbeddingParameterTyingUntied:
	default:
		return fmt.Errorf("unsupported parameter_tying %q", m.ParameterTying)
	}
	if m.OutputProjectionParam != "" {
		if strings.TrimSpace(m.OutputProjectionParam) == "" {
			return fmt.Errorf("output_projection_param must not be blank")
		}
	} else if m.ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 && m.OutputDim != m.ModelDim {
		return fmt.Errorf("output_projection_param is required when architecture_version=%q and output_dim (%d) differs from model_dim (%d)", m.ArchitectureVersion, m.OutputDim, m.ModelDim)
	}
	return nil
}

func (m EmbeddingManifest) validatePostPoolTransformDeclaration() error {
	switch m.PostPoolTransform {
	case "", EmbeddingPostPoolTransformNone, EmbeddingPostPoolTransformAOQTGivens:
		return nil
	default:
		return fmt.Errorf("unsupported post_pool_transform %q", m.PostPoolTransform)
	}
}

// ValidateLegacyEmbeddingTrainerSupported rejects manifests that need
// architecture execution paths the current tied trainer has not implemented.
func (m EmbeddingManifest) ValidateLegacyEmbeddingTrainerSupported() error {
	m = m.normalized()
	arch := m.ArchitectureVersion
	if arch == "" {
		arch = EmbeddingArchitectureLegacyV1
	}
	if arch != EmbeddingArchitectureLegacyV1 {
		return fmt.Errorf("%s is not supported by trainable package initialization yet", arch)
	}
	if m.ParameterTying != "" && m.ParameterTying != EmbeddingParameterTyingLegacyTied {
		return fmt.Errorf("%s with parameter_tying=%q is not supported by trainable package initialization yet", arch, m.ParameterTying)
	}
	if m.OutputProjectionParam != "" {
		return fmt.Errorf("%s with output_projection_param=%q is not supported by trainable package initialization yet", arch, m.OutputProjectionParam)
	}
	return nil
}

func validateEmbeddingRoleEntry(mod *eosartifact.Module, entryName, roleInput string, rank int) error {
	entry, err := findEntryPoint(mod, entryName)
	if err != nil {
		return err
	}
	input, err := requireEntryInput(entry, roleInput)
	if err != nil {
		return err
	}
	if input.Type.Kind != eosartifact.ValueTensor || input.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, roleInput)
	}
	if input.Type.Tensor.DType != "i32" {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want i32", entryName, roleInput, input.Type.Tensor.DType)
	}
	if got := len(input.Type.Tensor.Shape); got != rank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, roleInput, got, rank)
	}
	return nil
}

func moduleHasScaledAttentionMatMul(mod *eosartifact.Module) bool {
	if mod == nil {
		return false
	}
	for _, step := range mod.Steps {
		if step.Kind == eosartifact.StepMatMul && step.Attributes != nil && step.Attributes["scale"] == "rsqrt_rhs_rows" {
			return true
		}
	}
	return false
}

func (m *EmbeddingModel) validateTokenSequence(tokens []int32) error {
	if len(tokens) == 0 {
		return fmt.Errorf("tokens are empty")
	}
	if limit := m.manifest.Tokenizer.MaxSequence; limit > 0 && len(tokens) > limit {
		return fmt.Errorf("token sequence length %d exceeds max_sequence %d", len(tokens), limit)
	}
	if vocab := m.manifest.Tokenizer.VocabSize; vocab > 0 {
		for i, tok := range tokens {
			if tok < 0 || int(tok) >= vocab {
				return fmt.Errorf("token %d value %d is outside vocab_size %d", i, tok, vocab)
			}
		}
	}
	return nil
}

func (m *EmbeddingModel) attachTokenizer(file TokenizerFile) error {
	if m == nil {
		return fmt.Errorf("embedding model is not loaded")
	}
	tokenizer, err := NewBPETokenizer(file, m.manifest.Tokenizer)
	if err != nil {
		return err
	}
	cloned := cloneTokenizerFile(file)
	m.tokenizerFile = &cloned
	m.tokenizer = tokenizer
	return nil
}

func cloneTokenizerFile(in TokenizerFile) TokenizerFile {
	out := in
	out.Tokens = append([]string(nil), in.Tokens...)
	out.Merges = append([]TokenizerMerge(nil), in.Merges...)
	return out
}

func (m *EmbeddingModel) validateEmbeddingResult(result EmbeddingResult, batched bool) error {
	if m.manifest.OutputName != "" && result.OutputName != m.manifest.OutputName {
		return fmt.Errorf("embedding output name %q does not match manifest %q", result.OutputName, m.manifest.OutputName)
	}
	if result.Embeddings == nil {
		return fmt.Errorf("embedding output tensor is nil")
	}
	if want := m.manifest.OutputDType; want != "" && result.Embeddings.DType != want {
		return fmt.Errorf("embedding output dtype %q does not match manifest %q", result.Embeddings.DType, want)
	}
	wantRank := 1
	if batched {
		wantRank = 2
	}
	if got := len(result.Embeddings.Shape); got != wantRank {
		return fmt.Errorf("embedding output rank %d does not match expected %d", got, wantRank)
	}
	return nil
}

func validateEmbeddingEntry(mod *eosartifact.Module, entryName, tokenInput, maskInput string, tokenRank, outputRank int, outputDType string) error {
	entry, err := findEntryPoint(mod, entryName)
	if err != nil {
		return err
	}
	input, err := requireEntryInput(entry, tokenInput)
	if err != nil {
		return err
	}
	if input.Type.Kind != eosartifact.ValueTensor || input.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, tokenInput)
	}
	if input.Type.Tensor.DType != "i32" {
		return fmt.Errorf("entrypoint %q input %q dtype = %q, want i32", entryName, tokenInput, input.Type.Tensor.DType)
	}
	if got := len(input.Type.Tensor.Shape); got != tokenRank {
		return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, tokenInput, got, tokenRank)
	}
	if maskInput != "" {
		mask, err := requireEntryInput(entry, maskInput)
		if err != nil {
			return err
		}
		if mask.Type.Kind != eosartifact.ValueTensor || mask.Type.Tensor == nil {
			return fmt.Errorf("entrypoint %q input %q is not a tensor", entryName, maskInput)
		}
		if mask.Type.Tensor.DType != "i32" {
			return fmt.Errorf("entrypoint %q input %q dtype = %q, want i32", entryName, maskInput, mask.Type.Tensor.DType)
		}
		if got := len(mask.Type.Tensor.Shape); got != tokenRank {
			return fmt.Errorf("entrypoint %q input %q rank = %d, want %d", entryName, maskInput, got, tokenRank)
		}
	}
	if len(entry.Outputs) != 1 {
		return fmt.Errorf("entrypoint %q output count = %d, want 1", entryName, len(entry.Outputs))
	}
	output := entry.Outputs[0]
	if output.Type.Kind != eosartifact.ValueTensor || output.Type.Tensor == nil {
		return fmt.Errorf("entrypoint %q output %q is not a tensor", entryName, output.Name)
	}
	if outputDType != "" && output.Type.Tensor.DType != outputDType {
		return fmt.Errorf("entrypoint %q output dtype = %q, want %q", entryName, output.Type.Tensor.DType, outputDType)
	}
	if got := len(output.Type.Tensor.Shape); got != outputRank {
		return fmt.Errorf("entrypoint %q output rank = %d, want %d", entryName, got, outputRank)
	}
	return nil
}

func moduleHasKernelOp(mod *eosartifact.Module, op string) bool {
	if mod == nil || op == "" {
		return false
	}
	for _, kernel := range mod.Kernels {
		for _, bodyOp := range kernel.Body {
			if bodyOp.Op == op {
				return true
			}
		}
	}
	return false
}

func validateEmbeddingParam(mod *eosartifact.Module, name string) error {
	if name == "" {
		return nil
	}
	for _, param := range mod.Params {
		if param.Name == name {
			if param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
				return fmt.Errorf("param %q is not a tensor", name)
			}
			if got := len(param.Type.Tensor.Shape); got != 2 {
				return fmt.Errorf("param %q rank = %d, want 2", name, got)
			}
			return nil
		}
	}
	return fmt.Errorf("missing param %q", name)
}

func moduleParamDim(mod *eosartifact.Module, name string, axis int) (int, bool) {
	if mod == nil || name == "" || axis < 0 {
		return 0, false
	}
	for _, param := range mod.Params {
		if param.Name != name || param.Type.Kind != eosartifact.ValueTensor || param.Type.Tensor == nil {
			continue
		}
		if axis >= len(param.Type.Tensor.Shape) {
			return 0, false
		}
		dim, err := strconv.Atoi(param.Type.Tensor.Shape[axis])
		if err != nil || dim <= 0 {
			return 0, false
		}
		return dim, true
	}
	return 0, false
}

func validateAttentionParams(mod *eosartifact.Module, manifest EmbeddingManifest) error {
	names := []string{
		manifest.AttentionQueryParam,
		manifest.AttentionKeyParam,
		manifest.AttentionValueParam,
		manifest.AttentionOutputParam,
	}
	set := 0
	for _, name := range names {
		if name != "" {
			set++
		}
	}
	if set == 0 {
		return nil
	}
	if set != len(names) {
		return fmt.Errorf("attention params must declare query, key, value, and output together")
	}
	for _, name := range names {
		if err := validateEmbeddingParam(mod, name); err != nil {
			return err
		}
	}
	return nil
}

func findEntryPoint(mod *eosartifact.Module, name string) (eosartifact.EntryPoint, error) {
	if mod == nil {
		return eosartifact.EntryPoint{}, fmt.Errorf("nil module")
	}
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return eosartifact.EntryPoint{}, fmt.Errorf("unknown entrypoint %q", name)
}

func requireEntryInput(entry eosartifact.EntryPoint, name string) (eosartifact.ValueBinding, error) {
	for _, input := range entry.Inputs {
		if input.Name == name {
			return input, nil
		}
	}
	return eosartifact.ValueBinding{}, fmt.Errorf("entrypoint %q does not declare input %q", entry.Name, name)
}

func buildMaskedTokenInputs(batches [][]int32, padID int32, batched bool) (*backend.Tensor, *backend.Tensor, error) {
	if len(batches) == 0 {
		return nil, nil, fmt.Errorf("token batches are empty")
	}
	maxLen := 0
	for i, batch := range batches {
		if len(batch) == 0 {
			return nil, nil, fmt.Errorf("token batch %d is empty", i)
		}
		if len(batch) > maxLen {
			maxLen = len(batch)
		}
	}
	tokenData := make([]int32, 0, len(batches)*maxLen)
	maskData := make([]int32, 0, len(batches)*maxLen)
	for _, batch := range batches {
		tokenData = append(tokenData, batch...)
		activeLen := len(batch)
		for activeLen > 0 && batch[activeLen-1] == padID {
			activeLen--
		}
		for i := range batch {
			if i < activeLen {
				maskData = append(maskData, 1)
			} else {
				maskData = append(maskData, 0)
			}
		}
		for i := len(batch); i < maxLen; i++ {
			tokenData = append(tokenData, padID)
			maskData = append(maskData, 0)
		}
	}
	if !batched {
		return backend.NewTensorI32([]int{maxLen}, tokenData[:maxLen]), backend.NewTensorI32([]int{maxLen}, maskData[:maxLen]), nil
	}
	shape := []int{len(batches), maxLen}
	return backend.NewTensorI32(shape, tokenData), backend.NewTensorI32(shape, maskData), nil
}

func buildTokenInputTensor(batches [][]int32, padID int32, batched bool) (*backend.Tensor, error) {
	if len(batches) == 0 {
		return nil, fmt.Errorf("token batches are empty")
	}
	maxLen := 0
	for i, batch := range batches {
		if len(batch) == 0 {
			return nil, fmt.Errorf("token batch %d is empty", i)
		}
		if len(batch) > maxLen {
			maxLen = len(batch)
		}
	}
	tokenData := make([]int32, 0, len(batches)*maxLen)
	for _, batch := range batches {
		tokenData = append(tokenData, batch...)
		for i := len(batch); i < maxLen; i++ {
			tokenData = append(tokenData, padID)
		}
	}
	if !batched {
		return backend.NewTensorI32([]int{maxLen}, tokenData[:maxLen]), nil
	}
	return backend.NewTensorI32([]int{len(batches), maxLen}, tokenData), nil
}

func buildRoleInputTensor(batches [][]int32, role int32, batched bool) *backend.Tensor {
	maxLen := 0
	for _, batch := range batches {
		if len(batch) > maxLen {
			maxLen = len(batch)
		}
	}
	if !batched {
		data := make([]int32, maxLen)
		for i := range data {
			data[i] = role
		}
		return backend.NewTensorI32([]int{maxLen}, data)
	}
	data := make([]int32, len(batches)*maxLen)
	for i := range data {
		data[i] = role
	}
	return backend.NewTensorI32([]int{len(batches), maxLen}, data)
}
