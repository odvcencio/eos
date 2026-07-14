package eosruntime

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	"m31labs.dev/eos/runtime/backend"
	mll "m31labs.dev/mll"
)

// EmbeddingTrainCheckpointVersion identifies the trainer checkpoint schema.
const EmbeddingTrainCheckpointVersion = "manta/embed-train/v0alpha1"

var tagXCHK = [4]byte{'X', 'C', 'H', 'K'}

// EmbeddingTrainCheckpoint is a resumable state snapshot for the narrow embedder trainer.
type EmbeddingTrainCheckpoint struct {
	Version           string                     `json:"version"`
	Manifest          EmbeddingManifest          `json:"manifest"`
	Config            EmbeddingTrainConfig       `json:"config"`
	Step              int                        `json:"step"`
	TokenEmbedding    *backend.Tensor            `json:"token_embedding"`
	RoleEmbedding     *backend.Tensor            `json:"role_embedding,omitempty"`
	AttentionQuery    *backend.Tensor            `json:"attention_query,omitempty"`
	AttentionKey      *backend.Tensor            `json:"attention_key,omitempty"`
	AttentionValue    *backend.Tensor            `json:"attention_value,omitempty"`
	AttentionOutput   *backend.Tensor            `json:"attention_output,omitempty"`
	HiddenProjection  *backend.Tensor            `json:"hidden_projection,omitempty"`
	Projection        *backend.Tensor            `json:"projection"`
	TokenMoment1      *backend.Tensor            `json:"token_moment_1,omitempty"`
	TokenMoment2      *backend.Tensor            `json:"token_moment_2,omitempty"`
	RoleMoment1       *backend.Tensor            `json:"role_moment_1,omitempty"`
	RoleMoment2       *backend.Tensor            `json:"role_moment_2,omitempty"`
	AttentionQMoment1 *backend.Tensor            `json:"attention_query_moment_1,omitempty"`
	AttentionQMoment2 *backend.Tensor            `json:"attention_query_moment_2,omitempty"`
	AttentionKMoment1 *backend.Tensor            `json:"attention_key_moment_1,omitempty"`
	AttentionKMoment2 *backend.Tensor            `json:"attention_key_moment_2,omitempty"`
	AttentionVMoment1 *backend.Tensor            `json:"attention_value_moment_1,omitempty"`
	AttentionVMoment2 *backend.Tensor            `json:"attention_value_moment_2,omitempty"`
	AttentionOMoment1 *backend.Tensor            `json:"attention_output_moment_1,omitempty"`
	AttentionOMoment2 *backend.Tensor            `json:"attention_output_moment_2,omitempty"`
	HiddenMoment1     *backend.Tensor            `json:"hidden_projection_moment_1,omitempty"`
	HiddenMoment2     *backend.Tensor            `json:"hidden_projection_moment_2,omitempty"`
	ProjMoment1       *backend.Tensor            `json:"projection_moment_1,omitempty"`
	ProjMoment2       *backend.Tensor            `json:"projection_moment_2,omitempty"`
	Tensors           map[string]*backend.Tensor `json:"tensors,omitempty"`
	MomentTensors     map[string]*backend.Tensor `json:"moment_tensors,omitempty"`
}

type checkpointMLLMetadata struct {
	Version            string               `json:"version"`
	Manifest           EmbeddingManifest    `json:"manifest"`
	Config             EmbeddingTrainConfig `json:"config"`
	LogicalTensorDType map[string]string    `json:"logical_tensor_dtypes,omitempty"`
}

// DefaultEmbeddingCheckpointPath returns the conventional sibling checkpoint path for an artifact.
func DefaultEmbeddingCheckpointPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".embed-train.mll")
}

// Checkpoint snapshots the current training state.
func (t *EmbeddingTrainer) Checkpoint() (EmbeddingTrainCheckpoint, error) {
	if t == nil {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("embedding trainer is not initialized")
	}
	if err := t.syncOptimizerStateWithReason(true, "checkpoint"); err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	if t.isCompactTrainer() {
		tensors, moments := compactTrainStateCheckpointMaps(t.compactState)
		return EmbeddingTrainCheckpoint{
			Version:       EmbeddingTrainCheckpointVersion,
			Manifest:      t.manifest,
			Config:        t.config,
			Step:          t.step,
			Tensors:       tensors,
			MomentTensors: moments,
		}, nil
	}
	return EmbeddingTrainCheckpoint{
		Version:           EmbeddingTrainCheckpointVersion,
		Manifest:          t.manifest,
		Config:            t.config,
		Step:              t.step,
		TokenEmbedding:    t.tokenEmbed.Clone(),
		RoleEmbedding:     cloneTensorOrNil(t.roleEmbed),
		AttentionQuery:    cloneTensorOrNil(t.attentionQuery),
		AttentionKey:      cloneTensorOrNil(t.attentionKey),
		AttentionValue:    cloneTensorOrNil(t.attentionValue),
		AttentionOutput:   cloneTensorOrNil(t.attentionOutput),
		HiddenProjection:  cloneTensorOrNil(t.hiddenProjection),
		Projection:        t.projection.Clone(),
		TokenMoment1:      t.tokenMom1.Clone(),
		TokenMoment2:      t.tokenMom2.Clone(),
		RoleMoment1:       cloneTensorOrNil(t.roleMom1),
		RoleMoment2:       cloneTensorOrNil(t.roleMom2),
		AttentionQMoment1: cloneTensorOrNil(t.attnQMom1),
		AttentionQMoment2: cloneTensorOrNil(t.attnQMom2),
		AttentionKMoment1: cloneTensorOrNil(t.attnKMom1),
		AttentionKMoment2: cloneTensorOrNil(t.attnKMom2),
		AttentionVMoment1: cloneTensorOrNil(t.attnVMom1),
		AttentionVMoment2: cloneTensorOrNil(t.attnVMom2),
		AttentionOMoment1: cloneTensorOrNil(t.attnOMom1),
		AttentionOMoment2: cloneTensorOrNil(t.attnOMom2),
		HiddenMoment1:     cloneTensorOrNil(t.hiddenMom1),
		HiddenMoment2:     cloneTensorOrNil(t.hiddenMom2),
		ProjMoment1:       t.projMom1.Clone(),
		ProjMoment2:       t.projMom2.Clone(),
	}, nil
}

// WriteFile writes the checkpoint as an MLL checkpoint container.
func (c EmbeddingTrainCheckpoint) WriteFile(path string) error {
	if err := c.Validate(); err != nil {
		return err
	}
	data, err := encodeEmbeddingCheckpointMLL(c)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// ReadEmbeddingTrainCheckpointFile reads an MLL checkpoint.
func ReadEmbeddingTrainCheckpointFile(path string) (EmbeddingTrainCheckpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	if !eosartifact.IsMLLBytes(data) {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint %q is not an MLL file", path)
	}
	return decodeEmbeddingCheckpointMLL(data)
}

func encodeEmbeddingCheckpointMLL(c EmbeddingTrainCheckpoint) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	var dimsBuilder mll.DimsBuilder
	typeBuilder := mll.NewTypeBuilder()
	parmBuilder := mll.NewParmBuilder()
	entrBuilder := mll.NewEntrBuilder()
	tnsrBuilder := mll.NewTnsrBuilder()
	logicalDTypes := map[string]string{}
	tensorRefs := map[string]mll.Ref{}

	tensors := checkpointTensorMap(c)
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tensor := tensors[name]
		if tensor == nil {
			continue
		}
		shape, err := intShapeToMLLShape(strg, tensor.Shape)
		if err != nil {
			return nil, fmt.Errorf("checkpoint tensor %q shape: %w", name, err)
		}
		storageDType, raw, widened, err := encodeTensorStorage(tensor)
		if err != nil {
			return nil, fmt.Errorf("checkpoint tensor %q encode: %w", name, err)
		}
		if widened {
			logicalDTypes[name] = tensor.DType
		}
		nameIdx := strg.Intern(name)
		typeIndex := typeBuilder.AddTensorType(nameIdx, storageDType, shape)
		paramIndex := uint32(len(tensorRefs))
		parmBuilder.Add(mll.ParmDecl{
			NameIdx: nameIdx,
			TypeRef: mll.Ref{Tag: mll.TagTYPE, Index: typeIndex},
		})
		tensorRefs[name] = mll.Ref{Tag: mll.TagTNSR, Index: paramIndex}
		shape64 := make([]uint64, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim < 0 {
				return nil, fmt.Errorf("checkpoint tensor %q dim %d is negative", name, dim)
			}
			shape64[i] = uint64(dim)
		}
		tnsrBuilder.Add(mll.TensorEntry{
			NameIdx: nameIdx,
			DType:   storageDType,
			Shape:   shape64,
			Data:    raw,
		})
	}

	outputs := make([]mll.ValueBinding, 0, len(names))
	for _, name := range names {
		tensor := tensors[name]
		if tensor == nil {
			continue
		}
		shape, err := intShapeToMLLShape(strg, tensor.Shape)
		if err != nil {
			return nil, err
		}
		typeIdx := typeBuilder.AddTensorType(strg.Intern("entry:"+name), mustEosDTypeToMLL(tensor.DType), shape)
		outputs = append(outputs, mll.ValueBinding{
			NameIdx: strg.Intern(name),
			TypeRef: mll.Ref{Tag: mll.TagTYPE, Index: typeIdx},
		})
	}
	entrBuilder.Add(mll.EntryPoint{
		NameIdx: strg.Intern("trainer_checkpoint"),
		Kind:    mll.EntryKindPipeline,
		Outputs: outputs,
	})

	optmBuilder := mll.NewOptmBuilder(optimizerKindToMLL(c.Config.Optimizer))
	optmBuilder.SetStep(uint64(c.Step))
	optmBuilder.SetGeneration(uint64(c.Step))
	for _, name := range checkpointMomentNames(c) {
		if ref, ok := tensorRefs[name]; ok {
			optmBuilder.AddMomentTensor(ref)
		}
	}

	head := mll.HeadSection{
		Name:        strg.Intern(nonEmptyCheckpointName(c.Manifest.Name)),
		Description: strg.Intern("Eos training checkpoint"),
		Generation:  uint64(c.Step),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, "checkpoint_version", c.Version),
			headIntMeta(strg, "step", int64(c.Step)),
		},
	}
	metaBody, err := json.Marshal(checkpointMLLMetadata{
		Version:            c.Version,
		Manifest:           c.Manifest,
		Config:             c.Config,
		LogicalTensorDType: logicalDTypes,
	})
	if err != nil {
		return nil, err
	}

	sections := make([]mll.SectionInput, 0, 8)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileCheckpoint); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagHEAD, Body: body, DigestBody: digestBody, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSTRG, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagDIMS, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagPARM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagENTR, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTNSR, Body: body, Flags: mll.SectionFlagRequired | mll.SectionFlagAligned, SchemaVersion: 1})
	}
	if body, err := encodeSection(optmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagOPTM, Body: body, Flags: mll.SectionFlagRequired, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXCHK,
		Body:          metaBody,
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileCheckpoint, mll.V1_0, sections)
}

func decodeEmbeddingCheckpointMLL(data []byte) (EmbeddingTrainCheckpoint, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	if reader.Profile() != mll.ProfileCheckpoint {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint profile = %d, want %d", reader.Profile(), mll.ProfileCheckpoint)
	}
	body, ok := reader.Section(tagXCHK)
	if !ok {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint missing XCHK metadata section")
	}
	var meta checkpointMLLMetadata
	if err := json.Unmarshal(body, &meta); err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	if meta.Version == "" {
		meta.Version = EmbeddingTrainCheckpointVersion
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint missing STRG section")
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	tnsrBody, ok := reader.Section(mll.TagTNSR)
	if !ok {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint missing TNSR section")
	}
	tnsr, err := mll.ReadTnsrSection(tnsrBody)
	if err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	optmBody, ok := reader.Section(mll.TagOPTM)
	if !ok {
		return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint missing OPTM section")
	}
	optm, err := mll.ReadOptmSection(optmBody)
	if err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	tensors := make(map[string]*backend.Tensor, len(tnsr.Tensors))
	indexNames := make(map[uint32]string, len(tnsr.Tensors))
	for i, entry := range tnsr.Tensors {
		name := strg.At(entry.NameIdx)
		if name == "" {
			return EmbeddingTrainCheckpoint{}, fmt.Errorf("checkpoint tensor missing name for index %d", entry.NameIdx)
		}
		tensor, err := decodeTensorEntry(entry, meta.LogicalTensorDType[name], 0)
		if err != nil {
			return EmbeddingTrainCheckpoint{}, fmt.Errorf("decode checkpoint tensor %q: %w", name, err)
		}
		tensors[name] = tensor
		indexNames[uint32(i)] = name
	}
	momentNames := map[string]bool{}
	for _, ref := range optm.MomentTensors {
		if ref.Tag != mll.TagTNSR {
			continue
		}
		if name, ok := indexNames[ref.Index]; ok {
			momentNames[name] = true
		}
	}
	checkpoint := EmbeddingTrainCheckpoint{
		Version:           meta.Version,
		Manifest:          meta.Manifest,
		Config:            meta.Config,
		Step:              int(optm.Step),
		TokenEmbedding:    cloneTensorOrNil(tensors["token_embedding"]),
		RoleEmbedding:     cloneTensorOrNil(tensors["role_embedding"]),
		AttentionQuery:    cloneTensorOrNil(tensors["attention_query"]),
		AttentionKey:      cloneTensorOrNil(tensors["attention_key"]),
		AttentionValue:    cloneTensorOrNil(tensors["attention_value"]),
		AttentionOutput:   cloneTensorOrNil(tensors["attention_output"]),
		HiddenProjection:  cloneTensorOrNil(tensors["hidden_projection"]),
		Projection:        cloneTensorOrNil(tensors["projection"]),
		TokenMoment1:      cloneTensorOrNil(tensors["token_moment_1"]),
		TokenMoment2:      cloneTensorOrNil(tensors["token_moment_2"]),
		RoleMoment1:       cloneTensorOrNil(tensors["role_moment_1"]),
		RoleMoment2:       cloneTensorOrNil(tensors["role_moment_2"]),
		AttentionQMoment1: cloneTensorOrNil(tensors["attention_query_moment_1"]),
		AttentionQMoment2: cloneTensorOrNil(tensors["attention_query_moment_2"]),
		AttentionKMoment1: cloneTensorOrNil(tensors["attention_key_moment_1"]),
		AttentionKMoment2: cloneTensorOrNil(tensors["attention_key_moment_2"]),
		AttentionVMoment1: cloneTensorOrNil(tensors["attention_value_moment_1"]),
		AttentionVMoment2: cloneTensorOrNil(tensors["attention_value_moment_2"]),
		AttentionOMoment1: cloneTensorOrNil(tensors["attention_output_moment_1"]),
		AttentionOMoment2: cloneTensorOrNil(tensors["attention_output_moment_2"]),
		HiddenMoment1:     cloneTensorOrNil(tensors["hidden_projection_moment_1"]),
		HiddenMoment2:     cloneTensorOrNil(tensors["hidden_projection_moment_2"]),
		ProjMoment1:       cloneTensorOrNil(tensors["projection_moment_1"]),
		ProjMoment2:       cloneTensorOrNil(tensors["projection_moment_2"]),
	}
	if checkpoint.Manifest.normalized().ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 {
		checkpoint.TokenEmbedding = nil
		checkpoint.RoleEmbedding = nil
		checkpoint.AttentionQuery = nil
		checkpoint.AttentionKey = nil
		checkpoint.AttentionValue = nil
		checkpoint.AttentionOutput = nil
		checkpoint.HiddenProjection = nil
		checkpoint.Projection = nil
		checkpoint.TokenMoment1 = nil
		checkpoint.TokenMoment2 = nil
		checkpoint.RoleMoment1 = nil
		checkpoint.RoleMoment2 = nil
		checkpoint.AttentionQMoment1 = nil
		checkpoint.AttentionQMoment2 = nil
		checkpoint.AttentionKMoment1 = nil
		checkpoint.AttentionKMoment2 = nil
		checkpoint.AttentionVMoment1 = nil
		checkpoint.AttentionVMoment2 = nil
		checkpoint.AttentionOMoment1 = nil
		checkpoint.AttentionOMoment2 = nil
		checkpoint.HiddenMoment1 = nil
		checkpoint.HiddenMoment2 = nil
		checkpoint.ProjMoment1 = nil
		checkpoint.ProjMoment2 = nil
		checkpoint.Tensors, checkpoint.MomentTensors = genericCheckpointTensorMapsIncludingLegacy(tensors, momentNames)
	} else {
		checkpoint.Tensors, checkpoint.MomentTensors = genericCheckpointTensorMaps(tensors, momentNames)
	}
	if err := checkpoint.Validate(); err != nil {
		return EmbeddingTrainCheckpoint{}, err
	}
	return checkpoint, nil
}

func checkpointTensorMap(c EmbeddingTrainCheckpoint) map[string]*backend.Tensor {
	tensors := map[string]*backend.Tensor{
		"token_embedding":            c.TokenEmbedding,
		"role_embedding":             c.RoleEmbedding,
		"attention_query":            c.AttentionQuery,
		"attention_key":              c.AttentionKey,
		"attention_value":            c.AttentionValue,
		"attention_output":           c.AttentionOutput,
		"hidden_projection":          c.HiddenProjection,
		"projection":                 c.Projection,
		"token_moment_1":             c.TokenMoment1,
		"token_moment_2":             c.TokenMoment2,
		"role_moment_1":              c.RoleMoment1,
		"role_moment_2":              c.RoleMoment2,
		"attention_query_moment_1":   c.AttentionQMoment1,
		"attention_query_moment_2":   c.AttentionQMoment2,
		"attention_key_moment_1":     c.AttentionKMoment1,
		"attention_key_moment_2":     c.AttentionKMoment2,
		"attention_value_moment_1":   c.AttentionVMoment1,
		"attention_value_moment_2":   c.AttentionVMoment2,
		"attention_output_moment_1":  c.AttentionOMoment1,
		"attention_output_moment_2":  c.AttentionOMoment2,
		"hidden_projection_moment_1": c.HiddenMoment1,
		"hidden_projection_moment_2": c.HiddenMoment2,
		"projection_moment_1":        c.ProjMoment1,
		"projection_moment_2":        c.ProjMoment2,
	}
	includeLegacyGeneric := c.Manifest.normalized().ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1
	for name, tensor := range c.Tensors {
		if _, legacy := tensors[name]; legacy && !includeLegacyGeneric {
			continue
		}
		tensors[name] = tensor
	}
	for name, tensor := range c.MomentTensors {
		if _, legacy := tensors[name]; legacy && !includeLegacyGeneric {
			continue
		}
		tensors[name] = tensor
	}
	return tensors
}

func checkpointMomentNames(c EmbeddingTrainCheckpoint) []string {
	names := make([]string, 0, 10)
	for name, tensor := range map[string]*backend.Tensor{
		"token_moment_1":             c.TokenMoment1,
		"token_moment_2":             c.TokenMoment2,
		"role_moment_1":              c.RoleMoment1,
		"role_moment_2":              c.RoleMoment2,
		"attention_query_moment_1":   c.AttentionQMoment1,
		"attention_query_moment_2":   c.AttentionQMoment2,
		"attention_key_moment_1":     c.AttentionKMoment1,
		"attention_key_moment_2":     c.AttentionKMoment2,
		"attention_value_moment_1":   c.AttentionVMoment1,
		"attention_value_moment_2":   c.AttentionVMoment2,
		"attention_output_moment_1":  c.AttentionOMoment1,
		"attention_output_moment_2":  c.AttentionOMoment2,
		"hidden_projection_moment_1": c.HiddenMoment1,
		"hidden_projection_moment_2": c.HiddenMoment2,
		"projection_moment_1":        c.ProjMoment1,
		"projection_moment_2":        c.ProjMoment2,
	} {
		if tensor != nil {
			names = append(names, name)
		}
	}
	for name, tensor := range c.MomentTensors {
		if tensor != nil && !isLegacyCheckpointTensorName(name) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func genericCheckpointTensorMaps(tensors map[string]*backend.Tensor, momentNames map[string]bool) (map[string]*backend.Tensor, map[string]*backend.Tensor) {
	generic := map[string]*backend.Tensor{}
	moments := map[string]*backend.Tensor{}
	for name, tensor := range tensors {
		if tensor == nil || isLegacyCheckpointTensorName(name) {
			continue
		}
		if momentNames[name] {
			moments[name] = tensor.Clone()
		} else {
			generic[name] = tensor.Clone()
		}
	}
	if len(generic) == 0 {
		generic = nil
	}
	if len(moments) == 0 {
		moments = nil
	}
	return generic, moments
}

func genericCheckpointTensorMapsIncludingLegacy(tensors map[string]*backend.Tensor, momentNames map[string]bool) (map[string]*backend.Tensor, map[string]*backend.Tensor) {
	generic := map[string]*backend.Tensor{}
	moments := map[string]*backend.Tensor{}
	for name, tensor := range tensors {
		if tensor == nil {
			continue
		}
		if momentNames[name] {
			moments[name] = tensor.Clone()
		} else {
			generic[name] = tensor.Clone()
		}
	}
	if len(generic) == 0 {
		generic = nil
	}
	if len(moments) == 0 {
		moments = nil
	}
	return generic, moments
}

func isLegacyCheckpointTensorName(name string) bool {
	switch name {
	case "token_embedding",
		"role_embedding",
		"attention_query",
		"attention_key",
		"attention_value",
		"attention_output",
		"hidden_projection",
		"projection",
		"token_moment_1",
		"token_moment_2",
		"role_moment_1",
		"role_moment_2",
		"attention_query_moment_1",
		"attention_query_moment_2",
		"attention_key_moment_1",
		"attention_key_moment_2",
		"attention_value_moment_1",
		"attention_value_moment_2",
		"attention_output_moment_1",
		"attention_output_moment_2",
		"hidden_projection_moment_1",
		"hidden_projection_moment_2",
		"projection_moment_1",
		"projection_moment_2":
		return true
	default:
		return false
	}
}

func optimizerKindToMLL(name string) uint8 {
	switch name {
	case "", "adamw":
		return mll.OptimizerAdamW
	case "sgd":
		return mll.OptimizerSGD
	case "lamb":
		return mll.OptimizerLAMB
	default:
		return mll.OptimizerAdamW
	}
}

func nonEmptyCheckpointName(name string) string {
	if name != "" {
		return name
	}
	return "manta-train-checkpoint"
}

func mustEosDTypeToMLL(dtype string) mll.DType {
	value, err := eosDTypeToMLL(dtype)
	if err != nil {
		return mll.DTypeF32
	}
	return value
}

// Validate checks that the checkpoint is structurally usable.
func (c EmbeddingTrainCheckpoint) Validate() error {
	if c.Version == "" {
		return fmt.Errorf("checkpoint version is required")
	}
	if c.Version != EmbeddingTrainCheckpointVersion {
		return fmt.Errorf("checkpoint version %q is not supported, want %q", c.Version, EmbeddingTrainCheckpointVersion)
	}
	if c.Manifest.normalized().ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 {
		return validateGenericEmbeddingTrainCheckpoint(c)
	}
	if c.TokenEmbedding == nil {
		return fmt.Errorf("checkpoint token_embedding is required")
	}
	if c.Projection == nil {
		return fmt.Errorf("checkpoint projection is required")
	}
	if len(c.TokenEmbedding.Shape) != 2 {
		return fmt.Errorf("checkpoint token_embedding rank = %d, want 2", len(c.TokenEmbedding.Shape))
	}
	if c.Manifest.normalized().roleConditioned() {
		if c.RoleEmbedding == nil {
			return fmt.Errorf("checkpoint role_embedding is required for role-conditioned manifest")
		}
		if len(c.RoleEmbedding.Shape) != 2 {
			return fmt.Errorf("checkpoint role_embedding rank = %d, want 2", len(c.RoleEmbedding.Shape))
		}
		if c.RoleEmbedding.Shape[1] != c.TokenEmbedding.Shape[1] {
			return fmt.Errorf("checkpoint role/token embedding width mismatch %v x %v", c.RoleEmbedding.Shape, c.TokenEmbedding.Shape)
		}
	} else if c.RoleEmbedding != nil {
		return fmt.Errorf("checkpoint role_embedding requires role-conditioned manifest")
	}
	if len(c.Projection.Shape) != 2 {
		return fmt.Errorf("checkpoint projection rank = %d, want 2", len(c.Projection.Shape))
	}
	if c.HiddenProjection != nil {
		if c.Manifest.HiddenProjectionParam == "" {
			return fmt.Errorf("checkpoint hidden_projection requires manifest hidden_projection_param")
		}
		if len(c.HiddenProjection.Shape) != 2 {
			return fmt.Errorf("checkpoint hidden_projection rank = %d, want 2", len(c.HiddenProjection.Shape))
		}
		if c.TokenEmbedding.Shape[1] != c.HiddenProjection.Shape[0] {
			return fmt.Errorf("checkpoint embedding/hidden projection mismatch %v x %v", c.TokenEmbedding.Shape, c.HiddenProjection.Shape)
		}
		if c.HiddenProjection.Shape[1] != c.Projection.Shape[0] {
			return fmt.Errorf("checkpoint hidden/output projection mismatch %v x %v", c.HiddenProjection.Shape, c.Projection.Shape)
		}
	} else if c.TokenEmbedding.Shape[1] != c.Projection.Shape[0] {
		return fmt.Errorf("checkpoint embedding/projection mismatch %v x %v", c.TokenEmbedding.Shape, c.Projection.Shape)
	}
	if err := validateCheckpointAttention(c); err != nil {
		return err
	}
	if c.TokenMoment1 != nil && !sameTensorShape(c.TokenEmbedding, c.TokenMoment1) {
		return fmt.Errorf("checkpoint token_moment_1 shape %v does not match token_embedding %v", c.TokenMoment1.Shape, c.TokenEmbedding.Shape)
	}
	if c.TokenMoment2 != nil && !sameTensorShape(c.TokenEmbedding, c.TokenMoment2) {
		return fmt.Errorf("checkpoint token_moment_2 shape %v does not match token_embedding %v", c.TokenMoment2.Shape, c.TokenEmbedding.Shape)
	}
	if c.RoleMoment1 != nil && !sameTensorShape(c.RoleEmbedding, c.RoleMoment1) {
		return fmt.Errorf("checkpoint role_moment_1 shape %v does not match role_embedding %v", c.RoleMoment1.Shape, c.RoleEmbedding.Shape)
	}
	if c.RoleMoment2 != nil && !sameTensorShape(c.RoleEmbedding, c.RoleMoment2) {
		return fmt.Errorf("checkpoint role_moment_2 shape %v does not match role_embedding %v", c.RoleMoment2.Shape, c.RoleEmbedding.Shape)
	}
	if c.AttentionQMoment1 != nil && !sameTensorShape(c.AttentionQuery, c.AttentionQMoment1) {
		return fmt.Errorf("checkpoint attention_query_moment_1 shape %v does not match attention_query %v", c.AttentionQMoment1.Shape, c.AttentionQuery.Shape)
	}
	if c.AttentionQMoment2 != nil && !sameTensorShape(c.AttentionQuery, c.AttentionQMoment2) {
		return fmt.Errorf("checkpoint attention_query_moment_2 shape %v does not match attention_query %v", c.AttentionQMoment2.Shape, c.AttentionQuery.Shape)
	}
	if c.AttentionKMoment1 != nil && !sameTensorShape(c.AttentionKey, c.AttentionKMoment1) {
		return fmt.Errorf("checkpoint attention_key_moment_1 shape %v does not match attention_key %v", c.AttentionKMoment1.Shape, c.AttentionKey.Shape)
	}
	if c.AttentionKMoment2 != nil && !sameTensorShape(c.AttentionKey, c.AttentionKMoment2) {
		return fmt.Errorf("checkpoint attention_key_moment_2 shape %v does not match attention_key %v", c.AttentionKMoment2.Shape, c.AttentionKey.Shape)
	}
	if c.AttentionVMoment1 != nil && !sameTensorShape(c.AttentionValue, c.AttentionVMoment1) {
		return fmt.Errorf("checkpoint attention_value_moment_1 shape %v does not match attention_value %v", c.AttentionVMoment1.Shape, c.AttentionValue.Shape)
	}
	if c.AttentionVMoment2 != nil && !sameTensorShape(c.AttentionValue, c.AttentionVMoment2) {
		return fmt.Errorf("checkpoint attention_value_moment_2 shape %v does not match attention_value %v", c.AttentionVMoment2.Shape, c.AttentionValue.Shape)
	}
	if c.AttentionOMoment1 != nil && !sameTensorShape(c.AttentionOutput, c.AttentionOMoment1) {
		return fmt.Errorf("checkpoint attention_output_moment_1 shape %v does not match attention_output %v", c.AttentionOMoment1.Shape, c.AttentionOutput.Shape)
	}
	if c.AttentionOMoment2 != nil && !sameTensorShape(c.AttentionOutput, c.AttentionOMoment2) {
		return fmt.Errorf("checkpoint attention_output_moment_2 shape %v does not match attention_output %v", c.AttentionOMoment2.Shape, c.AttentionOutput.Shape)
	}
	if c.HiddenMoment1 != nil && !sameTensorShape(c.HiddenProjection, c.HiddenMoment1) {
		return fmt.Errorf("checkpoint hidden_projection_moment_1 shape %v does not match hidden_projection %v", c.HiddenMoment1.Shape, c.HiddenProjection.Shape)
	}
	if c.HiddenMoment2 != nil && !sameTensorShape(c.HiddenProjection, c.HiddenMoment2) {
		return fmt.Errorf("checkpoint hidden_projection_moment_2 shape %v does not match hidden_projection %v", c.HiddenMoment2.Shape, c.HiddenProjection.Shape)
	}
	if c.ProjMoment1 != nil && !sameTensorShape(c.Projection, c.ProjMoment1) {
		return fmt.Errorf("checkpoint projection_moment_1 shape %v does not match projection %v", c.ProjMoment1.Shape, c.Projection.Shape)
	}
	if c.ProjMoment2 != nil && !sameTensorShape(c.Projection, c.ProjMoment2) {
		return fmt.Errorf("checkpoint projection_moment_2 shape %v does not match projection %v", c.ProjMoment2.Shape, c.Projection.Shape)
	}
	return nil
}

func validateGenericEmbeddingTrainCheckpoint(c EmbeddingTrainCheckpoint) error {
	manifest := c.Manifest.normalized()
	if manifest.ArchitectureVersion != EmbeddingArchitectureCompactTransformerV1 {
		return fmt.Errorf("generic checkpoint validation requires architecture_version=%q", EmbeddingArchitectureCompactTransformerV1)
	}
	if len(c.Tensors) == 0 {
		return fmt.Errorf("checkpoint generic tensors are required for architecture_version=%q", manifest.ArchitectureVersion)
	}
	if c.TokenEmbedding != nil || c.Projection != nil {
		return fmt.Errorf("checkpoint for architecture_version=%q must use generic tensors, not legacy token/projection fields", manifest.ArchitectureVersion)
	}
	for name, tensor := range c.Tensors {
		if err := validateCheckpointGenericTensor("tensor", name, tensor); err != nil {
			return err
		}
	}
	for name, tensor := range c.MomentTensors {
		if err := validateCheckpointGenericTensor("moment tensor", name, tensor); err != nil {
			return err
		}
		baseName, ok := strings.CutSuffix(name, "_moment_1")
		if !ok {
			baseName, ok = strings.CutSuffix(name, "_moment_2")
		}
		if ok {
			base := c.Tensors[baseName]
			if base == nil {
				return fmt.Errorf("checkpoint moment tensor %q has no base tensor %q", name, baseName)
			}
			if !sameTensorShape(base, tensor) {
				return fmt.Errorf("checkpoint moment tensor %q shape %v does not match base tensor %q shape %v", name, tensor.Shape, baseName, base.Shape)
			}
		}
	}
	return nil
}

func validateCheckpointGenericTensor(kind, name string, tensor *backend.Tensor) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("checkpoint generic %s name is required", kind)
	}
	if tensor == nil {
		return fmt.Errorf("checkpoint generic %s %q is nil", kind, name)
	}
	if tensor.DType == "" {
		return fmt.Errorf("checkpoint generic %s %q dtype is required", kind, name)
	}
	if len(tensor.Shape) == 0 {
		return fmt.Errorf("checkpoint generic %s %q shape is required", kind, name)
	}
	if len(tensor.F32) != tensor.Elements() {
		return fmt.Errorf("checkpoint generic %s %q shape %v has %d f32 values, want %d", kind, name, tensor.Shape, len(tensor.F32), tensor.Elements())
	}
	return nil
}

// NewEmbeddingTrainerFromCheckpoint restores a trainer from a checkpoint and module contract.
func NewEmbeddingTrainerFromCheckpoint(mod *eosartifact.Module, checkpoint EmbeddingTrainCheckpoint) (*EmbeddingTrainer, error) {
	if err := checkpoint.Validate(); err != nil {
		return nil, err
	}
	checkpoint.Manifest = checkpoint.Manifest.normalized()
	if checkpoint.Manifest.ArchitectureVersion == EmbeddingArchitectureCompactTransformerV1 {
		state, err := LoadCompactEmbeddingTrainStateFromCheckpoint(checkpoint, checkpoint.Manifest)
		if err != nil {
			return nil, err
		}
		return newCompactEmbeddingTrainerFromTrainState(mod, state)
	}
	weights := map[string]*backend.Tensor{
		checkpoint.Manifest.TokenEmbeddingParam: checkpoint.TokenEmbedding,
		checkpoint.Manifest.ProjectionParam:     checkpoint.Projection,
	}
	if checkpoint.Manifest.roleConditioned() {
		weights[checkpoint.Manifest.RoleEmbeddingParam] = checkpoint.RoleEmbedding
	}
	if checkpoint.Manifest.AttentionQueryParam != "" {
		weights[checkpoint.Manifest.AttentionQueryParam] = checkpoint.AttentionQuery
		weights[checkpoint.Manifest.AttentionKeyParam] = checkpoint.AttentionKey
		weights[checkpoint.Manifest.AttentionValueParam] = checkpoint.AttentionValue
		weights[checkpoint.Manifest.AttentionOutputParam] = checkpoint.AttentionOutput
	}
	if checkpoint.Manifest.HiddenProjectionParam != "" {
		weights[checkpoint.Manifest.HiddenProjectionParam] = checkpoint.HiddenProjection
	}
	trainer, err := NewEmbeddingTrainer(mod, checkpoint.Manifest, weights, checkpoint.Config)
	if err != nil {
		return nil, err
	}
	trainer.step = checkpoint.Step
	if checkpoint.TokenMoment1 != nil {
		trainer.tokenMom1 = checkpoint.TokenMoment1.Clone()
	}
	if checkpoint.TokenMoment2 != nil {
		trainer.tokenMom2 = checkpoint.TokenMoment2.Clone()
	}
	if checkpoint.RoleMoment1 != nil {
		trainer.roleMom1 = checkpoint.RoleMoment1.Clone()
	}
	if checkpoint.RoleMoment2 != nil {
		trainer.roleMom2 = checkpoint.RoleMoment2.Clone()
	}
	if checkpoint.AttentionQMoment1 != nil {
		trainer.attnQMom1 = checkpoint.AttentionQMoment1.Clone()
	}
	if checkpoint.AttentionQMoment2 != nil {
		trainer.attnQMom2 = checkpoint.AttentionQMoment2.Clone()
	}
	if checkpoint.AttentionKMoment1 != nil {
		trainer.attnKMom1 = checkpoint.AttentionKMoment1.Clone()
	}
	if checkpoint.AttentionKMoment2 != nil {
		trainer.attnKMom2 = checkpoint.AttentionKMoment2.Clone()
	}
	if checkpoint.AttentionVMoment1 != nil {
		trainer.attnVMom1 = checkpoint.AttentionVMoment1.Clone()
	}
	if checkpoint.AttentionVMoment2 != nil {
		trainer.attnVMom2 = checkpoint.AttentionVMoment2.Clone()
	}
	if checkpoint.AttentionOMoment1 != nil {
		trainer.attnOMom1 = checkpoint.AttentionOMoment1.Clone()
	}
	if checkpoint.AttentionOMoment2 != nil {
		trainer.attnOMom2 = checkpoint.AttentionOMoment2.Clone()
	}
	if checkpoint.HiddenMoment1 != nil {
		trainer.hiddenMom1 = checkpoint.HiddenMoment1.Clone()
	}
	if checkpoint.HiddenMoment2 != nil {
		trainer.hiddenMom2 = checkpoint.HiddenMoment2.Clone()
	}
	if checkpoint.ProjMoment1 != nil {
		trainer.projMom1 = checkpoint.ProjMoment1.Clone()
	}
	if checkpoint.ProjMoment2 != nil {
		trainer.projMom2 = checkpoint.ProjMoment2.Clone()
	}
	return trainer, nil
}

func compactTrainStateCheckpointMaps(state *CompactEmbeddingTrainState) (map[string]*backend.Tensor, map[string]*backend.Tensor) {
	if state == nil {
		return nil, nil
	}
	tensors := map[string]*backend.Tensor{}
	moments := map[string]*backend.Tensor{}
	add := func(item CompactEmbeddingTrainTensor) {
		if item.Name == "" || item.Tensor == nil {
			return
		}
		tensors[item.Name] = item.Tensor.Clone()
		if item.Moment1 != nil {
			moments[item.Name+"_moment_1"] = item.Moment1.Clone()
		}
		if item.Moment2 != nil {
			moments[item.Name+"_moment_2"] = item.Moment2.Clone()
		}
	}
	add(state.TokenEmbedding)
	if state.RoleEmbedding != nil {
		add(*state.RoleEmbedding)
	}
	for _, layer := range state.Layers {
		add(layer.AttentionQuery)
		add(layer.AttentionKey)
		add(layer.AttentionValue)
		add(layer.AttentionOutput)
		add(layer.FFNUp)
		add(layer.FFNDown)
	}
	if state.OutputProjection != nil {
		add(*state.OutputProjection)
	}
	if len(tensors) == 0 {
		tensors = nil
	}
	if len(moments) == 0 {
		moments = nil
	}
	return tensors, moments
}

func cloneTensorOrNil(t *backend.Tensor) *backend.Tensor {
	if t == nil {
		return nil
	}
	return t.Clone()
}

func sameTensorShape(lhs, rhs *backend.Tensor) bool {
	if lhs == nil || rhs == nil {
		return false
	}
	if len(lhs.Shape) != len(rhs.Shape) {
		return false
	}
	for i := range lhs.Shape {
		if lhs.Shape[i] != rhs.Shape[i] {
			return false
		}
	}
	return true
}

func validateCheckpointAttention(c EmbeddingTrainCheckpoint) error {
	names := []string{
		c.Manifest.AttentionQueryParam,
		c.Manifest.AttentionKeyParam,
		c.Manifest.AttentionValueParam,
		c.Manifest.AttentionOutputParam,
	}
	tensors := []*backend.Tensor{c.AttentionQuery, c.AttentionKey, c.AttentionValue, c.AttentionOutput}
	set := 0
	for i, name := range names {
		if name != "" {
			set++
		}
		if name == "" && tensors[i] != nil {
			return fmt.Errorf("checkpoint %s requires corresponding manifest attention param", []string{"attention_query", "attention_key", "attention_value", "attention_output"}[i])
		}
	}
	if set == 0 {
		for _, tensor := range tensors {
			if tensor != nil {
				return fmt.Errorf("checkpoint attention tensors require manifest attention params")
			}
		}
		return nil
	}
	if set != len(names) {
		return fmt.Errorf("checkpoint attention params must declare query, key, value, and output together")
	}
	for i, tensor := range tensors {
		if tensor == nil {
			return fmt.Errorf("checkpoint missing %s tensor", []string{"attention_query", "attention_key", "attention_value", "attention_output"}[i])
		}
		if len(tensor.Shape) != 2 {
			return fmt.Errorf("checkpoint %s rank = %d, want 2", []string{"attention_query", "attention_key", "attention_value", "attention_output"}[i], len(tensor.Shape))
		}
	}
	d := c.TokenEmbedding.Shape[1]
	if c.AttentionQuery.Shape[0] != d || c.AttentionQuery.Shape[1] != d {
		return fmt.Errorf("checkpoint attention_query shape %v does not match embedding width %d", c.AttentionQuery.Shape, d)
	}
	if c.AttentionKey.Shape[0] != d || c.AttentionKey.Shape[1] != d {
		return fmt.Errorf("checkpoint attention_key shape %v does not match embedding width %d", c.AttentionKey.Shape, d)
	}
	if c.AttentionValue.Shape[0] != d || c.AttentionValue.Shape[1] != d {
		return fmt.Errorf("checkpoint attention_value shape %v does not match embedding width %d", c.AttentionValue.Shape, d)
	}
	if c.AttentionOutput.Shape[0] != d || c.AttentionOutput.Shape[1] != d {
		return fmt.Errorf("checkpoint attention_output shape %v does not match embedding width %d", c.AttentionOutput.Shape, d)
	}
	return nil
}

func requireTrainableParamByName(mod *eosartifact.Module, name string) (eosartifact.Param, error) {
	if mod == nil {
		return eosartifact.Param{}, fmt.Errorf("nil module")
	}
	for _, param := range mod.Params {
		if param.Name == name {
			return param, nil
		}
	}
	return eosartifact.Param{}, fmt.Errorf("missing param %q", name)
}
