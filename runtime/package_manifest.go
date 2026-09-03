package eosruntime

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
	mll "m31labs.dev/mll"
)

const PackageManifestVersion = "manta/package/v0alpha1"

var tagXPKG = [4]byte{'X', 'P', 'K', 'G'}

type PackageKind string

const (
	PackageEmbedding PackageKind = "embedding"
	PackageTraining  PackageKind = "training"
)

type PackageManifestFile struct {
	Role   string `json:"role"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

type PackageManifest struct {
	Version          string                          `json:"version"`
	Kind             PackageKind                     `json:"kind"`
	ModuleName       string                          `json:"module_name"`
	ArtifactVersion  string                          `json:"artifact_version"`
	Files            []PackageManifestFile           `json:"files"`
	ScoreSpectrum    EmbeddingScoreSpectrumPolicy    `json:"score_spectrum,omitempty"`
	ListwiseGeometry EmbeddingListwiseGeometryPolicy `json:"listwise_geometry,omitempty"`
	AOQTTransform    AOQTTransformPolicy             `json:"aoqt_transform,omitempty"`
}

const AOQTTransformPolicySchema = "eos.aoqt_transform_policy.v1"

// AOQTTransformPolicy records package-level AOQT transform provenance and usage gates.
type AOQTTransformPolicy struct {
	Schema                      string            `json:"schema,omitempty"`
	Enabled                     bool              `json:"enabled,omitempty"`
	ResearchOnly                bool              `json:"research_only,omitempty"`
	ResearchTrainAllowed        bool              `json:"research_train_allowed,omitempty"`
	ReleaseTrainAllowed         bool              `json:"release_train_allowed,omitempty"`
	CommercialUseAllowed        bool              `json:"commercial_use_allowed,omitempty"`
	FreeOpenReleaseAllowed      bool              `json:"free_open_release_allowed,omitempty"`
	QualityClaim                bool              `json:"quality_claim,omitempty"`
	AnchorArtifactSHA256        string            `json:"anchor_artifact_sha256,omitempty"`
	AnchorPackageManifestSHA256 string            `json:"anchor_package_manifest_sha256,omitempty"`
	AnchorEmbeddingSpaceID      string            `json:"anchor_embedding_space_id,omitempty"`
	DatasetManifestSHA256       string            `json:"dataset_manifest_sha256,omitempty"`
	QrelsSHA256ByDataset        map[string]string `json:"qrels_sha256_by_dataset,omitempty"`
	CompatibilityDigest         string            `json:"compatibility_digest,omitempty"`
	TransformSHA256             string            `json:"transform_sha256,omitempty"`
	PairingsSHA256              string            `json:"pairings_sha256,omitempty"`
	AnglesSHA256                string            `json:"angles_sha256,omitempty"`
}

func DefaultPackageManifestPath(artifactPath string) string {
	return defaultManifestPath(artifactPath, ".package.mll")
}

func ResolvePackageManifestPath(artifactPath string) string {
	return DefaultPackageManifestPath(artifactPath)
}

func (m PackageManifest) Validate() error {
	if m.Version == "" {
		return fmt.Errorf("package manifest version is required")
	}
	if m.Version != PackageManifestVersion {
		return fmt.Errorf("package manifest version %q is not supported, want %q", m.Version, PackageManifestVersion)
	}
	if m.Kind == "" {
		return fmt.Errorf("package manifest kind is required")
	}
	if m.ModuleName == "" {
		return fmt.Errorf("package manifest module_name is required")
	}
	if m.ArtifactVersion == "" {
		return fmt.Errorf("package manifest artifact_version is required")
	}
	if len(m.Files) == 0 {
		return fmt.Errorf("package manifest must list at least one file")
	}
	seen := map[string]bool{}
	for _, item := range m.Files {
		if item.Role == "" {
			return fmt.Errorf("package manifest file role is required")
		}
		if seen[item.Role] {
			return fmt.Errorf("duplicate package manifest file role %q", item.Role)
		}
		seen[item.Role] = true
		if item.Path == "" {
			return fmt.Errorf("package manifest file %q path is required", item.Role)
		}
		if item.SHA256 == "" {
			return fmt.Errorf("package manifest file %q sha256 is required", item.Role)
		}
		if len(item.SHA256) != sha256.Size*2 {
			return fmt.Errorf("package manifest file %q sha256 must be %d hex chars", item.Role, sha256.Size*2)
		}
		if _, err := hex.DecodeString(item.SHA256); err != nil {
			return fmt.Errorf("package manifest file %q sha256 is invalid: %w", item.Role, err)
		}
		if item.Bytes < 0 {
			return fmt.Errorf("package manifest file %q size must be non-negative", item.Role)
		}
	}
	if err := m.validateAOQTTransformRoleBinding(seen[EmbeddingPostPoolTransformRole]); err != nil {
		return err
	}
	return nil
}

func (m PackageManifest) WriteFile(path string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	data, err := encodePackageManifestMLL(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ReadPackageManifestFile(path string) (PackageManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PackageManifest{}, err
	}
	if !eosartifact.IsMLLBytes(data) {
		return PackageManifest{}, fmt.Errorf("package manifest %q is not an MLL file", path)
	}
	return decodePackageManifestMLL(data)
}

func encodePackageManifestMLL(manifest PackageManifest) ([]byte, error) {
	strg := mll.NewStringTableBuilder()
	strg.Intern("")
	for _, item := range manifest.Files {
		strg.Intern(item.Role)
		strg.Intern(item.Path)
	}

	head := mll.HeadSection{
		Name:        strg.Intern("manta-package-manifest"),
		Description: strg.Intern("Eos package manifest"),
		Metadata: []mll.HeadMetadataEntry{
			headStringMeta(strg, manifestVersionKey, manifest.Version),
			headStringMeta(strg, "package_kind", string(manifest.Kind)),
			headStringMeta(strg, "module_name", manifest.ModuleName),
			headStringMeta(strg, "artifact_version", manifest.ArtifactVersion),
			headIntMeta(strg, "file_count", int64(len(manifest.Files))),
			headBoolMeta(strg, "score_spectrum_research_only", manifest.ScoreSpectrum.ScoreSpectrumResearchOnly),
			headBoolMeta(strg, "train_allowed_for_research", manifest.ScoreSpectrum.TrainAllowedForResearch),
			headBoolMeta(strg, "release_train_allowed", manifest.ScoreSpectrum.ReleaseTrainAllowed),
			headBoolMeta(strg, "commercial_use_allowed", manifest.ScoreSpectrum.CommercialUseAllowed),
			headStringMeta(strg, "source_artifact_hashes", formatScoreSpectrumSourceHashes(manifest.ScoreSpectrum.SourceArtifactHashes)),
			headIntMeta(strg, "score_spectrum_row_count", int64(manifest.ScoreSpectrum.ScoreSpectrumRowCount)),
			headStringMeta(strg, "auto_cleared_objectives", formatScoreSpectrumObjectiveNames(manifest.ScoreSpectrum.AutoClearedObjectives)),
			headStringMeta(strg, "isolated_inherited_objectives", formatScoreSpectrumObjectiveNames(manifest.ScoreSpectrum.IsolatedInheritedObjectives)),
			headBoolMeta(strg, "listwise_geometry_research_only", manifest.ListwiseGeometry.ListwiseGeometryResearchOnly),
			headBoolMeta(strg, "listwise_geometry_train_allowed_for_research", manifest.ListwiseGeometry.TrainAllowedForResearch),
			headBoolMeta(strg, "listwise_geometry_release_train_allowed", manifest.ListwiseGeometry.ReleaseTrainAllowed),
			headBoolMeta(strg, "listwise_geometry_commercial_use_allowed", manifest.ListwiseGeometry.CommercialUseAllowed),
			headStringMeta(strg, "listwise_geometry_source_artifact_hashes", formatScoreSpectrumSourceHashes(manifest.ListwiseGeometry.SourceArtifactHashes)),
			headIntMeta(strg, "listwise_geometry_batch_count", int64(manifest.ListwiseGeometry.ListwiseGeometryBatchCount)),
			headStringMeta(strg, "listwise_geometry_auto_cleared_objectives", formatScoreSpectrumObjectiveNames(manifest.ListwiseGeometry.AutoClearedObjectives)),
			headStringMeta(strg, "listwise_geometry_isolated_inherited_objectives", formatScoreSpectrumObjectiveNames(manifest.ListwiseGeometry.IsolatedInheritedObjectives)),
			headStringMeta(strg, "aoqt_transform_schema", manifest.AOQTTransform.Schema),
			headBoolMeta(strg, "aoqt_transform_enabled", manifest.AOQTTransform.Enabled),
			headBoolMeta(strg, "aoqt_transform_research_only", manifest.AOQTTransform.ResearchOnly),
			headBoolMeta(strg, "aoqt_transform_research_train_allowed", manifest.AOQTTransform.ResearchTrainAllowed),
			headBoolMeta(strg, "aoqt_transform_release_train_allowed", manifest.AOQTTransform.ReleaseTrainAllowed),
			headBoolMeta(strg, "aoqt_transform_commercial_use_allowed", manifest.AOQTTransform.CommercialUseAllowed),
			headBoolMeta(strg, "aoqt_transform_free_open_release_allowed", manifest.AOQTTransform.FreeOpenReleaseAllowed),
			headBoolMeta(strg, "aoqt_transform_quality_claim", manifest.AOQTTransform.QualityClaim),
			headStringMeta(strg, "aoqt_transform_anchor_artifact_sha256", manifest.AOQTTransform.AnchorArtifactSHA256),
			headStringMeta(strg, "aoqt_transform_anchor_package_manifest_sha256", manifest.AOQTTransform.AnchorPackageManifestSHA256),
			headStringMeta(strg, "aoqt_transform_anchor_embedding_space_id", manifest.AOQTTransform.AnchorEmbeddingSpaceID),
			headStringMeta(strg, "aoqt_transform_dataset_manifest_sha256", manifest.AOQTTransform.DatasetManifestSHA256),
			headStringMeta(strg, "aoqt_transform_qrels_sha256_by_dataset", formatAOQTQrelsHashes(manifest.AOQTTransform.QrelsSHA256ByDataset)),
			headStringMeta(strg, "aoqt_transform_compatibility_digest", manifest.AOQTTransform.CompatibilityDigest),
			headStringMeta(strg, "aoqt_transform_transform_sha256", manifest.AOQTTransform.TransformSHA256),
			headStringMeta(strg, "aoqt_transform_pairings_sha256", manifest.AOQTTransform.PairingsSHA256),
			headStringMeta(strg, "aoqt_transform_angles_sha256", manifest.AOQTTransform.AnglesSHA256),
		},
	}

	var xbody bytes.Buffer
	writeU32 := func(v uint32) error {
		return binary.Write(&xbody, binary.LittleEndian, v)
	}
	writeI64 := func(v int64) error {
		return binary.Write(&xbody, binary.LittleEndian, v)
	}
	if err := writeU32(uint32(len(manifest.Files))); err != nil {
		return nil, err
	}
	for _, item := range manifest.Files {
		sum, err := hex.DecodeString(item.SHA256)
		if err != nil {
			return nil, fmt.Errorf("decode sha256 for %q: %w", item.Role, err)
		}
		if len(sum) != sha256.Size {
			return nil, fmt.Errorf("package manifest file %q sha256 bytes = %d, want %d", item.Role, len(sum), sha256.Size)
		}
		if err := writeU32(strg.Intern(item.Role)); err != nil {
			return nil, err
		}
		if err := writeU32(strg.Intern(item.Path)); err != nil {
			return nil, err
		}
		if err := writeI64(item.Bytes); err != nil {
			return nil, err
		}
		if _, err := xbody.Write(sum); err != nil {
			return nil, err
		}
	}

	var (
		dimsBuilder mll.DimsBuilder
		typeBuilder mll.TypeBuilder
		parmBuilder mll.ParmBuilder
		entrBuilder mll.EntrBuilder
		tnsrBuilder mll.TnsrBuilder
	)

	sections := make([]mll.SectionInput, 0, 9)
	if body, digestBody, err := encodeHeadSection(head, mll.ProfileSealed); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagHEAD,
			Body:          body,
			DigestBody:    digestBody,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(strg.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagSTRG,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(dimsBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagDIMS,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(typeBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagTYPE, Body: body, SchemaVersion: 1})
	}
	if body, err := encodeSection(parmBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagPARM,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(entrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagENTR,
			Body:          body,
			Flags:         mll.SectionFlagRequired,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection(tnsrBuilder.Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{
			Tag:           mll.TagTNSR,
			Body:          body,
			Flags:         mll.SectionFlagRequired | mll.SectionFlagAligned,
			SchemaVersion: 1,
		})
	}
	if body, err := encodeSection((mll.SchmSection{}).Write); err != nil {
		return nil, err
	} else {
		sections = append(sections, mll.SectionInput{Tag: mll.TagSCHM, Body: body, SchemaVersion: 1})
	}
	sections = append(sections, mll.SectionInput{
		Tag:           tagXPKG,
		Body:          xbody.Bytes(),
		Flags:         mll.SectionFlagSkippable | mll.SectionFlagSchemaless,
		SchemaVersion: 1,
	})
	return mll.WriteToBytes(mll.ProfileSealed, mll.V1_0, sections)
}

func decodePackageManifestMLL(data []byte) (PackageManifest, error) {
	reader, err := mll.ReadBytes(data, mll.WithDigestVerification())
	if err != nil {
		return PackageManifest{}, err
	}
	if reader.Profile() != mll.ProfileSealed {
		return PackageManifest{}, fmt.Errorf("package manifest profile = %d, want %d", reader.Profile(), mll.ProfileSealed)
	}
	headBody, ok := reader.Section(mll.TagHEAD)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing HEAD section")
	}
	strgBody, ok := reader.Section(mll.TagSTRG)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing STRG section")
	}
	xpkgBody, ok := reader.Section(tagXPKG)
	if !ok {
		return PackageManifest{}, fmt.Errorf("package manifest missing XPKG section")
	}
	head, err := mll.ReadHeadSection(headBody)
	if err != nil {
		return PackageManifest{}, err
	}
	strg, err := mll.ReadStringTable(strgBody)
	if err != nil {
		return PackageManifest{}, err
	}
	meta := make(map[string]mll.HeadMetadataEntry, len(head.Metadata))
	for _, entry := range head.Metadata {
		meta[strg.At(entry.Key)] = entry
	}
	readString := func(key string) (string, error) {
		entry, ok := meta[key]
		if !ok {
			return "", fmt.Errorf("package manifest missing %q", key)
		}
		if entry.Kind != mll.HeadValueString {
			return "", fmt.Errorf("package manifest key %q kind = %d, want string", key, entry.Kind)
		}
		return strg.At(entry.StringIdx), nil
	}
	readInt := func(key string) (int64, error) {
		entry, ok := meta[key]
		if !ok {
			return 0, fmt.Errorf("package manifest missing %q", key)
		}
		if entry.Kind != mll.HeadValueI64 {
			return 0, fmt.Errorf("package manifest key %q kind = %d, want int", key, entry.Kind)
		}
		return entry.I64, nil
	}
	readOptionalBool := func(key string) (bool, error) {
		entry, ok := meta[key]
		if !ok {
			return false, nil
		}
		if entry.Kind != mll.HeadValueBool {
			return false, fmt.Errorf("package manifest key %q kind = %d, want bool", key, entry.Kind)
		}
		return entry.Bool, nil
	}
	readOptionalString := func(key string) (string, error) {
		entry, ok := meta[key]
		if !ok {
			return "", nil
		}
		if entry.Kind != mll.HeadValueString {
			return "", fmt.Errorf("package manifest key %q kind = %d, want string", key, entry.Kind)
		}
		return strg.At(entry.StringIdx), nil
	}
	readOptionalInt := func(key string) (int64, error) {
		entry, ok := meta[key]
		if !ok {
			return 0, nil
		}
		if entry.Kind != mll.HeadValueI64 {
			return 0, fmt.Errorf("package manifest key %q kind = %d, want int", key, entry.Kind)
		}
		return entry.I64, nil
	}

	version, err := readString(manifestVersionKey)
	if err != nil {
		return PackageManifest{}, err
	}
	kind, err := readString("package_kind")
	if err != nil {
		return PackageManifest{}, err
	}
	moduleName, err := readString("module_name")
	if err != nil {
		return PackageManifest{}, err
	}
	artifactVersion, err := readString("artifact_version")
	if err != nil {
		return PackageManifest{}, err
	}
	expectedFiles, err := readInt("file_count")
	if err != nil {
		return PackageManifest{}, err
	}
	scorePolicy := EmbeddingScoreSpectrumPolicy{}
	if scorePolicy.ScoreSpectrumResearchOnly, err = readOptionalBool("score_spectrum_research_only"); err != nil {
		return PackageManifest{}, err
	}
	if scorePolicy.TrainAllowedForResearch, err = readOptionalBool("train_allowed_for_research"); err != nil {
		return PackageManifest{}, err
	}
	if scorePolicy.ReleaseTrainAllowed, err = readOptionalBool("release_train_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if scorePolicy.CommercialUseAllowed, err = readOptionalBool("commercial_use_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if value, err := readOptionalString("source_artifact_hashes"); err != nil {
		return PackageManifest{}, err
	} else {
		scorePolicy.SourceArtifactHashes = parseScoreSpectrumSourceHashes(value)
	}
	if value, err := readOptionalInt("score_spectrum_row_count"); err != nil {
		return PackageManifest{}, err
	} else {
		scorePolicy.ScoreSpectrumRowCount = int(value)
	}
	if value, err := readOptionalString("auto_cleared_objectives"); err != nil {
		return PackageManifest{}, err
	} else {
		scorePolicy.AutoClearedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	if value, err := readOptionalString("isolated_inherited_objectives"); err != nil {
		return PackageManifest{}, err
	} else {
		scorePolicy.IsolatedInheritedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	scorePolicy.ScoreSpectrumTrain = scorePolicy.ScoreSpectrumResearchOnly || scorePolicy.TrainAllowedForResearch || scorePolicy.ReleaseTrainAllowed || scorePolicy.CommercialUseAllowed || scorePolicy.ScoreSpectrumRowCount > 0 || len(scorePolicy.SourceArtifactHashes) > 0 || len(scorePolicy.AutoClearedObjectives) > 0 || len(scorePolicy.IsolatedInheritedObjectives) > 0
	listwisePolicy := EmbeddingListwiseGeometryPolicy{}
	if listwisePolicy.ListwiseGeometryResearchOnly, err = readOptionalBool("listwise_geometry_research_only"); err != nil {
		return PackageManifest{}, err
	}
	if listwisePolicy.TrainAllowedForResearch, err = readOptionalBool("listwise_geometry_train_allowed_for_research"); err != nil {
		return PackageManifest{}, err
	}
	if listwisePolicy.ReleaseTrainAllowed, err = readOptionalBool("listwise_geometry_release_train_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if listwisePolicy.CommercialUseAllowed, err = readOptionalBool("listwise_geometry_commercial_use_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if value, err := readOptionalString("listwise_geometry_source_artifact_hashes"); err != nil {
		return PackageManifest{}, err
	} else {
		listwisePolicy.SourceArtifactHashes = parseScoreSpectrumSourceHashes(value)
	}
	if value, err := readOptionalInt("listwise_geometry_batch_count"); err != nil {
		return PackageManifest{}, err
	} else {
		listwisePolicy.ListwiseGeometryBatchCount = int(value)
	}
	if value, err := readOptionalString("listwise_geometry_auto_cleared_objectives"); err != nil {
		return PackageManifest{}, err
	} else {
		listwisePolicy.AutoClearedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	if value, err := readOptionalString("listwise_geometry_isolated_inherited_objectives"); err != nil {
		return PackageManifest{}, err
	} else {
		listwisePolicy.IsolatedInheritedObjectives = parseScoreSpectrumObjectiveNames(value)
	}
	listwisePolicy.ListwiseGeometryTrain = listwisePolicy.ListwiseGeometryResearchOnly || listwisePolicy.TrainAllowedForResearch || listwisePolicy.ReleaseTrainAllowed || listwisePolicy.CommercialUseAllowed || listwisePolicy.ListwiseGeometryBatchCount > 0 || len(listwisePolicy.SourceArtifactHashes) > 0 || len(listwisePolicy.AutoClearedObjectives) > 0 || len(listwisePolicy.IsolatedInheritedObjectives) > 0
	aoqtPolicy := AOQTTransformPolicy{}
	if aoqtPolicy.Schema, err = readOptionalString("aoqt_transform_schema"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.Enabled, err = readOptionalBool("aoqt_transform_enabled"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.ResearchOnly, err = readOptionalBool("aoqt_transform_research_only"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.ResearchTrainAllowed, err = readOptionalBool("aoqt_transform_research_train_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.ReleaseTrainAllowed, err = readOptionalBool("aoqt_transform_release_train_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.CommercialUseAllowed, err = readOptionalBool("aoqt_transform_commercial_use_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.FreeOpenReleaseAllowed, err = readOptionalBool("aoqt_transform_free_open_release_allowed"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.QualityClaim, err = readOptionalBool("aoqt_transform_quality_claim"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.AnchorArtifactSHA256, err = readOptionalString("aoqt_transform_anchor_artifact_sha256"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.AnchorPackageManifestSHA256, err = readOptionalString("aoqt_transform_anchor_package_manifest_sha256"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.AnchorEmbeddingSpaceID, err = readOptionalString("aoqt_transform_anchor_embedding_space_id"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.DatasetManifestSHA256, err = readOptionalString("aoqt_transform_dataset_manifest_sha256"); err != nil {
		return PackageManifest{}, err
	}
	if value, err := readOptionalString("aoqt_transform_qrels_sha256_by_dataset"); err != nil {
		return PackageManifest{}, err
	} else {
		aoqtPolicy.QrelsSHA256ByDataset = parseAOQTQrelsHashes(value)
	}
	if aoqtPolicy.CompatibilityDigest, err = readOptionalString("aoqt_transform_compatibility_digest"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.TransformSHA256, err = readOptionalString("aoqt_transform_transform_sha256"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.PairingsSHA256, err = readOptionalString("aoqt_transform_pairings_sha256"); err != nil {
		return PackageManifest{}, err
	}
	if aoqtPolicy.AnglesSHA256, err = readOptionalString("aoqt_transform_angles_sha256"); err != nil {
		return PackageManifest{}, err
	}

	r := bytes.NewReader(xpkgBody)
	readU32 := func() (uint32, error) {
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}
	readI64 := func() (int64, error) {
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}

	count, err := readU32()
	if err != nil {
		return PackageManifest{}, err
	}
	if int64(count) != expectedFiles {
		return PackageManifest{}, fmt.Errorf("package manifest file_count = %d, XPKG count = %d", expectedFiles, count)
	}
	files := make([]PackageManifestFile, 0, count)
	for i := uint32(0); i < count; i++ {
		roleIdx, err := readU32()
		if err != nil {
			return PackageManifest{}, err
		}
		pathIdx, err := readU32()
		if err != nil {
			return PackageManifest{}, err
		}
		size, err := readI64()
		if err != nil {
			return PackageManifest{}, err
		}
		sum := make([]byte, sha256.Size)
		if _, err := io.ReadFull(r, sum); err != nil {
			return PackageManifest{}, err
		}
		files = append(files, PackageManifestFile{
			Role:   strg.At(roleIdx),
			Path:   strg.At(pathIdx),
			SHA256: hex.EncodeToString(sum),
			Bytes:  size,
		})
	}
	if r.Len() != 0 {
		return PackageManifest{}, fmt.Errorf("package manifest XPKG has %d trailing bytes", r.Len())
	}
	manifest := PackageManifest{
		Version:          version,
		Kind:             PackageKind(kind),
		ModuleName:       moduleName,
		ArtifactVersion:  artifactVersion,
		Files:            files,
		ScoreSpectrum:    scorePolicy,
		ListwiseGeometry: listwisePolicy,
		AOQTTransform:    aoqtPolicy,
	}
	if err := manifest.Validate(); err != nil {
		return PackageManifest{}, err
	}
	return manifest, nil
}

func BuildPackageManifest(kind PackageKind, mod *eosartifact.Module, files map[string]string) (PackageManifest, error) {
	manifest, err := buildPackageManifestUnchecked(kind, mod, files)
	if err != nil {
		return PackageManifest{}, err
	}
	return manifest, manifest.Validate()
}

func buildPackageManifestUnchecked(kind PackageKind, mod *eosartifact.Module, files map[string]string) (PackageManifest, error) {
	if mod == nil {
		return PackageManifest{}, fmt.Errorf("nil module")
	}
	roles := make([]string, 0, len(files))
	for role := range files {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	items := make([]PackageManifestFile, 0, len(roles))
	for _, role := range roles {
		path := files[role]
		if path == "" {
			return PackageManifest{}, fmt.Errorf("package file %q path is required", role)
		}
		sum, size, err := fileHash(path)
		if err != nil {
			return PackageManifest{}, fmt.Errorf("hash %q: %w", role, err)
		}
		items = append(items, PackageManifestFile{
			Role:   role,
			Path:   filepath.Base(path),
			SHA256: sum,
			Bytes:  size,
		})
	}
	manifest := PackageManifest{
		Version:         PackageManifestVersion,
		Kind:            kind,
		ModuleName:      mod.Name,
		ArtifactVersion: mod.Version,
		Files:           items,
	}
	return manifest, nil
}

func RebuildSiblingPackageManifest(artifactPath string) (PackageManifest, string, error) {
	manifestPath := ResolvePackageManifestPath(artifactPath)
	current, err := ReadPackageManifestFile(manifestPath)
	if err != nil {
		return PackageManifest{}, "", err
	}
	mod, err := eosartifact.ReadFile(artifactPath)
	if err != nil {
		return PackageManifest{}, "", err
	}
	files, err := siblingPackageFilesForKind(artifactPath, current.Kind)
	if err != nil {
		return PackageManifest{}, "", err
	}
	rebuilt, err := buildPackageManifestUnchecked(current.Kind, mod, files)
	if err != nil {
		return PackageManifest{}, "", err
	}
	rebuilt.ScoreSpectrum = current.ScoreSpectrum
	rebuilt.ListwiseGeometry = current.ListwiseGeometry
	rebuilt.AOQTTransform = current.AOQTTransform
	outPath := DefaultPackageManifestPath(artifactPath)
	if err := rebuilt.WriteFile(outPath); err != nil {
		return PackageManifest{}, "", err
	}
	return rebuilt, outPath, nil
}

func siblingPackageFilesForKind(artifactPath string, kind PackageKind) (map[string]string, error) {
	paths := map[string]string{
		"artifact": artifactPath,
	}
	addRequired := func(role, path string) error {
		if path == "" {
			return fmt.Errorf("package file %q path is required", role)
		}
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("package file %q: %w", role, err)
		}
		paths[role] = path
		return nil
	}
	addOptional := func(role, path string) {
		if path == "" {
			return
		}
		if _, err := os.Stat(path); err == nil {
			paths[role] = path
		}
	}
	if err := addRequired("embedding_manifest", ResolveEmbeddingManifestPath(artifactPath)); err != nil {
		return nil, err
	}
	if err := addRequired("weights", DefaultWeightFilePath(artifactPath)); err != nil {
		return nil, err
	}
	if err := addRequired("memory_plan", DefaultMemoryPlanPath(artifactPath)); err != nil {
		return nil, err
	}
	addOptional("tokenizer", DefaultTokenizerPath(artifactPath))
	addOptional(EmbeddingPostPoolTransformRole, DefaultPostPoolTransformPath(artifactPath))
	switch kind {
	case PackageEmbedding:
		return paths, nil
	case PackageTraining:
		if err := addRequired("train_manifest", ResolveEmbeddingTrainManifestPath(artifactPath)); err != nil {
			return nil, err
		}
		if err := addRequired("checkpoint", DefaultEmbeddingCheckpointPath(artifactPath)); err != nil {
			return nil, err
		}
		if err := addRequired("train_profile", DefaultEmbeddingTrainProfilePath(artifactPath)); err != nil {
			return nil, err
		}
		return paths, nil
	default:
		return nil, fmt.Errorf("unsupported package kind %q", kind)
	}
}

func (m PackageManifest) CacheKey() string {
	if err := m.Validate(); err != nil {
		return ""
	}
	h := sha256.New()
	write := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{'\n'})
	}
	write(m.Version)
	write(string(m.Kind))
	write(m.ModuleName)
	write(m.ArtifactVersion)
	write(fmt.Sprintf("%t", m.ScoreSpectrum.ScoreSpectrumResearchOnly))
	write(fmt.Sprintf("%t", m.ScoreSpectrum.TrainAllowedForResearch))
	write(fmt.Sprintf("%t", m.ScoreSpectrum.ReleaseTrainAllowed))
	write(fmt.Sprintf("%t", m.ScoreSpectrum.CommercialUseAllowed))
	write(formatScoreSpectrumSourceHashes(m.ScoreSpectrum.SourceArtifactHashes))
	write(fmt.Sprintf("%d", m.ScoreSpectrum.ScoreSpectrumRowCount))
	write(formatScoreSpectrumObjectiveNames(m.ScoreSpectrum.AutoClearedObjectives))
	write(formatScoreSpectrumObjectiveNames(m.ScoreSpectrum.IsolatedInheritedObjectives))
	write(fmt.Sprintf("%t", m.ListwiseGeometry.ListwiseGeometryResearchOnly))
	write(fmt.Sprintf("%t", m.ListwiseGeometry.TrainAllowedForResearch))
	write(fmt.Sprintf("%t", m.ListwiseGeometry.ReleaseTrainAllowed))
	write(fmt.Sprintf("%t", m.ListwiseGeometry.CommercialUseAllowed))
	write(formatScoreSpectrumSourceHashes(m.ListwiseGeometry.SourceArtifactHashes))
	write(fmt.Sprintf("%d", m.ListwiseGeometry.ListwiseGeometryBatchCount))
	write(formatScoreSpectrumObjectiveNames(m.ListwiseGeometry.AutoClearedObjectives))
	write(formatScoreSpectrumObjectiveNames(m.ListwiseGeometry.IsolatedInheritedObjectives))
	write(m.AOQTTransform.Schema)
	write(fmt.Sprintf("%t", m.AOQTTransform.Enabled))
	write(fmt.Sprintf("%t", m.AOQTTransform.ResearchOnly))
	write(fmt.Sprintf("%t", m.AOQTTransform.ResearchTrainAllowed))
	write(fmt.Sprintf("%t", m.AOQTTransform.ReleaseTrainAllowed))
	write(fmt.Sprintf("%t", m.AOQTTransform.CommercialUseAllowed))
	write(fmt.Sprintf("%t", m.AOQTTransform.FreeOpenReleaseAllowed))
	write(fmt.Sprintf("%t", m.AOQTTransform.QualityClaim))
	write(m.AOQTTransform.AnchorArtifactSHA256)
	write(m.AOQTTransform.AnchorPackageManifestSHA256)
	write(m.AOQTTransform.AnchorEmbeddingSpaceID)
	write(m.AOQTTransform.DatasetManifestSHA256)
	write(formatAOQTQrelsHashes(m.AOQTTransform.QrelsSHA256ByDataset))
	write(m.AOQTTransform.CompatibilityDigest)
	write(m.AOQTTransform.TransformSHA256)
	write(m.AOQTTransform.PairingsSHA256)
	write(m.AOQTTransform.AnglesSHA256)
	for _, item := range m.Files {
		write(item.Role)
		write(item.Path)
		write(item.SHA256)
		write(fmt.Sprintf("%d", item.Bytes))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func (m PackageManifest) VerifyFiles(paths map[string]string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	for _, item := range m.Files {
		path, ok := paths[item.Role]
		if !ok || path == "" {
			return fmt.Errorf("package manifest expects file role %q", item.Role)
		}
		if filepath.Base(path) != item.Path {
			return fmt.Errorf("package file %q path = %q, want %q", item.Role, filepath.Base(path), item.Path)
		}
		sum, size, err := fileHash(path)
		if err != nil {
			return fmt.Errorf("verify %q: %w", item.Role, err)
		}
		if sum != item.SHA256 {
			return fmt.Errorf("package file %q sha256 mismatch", item.Role)
		}
		if size != item.Bytes {
			return fmt.Errorf("package file %q size mismatch: got %d want %d", item.Role, size, item.Bytes)
		}
	}
	return nil
}

func (m PackageManifest) HasFileRole(role string) bool {
	for _, item := range m.Files {
		if item.Role == role {
			return true
		}
	}
	return false
}

func (m PackageManifest) validateAOQTTransformRoleBinding(hasTransformRole bool) error {
	if hasTransformRole && m.Kind != PackageEmbedding {
		return fmt.Errorf("package manifest file role %q is only supported for %q packages", EmbeddingPostPoolTransformRole, PackageEmbedding)
	}
	if hasTransformRole && !m.AOQTTransform.Enabled {
		return fmt.Errorf("package manifest file role %q requires enabled AOQT transform policy", EmbeddingPostPoolTransformRole)
	}
	if m.AOQTTransform.Enabled && m.Kind != PackageEmbedding {
		return fmt.Errorf("AOQT transform policy is only supported for %q packages", PackageEmbedding)
	}
	if m.AOQTTransform.Enabled && !hasTransformRole {
		return fmt.Errorf("AOQT transform policy requires package file role %q", EmbeddingPostPoolTransformRole)
	}
	return m.AOQTTransform.Validate()
}

func (p AOQTTransformPolicy) Validate() error {
	if !p.Enabled {
		return nil
	}
	if p.Schema != AOQTTransformPolicySchema {
		return fmt.Errorf("AOQT transform policy schema %q is not supported, want %q", p.Schema, AOQTTransformPolicySchema)
	}
	if !p.ResearchOnly || !p.ResearchTrainAllowed || p.ReleaseTrainAllowed || p.CommercialUseAllowed || p.FreeOpenReleaseAllowed || p.QualityClaim {
		return fmt.Errorf("AOQT transform policy requires research_only=true, research_train_allowed=true, release_train_allowed=false, commercial_use_allowed=false, free_open_release_allowed=false, quality_claim=false")
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{"anchor_artifact_sha256", p.AnchorArtifactSHA256},
		{"anchor_package_manifest_sha256", p.AnchorPackageManifestSHA256},
		{"dataset_manifest_sha256", p.DatasetManifestSHA256},
		{"compatibility_digest", p.CompatibilityDigest},
		{"transform_sha256", p.TransformSHA256},
		{"pairings_sha256", p.PairingsSHA256},
		{"angles_sha256", p.AnglesSHA256},
	} {
		if field.value == "" {
			return fmt.Errorf("AOQT transform policy %s is required", field.name)
		}
		if err := validateAOQTSHA256(field.value, "AOQT transform policy "+field.name); err != nil {
			return err
		}
	}
	if p.AnchorEmbeddingSpaceID == "" {
		return fmt.Errorf("AOQT transform policy anchor_embedding_space_id is required")
	}
	if len(p.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT transform policy qrels_sha256_by_dataset coverage is required")
	}
	for dataset, sum := range p.QrelsSHA256ByDataset {
		if strings.TrimSpace(dataset) == "" {
			return fmt.Errorf("AOQT transform policy qrels dataset name is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT transform policy qrels_sha256_by_dataset"); err != nil {
			return err
		}
	}
	return nil
}

func formatAOQTQrelsHashes(values map[string]string) string {
	if len(values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var b bytes.Buffer
	for i, key := range keys {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(key)
		b.WriteByte('=')
		b.WriteString(values[key])
	}
	return b.String()
}

func parseAOQTQrelsHashes(value string) map[string]string {
	if value == "" {
		return nil
	}
	out := map[string]string{}
	for _, row := range bytes.Split([]byte(value), []byte{'\n'}) {
		if len(row) == 0 {
			continue
		}
		parts := bytes.SplitN(row, []byte{'='}, 2)
		if len(parts) != 2 {
			continue
		}
		out[string(parts[0])] = string(parts[1])
	}
	return out
}

func fileHash(path string) (string, int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", 0, err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), int64(len(data)), nil
}
