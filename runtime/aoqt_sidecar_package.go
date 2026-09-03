package eosruntime

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	eosartifact "m31labs.dev/eos/artifact/eos"
)

// AOQTSidecarCandidatePackageConfig controls writing a sibling AOQT candidate package.
type AOQTSidecarCandidatePackageConfig struct {
	AnchorArtifactPath                  string
	OutputArtifactPath                  string
	ExpectedAnchorArtifactSHA256        string
	ExpectedAnchorPackageManifestSHA256 string
	AnchorEmbeddingSpaceID              string
	DatasetManifestSHA256               string
	QrelsSHA256ByDataset                map[string]string
	CompatibilityDigest                 string
	Transform                           AOQTGivensTransform
	UseHardlinks                        bool
}

type AOQTSidecarCandidatePackageResult struct {
	Paths           EmbeddingPackagePaths
	PackageManifest PackageManifest
	AOQTPolicy      AOQTTransformPolicy
}

var writeAOQTCandidatePackageManifestExclusive = writeAOQTPackageManifestExclusive

// WriteAOQTSidecarCandidatePackage copies an immutable anchor package into a
// fresh sibling package and binds a post-pool AOQT transform sidecar.
func WriteAOQTSidecarCandidatePackage(cfg AOQTSidecarCandidatePackageConfig) (AOQTSidecarCandidatePackageResult, error) {
	if err := validateAOQTCandidatePackageConfig(cfg); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}

	anchorMod, err := eosartifact.ReadFile(cfg.AnchorArtifactPath)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	anchorPackagePath := ResolvePackageManifestPath(cfg.AnchorArtifactPath)
	anchorPackage, err := ReadPackageManifestFile(anchorPackagePath)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if anchorPackage.Kind != PackageEmbedding {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor package kind = %q, want %q", anchorPackage.Kind, PackageEmbedding)
	}
	if anchorPackage.HasFileRole(EmbeddingPostPoolTransformRole) || anchorPackage.AOQTTransform.Enabled {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor must not already declare a post-pool transform")
	}
	if err := rejectResearchOnlyRestrictedEmbeddingPackage(anchorPackage); err != nil {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor is not release-clean for loading: %w", err)
	}
	anchorArtifactSHA, _, err := fileHash(cfg.AnchorArtifactPath)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if anchorArtifactSHA != cfg.ExpectedAnchorArtifactSHA256 {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor artifact sha256 mismatch")
	}
	anchorPackageSHA, _, err := fileHash(anchorPackagePath)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if anchorPackageSHA != cfg.ExpectedAnchorPackageManifestSHA256 {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor package manifest sha256 mismatch")
	}
	anchorPaths, err := packageRolePaths(cfg.AnchorArtifactPath, anchorPackage)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := anchorPackage.VerifyFiles(anchorPaths); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	anchorManifest, err := ReadEmbeddingManifestFile(anchorPaths["embedding_manifest"])
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if anchorManifest.requiresPostPoolTransform() {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT candidate anchor embedding manifest already declares %q", anchorManifest.PostPoolTransform)
	}
	if anchorManifest.OutputDim > 0 && cfg.Transform.Dim != anchorManifest.OutputDim {
		return AOQTSidecarCandidatePackageResult{}, fmt.Errorf("AOQT transform dim = %d, want anchor output_dim %d", cfg.Transform.Dim, anchorManifest.OutputDim)
	}

	outPaths := aoqtCandidateOutputPaths(cfg.OutputArtifactPath, anchorPackage.HasFileRole("tokenizer"))
	if err := ensureAOQTCandidateOutputAbsent(outPaths); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := os.MkdirAll(filepath.Dir(cfg.OutputArtifactPath), 0o755); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	cleanup := newAOQTCandidateCleanup()
	defer cleanup.run()

	if err := copyAOQTFile(anchorPaths["artifact"], outPaths.ArtifactPath, cfg.UseHardlinks, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if anchorPackage.HasFileRole("tokenizer") {
		if err := copyAOQTFile(anchorPaths["tokenizer"], outPaths.TokenizerPath, cfg.UseHardlinks, cleanup); err != nil {
			return AOQTSidecarCandidatePackageResult{}, err
		}
	}
	if err := copyAOQTFile(anchorPaths["weights"], outPaths.WeightFilePath, cfg.UseHardlinks, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := copyAOQTFile(anchorPaths["memory_plan"], outPaths.MemoryPlanPath, cfg.UseHardlinks, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}

	candidateManifest := anchorManifest
	candidateManifest.PostPoolTransform = EmbeddingPostPoolTransformAOQTGivens
	if err := writeAOQTEmbeddingManifestExclusive(candidateManifest, outPaths.ManifestPath, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := writeAOQTGivensTransformExclusive(cfg.Transform, outPaths.PostPoolTransformPath, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	transformSHA, _, err := fileHash(outPaths.PostPoolTransformPath)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	pairingsSHA, err := cfg.Transform.PairingsSHA256()
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	anglesSHA, err := cfg.Transform.AnglesSHA256()
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	policy := AOQTTransformPolicy{
		Schema:                      AOQTTransformPolicySchema,
		Enabled:                     true,
		ResearchOnly:                true,
		ResearchTrainAllowed:        true,
		ReleaseTrainAllowed:         false,
		CommercialUseAllowed:        false,
		FreeOpenReleaseAllowed:      false,
		QualityClaim:                false,
		AnchorArtifactSHA256:        anchorArtifactSHA,
		AnchorPackageManifestSHA256: anchorPackageSHA,
		AnchorEmbeddingSpaceID:      cfg.AnchorEmbeddingSpaceID,
		DatasetManifestSHA256:       cfg.DatasetManifestSHA256,
		QrelsSHA256ByDataset:        cloneAOQTStringMap(cfg.QrelsSHA256ByDataset),
		CompatibilityDigest:         cfg.CompatibilityDigest,
		TransformSHA256:             transformSHA,
		PairingsSHA256:              pairingsSHA,
		AnglesSHA256:                anglesSHA,
	}
	if err := policy.Validate(); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}

	packageFiles := map[string]string{
		"artifact":                     outPaths.ArtifactPath,
		"embedding_manifest":           outPaths.ManifestPath,
		"weights":                      outPaths.WeightFilePath,
		"memory_plan":                  outPaths.MemoryPlanPath,
		EmbeddingPostPoolTransformRole: outPaths.PostPoolTransformPath,
	}
	if anchorPackage.HasFileRole("tokenizer") {
		packageFiles["tokenizer"] = outPaths.TokenizerPath
	}
	candidatePackage, err := buildPackageManifestUnchecked(PackageEmbedding, anchorMod, packageFiles)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	candidatePackage.ScoreSpectrum = packageScoreSpectrumPolicy(anchorPackage.ScoreSpectrum)
	candidatePackage.ListwiseGeometry = packageListwiseGeometryPolicy(anchorPackage.ListwiseGeometry)
	candidatePackage.AOQTTransform = policy
	if err := verifyAOQTTransformPolicyBinding(candidatePackage, outPaths, cfg.Transform); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := writeAOQTCandidatePackageManifestExclusive(candidatePackage, outPaths.PackageManifestPath, cleanup); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	verifyPaths, err := packageRolePaths(outPaths.ArtifactPath, candidatePackage)
	if err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	if err := candidatePackage.VerifyFiles(verifyPaths); err != nil {
		return AOQTSidecarCandidatePackageResult{}, err
	}
	cleanup.commit()

	return AOQTSidecarCandidatePackageResult{
		Paths:           outPaths,
		PackageManifest: candidatePackage,
		AOQTPolicy:      policy,
	}, nil
}

func validateAOQTCandidatePackageConfig(cfg AOQTSidecarCandidatePackageConfig) error {
	if cfg.UseHardlinks {
		return fmt.Errorf("AOQT candidate package hardlinks are not supported; copy-only output is required")
	}
	if cfg.AnchorArtifactPath == "" {
		return fmt.Errorf("AOQT candidate anchor artifact path is required")
	}
	if cfg.OutputArtifactPath == "" {
		return fmt.Errorf("AOQT candidate output artifact path is required")
	}
	if cfg.AnchorEmbeddingSpaceID == "" {
		return fmt.Errorf("AOQT candidate anchor embedding space id is required")
	}
	if err := validateAOQTSHA256(cfg.ExpectedAnchorArtifactSHA256, "AOQT candidate expected anchor artifact sha256"); err != nil {
		return err
	}
	if err := validateAOQTSHA256(cfg.ExpectedAnchorPackageManifestSHA256, "AOQT candidate expected anchor package manifest sha256"); err != nil {
		return err
	}
	if cfg.DatasetManifestSHA256 == "" {
		return fmt.Errorf("AOQT candidate dataset manifest sha256 is required")
	}
	if err := validateAOQTSHA256(cfg.DatasetManifestSHA256, "AOQT candidate dataset manifest sha256"); err != nil {
		return err
	}
	if cfg.CompatibilityDigest == "" {
		return fmt.Errorf("AOQT candidate compatibility digest is required")
	}
	if err := validateAOQTSHA256(cfg.CompatibilityDigest, "AOQT candidate compatibility digest"); err != nil {
		return err
	}
	if len(cfg.QrelsSHA256ByDataset) == 0 {
		return fmt.Errorf("AOQT candidate qrels sha256 dataset coverage is required")
	}
	for dataset, sum := range cfg.QrelsSHA256ByDataset {
		if strings.TrimSpace(dataset) == "" {
			return fmt.Errorf("AOQT candidate qrels dataset is required")
		}
		if err := validateAOQTSHA256(sum, "AOQT candidate qrels sha256"); err != nil {
			return err
		}
	}
	return cfg.Transform.Validate()
}

type aoqtCandidateCleanup struct {
	entries []aoqtCandidateCleanupEntry
	active  bool
}

type aoqtCandidateCleanupEntry struct {
	path string
	file *os.File
}

func newAOQTCandidateCleanup() *aoqtCandidateCleanup {
	return &aoqtCandidateCleanup{active: true}
}

func (c *aoqtCandidateCleanup) createdFile(path string, file *os.File) {
	if c == nil || path == "" {
		return
	}
	hold, err := openAOQTCleanupHandle(path, file)
	if err != nil {
		return
	}
	c.entries = append(c.entries, aoqtCandidateCleanupEntry{path: path, file: hold})
}

func (c *aoqtCandidateCleanup) commit() {
	if c != nil {
		c.active = false
		c.closeEntries()
	}
}

func (c *aoqtCandidateCleanup) run() {
	if c == nil || !c.active {
		return
	}
	defer c.closeEntries()
	for i := len(c.entries) - 1; i >= 0; i-- {
		entry := c.entries[i]
		if entry.path == "" {
			continue
		}
		info, err := os.Lstat(entry.path)
		if err != nil || info.Mode()&os.ModeSymlink != 0 {
			continue
		}
		createdInfo, err := entry.file.Stat()
		if err != nil || !os.SameFile(createdInfo, info) {
			continue
		}
		_ = os.Remove(entry.path)
	}
}

func (c *aoqtCandidateCleanup) closeEntries() {
	if c == nil {
		return
	}
	for i := range c.entries {
		if c.entries[i].file == nil {
			continue
		}
		_ = c.entries[i].file.Close()
		c.entries[i].file = nil
	}
}

func openAOQTCleanupHandle(path string, file *os.File) (*os.File, error) {
	if file == nil {
		return nil, fmt.Errorf("AOQT candidate cleanup file handle is required")
	}
	createdInfo, err := file.Stat()
	if err != nil {
		return nil, err
	}
	hold, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	holdInfo, err := hold.Stat()
	if err != nil {
		_ = hold.Close()
		return nil, err
	}
	if !os.SameFile(createdInfo, holdInfo) {
		_ = hold.Close()
		return nil, fmt.Errorf("AOQT candidate cleanup path changed after create: %s", path)
	}
	return hold, nil
}

func aoqtCandidateOutputPaths(artifactPath string, includeTokenizer bool) EmbeddingPackagePaths {
	paths := EmbeddingPackagePaths{
		ArtifactPath:          artifactPath,
		ManifestPath:          DefaultEmbeddingManifestPath(artifactPath),
		WeightFilePath:        DefaultWeightFilePath(artifactPath),
		MemoryPlanPath:        DefaultMemoryPlanPath(artifactPath),
		PackageManifestPath:   DefaultPackageManifestPath(artifactPath),
		PostPoolTransformPath: DefaultPostPoolTransformPath(artifactPath),
	}
	if includeTokenizer {
		paths.TokenizerPath = DefaultTokenizerPath(artifactPath)
	}
	return paths
}

func ensureAOQTCandidateOutputAbsent(paths EmbeddingPackagePaths) error {
	for role, path := range aoqtCandidatePathMap(paths) {
		if path == "" {
			continue
		}
		if _, err := os.Lstat(path); err == nil {
			return fmt.Errorf("AOQT candidate output %q already exists: %s", role, path)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func aoqtCandidatePathMap(paths EmbeddingPackagePaths) map[string]string {
	return map[string]string{
		"artifact":                     paths.ArtifactPath,
		"embedding_manifest":           paths.ManifestPath,
		"tokenizer":                    paths.TokenizerPath,
		"weights":                      paths.WeightFilePath,
		"memory_plan":                  paths.MemoryPlanPath,
		"package_manifest":             paths.PackageManifestPath,
		EmbeddingPostPoolTransformRole: paths.PostPoolTransformPath,
	}
}

func packageRolePaths(artifactPath string, manifest PackageManifest) (map[string]string, error) {
	dir := filepath.Dir(artifactPath)
	out := make(map[string]string, len(manifest.Files))
	for _, item := range manifest.Files {
		switch item.Role {
		case "artifact", "embedding_manifest", "tokenizer", "weights", "memory_plan", EmbeddingPostPoolTransformRole:
			out[item.Role] = filepath.Join(dir, item.Path)
		default:
			return nil, fmt.Errorf("AOQT candidate package does not support anchor file role %q", item.Role)
		}
	}
	for _, role := range []string{"artifact", "embedding_manifest", "weights", "memory_plan"} {
		if out[role] == "" {
			return nil, fmt.Errorf("AOQT candidate anchor package missing required role %q", role)
		}
	}
	return out, nil
}

func copyAOQTFile(src, dst string, hardlink bool, cleanup *aoqtCandidateCleanup) error {
	if hardlink {
		return fmt.Errorf("AOQT candidate package hardlinks are not supported; copy-only output is required")
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	cleanup.createdFile(dst, out)
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func writeAOQTEmbeddingManifestExclusive(manifest EmbeddingManifest, path string, cleanup *aoqtCandidateCleanup) error {
	data, err := encodeAuthoredManifestMLL("embedding_manifest", EmbeddingManifestVersion, manifest.nameOrDefault(), "Eos embedding manifest", manifest.mllValues())
	if err != nil {
		return err
	}
	return writeAOQTExclusiveFile(path, data, 0o644, cleanup)
}

func writeAOQTGivensTransformExclusive(transform AOQTGivensTransform, path string, cleanup *aoqtCandidateCleanup) error {
	if err := transform.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(transform, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return writeAOQTExclusiveFile(path, data, 0o644, cleanup)
}

func writeAOQTPackageManifestExclusive(manifest PackageManifest, path string, cleanup *aoqtCandidateCleanup) error {
	if err := manifest.Validate(); err != nil {
		return err
	}
	data, err := encodePackageManifestMLL(manifest)
	if err != nil {
		return err
	}
	return writeAOQTExclusiveFile(path, data, 0o644, cleanup)
}

func writeAOQTExclusiveFile(path string, data []byte, perm os.FileMode, cleanup *aoqtCandidateCleanup) error {
	if path == "" {
		return fmt.Errorf("AOQT candidate output path is required")
	}
	if _, err := os.Lstat(path); err == nil {
		return fmt.Errorf("AOQT candidate output already exists: %s", path)
	} else if !os.IsNotExist(err) {
		return err
	}
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		return err
	}
	cleanup.createdFile(path, out)
	n, writeErr := out.Write(data)
	closeErr := out.Close()
	if writeErr != nil {
		return writeErr
	}
	if n != len(data) {
		return io.ErrShortWrite
	}
	return closeErr
}
