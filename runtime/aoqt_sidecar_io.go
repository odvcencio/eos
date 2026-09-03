package eosruntime

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
)

const AOQTSidecarCalibrationIOReportSchema = "eos.q3_aoqt_sidecar_calibration_io.v1"

type AOQTSidecarCalibrationIOConfig struct {
	ManifestPath                        string
	RowsJSONLPath                       string
	ExpectedManifestSHA256              string
	ExpectedRowsSHA256                  string
	ExpectedAnchorArtifactSHA256        string
	ExpectedAnchorPackageManifestSHA256 string
	ExpectedAnchorEmbeddingSpaceID      string
	ExpectedCompatibilityDigest         string
	ExpectedTurboQuantSeed              int64
	ExpectedTopology                    AOQTSidecarTopologyBinding
	ExpectedSourceArtifactHashes        []string
	ExpectedVectorCacheHashes           []string
}

type AOQTSidecarCalibrationIOReport struct {
	Schema                      string                       `json:"schema"`
	ManifestPath                string                       `json:"manifest_path"`
	RowsJSONLPath               string                       `json:"rows_jsonl_path"`
	ManifestSHA256              string                       `json:"manifest_sha256"`
	RowsSHA256                  string                       `json:"rows_sha256"`
	AnchorArtifactSHA256        string                       `json:"anchor_artifact_sha256"`
	AnchorPackageManifestSHA256 string                       `json:"anchor_package_manifest_sha256"`
	AnchorEmbeddingSpaceID      string                       `json:"anchor_embedding_space_id"`
	CompatibilityDigest         string                       `json:"compatibility_digest"`
	TurboQuantSeed              int64                        `json:"turboquant_seed"`
	Topology                    AOQTSidecarTopologyBinding   `json:"topology"`
	ObjectiveContract           AOQTSidecarObjectiveContract `json:"objective_contract"`
	LegalGates                  AOQTSidecarLegalGates        `json:"legal_gates"`
	RowCount                    int                          `json:"row_count"`
	CandidateCount              int                          `json:"candidate_count"`
	PairCount                   int                          `json:"pair_count"`
}

func LoadAOQTSidecarCalibrationSet(cfg AOQTSidecarCalibrationIOConfig) (AOQTSidecarCalibrationSet, AOQTSidecarCalibrationIOReport, error) {
	if strings.TrimSpace(cfg.ManifestPath) == "" {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, fmt.Errorf("AOQT calibration IO manifest path is required")
	}
	if strings.TrimSpace(cfg.RowsJSONLPath) == "" {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, fmt.Errorf("AOQT calibration IO rows JSONL path is required")
	}
	manifestData, err := os.ReadFile(cfg.ManifestPath)
	if err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	manifestSHA := sha256BytesAOQT(manifestData)
	if err := validateExpectedAOQTSHA256(manifestSHA, cfg.ExpectedManifestSHA256, "AOQT calibration manifest"); err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	var manifest AOQTSidecarCalibrationManifest
	if err := strictUnmarshalAOQT(manifestData, &manifest); err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, fmt.Errorf("%s: invalid AOQT calibration manifest JSON: %w", cfg.ManifestPath, err)
	}
	rowsData, err := os.ReadFile(cfg.RowsJSONLPath)
	if err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	rowsSHA := sha256BytesAOQT(rowsData)
	if err := validateExpectedAOQTSHA256(rowsSHA, cfg.ExpectedRowsSHA256, "AOQT calibration rows JSONL"); err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	rows, err := parseAOQTCalibrationRowsJSONL(rowsData, cfg.RowsJSONLPath)
	if err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	set := AOQTSidecarCalibrationSet{Manifest: manifest, Rows: rows}
	if err := set.Validate(); err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	if err := validateAOQTCalibrationIOBindings(cfg, manifest); err != nil {
		return AOQTSidecarCalibrationSet{}, AOQTSidecarCalibrationIOReport{}, err
	}
	report := AOQTSidecarCalibrationIOReport{
		Schema:                      AOQTSidecarCalibrationIOReportSchema,
		ManifestPath:                cfg.ManifestPath,
		RowsJSONLPath:               cfg.RowsJSONLPath,
		ManifestSHA256:              manifestSHA,
		RowsSHA256:                  rowsSHA,
		AnchorArtifactSHA256:        manifest.AnchorArtifactSHA256,
		AnchorPackageManifestSHA256: manifest.AnchorPackageManifestSHA256,
		AnchorEmbeddingSpaceID:      manifest.AnchorEmbeddingSpaceID,
		CompatibilityDigest:         manifest.CompatibilityDigest,
		TurboQuantSeed:              manifest.TurboQuantSeed,
		Topology:                    manifest.Topology,
		ObjectiveContract:           manifest.ObjectiveContract,
		LegalGates:                  manifest.LegalGates,
		RowCount:                    len(rows),
		CandidateCount:              countAOQTCandidates(rows),
		PairCount:                   countAOQTPairs(rows),
	}
	return set, report, nil
}

func parseAOQTCalibrationRowsJSONL(data []byte, path string) ([]AOQTSidecarCalibrationRow, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 1024), 64*1024*1024)
	var rows []AOQTSidecarCalibrationRow
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			return nil, fmt.Errorf("%s:%d: AOQT calibration rows JSONL must not contain blank lines", path, lineNo)
		}
		var row AOQTSidecarCalibrationRow
		if err := strictUnmarshalAOQT([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("%s:%d: invalid AOQT calibration row JSON: %w", path, lineNo, err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("%s: AOQT calibration rows JSONL is empty", path)
	}
	return rows, nil
}

func validateAOQTCalibrationIOBindings(cfg AOQTSidecarCalibrationIOConfig, manifest AOQTSidecarCalibrationManifest) error {
	if err := requireEqualAOQTString(manifest.AnchorArtifactSHA256, cfg.ExpectedAnchorArtifactSHA256, "anchor_artifact_sha256"); err != nil {
		return err
	}
	if err := requireEqualAOQTString(manifest.AnchorPackageManifestSHA256, cfg.ExpectedAnchorPackageManifestSHA256, "anchor_package_manifest_sha256"); err != nil {
		return err
	}
	if err := requireEqualAOQTString(manifest.AnchorEmbeddingSpaceID, cfg.ExpectedAnchorEmbeddingSpaceID, "anchor_embedding_space_id"); err != nil {
		return err
	}
	if err := requireEqualAOQTString(manifest.CompatibilityDigest, cfg.ExpectedCompatibilityDigest, "compatibility_digest"); err != nil {
		return err
	}
	if cfg.ExpectedTurboQuantSeed == 0 {
		return fmt.Errorf("AOQT calibration IO expected turboquant_seed is required")
	}
	if manifest.TurboQuantSeed != cfg.ExpectedTurboQuantSeed {
		return fmt.Errorf("AOQT calibration IO turboquant_seed = %d, want %d", manifest.TurboQuantSeed, cfg.ExpectedTurboQuantSeed)
	}
	if err := cfg.ExpectedTopology.Validate(); err != nil {
		return fmt.Errorf("AOQT calibration IO expected topology is required and must be valid: %w", err)
	}
	if manifest.Topology != cfg.ExpectedTopology {
		return fmt.Errorf("AOQT calibration IO topology binding mismatch")
	}
	if err := validateAOQTHashList(cfg.ExpectedSourceArtifactHashes, "AOQT calibration IO expected source_artifact_hashes"); err != nil {
		return err
	}
	if !slices.Equal(manifest.SourceArtifactHashes, cfg.ExpectedSourceArtifactHashes) {
		return fmt.Errorf("AOQT calibration IO source_artifact_hashes binding mismatch")
	}
	if err := validateAOQTHashList(cfg.ExpectedVectorCacheHashes, "AOQT calibration IO expected vector_cache_hashes"); err != nil {
		return err
	}
	if !slices.Equal(manifest.VectorCacheHashes, cfg.ExpectedVectorCacheHashes) {
		return fmt.Errorf("AOQT calibration IO vector_cache_hashes binding mismatch")
	}
	return nil
}

func strictUnmarshalAOQT(data []byte, out any) error {
	if err := rejectDuplicateObjectKeysAOQT(data); err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var extra struct{}
	if err := dec.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing JSON payload")
	}
	return nil
}

func rejectDuplicateObjectKeysAOQT(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := consumeJSONValueNoDuplicateKeysAOQT(dec, "$"); err != nil {
		return err
	}
	var extra struct{}
	if err := dec.Decode(&extra); err != io.EOF {
		if err != nil {
			return err
		}
		return fmt.Errorf("trailing JSON payload")
	}
	return nil
}

func consumeJSONValueNoDuplicateKeysAOQT(dec *json.Decoder, path string) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				return err
			}
			key, ok := keyTok.(string)
			if !ok {
				return fmt.Errorf("object key at %s must be a string", path)
			}
			if seen[key] {
				return fmt.Errorf("duplicate object key %q at %s", key, path)
			}
			seen[key] = true
			if err := consumeJSONValueNoDuplicateKeysAOQT(dec, path+"."+key); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim('}') {
			return fmt.Errorf("unterminated object at %s", path)
		}
	case '[':
		index := 0
		for dec.More() {
			if err := consumeJSONValueNoDuplicateKeysAOQT(dec, fmt.Sprintf("%s[%d]", path, index)); err != nil {
				return err
			}
			index++
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim(']') {
			return fmt.Errorf("unterminated array at %s", path)
		}
	default:
		return fmt.Errorf("unexpected JSON delimiter %q at %s", delim, path)
	}
	return nil
}

func validateExpectedAOQTSHA256(got, want, label string) error {
	if strings.TrimSpace(want) == "" {
		return fmt.Errorf("%s expected sha256 is required", label)
	}
	if err := validateAOQTSHA256(want, label+" expected sha256"); err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("%s sha256 = %s, want %s", label, got, want)
	}
	return nil
}

func requireEqualAOQTString(got, want, label string) error {
	if strings.TrimSpace(want) == "" {
		return fmt.Errorf("AOQT calibration IO expected %s is required", label)
	}
	if got != want {
		return fmt.Errorf("AOQT calibration IO %s = %q, want %q", label, got, want)
	}
	return nil
}

func sha256BytesAOQT(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
