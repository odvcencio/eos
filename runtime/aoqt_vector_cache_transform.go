package eosruntime

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const AOQTVectorCacheTransformBindingSchema = "eos.aoqt.vector_cache_transform_binding.v1"

type AOQTVectorCacheTransformConfig struct {
	TransformPath      string
	DocVectorPath      string
	QueryVectorPath    string
	OutputDocPath      string
	OutputQueryPath    string
	SidecarPath        string
	ExpectedDim        int
	ExpectedStages     int
	ExpectedPairsStage int
	ExpectedAngleCount int
	Dataset            string
	Artifact           string
}

type AOQTVectorCacheTransformBinding struct {
	Schema          string                       `json:"schema"`
	Dataset         string                       `json:"dataset,omitempty"`
	Artifact        string                       `json:"artifact,omitempty"`
	TransformPath   string                       `json:"transform_path"`
	TransformSHA256 string                       `json:"transform_sha256"`
	Topology        AOQTVectorCacheTopologyAudit `json:"topology"`
	Inputs          AOQTVectorCacheTransformIO   `json:"inputs"`
	Outputs         AOQTVectorCacheTransformIO   `json:"outputs"`
}

type AOQTVectorCacheTopologyAudit struct {
	Kind           string  `json:"kind"`
	Dim            int     `json:"dim"`
	Seed           int64   `json:"seed,omitempty"`
	StageCount     int     `json:"stage_count"`
	PairsPerStage  int     `json:"pairs_per_stage"`
	AngleCount     int     `json:"angle_count"`
	PairingsSHA256 string  `json:"pairings_sha256"`
	AnglesSHA256   string  `json:"angles_sha256"`
	Orthogonality  float64 `json:"orthogonality_frobenius_per_dim"`
	AuditBound     bool    `json:"audit_bound"`
	ExpectedDim    int     `json:"expected_dim,omitempty"`
	ExpectedStages int     `json:"expected_stages,omitempty"`
	ExpectedPairs  int     `json:"expected_pairs_per_stage,omitempty"`
	ExpectedAngles int     `json:"expected_angle_count,omitempty"`
}

type AOQTVectorCacheTransformIO struct {
	DocVectorPath     string `json:"doc_vector_path"`
	DocVectorSHA256   string `json:"doc_vector_sha256"`
	DocVectorCount    int    `json:"doc_vector_count"`
	QueryVectorPath   string `json:"query_vector_path"`
	QueryVectorSHA256 string `json:"query_vector_sha256"`
	QueryVectorCount  int    `json:"query_vector_count"`
	VectorFieldPolicy string `json:"vector_field_policy"`
	IDsAndRolesPolicy string `json:"ids_and_roles_policy"`
}

func TransformAOQTRetrievalVectorCaches(cfg AOQTVectorCacheTransformConfig) (AOQTVectorCacheTransformBinding, error) {
	if strings.TrimSpace(cfg.TransformPath) == "" {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("AOQT vector transform path is required")
	}
	if strings.TrimSpace(cfg.DocVectorPath) == "" || strings.TrimSpace(cfg.QueryVectorPath) == "" {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("AOQT input doc and query vector paths are required")
	}
	if strings.TrimSpace(cfg.OutputDocPath) == "" || strings.TrimSpace(cfg.OutputQueryPath) == "" || strings.TrimSpace(cfg.SidecarPath) == "" {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("AOQT output doc/query vector paths and sidecar path are required")
	}
	transform, err := ReadAOQTGivensTransformFile(cfg.TransformPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("read AOQT transform: %w", err)
	}
	topology, err := validateAOQTVectorCacheTransformTopology(transform, cfg)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	runtime := newAOQTGivensRuntimeAfterValidation(transform)
	cleanup := []string{}
	cleanupOnFailure := true
	defer func() {
		if cleanupOnFailure {
			for _, path := range cleanup {
				_ = os.Remove(path)
			}
		}
	}()
	docCount, err := transformAOQTVectorCacheFile(runtime, cfg.DocVectorPath, cfg.OutputDocPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("transform document vectors: %w", err)
	}
	cleanup = append(cleanup, cfg.OutputDocPath)
	queryCount, err := transformAOQTVectorCacheFile(runtime, cfg.QueryVectorPath, cfg.OutputQueryPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("transform query vectors: %w", err)
	}
	cleanup = append(cleanup, cfg.OutputQueryPath)
	transformSHA, err := sha256FileHexAOQTVectorCache(cfg.TransformPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	inputDocSHA, err := sha256FileHexAOQTVectorCache(cfg.DocVectorPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	inputQuerySHA, err := sha256FileHexAOQTVectorCache(cfg.QueryVectorPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	outputDocSHA, err := sha256FileHexAOQTVectorCache(cfg.OutputDocPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	outputQuerySHA, err := sha256FileHexAOQTVectorCache(cfg.OutputQueryPath)
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	binding := AOQTVectorCacheTransformBinding{
		Schema:          AOQTVectorCacheTransformBindingSchema,
		Dataset:         cfg.Dataset,
		Artifact:        cfg.Artifact,
		TransformPath:   cfg.TransformPath,
		TransformSHA256: transformSHA,
		Topology:        topology,
		Inputs: AOQTVectorCacheTransformIO{
			DocVectorPath:     cfg.DocVectorPath,
			DocVectorSHA256:   inputDocSHA,
			DocVectorCount:    docCount,
			QueryVectorPath:   cfg.QueryVectorPath,
			QueryVectorSHA256: inputQuerySHA,
			QueryVectorCount:  queryCount,
			VectorFieldPolicy: "replace_first_present_vector_embedding_values",
			IDsAndRolesPolicy: "preserve_all_non_vector_fields_including_id__id_parent_child_role",
		},
		Outputs: AOQTVectorCacheTransformIO{
			DocVectorPath:     cfg.OutputDocPath,
			DocVectorSHA256:   outputDocSHA,
			DocVectorCount:    docCount,
			QueryVectorPath:   cfg.OutputQueryPath,
			QueryVectorSHA256: outputQuerySHA,
			QueryVectorCount:  queryCount,
			VectorFieldPolicy: "replace_first_present_vector_embedding_values",
			IDsAndRolesPolicy: "preserve_all_non_vector_fields_including_id__id_parent_child_role",
		},
	}
	data, err := json.MarshalIndent(binding, "", "  ")
	if err != nil {
		return AOQTVectorCacheTransformBinding{}, err
	}
	data = append(data, '\n')
	if err := writeExclusiveAOQTVectorCacheFile(cfg.SidecarPath, data, 0o644); err != nil {
		return AOQTVectorCacheTransformBinding{}, fmt.Errorf("write AOQT vector transform sidecar: %w", err)
	}
	cleanup = append(cleanup, cfg.SidecarPath)
	cleanupOnFailure = false
	return binding, nil
}

func validateAOQTVectorCacheTransformTopology(transform AOQTGivensTransform, cfg AOQTVectorCacheTransformConfig) (AOQTVectorCacheTopologyAudit, error) {
	if err := transform.Validate(); err != nil {
		return AOQTVectorCacheTopologyAudit{}, err
	}
	if cfg.ExpectedDim > 0 && transform.Dim != cfg.ExpectedDim {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform dim = %d, want %d", transform.Dim, cfg.ExpectedDim)
	}
	pairingsSHA, err := transform.PairingsSHA256()
	if err != nil {
		return AOQTVectorCacheTopologyAudit{}, err
	}
	anglesSHA, err := transform.AnglesSHA256()
	if err != nil {
		return AOQTVectorCacheTopologyAudit{}, err
	}
	orth, err := transform.OrthogonalityFrobeniusPerDim()
	if err != nil {
		return AOQTVectorCacheTopologyAudit{}, err
	}
	if transform.Audit.PairingsSHA256 == "" || transform.Audit.AnglesSHA256 == "" || transform.Audit.Orthogonality == nil {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform audit must bind pairings_sha256, angles_sha256, and orthogonality")
	}
	if transform.Audit.PairingsSHA256 != pairingsSHA || transform.Audit.AnglesSHA256 != anglesSHA || math.Abs(*transform.Audit.Orthogonality-orth) > 1e-12 {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform audit binding mismatch")
	}
	pairsPerStage := 0
	angleCount := 0
	for index, stage := range transform.Stages {
		if index == 0 {
			pairsPerStage = len(stage.Pairs)
		} else if len(stage.Pairs) != pairsPerStage {
			return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform stage %d pairs = %d, want uniform %d", index, len(stage.Pairs), pairsPerStage)
		}
		angleCount += len(stage.Angles)
	}
	if cfg.ExpectedStages > 0 && len(transform.Stages) != cfg.ExpectedStages {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform stages = %d, want %d", len(transform.Stages), cfg.ExpectedStages)
	}
	if cfg.ExpectedPairsStage > 0 && pairsPerStage != cfg.ExpectedPairsStage {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform pairs per stage = %d, want %d", pairsPerStage, cfg.ExpectedPairsStage)
	}
	if cfg.ExpectedAngleCount > 0 && angleCount != cfg.ExpectedAngleCount {
		return AOQTVectorCacheTopologyAudit{}, fmt.Errorf("AOQT transform angle count = %d, want %d", angleCount, cfg.ExpectedAngleCount)
	}
	return AOQTVectorCacheTopologyAudit{
		Kind:           transform.Kind,
		Dim:            transform.Dim,
		Seed:           transform.Seed,
		StageCount:     len(transform.Stages),
		PairsPerStage:  pairsPerStage,
		AngleCount:     angleCount,
		PairingsSHA256: pairingsSHA,
		AnglesSHA256:   anglesSHA,
		Orthogonality:  orth,
		AuditBound:     true,
		ExpectedDim:    cfg.ExpectedDim,
		ExpectedStages: cfg.ExpectedStages,
		ExpectedPairs:  cfg.ExpectedPairsStage,
		ExpectedAngles: cfg.ExpectedAngleCount,
	}, nil
}

func transformAOQTVectorCacheFile(runtime aoqtGivensRuntime, inputPath, outputPath string) (int, error) {
	in, err := os.Open(inputPath)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return 0, err
	}
	out, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return 0, fmt.Errorf("output %q already exists", outputPath)
		}
		return 0, err
	}
	removeOnFailure := true
	defer func() {
		_ = out.Close()
		if removeOnFailure {
			_ = os.Remove(outputPath)
		}
	}()
	writer := bufio.NewWriter(out)
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	count := 0
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		transformed, err := transformAOQTVectorCacheRecord(runtime, line)
		if err != nil {
			return 0, fmt.Errorf("%s:%d: %w", inputPath, lineNo, err)
		}
		if _, err := writer.Write(transformed); err != nil {
			return 0, err
		}
		if err := writer.WriteByte('\n'); err != nil {
			return 0, err
		}
		count++
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	if count == 0 {
		return 0, fmt.Errorf("%s: no vector records found", inputPath)
	}
	if err := writer.Flush(); err != nil {
		return 0, err
	}
	if err := out.Sync(); err != nil {
		return 0, err
	}
	if err := out.Close(); err != nil {
		return 0, err
	}
	removeOnFailure = false
	return count, nil
}

func transformAOQTVectorCacheRecord(runtime aoqtGivensRuntime, line []byte) ([]byte, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(line, &object); err != nil {
		return nil, err
	}
	field, raw, err := firstAOQTVectorField(object)
	if err != nil {
		return nil, err
	}
	var vector []float32
	if err := json.Unmarshal(raw, &vector); err != nil {
		return nil, fmt.Errorf("%s must be a numeric vector: %w", field, err)
	}
	if err := validateAOQTFiniteVector(vector, field); err != nil {
		return nil, err
	}
	out, err := runtime.applyVector(vector)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	object[field] = data
	return json.Marshal(object)
}

func firstAOQTVectorField(object map[string]json.RawMessage) (string, json.RawMessage, error) {
	for _, field := range []string{"vector", "embedding", "values"} {
		if raw, ok := object[field]; ok && len(raw) > 0 && string(raw) != "null" {
			return field, raw, nil
		}
	}
	return "", nil, fmt.Errorf("record missing vector, embedding, or values field")
}

func sha256FileHexAOQTVectorCache(path string) (string, error) {
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

func writeExclusiveAOQTVectorCacheFile(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("output %q already exists", path)
		}
		return err
	}
	removeOnFailure := true
	defer func() {
		_ = file.Close()
		if removeOnFailure {
			_ = os.Remove(path)
		}
	}()
	if _, err := file.Write(data); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	removeOnFailure = false
	return nil
}
