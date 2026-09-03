package eosruntime

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"m31labs.dev/eos/runtime/backend"
)

const (
	EmbeddingPostPoolTransformNone       = "none"
	EmbeddingPostPoolTransformAOQTGivens = "aoqt_givens_v1"

	EmbeddingPostPoolTransformRole = "post_pool_transform"

	AOQTTransformVersion = "eos/aoqt-givens-transform/v1"
)

// AOQTGivensTransform is a shared post-pool orthogonal transform sidecar.
type AOQTGivensTransform struct {
	Version  string        `json:"version"`
	Kind     string        `json:"kind"`
	Dim      int           `json:"dim"`
	Seed     int64         `json:"seed,omitempty"`
	AngleCap float32       `json:"angle_cap,omitempty"`
	Stages   []AOQTStage   `json:"stages"`
	Audit    AOQTAuditInfo `json:"audit,omitempty"`
}

type AOQTStage struct {
	Pairs  [][2]int  `json:"pairs"`
	Angles []float32 `json:"angles"`
}

type AOQTAuditInfo struct {
	PairingsSHA256 string   `json:"pairings_sha256,omitempty"`
	AnglesSHA256   string   `json:"angles_sha256,omitempty"`
	Orthogonality  *float64 `json:"orthogonality_frobenius_per_dim,omitempty"`
}

func DefaultPostPoolTransformPath(artifactPath string) string {
	base := artifactPath
	if ext := filepath.Ext(base); ext != "" {
		base = base[:len(base)-len(ext)]
	}
	return base + ".aoqt.json"
}

func (m EmbeddingManifest) requiresPostPoolTransform() bool {
	switch m.PostPoolTransform {
	case "", EmbeddingPostPoolTransformNone:
		return false
	default:
		return true
	}
}

func validatePostPoolTransformForManifest(manifest EmbeddingManifest, transform *AOQTGivensTransform) error {
	switch manifest.PostPoolTransform {
	case "", EmbeddingPostPoolTransformNone:
		if transform != nil {
			return fmt.Errorf("post-pool transform supplied but embedding manifest declares none")
		}
		return nil
	case EmbeddingPostPoolTransformAOQTGivens:
		if transform == nil {
			return fmt.Errorf("embedding manifest declares %q but no post-pool transform was supplied", manifest.PostPoolTransform)
		}
		if err := transform.Validate(); err != nil {
			return err
		}
		if manifest.OutputDim > 0 && transform.Dim != manifest.OutputDim {
			return fmt.Errorf("AOQT transform dim = %d, want embedding output_dim %d", transform.Dim, manifest.OutputDim)
		}
		return nil
	default:
		return fmt.Errorf("unsupported post_pool_transform %q", manifest.PostPoolTransform)
	}
}

func ReadAOQTGivensTransformFile(path string) (AOQTGivensTransform, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AOQTGivensTransform{}, err
	}
	var transform AOQTGivensTransform
	if err := json.Unmarshal(data, &transform); err != nil {
		return AOQTGivensTransform{}, fmt.Errorf("decode AOQT transform: %w", err)
	}
	if err := transform.Validate(); err != nil {
		return AOQTGivensTransform{}, err
	}
	return transform, nil
}

func (t AOQTGivensTransform) WriteFile(path string) error {
	if err := t.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func (t AOQTGivensTransform) Validate() error {
	if err := t.validateStructure(); err != nil {
		return err
	}
	return t.validateAudit()
}

func (t AOQTGivensTransform) validateStructure() error {
	if t.Version == "" {
		return fmt.Errorf("AOQT transform version is required")
	}
	if t.Version != AOQTTransformVersion {
		return fmt.Errorf("AOQT transform version %q is not supported, want %q", t.Version, AOQTTransformVersion)
	}
	if t.Kind != EmbeddingPostPoolTransformAOQTGivens {
		return fmt.Errorf("AOQT transform kind %q is not supported, want %q", t.Kind, EmbeddingPostPoolTransformAOQTGivens)
	}
	if t.Dim <= 0 {
		return fmt.Errorf("AOQT transform dim must be positive")
	}
	if len(t.Stages) == 0 {
		return fmt.Errorf("AOQT transform must contain at least one stage")
	}
	for stageIndex, stage := range t.Stages {
		if len(stage.Pairs) == 0 {
			return fmt.Errorf("AOQT stage %d has no pairs", stageIndex)
		}
		if len(stage.Angles) != len(stage.Pairs) {
			return fmt.Errorf("AOQT stage %d angle count = %d, want %d", stageIndex, len(stage.Angles), len(stage.Pairs))
		}
		seen := map[int]bool{}
		for pairIndex, pair := range stage.Pairs {
			a, b := pair[0], pair[1]
			if a < 0 || a >= t.Dim || b < 0 || b >= t.Dim {
				return fmt.Errorf("AOQT stage %d pair %d = [%d %d] outside dim %d", stageIndex, pairIndex, a, b, t.Dim)
			}
			if a == b {
				return fmt.Errorf("AOQT stage %d pair %d repeats coordinate %d", stageIndex, pairIndex, a)
			}
			if seen[a] || seen[b] {
				return fmt.Errorf("AOQT stage %d coordinate appears in more than one pair", stageIndex)
			}
			seen[a], seen[b] = true, true
			angle := stage.Angles[pairIndex]
			if math.IsNaN(float64(angle)) || math.IsInf(float64(angle), 0) {
				return fmt.Errorf("AOQT stage %d angle %d is not finite", stageIndex, pairIndex)
			}
			if t.AngleCap > 0 && float32(math.Abs(float64(angle))) > t.AngleCap+1e-7 {
				return fmt.Errorf("AOQT stage %d angle %d exceeds angle_cap", stageIndex, pairIndex)
			}
		}
	}
	return nil
}

func (t AOQTGivensTransform) validateAudit() error {
	if t.Audit.PairingsSHA256 != "" {
		got, err := t.PairingsSHA256()
		if err != nil {
			return err
		}
		if got != t.Audit.PairingsSHA256 {
			return fmt.Errorf("AOQT audit pairings_sha256 mismatch")
		}
	}
	if t.Audit.AnglesSHA256 != "" {
		got, err := t.AnglesSHA256()
		if err != nil {
			return err
		}
		if got != t.Audit.AnglesSHA256 {
			return fmt.Errorf("AOQT audit angles_sha256 mismatch")
		}
	}
	if t.Audit.Orthogonality != nil {
		got, err := t.OrthogonalityFrobeniusPerDim()
		if err != nil {
			return err
		}
		if math.Abs(got-*t.Audit.Orthogonality) > 1e-12 {
			return fmt.Errorf("AOQT audit orthogonality mismatch: got %.12g want %.12g", got, *t.Audit.Orthogonality)
		}
	}
	return nil
}

func (t AOQTGivensTransform) ApplyVector(in []float32) ([]float32, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if len(in) != t.Dim {
		return nil, fmt.Errorf("AOQT vector dim = %d, want %d", len(in), t.Dim)
	}
	out := append([]float32(nil), in...)
	t.applyInPlace(out)
	return out, nil
}

func (t AOQTGivensTransform) ApplyTensor(tensor *backend.Tensor) (*backend.Tensor, error) {
	if tensor == nil {
		return nil, fmt.Errorf("embedding tensor is nil")
	}
	if err := t.Validate(); err != nil {
		return nil, err
	}
	out := tensor.Clone()
	switch len(out.Shape) {
	case 1:
		if out.Shape[0] != t.Dim {
			return nil, fmt.Errorf("AOQT tensor dim = %d, want %d", out.Shape[0], t.Dim)
		}
		if len(out.F32) < t.Dim {
			return nil, fmt.Errorf("AOQT tensor has %d values, want %d", len(out.F32), t.Dim)
		}
		t.applyInPlace(out.F32[:t.Dim])
	case 2:
		rows, cols := out.Shape[0], out.Shape[1]
		if cols != t.Dim {
			return nil, fmt.Errorf("AOQT tensor columns = %d, want %d", cols, t.Dim)
		}
		if len(out.F32) < rows*cols {
			return nil, fmt.Errorf("AOQT tensor has %d values, want %d", len(out.F32), rows*cols)
		}
		for row := 0; row < rows; row++ {
			t.applyInPlace(out.F32[row*cols : (row+1)*cols])
		}
	default:
		return nil, fmt.Errorf("AOQT tensor rank = %d, want 1 or 2", len(out.Shape))
	}
	return out, nil
}

func (t AOQTGivensTransform) applyInPlace(vec []float32) {
	for _, stage := range t.Stages {
		for i, pair := range stage.Pairs {
			a, b := pair[0], pair[1]
			theta := float64(stage.Angles[i])
			c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
			x, y := vec[a], vec[b]
			vec[a] = c*x - s*y
			vec[b] = s*x + c*y
		}
	}
}

func (t AOQTGivensTransform) OrthogonalityFrobeniusPerDim() (float64, error) {
	if err := t.validateStructure(); err != nil {
		return 0, err
	}
	matrix := make([]float64, t.Dim*t.Dim)
	for i := 0; i < t.Dim; i++ {
		basis := make([]float32, t.Dim)
		basis[i] = 1
		t.applyInPlace(basis)
		for r, v := range basis {
			matrix[r*t.Dim+i] = float64(v)
		}
	}
	var sum float64
	for i := 0; i < t.Dim; i++ {
		for j := 0; j < t.Dim; j++ {
			var dot float64
			for k := 0; k < t.Dim; k++ {
				dot += matrix[k*t.Dim+i] * matrix[k*t.Dim+j]
			}
			want := 0.0
			if i == j {
				want = 1
			}
			delta := dot - want
			sum += delta * delta
		}
	}
	return math.Sqrt(sum) / float64(t.Dim), nil
}

func (t AOQTGivensTransform) PairingsSHA256() (string, error) {
	if err := t.validateStructure(); err != nil {
		return "", err
	}
	h := sha256.New()
	for _, stage := range t.Stages {
		for _, pair := range stage.Pairs {
			_, _ = h.Write([]byte(fmt.Sprintf("%d,%d\n", pair[0], pair[1])))
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (t AOQTGivensTransform) AnglesSHA256() (string, error) {
	if err := t.validateStructure(); err != nil {
		return "", err
	}
	h := sha256.New()
	for _, stage := range t.Stages {
		for _, angle := range stage.Angles {
			_, _ = h.Write([]byte(fmt.Sprintf("%.9g\n", angle)))
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
