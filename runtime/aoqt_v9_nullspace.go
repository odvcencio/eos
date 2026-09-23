package eosruntime

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

const (
	AOQTV9NullspaceBasisVersion        = "aoqt-v9-active-set-nullspace-v1"
	AOQTV9ConeDerivativeVersion        = "exact-sign-zero-neutral-v1"
	AOQTV9ProtectedDerivativeTolerance = 1e-9
	AOQTV9ProtectedActiveSetMaxRows    = 16
	aoqtV9NullspaceMinNorm             = 1e-12
)

func AOQTV9NullspaceMagnitudes() []float64 {
	return []float64{0.00125, 0.000625, 0.0003125, 0.00015625}
}

type AOQTV9ProtectedSubset struct {
	Ordinal int   `json:"ordinal"`
	Rows    []int `json:"rows"`
}

type AOQTV9NullspaceCandidate struct {
	Ordinal                   int                       `json:"ordinal"`
	Subset                    AOQTV9ProtectedSubset     `json:"subset"`
	Direction                 []float64                 `json:"direction"`
	DirectionMaxAbs           float64                   `json:"direction_max_abs"`
	PrimaryDerivative         float64                   `json:"primary_derivative"`
	ProtectedDerivatives      []float64                 `json:"protected_derivatives"`
	ProtectedMaxDerivative    float64                   `json:"protected_max_derivative"`
	ProtectedMaxAbsDerivative float64                   `json:"protected_max_abs_derivative"`
	Cone                      AOQTV9ConeDerivativeCheck `json:"cone"`
	RankOrder                 int                       `json:"rank_order"`
}

type AOQTV9ConeDerivativeCheck struct {
	Version                string    `json:"version"`
	PrimaryDerivative      float64   `json:"primary_derivative"`
	ProtectedDerivatives   []float64 `json:"protected_derivatives"`
	ProtectedMaxDerivative float64   `json:"protected_max_derivative"`
	Accepted               bool      `json:"accepted"`
	RejectionReason        string    `json:"rejection_reason,omitempty"`
}

func AOQTV9CanonicalProtectedSubsets(protectedRowCount int) ([]AOQTV9ProtectedSubset, error) {
	if protectedRowCount < 0 {
		return nil, fmt.Errorf("AOQT V9 nullspace protected row count must be nonnegative")
	}
	if protectedRowCount > AOQTV9ProtectedActiveSetMaxRows {
		return nil, fmt.Errorf("AOQT V9 nullspace protected row count %d exceeds bounded active-set guard %d", protectedRowCount, AOQTV9ProtectedActiveSetMaxRows)
	}
	var subsets []AOQTV9ProtectedSubset
	current := make([]int, 0, protectedRowCount)
	var visit func(start, size int)
	visit = func(start, size int) {
		if len(current) == size {
			rows := append([]int(nil), current...)
			subsets = append(subsets, AOQTV9ProtectedSubset{Ordinal: len(subsets) + 1, Rows: rows})
			return
		}
		remaining := size - len(current)
		for row := start; row <= protectedRowCount-remaining; row++ {
			current = append(current, row)
			visit(row+1, size)
			current = current[:len(current)-1]
		}
	}
	for size := 0; size <= protectedRowCount; size++ {
		visit(0, size)
	}
	return subsets, nil
}

func AOQTV9BuildNullspaceCandidates(q3Gradient []float64, protectedGradients [][]float64) ([]AOQTV9NullspaceCandidate, error) {
	if err := aoqtV9ValidateGradientMatrix(q3Gradient, protectedGradients); err != nil {
		return nil, err
	}
	subsets, err := AOQTV9CanonicalProtectedSubsets(len(protectedGradients))
	if err != nil {
		return nil, err
	}
	candidates := make([]AOQTV9NullspaceCandidate, 0, len(subsets))
	var lastProjectionErr error
	for _, subset := range subsets {
		rows := make([][]float64, 0, len(subset.Rows))
		for _, row := range subset.Rows {
			rows = append(rows, protectedGradients[row])
		}
		direction, err := AOQTV9ProjectDescentOntoProtectedNullspace(q3Gradient, rows)
		if err != nil {
			lastProjectionErr = err
			continue
		}
		cone, err := AOQTV9CheckConeDerivatives(q3Gradient, protectedGradients, direction)
		if err != nil {
			return nil, err
		}
		if !cone.Accepted {
			continue
		}
		candidates = append(candidates, AOQTV9NullspaceCandidate{
			Ordinal:                   subset.Ordinal,
			Subset:                    AOQTV9ProtectedSubset{Ordinal: subset.Ordinal, Rows: append([]int(nil), subset.Rows...)},
			Direction:                 direction,
			DirectionMaxAbs:           aoqtV9MaxAbs(direction),
			PrimaryDerivative:         cone.PrimaryDerivative,
			ProtectedDerivatives:      append([]float64(nil), cone.ProtectedDerivatives...),
			ProtectedMaxDerivative:    cone.ProtectedMaxDerivative,
			ProtectedMaxAbsDerivative: aoqtV9MaxAbs(cone.ProtectedDerivatives),
			Cone:                      cone,
		})
	}
	if len(candidates) == 0 {
		if lastProjectionErr != nil {
			return nil, lastProjectionErr
		}
		return nil, fmt.Errorf("AOQT V9 nullspace produced no accepted candidates")
	}
	return AOQTV9RankNullspaceCandidates(AOQTV9DedupeNullspaceCandidates(candidates)), nil
}

func AOQTV9ProjectDescentOntoProtectedNullspace(q3Gradient []float64, activeProtectedRows [][]float64) ([]float64, error) {
	if err := aoqtV9ValidateGradientMatrix(q3Gradient, activeProtectedRows); err != nil {
		return nil, err
	}
	direction := make([]float64, len(q3Gradient))
	for i, v := range q3Gradient {
		direction[i] = -v
	}
	basis, err := aoqtV9OrthonormalRows(activeProtectedRows, len(q3Gradient))
	if err != nil {
		return nil, err
	}
	for pass := 0; pass < 2; pass++ {
		for _, row := range basis {
			scale := aoqtV9Dot(direction, row)
			for i := range direction {
				direction[i] -= scale * row[i]
			}
		}
	}
	return aoqtV9NormalizeMaxAbs(direction)
}

func AOQTV9CheckConeDerivatives(q3Gradient []float64, protectedGradients [][]float64, direction []float64) (AOQTV9ConeDerivativeCheck, error) {
	if err := aoqtV9ValidateGradientMatrix(q3Gradient, protectedGradients); err != nil {
		return AOQTV9ConeDerivativeCheck{}, err
	}
	if len(direction) != len(q3Gradient) {
		return AOQTV9ConeDerivativeCheck{}, fmt.Errorf("AOQT V9 direction length = %d, want %d", len(direction), len(q3Gradient))
	}
	if !aoqtV9AllFinite(direction) || aoqtV9MaxAbs(direction) <= 0 {
		return AOQTV9ConeDerivativeCheck{}, fmt.Errorf("AOQT V9 direction must be finite and nonzero")
	}
	check := AOQTV9ConeDerivativeCheck{Version: AOQTV9ConeDerivativeVersion}
	check.PrimaryDerivative = aoqtV9Dot(q3Gradient, direction)
	check.ProtectedDerivatives = make([]float64, len(protectedGradients))
	check.ProtectedMaxDerivative = math.Inf(-1)
	if !aoqtV9Finite(check.PrimaryDerivative) {
		return AOQTV9ConeDerivativeCheck{}, fmt.Errorf("AOQT V9 primary derivative must be finite")
	}
	if check.PrimaryDerivative >= 0 {
		check.RejectionReason = "primary_not_descent"
		return check, nil
	}
	for i, row := range protectedGradients {
		derivative := aoqtV9Dot(row, direction)
		if !aoqtV9Finite(derivative) {
			return AOQTV9ConeDerivativeCheck{}, fmt.Errorf("AOQT V9 protected derivative[%d] must be finite", i)
		}
		check.ProtectedDerivatives[i] = derivative
		if derivative > check.ProtectedMaxDerivative {
			check.ProtectedMaxDerivative = derivative
		}
		if derivative > AOQTV9ProtectedDerivativeTolerance {
			check.RejectionReason = "protected_positive_derivative"
			return check, nil
		}
	}
	check.Accepted = true
	return check, nil
}

func AOQTV9DedupeNullspaceCandidates(candidates []AOQTV9NullspaceCandidate) []AOQTV9NullspaceCandidate {
	bestByDirection := make(map[string]AOQTV9NullspaceCandidate, len(candidates))
	for _, candidate := range candidates {
		key := aoqtV9DirectionKey(candidate.Direction)
		best, ok := bestByDirection[key]
		if !ok || aoqtV9CandidateLess(candidate, best) {
			bestByDirection[key] = candidate
		}
	}
	out := make([]AOQTV9NullspaceCandidate, 0, len(bestByDirection))
	for _, candidate := range bestByDirection {
		out = append(out, candidate)
	}
	return AOQTV9RankNullspaceCandidates(out)
}

func AOQTV9RankNullspaceCandidates(candidates []AOQTV9NullspaceCandidate) []AOQTV9NullspaceCandidate {
	out := append([]AOQTV9NullspaceCandidate(nil), candidates...)
	sort.SliceStable(out, func(i, j int) bool {
		return aoqtV9CandidateLess(out[i], out[j])
	})
	for i := range out {
		out[i].RankOrder = i + 1
	}
	return out
}

func aoqtV9CandidateLess(a, b AOQTV9NullspaceCandidate) bool {
	if a.Cone.Accepted != b.Cone.Accepted {
		return a.Cone.Accepted
	}
	if a.PrimaryDerivative != b.PrimaryDerivative {
		return a.PrimaryDerivative < b.PrimaryDerivative
	}
	if a.ProtectedMaxDerivative != b.ProtectedMaxDerivative {
		return a.ProtectedMaxDerivative < b.ProtectedMaxDerivative
	}
	if a.ProtectedMaxAbsDerivative != b.ProtectedMaxAbsDerivative {
		return a.ProtectedMaxAbsDerivative < b.ProtectedMaxAbsDerivative
	}
	if a.Subset.Ordinal != b.Subset.Ordinal {
		return a.Subset.Ordinal < b.Subset.Ordinal
	}
	return a.Ordinal < b.Ordinal
}

func aoqtV9OrthonormalRows(rows [][]float64, dim int) ([][]float64, error) {
	basis := make([][]float64, 0, len(rows))
	for rowIndex, row := range rows {
		if len(row) != dim {
			return nil, fmt.Errorf("AOQT V9 protected row[%d] length = %d, want %d", rowIndex, len(row), dim)
		}
		vector := append([]float64(nil), row...)
		for pass := 0; pass < 2; pass++ {
			for _, q := range basis {
				scale := aoqtV9Dot(vector, q)
				for i := range vector {
					vector[i] -= scale * q[i]
				}
			}
		}
		norm := math.Sqrt(aoqtV9Dot(vector, vector))
		if !aoqtV9Finite(norm) {
			return nil, fmt.Errorf("AOQT V9 protected row[%d] norm must be finite", rowIndex)
		}
		if norm <= aoqtV9NullspaceMinNorm {
			continue
		}
		for i := range vector {
			vector[i] /= norm
		}
		basis = append(basis, vector)
	}
	return basis, nil
}

func aoqtV9ValidateGradientMatrix(primary []float64, rows [][]float64) error {
	if len(primary) == 0 {
		return fmt.Errorf("AOQT V9 primary gradient must be nonempty")
	}
	if !aoqtV9AllFinite(primary) || aoqtV9MaxAbs(primary) <= 0 {
		return fmt.Errorf("AOQT V9 primary gradient must be finite and nonzero")
	}
	for i, row := range rows {
		if len(row) != len(primary) {
			return fmt.Errorf("AOQT V9 protected row[%d] length = %d, want %d", i, len(row), len(primary))
		}
		if !aoqtV9AllFinite(row) {
			return fmt.Errorf("AOQT V9 protected row[%d] must be finite", i)
		}
	}
	return nil
}

func aoqtV9NormalizeMaxAbs(values []float64) ([]float64, error) {
	if !aoqtV9AllFinite(values) {
		return nil, fmt.Errorf("AOQT V9 vector must be finite")
	}
	maxAbs := aoqtV9MaxAbs(values)
	if maxAbs <= aoqtV9NullspaceMinNorm {
		return nil, fmt.Errorf("AOQT V9 projected direction is zero")
	}
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = v / maxAbs
		if out[i] == 0 {
			out[i] = 0
		}
		if !aoqtV9Finite(out[i]) {
			return nil, fmt.Errorf("AOQT V9 normalized vector[%d] must be finite", i)
		}
	}
	return out, nil
}

func aoqtV9Dot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func aoqtV9MaxAbs(values []float64) float64 {
	var maxAbs float64
	for _, value := range values {
		abs := math.Abs(value)
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

func aoqtV9AllFinite(values []float64) bool {
	for _, value := range values {
		if !aoqtV9Finite(value) {
			return false
		}
	}
	return true
}

func aoqtV9Finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func aoqtV9DirectionKey(direction []float64) string {
	var builder strings.Builder
	for i, value := range direction {
		if i > 0 {
			builder.WriteByte(',')
		}
		if value == 0 {
			value = 0
		}
		builder.WriteString(strconv.FormatUint(math.Float64bits(value), 16))
	}
	return builder.String()
}
