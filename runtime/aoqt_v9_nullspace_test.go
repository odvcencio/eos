package eosruntime

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func TestAOQTV9NullspaceProjectsDescentOrthogonalToActiveRows(t *testing.T) {
	q3 := []float64{-2, -3, -4, -5, -6, -7}
	active := [][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
		{0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 1, 0},
	}
	direction, err := AOQTV9ProjectDescentOntoProtectedNullspace(q3, active)
	if err != nil {
		t.Fatalf("project descent: %v", err)
	}
	if got := aoqtV9MaxAbs(direction); math.Abs(got-1) > 1e-12 {
		t.Fatalf("direction max abs = %.17g, want 1", got)
	}
	for i, row := range active {
		if got := aoqtV9Dot(row, direction); math.Abs(got) > 1e-12 {
			t.Fatalf("active row %d derivative = %.17g, want orthogonal", i, got)
		}
	}
	if got := aoqtV9Dot(q3, direction); got >= 0 {
		t.Fatalf("primary derivative = %.17g, want descent", got)
	}
}

func TestAOQTV9NullspaceConeRejectsProtectedPositiveDerivative(t *testing.T) {
	q3 := []float64{1, 0}
	protected := [][]float64{{-1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}
	check, err := AOQTV9CheckConeDerivatives(q3, protected, []float64{-1, 0})
	if err != nil {
		t.Fatalf("cone check: %v", err)
	}
	if check.Accepted || check.RejectionReason != "protected_positive_derivative" {
		t.Fatalf("cone check = %+v, want protected positive derivative rejection", check)
	}
}

func TestAOQTV9NullspaceConeTreatsZeroProtectedDerivativeAsNeutral(t *testing.T) {
	q3 := []float64{0, 1}
	protected := [][]float64{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}
	check, err := AOQTV9CheckConeDerivatives(q3, protected, []float64{0, -1})
	if err != nil {
		t.Fatalf("cone check: %v", err)
	}
	if !check.Accepted || check.ProtectedMaxDerivative != 0 {
		t.Fatalf("cone check = %+v, want accepted with zero protected derivative", check)
	}
}

func TestAOQTV9NullspaceCanonicalSubsetsDeterministicOrder(t *testing.T) {
	subsets, err := AOQTV9CanonicalProtectedSubsets(3)
	if err != nil {
		t.Fatalf("subsets: %v", err)
	}
	wantRows := [][]int{
		[]int{},
		{0},
		{1},
		{2},
		{0, 1},
		{0, 2},
		{1, 2},
		{0, 1, 2},
	}
	if len(subsets) != len(wantRows) {
		t.Fatalf("subset count = %d, want %d", len(subsets), len(wantRows))
	}
	for i, subset := range subsets {
		if subset.Ordinal != i+1 || !slices.Equal(subset.Rows, wantRows[i]) {
			t.Fatalf("subset[%d] = %+v, want ordinal %d rows %v", i, subset, i+1, wantRows[i])
		}
	}
}

func TestAOQTV9NullspaceBuildRanksDeterministically(t *testing.T) {
	q3 := []float64{-1, -2, -3, -4, -5, -6}
	protected := [][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
		{0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 1, 0},
		{1, 0, 0, 0, 0, 0},
	}
	first, err := AOQTV9BuildNullspaceCandidates(q3, protected)
	if err != nil {
		t.Fatalf("build first: %v", err)
	}
	second, err := AOQTV9BuildNullspaceCandidates(q3, protected)
	if err != nil {
		t.Fatalf("build second: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("candidate build was nondeterministic")
	}
	if len(first) == 0 {
		t.Fatal("candidate build returned no directions")
	}
	for i, candidate := range first {
		if candidate.RankOrder != i+1 {
			t.Fatalf("candidate[%d] rank_order = %d, want %d", i, candidate.RankOrder, i+1)
		}
	}
}

func TestAOQTV9NullspaceDedupeKeepsBestDuplicateDirection(t *testing.T) {
	direction := []float64{0, -1}
	weaker := AOQTV9NullspaceCandidate{
		Ordinal:                   2,
		Subset:                    AOQTV9ProtectedSubset{Ordinal: 2, Rows: []int{0, 1, 2, 3, 4}},
		Direction:                 append([]float64(nil), direction...),
		DirectionMaxAbs:           1,
		PrimaryDerivative:         -1,
		ProtectedMaxDerivative:    0,
		ProtectedMaxAbsDerivative: 0,
		Cone:                      AOQTV9ConeDerivativeCheck{Version: AOQTV9ConeDerivativeVersion, Accepted: true},
	}
	stronger := weaker
	stronger.Ordinal = 1
	stronger.Subset.Ordinal = 1
	stronger.PrimaryDerivative = -2
	got := AOQTV9DedupeNullspaceCandidates([]AOQTV9NullspaceCandidate{weaker, stronger})
	if len(got) != 1 {
		t.Fatalf("deduped candidate count = %d, want 1", len(got))
	}
	if got[0].Ordinal != 1 || got[0].RankOrder != 1 || got[0].PrimaryDerivative != -2 {
		t.Fatalf("deduped candidate = %+v, want stronger duplicate at rank 1", got[0])
	}
}

func TestAOQTV9NullspaceRejectsNaNAndZeroProjection(t *testing.T) {
	if _, err := AOQTV9ProjectDescentOntoProtectedNullspace([]float64{math.NaN(), 1}, [][]float64{{1, 0}, {0, 1}, {0, 0}, {0, 0}, {0, 0}}); err == nil {
		t.Fatal("NaN primary gradient unexpectedly accepted")
	}
	if _, err := AOQTV9ProjectDescentOntoProtectedNullspace([]float64{1, 0, 0, 0, 0}, [][]float64{
		{1, 0, 0, 0, 0},
		{0, 1, 0, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 0, 1, 0},
		{0, 0, 0, 0, 1},
	}); err == nil {
		t.Fatal("zero nullspace projection unexpectedly accepted")
	}
	if _, err := AOQTV9CanonicalProtectedSubsets(AOQTV9ProtectedActiveSetMaxRows + 1); err == nil {
		t.Fatal("too many protected rows unexpectedly accepted")
	}
}

func TestAOQTV9NullspaceMagnitudesCanonical(t *testing.T) {
	want := []float64{0.00125, 0.000625, 0.0003125, 0.00015625}
	if got := AOQTV9NullspaceMagnitudes(); !reflect.DeepEqual(got, want) {
		t.Fatalf("magnitudes = %v, want %v", got, want)
	}
}

func TestAOQTV9NullspaceFiveProtectedRowsEnumeratesAllCardinalities(t *testing.T) {
	subsets, err := AOQTV9CanonicalProtectedSubsets(5)
	if err != nil {
		t.Fatalf("subsets: %v", err)
	}
	if len(subsets) != 32 {
		t.Fatalf("subset count = %d, want all 2^5 active sets", len(subsets))
	}
	if got := subsets[0].Rows; len(got) != 0 {
		t.Fatalf("first subset = %v, want empty active set", got)
	}
	if got := subsets[len(subsets)-1].Rows; !reflect.DeepEqual(got, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("last subset = %v, want full active set", got)
	}
}

func TestAOQTV9NullspaceBuildSkipsConeRejectedSubsets(t *testing.T) {
	q3 := []float64{1, 1}
	protected := [][]float64{
		{-1, 0},
		{0, 1},
		{0, 0},
		{0, 0},
		{0, 0},
	}
	candidates, err := AOQTV9BuildNullspaceCandidates(q3, protected)
	if err != nil {
		t.Fatalf("build candidates: %v", err)
	}
	for _, candidate := range candidates {
		if !candidate.Cone.Accepted {
			t.Fatalf("candidate retained rejected cone: %+v", candidate)
		}
	}
}

func TestAOQTV9NullspaceBuildErrorsWhenAllSubsetsConeRejected(t *testing.T) {
	q3 := []float64{1}
	protected := [][]float64{
		{-1},
		{-1},
		{-1},
		{-1},
		{-1},
	}
	if _, err := AOQTV9BuildNullspaceCandidates(q3, protected); err == nil {
		t.Fatal("all cone-rejected candidates unexpectedly accepted")
	}
}

func TestAOQTV9NullspaceConeUsesProtectedDerivativeTolerance(t *testing.T) {
	q3 := []float64{0, 1}
	protected := [][]float64{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}
	check, err := AOQTV9CheckConeDerivatives(q3, protected, []float64{AOQTV9ProtectedDerivativeTolerance / 2, -1})
	if err != nil {
		t.Fatalf("cone check tiny residual: %v", err)
	}
	if !check.Accepted {
		t.Fatalf("tiny positive residual rejected: %+v", check)
	}
	check, err = AOQTV9CheckConeDerivatives(q3, protected, []float64{AOQTV9ProtectedDerivativeTolerance * 2, -1})
	if err != nil {
		t.Fatalf("cone check positive residual: %v", err)
	}
	if check.Accepted || check.RejectionReason != "protected_positive_derivative" {
		t.Fatalf("large positive residual check = %+v, want protected rejection", check)
	}
}
