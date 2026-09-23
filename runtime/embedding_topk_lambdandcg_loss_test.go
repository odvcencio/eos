package eosruntime

import (
	"math"
	"testing"
)

func TestTopKLambdaNDCGSwapDeltaUsesRawGradedGains(t *testing.T) {
	scores := []float32{0.9, 0.8, 0.7}
	gains := []float32{0, 2, 1}
	ids := []string{"d0", "d1", "d2"}

	got, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 0.5, 0, nil)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.EligiblePairs != 3 || got.ContributingPairs != 3 {
		t.Fatalf("pairs = eligible %d contributing %d, want 3/3", got.EligiblePairs, got.ContributingPairs)
	}
	discount0 := 1.0
	discount1 := 1 / math.Log2(3)
	idcg := 2*discount0 + 1*discount1
	w10 := 2 * (discount0 - discount1) / idcg
	w20 := (discount0 - 0) / idcg
	w12 := discount1 / idcg
	weightSum := w10 + w20 + w12
	wantLoss := float32((w10*float64(softplus32((scores[0]-scores[1])/0.5)) + w20*float64(softplus32((scores[0]-scores[2])/0.5)) + w12*float64(softplus32((scores[2]-scores[1])/0.5))) / weightSum)
	assertClose32(t, got.Loss, wantLoss, 1e-6, "loss")
	if got.Grad[1] >= 0 || got.Grad[2] >= 0 || got.Grad[0] <= 0 {
		t.Fatalf("grad signs = %+v, want positives pulled up and cutoff intruder pushed down", got.Grad)
	}
}

func TestTopKLambdaNDCGMultiPositiveCutoffCrossing(t *testing.T) {
	scores := []float32{0.8, 0.7, 0.6, 0.5}
	gains := []float32{0, 3, 2, 1}
	ids := []string{"n", "p3", "p2", "p1"}

	got, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 0.25, 0, nil)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.EligiblePairs != 6 {
		t.Fatalf("eligible pairs = %d, want all 6 graded higher/lower pairs", got.EligiblePairs)
	}
	if got.ContributingPairs != 5 {
		t.Fatalf("contributing pairs = %d, want 5: the rank-1/rank-2 equal-discount-outside swap should be zero", got.ContributingPairs)
	}
	if got.Grad[3] == 0 {
		t.Fatalf("outside-cutoff positive grad = %f, want nonzero from cutoff-crossing swaps", got.Grad[3])
	}
	if got.Grad[0] <= 0 {
		t.Fatalf("top-ranked nonrelevant grad = %f, want positive", got.Grad[0])
	}
}

func TestTopKLambdaNDCGCutoffOutsideKCanZeroBranch(t *testing.T) {
	scores := []float32{0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.20, 0.10}
	gains := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
	ids := []string{"n00", "n01", "n02", "n03", "n04", "n05", "n06", "n07", "n08", "n09", "p", "tail"}
	mask := make([][]bool, len(scores))
	for i := range mask {
		mask[i] = make([]bool, len(scores))
	}
	mask[10][11] = true

	narrow, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 10, 0.05, 0, mask)
	if err != nil {
		t.Fatalf("narrow cutoff loss: %v", err)
	}
	if narrow.EligiblePairs != 1 || narrow.ContributingPairs != 0 || narrow.Loss != 0 {
		t.Fatalf("narrow cutoff result = %+v, want one eligible outside-K pair with zero contribution", narrow)
	}

	wide, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 100, 0.05, 0, mask)
	if err != nil {
		t.Fatalf("wide cutoff loss: %v", err)
	}
	if wide.EligiblePairs != 1 || wide.ContributingPairs != 1 || wide.Loss <= 0 {
		t.Fatalf("wide cutoff result = %+v, want same pair to contribute", wide)
	}

	defaulted, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, effectiveTurboQuantTopKRecallCutoff(0), 0.05, 0, mask)
	if err != nil {
		t.Fatalf("default recall cutoff loss: %v", err)
	}
	if defaulted.EligiblePairs != 1 || defaulted.ContributingPairs != 1 || defaulted.Loss <= 0 {
		t.Fatalf("default recall cutoff result = %+v, want omitted recall cutoff to default to cutoff100 and contribute", defaulted)
	}
}

func TestTopKLambdaNDCGDeterministicTiesUseCandidateID(t *testing.T) {
	scores := []float32{1, 1, 1}
	gains := []float32{0, 2, 1}
	ids := []string{"b", "a", "c"}

	ranks := topkLambdaNDCGCurrentRanks(scores, ids)
	if ranks[1] != 0 || ranks[0] != 1 || ranks[2] != 2 {
		t.Fatalf("ranks = %+v, want ID-ascending tie order a,b,c", ranks)
	}
	got, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 1, 0, nil)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.ContributingPairs != 3 {
		t.Fatalf("contributing pairs = %d, want 3", got.ContributingPairs)
	}
}

func TestTopKLambdaNDCGEligibilityMask(t *testing.T) {
	scores := []float32{0.9, 0.8, 0.7}
	gains := []float32{0, 2, 1}
	ids := []string{"d0", "d1", "d2"}
	mask := [][]bool{
		{false, false, false},
		{true, false, false},
		{false, false, false},
	}

	got, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 0.5, 0, mask)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.EligiblePairs != 1 || got.ContributingPairs != 1 {
		t.Fatalf("pairs = eligible %d contributing %d, want 1/1", got.EligiblePairs, got.ContributingPairs)
	}
	if got.Grad[2] != 0 {
		t.Fatalf("masked candidate grad = %f, want zero", got.Grad[2])
	}
	if got.Grad[1] >= 0 || got.Grad[0] <= 0 {
		t.Fatalf("grad signs = %+v, want only p2>d0 pair active", got.Grad)
	}
}

func TestTopKLambdaNDCGZeroIDCGReturnsSafeZero(t *testing.T) {
	got, err := topkLambdaNDCGLossAndGrad(
		[]float32{2, 1},
		[]float32{0, 0},
		[]string{"a", "b"},
		10,
		0.5,
		0,
		nil,
	)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.Loss != 0 || got.EligiblePairs != 0 || got.ContributingPairs != 0 {
		t.Fatalf("zero-idcg result = %+v, want safe zero", got)
	}
	for i, grad := range got.Grad {
		if grad != 0 {
			t.Fatalf("grad[%d] = %f, want zero", i, grad)
		}
	}
}

func TestTopKLambdaNDCGValidation(t *testing.T) {
	tests := []struct {
		name   string
		scores []float32
		gains  []float32
		ids    []string
		tau    float32
		mask   [][]bool
	}{
		{name: "score-gain length", scores: []float32{1}, gains: []float32{1, 0}, ids: []string{"a"}, tau: 1},
		{name: "id length", scores: []float32{1}, gains: []float32{1}, ids: nil, tau: 1},
		{name: "tau zero", scores: []float32{1}, gains: []float32{1}, ids: []string{"a"}, tau: 0},
		{name: "tau nan", scores: []float32{1}, gains: []float32{1}, ids: []string{"a"}, tau: float32(math.NaN())},
		{name: "score nan", scores: []float32{float32(math.NaN())}, gains: []float32{1}, ids: []string{"a"}, tau: 1},
		{name: "gain inf", scores: []float32{1}, gains: []float32{float32(math.Inf(1))}, ids: []string{"a"}, tau: 1},
		{name: "mask rows", scores: []float32{1}, gains: []float32{1}, ids: []string{"a"}, tau: 1, mask: [][]bool{{false}, {false}}},
		{name: "mask cols", scores: []float32{1}, gains: []float32{1}, ids: []string{"a"}, tau: 1, mask: [][]bool{{false, false}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := topkLambdaNDCGLossAndGrad(tc.scores, tc.gains, tc.ids, 10, tc.tau, 0, tc.mask); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestTopKLambdaNDCGFiniteDifferenceAndRowNormalization(t *testing.T) {
	scores := []float32{0.3, 0.1, 0.2, -0.4}
	gains := []float32{0, 2, 1, 0}
	ids := []string{"n0", "p2", "p1", "n1"}

	got, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 0.7, 0.05, nil)
	if err != nil {
		t.Fatalf("lambda ndcg loss: %v", err)
	}
	if got.Loss <= 0 {
		t.Fatalf("loss = %f, want positive", got.Loss)
	}
	eps := float32(1e-3)
	for i := range scores {
		plusScores := append([]float32(nil), scores...)
		minusScores := append([]float32(nil), scores...)
		plusScores[i] += eps
		minusScores[i] -= eps
		plus, err := topkLambdaNDCGLossAndGrad(plusScores, gains, ids, 2, 0.7, 0.05, nil)
		if err != nil {
			t.Fatalf("plus loss: %v", err)
		}
		minus, err := topkLambdaNDCGLossAndGrad(minusScores, gains, ids, 2, 0.7, 0.05, nil)
		if err != nil {
			t.Fatalf("minus loss: %v", err)
		}
		numeric := (plus.Loss - minus.Loss) / (2 * eps)
		assertClose32(t, got.Grad[i], numeric, 2e-3, "finite diff grad")
	}

	wide, err := topkLambdaNDCGLossAndGrad(scores, gains, ids, 2, 0.7, 0.05, nil)
	if err != nil {
		t.Fatalf("wide loss: %v", err)
	}
	scaledScores := append([]float32(nil), scores...)
	for i := range scaledScores {
		scaledScores[i] *= 10
	}
	scaled, err := topkLambdaNDCGLossAndGrad(scaledScores, gains, ids, 2, 0.7, 0.05, nil)
	if err != nil {
		t.Fatalf("scaled loss: %v", err)
	}
	if wide.EligiblePairs != scaled.EligiblePairs || wide.ContributingPairs != scaled.ContributingPairs {
		t.Fatalf("pair counts changed after score scaling: %+v vs %+v", wide, scaled)
	}
}
