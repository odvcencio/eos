package eosruntime

import (
	"fmt"
	"math"
	"slices"
)

type topkLambdaNDCGLossResult struct {
	Loss              float32
	Grad              []float32
	EligiblePairs     int
	ContributingPairs int
}

func topkLambdaNDCGLossAndGrad(scores, gains []float32, candidateIDs []string, cutoff int, tau, margin float32, eligiblePairs [][]bool) (topkLambdaNDCGLossResult, error) {
	var result topkLambdaNDCGLossResult
	if len(scores) != len(gains) {
		return result, fmt.Errorf("top-k LambdaNDCG loss requires scores and gains with matching lengths")
	}
	if len(scores) != len(candidateIDs) {
		return result, fmt.Errorf("top-k LambdaNDCG loss requires scores and candidate IDs with matching lengths")
	}
	if tau <= 0 || !isFinite32(tau) {
		return result, fmt.Errorf("top-k LambdaNDCG tau must be finite and positive")
	}
	if !isFinite32(margin) {
		return result, fmt.Errorf("top-k LambdaNDCG margin must be finite")
	}
	n := len(scores)
	result.Grad = make([]float32, n)
	if n == 0 || cutoff <= 0 {
		return result, nil
	}
	for i := range scores {
		if !isFinite32(scores[i]) {
			return result, fmt.Errorf("top-k LambdaNDCG score[%d] must be finite", i)
		}
		if !isFinite32(gains[i]) {
			return result, fmt.Errorf("top-k LambdaNDCG gain[%d] must be finite", i)
		}
	}
	if len(eligiblePairs) > 0 {
		if len(eligiblePairs) != n {
			return result, fmt.Errorf("top-k LambdaNDCG eligibility mask row count = %d, want %d", len(eligiblePairs), n)
		}
		for i := range eligiblePairs {
			if len(eligiblePairs[i]) != n {
				return result, fmt.Errorf("top-k LambdaNDCG eligibility mask row %d length = %d, want %d", i, len(eligiblePairs[i]), n)
			}
		}
	}
	k := cutoff
	if k > n {
		k = n
	}
	idcg := topkLambdaNDCGIDCG(gains, k)
	if idcg == 0 {
		return result, nil
	}
	ranks := topkLambdaNDCGCurrentRanks(scores, candidateIDs)
	discounts := make([]float64, n)
	for rank := range discounts {
		if rank < k {
			discounts[rank] = 1 / math.Log2(float64(rank)+2)
		}
	}

	type weightedPair struct {
		high   int
		low    int
		weight float64
	}
	pairs := make([]weightedPair, 0)
	weightSum := float64(0)
	for high := 0; high < n; high++ {
		for low := 0; low < n; low++ {
			if gains[high] <= gains[low] {
				continue
			}
			if len(eligiblePairs) > 0 && !eligiblePairs[high][low] {
				continue
			}
			result.EligiblePairs++
			weight := topkLambdaNDCGSwapDelta(gains, discounts, idcg, ranks[high], ranks[low], high, low)
			if weight == 0 {
				continue
			}
			result.ContributingPairs++
			weightSum += weight
			pairs = append(pairs, weightedPair{high: high, low: low, weight: weight})
		}
	}
	if weightSum == 0 {
		return result, nil
	}
	for _, pair := range pairs {
		weight := float32(pair.weight / weightSum)
		z := (scores[pair.low] - scores[pair.high] + margin) / tau
		loss := softplus32(z)
		scale := weight * sigmoid32(z) / tau
		result.Loss += weight * loss
		result.Grad[pair.high] -= scale
		result.Grad[pair.low] += scale
	}
	return result, nil
}

func topkLambdaNDCGCurrentRanks(scores []float32, candidateIDs []string) []int {
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		if scores[a] > scores[b] {
			return -1
		}
		if scores[a] < scores[b] {
			return 1
		}
		if candidateIDs[a] < candidateIDs[b] {
			return -1
		}
		if candidateIDs[a] > candidateIDs[b] {
			return 1
		}
		return a - b
	})
	ranks := make([]int, len(scores))
	for rank, idx := range order {
		ranks[idx] = rank
	}
	return ranks
}

func topkLambdaNDCGIDCG(gains []float32, cutoff int) float64 {
	positive := make([]float32, 0, len(gains))
	for _, gain := range gains {
		if gain > 0 {
			positive = append(positive, gain)
		}
	}
	slices.SortFunc(positive, func(a, b float32) int {
		if a > b {
			return -1
		}
		if a < b {
			return 1
		}
		return 0
	})
	dcg := float64(0)
	limit := cutoff
	if limit > len(positive) {
		limit = len(positive)
	}
	for i := 0; i < limit; i++ {
		dcg += float64(positive[i]) / math.Log2(float64(i)+2)
	}
	return dcg
}

func topkLambdaNDCGSwapDelta(gains []float32, discounts []float64, idcg float64, rankA, rankB, indexA, indexB int) float64 {
	if idcg == 0 || rankA == rankB {
		return 0
	}
	aGain := topkLambdaNDCGPositiveGain(gains[indexA])
	bGain := topkLambdaNDCGPositiveGain(gains[indexB])
	before := aGain*discounts[rankA] + bGain*discounts[rankB]
	after := aGain*discounts[rankB] + bGain*discounts[rankA]
	delta := (before - after) / idcg
	if delta < 0 {
		return -delta
	}
	return delta
}

func topkLambdaNDCGPositiveGain(gain float32) float64 {
	if gain <= 0 {
		return 0
	}
	return float64(gain)
}
