//go:build linux && cgo

package cuda

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"m31labs.dev/eos/runtime/backend"
)

func TestCUDAContrastiveAcceleratorMatchesHostInfoNCE(t *testing.T) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		t.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query := backend.NewTensorF32([]int{3, 4}, []float32{
		0.25, -0.5, 0.75, 1.25,
		-0.25, 0.8, 0.3, -0.6,
		0.9, 0.1, -0.4, 0.2,
	})
	positive := backend.NewTensorF32([]int{3, 4}, []float32{
		-0.75, 0.5, 0.125, 1.5,
		0.15, 0.9, -0.25, -0.35,
		0.8, -0.2, -0.55, 0.45,
	})
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}

	got, err := accelAny.RunInfoNCE(query, positive, cfg)
	if err != nil {
		t.Fatalf("run infonce: %v", err)
	}
	wantQ, wantP, wantLoss, wantScore := hostInfoNCEGrad(query.F32, positive.F32, 3, 4, cfg.Temperature)
	assertTensorClose(t, got.QueryGrads, []int{3, 4}, wantQ)
	assertTensorClose(t, got.PositiveGrads, []int{3, 4}, wantP)
	assertCloseF32(t, got.LossSum, wantLoss, 0.0001)
	assertCloseF32(t, got.ScoreSum, wantScore, 0.0001)

	stats := accelAny.Stats()
	if stats.RunCalls != 1 {
		t.Fatalf("run calls = %d, want 1", stats.RunCalls)
	}
	if stats.UploadedBytes <= 0 {
		t.Fatalf("uploaded bytes = %d, want positive", stats.UploadedBytes)
	}
	if stats.DownloadedBytes <= 0 {
		t.Fatalf("downloaded bytes = %d, want positive", stats.DownloadedBytes)
	}
	if stats.RunNanos <= 0 {
		t.Fatalf("run nanos = %d, want positive", stats.RunNanos)
	}
}

func TestCUDAContrastiveAcceleratorMatchesHostRectangularInfoNCE(t *testing.T) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		t.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query := backend.NewTensorF32([]int{2, 4}, []float32{
		0.25, -0.5, 0.75, 1.25,
		-0.25, 0.8, 0.3, -0.6,
	})
	candidates := backend.NewTensorF32([]int{5, 4}, []float32{
		-0.75, 0.5, 0.125, 1.5,
		0.15, 0.9, -0.25, -0.35,
		0.8, -0.2, -0.55, 0.45,
		0.4, -0.1, 0.2, 0.7,
		-0.2, 0.6, 0.5, -0.4,
	})
	targets := []int{1, 3}
	cfg := backend.ContrastiveLossConfig{Temperature: 0.07}

	got, err := accelAny.RunInfoNCEWithTargets(query, candidates, targets, cfg)
	if err != nil {
		t.Fatalf("run rectangular infonce: %v", err)
	}
	wantQ, wantC, wantLoss, wantScore := hostInfoNCEGradTargets(query.F32, candidates.F32, 2, 5, 4, targets, cfg.Temperature)
	assertTensorClose(t, got.QueryGrads, []int{2, 4}, wantQ)
	assertTensorClose(t, got.PositiveGrads, []int{5, 4}, wantC)
	assertCloseF32(t, got.LossSum, wantLoss, 0.0001)
	assertCloseF32(t, got.ScoreSum, wantScore, 0.0001)
}

func TestCUDAContrastiveKernelsUse64BitPairIndexing(t *testing.T) {
	for name, source := range map[string]string{
		"scores": contrastiveScoresKernelSource,
		"grad":   contrastiveGradKernelSource,
	} {
		if !strings.Contains(source, "long long idx = (long long)blockIdx.x") {
			t.Fatalf("%s kernel does not use 64-bit global index", name)
		}
	}
	if !strings.Contains(contrastiveGradKernelSource, "(long long)query_rows * (long long)candidate_rows * (long long)width") {
		t.Fatal("grad kernel does not compute total grad elements in 64-bit")
	}
}

func BenchmarkCUDAContrastiveAccelerator256x64(b *testing.B) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		b.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		b.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, positive := syntheticContrastiveTensors(256, 64)
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := accelAny.RunInfoNCE(query, positive, cfg); err != nil {
			b.Fatalf("run infonce: %v", err)
		}
	}
}

func BenchmarkHostContrastiveInfoNCE256x64(b *testing.B) {
	query, positive := syntheticContrastiveTensors(256, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostInfoNCEGrad(query.F32, positive.F32, 256, 64, 0.05)
	}
}

func BenchmarkHostContrastiveInfoNCE128x512x64(b *testing.B) {
	query, _ := syntheticContrastiveTensors(128, 64)
	candidates, _ := syntheticContrastiveTensors(512, 64)
	targets := make([]int, 128)
	for i := range targets {
		targets[i] = i * 4
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostInfoNCEGradTargets(query.F32, candidates.F32, 128, 512, 64, targets, 0.05)
	}
}

func BenchmarkCUDAContrastiveAccelerator128x512x64(b *testing.B) {
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		b.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		b.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, _ := syntheticContrastiveTensors(128, 64)
	candidates, _ := syntheticContrastiveTensors(512, 64)
	targets := make([]int, 128)
	for i := range targets {
		targets[i] = i * 4
	}
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := accelAny.RunInfoNCEWithTargets(query, candidates, targets, cfg); err != nil {
			b.Fatalf("run rectangular infonce: %v", err)
		}
	}
}

// contrastiveGEMMBenchBatch/Width/Negatives fix the S1a before/after
// benchmark shape: batch 256, 5 negatives/query (own positive + 5 mined
// negatives per query, so a 6x shared candidate pool), d=256.
const (
	contrastiveGEMMBenchBatch      = 256
	contrastiveGEMMBenchWidth      = 256
	contrastiveGEMMBenchNegatives  = 5
	contrastiveGEMMBenchMultiplier = 1 + contrastiveGEMMBenchNegatives
)

func contrastiveGEMMBenchFixture() (*backend.Tensor, *backend.Tensor, []int) {
	query, _ := syntheticContrastiveTensors(contrastiveGEMMBenchBatch, contrastiveGEMMBenchWidth)
	candidates, _ := syntheticContrastiveTensors(contrastiveGEMMBenchBatch*contrastiveGEMMBenchMultiplier, contrastiveGEMMBenchWidth)
	targets := make([]int, contrastiveGEMMBenchBatch)
	for i := range targets {
		targets[i] = i * contrastiveGEMMBenchMultiplier
	}
	return query, candidates, targets
}

// BenchmarkCUDAContrastiveAcceleratorAtomic256x5neg256 is the "before"
// measurement for S1a: the retired one-thread-per-pair scoring kernel plus
// the atomicAdd gradient kernel, forced on via EOS_CUDA_CONTRASTIVE_GEMM=0.
func BenchmarkCUDAContrastiveAcceleratorAtomic256x5neg256(b *testing.B) {
	b.Setenv("EOS_CUDA_CONTRASTIVE_GEMM", "0")
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		b.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		b.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, candidates, targets := contrastiveGEMMBenchFixture()
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := accelAny.RunInfoNCEWithTargets(query, candidates, targets, cfg); err != nil {
			b.Fatalf("run rectangular infonce: %v", err)
		}
	}
}

// BenchmarkCUDAContrastiveAcceleratorGEMM256x5neg256 is the "after"
// measurement for S1a: the cuBLAS GEMM-based score/gradient path, forced on
// via EOS_CUDA_CONTRASTIVE_GEMM=1.
func BenchmarkCUDAContrastiveAcceleratorGEMM256x5neg256(b *testing.B) {
	b.Setenv("EOS_CUDA_CONTRASTIVE_GEMM", "1")
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		b.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		b.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, candidates, targets := contrastiveGEMMBenchFixture()
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := accelAny.RunInfoNCEWithTargets(query, candidates, targets, cfg); err != nil {
			b.Fatalf("run rectangular infonce: %v", err)
		}
	}
}

func syntheticContrastiveTensors(rows, width int) (*backend.Tensor, *backend.Tensor) {
	query := make([]float32, rows*width)
	positive := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for col := 0; col < width; col++ {
			idx := row*width + col
			value := float32(((row+1)*(col+3))%23) / 23
			query[idx] = value - 0.5
			positive[idx] = value*0.75 + float32((row+col)%7)/17 - 0.35
		}
	}
	return backend.NewTensorF32([]int{rows, width}, query), backend.NewTensorF32([]int{rows, width}, positive)
}

func hostInfoNCEGrad(query, positive []float32, rows, width int, temperature float32) ([]float32, []float32, float32, float32) {
	targets := make([]int, rows)
	for i := range targets {
		targets[i] = i
	}
	return hostInfoNCEGradTargets(query, positive, rows, rows, width, targets, temperature)
}

func hostInfoNCEGradTargets(query, candidates []float32, queryRows, candidateRows, width int, targetIndexes []int, temperature float32) ([]float32, []float32, float32, float32) {
	if temperature <= 0 {
		temperature = 0.05
	}
	queryNorms := tensorRowNorms(query, queryRows, width)
	candidateNorms := tensorRowNorms(candidates, candidateRows, width)
	queryGrads := make([]float32, queryRows*width)
	candidateGrads := make([]float32, candidateRows*width)
	totalLoss := float32(0)
	totalScore := float32(0)
	rowScores := make([]float32, candidateRows)
	probs := make([]float32, candidateRows)
	for row := 0; row < queryRows; row++ {
		for col := 0; col < candidateRows; col++ {
			score := hostCosineScore(query[row*width:(row+1)*width], candidates[col*width:(col+1)*width], queryNorms[row], candidateNorms[col])
			rowScores[col] = score
			totalScore += score
		}
		targetIndex := targetIndexes[row]
		totalLoss += hostInfoNCEProbs(rowScores, targetIndex, temperature, probs)
		for col, prob := range probs {
			target := float32(0)
			if col == targetIndex {
				target = 1
			}
			scale := (prob - target) / temperature
			hostAccumulateCosineGrad(
				query[row*width:(row+1)*width],
				candidates[col*width:(col+1)*width],
				queryNorms[row],
				candidateNorms[col],
				rowScores[col],
				scale,
				queryGrads[row*width:(row+1)*width],
				candidateGrads[col*width:(col+1)*width],
			)
		}
	}
	return queryGrads, candidateGrads, totalLoss, totalScore
}

func hostCosineScore(left, right []float32, leftNorm, rightNorm float32) float32 {
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	dot := float32(0)
	for i := range left {
		dot += left[i] * right[i]
	}
	return dot / (leftNorm * rightNorm)
}

func hostAccumulateCosineGrad(left, right []float32, leftNorm, rightNorm, score, scale float32, gradLeft, gradRight []float32) {
	if leftNorm == 0 || rightNorm == 0 {
		return
	}
	denom := leftNorm * rightNorm
	leftScale := score / (leftNorm * leftNorm)
	rightScale := score / (rightNorm * rightNorm)
	for i := range left {
		gradLeft[i] += scale * (right[i]/denom - left[i]*leftScale)
		gradRight[i] += scale * (left[i]/denom - right[i]*rightScale)
	}
}

func hostInfoNCEProbs(scores []float32, target int, temperature float32, probs []float32) float32 {
	maxLogit := scores[0] / temperature
	for _, score := range scores[1:] {
		logit := score / temperature
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	sum := float32(0)
	for i, score := range scores {
		value := float32(math.Exp(float64(score/temperature - maxLogit)))
		probs[i] = value
		sum += value
	}
	for i := range probs {
		probs[i] /= sum
	}
	prob := probs[target]
	if prob < 1e-12 {
		prob = 1e-12
	}
	return -float32(math.Log(float64(prob)))
}

func assertCloseF32(t *testing.T, got, want, tol float32) {
	t.Helper()
	diff := got - want
	if diff < -tol || diff > tol {
		t.Fatalf("got %f, want %f", got, want)
	}
}

// assertTensorCloseTol is assertTensorClose (cuda_integration_test.go) with a
// caller-supplied tolerance instead of the fixed 0.0005 default. The GEMM
// path sums in a different order than the host reference loop (S1a), so
// larger shapes need a wider tolerance than the small fixtures elsewhere in
// this file.
func assertTensorCloseTol(t *testing.T, tensor *backend.Tensor, wantShape []int, want []float32, tol float32) {
	t.Helper()
	if tensor == nil {
		t.Fatal("tensor is nil")
	}
	if len(tensor.Shape) != len(wantShape) {
		t.Fatalf("rank = %d, want %d", len(tensor.Shape), len(wantShape))
	}
	for i := range wantShape {
		if tensor.Shape[i] != wantShape[i] {
			t.Fatalf("shape[%d] = %d, want %d", i, tensor.Shape[i], wantShape[i])
		}
	}
	if len(tensor.F32) != len(want) {
		t.Fatalf("len(F32) = %d, want %d", len(tensor.F32), len(want))
	}
	for i, got := range tensor.F32 {
		diff := got - want[i]
		if diff < -tol || diff > tol {
			t.Fatalf("tensor[%d] = %f, want %f (tol %f)", i, got, want[i], tol)
		}
	}
}

// TestCUDAContrastiveGEMMMatchesHostAcrossShapes is the S1a parity gate for
// the cuBLAS GEMM-based InfoNCE path (runInfoNCEGEMM): tolerance-based (not
// bit-exact -- GEMM reduction order differs from the host's sequential sum)
// agreement with the host reference across batch, candidate-pool, and width
// shapes. It forces EOS_CUDA_CONTRASTIVE_GEMM=1 so the result reflects the
// new path regardless of the process default.
func TestCUDAContrastiveGEMMMatchesHostAcrossShapes(t *testing.T) {
	t.Setenv("EOS_CUDA_CONTRASTIVE_GEMM", "1")
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		t.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	const negativeMultiplier = 1 + 4 // hard-negative shape: own positive + 4 mined negatives per query
	batches := []int{8, 64, 256}
	widths := []int{64, 256}

	for _, batch := range batches {
		for _, width := range widths {
			for _, hardNegative := range []bool{false, true} {
				candidateRows := batch
				if hardNegative {
					candidateRows = batch * negativeMultiplier
				}
				name := fmt.Sprintf("batch%d_width%d_candidates%d", batch, width, candidateRows)
				t.Run(name, func(t *testing.T) {
					query, _ := syntheticContrastiveTensors(batch, width)
					candidates, _ := syntheticContrastiveTensors(candidateRows, width)
					targets := make([]int, batch)
					stride := 1
					if hardNegative {
						stride = negativeMultiplier
					}
					for i := range targets {
						targets[i] = i * stride
					}
					cfg := backend.ContrastiveLossConfig{Temperature: 0.05}

					got, err := accelAny.RunInfoNCEWithTargets(query, candidates, targets, cfg)
					if err != nil {
						t.Fatalf("run infonce: %v", err)
					}
					wantQ, wantC, wantLoss, wantScore := hostInfoNCEGradTargets(query.F32, candidates.F32, batch, candidateRows, width, targets, cfg.Temperature)

					// Per-element gradient tolerance: matches the fixed-shape
					// 0.0005 default (assertTensorClose). Measured max diff
					// across these shapes tops out around 2.4e-5 (batch 256,
					// width 64, 1280 candidates), so this keeps ~20x headroom
					// while still catching a real algebra/summation regression.
					gradTol := float32(0.0005)
					assertTensorCloseTol(t, got.QueryGrads, []int{batch, width}, wantQ, gradTol)
					assertTensorCloseTol(t, got.PositiveGrads, []int{candidateRows, width}, wantC, gradTol)
					assertCloseF32(t, got.LossSum, wantLoss, 0.01)

					// ScoreSum sums batch*candidateRows terms (up to 256*1280
					// here); bound the tolerance relative to its own
					// magnitude instead of a fixed absolute value.
					scoreTol := float32(0.001)*float32(math.Abs(float64(wantScore))) + 0.01
					assertCloseF32(t, got.ScoreSum, wantScore, scoreTol)
				})
			}
		}
	}
}

// TestCUDAContrastiveGEMMFlagDisabledUsesAtomicPath is the rollback check for
// S1a: EOS_CUDA_CONTRASTIVE_GEMM=0 must still route through the retired
// atomic kernels (runInfoNCEAtomic) and match the host reference, so the flag
// is a genuine escape hatch and not dead code.
func TestCUDAContrastiveGEMMFlagDisabledUsesAtomicPath(t *testing.T) {
	t.Setenv("EOS_CUDA_CONTRASTIVE_GEMM", "0")
	accelAny, err := NewContrastiveAccelerator()
	if err != nil {
		t.Fatalf("new contrastive accelerator: %v", err)
	}
	if accelAny == nil {
		t.Skip("no cuda contrastive accelerator available")
	}
	defer accelAny.Close()

	query, candidates := syntheticContrastiveTensors(32, 64)
	cfg := backend.ContrastiveLossConfig{Temperature: 0.05}
	got, err := accelAny.RunInfoNCE(query, candidates, cfg)
	if err != nil {
		t.Fatalf("run infonce: %v", err)
	}
	wantQ, wantP, wantLoss, wantScore := hostInfoNCEGrad(query.F32, candidates.F32, 32, 64, cfg.Temperature)
	assertTensorClose(t, got.QueryGrads, []int{32, 64}, wantQ)
	assertTensorClose(t, got.PositiveGrads, []int{32, 64}, wantP)
	assertCloseF32(t, got.LossSum, wantLoss, 0.0001)
	assertCloseF32(t, got.ScoreSum, wantScore, 0.0001)
}

// TestCUDAContrastiveScoreMatrixMemoryEnvelope is a pure size-arithmetic
// guard, not a real allocation ("a unit test on the size calculation, not an
// actual 4096 allocation" -- S1a). docs/benchmarks.md's Batch Sweep table
// measured 4.47 GB (GiB) max RSS for the *whole* training process at batch
// 4096 with in-batch negatives; this asserts the B x C score-matrix bytes
// (the quadratic-in-batch term this change touches) stay a small, sane
// fraction of that documented total envelope, so a future shape change (for
// example growing the hard-negative multiplier) cannot silently blow past
// it without failing a fast, GPU-free test.
func TestCUDAContrastiveScoreMatrixMemoryEnvelope(t *testing.T) {
	const documentedTotalEnvelopeGiB = 4.47
	const gib = int64(1024) * 1024 * 1024
	const negativeMultiplier = 1 + 4

	cases := []struct {
		name          string
		queryRows     int
		candidateRows int
	}{
		{"in-batch-4096", 4096, 4096},
		{"hard-negative-4096x5", 4096, 4096 * negativeMultiplier},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			elements := tc.queryRows * tc.candidateRows
			scoreBytes, err := checkedCUDABytes("cuda contrastive score matrix", elements, 4)
			if err != nil {
				t.Fatalf("score matrix byte calculation: %v", err)
			}
			// The scores and scales buffers are both allocated at this
			// shape (see runInfoNCEGEMM / runInfoNCEAtomic); count both.
			totalBytes := int64(scoreBytes) * 2
			totalGiB := float64(totalBytes) / float64(gib)
			if totalGiB >= documentedTotalEnvelopeGiB {
				t.Fatalf("score+scale matrix envelope %.4f GiB at batch %d candidates %d meets or exceeds the documented %.2f GiB total training envelope from docs/benchmarks.md", totalGiB, tc.queryRows, tc.candidateRows, documentedTotalEnvelopeGiB)
			}
			t.Logf("batch=%d candidates=%d width-independent score+scale bytes = %.4f GiB (documented total training envelope %.2f GiB)", tc.queryRows, tc.candidateRows, totalGiB, documentedTotalEnvelopeGiB)
		})
	}
}
