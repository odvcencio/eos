package eosruntime

import (
	"math"
	"testing"
)

// TestVectorDistillLossAndGradMSEPlusCosineCombined verifies the combined
// MSE+cosine loss on a fixed 2-d input against analytically computed values.
func TestVectorDistillLossAndGradMSEPlusCosineCombined(t *testing.T) {
	// proj = [1, 0], teacher = [0, 1]
	// MSE = (1/2)*[(1-0)^2 + (0-1)^2] = 1.0
	// cosine(proj,teacher) = dot/|proj||teacher| = 0/(1*1) = 0
	// cosine_loss = 1 - 0 = 1.0
	// total = 0.5*1.0 + 0.5*1.0 = 1.0
	proj := []float32{1, 0}
	teacher := []float32{0, 1}
	result, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("VectorDistillLossAndGrad error: %v", err)
	}
	wantLoss := float32(1.0)
	if math.Abs(float64(result.Loss-wantLoss)) > 1e-5 {
		t.Errorf("loss = %f, want %f", result.Loss, wantLoss)
	}

	// Gradient w.r.t. proj[k]:
	// d_total/d_proj[k] = 0.5*(2/d)*(proj[k]-teacher[k]) - 0.5*d_cosine/d_proj[k]
	// For proj=[1,0], teacher=[0,1]:
	//   projNorm=1, teacherNorm=1, denom=1, cosine=0
	//   d_cosine/d_proj[0] = teacher[0]/denom - proj[0]*cosine/projNorm^2 = 0/1 - 1*0/1 = 0
	//   d_cosine/d_proj[1] = teacher[1]/denom - proj[1]*cosine/projNorm^2 = 1/1 - 0*0/1 = 1
	//
	// d_total/d_proj[0] = 0.5*(2/2)*(1-0) - 0.5*0 = 0.5
	// d_total/d_proj[1] = 0.5*(2/2)*(0-1) - 0.5*1 = -0.5 - 0.5 = -1.0
	wantGrad := []float32{0.5, -1.0}
	for i, g := range result.GradProj {
		if math.Abs(float64(g-wantGrad[i])) > 1e-5 {
			t.Errorf("GradProj[%d] = %f, want %f", i, g, wantGrad[i])
		}
	}
}

// TestVectorDistillLossAndGradPerfectMatchIsZero confirms that when the student
// projection exactly matches the teacher, the loss is zero.
func TestVectorDistillLossAndGradPerfectMatchIsZero(t *testing.T) {
	v := []float32{0.6, 0.8} // already unit-norm
	result, err := VectorDistillLossAndGrad(v, v)
	if err != nil {
		t.Fatalf("VectorDistillLossAndGrad error: %v", err)
	}
	if result.Loss > 1e-5 {
		t.Errorf("perfect-match loss = %f, want ~0", result.Loss)
	}
	for i, g := range result.GradProj {
		if math.Abs(float64(g)) > 1e-5 {
			t.Errorf("perfect-match GradProj[%d] = %f, want ~0", i, g)
		}
	}
}

// TestVectorDistillLossAndGradLossDecreasesWithGradientStep confirms that
// taking a gradient step reduces the loss.
func TestVectorDistillLossAndGradLossDecreasesWithGradientStep(t *testing.T) {
	proj := []float32{1, 0, 0}
	teacher := []float32{0, 1, 0}

	result1, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("first loss: %v", err)
	}

	// Gradient descent step
	lr := float32(0.1)
	proj2 := make([]float32, len(proj))
	for i := range proj {
		proj2[i] = proj[i] - lr*result1.GradProj[i]
	}

	result2, err := VectorDistillLossAndGrad(proj2, teacher)
	if err != nil {
		t.Fatalf("second loss: %v", err)
	}
	if result2.Loss >= result1.Loss {
		t.Errorf("loss did not decrease: before=%f after=%f", result1.Loss, result2.Loss)
	}
}

// TestVectorDistillLossAndGradRejectsLengthMismatch verifies the dimension check.
func TestVectorDistillLossAndGradRejectsLengthMismatch(t *testing.T) {
	_, err := VectorDistillLossAndGrad([]float32{1, 0}, []float32{0, 1, 0})
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
}

func TestVectorDistillLossAndGradScratchMatchesPublicExact(t *testing.T) {
	cases := []struct {
		name    string
		proj    []float32
		teacher []float32
	}{
		{name: "cosine", proj: []float32{0.7, -0.2, 0.1, 0.5}, teacher: []float32{-0.3, 0.4, 0.8, -0.1}},
		{name: "zero_proj", proj: []float32{0, 0, 0, 0}, teacher: []float32{-0.3, 0.4, 0.8, -0.1}},
		{name: "zero_teacher", proj: []float32{0.7, -0.2, 0.1, 0.5}, teacher: []float32{0, 0, 0, 0}},
	}
	scratch := make([]float32, 0, 8)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want, err := VectorDistillLossAndGrad(tc.proj, tc.teacher)
			if err != nil {
				t.Fatalf("VectorDistillLossAndGrad: %v", err)
			}
			got, err := vectorDistillLossAndGradInto(tc.proj, tc.teacher, scratch)
			if err != nil {
				t.Fatalf("vectorDistillLossAndGradInto: %v", err)
			}
			if got.Loss != want.Loss {
				t.Fatalf("loss = %v, want exact %v", got.Loss, want.Loss)
			}
			if len(got.GradProj) != len(want.GradProj) {
				t.Fatalf("grad length = %d, want %d", len(got.GradProj), len(want.GradProj))
			}
			for i := range got.GradProj {
				if got.GradProj[i] != want.GradProj[i] {
					t.Fatalf("grad[%d] = %v, want exact %v", i, got.GradProj[i], want.GradProj[i])
				}
			}
			scratch = got.GradProj
		})
	}
}

// TestAccumulateVectorDistillProjectionGradsBackpropIsCorrect verifies that
// the projection back-propagation computes correct d_loss/d_student and d_loss/d_W
// against a finite-difference approximation.
func TestAccumulateVectorDistillProjectionGradsBackpropIsCorrect(t *testing.T) {
	inputDim := 3
	outDim := 4
	// Fixed W, student, teacher
	W := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 0.1, 0.2, 0.3,
	}
	student := []float32{0.6, -0.5, 0.3}
	teacher := []float32{0.2, 0.8, -0.1, 0.5}

	// Forward
	proj := make([]float32, outDim)
	for i := 0; i < inputDim; i++ {
		base := i * outDim
		for k := 0; k < outDim; k++ {
			proj[k] += W[base+k] * student[i]
		}
	}

	lossResult, err := VectorDistillLossAndGrad(proj, teacher)
	if err != nil {
		t.Fatalf("loss: %v", err)
	}

	// Analytic gradients
	gradStudent := make([]float32, inputDim)
	gradW := make([]float32, inputDim*outDim)
	accumulateVectorDistillProjectionGrads(student, lossResult.GradProj, W, gradStudent, gradW)

	// Finite-difference check for gradStudent
	eps := float32(1e-4)
	for i := 0; i < inputDim; i++ {
		studentPlus := append([]float32(nil), student...)
		studentPlus[i] += eps
		projPlus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for k := 0; k < outDim; k++ {
				projPlus[k] += W[base+k] * studentPlus[si]
			}
		}
		rPlus, _ := VectorDistillLossAndGrad(projPlus, teacher)

		studentMinus := append([]float32(nil), student...)
		studentMinus[i] -= eps
		projMinus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for k := 0; k < outDim; k++ {
				projMinus[k] += W[base+k] * studentMinus[si]
			}
		}
		rMinus, _ := VectorDistillLossAndGrad(projMinus, teacher)

		fdGrad := (rPlus.Loss - rMinus.Loss) / (2 * eps)
		if math.Abs(float64(gradStudent[i]-fdGrad)) > 1e-3 {
			t.Errorf("gradStudent[%d] analytic=%f fd=%f", i, gradStudent[i], fdGrad)
		}
	}

	// Finite-difference check for gradW[0] (first row)
	for k := 0; k < outDim; k++ {
		Wplus := append([]float32(nil), W...)
		Wplus[k] += eps
		projPlus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for kk := 0; kk < outDim; kk++ {
				projPlus[kk] += Wplus[base+kk] * student[si]
			}
		}
		rPlus, _ := VectorDistillLossAndGrad(projPlus, teacher)

		Wminus := append([]float32(nil), W...)
		Wminus[k] -= eps
		projMinus := make([]float32, outDim)
		for si := 0; si < inputDim; si++ {
			base := si * outDim
			for kk := 0; kk < outDim; kk++ {
				projMinus[kk] += Wminus[base+kk] * student[si]
			}
		}
		rMinus, _ := VectorDistillLossAndGrad(projMinus, teacher)

		fdGrad := (rPlus.Loss - rMinus.Loss) / (2 * eps)
		if math.Abs(float64(gradW[k]-fdGrad)) > 1e-3 {
			t.Errorf("gradW[0][%d] analytic=%f fd=%f", k, gradW[k], fdGrad)
		}
	}
}

// TestVectorDistillRelationalLossAndGradHandComputedThreeVectorCase checks the
// forward loss value against a hand-computed 3-vector case:
//
//	students: s0=[1,0], s1=[0,1], s2=[1,0]   (s0,s2 identical; s1 orthogonal)
//	teachers: t0=[1,0], t1=[0,1], t2=[0,1]   (t1,t2 identical; t0 orthogonal to both)
//
// Off-diagonal cosine entries (both directions, matrix is symmetric):
//
//	S_student: (0,1)=0 (0,2)=1 (1,2)=0   S_teacher: (0,1)=0 (0,2)=0 (1,2)=1
//	diffs:     (0,1)=0 (0,2)=1 (1,2)=-1
//
// mean((S_student-S_teacher)^2) over all 6 off-diagonal entries
// = (0+0+1+1+1+1)/6 = 2/3, so loss = weight * 2/3.
func TestVectorDistillRelationalLossAndGradHandComputedThreeVectorCase(t *testing.T) {
	students := [][]float32{{1, 0}, {0, 1}, {1, 0}}
	teachers := [][]float32{{1, 0}, {0, 1}, {0, 1}}
	weight := float32(1)

	loss, grads, err := VectorDistillRelationalLossAndGrad(students, teachers, weight)
	if err != nil {
		t.Fatalf("VectorDistillRelationalLossAndGrad error: %v", err)
	}
	wantLoss := float32(2.0 / 3.0)
	if math.Abs(float64(loss-wantLoss)) > 1e-5 {
		t.Errorf("loss = %f, want %f", loss, wantLoss)
	}
	if len(grads) != 3 {
		t.Fatalf("len(grads) = %d, want 3", len(grads))
	}
	// s0 and s2 are identical inputs but are not symmetric in the loss (s0
	// pairs with a mismatched teacher pair (0,2), s2 pairs with a different
	// mismatched teacher pair via (1,2) through s1), so nothing forces their
	// gradients to match; just confirm every component is finite and at
	// least one gradient is non-zero (the term is active).
	var anyNonZero bool
	for _, g := range grads {
		if len(g) != 2 {
			t.Fatalf("grad dim = %d, want 2", len(g))
		}
		for _, v := range g {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("grad component is not finite: %v", grads)
			}
			if v != 0 {
				anyNonZero = true
			}
		}
	}
	if !anyNonZero {
		t.Fatalf("all relational gradients are zero, want at least one non-zero: %v", grads)
	}
}

// TestVectorDistillRelationalLossAndGradFiniteDifferenceCheck verifies the
// analytic per-vector gradient against a central-difference approximation on
// a small (3 vectors, dim 3) synthetic batch with non-axis-aligned values.
func TestVectorDistillRelationalLossAndGradFiniteDifferenceCheck(t *testing.T) {
	students := [][]float32{
		{0.6, -0.3, 0.2},
		{0.1, 0.8, -0.4},
		{-0.5, 0.2, 0.9},
	}
	teachers := [][]float32{
		{0.2, 0.5, -0.1},
		{-0.3, 0.4, 0.6},
		{0.7, -0.2, 0.1},
	}
	weight := float32(0.7)

	_, grads, err := VectorDistillRelationalLossAndGrad(students, teachers, weight)
	if err != nil {
		t.Fatalf("VectorDistillRelationalLossAndGrad error: %v", err)
	}

	eps := float32(1e-3)
	for i := range students {
		for k := range students[i] {
			plus := cloneVectorBatch(students)
			plus[i][k] += eps
			lossPlus, _, err := VectorDistillRelationalLossAndGrad(plus, teachers, weight)
			if err != nil {
				t.Fatalf("loss plus: %v", err)
			}

			minus := cloneVectorBatch(students)
			minus[i][k] -= eps
			lossMinus, _, err := VectorDistillRelationalLossAndGrad(minus, teachers, weight)
			if err != nil {
				t.Fatalf("loss minus: %v", err)
			}

			fdGrad := (lossPlus - lossMinus) / (2 * eps)
			if math.Abs(float64(grads[i][k]-fdGrad)) > 1e-2 {
				t.Errorf("grads[%d][%d] analytic=%f fd=%f", i, k, grads[i][k], fdGrad)
			}
		}
	}
}

// TestVectorDistillRelationalLossAndGradInactiveForSmallBatchOrZeroWeight
// verifies the term is a no-op (loss=0, nil grads, nil error) when the batch
// has fewer than 2 examples or the weight is non-positive, matching the
// "0 disables" contract of --vector-distill-relational-weight.
func TestVectorDistillRelationalLossAndGradInactiveForSmallBatchOrZeroWeight(t *testing.T) {
	single := [][]float32{{1, 0}}
	pair := [][]float32{{1, 0}, {0, 1}}

	if loss, grads, err := VectorDistillRelationalLossAndGrad(single, single, 1); err != nil || loss != 0 || grads != nil {
		t.Fatalf("batch size 1: loss=%v grads=%v err=%v, want inactive", loss, grads, err)
	}
	if loss, grads, err := VectorDistillRelationalLossAndGrad(pair, pair, 0); err != nil || loss != 0 || grads != nil {
		t.Fatalf("zero weight: loss=%v grads=%v err=%v, want inactive", loss, grads, err)
	}
	if loss, grads, err := VectorDistillRelationalLossAndGrad(nil, nil, 1); err != nil || loss != 0 || grads != nil {
		t.Fatalf("empty batch: loss=%v grads=%v err=%v, want inactive", loss, grads, err)
	}
}

// TestVectorDistillRelationalLossAndGradRejectsCountMismatch verifies the
// student/teacher count check.
func TestVectorDistillRelationalLossAndGradRejectsCountMismatch(t *testing.T) {
	students := [][]float32{{1, 0}, {0, 1}}
	teachers := [][]float32{{1, 0}}
	if _, _, err := VectorDistillRelationalLossAndGrad(students, teachers, 1); err == nil {
		t.Fatal("expected error for student/teacher count mismatch, got nil")
	}
}

func cloneVectorBatch(batch [][]float32) [][]float32 {
	out := make([][]float32, len(batch))
	for i, v := range batch {
		out[i] = append([]float32(nil), v...)
	}
	return out
}

func BenchmarkVectorDistillLossAndGradScratch(b *testing.B) {
	proj := make([]float32, 384)
	teacher := make([]float32, 384)
	for i := range proj {
		proj[i] = float32(i%17-8) * 0.01
		teacher[i] = float32(i%23-11) * 0.008
	}
	b.Run("public", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := VectorDistillLossAndGrad(proj, teacher); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("scratch", func(b *testing.B) {
		b.ReportAllocs()
		scratch := make([]float32, 0, len(proj))
		for i := 0; i < b.N; i++ {
			result, err := vectorDistillLossAndGradInto(proj, teacher, scratch)
			if err != nil {
				b.Fatal(err)
			}
			scratch = result.GradProj
		}
	})
}
