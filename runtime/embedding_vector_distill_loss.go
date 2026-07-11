package eosruntime

import (
	"fmt"
	"math"
)

// VectorDistillLossResult holds the scalar loss and d_loss/d_proj gradient.
type VectorDistillLossResult struct {
	Loss     float32
	GradProj []float32 // gradient w.r.t. the projected 384-d vector
}

// VectorDistillLossAndGrad computes the combined MSE + cosine distillation loss
// between a projected student embedding (proj, teacherDim-d) and a teacher
// embedding (teacher, teacherDim-d):
//
//	loss = 0.5 * MSE(proj, teacher) + 0.5 * (1 − cosine(proj, teacher))
//
// Returns the scalar loss and d_loss/d_proj so the caller can back-propagate
// through the linear distillation projection.
func VectorDistillLossAndGrad(proj, teacher []float32) (VectorDistillLossResult, error) {
	return vectorDistillLossAndGradInto(proj, teacher, nil)
}

func vectorDistillLossAndGradInto(proj, teacher, gradScratch []float32) (VectorDistillLossResult, error) {
	if len(proj) == 0 || len(teacher) == 0 {
		return VectorDistillLossResult{}, fmt.Errorf("vector_distill: embeddings must be non-empty")
	}
	if len(proj) != len(teacher) {
		return VectorDistillLossResult{}, fmt.Errorf("vector_distill: proj length %d != teacher length %d", len(proj), len(teacher))
	}
	d := len(proj)

	// --- MSE ---
	var mseLoss float32
	for i := range proj {
		diff := proj[i] - teacher[i]
		mseLoss += diff * diff
	}
	mseLoss /= float32(d)

	// --- cosine ---
	var projSqNorm, teacherSqNorm, dot float64
	for i := range proj {
		projSqNorm += float64(proj[i]) * float64(proj[i])
		teacherSqNorm += float64(teacher[i]) * float64(teacher[i])
		dot += float64(proj[i]) * float64(teacher[i])
	}
	projNorm := float32(math.Sqrt(projSqNorm))
	teacherNorm := float32(math.Sqrt(teacherSqNorm))

	var cosine float32
	hasCosineGrad := false
	if projNorm > 0 && teacherNorm > 0 {
		denom := projNorm * teacherNorm
		cosine = float32(dot) / denom
		// d(cosine)/d(proj[k]) = teacher[k]/denom − proj[k]*cosine/projNorm²
		projScale := cosine / (projNorm * projNorm)
		gradScratch = ensureFloat32Scratch(gradScratch, d)
		for k := range proj {
			gradScratch[k] = teacher[k]/denom - proj[k]*projScale
		}
		hasCosineGrad = true
	}

	// --- combined ---
	// total = 0.5*mse + 0.5*(1−cosine)
	// d_total/d_proj[k]
	//   = 0.5 * d_mse/d_proj[k]  + 0.5 * d(1−cosine)/d_proj[k]
	//   = 0.5 * (2/d)*(proj[k]−teacher[k]) − 0.5 * d_cosine/d_proj[k]
	totalLoss := 0.5*mseLoss + 0.5*(1-cosine)
	gradProj := gradScratch
	if !hasCosineGrad {
		gradProj = ensureFloat32Scratch(gradScratch, d)
	}
	cosineLossScale := float32(0.5)
	for k := range proj {
		mseGrad := float32(1) / float32(d) * (proj[k] - teacher[k]) // 0.5 * 2/d * diff
		var cosGrad float32
		if hasCosineGrad {
			cosGrad = -cosineLossScale * gradProj[k] // -0.5 * d_cosine/d_proj
		}
		gradProj[k] = mseGrad + cosGrad
	}

	return VectorDistillLossResult{Loss: totalLoss, GradProj: gradProj}, nil
}

func ensureFloat32Scratch(s []float32, n int) []float32 {
	if cap(s) < n {
		return make([]float32, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// accumulateVectorDistillProjectionGrads back-propagates through the linear
// distillation projection W (shape [inputDim, teacherDim], stored row-major)
// and accumulates into gradStudent (inputDim) and gradW (inputDim*teacherDim).
//
// forward: proj[k] = Σ_i W[i*teacherDim + k] * student[i]
// backward:
//
//	gradStudent[i] += Σ_k W[i*teacherDim + k] * gradProj[k]
//	gradW[i*teacherDim + k] += student[i] * gradProj[k]
func accumulateVectorDistillProjectionGrads(
	student []float32, // inputDim
	gradProj []float32, // teacherDim — d_loss/d_proj
	W []float32, // inputDim * teacherDim
	gradStudent []float32, // accumulate in-place
	gradW []float32, // accumulate in-place
) {
	if len(gradProj) == 0 || len(student) == 0 {
		return
	}
	inputDim := len(student)
	teacherDim := len(gradProj)
	for i := 0; i < inputDim; i++ {
		si := student[i]
		base := i * teacherDim
		var gs float32
		for k := 0; k < teacherDim; k++ {
			gs += W[base+k] * gradProj[k]
			gradW[base+k] += si * gradProj[k]
		}
		gradStudent[i] += gs
	}
}

// VectorDistillRelationalLossAndGrad computes the opt-in in-batch relational
// (similarity-matrix) distillation term between RAW student pooled vectors
// (pre-projection) and teacher vectors:
//
//	S_student[i][j] = cosine(student[i], student[j])
//	S_teacher[i][j] = cosine(teacher[i], teacher[j])
//	loss = weight * mean_{i != j} (S_student[i][j] - S_teacher[i][j])^2
//
// Unlike the pointwise term, gradients flow directly into the raw student
// vectors (never through the ephemeral projection), so this term supervises
// the geometry actually served at inference time. The term is inactive
// (loss=0, nil grads, nil error) when weight <= 0 or fewer than 2 students are
// given, since an in-batch similarity matrix needs at least 2 rows.
func VectorDistillRelationalLossAndGrad(students, teachers [][]float32, weight float32) (float32, [][]float32, error) {
	return vectorDistillRelationalLossAndGradWithScores(students, teachers, weight, nil, nil)
}

func vectorDistillRelationalLossAndGradWithScores(students, teachers [][]float32, weight float32, studentScores, teacherScores []float32) (float32, [][]float32, error) {
	n := len(students)
	if weight <= 0 || n < 2 {
		return 0, nil, nil
	}
	if len(teachers) != n {
		return 0, nil, fmt.Errorf("vector_distill_relational: student count %d != teacher count %d", n, len(teachers))
	}

	normS := make([]float32, n)
	normT := make([]float32, n)
	for i := 0; i < n; i++ {
		normS[i] = vectorNorm(students[i])
		normT[i] = vectorNorm(teachers[i])
	}

	grads := make([][]float32, n)
	for i := range grads {
		grads[i] = make([]float32, len(students[i]))
	}

	// M counts every off-diagonal entry of the N×N similarity matrix (both
	// S[i][j] and S[j][i], which are equal). Each unordered pair below stands
	// in for both, hence the factor-of-2 forward accumulation and factor-of-4
	// gradient scale (see runtime/embedding_vector_distill_loss_test.go for the
	// derivation check against a hand-computed case).
	m := float32(n * (n - 1))
	var sumSq float32
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sStudent := vectorDistillRelationalScore(students[i], students[j], normS[i], normS[j], studentScores, n, i, j)
			sTeacher := vectorDistillRelationalScore(teachers[i], teachers[j], normT[i], normT[j], teacherScores, n, i, j)
			diff := sStudent - sTeacher
			sumSq += 2 * diff * diff

			scale := weight * 4 * diff / m
			accumulateCosineGradFromScore(students[i], students[j], normS[i], normS[j], sStudent, scale, grads[i], grads[j])
		}
	}
	loss := weight * sumSq / m
	return loss, grads, nil
}

func vectorDistillRelationalScore(left, right []float32, leftNorm, rightNorm float32, scores []float32, rows, i, j int) float32 {
	if len(scores) == rows*rows && rows > 0 && i >= 0 && i < rows && j >= 0 && j < rows && len(left) == len(right) && len(left) > 0 && leftNorm != 0 && rightNorm != 0 {
		return scores[i*rows+j] / (leftNorm * rightNorm)
	}
	return cosineScoreWithNorms(left, right, leftNorm, rightNorm)
}

// applyVectorDistillProjectionAdamW applies an AdamW step to the distillation
// projection weights using the same hyperparams as the student encoder.
func applyVectorDistillProjectionAdamW(
	W, m1, m2, gradW []float32,
	lr, beta1, beta2, eps, weightDecay float32,
	step int,
) {
	if len(W) == 0 || step <= 0 {
		return
	}
	b1 := float64(beta1)
	b2 := float64(beta2)
	corr1 := float32(1 - math.Pow(b1, float64(step)))
	corr2 := float32(1 - math.Pow(b2, float64(step)))
	for i := range W {
		g := gradW[i]
		if weightDecay != 0 {
			g += weightDecay * W[i]
		}
		m1[i] = beta1*m1[i] + (1-beta1)*g
		m2[i] = beta2*m2[i] + (1-beta2)*g*g
		mHat := m1[i]
		vHat := m2[i]
		if corr1 != 0 {
			mHat /= corr1
		}
		if corr2 != 0 {
			vHat /= corr2
		}
		W[i] -= lr * (mHat / (float32(math.Sqrt(float64(vHat))) + eps))
	}
}
