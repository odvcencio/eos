package eosruntime

import (
	"testing"

	"m31labs.dev/turboquant"
)

// TestNewEosQuantizedVectorRejectsDroppedNormFootgun is the negative case for
// eos's Norm-carry policy: a struct with Norm <= 0 that the caller has not
// flagged as a genuine zero vector looks like a hand-built
// turboquant.IPQuantized{...} literal that forgot to carry Norm (Go's zero
// value for an omitted float32 field). It must be rejected, never silently
// scored as 0.
func TestNewEosQuantizedVectorRejectsDroppedNormFootgun(t *testing.T) {
	dim, bitWidth := 8, 4
	mseBytes, signBytes := turboquant.IPQuantizedSizes(dim, bitWidth)
	mse := make([]byte, mseBytes)
	signs := make([]byte, signBytes)
	for i := range signs {
		signs[i] = 0xFF // nonzero payload, as a real non-zero-vector quantization would carry.
	}

	if _, err := newEosQuantizedVector(mse, signs, 1.0 /* ResNorm */, 0 /* Norm */, false /* zeroVectorAllowed */); err == nil {
		t.Fatalf("newEosQuantizedVector accepted Norm=0 without zeroVectorAllowed; the dropped-Norm footgun must be rejected")
	}
	if err := validateEosQuantizedVectorNorm(turboquant.IPQuantized{MSE: mse, Signs: signs, ResNorm: 1.0, Norm: 0}, false); err == nil {
		t.Fatalf("validateEosQuantizedVectorNorm accepted Norm=0 without zeroVectorAllowed")
	}
	if err := validateEosQuantizedVectorNorm(turboquant.IPQuantized{MSE: mse, Signs: signs, ResNorm: 1.0, Norm: -1}, true); err == nil {
		t.Fatalf("validateEosQuantizedVectorNorm accepted a negative Norm even with zeroVectorAllowed")
	}
}

// TestNewEosQuantizedVectorAllowsGenuineZeroVector documents and locks the
// accept path of eos's Norm-carry policy: a genuine zero-vector Norm of 0 is
// accepted when the caller states that intent via zeroVectorAllowed.
// turboquant.IPQuantizer.Quantize on a real zero vector reports Norm 0, but
// (at MSE bit widths above 1) a nonzero ResNorm residual, so this test also
// pins that ResNorm cannot be used to infer "genuinely zero" on its own.
func TestNewEosQuantizedVectorAllowsGenuineZeroVector(t *testing.T) {
	dim, bitWidth := 8, 4
	q := turboquant.NewIPWithSeed(dim, bitWidth, 1)
	qx := q.Quantize(make([]float32, dim)) // the zero vector.
	if qx.Norm != 0 {
		t.Fatalf("precondition failed: Quantize(zero vector).Norm = %v, want 0", qx.Norm)
	}

	got, err := newEosQuantizedVector(qx.MSE, qx.Signs, qx.ResNorm, qx.Norm, true /* zeroVectorAllowed */)
	if err != nil {
		t.Fatalf("newEosQuantizedVector rejected a caller-confirmed genuine zero vector: %v", err)
	}
	if got.Norm != 0 {
		t.Fatalf("newEosQuantizedVector.Norm = %v, want 0", got.Norm)
	}
	if err := validateEosQuantizedVectorNorm(qx, true); err != nil {
		t.Fatalf("validateEosQuantizedVectorNorm rejected a caller-confirmed genuine zero vector: %v", err)
	}
}

// TestValidateEosQuantizedVectorNormAcceptsPositiveNorm covers the ordinary
// accept path: any strictly positive Norm is valid regardless of ResNorm or
// zeroVectorAllowed.
func TestValidateEosQuantizedVectorNormAcceptsPositiveNorm(t *testing.T) {
	if err := validateEosQuantizedVectorNorm(turboquant.IPQuantized{Norm: 1, ResNorm: 0}, false); err != nil {
		t.Fatalf("validateEosQuantizedVectorNorm rejected Norm=1 ResNorm=0: %v", err)
	}
	if err := validateEosQuantizedVectorNorm(turboquant.IPQuantized{Norm: 3.5, ResNorm: 2.1}, false); err != nil {
		t.Fatalf("validateEosQuantizedVectorNorm rejected Norm=3.5 ResNorm=2.1: %v", err)
	}
}
