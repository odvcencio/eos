package eosruntime

import (
	"fmt"

	"m31labs.dev/turboquant"
)

// validateEosQuantizedVectorNorm enforces eos's Norm-carry policy for a
// turboquant.IPQuantized value: turboquant.IPQuantizer.InnerProductPrepared
// multiplies its whole estimate by Norm, so a struct with Norm <= 0 scores as
// the zero vector against every query. That is correct only for a genuine
// zero-vector input.
//
// A zero Norm cannot be told apart from the "forgot to carry Norm" footgun
// (Go's zero value for an omitted float32 field) by inspecting the other
// fields alone: turboquant.IPQuantizer.Quantize on a real zero vector reports
// Norm 0, but at MSE bit widths above 1 its ResNorm is a small nonzero
// dequantization residual, not 0 (verified against turboquant v0.2.0). So the
// caller, who alone knows whether the source vector was a genuine zero
// vector, must state that intent explicitly with zeroVectorAllowed. Every eos
// path that reconstructs an IPQuantized from persisted fields, instead of a
// fresh Quantize call, must validate the result with this function before
// scoring it.
func validateEosQuantizedVectorNorm(qx turboquant.IPQuantized, zeroVectorAllowed bool) error {
	if qx.Norm > 0 {
		return nil
	}
	if qx.Norm == 0 && zeroVectorAllowed {
		return nil
	}
	return fmt.Errorf("eos: quantized vector has non-positive Norm %v; refusing to score a vector that dropped its input-norm field (pass zeroVectorAllowed only for a confirmed genuine zero-vector input)", qx.Norm)
}

// newEosQuantizedVector builds a turboquant.IPQuantized from persisted
// MSE/Signs/ResNorm/Norm fields (for example a future bank or wire format
// that reconstructs vectors rather than calling Quantize directly) and
// rejects the zero-Norm footgun described by validateEosQuantizedVectorNorm.
// Callers must set zeroVectorAllowed only when they know, independently of
// these fields, that the source vector is a genuine zero vector. It does not
// validate MSE/Signs lengths; callers that need that check should also call
// turboquant.ValidateIPQuantized.
func newEosQuantizedVector(mse, signs []byte, resNorm, norm float32, zeroVectorAllowed bool) (turboquant.IPQuantized, error) {
	qx := turboquant.IPQuantized{MSE: mse, Signs: signs, ResNorm: resNorm, Norm: norm}
	if err := validateEosQuantizedVectorNorm(qx, zeroVectorAllowed); err != nil {
		return turboquant.IPQuantized{}, err
	}
	return qx, nil
}
