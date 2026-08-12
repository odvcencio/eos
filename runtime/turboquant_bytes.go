package eosruntime

import "m31labs.dev/turboquant"

// turboquantVectorBytesNormFields is the byte cost of the two float32 norm
// fields turboquant.IPQuantized carries alongside its packed MSE and sign
// payloads: Norm (the original input's L2 norm) and ResNorm (the L2 norm of
// the unit-space MSE residual). Both fields are required to reconstruct an
// inner-product estimate; storing only one silently drops Norm and scores
// every reconstructed vector as zero (see turboquant.IPQuantized's doc
// comment). Every eos accounting site that sizes a stored quantized vector
// must add this constant on top of turboquant.IPQuantizedSizes, not a stale
// single-float32 "+4".
const turboquantVectorBytesNormFields = 8

// turboquantVectorBytes returns the total on-disk byte cost of a single
// turboquant.IPQuantized vector for the given dimension and bit width: its
// packed MSE payload, its packed sign payload, and both float32 norm fields
// (Norm and ResNorm). This is the single source of truth for eos's TurboQuant
// storage accounting; call it instead of hand-summing
// turboquant.IPQuantizedSizes plus a literal norm-field byte count.
func turboquantVectorBytes(dim, bitWidth int) int64 {
	mseBytes, signBytes := turboquant.IPQuantizedSizes(dim, bitWidth)
	return int64(mseBytes+signBytes) + turboquantVectorBytesNormFields
}
