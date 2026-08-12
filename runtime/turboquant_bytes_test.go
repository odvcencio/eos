package eosruntime

import (
	"testing"

	"m31labs.dev/turboquant"
)

// TestTurboquantVectorBytesMatchesIPQuantizedSizesPlusNormFields pins the
// turboquantVectorBytes formula against turboquant.IPQuantizedSizes: the
// packed MSE and sign payload sizes, plus 8 bytes for the Norm and ResNorm
// float32 fields turboquant.IPQuantized carries alongside them. A future
// change that drops back to a single 4-byte norm field must fail this test.
func TestTurboquantVectorBytesMatchesIPQuantizedSizesPlusNormFields(t *testing.T) {
	cases := []struct {
		dim      int
		bitWidth int
	}{
		{dim: 8, bitWidth: 1},
		{dim: 8, bitWidth: 2},
		{dim: 128, bitWidth: 2},
		{dim: 128, bitWidth: 4},
		{dim: 128, bitWidth: 8},
		{dim: 256, bitWidth: 4},
		{dim: 3072, bitWidth: 8},
	}
	for _, tc := range cases {
		mseBytes, signBytes := turboquant.IPQuantizedSizes(tc.dim, tc.bitWidth)
		want := int64(mseBytes+signBytes) + 8
		got := turboquantVectorBytes(tc.dim, tc.bitWidth)
		if got != want {
			t.Fatalf("turboquantVectorBytes(%d,%d) = %d, want %d (IPQuantizedSizes mse=%d signs=%d, +8 norm fields)",
				tc.dim, tc.bitWidth, got, want, mseBytes, signBytes)
		}
	}
}

// TestTurboquantVectorBytesMatchesRealQuantizeResultSize pins
// turboquantVectorBytes against the actual serialized size of a real
// q.Quantize(vec) result: len(MSE) + len(Signs) + 8 (Norm and ResNorm). This
// catches drift between the formula and what turboquant.IPQuantizer.Quantize
// actually allocates, not just between the formula and IPQuantizedSizes.
func TestTurboquantVectorBytesMatchesRealQuantizeResultSize(t *testing.T) {
	cases := []struct {
		dim      int
		bitWidth int
	}{
		{dim: 8, bitWidth: 1},
		{dim: 8, bitWidth: 2},
		{dim: 128, bitWidth: 2},
		{dim: 128, bitWidth: 4},
		{dim: 128, bitWidth: 8},
		{dim: 256, bitWidth: 4},
	}
	for _, tc := range cases {
		q := turboquant.NewIPWithSeed(tc.dim, tc.bitWidth, 1)
		vec := make([]float32, tc.dim)
		for i := range vec {
			vec[i] = float32(i%7) - 3
		}
		qx := q.Quantize(vec)
		gotSerialized := int64(len(qx.MSE) + len(qx.Signs) + 8)
		want := turboquantVectorBytes(tc.dim, tc.bitWidth)
		if gotSerialized != want {
			t.Fatalf("dim=%d bitWidth=%d: real Quantize result len(MSE)+len(Signs)+8 = %d, turboquantVectorBytes = %d",
				tc.dim, tc.bitWidth, gotSerialized, want)
		}
	}
}
