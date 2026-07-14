//go:build linux && cgo

package cuda

import (
	"math"
	"os"
	"strings"
	"testing"
)

func TestBERTAttentionContextKernelUsesCooperativeSharedLogits(t *testing.T) {
	source := bertAttentionContextKernelSource
	for _, want := range []string{
		"__shared__ double logits[512];",
		"__shared__ float qvec[32];",
		"int job = blockIdx.x;",
		"for (int key_token = threadIdx.x; key_token < tokens; key_token += blockDim.x)",
		"logits[key_token] = logit;",
		"double prob = exp(logits[key_token] - max_logit) / sum_exp;",
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("attention context kernel source missing %q", want)
		}
	}
	if strings.Contains(source, "int job = blockIdx.x * blockDim.x + threadIdx.x") {
		t.Fatal("attention context kernel still uses one-thread-per-row job mapping")
	}
}

func TestBERTAttentionContextKernelFailsClosedForUnsupportedSharedShape(t *testing.T) {
	source := bertAttentionContextKernelSource
	for _, want := range []string{
		"batch <= 0",
		"tokens > 512",
		"head_dim > 32",
		"hidden != heads * head_dim",
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("attention context kernel source missing fail-closed guard %q", want)
		}
	}
	if strings.Contains(source, "batch > 64") {
		t.Fatal("attention context kernel still rejects batch > 64")
	}
}

func TestBERTAttentionContextBlockSizeSelection(t *testing.T) {
	for _, tt := range []struct {
		tokens int
		want   uint
	}{
		{tokens: 1, want: 128},
		{tokens: 256, want: 128},
		{tokens: 257, want: 256},
		{tokens: 512, want: 256},
	} {
		if got := bertAttentionContextBlockSize(tt.tokens); got != tt.want {
			t.Fatalf("bertAttentionContextBlockSize(%d)=%d want %d", tt.tokens, got, tt.want)
		}
	}
}

func TestBGEHiddenMetadataReportsAttentionKernelVariant(t *testing.T) {
	source := readNativeLinuxSourceForTest(t)
	for _, want := range []string{
		`"attention_kernel_variant":            "cooperative_shared_logit_v2"`,
		`"attention_kernel_block_size":         int(bertAttentionContextBlockSize(tokens))`,
		`"attention_kernel_max_tokens":         512`,
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("native source missing hidden attention metadata %q", want)
		}
	}
}

func TestBERTAttentionContextKernelEdgeMaskMatrix(t *testing.T) {
	rt, err := newDeviceRuntime()
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer rt.close()

	for _, tc := range []struct {
		name   string
		batch  int
		tokens int
		hidden int
		heads  int
		mask   func(batch, tokens int) []int32
	}{
		{
			name:   "T1",
			batch:  1,
			tokens: 1,
			hidden: 8,
			heads:  2,
			mask: func(batch, tokens int) []int32 {
				return filledAttentionMask(batch, tokens, 1)
			},
		},
		{
			name:   "partial_mask",
			batch:  1,
			tokens: 7,
			hidden: 8,
			heads:  2,
			mask: func(batch, tokens int) []int32 {
				mask := filledAttentionMask(batch, tokens, 1)
				mask[1], mask[4] = 0, 0
				return mask
			},
		},
		{
			name:   "all_key_masked_row",
			batch:  2,
			tokens: 5,
			hidden: 8,
			heads:  2,
			mask: func(batch, tokens int) []int32 {
				mask := filledAttentionMask(batch, tokens, 1)
				for i := 0; i < tokens; i++ {
					mask[tokens+i] = 0
				}
				return mask
			},
		},
		{
			name:   "batch_gt_64_short_sequence",
			batch:  65,
			tokens: 2,
			hidden: 8,
			heads:  2,
			mask: func(batch, tokens int) []int32 {
				return filledAttentionMask(batch, tokens, 1)
			},
		},
		{
			name:   "T512",
			batch:  1,
			tokens: 512,
			hidden: 8,
			heads:  2,
			mask: func(batch, tokens int) []int32 {
				return filledAttentionMask(batch, tokens, 1)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			elements := tc.batch * tc.tokens * tc.hidden
			query := deterministicAttentionValues(elements, 3, 0.017)
			key := deterministicAttentionValues(elements, 5, -0.011)
			value := deterministicAttentionValues(elements, 7, 0.013)
			mask := tc.mask(tc.batch, tc.tokens)
			want := hostBERTAttentionContextForTest(query, key, value, mask, tc.batch, tc.tokens, tc.hidden, tc.heads)

			qBuf, err := rt.uploadFloat32(query)
			if err != nil {
				t.Fatalf("upload query: %v", err)
			}
			defer rt.freeBuffer(qBuf)
			kBuf, err := rt.uploadFloat32(key)
			if err != nil {
				t.Fatalf("upload key: %v", err)
			}
			defer rt.freeBuffer(kBuf)
			vBuf, err := rt.uploadFloat32(value)
			if err != nil {
				t.Fatalf("upload value: %v", err)
			}
			defer rt.freeBuffer(vBuf)
			maskBuf, err := rt.uploadInt32(mask)
			if err != nil {
				t.Fatalf("upload mask: %v", err)
			}
			defer rt.freeBuffer(maskBuf)
			outBuf, err := rt.allocFloat32(elements)
			if err != nil {
				t.Fatalf("alloc output: %v", err)
			}
			defer rt.freeBuffer(outBuf)

			if err := rt.runBERTAttentionContextDevice(qBuf, kBuf, vBuf, maskBuf, outBuf, tc.batch, tc.tokens, tc.hidden, tc.heads); err != nil {
				t.Fatalf("run attention context: %v", err)
			}
			got := make([]float32, elements)
			if err := rt.downloadFloat32(got, outBuf); err != nil {
				t.Fatalf("download output: %v", err)
			}
			maxAbs := float32(0)
			maxIdx := -1
			for i := range got {
				if !isFinite32(got[i]) {
					t.Fatalf("output[%d]=%g is not finite", i, got[i])
				}
				d := got[i] - want[i]
				if d < 0 {
					d = -d
				}
				if d > maxAbs {
					maxAbs, maxIdx = d, i
				}
			}
			if maxAbs > 1e-4 {
				t.Fatalf("max abs=%g at %d, want <=1e-4 got=%g want=%g", maxAbs, maxIdx, got[maxIdx], want[maxIdx])
			}
			t.Logf("attention edge case %s max_abs=%g finite_outputs=%d", tc.name, maxAbs, len(got))
		})
	}
}

func readNativeLinuxSourceForTest(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("native_linux.go")
	if err != nil {
		t.Fatalf("read native_linux.go: %v", err)
	}
	return string(data)
}

func filledAttentionMask(batch, tokens int, value int32) []int32 {
	mask := make([]int32, batch*tokens)
	for i := range mask {
		mask[i] = value
	}
	return mask
}

func deterministicAttentionValues(elements, multiplier int, offset float32) []float32 {
	values := make([]float32, elements)
	for i := range values {
		centered := float32((i*multiplier)%23 - 11)
		values[i] = centered*0.03125 + offset
	}
	return values
}

func hostBERTAttentionContextForTest(q, k, v []float32, mask []int32, batch, tokens, hidden, heads int) []float32 {
	out := make([]float32, batch*tokens*hidden)
	headDim := hidden / heads
	scale := 1 / math.Sqrt(float64(headDim))
	for b := 0; b < batch; b++ {
		for queryToken := 0; queryToken < tokens; queryToken++ {
			queryRow := b*tokens + queryToken
			for head := 0; head < heads; head++ {
				headBase := head * headDim
				logits := make([]float64, tokens)
				maxLogit := math.Inf(-1)
				active := 0
				for keyToken := 0; keyToken < tokens; keyToken++ {
					logit := math.Inf(-1)
					if mask[b*tokens+keyToken] != 0 {
						keyRow := b*tokens + keyToken
						dot := 0.0
						for d := 0; d < headDim; d++ {
							dot += float64(q[queryRow*hidden+headBase+d]) * float64(k[keyRow*hidden+headBase+d])
						}
						logit = dot * scale
						if logit > maxLogit {
							maxLogit = logit
						}
						active++
					}
					logits[keyToken] = logit
				}
				if active == 0 {
					continue
				}
				sumExp := 0.0
				for keyToken := 0; keyToken < tokens; keyToken++ {
					if mask[b*tokens+keyToken] != 0 {
						sumExp += math.Exp(logits[keyToken] - maxLogit)
					}
				}
				for d := 0; d < headDim; d++ {
					acc := 0.0
					for keyToken := 0; keyToken < tokens; keyToken++ {
						if mask[b*tokens+keyToken] == 0 {
							continue
						}
						keyRow := b*tokens + keyToken
						prob := math.Exp(logits[keyToken]-maxLogit) / sumExp
						acc += prob * float64(v[keyRow*hidden+headBase+d])
					}
					out[queryRow*hidden+headBase+d] = float32(acc)
				}
			}
		}
	}
	return out
}

func isFinite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}
