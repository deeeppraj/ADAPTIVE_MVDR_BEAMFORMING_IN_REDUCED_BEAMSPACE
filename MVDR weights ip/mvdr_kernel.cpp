// mvdr_kernel.cpp
// =============================================================================
// MVDR Weight Computation — Givens QR + CORDIC Implementation
// =============================================================================
//
// Interface fixed-point types (ap_fixed<32,9>, FRAC=23) are preserved exactly.
// Internal computation uses float32 (sufficient: input FRAC=23 matches float mantissa).
//
//   The covariance matrix R can have condition number ~100–1000 when strong
//   jammers are present (as in our scenario: 2 jammers at 20 dB above noise).
//   The smallest singular value of R can be ~0.004, making intermediate
//   values in back-substitution (1/|U_diag|²) reach ~60,000 — far above
//   any practical ap_fixed INT range without massive type widening.
//
//   Using float32 avoids this entirely. For synthesis, float maps to DSP-backed
//   IEEE-754 SP FP units; hls::sqrtf maps to a CORDIC-based FP core.
//
// Algorithm: Complex QR decomposition via Givens rotations
// ─────────────────────────────────────────────────────────────────────
//   Phase 1: Load R and a into augmented float matrix A[N][N+1]
//   Phase 2: Givens QR — for each pivot col p, zero all rows q > p
//            r = hls::sqrt(|A[p][p]|² + |A[q][p]|²)   ← CORDIC via hls::sqrt
//            c = conj(A[p][p])/r,  s = conj(A[q][p])/r
//            apply [c s; -s* c*] rotation to rows p,q across all cols
//   Phase 3: Complex back-substitution  x = U⁻¹ · c  →  R⁻¹ · a
//   Phase 4: Normalise  w = x / Re(aᴴ · x)
//   Phase 5: Quantise to ap_fixed<32,9> output
// =============================================================================

#include "mvdr_kernel.h"
#include "hls_math.h"   // hls::sqrt → CORDIC-based FP core in synthesis

// ─── CORDIC: Complex pair magnitude ──────────────────────────────────────────
// r = sqrt(|a|² + |b|²) for two complex numbers a, b.
// hls::sqrt maps to CORDIC vectoring mode in synthesis.
// Using float32 arguments — sufficient since input ap_fixed<32,9> FRAC=23
// matches float32's 23-bit mantissa exactly.
static float cordic_cmag_d(float a_re, float a_im,
                            float b_re, float b_im)
{
#pragma HLS INLINE
    float sum_sq = a_re*a_re + a_im*a_im + b_re*b_re + b_im*b_im;
    if (sum_sq <= 0.0f) return 0.0;
    return hls::sqrtf((float)sum_sq);   // CORDIC sqrt, float precision sufficient
}

// ─── Apply one complex Givens rotation to rows p and q ────────────────────────
// Zeroes A[q][p_col] and updates all columns j = p_col..N_BEAMS.
// A is stored as separate real/imaginary arrays for HLS resource efficiency.
//
// Rotation:  G = [c  s; -s*  c*],   c = conj(A[p][p])/r,  s = conj(A[q][p])/r
// Row update: new_p[j] =  c·A[p][j] + s·A[q][j]
//             new_q[j] = -s*·A[p][j] + c*·A[q][j]
static void givens_rotate_d(float Ar[N_BEAMS][N_BEAMS+1],
                            float Ai[N_BEAMS][N_BEAMS+1],
                             int p, int q, int p_col)
{
#pragma HLS INLINE off

    // ── CORDIC: compute rotation radius r ────────────────────────────────
    float r = cordic_cmag_d(Ar[p][p_col], Ai[p][p_col],
                              Ar[q][p_col], Ai[q][p_col]);

    if (r < 1e-15f) return;  // already negligible — skip rotation

    // ── Rotation coefficients ─────────────────────────────────────────────
    // c = conj(A[p][p_col]) / r  →  c_re =  A[p][p].re / r
    //                                c_im = -A[p][p].im / r  (conjugate)
    // s = conj(A[q][p_col]) / r  →  s_re =  A[q][p].re / r
    //                                s_im = -A[q][p].im / r
    float inv_r = 1.0f / r;
    float c_re  =  Ar[p][p_col] * inv_r;
    float c_im  = -Ai[p][p_col] * inv_r;
    float s_re  =  Ar[q][p_col] * inv_r;
    float s_im  = -Ai[q][p_col] * inv_r;

    // ── Apply rotation to all columns j = p_col .. N_BEAMS ───────────────
    // new_p[j] =  c·p + s·q   where c=(c_re,c_im), s=(s_re,s_im)
    // new_q[j] = -s*·p + c*·q  where -s*=(-s_re,s_im), c*=(c_re,-c_im)
GIVENS_COL:
    for (int j = p_col; j <= N_BEAMS; j++) {
#pragma HLS PIPELINE II=1
        float ap_re = Ar[p][j],  ap_im = Ai[p][j];
        float aq_re = Ar[q][j],  aq_im = Ai[q][j];

        // new A[p][j] = c*ap + s*aq
        Ar[p][j] = c_re*ap_re - c_im*ap_im + s_re*aq_re - s_im*aq_im;
        Ai[p][j] = c_re*ap_im + c_im*ap_re + s_re*aq_im + s_im*aq_re;

        // new A[q][j] = -s* * ap + c* * aq
        //   -s* = (-s_re, +s_im),   c* = (c_re, -c_im)
        Ar[q][j] = -s_re*ap_re - s_im*ap_im + c_re*aq_re + c_im*aq_im;
        Ai[q][j] = -s_re*ap_im + s_im*ap_re + c_re*aq_im - c_im*aq_re;
    }
    // Ar[q][p_col] and Ai[q][p_col] are now ~0 by construction.
}

// =============================================================================
// Top-Level Kernel
// =============================================================================
void mvdr_weights(
    ccov_t    R_in[N_BEAMS][N_BEAMS],
    cweight_t a_in[N_BEAMS],
    cweight_t w_out[N_BEAMS])
{
    // ── AXI4 master interfaces ────────────────────────────────────────────
#pragma HLS INTERFACE m_axi port=R_in  offset=slave bundle=gmem0 \
        depth=441 max_read_burst_length=256 latency=64
#pragma HLS INTERFACE m_axi port=a_in  offset=slave bundle=gmem1 \
        depth=21  max_read_burst_length=32  latency=64
#pragma HLS INTERFACE m_axi port=w_out offset=slave bundle=gmem2 \
        depth=21  max_write_burst_length=32 latency=64

    // ── AXI-Lite control ─────────────────────────────────────────────────
#pragma HLS INTERFACE s_axilite port=R_in   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=a_in   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=w_out  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // ─────────────────────────────────────────────────────────────────────
    // PHASE 1 — Load into augmented float matrix A[N][N+1]
    // ─────────────────────────────────────────────────────────────────────
    // Real and imaginary parts kept in separate arrays for clean indexing.
    // A_r[i][0..N-1] = R[i][j].re,   A_r[i][N] = a[i].re
    // A_i[i][0..N-1] = R[i][j].im,   A_i[i][N] = a[i].im
    //
    // dim=1 complete: partitions by row — each of the 21 rows becomes a
    // separate 22-element RAM. givens_rotate_d needs rows p and q accessible
    // simultaneously, which this provides (p and q are always different rows).
    //
    // NOTE: GIVENS_COL achieves II=2 (not II=1) because the 22-element row
    // RAMs have 1R1W ports and HLS conservatively schedules the write one
    // cycle after the read of the same row. This is acceptable:
    //   210 rotations x 22 cols x II=2 = ~9240 cycles for Givens phase.
    // Attempting dim=0 complete causes a 6x instruction explosion (15K->2.7K)
    // because p and q are variable function parameters — HLS builds 21-way
    // MUX trees for all 462 elements, hanging synthesis at >1.9 GB RAM.
    float A_r[N_BEAMS][N_BEAMS+1];
    float A_i[N_BEAMS][N_BEAMS+1];
#pragma HLS ARRAY_PARTITION variable=A_r dim=1 complete
#pragma HLS ARRAY_PARTITION variable=A_i dim=1 complete

    // Save a for Phase 4 normalisation
    float a_r[N_BEAMS], a_i[N_BEAMS];
#pragma HLS ARRAY_PARTITION variable=a_r complete
#pragma HLS ARRAY_PARTITION variable=a_i complete

LOAD_R:
    for (int i = 0; i < N_BEAMS; i++) {
        LOAD_COL:
        for (int j = 0; j < N_BEAMS; j++) {
#pragma HLS PIPELINE II=1
            // ap_fixed<32,9> → float: FRAC=23 matches float32 mantissa exactly
            A_r[i][j] = (float)R_in[i][j].re;
            A_i[i][j] = (float)R_in[i][j].im;
        }
        // Augmented column N: steering vector
        A_r[i][N_BEAMS] = (float)a_in[i].re;
        A_i[i][N_BEAMS] = (float)a_in[i].im;
        // Save for normalisation
        a_r[i] = (float)a_in[i].re;
        a_i[i] = (float)a_in[i].im;
    }

    // ─────────────────────────────────────────────────────────────────────
    // PHASE 2 — Complex QR via Givens Rotations (CORDIC for each rotation)
    // ─────────────────────────────────────────────────────────────────────
    // After this phase:
    //   A[0..N-1][0..N-1] = U  (upper triangular, complex)
    //   A[0..N-1][N]      = c  = Q^H · a  (transformed RHS)
    //
    // Loops are SEQUENTIAL — each rotation reads results from the previous
    // one (true loop-carried dependence on A).  The GIVENS_COL inner loop
    // inside givens_rotate_d is pipelined (II=1).
GIVENS_OUTER:
    for (int p = 0; p < N_BEAMS - 1; p++) {
        GIVENS_INNER:
        for (int q = p + 1; q < N_BEAMS; q++) {
            givens_rotate_d(A_r, A_i, p, q, p);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // PHASE 3 — Complex Back-Substitution:  U·x = c  →  x = R⁻¹·a
    // ─────────────────────────────────────────────────────────────────────
    // For i = N-1 downto 0:
    //   acc = c[i] - Σ_{j>i} U[i][j]·x[j]
    //   x[i] = acc / U[i][i]    ← complex division
    //
    // Complex division:  x = acc/d  =  acc·conj(d) / |d|²
    // Using float avoids the overflow that occurs with fixed-point when
    // |d|² is small (condition number ~100 → |d|_min ~0.005 → 1/|d|² ~40000).
    float x_r[N_BEAMS], x_i[N_BEAMS];
#pragma HLS ARRAY_PARTITION variable=x_r complete
#pragma HLS ARRAY_PARTITION variable=x_i complete

BACK_SUB_OUTER:
    for (int i = N_BEAMS - 1; i >= 0; i--) {

        float acc_re = A_r[i][N_BEAMS];
        float acc_im = A_i[i][N_BEAMS];

        BACK_SUB_INNER:
        for (int j = i + 1; j < N_BEAMS; j++) {
#pragma HLS PIPELINE II=1
            // acc -= U[i][j] · x[j]   (complex multiply-subtract)
            acc_re -= A_r[i][j]*x_r[j] - A_i[i][j]*x_i[j];
            acc_im -= A_r[i][j]*x_i[j] + A_i[i][j]*x_r[j];
        }

        // x[i] = acc / U[i][i]
        // x = acc·conj(d) / |d|²   where d = U[i][i]
        float d_re  = A_r[i][i];
        float d_im  = A_i[i][i];
        float d_sq  = d_re*d_re + d_im*d_im;

        if (d_sq < 1e-30f) {
            // Numerically singular diagonal — zero out weight
            // (Should not happen for a well-estimated HPD covariance)
            x_r[i] = 0.0f;
            x_i[i] = 0.0f;
        } else {
            float inv_dsq = 1.0f / d_sq;
            x_r[i] = (acc_re*d_re + acc_im*d_im) * inv_dsq;
            x_i[i] = (acc_im*d_re - acc_re*d_im) * inv_dsq;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // PHASE 4 — Normalise:  w = x / Re(aᴴ · x)
    // ─────────────────────────────────────────────────────────────────────
    // denom = Re(aᴴ · x) = Σ_i (a_re[i]·x_re[i] + a_im[i]·x_im[i])
    // Real and positive for Hermitian PD R.
    float denom = 0.0f;

NORM_DENOM:
    for (int i = 0; i < N_BEAMS; i++) {
#pragma HLS PIPELINE II=1
        denom += a_r[i]*x_r[i] + a_i[i]*x_i[i];
    }

    float inv_denom = (denom > 1e-30) ? 1.0f / denom : 0.0;

    // ─────────────────────────────────────────────────────────────────────
    // PHASE 5 — Quantise output: float → ap_fixed<32,9>
    // ─────────────────────────────────────────────────────────────────────
    // AP_RND_CONV and AP_SAT applied automatically by the ap_fixed cast.
WRITE_WEIGHTS:
    for (int i = 0; i < N_BEAMS; i++) {
#pragma HLS PIPELINE II=1
        w_out[i].re = weight_t(x_r[i] * inv_denom);
        w_out[i].im = weight_t(x_i[i] * inv_denom);
    }
}
