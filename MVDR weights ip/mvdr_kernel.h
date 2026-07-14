// mvdr_kernel.h
// =============================================================================
// MVDR (Capon) Beamforming Weight Computation — Vitis HLS Kernel
// =============================================================================
//
// Solves:   w = R⁻¹ · a  /  (aᴴ · R⁻¹ · a)
//
// Algorithm: Complex QR decomposition via Givens rotations with CORDIC
// ─────────────────────────────────────────────────────────────────────
//   Phase 1 ─ Load R (21×21) and a (21×1) into on-chip augmented buffer
//             A[N][N+1] = [R | a]  (working type: ap_fixed<48,12>)
//
//   Phase 2 ─ Givens QR elimination (N-1 pivot columns × up to N-1 rows each)
//             For each (p, q) pair with q > p:
//               r   = sqrt(|A[p][p]|² + |A[q][p]|²)  ← CORDIC via hls::sqrtf
//               c   = conj(A[p][p]) / r               ← CORDIC reciprocal
//               s   = conj(A[q][p]) / r
//             Apply unitary rotation [c s; -s* c*] to rows p,q across all cols
//             → A becomes [U | c] where U is upper triangular
//
//   Phase 3 ─ Complex back-substitution: x = U⁻¹ · c  ==>  R⁻¹ · a
//             Diagonal divisions: x[i] = acc * conj(U[i][i]) / |U[i][i]|²
//
//   Phase 4 ─ Normalise: w[i] = x[i] / Re(aᴴ · x)
//             Denominator is real positive since R is Hermitian PD.
//
// Fixed-point types
// ─────────────────
//   Input  R : ap_fixed<32,9>  FRAC=23  range ±256   — matches cov result_t
//   Input  a : ap_fixed<32,9>  FRAC=23  range ±256   — beamspace steering vec
//   Working  : ap_fixed<48,12> FRAC=36  range ±2048  — absorbs Givens growth
//              WHY INT=12: Givens rotations are unitary → row 2-norm preserved.
//              Max initial row norm ≤ √21 × 256 ≈ 1173 → INT=11 min → 12 safe.
//   Output w : ap_fixed<32,9>  FRAC=23  range ±256   — matches MATLAB output
//
// Interface (AXI4 master + AXI-Lite control)
// ─────────────────────────────────────────────────────────────────────────────
//   R_in  [N×N]  — AXI4 master (bundle gmem0), offset=slave
//   a_in  [N]    — AXI4 master (bundle gmem1), offset=slave
//   w_out [N]    — AXI4 master (bundle gmem2), offset=slave
//   return       — AXI-Lite (bundle ctrl)
// =============================================================================
#pragma once

#include <ap_fixed.h>
#include <ap_int.h>

// ─── Dimensions ──────────────────────────────────────────────────────────────
static constexpr int N_BEAMS = 21;      // beamspace channels (= M_CH, = B_BEAMS)

// ─── Fixed-Point Types ───────────────────────────────────────────────────────

// I/O types — MUST match covariance kernel result_t  (ap_fixed<32,9>, FRAC=23)
typedef ap_fixed<32, 9, AP_RND_CONV, AP_SAT> cov_t;    // covariance matrix entry
typedef ap_fixed<32, 9, AP_RND_CONV, AP_SAT> weight_t; // MVDR weight output

// Working type inside QR — wider to absorb Givens accumulation and rotation
// growth without saturation. FRAC=36 keeps sub-LSB precision in coefficients.
typedef ap_fixed<48, 12, AP_RND_CONV, AP_SAT> work_t;

// ─── Complex Structs ─────────────────────────────────────────────────────────
struct ccov_t    { cov_t    re; cov_t    im; }; // complex covariance entry (I/O)
struct cweight_t { weight_t re; weight_t im; }; // complex MVDR weight     (I/O)
struct cwork_t   { work_t   re; work_t   im; }; // complex working entry   (internal)

// ─── Top-Level Kernel ─────────────────────────────────────────────────────────
void mvdr_weights(
    ccov_t    R_in[N_BEAMS][N_BEAMS],  // input : 21×21 Hermitian covariance matrix
    cweight_t a_in[N_BEAMS],           // input : 21×1  beamspace steering vector
    cweight_t w_out[N_BEAMS]           // output: 21×1  MVDR weight vector
);
