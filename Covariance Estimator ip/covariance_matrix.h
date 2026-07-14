// covariance_matrix.h
// =============================================================================
// Sample Covariance Estimation - AXI-Stream input from RDBS kernel
// =============================================================================
// Takes RDBS compressed output (21 beams × 1000 snapshots) via AXI-Stream
// Produces 21×21 Hermitian covariance matrix via AXI4 master to DDR
//
// Estimator:  R_hat = (1/K) * Y * Y^H    where Y is the beamspace data
// =============================================================================
#pragma once

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// ─── Dimensions ──────────────────────────────────────────────────────────────
static constexpr int M_CH    = 21;     // beamspace channels (B from RDBS)
static constexpr int K_SAMP  = 1000;   // temporal snapshots

// Upper-triangle entries (Hermitian: only compute j >= i)
static constexpr int UTRI_ENTRIES = M_CH * (M_CH + 1) / 2;  // = 231

// ─── Fixed-point types ───────────────────────────────────────────────────────
//
// samp_t:   ap_fixed<16,4>  range [-8, +8)
//           MUST match RDBS out_data_t exactly (same bit pattern on the wire)
//
// accum_t:  ap_fixed<48,18>
//           Worst case: |sample|^2 * K = 8 * 8 * 1000 = 64000 → needs 17 int bits
//           18 int bits gives range up to 131072, safe headroom
//
// result_t: ap_fixed<32,9>
//           After dividing by K: 64000 / 1000 = 64 → needs 7 int bits
//           9 int bits gives range up to 256, comfortable

typedef ap_fixed<16, 4,  AP_RND_CONV, AP_SAT> samp_t;
typedef ap_fixed<48, 18, AP_RND_CONV, AP_SAT> accum_t;
typedef ap_fixed<32, 9,  AP_RND_CONV, AP_SAT> result_t;

// ─── Complex structs ─────────────────────────────────────────────────────────
struct csamp_t {
    samp_t re;
    samp_t im;
};

struct cresult_t {
    result_t re;
    result_t im;
};

// ─── AXI-Stream packet (IDENTICAL to RDBS kernel's axis_data_t) ──────────────
// bits [31:16] = real component (ap_fixed<16,4> bit pattern)
// bits [15:0]  = imag component (ap_fixed<16,4> bit pattern)
// last         = TLAST, high on the final channel of each snapshot (channel 20)
struct axis_data_t {
    ap_int<32> data;
    bool last;
};

// ─── Top-level kernel ────────────────────────────────────────────────────────
void covariance_estimation(
    hls::stream<axis_data_t> &y_in,    // AXI-Stream from RDBS (21 samples/snapshot)
    cresult_t R[M_CH][M_CH]            // 21×21 output via AXI4 master to DDR
);