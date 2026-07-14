// covariance_matrix.cpp
// =============================================================================
// Sample Covariance Estimation - AXI-Stream input version
// =============================================================================
// Phase 1: Stream-read 21 × 1000 samples from AXI-Stream into on-chip X_buf
// Phase 2: Compute R = (1/K) Y Y^H (upper triangle + Hermitian fill)
// Phase 3: Burst-write 21×21 result to DDR via AXI4 master
// =============================================================================

#include "covariance_matrix.h"

// ─── Complex MAC with conjugate: acc += a * conj(b) ──────────────────────────
static void cmac_conj(accum_t &acc_re, accum_t &acc_im,
                      const samp_t a_re,  const samp_t a_im,
                      const samp_t b_re,  const samp_t b_im)
{
#pragma HLS INLINE
    acc_re += a_re * b_re + a_im * b_im;   // real part of a · conj(b)
    acc_im += a_im * b_re - a_re * b_im;   // imag part of a · conj(b)
}

// ─── Top-level kernel ────────────────────────────────────────────────────────
void covariance_estimation(
    hls::stream<axis_data_t> &y_in,
    cresult_t R[M_CH][M_CH])
{
    // ── Interface pragmas ────────────────────────────────────────────────
#pragma HLS INTERFACE axis port=y_in

#pragma HLS INTERFACE m_axi port=R offset=slave bundle=gmem \
        depth=441 max_write_burst_length=256 latency=64 num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=R      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // ── On-chip buffers ──────────────────────────────────────────────────

    // X_buf: 21 channels × 1000 snapshots
    // Partitioned on dim=1 (channels) so rows i and j are always readable
    // in the same clock cycle during the MAC loop → II=1
    csamp_t X_buf[M_CH][K_SAMP];
#pragma HLS ARRAY_PARTITION variable=X_buf dim=1 complete

    // R_buf: 21×21 output staging (on-chip, avoids per-entry AXI latency)
    cresult_t R_buf[M_CH][M_CH];
#pragma HLS ARRAY_PARTITION variable=R_buf dim=1 complete

    // ── PHASE 1: Stream-read from RDBS into X_buf ────────────────────────
    // Data arrives as: all 21 channels of snapshot 0, then snapshot 1, ...
    // This matches RDBS output order (BEAM_LOOP inside SNAPSHOT_LOOP)
    LOAD_SNAPSHOT:
    for (int t = 0; t < K_SAMP; t++) {
        LOAD_CHANNEL:
        for (int ch = 0; ch < M_CH; ch++) {
#pragma HLS PIPELINE II=1
            axis_data_t pkt = y_in.read();

            // Bit-copy: same unpacking as RDBS uses for packing
            samp_t xr, xi;
            xr.range(15, 0) = pkt.data.range(31, 16);  // real from upper 16
            xi.range(15, 0) = pkt.data.range(15, 0);    // imag from lower 16
            X_buf[ch][t].re = xr;
            X_buf[ch][t].im = xi;
        }
    }

    // ── Normalization constant 1/K (compile-time, no divider hardware) ───
    const accum_t inv_K = accum_t(1.0 / K_SAMP);

    // ── PHASE 2: Compute R = (1/K) Y Y^H ────────────────────────────────
    // Only upper triangle (j >= i), fill lower by Hermitian conjugate
    // 231 (i,j) pairs × 1000 MACs each = 231,000 total MACs
    OUTER_I:
    for (int i = 0; i < M_CH; i++) {
        OUTER_J:
        for (int j = i; j < M_CH; j++) {

            accum_t acc_re = accum_t(0);
            accum_t acc_im = accum_t(0);

            INNER_K:
            for (int t = 0; t < K_SAMP; t++) {
#pragma HLS PIPELINE II=1
                cmac_conj(acc_re, acc_im,
                          X_buf[i][t].re, X_buf[i][t].im,
                          X_buf[j][t].re, X_buf[j][t].im);
            }

            // Normalize: multiply by 1/K instead of dividing
            result_t r_re = result_t(acc_re * inv_K);
            result_t r_im = result_t(acc_im * inv_K);

            // Upper triangle
            R_buf[i][j].re = r_re;
            R_buf[i][j].im = r_im;

            // Lower triangle (Hermitian: R[j][i] = conj(R[i][j]))
            R_buf[j][i].re = r_re;
            R_buf[j][i].im = result_t(-r_im);
        }
    }

    // ── PHASE 3: Burst-write R_buf to DDR ────────────────────────────────
    WRITE_ROW:
    for (int i = 0; i < M_CH; i++) {
        WRITE_COL:
        for (int j = 0; j < M_CH; j++) {
#pragma HLS PIPELINE II=1
            R[i][j] = R_buf[i][j];
        }
    }
}