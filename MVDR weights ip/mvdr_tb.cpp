// mvdr_tb.cpp
// =============================================================================
// Testbench for mvdr_weights() — File-Driven, MATLAB Golden Reference
// =============================================================================
//
// Reads THREE files produced by golden_reference.m:
//   cov_output.dat   — 441 lines, 21×21 complex R matrix (ap_fixed<32,9>)
//   mvdr_steer.dat   —  21 lines, beamspace SOI steering vector (ap_fixed<32,9>)
//   mvdr_output.dat  —  21 lines, MVDR reference weights  (ap_fixed<32,9>)
//
// File format (all files):
//   One line per complex sample:  "<re_int> <im_int>"
//   re_int = round(real_value × 2^FRAC) — raw ap_fixed<32,9> bit pattern
//   Read with fscanf "%d %d" → signed 32-bit int → bit-copy into ap_fixed.
//
// No dependency on w_coeffs.h, rdbs_params.h, or any RDBS component.
// The beamspace steering vector a_s_bs was already computed by MATLAB as
//   a_s_bs = W_fix' * a_s   (Step 7 of golden_reference.m)
// and written verbatim to mvdr_steer.dat — no recomputation needed here.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "mvdr_kernel.h"

// ─── Fixed-Point Scaling ─────────────────────────────────────────────────────
static constexpr int FRAC_COV   = 23;  // ap_fixed<32,9> for R
static constexpr int FRAC_STEER = 23;  // ap_fixed<32,9> for a_s_bs
static constexpr int FRAC_MVDR  = 23;  // ap_fixed<32,9> for w

// ─── Error Threshold ─────────────────────────────────────────────────────────
// Givens QR on 21×21 complex matrix: ~210 rotation steps, each ±1 LSB of
// work_t (FRAC=36). Projected back to weight_t (FRAC=23): empirically ~10-80 LSBs.
// Threshold set to 128 LSBs = 128 × 2^{-23} ≈ 1.5×10^{-5}.
// Raise to 512 if SNR is still > 30 dB but some weights fail.
static constexpr int    ERR_THRESH_LSB = 128;
static constexpr double ERR_THRESH_VAL = ERR_THRESH_LSB / (double)(1LL << FRAC_MVDR);

// ─── Bit-pattern helpers ──────────────────────────────────────────────────────
// Direct bit-copy: ri is round(value × 2^FRAC) as a signed 32-bit integer.
// Assigning via .range() avoids any floating-point rounding on the way in.
static ccov_t make_ccov(int ri, int ii)
{
    ccov_t c;
    c.re.range(31, 0) = ap_int<32>(ri);
    c.im.range(31, 0) = ap_int<32>(ii);
    return c;
}

static cweight_t make_cweight(int ri, int ii)
{
    cweight_t c;
    c.re.range(31, 0) = ap_int<32>(ri);
    c.im.range(31, 0) = ap_int<32>(ii);
    return c;
}

// =============================================================================
int main()
{
    printf("====================================================\n");
    printf("  MVDR HLS Testbench\n");
    printf("  N_BEAMS = %d\n", N_BEAMS);
    printf("  Input  R     : ap_fixed<32,9>  (FRAC=%d)\n", FRAC_COV);
    printf("  Input  a_s_bs: ap_fixed<32,9>  (FRAC=%d)\n", FRAC_STEER);
    printf("  Output w     : ap_fixed<32,9>  (FRAC=%d)\n", FRAC_MVDR);
    printf("  Error threshold: %d LSBs  (%.2e)\n",
           ERR_THRESH_LSB, ERR_THRESH_VAL);
    printf("====================================================\n\n");

    // ─── 1. Read covariance matrix R from cov_output.dat ─────────────────
    static ccov_t R_in[N_BEAMS][N_BEAMS];

    FILE *fcov = fopen("cov_output.dat", "r");
    if (!fcov) {
        printf("ERROR: Cannot open cov_output.dat\n");
        printf("       Run golden_reference.m first.\n");
        return 1;
    }
    for (int i = 0; i < N_BEAMS; i++) {
        for (int j = 0; j < N_BEAMS; j++) {
            int ri, ii;
            if (fscanf(fcov, "%d %d", &ri, &ii) != 2) {
                printf("ERROR: Unexpected EOF in cov_output.dat at [%d][%d]\n", i, j);
                fclose(fcov);
                return 1;
            }
            R_in[i][j] = make_ccov(ri, ii);
        }
    }
    fclose(fcov);
    printf("Loaded cov_output.dat  — %d×%d complex R (FRAC=%d)\n",
           N_BEAMS, N_BEAMS, FRAC_COV);

    // Sanity: diagonal must be real positive (imag ≈ 0)
    printf("R diagonal (first 5):\n");
    for (int i = 0; i < 5; i++)
        printf("  R[%d][%d] = %+10.6f + j·%+10.6f\n", i, i,
               (double)R_in[i][i].re, (double)R_in[i][i].im);
    printf("\n");

    // ─── 2. Read beamspace steering vector from mvdr_steer.dat ───────────
    // a_s_bs = W_fix' * a_s  — computed and written by golden_reference.m.
    // This is the SOI steering vector already projected into beamspace (21×1).
    // No W matrix, no element-space geometry needed here.
    static cweight_t a_in[N_BEAMS];

    FILE *fsteer = fopen("mvdr_steer.dat", "r");
    if (!fsteer) {
        printf("ERROR: Cannot open mvdr_steer.dat\n");
        printf("       Run golden_reference.m first (it writes this file).\n");
        return 1;
    }
    for (int b = 0; b < N_BEAMS; b++) {
        int ri, ii;
        if (fscanf(fsteer, "%d %d", &ri, &ii) != 2) {
            printf("ERROR: Unexpected EOF in mvdr_steer.dat at b=%d\n", b);
            fclose(fsteer);
            return 1;
        }
        a_in[b] = make_cweight(ri, ii);
    }
    fclose(fsteer);
    printf("Loaded mvdr_steer.dat  — %d-element beamspace steering vector (FRAC=%d)\n",
           N_BEAMS, FRAC_STEER);

    printf("a_s_bs (first 5 entries):\n");
    for (int b = 0; b < 5; b++)
        printf("  a[%d] = %+8.5f + j·%+8.5f  |a|=%.5f\n", b,
               (double)a_in[b].re, (double)a_in[b].im,
               hypot((double)a_in[b].re, (double)a_in[b].im));
    printf("\n");

    // ─── 3. Run MVDR HLS Kernel ───────────────────────────────────────────
    static cweight_t w_hls[N_BEAMS];

    printf("Running mvdr_weights() ...\n");
    mvdr_weights(R_in, a_in, w_hls);
    printf("Kernel done.\n\n");

    // ─── 4. Read MATLAB reference weights from mvdr_output.dat ───────────
    static int ref_re_int[N_BEAMS];
    static int ref_im_int[N_BEAMS];

    FILE *fref = fopen("mvdr_output.dat", "r");
    if (!fref) {
        printf("ERROR: Cannot open mvdr_output.dat\n");
        printf("       Run golden_reference.m first.\n");
        return 1;
    }
    for (int b = 0; b < N_BEAMS; b++) {
        if (fscanf(fref, "%d %d", &ref_re_int[b], &ref_im_int[b]) != 2) {
            printf("ERROR: Unexpected EOF in mvdr_output.dat at b=%d\n", b);
            fclose(fref);
            return 1;
        }
    }
    fclose(fref);
    printf("Loaded mvdr_output.dat — %d reference weights (FRAC=%d)\n\n",
           N_BEAMS, FRAC_MVDR);

    // ─── 5. Compare and print all 21 weights ─────────────────────────────
    const double lsb_val = 1.0 / (double)(1LL << FRAC_MVDR);

    printf("============================================================\n");
    printf("  MVDR WEIGHTS — HLS vs MATLAB  (%d beams)\n", N_BEAMS);
    printf("  idx |    HLS real    HLS imag  |    MAT real    MAT imag  | err_re err_im (LSBs)\n");
    printf("------------------------------------------------------------\n");

    int    max_err_re = 0, max_err_im = 0;
    int    fail_count = 0;
    double sum_err_sq = 0.0, sum_ref_sq = 0.0;

    for (int b = 0; b < N_BEAMS; b++) {

        // Extract HLS output as raw 32-bit signed integer bit-pattern
        int hls_re_int = (int)(ap_int<32>)w_hls[b].re.range(31, 0);
        int hls_im_int = (int)(ap_int<32>)w_hls[b].im.range(31, 0);

        int err_re = abs(hls_re_int - ref_re_int[b]);
        int err_im = abs(hls_im_int - ref_im_int[b]);
        if (err_re > max_err_re) max_err_re = err_re;
        if (err_im > max_err_im) max_err_im = err_im;

        double hls_re = (double)w_hls[b].re;
        double hls_im = (double)w_hls[b].im;
        double ref_re = ref_re_int[b] * lsb_val;
        double ref_im = ref_im_int[b] * lsb_val;

        double err_re_f = hls_re - ref_re;
        double err_im_f = hls_im - ref_im;
        sum_err_sq += err_re_f*err_re_f + err_im_f*err_im_f;
        sum_ref_sq += ref_re*ref_re      + ref_im*ref_im;

        bool fail = (err_re > ERR_THRESH_LSB || err_im > ERR_THRESH_LSB);
        if (fail) fail_count++;

        printf("  w[%2d]%c  %+11.7f  %+11.7f  |  %+11.7f  %+11.7f  |  %5d  %5d  %s\n",
               b, fail ? '!' : ' ',
               hls_re, hls_im, ref_re, ref_im,
               err_re, err_im,
               fail ? "<-- FAIL" : "");
    }
    printf("------------------------------------------------------------\n\n");

    // ─── 6. Phase error per weight ────────────────────────────────────────
    printf("PHASE ERROR PER WEIGHT (degrees):\n");
    for (int b = 0; b < N_BEAMS; b++) {
        double hls_re = (double)w_hls[b].re,  hls_im = (double)w_hls[b].im;
        double ref_re = ref_re_int[b]*lsb_val, ref_im = ref_im_int[b]*lsb_val;

        double ph_hls = atan2(hls_im, hls_re) * 180.0 / M_PI;
        double ph_ref = atan2(ref_im,  ref_re) * 180.0 / M_PI;
        double ph_err = ph_hls - ph_ref;
        while (ph_err >  180.0) ph_err -= 360.0;
        while (ph_err < -180.0) ph_err += 360.0;

        printf("  w[%2d]  |HLS|=%.5f  |REF|=%.5f  Δφ=%+7.3f°\n",
               b, hypot(hls_re, hls_im), hypot(ref_re, ref_im), ph_err);
    }
    printf("\n");

    // ─── 7. Summary ───────────────────────────────────────────────────────
    double snr_db = (sum_err_sq > 0.0)
                    ? 10.0 * log10(sum_ref_sq / sum_err_sq)
                    : 999.0;

    printf("====================================================\n");
    printf("  MVDR RESULTS SUMMARY\n");
    printf("====================================================\n");
    printf("  Max |error| real     : %d LSBs  (%.3e)\n", max_err_re, max_err_re*lsb_val);
    printf("  Max |error| imag     : %d LSBs  (%.3e)\n", max_err_im, max_err_im*lsb_val);
    printf("  Threshold            : %d LSBs  (%.3e)\n", ERR_THRESH_LSB, ERR_THRESH_VAL);
    printf("  Weight vector SNR    : %.2f dB\n", snr_db);
    printf("  Weights failed       : %d / %d\n", fail_count, N_BEAMS);
    printf("----------------------------------------------------\n");
    if (fail_count == 0) {
        printf("  *** PASS ***\n");
    } else {
        printf("  *** FAIL *** (%d weights exceed %d LSB threshold)\n",
               fail_count, ERR_THRESH_LSB);
        if (snr_db > 30.0)
            printf("  SNR=%.1f dB is acceptable — raise ERR_THRESH_LSB to 512.\n", snr_db);
        else
            printf("  Low SNR indicates a real problem (singular R or file order error).\n");
    }
    printf("====================================================\n");

    return (fail_count == 0) ? 0 : 1;
}
