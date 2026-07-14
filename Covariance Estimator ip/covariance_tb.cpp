// covariance_tb.cpp
// =============================================================================
// Testbench for covariance_estimation() with AXI-Stream input
// =============================================================================
// Generates 21-channel beamspace data (simulating RDBS output),
// packs it into the same AXI-Stream format as the RDBS kernel,
// feeds it to the covariance kernel, and verifies the output.
//
// Verification:
//   1. Max error vs double-precision reference  (< ERR_THRESH)
//   2. Hermitian symmetry: R[i][j] == conj(R[j][i])
//   3. Diagonal: real part > 0, imaginary part == 0
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "covariance_matrix.h"

using namespace std;
typedef complex<double> cdbl;

// ─── Utility: complex Gaussian sample ────────────────────────────────────────
static cdbl cawgn(mt19937 &rng, double sigma)
{
    uniform_real_distribution<double> U(1e-12, 1.0 - 1e-12);
    const double u1 = U(rng), u2 = U(rng);
    const double r  = sigma * sqrt(-2.0 * log(u1));
    return cdbl(r * cos(2.0 * M_PI * u2),
                r * sin(2.0 * M_PI * u2));
}

// ─── Reference: double-precision covariance R = (1/K) Y Y^H ─────────────────
static void ref_cov(const cdbl Y[M_CH][K_SAMP],
                          cdbl R[M_CH][M_CH])
{
    for (int i = 0; i < M_CH; i++)
        for (int j = 0; j < M_CH; j++) {
            cdbl acc(0.0, 0.0);
            for (int t = 0; t < K_SAMP; t++)
                acc += Y[i][t] * conj(Y[j][t]);
            R[i][j] = acc / (double)K_SAMP;
        }
}

// ─── Print 4×4 sub-block ─────────────────────────────────────────────────────
static void print_block(const char *label, const double blk[4][4])
{
    cout << label << ":\n";
    for (int i = 0; i < 4; i++) {
        cout << "  [";
        for (int j = 0; j < 4; j++)
            cout << setw(10) << fixed << setprecision(4) << blk[i][j];
        cout << " ]\n";
    }
}

// =============================================================================
int main()
{
    mt19937 rng(42);

    // ─── Signal scenario in beamspace ────────────────────────────────────
    // We simulate what RDBS would produce: 21-channel data with
    // a few correlated sources plus noise.
    // For simplicity, we use random steering-like vectors in the
    // 21-dim beamspace rather than computing the full 64-element
    // array model + DFT compression.

    constexpr int N_SRC = 3;
    const double src_pwr[N_SRC]   = { 1.0, 2.0, 1.5 };
    const double noise_var        = 0.1;

    // Random beamspace "steering vectors" (21×1 for each source)
    static cdbl A_bs[M_CH][N_SRC];
    for (int s = 0; s < N_SRC; s++) {
        double norm = 0.0;
        for (int ch = 0; ch < M_CH; ch++) {
            A_bs[ch][s] = cawgn(rng, 1.0);
            norm += std::norm(A_bs[ch][s]);
        }
        // Normalize so each source contributes its specified power
        norm = sqrt(norm);
        for (int ch = 0; ch < M_CH; ch++)
            A_bs[ch][s] /= norm;
    }

    // ─── Generate beamspace data Y [M_CH × K_SAMP] ──────────────────────
    static cdbl    Yd[M_CH][K_SAMP];    // double-precision
    static csamp_t Yq[M_CH][K_SAMP];    // fixed-point (what RDBS would produce)

    cout << "Generating beamspace test data (M_CH=" << M_CH
         << ", K=" << K_SAMP << ", sources=" << N_SRC << ")\n";

    for (int t = 0; t < K_SAMP; t++) {
        cdbl st[N_SRC];
        for (int s = 0; s < N_SRC; s++)
            st[s] = cawgn(rng, sqrt(src_pwr[s] / 2.0));

        for (int ch = 0; ch < M_CH; ch++) {
            cdbl y_ct(0.0, 0.0);
            for (int s = 0; s < N_SRC; s++)
                y_ct += A_bs[ch][s] * st[s];
            y_ct += cawgn(rng, sqrt(noise_var / 2.0));

            Yd[ch][t] = y_ct;
            Yq[ch][t].re = samp_t(y_ct.real());
            Yq[ch][t].im = samp_t(y_ct.imag());
        }
    }

    // ─── Pack into AXI-Stream (same format as RDBS output) ───────────────
    // Order: all 21 channels of snapshot 0, then snapshot 1, ...
    // Packing: bits[31:16] = real, bits[15:0] = imag (ap_fixed<16,4> patterns)
    // TLAST: high on last channel (ch=20) of each snapshot

    hls::stream<axis_data_t> y_stream("y_in");

    cout << "Packing into AXI-Stream (" << M_CH * K_SAMP << " packets)\n";

    for (int t = 0; t < K_SAMP; t++) {
        for (int ch = 0; ch < M_CH; ch++) {
            axis_data_t pkt;
            pkt.data.range(31, 16) = Yq[ch][t].re.range(15, 0);
            pkt.data.range(15, 0)  = Yq[ch][t].im.range(15, 0);
            pkt.last = (ch == M_CH - 1);
            y_stream.write(pkt);
        }
    }

    // ─── Double-precision reference covariance ───────────────────────────
    // Computed from the QUANTIZED data (same numbers the kernel sees)
    // so the comparison isolates accumulation errors, not input quantization
    static cdbl Yd_from_q[M_CH][K_SAMP];
    for (int ch = 0; ch < M_CH; ch++)
        for (int t = 0; t < K_SAMP; t++)
            Yd_from_q[ch][t] = cdbl((double)Yq[ch][t].re, (double)Yq[ch][t].im);

    static cdbl Rd[M_CH][M_CH];
    cout << "Computing double-precision reference R\n";
    ref_cov(Yd_from_q, Rd);

    // ─── Run HLS kernel ──────────────────────────────────────────────────
    static cresult_t Rq[M_CH][M_CH];
    cout << "Running covariance_estimation()\n";
    covariance_estimation(y_stream, Rq);
    cout << "Kernel finished\n\n";

    // ─── Verification ────────────────────────────────────────────────────
    const double ERR_THRESH  = 1e-2;
    const double HERM_THRESH = 1e-7;

    double max_err      = 0.0;
    double max_herm_err = 0.0;
    double max_diag_im  = 0.0;
    double min_diag_re  = 1e30;
    int    n_err        = 0;

    for (int i = 0; i < M_CH; i++) {
        for (int j = 0; j < M_CH; j++) {
            double hls_re = (double)Rq[i][j].re;
            double hls_im = (double)Rq[i][j].im;

            // Error vs reference
            double err = hypot(hls_re - Rd[i][j].real(),
                               hls_im - Rd[i][j].imag());
            if (err > max_err) max_err = err;
            if (err > ERR_THRESH) n_err++;

            // Hermitian check: R[i][j] should equal conj(R[j][i])
            double hr = fabs(hls_re - (double)Rq[j][i].re);
            double hi = fabs(hls_im + (double)Rq[j][i].im);
            double herr = hypot(hr, hi);
            if (herr > max_herm_err) max_herm_err = herr;
        }

        // Diagonal checks
        double d_im = fabs((double)Rq[i][i].im);
        double d_re = (double)Rq[i][i].re;
        if (d_im > max_diag_im) max_diag_im = d_im;
        if (d_re < min_diag_re) min_diag_re = d_re;
    }

    // ─── Print 4×4 comparison ────────────────────────────────────────────
    double blk_hls[4][4], blk_ref[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            blk_hls[i][j] = (double)Rq[i][j].re;
            blk_ref[i][j] = Rd[i][j].real();
        }
    print_block("R_hls top-left 4x4 (real part)", blk_hls);
    cout << "\n";
    print_block("R_ref top-left 4x4 (real part)", blk_ref);

    // ─── Diagonal summary ────────────────────────────────────────────────
    cout << "\nDiagonal (power per channel) - first 8 channels:\n";
    cout << "  Channel   R_hls[i][i].re   R_ref[i][i].re\n";
    for (int i = 0; i < min(8, M_CH); i++) {
        cout << "    " << setw(3) << i
             << "      " << fixed << setprecision(5)
             << setw(10) << (double)Rq[i][i].re
             << "       "
             << setw(10) << Rd[i][i].real() << "\n";
    }

    // ─── Summary ─────────────────────────────────────────────────────────
    cout << "\n========================================================\n";
    cout << "  COVARIANCE KERNEL RESULTS  (M=" << M_CH << ", K=" << K_SAMP << ")\n";
    cout << "========================================================\n";
    cout << fixed << setprecision(6);
    cout << "  Max |R_hls - R_ref|      : " << max_err     << "   (thresh: " << ERR_THRESH << ")\n";
    cout << "  Max Hermitian error       : " << max_herm_err << "   (thresh: " << HERM_THRESH << ")\n";
    cout << "  Max |diag imaginary|      : " << max_diag_im << "   (want: 0)\n";
    cout << "  Min diagonal real         : " << min_diag_re << "   (want: > 0)\n";
    cout << "  Entries exceeding thresh  : " << n_err << "\n";
    cout << "========================================================\n";

    bool pass = (max_err < ERR_THRESH)
             && (max_herm_err < HERM_THRESH)
             && (max_diag_im < 1e-6)
             && (min_diag_re > 0.0)
             && (n_err == 0);

    cout << "\n  " << (pass ? "*** PASS ***" : "*** FAIL ***") << "\n\n";

    return pass ? 0 : 1;
}