// rdbs_tb.cpp
// =============================================================================
// RDBS Testbench (FIXED)
// =============================================================================
// Changes from original:
//   - Input packing: 16-bit integers are ap_fixed<16,1> BIT PATTERNS (FRAC=15)
//   - Output unpacking: 16-bit integers are ap_fixed<16,4> BIT PATTERNS (FRAC=12)
//   - MATLAB must export output with 2^12 scaling (not 2^15)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "rdbs_kernel.h"

#define ERROR_TOLERANCE 2   // max allowed difference in LSBs

int main() {
    printf("====================================================\n");
    printf("  RDBS HLS Testbench\n");
    printf("  M = %d elements, B = %d beams, K = %d snapshots\n",
           M_ELEMENTS, B_BEAMS, K_SNAPSHOTS);
    printf("  Input:  ap_fixed<16,1> (15 frac bits)\n");
    printf("  Output: ap_fixed<16,4> (12 frac bits)\n");
    printf("====================================================\n\n");

    // ---- 1. Read test input ----
    FILE *fin = fopen("test_input.dat", "r");
    if (!fin) {
        printf("ERROR: Cannot open test_input.dat\n");
        printf("       Run the MATLAB script first.\n");
        return 1;
    }

    // Arrays are static to avoid stack overflow with large K
    static int x_real_int[K_SNAPSHOTS][M_ELEMENTS];
    static int x_imag_int[K_SNAPSHOTS][M_ELEMENTS];

    for (int k = 0; k < K_SNAPSHOTS; k++) {
        for (int m = 0; m < M_ELEMENTS; m++) {
            if (fscanf(fin, "%d %d", &x_real_int[k][m], &x_imag_int[k][m]) != 2) {
                printf("ERROR: Unexpected end of test_input.dat at k=%d, m=%d\n", k, m);
                fclose(fin);
                return 1;
            }
        }
    }
    fclose(fin);
    printf("Read %d input samples from test_input.dat\n", M_ELEMENTS * K_SNAPSHOTS);

    // ---- 2. Read expected output ----
    // NOTE: MATLAB now exports output scaled by 2^12 (not 2^15)
    //       because output type is ap_fixed<16,4> with 12 frac bits
    FILE *fref = fopen("test_output.dat", "r");
    if (!fref) {
        printf("ERROR: Cannot open test_output.dat\n");
        return 1;
    }

    static int y_ref_real[K_SNAPSHOTS][B_BEAMS];
    static int y_ref_imag[K_SNAPSHOTS][B_BEAMS];

    for (int k = 0; k < K_SNAPSHOTS; k++) {
        for (int b = 0; b < B_BEAMS; b++) {
            if (fscanf(fref, "%d %d", &y_ref_real[k][b], &y_ref_imag[k][b]) != 2) {
                printf("ERROR: Unexpected end of test_output.dat at k=%d, b=%d\n", k, b);
                fclose(fref);
                return 1;
            }
        }
    }
    fclose(fref);
    printf("Read %d expected output samples from test_output.dat\n\n",
           B_BEAMS * K_SNAPSHOTS);

    // ---- 3. Feed input to kernel ----
    hls::stream<axis_data_t> x_stream("x_in");
    hls::stream<axis_data_t> y_stream("y_out");

    for (int k = 0; k < K_SNAPSHOTS; k++) {
        for (int m = 0; m < M_ELEMENTS; m++) {
            axis_data_t pkt;
            // Pack: these integers ARE the ap_fixed<16,1> bit patterns
            ap_int<16> ri = (ap_int<16>)x_real_int[k][m];
            ap_int<16> ii = (ap_int<16>)x_imag_int[k][m];
            pkt.data.range(31, 16) = ri;
            pkt.data.range(15, 0)  = ii;
            pkt.last = (m == M_ELEMENTS - 1);
            x_stream.write(pkt);
        }
    }
    printf("Input stream loaded: %d packets\n", M_ELEMENTS * K_SNAPSHOTS);

    // ---- 4. Run kernel ----
    printf("Running rdbs_kernel...\n");
    rdbs_kernel(x_stream, y_stream, K_SNAPSHOTS);
    printf("Kernel finished. Output stream has %d packets.\n\n",
           (int)y_stream.size());

    // ---- 5. Compare output ----
    int max_err_real = 0, max_err_imag = 0;
    long long sum_err_sq = 0;
    long long sum_sig_sq = 0;
    int fail_count = 0;
    int total_samples = B_BEAMS * K_SNAPSHOTS;

    printf("Comparing output against MATLAB reference...\n");

    for (int k = 0; k < K_SNAPSHOTS; k++) {
        for (int b = 0; b < B_BEAMS; b++) {
            if (y_stream.empty()) {
                printf("ERROR: Output stream empty at k=%d, b=%d\n", k, b);
                return 1;
            }
            axis_data_t pkt = y_stream.read();

            // Unpack: these 16-bit values are ap_fixed<16,4> bit patterns
            // As signed integers, they equal round(float_value * 2^12)
            int y_real_hls = (int)(short)(ap_int<16>)pkt.data.range(31, 16);
            int y_imag_hls = (int)(short)(ap_int<16>)pkt.data.range(15, 0);

            int err_r = abs(y_real_hls - y_ref_real[k][b]);
            int err_i = abs(y_imag_hls - y_ref_imag[k][b]);

            if (err_r > max_err_real) max_err_real = err_r;
            if (err_i > max_err_imag) max_err_imag = err_i;

            sum_err_sq += (long long)err_r * err_r + (long long)err_i * err_i;
            sum_sig_sq += (long long)y_ref_real[k][b] * y_ref_real[k][b]
                        + (long long)y_ref_imag[k][b] * y_ref_imag[k][b];

            if (err_r > ERROR_TOLERANCE || err_i > ERROR_TOLERANCE) {
                if (fail_count < 10) {
                    printf("  MISMATCH at k=%3d, b=%2d: "
                           "HLS=(%6d,%6d) REF=(%6d,%6d) err=(%d,%d)\n",
                           k, b, y_real_hls, y_imag_hls,
                           y_ref_real[k][b], y_ref_imag[k][b], err_r, err_i);
                }
                fail_count++;
            }

            // Check TLAST
            if (pkt.last != (b == B_BEAMS - 1)) {
                printf("  WARNING: TLAST mismatch at k=%d, b=%d\n", k, b);
            }
        }
    }

    double snr_db = (sum_err_sq > 0) ?
        10.0 * log10((double)sum_sig_sq / (double)sum_err_sq) : 999.0;

    printf("\n====================================================\n");
    printf("  RESULTS\n");
    printf("====================================================\n");
    printf("  Total output samples    : %d\n", total_samples);
    printf("  Max error (real)        : %d LSBs\n", max_err_real);
    printf("  Max error (imag)        : %d LSBs\n", max_err_imag);
    printf("  Signal-to-error ratio   : %.1f dB\n", snr_db);
    printf("  Failed samples (|err|>%d): %d of %d\n",
           ERROR_TOLERANCE, fail_count, total_samples);
    printf("----------------------------------------------------\n");

    if (fail_count == 0) {
        printf("  *** PASS ***\n");
    } else {
        printf("  *** FAIL *** (%d mismatches)\n", fail_count);
    }
    printf("====================================================\n");

    return (fail_count > 0) ? 1 : 0;
}