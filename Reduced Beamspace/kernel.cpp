// rdbs_kernel.cpp
// =============================================================================
// RDBS Beamspace Compressor - Implementation (FIXED)
// =============================================================================
// Fix 1: Output type is now ap_fixed<16,4> (range [-8,+8)) to handle
//        coherent beamformer gain of up to sqrt(64) = 8
// Fix 2: Data packing uses .range() bit copy instead of cast+divide
// =============================================================================

#include "rdbs.h"
#include "w_coeffs.h"

void rdbs_kernel(
    hls::stream<axis_data_t> &x_in,
    hls::stream<axis_data_t> &y_out,
    int num_snapshots
) {
    #pragma HLS INTERFACE axis port=x_in
    #pragma HLS INTERFACE axis port=y_out
    #pragma HLS INTERFACE s_axilite port=num_snapshots
    #pragma HLS INTERFACE s_axilite port=return

    // Load W^H coefficients (converted from MATLAB float tables once)
    static coeff_t w_real[B_BEAMS][M_ELEMENTS];
    static coeff_t w_imag[B_BEAMS][M_ELEMENTS];
    #pragma HLS ARRAY_PARTITION variable=w_real dim=2 complete
    #pragma HLS ARRAY_PARTITION variable=w_imag dim=2 complete

    static bool initialized = false;
    if (!initialized) {
        INIT_COEFFS:
        for (int b = 0; b < B_BEAMS; b++) {
            for (int m = 0; m < M_ELEMENTS; m++) {
                #pragma HLS PIPELINE II=1
                w_real[b][m] = (coeff_t)W_H_REAL_F[b][m];
                w_imag[b][m] = (coeff_t)W_H_IMAG_F[b][m];
            }
        }
        initialized = true;
    }

    // Local buffer for one snapshot
    in_data_t x_real[M_ELEMENTS];
    in_data_t x_imag[M_ELEMENTS];
    #pragma HLS ARRAY_PARTITION variable=x_real complete
    #pragma HLS ARRAY_PARTITION variable=x_imag complete

    SNAPSHOT_LOOP:
    for (int k = 0; k < num_snapshots; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=129 max=129

        // ---- Read 64 input samples ----
        READ_INPUT:
        for (int m = 0; m < M_ELEMENTS; m++) {
            #pragma HLS PIPELINE II=1
            axis_data_t pkt = x_in.read();

            // BIT COPY: the 16-bit integer on the wire IS the bit pattern
            // of ap_fixed<16,1>. We copy bits directly, no cast or divide.
            in_data_t xr, xi;
            xr.range(15, 0) = pkt.data.range(31, 16);
            xi.range(15, 0) = pkt.data.range(15, 0);
            x_real[m] = xr;
            x_imag[m] = xi;
        }

        // ---- Compute B dot products ----
        BEAM_LOOP:
        for (int b = 0; b < B_BEAMS; b++) {
            #pragma HLS PIPELINE II=1

            accum_t acc_real = 0;
            accum_t acc_imag = 0;

            DOT_PRODUCT:
            for (int m = 0; m < M_ELEMENTS; m++) {
                #pragma HLS UNROLL
                acc_real += w_real[b][m] * x_real[m] - w_imag[b][m] * x_imag[m];
                acc_imag += w_real[b][m] * x_imag[m] + w_imag[b][m] * x_real[m];
            }

            // Truncate accumulator to output type: ap_fixed<16,4>
            // Range [-8, +8) with 12 fractional bits -- fits the beamformer gain
            out_data_t y_r = (out_data_t)acc_real;
            out_data_t y_i = (out_data_t)acc_imag;

            // BIT COPY to AXI stream: the 16-bit ap_fixed<16,4> pattern
            // goes on the wire as-is. The receiver interprets it the same way.
            axis_data_t out_pkt;
            out_pkt.data.range(31, 16) = y_r.range(15, 0);
            out_pkt.data.range(15, 0)  = y_i.range(15, 0);
            out_pkt.last = (b == B_BEAMS - 1);
            y_out.write(out_pkt);
        }
    }
}