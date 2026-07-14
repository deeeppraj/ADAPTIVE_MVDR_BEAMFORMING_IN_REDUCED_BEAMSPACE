// rdbs_kernel.h
// =============================================================================
// RDBS Beamspace Compressor - Vitis HLS Kernel (FIXED)
// =============================================================================
#ifndef RDBS_KERNEL_H
#define RDBS_KERNEL_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "params.h"

// ---- Type definitions ----

// Input data: ap_fixed<16,1>  => range [-1, +1), 15 fractional bits
// This matches the MATLAB normalization of raw ADC data to [-1,1)
typedef ap_fixed<16, 1>  in_data_t;

// Coefficients: ap_fixed<16,1> => range [-1, +1), 15 fractional bits
// W^H elements have magnitude 1/sqrt(64) = 0.125, fits easily
typedef ap_fixed<16, 1>  coeff_t;

// Output data: ap_fixed<16,4> => range [-8, +8), 12 fractional bits
// WHY 4 integer bits: the beamformer sums 64 products, each up to
// (1/sqrt(64)) * 1.0 = 0.125. Coherent sum can reach 64 * 0.125 = 8.
// So we need range [-8, +8) => 4 integer bits (includes sign).
typedef ap_fixed<16, 4>  out_data_t;

// Accumulator: wide enough for the full dot product without any loss
typedef ap_fixed<40, 8>  accum_t;

// AXI-Stream packet: 32-bit data word (16 real + 16 imag) + TLAST
struct axis_data_t {
    ap_int<32> data;   // bits [31:16] = real, bits [15:0] = imag
    bool last;
};

// ---- Top-level function ----
void rdbs_kernel(
    hls::stream<axis_data_t> &x_in,
    hls::stream<axis_data_t> &y_out,
    int num_snapshots
);

#endif