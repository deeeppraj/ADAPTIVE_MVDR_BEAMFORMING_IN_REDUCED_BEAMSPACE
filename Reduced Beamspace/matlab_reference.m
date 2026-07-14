%% RDBS Fixed-Point Model & HLS Export
% =========================================================================
% PURPOSE:
%   1. Compute W (DFT submatrix) for +-45 deg scan cone
%   2. Quantize everything to fixed-point matching HLS types
%   3. Export W as a C header file for the HLS kernel ROM
%   4. Generate test vectors (input X, expected output Y) as hex files
%   5. Report quantization error so you know the bit widths are safe
%
% OUTPUTS (saved to ./hls_export/ folder):
%   w_coeffs.h      - C header with W coefficients (HLS ROM)
%   test_input.dat   - 64x129 complex input in hex (I and Q interleaved)
%   test_output.dat  - Bx129 complex expected output in hex
%   rdbs_params.h    - #defines for B, M, K, bit widths
% =========================================================================
clear; clc; close all; rng(42);

%% ========================================================================
%  STEP 1: DECIDE BIT WIDTHS
%  These are the numbers HLS needs. Change here, everything follows.
% =========================================================================
BW_DATA   = 16;   % total bits for input I/Q samples (ADC resolution)
BW_COEFF  = 16;   % total bits for W coefficients
BW_ACCUM  = 32;   % total bits for MAC accumulator
BW_OUT    = 16;   % total bits for output samples
INT_IN    = 1;    % integer bits for input:  ap_fixed<16,1>, range [-1,+1)
INT_OUT   = 4;    % integer bits for output: ap_fixed<16,4>, range [-8,+8)
                  % WHY 4: coherent sum of 64 elements * (1/sqrt(64)) = 8
FRAC_DATA  = BW_DATA  - INT_IN;    % 15 fractional bits (input)
FRAC_OUT   = BW_OUT   - INT_OUT;   % 12 fractional bits (output)
FRAC_COEFF = BW_COEFF - 1;         % 15 fractional bits (coefficients)

%% ========================================================================
%  STEP 2: BUILD THE ARRAY AND SCAN CONE (same as before)
% =========================================================================
Mx = 8; My = 8; M = Mx*My;   % 8x8 UPA, 64 elements
mx = (0:Mx-1)'; my = (0:My-1)';

% Steering vector in (u,v) direction-cosine space
% d = lambda/2 spacing => phase = exp(-j*pi*m*u)
steering_uv = @(u,v) kron( exp(-1j*pi*my*v), exp(-1j*pi*mx*u) );

% Scan cone: +-45 degrees (GESTRA max scan area, Table 2.1 of dissertation)
theta_max = 45 * pi/180;
r_cone    = sin(theta_max);   % = 0.707

% DFT beam grid: 64 beams, centers at u = 2k/Mx, k = -4..3
ku = (-Mx/2 : Mx/2-1);  u_grid = 2*ku/Mx;  % [-1 -0.75 -0.5 ... 0.75]
kv = (-My/2 : My/2-1);  v_grid = 2*kv/My;
[Ug, Vg] = meshgrid(u_grid, v_grid);

% Keep beams whose center falls inside the scan cone circle
inside = (Ug.^2 + Vg.^2) <= r_cone^2;
u_sel  = Ug(inside);
v_sel  = Vg(inside);
B      = numel(u_sel);        % number of kept beams

fprintf('Scan cone: +/- %d deg => circle radius %.3f in u-v space\n', ...
    round(theta_max*180/pi), r_cone);
fprintf('Beams kept: B = %d out of %d\n', B, M);
fprintf('Reduction ratio: %.1f%%\n\n', 100*B/M);

%% ========================================================================
%  STEP 3: BUILD W (floating point, then quantize)
% =========================================================================
% Each column of W is one normalized DFT basis vector
W_float = zeros(M, B);
for b = 1:B
    W_float(:,b) = steering_uv(u_sel(b), v_sel(b)) / sqrt(M);
end

% Quantize W to fixed-point: round to nearest 2^(-FRAC_COEFF)
W_fix = round(W_float * 2^FRAC_COEFF) / 2^FRAC_COEFF;

% Report coefficient quantization error
coeff_err = max(abs(W_float(:) - W_fix(:)));
fprintf('W coefficient quantization:\n');
fprintf('  Bit width: %d bits (1 sign + %d fractional)\n', BW_COEFF, FRAC_COEFF);
fprintf('  Max error: %.6e (LSB = %.6e)\n', coeff_err, 2^(-FRAC_COEFF));
fprintf('  Error in dB below full scale: %.1f dB\n\n', 20*log10(coeff_err));

%% ========================================================================
%  STEP 4: GENERATE TEST INPUT X (64x129, fixed-point)
% =========================================================================
K = 1000;   % snapshots (increased for better covariance estimation downstream)

% Signal scenario (same as your model)
theta_s  = 20*pi/180;  phi_s  =  30*pi/180;
theta_i1 = 35*pi/180;  phi_i1 = -45*pi/180;
theta_i2 = 15*pi/180;  phi_i2 = 120*pi/180;
jammer_amp = 10;
fs_sig = 1000;  t = (0:K-1)/fs_sig;

a_s  = steering_uv(sin(theta_s)*cos(phi_s),   sin(theta_s)*sin(phi_s));
a_i1 = steering_uv(sin(theta_i1)*cos(phi_i1), sin(theta_i1)*sin(phi_i1));
a_i2 = steering_uv(sin(theta_i2)*cos(phi_i2), sin(theta_i2)*sin(phi_i2));

s  = cos(2*pi*50*t);
i1 = sqrt(1/2)*(randn(1,K)+1j*randn(1,K));
i2 = sqrt(1/2)*(randn(1,K)+1j*randn(1,K));
noise = sqrt(1/2)*(randn(M,K)+1j*randn(M,K));
X_float = a_s*s + jammer_amp*a_i1*i1 + jammer_amp*a_i2*i2 + noise;

% Normalize X to [-1,1) range for fixed-point representation
X_scale = max(abs(X_float(:))) * 1.1;   % 10% headroom to avoid clipping
X_norm  = X_float / X_scale;
X_fix   = round(X_norm * 2^FRAC_DATA) / 2^FRAC_DATA;

% Clip any stragglers (shouldn't happen with 10% headroom, but safety)
% max_val = (2^(BW_DATA-1) - 1) / 2^FRAC_DATA;
% X_fix   = max(min(X_fix, max_val), -max_val);
% max_val = (2^(BW_DATA-1) - 1) / 2^FRAC_DATA;
% X_fix   = complex(max(min(real(X_fix), max_val), -max_val), ...
%                   max(min(imag(X_fix), max_val), -max_val));

fprintf('Input data:\n');
fprintf('  Scale factor: %.4f (divide raw ADC by this to normalize)\n', X_scale);
fprintf('  Max |X_fix|: %.6f (should be < 1.0)\n', max(abs(X_fix(:))));
fprintf('  Bit width: %d bits\n\n', BW_DATA);

%% ========================================================================
%  STEP 5: COMPUTE EXPECTED OUTPUT Y = W_fix' * X_fix (golden reference)
% =========================================================================
% This is EXACTLY what HLS must reproduce.
% W_fix' is BxM (conjugate transpose), X_fix is Mx129 => Y is Bx129
Y_float_ref = W_fix' * X_fix;   % using fixed-point W and X

% Quantize output to match HLS output type: ap_fixed<16,4>, 12 frac bits
Y_fix = round(Y_float_ref * 2^FRAC_OUT) / 2^FRAC_OUT;

fprintf('Output data:\n');
fprintf('  Y size: %d x %d (B x K)\n', B, K);
fprintf('  Max |Y_fix|: %.6f (must be < 8.0 for ap_fixed<16,4>)\n\n', max(abs(Y_fix(:))));

%% ========================================================================
%  STEP 6: QUANTIZATION QUALITY CHECK
% =========================================================================
% Compare floating-point RDBS output vs fixed-point output
Y_ideal = W_float' * X_norm;   % pure floating-point reference
quant_error = Y_ideal - Y_fix;
snr_quant = 10*log10( mean(abs(Y_ideal(:)).^2) / mean(abs(quant_error(:)).^2) );

fprintf('=== QUANTIZATION QUALITY ===\n');
fprintf('  Signal-to-Quantization-Noise Ratio: %.1f dB\n', snr_quant);
fprintf('  (Should be >> thermal SNR floor, typically want > 60 dB)\n');
fprintf('  Max absolute error: %.6e\n\n', max(abs(quant_error(:))));

fprintf('DEBUG: X_float is complex? %d\n', ~isreal(X_float));
fprintf('DEBUG: X_norm  is complex? %d\n', ~isreal(X_norm));
fprintf('DEBUG: X_fix   is complex? %d\n', ~isreal(X_fix));
fprintf('DEBUG: X_fix(1,1) = %.6f + %.6fi\n', real(X_fix(1,1)), imag(X_fix(1,1)));
fprintf('DEBUG: X_fix(2,1) = %.6f + %.6fi\n', real(X_fix(2,1)), imag(X_fix(2,1)));
fprintf('DEBUG: max real = %.6f, max imag = %.6f\n', max(real(X_fix(:))), max(imag(X_fix(:))));
fprintf('DEBUG: min real = %.6f, min imag = %.6f\n', min(real(X_fix(:))), min(imag(X_fix(:))));

%% ========================================================================
%  STEP 7: EXPORT FILES FOR HLS
% =========================================================================
outdir = './hls_export';
if ~exist(outdir,'dir'), mkdir(outdir); end

% ---- 7a: Parameters header ----
fid = fopen(fullfile(outdir, 'rdbs_params.h'), 'w');
fprintf(fid, '#ifndef RDBS_PARAMS_H\n#define RDBS_PARAMS_H\n\n');
fprintf(fid, '// Auto-generated by MATLAB fixed-point model\n');
fprintf(fid, '// Scan cone: +/- %d degrees\n\n', round(theta_max*180/pi));
fprintf(fid, '#define M_ELEMENTS   %d    // total antenna elements\n', M);
fprintf(fid, '#define B_BEAMS      %d    // kept beams after RDBS\n', B);
fprintf(fid, '#define K_SNAPSHOTS  %d    // time snapshots\n', K);
fprintf(fid, '#define BW_DATA      %d    // input bit width\n', BW_DATA);
fprintf(fid, '#define BW_OUT       %d    // output bit width\n', BW_OUT);
fprintf(fid, '#define BW_COEFF     %d    // coefficient bit width\n', BW_COEFF);
fprintf(fid, '#define BW_ACCUM     %d    // accumulator bit width\n', BW_ACCUM);
fprintf(fid, '#define INT_IN       %d    // input integer bits (ap_fixed<16,%d>)\n', INT_IN, INT_IN);
fprintf(fid, '#define INT_OUT      %d    // output integer bits (ap_fixed<16,%d>)\n', INT_OUT, INT_OUT);
fprintf(fid, '#define FRAC_DATA    %d    // fractional bits for input\n', FRAC_DATA);
fprintf(fid, '#define FRAC_OUT     %d    // fractional bits for output\n', FRAC_OUT);
fprintf(fid, '#define FRAC_COEFF   %d    // fractional bits for coeffs\n', FRAC_COEFF);
fprintf(fid, '#define X_SCALE      %.10f // multiply HLS output by this to get volts\n', X_scale);
fprintf(fid, '\n#endif\n');
fclose(fid);

% ---- 7b: W coefficients header (the ROM) ----
% Layout: W_H[b][m] = conj(W[m][b]) since HLS computes W^H * x
% Export as FLOAT LITERALS -- ap_fixed<16,1> constructor converts them
% correctly at compile time. Never export raw integers into ap_fixed.
fid = fopen(fullfile(outdir, 'w_coeffs.h'), 'w');
fprintf(fid, '#ifndef W_COEFFS_H\n#define W_COEFFS_H\n\n');
fprintf(fid, '#include "rdbs_params.h"\n\n');
fprintf(fid, '// W_H[b][m] = conj(W[m][b])\n');
fprintf(fid, '// Generated from MATLAB fixed-point model\n');
fprintf(fid, '// Values are the quantized floats; ap_fixed converts at compile time\n\n');

W_H = W_fix';   % B x M (conjugate transpose)

fprintf(fid, 'static const float W_H_REAL_F[B_BEAMS][M_ELEMENTS] = {\n');
for b = 1:B
    fprintf(fid, '  {');
    for m = 1:M
        if m < M, fprintf(fid, '%.8ff, ', real(W_H(b,m)));
        else,     fprintf(fid, '%.8ff',   real(W_H(b,m)));
        end
    end
    if b < B, fprintf(fid, '},\n');
    else,     fprintf(fid, '}\n');
    end
end
fprintf(fid, '};\n\n');

fprintf(fid, 'static const float W_H_IMAG_F[B_BEAMS][M_ELEMENTS] = {\n');
for b = 1:B
    fprintf(fid, '  {');
    for m = 1:M
        if m < M, fprintf(fid, '%.8ff, ', imag(W_H(b,m)));
        else,     fprintf(fid, '%.8ff',   imag(W_H(b,m)));
        end
    end
    if b < B, fprintf(fid, '},\n');
    else,     fprintf(fid, '}\n');
    end
end
fprintf(fid, '};\n\n#endif\n');
fclose(fid);

% ---- 7c: Test input (X) ----
% Format: one line per sample, "real_hex imag_hex" (16-bit signed integers)
% Reading order: column-major (all 64 elements of snapshot 0, then snapshot 1, ...)
% This is the streaming order: the FPGA receives one snapshot at a time
fid = fopen(fullfile(outdir, 'test_input.dat'), 'w');
for k = 1:K
    for m = 1:M
        ri = round(real(X_fix(m,k)) * 2^FRAC_DATA);
        ii = round(imag(X_fix(m,k)) * 2^FRAC_DATA);
        fprintf(fid, '%d %d\n', ri, ii);
    end
end
fclose(fid);

% ---- 7d: Expected output (Y) ----
% Same format: one line per sample, column-major
% IMPORTANT: scaled by 2^FRAC_OUT (=12), NOT 2^FRAC_DATA (=15)
% because output type is ap_fixed<16,4>, which has 12 fractional bits
fid = fopen(fullfile(outdir, 'test_output.dat'), 'w');
for k = 1:K
    for b = 1:B
        ri = round(real(Y_fix(b,k)) * 2^FRAC_OUT);
        ii = round(imag(Y_fix(b,k)) * 2^FRAC_OUT);
        fprintf(fid, '%d %d\n', ri, ii);
    end
end
fclose(fid);

fprintf('=== FILES EXPORTED to %s/ ===\n', outdir);
fprintf('  rdbs_params.h   - #defines for HLS\n');
fprintf('  w_coeffs.h      - W^H coefficient ROM (%d x %d entries)\n', B, M);
fprintf('  test_input.dat  - %d lines (M*K = %d x %d)\n', M*K, M, K);
fprintf('  test_output.dat - %d lines (B*K = %d x %d)\n', B*K, B, K);
fprintf('\nDone. These files go into your Vitis HLS project.\n');

% Right after fclose(fid) for test_input.dat:
fid_check = fopen(fullfile(outdir, 'test_input.dat'), 'r');
line1 = fgetl(fid_check);
line2 = fgetl(fid_check);
line3 = fgetl(fid_check);
fclose(fid_check);
fprintf('DEBUG: File written to: %s\n', fullfile(outdir, 'test_input.dat'));
fprintf('DEBUG: Line 1: %s\n', line1);
fprintf('DEBUG: Line 2: %s\n', line2);
fprintf('DEBUG: Line 3: %s\n', line3);