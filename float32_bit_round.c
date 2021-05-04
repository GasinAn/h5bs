// Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).
// Warning: undefined behavior when delta_exponent >= 32.
// Warning: if val or g_max is NAN, the output may not be NAN.

// Include math.h and stdio.h just for testing.
#include <math.h>
#include <stdio.h>

#define HEX_00400000 4194304
#define HEX_7F800000 2139095040
#define HEX_80000000 -2147483648
#define HEX_FF800000 -8388608

float float32_bit_round(float val, float g_max){
    unsigned int *p_val = (int*) &val;
    unsigned int *p_g_max = (int*) &g_max;

    int exponent_val = *p_val & HEX_7F800000;
    int exponent_g_max = *p_g_max & HEX_7F800000;

    int delta_exponent = (exponent_val - exponent_g_max) >> 23;

    // Situation: delta_exponent >= 0,
    // return trunc_to_mul_of_2_to_b(val + sgn(val) * 2 ** (b - 1)).
    unsigned int val_r_dexp_ge_0;
    val_r_dexp_ge_0 = *p_val + (HEX_00400000 >> delta_exponent);
    val_r_dexp_ge_0 = val_r_dexp_ge_0 & (HEX_FF800000 >> delta_exponent);
    val_r_dexp_ge_0 = (delta_exponent > -1) * val_r_dexp_ge_0;

    // Situation: delta_exponent == -1,
    // return sgn(val) * 2 ** b.
    unsigned int val_r_dexp_eq_m1;
    val_r_dexp_eq_m1 = (*p_val & HEX_80000000) | exponent_g_max;
    val_r_dexp_eq_m1 = (delta_exponent == -1) * val_r_dexp_eq_m1;

    unsigned int val_r = val_r_dexp_ge_0 + val_r_dexp_eq_m1;
    return *((float*) &val_r);
}

// Add main function just for testing.
int main(){
    return 0;
}
