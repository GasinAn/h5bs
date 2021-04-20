// Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).
// Warning: undefined behavior when delta_exponent >= 32.

#include <stdio.h>

float float32_bit_round(float val, float g_max){
    unsigned int *p_val = (int*) &val;
    unsigned int *p_g_max = (int*) &g_max;

    int exponent_val = *p_val & 0x7f800000;
    int exponent_g_max = *p_g_max & 0x7f800000;

    int delta_exponent = (exponent_val - exponent_g_max) >> 23;

    // Situation: delta_exponent >= 0,
    // return trunc_to_mul_of_2_to_b(val + sgn(val) * 2 ** (b - 1)).
    unsigned int val_r_dexp_ge_0;
    val_r_dexp_ge_0 = *p_val + (0x00400000 >> delta_exponent);
    val_r_dexp_ge_0 = val_r_dexp_ge_0 & (-8388608 >> delta_exponent);
    val_r_dexp_ge_0 = (delta_exponent > -1) * val_r_dexp_ge_0;

<<<<<<< HEAD
    if (delta_exponent > -1)
    {
        return *((float*) &val_r);
    }
    else if (delta_exponent > -2)
    {
        int signed_g = (*p_val & -2147483648) | exponent_g_max;
        return *((float*) &signed_g);
    }
    else
    {
        return 0.0;
    }
=======
    // Situation: delta_exponent == -1,
    // return sgn(val) * g_max.
    unsigned int val_r_dexp_eq_m1;
    val_r_dexp_eq_m1 = (*p_val & -2147483648) | exponent_g_max;
    val_r_dexp_eq_m1 = (delta_exponent == -1) * val_r_dexp_eq_m1;

    unsigned int val_r = val_r_dexp_ge_0 + val_r_dexp_eq_m1;
    return *((float*) &val_r);
>>>>>>> speed-up
}

int main(){
    return 0;
}
