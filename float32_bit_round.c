// Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).

#include <stdio.h>

float float32_bit_round(float val, float g_max){
    int *p_val = (int*) &val;
    int *p_g_max = (int*) &g_max;

    int exponent_val = *p_val & 0x7f800000;
    int exponent_g_max = *p_g_max & 0x7f800000;

    int delta_exponent = (exponent_val - exponent_g_max) >> 23;

    int g;
    float *p_g;

    float val_;
    int *p_val_;

    int val_r;
    float *p_val_r;

    if (delta_exponent >> 31) // delta_exponent < 0
    {
        if (delta_exponent ^ -1) // delta_exponent != -1
        {
            return 0.0;
        }
        else
        {
            g = *p_g_max & 0x7f800000;
            g = (*p_val & -2147483648) | g; // -2147483648: 80000000
            p_g = (float*) &g;
            return *p_g;
        }
    }
    else if (delta_exponent < 23)
    {
        g = *p_g_max & -8388608; // -8388608: ff800000
        p_g = (float*) &g;

        if (*p_val >> 31) // *p_val < 0
        {
            val_ = val + (*p_g / 2.0);
        }
        else
        {
            val_ = val - (*p_g / 2.0);
        }
        p_val_ = (int*) &val_;

        val_r = *p_val_ & (-8388608 >> delta_exponent); // -8388608: ff800000
        p_val_r = (float*) &val_r;
        return *p_val_r;
    }
    else
    {
        return val;
    } 
}

int main(){
    return 0;
}