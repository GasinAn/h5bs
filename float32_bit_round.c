// Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).

#include <stdio.h>

float float32_bit_round(float val, float g_max){
    // uint32_t may be better; since int is signed and can be 64bit
    int *p_val = (int*) &val;
    int *p_g_max = (int*) &g_max;

    int exponent_val = *p_val & 0x7f800000;
    int exponent_g_max = *p_g_max & 0x7f800000;

    int delta_exponent = (exponent_val - exponent_g_max) >> 23;

    int g;
    float *p_g;

    float val_;
    // same as above, uint32_t
    int *p_val_;

    int val_r;
    float *p_val_r;

    if (delta_exponent >> 31) // delta_exponent < 0
    {
        g = *p_g_max & 0x7f800000;
        // if use uint32_t, then maybe can write something like 0x80000000u (not sure)
        g = (*p_val & -2147483648) | g; // -2147483648: 80000000
        g = (delta_exponent == -1) * g;
        // directly do *((float*)&g) is enough
        p_g = (float*) &g;
        return *p_g;
    }
    else if (delta_exponent < 23)
    {
        g = *p_g_max & -8388608; // -8388608: ff800000
        p_g = (float*) &g;

        // try to eliminate this `if`, since (failed) branch prediction can be expensive
        if (*p_val >> 31) // *p_val < 0
        {
            // floating point divide maybe slow (not sure, need benchmark)
            // can use the method I proposed yesterday (1x bitwise operation on the "exponent" part of a float, may be faster)
            val_ = val + (*p_g / 2.0);
        }
        else
        {
            // same as above
            val_ = val - (*p_g / 2.0);
        }
        p_val_ = (int*) &val_;

        // may be write as 0xff800000u
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
