// Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).

#include <stdio.h>

float float32_bit_round(float val, float g_max){
    unsigned int *p_val = (int*) &val;
    unsigned int *p_g_max = (int*) &g_max;

    int exponent_val = *p_val & 0x7f800000;
    int exponent_g_max = *p_g_max & 0x7f800000;

    int delta_exponent = (exponent_val - exponent_g_max) >> 23;

    unsigned int val_ = *p_val + (0x00400000 >> delta_exponent);
    unsigned int val_r = val_ & (-8388608 >> delta_exponent);

    return *((float*) &val_r);
}

int main(){
    int delta_exponent = 34;
    printf("%x",-8388608>>delta_exponent);
    return 0;
}
