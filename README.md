# h5bs
Hello everyone. The core func in this repo is bit_round in dnb_float32.pyx. I have translated it into a C version in float32_bit_round.c: float float32_bit_round(float val, float g_max). It rounds float val to float val_r = n*2\**b (int n; int b = max(b: 2\**b <= g_max)). I think in most of situations it works correctly, but it is needed to be speeded up. Please help me to find bug and speed up this func, thank you!

---

Oh! Maybe I don't need it any more! You may be interested in [bitround](https://github.com/GasinAn/bitround)!
