"""
Visibility float32 Data Precision Reduction.

See arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015 for description of method.

Author: Jiachen An <Gasin185@163.com>
Website: https://github.com/GasinAn/h5bs/blob/main/dnb.pyx

Copyright (c) 2021 Jiachen An (Gasin185@163.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdio cimport printf

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_precision(
        np.ndarray[np.complex64_t, ndim=3, mode='c'] vis not None,
        np.int32_t nchan,
        np.ndarray[np.int32_t, ndim=1, mode='c'] chan_a not None,
        np.ndarray[np.int32_t, ndim=1, mode='c'] chan_b not None,
        np.float64_t f_N,
        ):
    """
    Reduce visibility precision for float32 data.

    Parameters
    ----------
    vis: array of complex64 with shape (nfreq, nprod, ntime)
        Visibilities to be processed.
    nchan: int
        The number of channels.
    chan_a: array of integers with shape (nprod)
        1st channel (channel A) in correlation product.
    chan_b: array of integers with shape (nprod)
        2nd channel (channel B) in correlation product.
    f_N: float
        f/N. Controls degree of precision reduction.

    Returns
    -------
    vis_r : array of float32 with shape (nfreq, nprod, ntime)
        Re of visibilities after rounding.
    vis_i : array of float32 with shape (nfreq, nprod, ntime)
        Im of visibilities after rounding.

    Notes
    -----
    The number of products should be consistent with the number of channels,
    nprod = nchan*(nchan+1)/2, and channels should be labeled from 0 to
    (nchan-1). Products can be in any order.
    It is assumed that the cross-correlations between channels are much more
    smaller than the auto-correlations.
    See arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015 for details.

    """

    cdef np.int32_t nfreq = vis.shape[0]
    cdef np.int32_t nprod = vis.shape[1]
    cdef np.int32_t ntime = vis.shape[2]

    cdef np.int32_t i
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] auto_inds
    auto_inds = np.empty(nchan, np.int32)
    for i in xrange(nprod):
        if chan_a[i]==chan_b[i]:
            auto_inds[chan_a[i]] = i

    cdef np.int32_t n0, n1, n2

    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] auto_vis
    auto_vis = np.empty((nchan, ntime), np.float32)

    cdef np.int32_t n_chan_a, n_chan_b
    cdef np.int32_t is_auto

    cdef np.float32_t auto_a, auto_b

    cdef np.float64_t auto_g_factor = f_N*12
    cdef np.float64_t corr_g_factor = f_N*6

    cdef np.float32_t g_max

    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] vis_r
    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] vis_i
    vis_r = np.empty((nfreq, nprod, ntime), np.float32)
    vis_i = np.empty((nfreq, nprod, ntime), np.float32)

    for n0 in xrange(nfreq):
        auto_vis[:] = vis[n0,auto_inds].real

        for n1 in xrange(nprod):
            n_chan_a = chan_a[n1]
            n_chan_b = chan_b[n1]
            is_auto = n_chan_a==n_chan_b

            for n2 in xrange(ntime):
                auto_a = auto_vis[n_chan_a,n2]
                auto_b = auto_vis[n_chan_b,n2]

                if is_auto:
                    g_max = <np.float32_t> sqrt(auto_a*auto_b*auto_g_factor)

                    vis_r[n0,n1,n2] = bit_round(vis[n0,n1,n2].real, g_max)
                    vis_i[n0,n1,n2] = 0

                else:
                    g_max = <np.float32_t> sqrt(auto_a*auto_b*corr_g_factor)

                    vis_r[n0,n1,n2] = bit_round(vis[n0,n1,n2].real, g_max)
                    vis_i[n0,n1,n2] = bit_round(vis[n0,n1,n2].imag, g_max)

    return vis_r, vis_i

def bit_round_py(np.float32_t val, np.float32_t g_max):
    """Python wrapper of C version, for testing."""
    return bit_round(val, g_max)

cdef inline np.float32_t bit_round(np.float32_t val, np.float32_t g_max):
    """Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max))."""

    cdef np.uint32_t *p_val = <np.uint32_t*> &val
    cdef np.uint32_t *p_g_max = <np.uint32_t*> &g_max

    cdef np.int32_t exponent_val = p_val[0] & 0x7f800000
    cdef np.int32_t exponent_g_max = p_g_max[0] & 0x7f800000

    cdef np.int32_t delta_exponent = (exponent_val - exponent_g_max) >> 23

    cdef np.uint32_t val_ = p_val[0] + (0x00400000 >> delta_exponent)
    cdef np.uint32_t val_r = val_ & (-8388608 >> delta_exponent)

    return (<np.float32_t*> &val_r)[0]

def test():
    """Test reduce_precision."""

    import time
    from numpy.random import randn

    nfreq = 5     # Number of spectral frequencies.
    nchan = 16    # Number of channels correlated.
    ntime = 1000  # Number of temporal integrations.

    f = 0.01 # Precision reduction parameter.
    N = 100  # Number of samples integrated (delta_f*delta_t).

    T = 50 # System temperature.

    band_pass = np.arange(nfreq, 2*nfreq)**2
    gain_chan = np.arange(nchan, 2*nchan)

    nprod = (nchan*(nchan+1))//2
    vis = np.empty((nfreq, nprod, ntime), np.complex64)
    chan_a = np.empty(nprod, np.int32)
    chan_b = np.empty(nprod, np.int32)

    k = 0
    for i in range(nchan):
        for j in range(i+1):
            chan_a[k], chan_b[k] = i, j
            k += 1

    for k0 in range(nfreq):
        k1 = 0
        for i in range(nchan):
            for j in range(i+1):
                A = T*gain_chan[i]*gain_chan[j]*band_pass[k0]
                if (i==j):
                    vis_r = A*abs(1+randn(ntime)/np.sqrt(N))
                    vis_i = 0
                else:
                    vis_r = A*randn(ntime)/np.sqrt(2*N)
                    vis_i = A*randn(ntime)/np.sqrt(2*N)
                vis[k0,k1] = vis_r+vis_i*1j
                k1 += 1

    t_s = time.perf_counter()
    vis_r, vis_i = reduce_precision(vis, nchan, chan_a, chan_b, f/N)
    t_e = time.perf_counter()

    rate = nfreq*nprod*ntime*np.dtype(np.complex64).itemsize/(t_e-t_s)
    print("Throughput: %f MiB/s" %(rate/1024**2))


if __name__=='__main__':
    test()
