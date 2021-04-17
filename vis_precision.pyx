"""Visibility data precision reduction.

Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
Website: https://gist.github.com/kiyo-masui/b61c7fa4f11fca453bdd

Copyright (c) 2015 Kiyoshi Masui (kiyo@physics.ubc.ca)

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

Modified by Jiachen An <Gasin185@163.com>
Website: http://www.github.com/GasinAn/h5bs/vis_precision.pyx

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

import math

import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt

np.import_array()


ctypedef struct VIS_DTYPE_t:
    np.int32_t r
    np.int32_t i


VIS_DTYPE = np.dtype([('r', np.int32), ('i', np.int32)])


@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_precision(
        np.ndarray[VIS_DTYPE_t, ndim=3, mode='c'] vis not None,
        np.ndarray[np.int64_t, ndim=1, mode='c'] prod_a not None,
        np.ndarray[np.int64_t, ndim=1, mode='c'] prod_b not None,
        np.float64_t f_nsamp,
        ):
    """Reduce visibility precision.

    Parameters
    ----------
    vis : array of VIS_DTYPE with shape (nfreq, nprod, ntime)
        Visibilities to be processed.
    chan_a : array of integers with shape (nprod)
        First channel in correlation product.
    chan_b : array of integers with shape (nprod)
        Second channel in correlation product.
    f_nsamp : float
        f / nsamples. Controls degree of precision reduction.

    Returns
    -------
    vis_rounded : array like vis
        Visibilities after rounding.

    Notes
    -----
    The number of products should be consistent with the number of channels,
    nprod = nchan * (nchan + 1) / 2, and channels should be labelled from 0 to
    nchan - 1.  Products can be in any order.

    """

    cdef int nprod = vis.shape[1]
    cdef int ntime = vis.shape[2]
    cdef int ii, jj
    cdef int nchan
    nchan = np.max(prod_a) + 1

    # Need the extra bits otherwise we get overflows.
    cdef np.int64_t auto_a, auto_b
    cdef np.int32_t gran_max
    cdef float gran_max_sq
    cdef float gran_sq_factor = 12 * f_nsamp

    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] auto_inds
    auto_inds = np.empty(nchan, np.int64)
    auto_inds[:] = -1
    for ii in xrange(nprod):
        if prod_a[ii] == prod_b[ii]:
            auto_inds[prod_a[ii]] = ii
    if np.any(auto_inds < 0):
        msg = "Not all auto-correlations present in visibilities."
        raise ValueError(msg)

    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] auto_vis
    auto_vis = np.empty((nchan, ntime), np.int32)

    # Allow attribute access to struct members.
    vis = vis.view(np.recarray)

    cdef np.ndarray[VIS_DTYPE_t, ndim=3, mode='c'] vis_rounded
    vis_rounded = np.empty_like(vis)

    for jj in xrange(vis.shape[0]):
        for ii in xrange(nchan):
            auto_vis[ii] = vis[jj,auto_inds[ii]].r

        for ii in xrange(nprod):
            for kk in xrange(ntime):
                # This casts to 64 bits.
                auto_a = auto_vis[prod_a[ii],kk]
                auto_b = auto_vis[prod_b[ii],kk]

                # Calculate the maximum granularity.
                gran_max_sq = auto_a * auto_b * gran_sq_factor / 2.
                # Granularity is factor of 2 larger for autos.
                gran_max_sq *= 1 + (prod_a[ii] == prod_b[ii])
                # Truncating the float rounds granularities down and
                # is conservative.
                gran_max = <np.int32_t> sqrt(gran_max_sq)

                # Round.
                vis_rounded[jj,ii,kk].r = bit_round(vis[jj,ii,kk].r, gran_max)
                vis_rounded[jj,ii,kk].i = bit_round(vis[jj,ii,kk].i, gran_max)

    return vis_rounded


cdef inline np.int32_t bit_round(np.int32_t val, np.int32_t gran_max):
    """Round value to multiple of largest power-of-2 smaller than gran_max.

    Undefined results for gran_max < 0 and gran_max > 2**30.

    """

    # gran is the granularity. It is the largest power-of-2 less than
    # gran_max.
    cdef np.int32_t gran = gran_max >> 1
    gran |= gran >> 1
    gran |= gran >> 2
    gran |= gran >> 4
    gran |= gran >> 8
    gran |= gran >> 16
    gran += 1

    # Bitmask selects bits to be rounded.
    cdef np.int32_t bitmask = gran - 1

    # Determine if there is a round-up/round-down tie.
    # This operation gets the `gran = 1` case correct (non tie).
    cdef np.int32_t tie = ((val & bitmask) << 1) == gran

    # The actual rounding.
    cdef np.int32_t val_t = (val - (gran >> 1)) | bitmask
    val_t += 1
    # There is a bit of extra bit twiddling for the gran_max <= 1.
    val_t -= (gran_max >> 1) == 0

    # Break any tie by rounding to even.
    val_t -= val_t & (tie * gran)

    return val_t


def bit_round_py(np.int32_t val, np.int32_t gran_max):
    """Python wrapper of C version, for testing."""
    return bit_round(val, gran_max)


def test():
    """Generate mock data then reduce its precision."""

    from numpy import random
    import time

    # Parameters.
    nchan = 16          # Number of channels correlated
    nsamples = 100      # Number of samples integrated, delta_f*delta_t
    Tsys = 50           # System temperature
    f = 0.01            # Precision reduction parameter
    nfreq = 5           # Added dimensionality, spectral frequencies.
    ntime = 1000        # Added dimensionality, temporal integrations.

    # Made up channel dependant gain.
    gain_chan = np.arange(nchan) + nchan
    # Made up frequency dependant gain.
    bandpass = (np.arange(nfreq) + nfreq)**2

    # Generate mock data. Model is pure uncorrelated receiver noise.  Auto
    # correlations are a number, everything else is noise.
    nprod = (nchan * (nchan + 1)) // 2
    vis = np.recarray((nfreq, nprod, ntime), VIS_DTYPE)
    chan_a = np.empty(nprod, np.int64)
    chan_b = np.empty(nprod, np.int64)

    for ff in range(nfreq):
        kk = 0
        for ii in range(nchan):
            for jj in range(ii, nchan):
                chan_a[kk] = ii
                chan_b[kk] = jj

                amp = Tsys * gain_chan[ii] * gain_chan[jj] * bandpass[ff]
                if ii == jj:
                    vis[ff,kk].r = np.round(
                        amp * (1. + random.randn(ntime) / math.sqrt(nsamples)))
                    vis[ff,kk].i = 0.
                else:
                    vis[ff,kk].r = np.round(
                        amp * random.randn(ntime) / math.sqrt(2 * nsamples))
                    vis[ff,kk].i = np.round(
                        amp * random.randn(ntime) / math.sqrt(2 * nsamples))
                kk += 1

    t0 = time.perf_counter()
    vis_rounded = reduce_precision(vis, chan_a, chan_b, f / nsamples)
    t = time.perf_counter() - t0

    rate = nprod * nfreq * ntime * VIS_DTYPE.itemsize / t
    print "Throughput: %f MiB/s" % (rate / 1024**2)

    rounding_error = (vis_rounded.r - vis.r).astype(np.int64)
    #print "Rounding bias:"
    #print np.mean(rounding_error, -1)
    #print "Rounding RMS:"
    #print np.sqrt(np.mean(rounding_error**2, -1))
    #print "Relative to thermal noise:"
    #print np.mean(rounding_error**2, -1) / np.var(vis.r, -1)


if __name__ == "__main__":
    test()
