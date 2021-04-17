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
        int nchan,
        np.ndarray[np.int64_t, ndim=1, mode='c'] chan_a not None,
        np.ndarray[np.int64_t, ndim=1, mode='c'] chan_b not None,
        np.float64_t f_nsamp,
        ):
    """Reduce visibility precision.

    Parameters
    ----------
    vis: array of VIS_DTYPE with shape (nfreq, nprod, ntime)
        Visibilities to be processed.
    nchan: int
        The number of channels.
    chan_a : array of integers with shape (nprod)
        1st channel (channel A) in correlation product.
    chan_b : array of integers with shape (nprod)
        2nd channel (channel B) in correlation product.
    f_nsamp : float
        f/nsamples. Controls degree of precision reduction.

    Returns
    -------
    vis_r : array like vis
        Visibilities after rounding.

    Notes
    -----
    The number of products should be consistent with the number of channels,
    nprod = nchan*(nchan+1)/2, and channels should be labelled from 0 to
    (nchan-1). Products can be in any order.

    """

    cdef int nprod = vis.shape[1]

    if (2*nprod != nchan*(nchan+1)):
        msg = "nprod should be equal to nchan*(nchan+1)/2."
        raise ValueError(msg)

    if np.any(chan_a>=nchan):
        msg = "Channel A should be labelled by integers less than nchan."
        raise ValueError(msg)

    if np.any(chan_b>=nchan):
        msg = "Channel B should be labelled by integers less than nchan."
        raise ValueError(msg)

    if np.any(chan_a<0):
        msg = "Channel A should be labelled by non-negtive integers."
        raise ValueError(msg)

    if np.any(chan_b<0):
        msg = "Channel B should be labelled by non-negtive integers."
        raise ValueError(msg)

    cdef int ntime = vis.shape[2]
    cdef int ii, jj, kk

    # Need the extra bits otherwise we get overflows.
    cdef np.int64_t auto_a, auto_b
    cdef np.int32_t gran_max
    cdef float gran_max_sq
    cdef float gran_sq_factor = 12*f_nsamp
    cdef float half_gran_sq_factor = 6*f_nsamp

    cdef np.ndarray[np.int64_t, ndim=1, mode='c'] auto_inds
    auto_inds = np.empty(nchan, np.int64)
    auto_inds[:] = -1

    # This is equivalent to the following loop:
    # whether_auto = chan_a==chan_b
    # auto_inds[chan_a[whether_auto]] = np.arange(nprod)[whether_auto]
    # they run almost as fast as each other.
    for ii in xrange(nprod):
        if (chan_a[ii] == chan_b[ii]):
            auto_inds[chan_a[ii]] = ii
    # Now chan_pairs[auto_inds[i]]==(i,i) (chan_pairs:=[(chan_a,chan_b)]).

    if np.any(auto_inds<0):
        msg = "Not all auto-correlations present in visibilities."
        raise ValueError(msg)

    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] auto_vis
    auto_vis = np.empty((nchan, ntime), np.int32)

    # Allow attribute access to struct members.
    vis = vis.view(np.recarray)

    cdef np.ndarray[VIS_DTYPE_t, ndim=3, mode='c'] vis_r
    vis_r = np.empty_like(vis)

    cdef np.int64_t n_a, n_b
    cdef int is_auto

    for jj in xrange(vis.shape[0]):
        # auto_vis = vis[nfreq=jj][chan_pair=(0:nchan,0:nchan)][ntime=:].r
        auto_vis[:] = vis[jj,auto_inds].r

        for ii in xrange(nprod):
            # nprod=ii <=> chan_pair=(n_a,n_b) := (chan_a[ii],chan_b[ii]).
            n_a = chan_a[ii]
            n_b = chan_b[ii]
            is_auto = n_a==n_b

            for kk in xrange(ntime):
                # This casts to 64 bits.
                # nprod=ii <=> chan_pair=(n_a,n_b).
                # auto_a=vis[nfreq=jj][chan_pair=(n_a,n_a)][ntime=kk].r
                # auto_b=vis[nfreq=jj][chan_pair=(n_b,n_b)][ntime=kk].r
                auto_a = auto_vis[n_a,kk]
                auto_b = auto_vis[n_b,kk]
                # V_{ij} <=> vis[nfreq=jj][chan_pair=(n_a,n_b)][ntime=kk],
                # V_{ii}, V_{jj} <=> auto_a, auto_b.
                # see arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015.

                # Calculate the maximum granularity.
                # gran**2 < 12*f*s**2.
                # see arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015.
                if is_auto:
                    # if n_a==n_b, s**2 = (auto_a*auto_b)/nsamp.
                    # (sigma_{Re,ii}**2 = V_{ii}**2/N)
                    # gran**2 < 12*f*(auto_a*auto_b)/nsamp,
                    # gran**2 < auto_a*auto_b*(12*f/nsamp).
                    gran_max_sq = auto_a*auto_b*gran_sq_factor
                    # Truncating the float rounds granularities.
                    # gran < sqrt(12*f*s**2).
                    gran_max = <np.int32_t> sqrt(gran_max_sq)
                    # Round.
                    vis_r[jj,ii,kk].r = bit_round(vis[jj,ii,kk].r, gran_max)
                    vis_r[jj,ii,kk].i = 0
                else:
                    # if n_a!=n_b, s**2 = (auto_a*auto_b)/(2*nsamp).
                    # (sigma_{Re/Im,ij}**2 = (V_{ii}*V_{jj})/(2*N) (i!=j))
                    # gran**2 < 12*f*(auto_a*auto_b)/(2*nsamp),
                    # gran**2 < auto_a*auto_b*(12*f/nsamp)/2.
                    gran_max_sq = auto_a*auto_b*half_gran_sq_factor
                    # Truncating the float rounds granularities.
                    # gran < sqrt(12*f*s**2).
                    gran_max = <np.int32_t> sqrt(gran_max_sq)
                    # Round.
                    vis_r[jj,ii,kk].r = bit_round(vis[jj,ii,kk].r, gran_max)
                    vis_r[jj,ii,kk].i = bit_round(vis[jj,ii,kk].i, gran_max)

    return vis_r


cdef inline np.int32_t bit_round(np.int32_t val, np.int32_t gran_max):
    """Round value to multiple of largest power-of-2 not greater than gran_max.

    Undefined results for gran_max < 0 and gran_max > 2**30.

    """

    if (gran_max > 1073741824):
        raise ValueError('Sorry, maximum of granularity is too large.')

    # gran is the granularity.
    # It is the largest power-of-2 not greater than gran_max.
    # (unless gran_max is equal to 0)

    # gran_max == 000...01????...?
    cdef np.int32_t gran = gran_max >> 1
    # gran == 000...001???...?

    gran |= gran >> 1
    # gran == 000...0011???...?
    gran |= gran >> 2
    # gran == 000...001111???...?
    gran |= gran >> 4
    # gran == 000...0011111111???...?
    gran |= gran >> 8
    # gran == 000...001111111111111111???...?

    # Bitmask selects bits to be rounded.
    cdef np.int32_t bitmask = gran | (gran >> 16)
    # bitmask == 000...001111...1
    # if gran_max <= 1, bitmask == 0

    gran = bitmask + 1
    # gran == 000...010000...0, bingo!
    # if gran_max <= 1, gran == 1

    # Determine if there is a round-up/round-down tie.
    # This operation gets the "gran == 1" case correct (non tie).
    cdef np.int32_t tie = ((val & bitmask) << 1) == gran
    # val == ???...??abcd...?
    # val & bitmask == 000...00abcd...?
    # (val & bitmask) << 1 == 000...0abcde...?
    # gran == 000...010000...0
    # ((val & bitmask) << 1) == gran <=> abcd...? == 1000...0
    # if gran_max <= 1, tie = 0

    # The actual rounding.
    cdef np.int32_t val_t = (val - (gran >> 1)) | bitmask
    # val == ???...??abcd...?
    # gran >> 1 == 000...001000...0
    # if val == ???...??1bcd...?
    # val - (gran >> 1) == ???...??0bcd...?
    # (val - (gran >> 1)) | bitmask == ???...??1111...1
    # if val == ???...??0bcd...?
    # val - (gran >> 1) == (???...??-1)1bcd...?
    # (val - (gran >> 1)) | bitmask == (???...??-1)1111...1
    # Special situation:
    # if gran_max <= 1, (val - (gran >> 1)) | bitmask == val !
    val_t += (gran_max >> 1) != 0
    # if val == ???...??1bcd...?
    # val_t == (???...??+1)0000...0
    # if val == ???...??0bcd...?
    # val_t == ???...??0000...0

    # Break any tie by rounding to even.
    # if tie == 1, val == ???...??1000...0
    # val_t == (???...??+1)0000...0
    val_t -= val_t & (tie * gran)
    # if val == ???...?01000...0
    # val_t == !!!...!10000...0
    # gran == 000...010000...0
    # val_t & (tie * gran) == 000...010000...0
    # val_t == !!!...!00000...0
    # if val == ???...?11000...0
    # val_t == !!!...!00000...0
    # gran == 000...010000...0
    # val_t & (tie * gran) == 000...000000...0
    # val_t == !!!...!00000...0
    # In conclusion: val = ???...??1000...0 => val_t == !!!...!00000...0,
    # which means rounding "by" gran then truncating "by" gran*2.

    return val_t


def bit_round_py(np.int32_t val, np.int32_t gran_max):
    """Python wrapper of C version, for testing."""
    return bit_round(val, gran_max)


def test(*, l=False):
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

    # Generate mock data. Model is pure uncorrelated receiver noise.
    # Auto correlations are a number, everything else is noise.
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
    vis_rounded = reduce_precision(vis, nchan, chan_a, chan_b, f / nsamples)
    t = time.perf_counter() - t0

    rate = nprod * nfreq * ntime * VIS_DTYPE.itemsize / t
    print "Throughput: %f MiB/s" % (rate / 1024**2)

    rounding_error = (vis_rounded.r - vis.r).astype(np.int64)
    if l:
        print "Rounding bias:"
        print np.mean(rounding_error, -1)
        print "Rounding RMS:"
        print np.sqrt(np.mean(rounding_error**2, -1))
        print "Relative to thermal noise:"
        print np.mean(rounding_error**2, -1) / np.var(vis.r, -1)


if __name__ == "__main__":
    test()
