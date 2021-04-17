import numpy
import h5py
import hdf5plugin

from dnb import reduce_precision


DTYPE = numpy.dtype([('r', numpy.int32), ('i', numpy.int32)])


def test(*, l=False):
    """Test dnb.reduce_precision and hdf5plugin.Bitshuffle."""

    from math import sqrt
    from time import perf_counter

    # Parameters.
    nchan = 16          # Number of channels correlated
    nsamples = 100      # Number of samples integrated, delta_f*delta_t
    Tsys = 50           # System temperature
    f = 0.01            # Precision reduction parameter
    nfreq = 5           # Added dimensionality, spectral frequencies.
    ntime = 1000        # Added dimensionality, temporal integrations.

    # Made up channel dependant gain.
    gain_chan = numpy.arange(nchan)+nchan
    # Made up frequency dependant gain.
    bandpass = (numpy.arange(nfreq)+nfreq)**2

    # Generate mock data. Model is pure uncorrelated receiver noise.
    # Auto correlations are a number, everything else is noise.
    nprod = (nchan*(nchan+1))//2
    vis = numpy.recarray((nfreq, nprod, ntime), DTYPE)
    chan_a = numpy.empty(nprod, numpy.int64)
    chan_b = numpy.empty(nprod, numpy.int64)
    
    for ff in range(nfreq):
        kk = 0
        for ii in range(nchan):
            for jj in range(ii, nchan):
                chan_a[kk] = ii
                chan_b[kk] = jj

                amp = Tsys*gain_chan[ii]*gain_chan[jj]*bandpass[ff]
                if (ii == jj):
                    vis[ff,kk].r = numpy.round(
                        amp*(1.0+numpy.random.randn(ntime)/sqrt(nsamples)))
                    vis[ff,kk].i = 0.0
                else:
                    vis[ff,kk].r = numpy.round(
                        amp*numpy.random.randn(ntime)/sqrt(2*nsamples))
                    vis[ff,kk].i = numpy.round(
                        amp*numpy.random.randn(ntime)/sqrt(2*nsamples))
                kk += 1

    # Reduce precision.
    t0 = perf_counter()
    vis_rounded = reduce_precision(vis, nchan, chan_a, chan_b, f/nsamples)
    t = perf_counter()-t0

    rate = nfreq*nprod*ntime*DTYPE.itemsize/t
    print("Throughput(reduce_precision): %f MiB/s" %(rate/1024**2))

    # Compress.
    with h5py.File('test.h5', 'w') as f:
        t0 = perf_counter()
        f.create_dataset('mock_data',
            data=vis_rounded, **hdf5plugin.Bitshuffle())
        t = perf_counter()-t0

    rate = nfreq*nprod*ntime*DTYPE.itemsize/t
    print("Throughput(bitshuffle_compress): %f MiB/s" %(rate/1024**2))

    # Decompress.
    with h5py.File('test.h5', 'r') as f:
        t0 = perf_counter()
        vis_decompressed = f['mock_data'][()]
        t = perf_counter()-t0

    rate = nfreq*nprod*ntime*DTYPE.itemsize/t
    print("Throughput(bitshuffle_decompress): %f MiB/s" %(rate/1024**2))

    if numpy.any(vis_rounded!=vis_decompressed):
        raise ValueError('Data changed after I/O.')

    # Calculate compression rate.
    import os
    rate = os.path.getsize('test.h5')/(nfreq*nprod*ntime*DTYPE.itemsize)
    print('Compression rate: %f %%' %(100*rate))

    rounding_error = (vis_rounded.r-vis.r).astype(numpy.int64)
    if l:
        print("Rounding bias:")
        print(numpy.mean(rounding_error, -1))
        print("Rounding RMS:")
        print(numpy.sqrt(numpy.mean(rounding_error**2, -1)))
        print("Relative to thermal noise:")
        print(numpy.mean(rounding_error**2, -1)/numpy.var(vis.r, -1))


if __name__=='__main__':
    test()
