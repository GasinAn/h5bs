import numpy
import h5py
import hdf5plugin

from dnb_float32 import reduce_precision


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

    band_pass = numpy.arange(nfreq, 2*nfreq)**2
    gain_chan = numpy.arange(nchan, 2*nchan)

    nprod = (nchan*(nchan+1))//2
    vis = numpy.empty((nfreq, nprod, ntime), numpy.complex64)
    chan_a = numpy.empty(nprod, numpy.int32)
    chan_b = numpy.empty(nprod, numpy.int32)

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
                    vis_r = A*abs(1+randn(ntime)/numpy.sqrt(N))
                    vis_i = 0
                else:
                    vis_r = A*randn(ntime)/numpy.sqrt(2*N)
                    vis_i = A*randn(ntime)/numpy.sqrt(2*N)
                vis[k0,k1] = vis_r+vis_i*1j
                k1 += 1

    # Reduce precision.
    t_s = time.perf_counter()
    vis_r, vis_i = reduce_precision(vis, nchan, chan_a, chan_b, f/N)
    t_e = time.perf_counter()

    rate = nfreq*nprod*ntime*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
    print("Throughput(reduce_precision): %f MiB/s" %(rate/1024**2))

    # Compress.
    with h5py.File('test_float32.h5', 'w') as f:
        t_s = time.perf_counter()
        f.create_dataset('vis_r', data=vis_r, **hdf5plugin.Bitshuffle())
        f.create_dataset('vis_i', data=vis_i, **hdf5plugin.Bitshuffle())
        t_e = time.perf_counter()

    rate = nfreq*nprod*ntime*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
    print("Throughput(bitshuffle_compress): %f MiB/s" %(rate/1024**2))

    # Decompress.
    with h5py.File('test_float32.h5', 'r') as f:
        t_s = time.perf_counter()
        vis_r_ = f['vis_r'][...]
        vis_i_ = f['vis_i'][...]
        t_e = time.perf_counter()

    rate = nfreq*nprod*ntime*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
    print("Throughput(bitshuffle_decompress): %f MiB/s" %(rate/1024**2))

    if numpy.any(vis_r_!=vis_r) or numpy.any(vis_i_!=vis_i):
        raise ValueError('Data changed after I/O.')

    # Calculate compression rate.
    import os
    fsize = os.path.getsize('test_float32.h5')
    rate = fsize/(nfreq*nprod*ntime*numpy.dtype(numpy.complex64).itemsize)
    print('Compression rate: %f %%' %(100*rate))

if __name__=='__main__':
    test()
