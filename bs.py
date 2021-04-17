import h5py
import hdf5plugin

df = h5py.File('example.hdf5', 'r')

vis = df['vis'][...]
vis_r, vis_i = vis.real, vis.imag

with h5py.File('vis.h5', 'w') as f:
    f.create_dataset('vis', data=vis)

bs = hdf5plugin.Bitshuffle()
with h5py.File('vis_ri.h5', 'w') as f:
    f.create_dataset('vis_r', data=vis_r, **bs)
    f.create_dataset('vis_i', data=vis_i, **bs)
