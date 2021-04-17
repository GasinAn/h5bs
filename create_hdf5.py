# Author: Jixia Li
# Modified by Jiachen An
# Website: http://cosmology.bao.ac.cn/~lijixia/create_hdf5.py

import h5py
import numpy as np

# Function to create a fake visibility with random values.
def vis():
    rp = np.random.random((5, 512, 200)).astype(np.float32)
    ip = np.random.random((5, 512, 200)).astype(np.float32)
    return rp + 1j * ip
# Function to create a fake antenna pointing with random values.
def antpointing(nfeeds):
    a = np.zeros((nfeeds, 4), dtype = np.float32)
    a[:, 0] = np.ones(16, dtype = np.float32) * np.random.random(1)[0] * 360
    a[:, 1] = np.ones(16, dtype = np.float32) * np.random.random(1)[0] * 90
    a[:, 2] = np.random.random(16) * 0.1
    a[:, 3] = np.random.random(16) * 0.03
    return a

###########################################################
# This code will create an example hdf5 file according to #
# the definition documentation.                           #
# Contact info: jxli@bao.ac.cn                            #
###########################################################
if __name__ == '__main__':
    # Create a hdf5 file object.
    df = h5py.File('example.hdf5', 'w')
    
    # Type A Keywords
    df.attrs['nickname'] = 'Keyword Example Data' # Any nick name for the data file.
    df.attrs['comment'] = 'Here is comment.'
    df.attrs['observer'] = 'Someone'
    df.attrs['history'] = 'No history.'
    df.attrs['keywordver'] = '0.0' # Keyword version.
    # Type B Keywords
    df.attrs['sitename'] = 'Hongliuxia Observatory'
    df.attrs['sitelat'] = 44.17639   # Not precise
    df.attrs['sitelon'] = 91.7413861 # Not precise
    df.attrs['siteelev'] = 1500.0    # Not precise
    df.attrs['timezone'] = 'UTC+08'  # Beijing time
    df.attrs['epoch'] = 2000.0  # year
    # Type C Keywords
    df.attrs['telescope'] = 'Tianlai-Dish-I'
    # "Tianlai-Cylinder-I", "Tianlai-Cylinder-II" ...
    df.attrs['dishdiam'] = 6.0  # meters; For cylinder: -1.0
    df.attrs['nants'] = 16 # For Cylinder: 3
    df.attrs['npols'] = 2
    df.attrs['cylen'] = 50 # For dish: -1
    df.attrs['cywid'] = 50 # For dish: -1
    # Type D Keywords
    df.attrs['recvver'] = '0.0'    # Receiver version.
    df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.
    # Type E Keywords
    df.attrs['corrver'] = '0.0'    # Correlator version.
    df.attrs['samplingbits'] = 8 # ADC sampling bits.
    df.attrs['corrmode'] = 1 # 2, 3
    df.attrs['inttime'] = 1.0
    df.attrs['obstime'] = '2016/02/29 09:30:22' # Y/M/D H:M:S
    df.attrs['nfreq'] = 512 # Number of Frequency Points
    df.attrs['freqstart'] = 685.0 # MHz; Frequency starts.
    df.attrs['freqstep'] = 0.244140625 # MHz; Frequency step.
    #df.attrs[''] = 

    # Data Array
    df['vis'] = vis()
    df['vis'].attrs['dimname'] = 'Time, Frequency, Baseline'

    df['feedno'] = np.arange(1, 17, dtype = np.int32)
    # For Cylinder: 1-192

    df['channo'] = np.arange(1, 33, dtype = np.int32).reshape(-1,2)
    # -1 for invalid channel
    df['channo'].attrs['dimname'] = 'Feed No., (XPolarization YPolarization)'

    df['blorder'] = np.array([[2,2],[1,1],[4,4],[3,3], [1,4], [1,3]])
    df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'

    df['feedpos'] = np.random.random((16, 3)).astype(np.float32)
    # Feeds' positions in horizontal coordinate.
    df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
    df['feedpos'].attrs['unit'] = 'degree'

    df['antpointing'] = antpointing(16)
    df['antpointing'].attrs['dimname'] = 'Feed No., (Az,Alt,AzErr,AltErr)'
    df['antpointing'].attrs['unit'] = 'degree'

    df['polerr'] = np.zeros((16,2), dtype=np.float32)
    # Clockwise? Anti-Clockwise?
    df['polerr'].attrs['dimname'] = 'Feed No., (XPolErr,YPolErr)'
    df['polerr'].attrs['unit'] = 'degree'

    df['noisesource'] = np.array(
        [[60.0, 0.3], [0.0, 0.0], [300.0, 3.5]], 
        np.float32,
    ) # Unit: seconds; Cycle < 0 means turned off.
    df['noisesource'].attrs['dimname'] = 'Source No., (Cycle Duration)' #
    df['noisesource'].attrs['unit'] = 'second'

    df['transitsource'] = np.array([['2016/2/29 11:03:15', 'Cygnus A'], 
                                    ['2016/2/29 15:32:09', 'Sun'], 
                                    ['2016/2/29 21:54:20', 'Cassiopeia A']],
                                    dtype=h5py.special_dtype(vlen=str))
    df['transitsource'].attrs['dimname'] = 'Source, (DateTime, SourceName)'

    df['weather'] = np.array(
        [[0.0,   17.7, -5.6, -9.5,  85.2, 0.0, 29.2, 1.2, 0.0],
         [300.0, 18.2, -6.9, -11.2, 80.5, 0.0, 31.8, 1.0, 0.0],
         [600.0, 18.0, -7.5, -13.8, 78.3, 0.0, 30.0, 1.1, 0.0],
         [900.0, 18.5, -8.0, -14.2, 75.2, 0.0, 29.5, 1.2, 0.0]],
         dtype = np.float32,
    )

    df['weather'].attrs['dimname'] = (
        'Weather Data, '
        '(TimeOffset, RoomTemperature, SiteTemperature, Dewpoint, Humidity, '
        'Precipitation, WindDirection, WindSpeed, Pressure)'
    )
    df['weather'].attrs['unit'] = (
        'second, Celcius, Celcius, Celcius, %, millimeter, degree, m/s, mmHg'
    )

    df.close()
