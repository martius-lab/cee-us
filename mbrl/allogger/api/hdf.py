from glob import glob
import os
import numpy as np
import h5py

def list_keys(path):
    keys = []

    def get_flat_keys(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    file = sorted(glob(os.path.join(path, '*.h5')))[0]
    with h5py.File(file, 'r') as hdf:
        hdf.visititems(get_flat_keys)

    scalars = [k for k in keys if k.startswith('scalar')]
    histograms = [k for k in keys if k.startswith('histrogram')]
    images = [k for k in keys if k.startswith('image')]
    arrays = [k for k in keys if k.startswith('array')]

    print('Scalars:')
    print('\t' + '\n\t'.join(scalars))
    print('Histograms:')
    print('\t' + '\n\t'.join(histograms))
    print('Images:')
    print('\t' + '\n\t'.join(images))
    print('Arrays:')
    print('\t' + '\n\t'.join(arrays))

def read_from_key(path, key):
    datas = []

    for file in sorted(glob(os.path.join(path, '*.h5'))):
        with h5py.File(file, 'r') as hdf:
            data = np.asarray(hdf[key])
        print(f'Extracting {key} from {file}, index {data["step"][0]} to {data["step"][-1]}')
        datas.append(data)

    datas = np.concatenate(datas, axis=0)

    return datas