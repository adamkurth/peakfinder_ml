import scipy
import sklearn as skl
import numpy as np
import h5py as h5

def load_data(file_path):
    with h5.File(file_path, 'r') as f:
        data = f['entry/data/data'][:]
    return data

def main():
    file_path = 'images/DATASET1-1.h5'
    data = load_data(file_path)
    print(data)
    
if __name__ == '__main__':
    main()
    

