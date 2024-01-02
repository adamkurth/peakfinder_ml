import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn as sk
import numpy as np
import h5py as h5
import os
import glob


def load_data(work=True):
    if work: # whether at work or not
        file_path = 'images/DATASET1-1.h5'
    else:
        water_background_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
        file_path = os.path.join(water_background_dir,'9_18_23_high_intensity_3e8keV-2.h5')
        
    matching_files = glob.glob(file_path)
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: {file_path}")
        
    try:
        with h5.File(file_path, 'r') as f:
            data = f['entry/data/data'][:]
        return data
    except Exception as e:
        raise OSError(f"Failed to read {file_path}: {e}")
    return file_path

def main():
    work = False # load home image
    data = load_data(work)
    return data
    
if __name__ == '__main__':
    image = main()
    X = image.reshape(image.shape[0], -1) # (4371, 4150)
    
    # standardize data (mean = 0, std = 1)
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, X_scaled, test_size=0.2)

