import os
import numpy as np
import h5py as h5
import glob 

from sklearn import svm 
from sklearn.model_selection import train_test_split

from label_finder import(
    PeakThresholdProcessor,
    ArrayRegion,
    view_neighborhood,
    main,
    validate,
    is_peak,
)

def load_data(choice):
    if choice: # whether at work or not
        file_path = 'images/DATASET1-1.h5'
    elif choice == False:
        water_background_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
        files = glob.glob(os.path.join(water_background_dir, '*.h5'))
        print("Select a file from the following options:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")
        file_index = int(input("Enter the file number: ")) - 1
        file_path = files[file_index]
        
    matching_files = glob.glob(file_path)
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: \n{file_path}")
        
    try:
        with h5.File(file_path, 'r') as f:
            data = f['entry/data/data'][:]
        return data, file_path
    except Exception as e:
        raise OSError(f"Failed to read {file_path}: {e}")

def pre_process(image_choice):
    image_array, path = load_data(image_choice)
    processor = PeakThresholdProcessor(image_array, 200)
    coord_manual = main(path, 200, display=False)
    coord_manual = [tuple(coord) for coord in coord_manual]
    coord_script = processor.get_local_maxima()
    confirmed_coordinates, coord_manual, coord_script = validate(coord_manual, coord_script, image_array)
    # visualize(confirmed_coordinates, coord_manual, coord_script, image_array)
    confirmed_coordinates = list(confirmed_coordinates)
    return confirmed_coordinates, image_array  

def labeled_array(image_array, coord_list):
    labeled = np.zeros(image_array.shape, dtype=int)
    for coord in coord_list: 
        x, y = coord
        if is_peak(image_array, coord, neighborhood_size=5):
            labeled[x, y] = 1
    return labeled # array

def svm(image_array, conf_coord, downsample=False, sample_size=None):
    label_array = labeled_array(image_array, conf_coord)
    
    X = image_array.reshape(-1, 1)
    y = label_array.reshape(-1, 1)
    view_neighborhood(conf_coord, label_array)
    
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    
    # if downsample:
    #     label_array = label_array[::downsample, ::downsample]
    #     image_array = image_array[::downsample, ::downsample]
    
    
    return 0

def main_():
    image_choice = False
    conf_coord, image_array = pre_process(image_choice)
    conf_coord = list(conf_coord)
    svm(image_array, conf_coord, downsample=1, sample_size=None)
    
if __name__ == '__main__':
    main_()
    