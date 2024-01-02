import os
import glob
import h5py as h5
import numpy as np
from label_finder import(
    PeakThresholdProcessor,
    ArrayRegion, 
    load_data, 
    load_file_h5,
    display_peak_regions, 
    validate,
    is_peak,
    view_neighborhood, 
    generate_labeled_image, 
    main,
    )                     
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def split_into_patches(image_data, patch_size):
    """
    Split image into patches of size patch_size x patch_size
    
    :param image_data: 2d numpy array representing the image.
    :param patch_size: int, size of the patch.
    :return: list of 2d numpy arrays representing the patches.
    """
    patches = []
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            # Determine the size of the patch, adjusting for edges
            patch_height = min(patch_size, image.shape[0] - i)
            patch_width = min(patch_size, image.shape[1] - j)

            # Extract the patch
            patch = image[i:i+patch_height, j:j+patch_width]

            # If necessary, pad the patch so all patches have the same size
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), 'constant')
            
            patches.append(patch)

def label_for_each_patch(patch, labeled_image, patch_coordinates, peak_coordinates):
    """
    Determines the label for a patch based on the labeled image
    :param patch: The patch of the original image
    :param labeled_image: the fully labeled image from generate_labeled_image.
    :param patch_coordinates: the (row, col) coordinates of the top-left pixel in the patch of the full image
    :param peak_coordinates: the (row, col) coordinates of the peak in the full image
    :return: the label for the patch (1 if contains peak, 0 otherwise)
    """
    patch_size = patch.shape[0]
    start_row, start_col = patch_coordinates
    end_row, end_col = start_row + patch_size, start_col + patch_size
    
    # extract the corresponding part of the labeled_image
    patch_labeled_image = labeled_image[start_row:end_row, start_col:end_col]
    
    # check if any parts are labeled as a peak
    if np.any(patch_labeled_image == 1):
        return 1
    else: 
        return 0
    

if __name__ == '__main__':
    work = False
    threshold = 1000
    image_data, file_path = load_data(work)
    coordinates = main(file_path, threshold, display=False)
    coordinates = [tuple(coord) for coord in coordinates]
    
    labeled_image = generate_labeled_image(image_data, coordinates)

    if len(image_data.shape) == 2:
        image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1]) # 3d
        print(image_data.shape)
        
    # split first image into patches (shape: (1, 4371, 4150) )
    patch_size = 50
    patches = split_into_patches(image_data[0], patch_size)
    
    # X is the data matrix and y is the labeled matrix
    # data prep for machine learning
    X = np.array([patch.reshape(-1) for patch in patches])
    # extracts a patch from the image data 
    y = np.array([
        label_for_each_patch(
            image_data[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size], # patch
            labeled_image, # labels
            (i * patch_size, j * patch_size), # patch coordinates
            coordinates
        )
        for i in range(image_data.shape[0] // patch_size) 
        for j in range(image_data.shape[1] // patch_size)
    ]) 
    # standardize data (mean = 0, std = 1)
    scaler = preprocessing.StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  

    # evaluate model performance
    print("Model accuracy: ", model.score(X_test, y_test))