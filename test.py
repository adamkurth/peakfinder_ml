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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def data_preparation(image_data, labeled_image, coordinates, patch_size):
    def split_into_patches(image_data, patch_size):
        """
        Split image into patches of size patch_size x patch_size
        
        :param image_data: 2d numpy array representing the image.
        :param patch_size: int, size of the patch.
        :return: list of 2d numpy arrays representing the patches.
        """
        patches = []
        for i in range(0, image_data.shape[0], patch_size):
            for j in range(0, image_data.shape[1], patch_size):
                # Adjust the patch size for edge cases
                patch = image_data[i:min(i + patch_size, image_data.shape[0]), j:min(j + patch_size, image_data.shape[1])]
                
                # Pad the patch to ensure uniform size if it's an edge case
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), 'constant')
                
                patches.append(patch)
        return patches

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
    
    def generate_labels(patches, labeled_image, coordinates, patch_size):
        """
        Generates labels for each patch in patches.
        :param patches: list of 2d numpy arrays representing the patches.
        :param labeled_image: the fully labeled image from generate_labeled_image.
        :param coordinates: the (row, col) coordinates of the peaks in the full image
        :param patch_size: int, size of the patch.
        :return: list of labels for each patch.
        """
        y = np.array([
        label_for_each_patch(
            image_data[i*patch_size:min((i+1)*patch_size, image_data.shape[1]),
                    j*patch_size:min((j+1)*patch_size, image_data.shape[2])],
            labeled_image,
            (i * patch_size, j * patch_size),
            coordinates
        )
        for i in range((image_data.shape[1] + patch_size - 1) // patch_size)        for j in range((image_data.shape[2] + patch_size - 1) // patch_size)
        ])
        return y
    
    # preprocessing
    patches = split_into_patches(image_data[0], patch_size)
    X = np.array([patch.reshape(-1) for patch in patches])
    y = generate_labels(patches, labeled_image, coordinates, patch_size)
    
    return X, y

def train(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, y_pred):
    """Evaluates the trained model using various metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")
    print(f"Model Precision: {precision}")
    print(f"Model Recall: {recall}")
    print(f"Model F1 Score: {f1}")


def main():
    work = False
    threshold = 1000
    image_data, file_path = load_data(work)
    coordinates = main(file_path, threshold, display=False)
    coordinates = [tuple(coord) for coord in coordinates]
    
    labeled_image = generate_labeled_image(image_data, coordinates)

    if len(image_data.shape) == 2:
        image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1]) # 3d
        print(image_data.shape)
       
    X, y = data_preparation(image_data, labeled_image, coordinates, patch_size=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = train(X_train, y_train)    
    
    y_pred = model.predict(X_test)
    
    evaluate_model(model, X_test, y_test, y_pred)

if __name__ == '__main__'
    main()