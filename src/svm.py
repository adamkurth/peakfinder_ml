import os
import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from label_finder import(
    PeakThresholdProcessor,
    ArrayRegion, 
    # load_data, 
    load_file_h5,
    display_peak_regions, 
    validate,
    is_peak,
    view_neighborhood, 
    # generate_labeled_image, 
    visualize,
    main,
    )                     
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm 
# import sklearn.svm as sklearn_svm

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_gradient_magnitude, generic_filter
from skimage.feature import peak_local_max
from skimage.morphology import dilation, square

class PeakThresholdProcessor: 
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates
    
    def get_local_maxima(self):
        image_1d = self.image_array.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def flat_to_2d(self, index):
        shape = self.image_array.shape
        rows, cols = shape
        return (index // cols, index % cols) 
class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0
    
    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y
    
    def set_region_size(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) //2
        self.region_size = min(size, max_printable_region)
    
    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

    def extract_region(self, x_center, y_center, region_size):
            self.set_peak_coordinate(x_center, y_center)
            self.set_region_size(region_size)
            region = self.get_region()

            # Set print options for better readability
            np.set_printoptions(precision=8, suppress=True, linewidth=120, edgeitems=7)
            return region
    
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
    
def load_file_h5(file_path):
    try:
        with h5.File(file_path, "r") as f:
            data = np.array(f["entry/data/data"][()])
            print(f"File loaded successfully: \n {file_path}")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        
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

def svm(image_array, confirmed_coordinates):
    def downsample_data(X, y, sample_size):
        if sample_size >= X.shape[0]:
            raise ValueError("Sample size must be smaller than the total number of samples.")
        
        # Get random indices for each class
        indices_class_0 = np.where(y == 0)[0]
        indices_class_1 = np.where(y == 1)[0]
        
        if len(indices_class_0) < sample_size // 2 or len(indices_class_1) < sample_size // 2:
            raise ValueError("Insufficient samples in one or both classes to achieve the desired sample size.")
    
        # Randomly select samples for each class
        selected_indices_class_0 = np.random.choice(indices_class_0, sample_size // 2, replace=False)
        selected_indices_class_1 = np.random.choice(indices_class_1, sample_size // 2, replace=False)
        
        # Combine the selected samples
        selected_indices = np.concatenate([selected_indices_class_0, selected_indices_class_1])
            
        # Shuffle the selected indices to mix both classes
        np.random.shuffle(selected_indices)
        
        # Extract the downsampled data
        X_downsampled = X[selected_indices]
        y_downsampled = y[selected_indices]
        
        print(f'Downsampled data to {sample_size} samples. New shape: {X_downsampled.shape}, {y_downsampled.shape}\n')  
        print(f'Unique labels: {np.unique(y_downsampled)}, Class counts: {np.bincount(y_downsampled)}\n')
        print(f'Downsampling completed.\n')
        return X_downsampled, y_downsampled
        
    def svm_hyperparameter_tuning(X_train, y_train):
        import sklearn.svm as sklearn_svm

        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(f"The number of classes in the training set must be greater than one; got {len(unique_classes)} class(es).")
        
        param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                    'kernel': ['rbf']}        
        # tuning
        grid_search = GridSearchCV(sklearn_svm.SVC(), param_grid, refit = True, verbose = 3, error_score='raise') #5 fold cross validation

        # fit the grid search model
        grid_search.fit(X_train, y_train)
        
        print(f'Best parameters found: {grid_search.best_params_}')
        return grid_search.best_estimator_ 
        
    def svm_cross_validation(svm_model, X, y, cv=5):
        scores = cross_val_score(svm_model, X, y, cv=cv)
        print(f"Cross validation scores: {scores}")
        print(f"Mean cross validation score: {scores.mean():.3f}")
        print(f"Standard deviation of cross validation scores: {scores.std():.3f}")

    def apply_pca(X, n_components=2):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print(f'Original number of features: {X.shape[1]}')
        print(f'Reduced number of features: {X_pca.shape[1]}')
        return X_pca
    
    def extract_features(image_array, confirmed_coordinates, neighborhood_size=5, threshold=200):
        """Extracts features from the image array.
        Args:
            image_array (np.array): image data
            confirmed_coordinates (list(tuples)): size of neighborhood to be considered
            neighborhood_size (int, optional): defaults to 5
            threshold (int, optional): threshold to be considered a peak
        Returns:
            features (np.array): a feature array where each row corresponds to a peak
        """
        # gradient magnitude
        grad_mag = gaussian_gradient_magnitude(image_array, sigma=1)
        dilated_image = dilation(image_array, square(neighborhood_size))
        
        def local_stats(neighborhood, threshold=400):
            center = neighborhood[neighborhood_size // 2, neighborhood_size // 2]
            return [
                np.max(neighborhood) - center, # max - center
                np.mean(neighborhood > threshold), # fraction of high values
            ]
        
        features_list = []
        
        for coord in confirmed_coordinates:
            x, y = coord
            # region around peak
            region = ArrayRegion(image_array).extract_region(x, y, neighborhood_size)
            # compute the local stats for the region
            local_feature = local_stats(region, threshold)
            
            # extract other features such as gradient magnitude and dilated image
            grad_feature = grad_mag[x, y]
            dilated_feature = dilated_image[x, y]
            
            # combine features
            peak_features = np.array([image_array[x, y], grad_feature, dilated_feature] + local_feature)
            
            features_list.append(peak_features)
            
        features = np.stack(features_list, axis=0)
        return features
    
    def labeled_peaks(image_array, conf_coordinates):
        labeled_array = np.zeros_like(image_array, dtype=int)
        for coord in conf_coordinates:
            x, y = coord
            if is_peak(image_array, (x, y), neighborhood_size=3):
                labeled_array[x, y] = 1
        return labeled_array    
    
    def svm_classification(image_array, confirmed_coordinates, downsample=False, sample_size=None):
        # Initialize an empty labeled array with the same shape as the image_array
        labeled_array = labeled_peaks(image_array, confirmed_coordinates)
        
        unique_labels = np.unique(labeled_array)
        print(f"Unique labels: {unique_labels}")        
        
        num_peaks = np.sum(labeled_array == 1)
        num_non_peaks = np.sum(labeled_array == 0)
        print(f"Number of peaks: {num_peaks}, Number of non-peaks: {num_non_peaks}")

        X = extract_features(image_array, confirmed_coordinates, neighborhood_size=5, threshold=200)
        X_flat = X.reshape(-1, 1)
        y_flat = labeled_array.ravel()
        y_flat = y_flat.astype(int)
        
        classes = np.unique(y_flat)
        print(f"Unique classes: {classes}")
        
        if downsample and sample_size:
            X, y = downsample_data(X_flat, y_flat, sample_size)
            
        print(f'Shape after downsampling: {X.shape}, {y.shape}')            
        
        print(f'Shape before splitting: {X.shape}, {y.shape}')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        print(f'Shape after splitting: {X_train.shape}, {y_train.shape}')
                
        # # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Hyperparameter tuning
        best_svm_model = svm_hyperparameter_tuning(X_train, y_train)
        
        # Training best model from hyperparameter tuning
        print("Training the SVM...")
        best_svm_model.fit(X_train, y_train)
        print("Training completed.")
        
        # # Predict and evaluate
        y_pred = best_svm_model.predict(X_test)
        
        print('Classification report for SVM: \n', classification_report(y_test, y_pred))
        print('Confusion matrix for SVM: \n', confusion_matrix(y_test, y_pred))
        
        return best_svm_model, X_test, y_test, y_pred, confusion_matrix(y_test, y_pred)
    

    best_svm_model, X_test, y_test, y_pred, confusion_matrix = svm_classification(image_array, confirmed_coordinates, downsample=True, sample_size=10)
    
if __name__ == "__main__":
    image_choice = False    # True = work, False = home
    confirmed_coordinates, image_array = pre_process(image_choice)
    svm(image_array, confirmed_coordinates)



#    visualizing the confusion matrix: 
#       - True Negative (top left) negative samples correctly identified of not peak
#       - False Positive (top right) negative samples incorrectly identified as peak
#       - False Negative (bottom left) positive samples incorrectly identified as not peak
#       - True Positive (bottom right) positive samples correctly identified as peak
# 
#    - Precision: TP / (TP + FP) = TP / predicted positive
#    - Recall: TP / (TP + FN) = TP / actual positive