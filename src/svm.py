import os
import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from label_finder import(
    PeakThresholdProcessor,
    ArrayRegion, 
    # load_data, 
    load_file_h5,
    display_peak_regions, 
    validate,
    is_peak,
    view_neighborhood, 
    generate_labeled_image, 
    visualize,
    main,
    )                     
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm 
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

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

def downsample_data(X, y, sample_size):
    # ensure sample size is smaller than total
    assert sample_size < X.shape[0], "Sample size must be smaller than total number of samples."
    # get random indices
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    # extract
    X_downsampled = X[indices]
    y_downsampled = y[indices]
    return X_downsampled, y_downsampled
    
def svm_hyperparameter_tuning(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']}
    grid_search = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3) #5 fold cross validation
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters found : {grid_search.best_params_}')
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

def svm_classification(X, y, downsample=False, sample_size=None):
    if downsample and sample_size:
        X, y = downsample_data(X, y, sample_size)
    
    X_flattened = X.reshape(-1, 1)  # Reshape X to (n_samples, n_features)
    y_flattened = y.ravel()  # Flatten y to 1D
    
    print("Shape of X before splitting:", X_flattened.shape)
    print("Shape of y before splitting:", y_flattened.shape)
    
    if X_flattened.shape[0] != y_flattened.shape[0]:
        raise ValueError("The number of samples in X and y must match.")
    
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_flattened, test_size=0.20, random_state=42)
    
    # removing mean and scaling to unit variance
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    
    # hyperparameter tuning 
    best_svm_model = svm_hyperparameter_tuning(X_train, y_train)

    # training best model from hyperparameter tuning
    print("Training the SVM...")
    best_svm_model.fit(X_train, y_train)
    print("Training completed.")
    
    # predict and evaluate
    y_pred = best_svm_model.predict(X_test)
    print('Classification report for SVM: \n', classification_report(y_test, y_pred))
    print('Confusion matrix for SVM: \n', confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    image_choice = False    # True = work, False = home
    confirmed_coordinates, image_array = pre_process(image_choice)
    labeled_image = generate_labeled_image(image_array, confirmed_coordinates, 5)

    svm_classification(image_array, labeled_image, downsample=True, sample_size=1000)
