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
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest

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

def isolation_forest(image_array, n_estimators=100, contamination='auto'):
    """Applies the Isolation Forest to the features array.
    Args:
        :image_array (np.array): the data array to be clustered.
        :n_estimators (int): The number of base estimators in the the ensemble.
        :contamination (str): Isolation forest model and anaomaly score for each data point.
    """
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    clf.fit(image_array)
    
    # anomaly scores (lower = more abnormal)
    scores = clf.decision_function(image_array)

    # prediction: (-1 = outlier, 1 = inlier)
    predictions = clf.predict(image_array)
    print('count', np.count_nonzero(predictions != 1))   
    print("Isolation Forest Evaluation:")
    print("----------------------------")
    print("Number of estimators:", clf.n_estimators)
    print("Contamination:", clf.contamination)
    print("Anomaly scores:", scores)
    print("Predictions:", predictions)
    return clf, scores, predictions

def plot_results(data, anomalies):
    data_x = [coord[0] for coord in data]
    data_y = [coord[1] for coord in data]
    anomalies_x = [anomaly[0] for anomaly in anomalies]
    anomalies_y = [anomaly[1] for anomaly in anomalies]
    
    plt.scatter(data_x, data_y, color='k', s=3, label='Data Points')  # All data points
    plt.scatter(anomalies_x, anomalies_y, color='r', s=10, label='Anomalies')  # Anomalies in red
    plt.title('Isolation Forest Anomalies')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    image_choice = False    # True = work, False = home
    confirmed_coordinates, image_array = pre_process(image_choice)
    
    clf, scores, predictions = isolation_forest(image_array)
    anomalies = image_array[predictions == -1]
    plot_results(confirmed_coordinates, anomalies)
    
    
    