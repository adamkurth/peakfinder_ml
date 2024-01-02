import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from test import load_data

class PeakThresholdProcessor: 
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates

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
    
def load_file_h5(file_path):
    try:
        with h5.File(file_path, "r") as f:
            data = np.array(f["entry/data/data"][()])
            print("File loaded successfully.")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
    return None
               
def extract_region(image_array, region_size, x_center, y_center):
    extract = ArrayRegion(image_array)
    extract.set_peak_coordinate(x_center,y_center)
    extract.set_region_size(region_size)
    np.set_printoptions(floatmode='fixed', precision=10)
    np.set_printoptions(edgeitems=3, suppress=True, linewidth=200)
    region = extract.get_region()
    return region        
    
def display_peak_regions(image_array, coordinates, region_size=3):
    region = ArrayRegion(image_array)
    for i, (x, y) in enumerate(coordinates, 1):
        region = extract_region(image_array, region_size, x, y)
        plt.imshow(region, cmap='viridis')
        plt.title(f"Peak {i} at ({x}, {y})")
        plt.colorbar(label='Intensity')
        plt.show()

def main(file_path, threshold_value):
    image_array = load_file_h5(file_path) # load_file_h5
    threshold_processor = PeakThresholdProcessor(image_array, threshold_value)
    coordinates = threshold_processor.get_coordinates_above_threshold()
    display_peak_regions(image_array, coordinates, region_size=5)
    
    print(f'Found {len(coordinates)} peaks above threshold {threshold_value}')
    display_peak_regions(image_array, coordinates, region_size=5)

if __name__ == "__main__":
    work = False
    image_data, file_path = load_data(work) # from test.py in peakfinder_ml
    threshold = 1000
    main(file_path, threshold)