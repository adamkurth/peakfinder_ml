import os
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import ast # for string to list conversion

from scipy.signal import find_peaks
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
                  
def display_peak_regions(image_array, coordinates):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_array, cmap='viridis')
    # plt.title("Intensity")
    for i, (x, y) in enumerate(coordinates, 1):
        plt.scatter(y, x, marker='x', color='red')
        plt.text(y, x, f'{i+1}', color='white', ha='right')    

    plt.title(f"Peak Regions (size={image_array.shape})")
    plt.show()

def validate(manual, script, image_array):
    manual_set = set(manual)
    script_set = set(script)
    common = manual_set.intersection(script_set)
    unique_manual = manual_set.difference(script_set)
    unique_script = script_set.difference(manual_set)
    
    print(f'common: {common}\n')
    print(f'unique_manual: {unique_manual}\n')
    print(f'unique_script: {unique_script}\n')
    
    confirmed_common = {coord for coord in common if is_peak(image_array, coord)}
    confirmed_unique_manual = {coord for coord in unique_manual if is_peak(image_array, coord)}
    confirmed_unique_script = {coord for coord in unique_script if is_peak(image_array, coord)}
    
    print(f'confirmed_common: {confirmed_common}\n')
    print(f'confirmed_unique_manual: {confirmed_unique_manual}\n')
    print(f'confirmed_unique_script: {confirmed_unique_script}\n')

    return confirmed_common, confirmed_unique_manual, confirmed_unique_script

def is_peak(image_data, coordinate, neighborhood_size=3):
    x, y = coordinate
    region = ArrayRegion(image_data)
    
    neighborhood = region.extract_region(x, y, neighborhood_size)    
    center = neighborhood_size, neighborhood_size
    is_peak = neighborhood[center] == np.max(neighborhood)
    if is_peak:
        print(f'Peak found at {coordinate}')
    return is_peak

def view_neighborhood(coordinates, image_data):
    coordinates = list(coordinates)
    
    print("List of coordinates:")
    for i, (x, y) in enumerate(coordinates, 1):
        print(f'{i}. ({x}, {y})')

    while True:
        ans = input(f'Which coordinate do you want to view? (1-{len(coordinates)} or "q" to quit) \n')

        if ans.lower() == "q":
            print("Exiting")
            break

        try:
            ans = int(ans) - 1  # Convert to 0-based index
            if 0 <= ans < len(coordinates):
                coordinate = coordinates[ans]
                x, y = coordinate
                
                region = ArrayRegion(image_data)
                neighborhood = region.extract_region(x_center=x, y_center=y, region_size=3)
                                
                # Determine if the coordinate is a peak
                center = neighborhood.shape[0] // 2, neighborhood.shape[1] // 2
                is_peak = neighborhood[center] == np.max(neighborhood)
                
                print(f'Neighborhood for ({x}, {y}):')
                print(neighborhood)
                
                if is_peak:
                    print("This is a peak.")
                else:
                    print("This is not a peak.")
        
                # continue?
                cont = input('Do you want to view another neighborhood? (Y/n) ').strip().lower()
                if cont in ['n', 'no']:
                    print("Exiting")
                    break
                else:
                    view_neighborhood(coordinates, image_data)  # Recursive call
            else:
                print(f"Please enter a number between 1 and {len(coordinates)}.")
                
        except ValueError:
            print("Invalid choice. Please enter a number or 'q' to quit.")
        except IndexError:
            print("Invalid choice. Try again.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
def main(file_path, threshold_value):
    image_array = load_file_h5(file_path) # load_file_h5
    threshold_processor = PeakThresholdProcessor(image_array, threshold_value)
    coordinates = threshold_processor.get_coordinates_above_threshold()
    # display    
    print(f'Found {len(coordinates)} peaks above threshold {threshold_value}')
    # display_peak_regions(image_array, coordinates)
    return coordinates

if __name__ == "__main__":
    work = False
    image_data, file_path = load_data(work) # from test.py in peakfinder_ml
    threshold = 1000
    coordinates_array = main(file_path, threshold)
    coordinates = [tuple(coord) for coord in coordinates_array]
    print(f'manually found coordinates {coordinates}\n')
    
    threshold_processor = PeakThresholdProcessor(image_data, threshold)
    peaks = threshold_processor.get_local_maxima()
    
    confirmed_common_peaks, _, _ = validate(coordinates, peaks, image_data) # manually found and script found
    confirmed_common_peaks = list(confirmed_common_peaks)
    print(confirmed_common_peaks)
    view_neighborhood(confirmed_common_peaks, image_data)
    