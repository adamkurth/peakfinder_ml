import os
import glob
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from label_finder import(
    load_data, 
    load_file_h5,
    display_peak_regions, 
    # validate,
    is_peak,
    view_neighborhood, 
    generate_labeled_image, 
    main,
    )                     


class PeakThresholdProcessor:
    def __init__(self, image_tensor, threshold_value=0):
        self.image_tensor = image_tensor
        self.threshold_value = threshold_value

    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value

    def get_coordinates_above_threshold(self, image):
        # convert to boolean mask 'image' is a 2D tensor (H, W)
        mask = image > self.threshold_value
        coordinates = torch.nonzero(mask).cpu().numpy()
        return coordinates

    def get_local_maxima(self, image):
        # from 2D image to 1D
        image_1d = image.flatten().cpu().numpy()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx, image.shape[1]) for idx in peaks]
        return coordinates
    
    def flat_to_2d(self, index, width):
        return (index // width, index % width)
    
class ArrayRegion:
    def __init__(self, tensor):
        self.tensor = tensor
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0

    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y

    def set_region_size(self, size):
        max_printable_region = min(self.tensor.shape[0], self.tensor.shape[1]) // 2
        self.region_size = min(size, max_printable_region)

    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size + 1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size + 1)
        region = self.tensor[x_range, y_range] # tensor slicing
        return region

    def extract_region(self, x_center, y_center, region_size):
        self.set_peak_coordinate(x_center, y_center)
        self.set_region_size(region_size)
        return self.get_region()
    
def load_tensor(directory_path):
    file_pattern = directory_path + '*.h5'
    tensor_list = []

    for file_path in glob.glob(file_pattern):
        with h5.File(file_path, 'r') as file:
            # Assuming 'entry/data/data' is the correct path within your .h5 files
            data = np.array(file['entry/data/data'][:])
            # Ensure data is 2D (H, W), if not, you might need to adjust or provide additional context
            if len(data.shape) == 3 and data.shape[2] == 1:
                data = data[:, :, 0]  # If it's (H, W, 1), convert to (H, W)
            elif len(data.shape) != 2:
                raise ValueError(f"Data in {file_path} has an unexpected shape: {data.shape}")

            tensor = torch.from_numpy(data).unsqueeze(0).float()  # Add a batch dimension (1, H, W)
            print(f'Loaded data from {file_path} with shape: {tensor.shape}')
            tensor_list.append(tensor)

    if not tensor_list:
        raise ValueError("No .h5 files found or empty dataset in files.")

    # Stack all tensors along the first dimension to create a single tensor
    combined_tensor = torch.cat(tensor_list, dim=0)  # (N, H, W)

    print(f"Combined tensor shape: {combined_tensor.shape}")
    return combined_tensor, directory_path

def is_peak(image_tensor, coordinate, neighborhood_size=3):
    x, y = coordinate
    region = ArrayRegion(image_tensor)
    
    neighborhood = region.extract_region(x, y, neighborhood_size)
    if torch.numel(neighborhood) == 0:  # empty
        return False
    
    center = neighborhood_size // 2, neighborhood_size // 2
    is_peak = neighborhood[center] == torch.max(neighborhood)
    return is_peak

def generate_label_tensor(image_tensor, neighborhood_size=3):
    """Generate a tensor of the same shape as image_tensor, marking 1 at peaks and 0 elsewhere."""
    #N: number of images, 
    # C: number of channels, 
    # H: height,
    # W: width
    
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor should be a PyTorch tensor with shape (N, H, W)")

    if len(image_tensor.shape) == 4 and image_tensor.shape[1] == 1:  # If it's (N, C, H, W) with C=1
        image_tensor = image_tensor.squeeze(1)  # Remove the channel dimension

    elif len(image_tensor.shape) != 3:  # Ensure it's (N, H, W)
        raise ValueError("Unsupported tensor shape. Expected (N, H, W)")

    label_tensor_list = []
    for img_idx in range(image_tensor.shape[0]):  # Loop over images in tensor
        label_tensor = torch.zeros_like(image_tensor[img_idx, :, :])  # (H, W)

        for x in range(image_tensor.shape[1]):  # height
            for y in range(image_tensor.shape[2]):  # width
                if is_local_max(image_tensor[img_idx, :, :], x, y, neighborhood_size):
                    label_tensor[x, y] = 1  # peak

        label_tensor_list.append(label_tensor)

    combined_label_tensor = torch.stack(label_tensor_list)
    return combined_label_tensor

def find_coordinates(combined_tensor):
    coord_list_manual = []
    coord_list_script = []
    processor = PeakThresholdProcessor(combined_tensor, threshold_value=1000)
    confirmed_common_list = []

    for img_idx, img in enumerate(combined_tensor):
        print(f'Processing Image {img_idx}')
        # manual 
        coord_manual = processor.get_coordinates_above_threshold(img)
        coord_list_manual.append(coord_manual)
        # script
        coord_script = processor.get_local_maxima(img)
        coord_list_script.append(coord_script)
        # validate for img 
        confirmed_common, _, _ = validate(coord_manual, coord_script, img)
        confirmed_common_list.append(confirmed_common)

    return confirmed_common_list
    
def validate(manual, script, image_array):
    manual_set = set([tuple(x) for x in manual])
    script_set = set([tuple(x) for x in script])
    
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
### INSERT KNOWN COORDINATES FOR PEAKS IN TENSOR ^^

if __name__ == '__main__':
    # work_dir = ''
    home_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
    combined_tensor, directory_path = load_tensor(home_dir)
    print("Type of combined_tensor:", type(combined_tensor))
    print("Shape of combined_tensor:", combined_tensor.shape)
    
    coordinates_list = find_coordinates(combined_tensor)
    # combined_label_tensor = generate_label_tensor(combined_tensor)
    
    # combined_label_tensor = generate_label_tensor(combined_tensor)
    # print(combined_tensor, combined_label_tensor)
    