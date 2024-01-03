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
    validate,
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

    def get_coordinates_above_threshold(self):
        # convert to boolean mask
        mask = self.image_tensor > self.threshold_value
        # indices of True values in the mask
        coordinates = torch.nonzero(mask).cpu().numpy()
        return coordinates

    def get_local_maxima(self):
        # relies on 'find_peaks' which works on 1D arrays.
        image_1d = self.image_tensor.flatten().cpu().numpy()  # to numpy for compatibility with 'find_peaks'
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates

    def flat_to_2d(self, index):
        rows, cols = self.image_tensor.shape
        return (index // cols, index % cols)

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
            # Check if the data is already 3D (H, W, C), and if not, add a channel dimension
            if len(data.shape) == 2:
                data = data[np.newaxis, :, :]  # Add a channel dimension (C, H, W)
            tensor = torch.from_numpy(data).float()
            print(f'Loaded data from {file_path} with shape: {tensor.shape}')
            tensor_list.append(tensor)

    if not tensor_list:
        raise ValueError("No .h5 files found or empty dataset in files.")

    # Stack all tensors along the first dimension to create a single tensor
    combined_tensor = torch.stack(tensor_list)

    # Check if the combined tensor has 4 dimensions (N, H, W, C), and if so, permute to (N, C, H, W)
    if combined_tensor.dim() == 4:
        combined_tensor = combined_tensor.permute(0, 3, 1, 2)

    return combined_tensor, directory_path

def is_local_max(image, x, y, neighborhood_size):
    """Check if the pixel at (x, y) is a local maximum within the specified neighborhood."""
    half_size = neighborhood_size // 2
    neighborhood = image[max(0, x - half_size):x + half_size + 1, 
                         max(0, y - half_size):y + half_size + 1]

    return image[x, y] == torch.max(neighborhood)

def generate_label_tensor(image_tensor, neighborhood_size=3):
    """Generate a tensor of the same shape as image_tensor, marking 1 at peaks and 0 elsewhere."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor should be a PyTorch tensor")
    
    label_tensor_list = []
    for img_idx in range(image_tensor.shape[0]):
        label_tensor = torch.zeros_like(image_tensor[img_idx, :, :])
        # loop over every pixel in image
        for x in range(image_tensor.shape[1]):
            for y in range(image_tensor.shape[2]):
                if is_local_max(image_tensor[img_idx, :, :], x, y, neighborhood_size):
                    label_tensor[x, y] = 1  # Mark as peak
                    print(f'Peak found in image {img_idx} at ({x}, {y})')
                    
        label_tensor_list.append(label_tensor) 
    combined_label_tensor = torch.stack(label_tensor_list)
    return combined_label_tensor



if __name__ == '__main__':
    # work_dir = ''
    home_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
    combined_tensor, directory_path = load_tensor(home_dir)
    print("Type of combined_tensor:", type(combined_tensor))
    print("Shape of combined_tensor:", combined_tensor.shape)
    
    if isinstance(combined_tensor, torch.Tensor):
        combined_label_tensor = generate_label_tensor(combined_tensor)
    else:
        print("combined_tensor is not a PyTorch tensor.")
    
    # combined_label_tensor = generate_label_tensor(combined_tensor)
    # print(combined_tensor, combined_label_tensor)
    