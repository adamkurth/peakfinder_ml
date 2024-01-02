import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from test import load_data


def detect_peaks(image):
    # convert to 1D array
    image_1d = image.flatten()
    
    # detect peaks using simple max threshold 
    peaks, _ = find_peaks(image_1d, height=500)
    
    # create empty 1d array
    labels = np.zeros_like(image_1d)
    
    # mark peaks in label array
    labels[peaks] = 1
    return labels

def manual_label_correction(labels, image):
    plt.imshow(image, cmap='gray')
    plt.title('Enter the index of the incorrect peaks (comma-separated): ')
    plt.show()
    
    incorrect_indices = input('Indices of incorrect peaks: ')
    incorrect_indices = [int(idx) for idx in incorrect_indices.split(',')]
    
    for idx in incorrect_indices:
        #flip te label for specified index
        labels[idx] = 1 - labels[idx]

    return labels

def main():
    test_dir_work = False
    image_data = load_data(test_dir_work)

    initial_labels = detect_peaks(image_data)
    print('initial labels\n', initial_labels)
    corrected_labels = manual_label_correction(initial_labels, image_data)
    print('corrected labels\n', corrected_labels)
    
if __name__ == '__main__':
    main()