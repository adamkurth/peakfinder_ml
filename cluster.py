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
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN

image_array = load_data(False)
threshold = 1000
processor = PeakThresholdProcessor(image_array, threshold)
coordinates = processor.get_coordinates_above_threshold()

# find corresponding intensities
intensities = image_array[coordinates[:, 0], coordinates[:, 1]]
features = np.column_stack((coordinates, intensities))
features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# clustering 
clustering = DBSCAN(eps=0.5, min_samples=5).fit(features_normalized)

# labels 
labels = clustering.labels_
