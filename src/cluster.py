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
    visualize,
    main,
    )                     
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN



image_array, path = load_data(False)
processor = PeakThresholdProcessor(image_array, 500)
coord_manual = main(path, 500, display=False)
coord_manual = [tuple(coord) for coord in coord_manual]
coord_script = processor.get_local_maxima()

confirmed_coordinates, _, _ = validate(coord_manual, coord_script, image_array)
# visualize(confirmed_coordinates, coord_manual, coord_script, image_array)
confirmed_coordinates = list(confirmed_coordinates)

# find corresponding intensities
intensities = [image_array[coord[0], coord[1]] for coord in confirmed_coordinates]

features = np.column_stack((confirmed_coordinates, intensities))
features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# clustering 
clustering = DBSCAN(eps=0.5, min_samples=5).fit(features_normalized)

# labels 
labels = clustering.labels_
# print(labels)
