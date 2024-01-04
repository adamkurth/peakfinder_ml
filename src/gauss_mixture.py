import os
import glob
import h5py as h5
import numpy as np
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
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture

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

def cluster(confirmed_coordinates, image_array):
    # find corresponding intensities
    intensities = [image_array[x, y] for x,y in confirmed_coordinates]
    #combine coordinates and intensities
    features = [[x, y, intensity] for (x, y), intensity in zip(confirmed_coordinates, intensities)] 
    features_array = np.array(features)
    return features_array    

def guassian_mixture_model(n, features, covariance_type='full', random_state=None):
    """
    Guassian Mixture Model:
    - probabilistic model for representing normally distributed subpopulations within an overall population
    - think of best as a generalization of k-means clustering
    - when calculating the covariance matrix becomes too difficult, known to diverge (infinite likelihoods)
    - Recall: A covariance matrix represents the relationships between different variables in a multivariate distribution. 
    Args:
        n (int): number of clusters
        features (array): Array of shape (n_samples, n_features) representing the input data. Each row corresponds to a sample, and each column corresponds to a feature.
        covariance_type (str, optional): Covariance type for the Gaussian Mixture Model. Options are 'full', 'tied', 'diag', 'spherical'. Defaults to 'full'.
        random_state (int, RandomState instance or None, optional): Random state for reproducible results. Defaults to None.

    Returns:
        labels (array): Array of shape (n_samples,) representing the predicted cluster labels for each sample.
    """    
    gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=random_state)
    gmm.fit(features)
    labels = gmm.predict(features)
    return labels


def plot_clusters(n, labels, features_array):
    colors = plt.cm.Pastel1(np.linspace(0, 1, n))
    plt.imshow(image_array, cmap='grey')
    for i in range(n):
        cluster = features_array[labels==i, :]
        plt.scatter(cluster[:, 1], cluster[:, 0], s=5, label=f'Cluster {i}', color=colors[i])
        
        if np.count_nonzero(labels == i) > 0:
            avg_cluster = np.mean(cluster[:, 2])
            plt.text(np.mean(cluster[:, 1]), np.mean(cluster[:, 0]), f'Avg: {avg_cluster:.4f}', color=colors[i], fontsize=8, ha='center', va='center')
    plt.legend(loc='upper right', numpoints=1)
    plt.show()

def plot_hist(labels, features_array):
    n_components = len(np.unique(labels))
    fig, axs = plt.subplots(1, n_components, figsize=(12, 4))
    
    for i in range(n_components):
        component = features_array[labels == i, 2]
        axs[i].hist(component, bins=10, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Component {i}')
        axs[i].set_xlabel('Intensity')
        axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    image_choice = False    # True = work, False = home
    confirmed_coordinates, image_array = pre_process(image_choice)
    features_array = cluster(confirmed_coordinates, image_array)
    labels = guassian_mixture_model(2, features_array)
    plot_clusters(2, labels, features_array)
    # plot_hist(labels, features_array)
   