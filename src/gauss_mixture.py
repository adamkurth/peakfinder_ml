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

def guassian_mixture_model(features, random_state=None):
    """
    Guassian Mixture Model:
    - probabilistic model for representing normally distributed subpopulations within an overall population
    - think of best as a generalization of k-means clustering
    - when calculating the covariance matrix becomes too difficult, known to diverge (infinite likelihoods)
    - Recall: A covariance matrix represents the relationships between different variables in a multivariate distribution. 

    Args:
        features (array): Array of shape (n_samples, n_features) representing the input data. Each row corresponds to a sample, and each column corresponds to a feature.
        random_state (int, RandomState instance or None, optional): Random state for reproducible results. Defaults to None.

    Returns:
        best_n (int): The optimal number of clusters determined by BIC.
        labels (array): Array of shape (n_samples,) representing the predicted cluster labels for each sample.
        gmm (GaussianMixture): The trained Gaussian Mixture Model.
    """    
    
    lowest_bic = np.infty
    best_n = None
    best_labels = None
    best_gmm = None

    for n in range(1, 11):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=random_state)
        gmm.fit(features)
        bic = gmm.bic(features)
        if bic < lowest_bic:
            lowest_bic = bic
            best_n = n
            best_labels = gmm.predict(features)
            best_gmm = gmm

    print("Initial BIC:", gmm.bic(features))
    print("Number of Iterations:", gmm.n_iter_)
    print("Optimal BIC:", lowest_bic, '\n')

    print("Optimal Number of Components:", best_n)
    print("Optimal Covariance Type:", best_gmm.covariance_type)
    print("Optimal Means:\n", best_gmm.means_, '\n')
    
    print("Optimal Weights:", best_gmm.weights_)
    
    avg_cluster_values = [np.mean(features[best_labels==i, 2]) for i in range(best_n)]
    sorted_indices = np.argsort(avg_cluster_values)
    sorted_cluster_values = [np.mean(features[best_labels==i]) for i in sorted_indices]
    percentage_increase = [(sorted_cluster_values[i+1] - sorted_cluster_values[i]) / sorted_cluster_values[i] * 100 for i in range(best_n-1)]
    for i, increase in enumerate(percentage_increase):
        print(f"Percentage Increase from Component {sorted_indices[i]} to Component {sorted_indices[i+1]}: {increase:.2f}%")

    return best_n, best_labels, best_gmm


def visualize_clusters(n, labels, features_array, image_array):
    def plot_clusters(n, labels, features_array):
        colors = plt.cm.Set1(np.linspace(0, 1, n))
        plt.imshow(image_array, cmap='gray')
        plt.gca().set_facecolor('black')
        
        avg_cluster_values = [np.mean(features_array[labels==i, 2]) for i in range(n)]
        sorted_indices = np.argsort(avg_cluster_values)[::-1]  # Sort the indices in descending order
        
        for i in sorted_indices:
            cluster = features_array[labels==i, :]
            plt.scatter(cluster[:, 1], cluster[:, 0], s=5, label=f'Cluster {i}', color=colors[i])
            
            if np.count_nonzero(labels == i) > 0:
                avg_cluster = np.mean(cluster[:, 2])
                plt.text(np.mean(cluster[:, 1]), np.mean(cluster[:, 0]), f'Avg: {avg_cluster:.4f}', color=colors[i], fontsize=8, ha='center', va='center')
        
        avg_legend = [plt.Line2D([0], [0], marker='o', color=colors[i], markerfacecolor='black', markersize=5) for i in sorted_indices]
        avg_labels = [f'Avg: {np.mean(features_array[labels==i, 2]):.4f}' for i in sorted_indices]
        plt.legend(avg_legend + avg_legend, avg_labels + avg_labels, loc='upper right', fontsize='small', numpoints=1)
        plt.show()
        
    def plot_hist(labels, features_array):
        n_components = len(np.unique(labels))
        num_cells = n_components // 2  # Divide the number of components by 2 to get the number of cells
        fig, axs = plt.subplots(2, num_cells, figsize=(12, 8), sharey=True)  # Create a 2xnum_cells grid of subplots
        
        for i in range(n_components):
            if i < len(features_array):
                component = features_array[labels == i, 2]
                row = i // num_cells  # Calculate the row index
                col = i % num_cells  # Calculate the column index
                axs[row, col].hist(component, bins=10, color='skyblue', edgecolor='black')
                axs[row, col].set_title(f'Component {i}')
                axs[row, col].set_xlabel('Intensity')
                axs[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
    def plotly_clusters(labels, features_array):
        fig = go.Figure()
        n_components = len(np.unique(labels))
        
        for i in range(n_components):
            if i < len(features_array):
                component = features_array[labels == i, 2]
                fig.add_trace(go.Histogram(x=component, nbinsx=10, name=f'Component {i}'))
        
        fig.update_layout(
            title="Histogram of Intensity",
            xaxis_title="Intensity",
            yaxis_title="Frequency",
            barmode="overlay",
            bargap=0.1
        )
        
        fig.show()
    
    plot_clusters(n, labels, features_array)
    plot_hist(labels, features_array)
    plotly_clusters(labels, features_array)

if __name__ == '__main__':
    image_choice = False    # True = work, False = home
    confirmed_coordinates, image_array = pre_process(image_choice)
    features_array = cluster(confirmed_coordinates, image_array)
    best_n, labels, best_gmm = guassian_mixture_model(features_array)
    # visualize_clusters(best_n, labels, features_array, image_array)
 