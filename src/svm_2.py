import os
import numpy as np
import h5py as h5
import glob 
import matplotlib.pyplot as plt


from sklearn import svm 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC as svc
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import pair_confusion_matrix
from label_finder import(
    PeakThresholdProcessor,
    ArrayRegion,
    view_neighborhood,
    main,
    validate,
    is_peak,
)

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

def labeled_array(image_array, coord_list):
    labeled = np.zeros(image_array.shape, dtype=int)
    for coord in coord_list: 
        x, y = coord
        if is_peak(image_array, coord, neighborhood_size=5):
            labeled[x, y] = 1
    return labeled # array

def svm_hyperparameter_tuning(X_train, y_train):
    """Hyperparameter tuning for SVM model using GridSearchCV.
    Args:
        X_train (np.array): training data
        y_train (np.array): target values for training data
    Returns:
        grid_search: GridSearchCV object that is the result of the hyperparameter tuning
    """
    print(f'Starting Hyperparameter tuning...')
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                }
    grid_search = GridSearchCV(svc(), param_grid, refit=True, verbose=2, cv=3)
    y_train_flat = y_train.ravel()
    grid_search.fit(X_train, y_train_flat)
    print(f'Finished grid search...\n')
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best estimator: {grid_search.best_estimator_}')
    print(f'Best score: {grid_search.best_score_}')
    print(f'Finishing Hyperparameter tuning...\n')
    return grid_search   
    
def svm_cross_validation(best_model, X, y, cv=5):
    """Cross validation for SVM model.
    Args:
        best_model (SVC): the best model from hyperparameter tuning
        X (np.array): image array data
        y (np.array): labeled array data
        cv (int, optional): Number of folds. Defaults to 5.
    Returns:
        scores, mean_scores: cross validation scores and mean scores
    """
    print(f'Starting cross validation...')
    print(f'Cross validation with {cv} folds...')
    y = y.ravel()
    scores = cross_val_score(best_model, X, y, cv=cv)
    print(f'Cross validation scores: {scores}')
    print(f'Average cross validation score: {np.mean(scores)}')
    print(f'Finishing cross validation...\n')
    return scores, np.mean(scores)

def downsample_data(X, y, random_state=42):
    """Downsamples th majority class to the size of the minority class.
    Args:
        X (np.array): Feature array
        y (np.array): label array
        random_state (int): random state for reproducibility
        
    Returns: X_downsampled, y_downsampled: downsampled feature and label arrays
    """
    #combine X, y into single dataset for resampling
    print(f'Starting downsampling...')
    data = np.hstack((X, y.reshape(-1, 1)))
    
    # identify the major and minority classes
    majority_class = data[data[:, -1] == 0]
    minority_class = data[data[:, -1] == 1]
    
    # downsample the majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,
                                    n_samples=len(minority_class),
                                    random_state=random_state)
    
    # reassemble the downsampled dataset and shuffle
    data_downsampled = np.vstack((majority_downsampled, minority_class))
    np.random.shuffle(data_downsampled)
    
    # split downsampled data into feature and labeled arrays
    X_downsampled = data_downsampled[:, :-1]
    y_downsampled = data_downsampled[:, -1]
    print(f'Finishing downsampling...')
    return X_downsampled, y_downsampled
    
def accuracy_metrics(y_test, y_pred):
    """Prints the accuracy metrics for the model.
    Args:
        y_test (np.array): testing data
        y_pred (np.array): prediction data
    Description:
        Accuracy: Proportion of true positives and negatives from the total num of predictions
        Confusion Matrix: 2x2 matrix of true negative (upper left), false positives (top right), 
            false negatives (bottom left), and true positives (bottom right)
        Precision: Ability of the classifier to not label poisitive sample that is negative.
            .1 means 10% of positive predictions were incorrect
        Recall: Ability of the classifier to find all positive samples.     
        F1 Score: Harmonoic mean of the precision and recall, 1 is best, 0 is worst.
    """
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 score: {f1}')
    
    
def svm(image_array, conf_coord, downsample=True):
    label_array = labeled_array(image_array, conf_coord)
    X = image_array
    y = label_array
        
    # convert to 1D arrays
    X_flat = image_array.reshape(-1, 1) 
    y_flat = label_array.reshape(-1, 1)
    # view_neighborhood(conf_coord, label_array)
    
    print(f'Number of non-zero elements in X_flat: {np.count_nonzero(X_flat)}') # return 7400397 for X_flat
    print(f'Number of non-zero elements in y_flat: {np.count_nonzero(y_flat)}') # return 93 for y_flat
    
    classes = np.unique(y_flat)
    print(f'Types of classes: {classes}') # return [0 1]
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

    # store for later
    X_test_2 = X_test
    y_test_2 = y_test 
    # downsample the majority class
    if downsample: 
        X_train, y_train = downsample_data(X_train, y_train)
        
    # hyperparameter tuning
    grid_search = svm_hyperparameter_tuning(X_train, y_train)
    
    # predict
    y_pred = grid_search.predict(X_test)
    
    # cross validation
    scores, mean_scores = svm_cross_validation(grid_search.best_estimator_, X_test, y_test, cv=5)
    
    # accuracy
    accuracy = grid_search.score(X_test, y_test)
    print(f'Accuracy with the best estimator: {accuracy}') # return 0.9999988974428944
    
    # confusion matrix
    cm = confusion_matrix(y_test.ravel(), y_pred)
    print(f'Confusion matrix:\n {cm}') 

    # accuracy metrics
    accuracy_metrics(y_test, y_pred)
    
    return grid_search.best_estimator_, y_pred, accuracy

def main_():
    image_choice = False
    conf_coord, image_array = pre_process(image_choice)
    conf_coord = list(conf_coord)
    best_model, y_pred, accuracy = svm(image_array, conf_coord, downsample=True)
    
if __name__ == '__main__':
    main_()
    