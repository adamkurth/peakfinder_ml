import os
import glob
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
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

class CCN(nn.Module):
    # CNN using pytorch
    def __init__(self, num_channels, img_height, img_width):
        super(CCN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1) # 32 neurons
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 neurons
        self.dropout = nn.Dropout(0.5)
        self.flattened_size = 64 * (img_height // 4) * (img_width // 4) # 64 neurons
        self.fc1 = nn.Linear(self.flattened_size, 128) # 128 neurons
        self.fc2 = nn.Linear(128, 2) # 2 for binary classification
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.MaxPool2d(2)(x) # 2x2 pooling
        x = torch.relu(self.conv2(x)) # 64 neurons
        x = nn.MaxPool2d(2)(x) # 2x2 pooling
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(x) # regularization
        x = torch.relu(self.fc1(x)) # 128 neurons
        x = self.fc2(x) # 2 for binary classification
        return x
    
def preprocess(image_data):
    if len(image_data.shape) == 2:  # If grayscale and 2D
        image_data = image_data.reshape(1, 1, image_data.shape[0], image_data.shape[1])
    elif len(image_data.shape) == 3:  # If 3D but no channel dimension
        image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], image_data.shape[2])
    # Move channel to the second dimension for PyTorch
    image_data = np.moveaxis(image_data, -1, 1)
    return image_data

def data_preparation(image_data, labeled_image):
    labeled_image = labeled_image.reshape(-1).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(image_data, labeled_image, test_size=0.2)
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return X_train, X_test, y_train, y_test, train_loader, test_loader, labeled_image
    
def train(train_loader, num_channels, img_height, img_width):
    model = CCN(num_channels, img_height, img_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}')
    return model

def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on test images: {100 * correct / total}%')


def test_main():
    threshold = 1000
    image_data, file_path = load_data(work=False)
    coordinates = main(file_path, threshold, display=False)
    coordinates = [tuple(coord) for coord in coordinates]
    
    labeled_image = generate_labeled_image(image_data, coordinates, neighborhood_size=5)

    # preprocessing
    image_data = preprocess(image_data)
    image_data = preprocess(image_data)
    print("Preprocessed image data shape:", image_data.shape)
    labeled_image = generate_labeled_image(image_data, coordinates)
    
    
    # data prep
    X_train, X_test, y_train, y_test, train_loader, test_loader, labeled_image = data_preparation(image_data, labeled_image)
    
    img_height, img_width = image_data.shape[1:3]
    img_height = image_data.shape[0]
    img_width = image_data.shape[1]
    num_channels = image_data.shape[2]
    
    model = train(train_loader, num_channels, img_height, img_width)
    
    evaluate_model(model, test_loader)
    
if __name__ == '__main__':
    test_main()