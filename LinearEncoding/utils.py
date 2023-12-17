import os
import importlib
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchlens as tl

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_pretrained_model(model_name, weights_available=False, weights_path=None):
    # Dynamically import the module based on the model_name
    model_module = importlib.import_module(f'torchvision.models')
    # Access the specific model class
    model_class = getattr(model_module, model_name)
    # Instantiate the model
    if weights_available:
        model = model_class(weights=None)
        weight = torch.load(weights_path)
        model.load_state_dict(weight)
    else:
        model = model_class(pretrained=True)
    return model    

def Load_images(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            image_paths.append(os.path.join(root, file))
    return np.sort(np.array(image_paths))

def generate_dataloader(image_paths, batch_size=10,preprocess=None):
    dataset = ImageDataset(image_paths,preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    return dataloader

def train_test_split(image_paths, train_idx, test_idx, batch_size=10,preprocess=None):
    train_paths, test_paths = image_paths[train_idx], image_paths[test_idx]
    Train_dataset = ImageDataset(train_paths,preprocess)
    Test_dataset = ImageDataset(test_paths,preprocess)
    Train_dataloader = DataLoader(Train_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    Test_dataloader = DataLoader(Test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    return Train_dataloader, Test_dataloader

def Extract_features(model, device, dataloader, feature_name):
    features = []
    for batch in dataloader:
        batch = batch.to(device)
        model_his = tl.log_forward_pass(model, batch, vis_opt="none",layers_to_save=[feature_name])
        feature = model_his[feature_name].tensor_contents
        features.append(feature.cpu().detach().numpy())
    features = np.concatenate(features, axis=0)
    features = features.reshape(features.shape[0],-1)
    return features

def Explained_variance(y_true, y_pred):
    total_var = np.var(y_true,axis=0)
    residual_var = np.sum((y_pred - y_true)**2,axis=0)/y_true.shape[0]
    print(total_var[0:10],residual_var[0:10])
    return 1 - np.mean(residual_var/total_var)