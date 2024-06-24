import os
import sys
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, 
                               QVBoxLayout, QTableWidget, QTableWidgetItem, QMenuBar,
                               QDialog, QDialogButtonBox, QComboBox, QLineEdit)
from PySide6.QtGui import QIcon
from utils import *

# Dict of parameters available
model_dict = {
    'alexnet': ['conv2', 'conv4', 'fc6'],
    'densenet169': ['denseblock2', 'denseblock3', 'denseblock4', 'classifier'],
    'inception_v3': ['avgpool', 'lastconv', 'midconv'],
    'resnet101': ['avgpool', 'block4', 'block3', 'block2'],
    'vgg19': ['classifier', 'lastconv', 'midconv'],
    # 'vit_l_16': ['lastlayer', 'midlayer']
}

reg_dict = {'Ridge':[1e-3,1e-2,1e2,1e3,1e4,1e5],
            # 'Lasso':[1e-3,1e-2,1e2,1e3,1e4,1e5],
            'PLS':[2,5,10,25,50,100]}

# Preprocess settings
default_preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),             
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set device and random seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))

## Functions for encoding analysis

# Baic encoding analysis
def Encoding(image_file, resp_file, para_dict):
    preprocess = para_dict['preprocess']
    model = para_dict['model']
    model_name = para_dict['model_name']
    feature_layer = para_dict['feature_layer']
    device = para_dict['device']
    pca_used = para_dict['pca_used']
    reg = para_dict['reg']

    Register_hook(model, model_name, feature_layer, fn)
    image_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)
    Extract_features(model, device, image_dataloader)
    feat = np.concatenate(result, axis=0)
    feat = feat.reshape(feat.shape[0], -1)
    result.clear()

    _, mean_val_EV, mean_val_R = CrossValidationEncoding(CV=5, feature=feat, resp=Resp, pca_used=pca_used, reg=reg)

    print("Mean R^2(Explained variance) for val_Resp: ", mean_val_EV)
    print("Mean Pearson correlation for val_Resp: ", mean_val_R)

# Encoding and select images with maximum and minimum response
def Select(image_file, resp_file, para_dict):
    preprocess = para_dict['preprocess']
    model = para_dict['model']
    model_name = para_dict['model_name']
    feature_layer = para_dict['feature_layer']
    device = para_dict['device']
    pca_used = para_dict['pca_used']
    reg = para_dict['reg']
    unit_num = para_dict['unit_num']
    num_extreme = para_dict['num_extreme']

    Register_hook(model, model_name, feature_layer, fn)
    image_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)
    Extract_features(model, device, image_dataloader)
    feat = np.concatenate(result, axis=0)
    feat = feat.reshape(feat.shape[0], -1)
    result.clear()

    pred_resp, mean_val_EV, mean_val_R = CrossValidationEncoding(CV=5, feature=feat, resp=Resp, pca_used=pca_used, reg=reg)
    print("Mean R^2(Explained variance) for val_Resp: ", mean_val_EV)
    print("Mean Pearson correlation for val_Resp: ", mean_val_R)

    # Find the extreme response idx
    true_max_idx = np.argsort(Resp[:,unit_num])[-num_extreme:][::-1]
    true_min_idx = np.argsort(Resp[:,unit_num])[:num_extreme]
    pred_max_idx = np.argsort(pred_resp[:,unit_num])[-num_extreme:][::-1]
    pred_min_idx = np.argsort(pred_resp[:,unit_num])[:num_extreme]
    images = Load_data_from_file(image_file)

    # Plot the images
    fig, ax = plt.subplots(4, num_extreme, figsize=(40, 40))
    for i in range(num_extreme):
        true_max_img = images[true_max_idx[i]]
        true_min_img = images[true_min_idx[i]]
        pred_max_img = images[pred_max_idx[i]]
        pred_min_img = images[pred_min_idx[i]]

        ax[0, i].imshow(true_max_img)
        ax[0, i].set_title('True Max {}'.format(i+1), fontdict={'fontsize': 10})
        ax[0, i].axis('off')
        ax[2, i].imshow(true_min_img)
        ax[2, i].set_title('True Min {}'.format(i+1), fontdict={'fontsize': 10})
        ax[2, i].axis('off')
        ax[1, i].imshow(pred_max_img)
        ax[1, i].set_title('Pred Max {}'.format(i+1),fontdict={'fontsize':10})
        ax[1, i].axis('off')
        ax[3, i].imshow(pred_min_img)
        ax[3, i].set_title('Pred Min {}'.format(i+1),fontdict={'fontsize':10})
        ax[3, i].axis('off')
    plt.show()

# Search for the best model, layer, regression type and parameter combination for encoding analysis
def Search(image_file, resp_file, split_ratio):
    preprocess = default_preprocess
    images, Resp = Load_data_from_file(image_file), Load_data_from_file(resp_file)
    
    permuted_idx = np.random.permutation(Resp.shape[0])
    test_idx, train_idx = permuted_idx[:int(split_ratio*Resp.shape[0])], permuted_idx[int(split_ratio*Resp.shape[0]):]
    train_images, te_images = images[train_idx], images[test_idx]
    train_Resp, te_Resp = Resp[train_idx], Resp[test_idx]
    
    tr_set, te_set = ImageDataset(train_images, preprocess), ImageDataset(te_images, preprocess)
    tr_dataloader = DataLoader(tr_set, batch_size=10, shuffle=False, pin_memory=True)
    te_dataloader = DataLoader(te_set, batch_size=10, shuffle=False, pin_memory=True)
    
    # First step: Find best aligned model and layer
    Best_model, Best_layer = 'alexnet', 'fc6'
    Best_EV, Best_r = 0,0

    for model_name in model_dict.keys():
        for feature_layer in model_dict[model_name]:
            # Load pretrained model
            print("Model: ", model_name)
            print("Layer: ", feature_layer)
            model = load_pretrained_model(model_name)
            model.eval().to(device)
    
            Register_hook(model, model_name, feature_layer, fn)
            reg = Register_regression('Ridge', 1000)

            Extract_features(model, device, tr_dataloader)
            tr_feat = np.concatenate(result, axis=0)
            tr_feat = tr_feat.reshape(tr_feat.shape[0], -1)
            result.clear()

            _,mean_val_EV, mean_val_R = CrossValidationEncoding(CV=5, feature=tr_feat, resp=train_Resp, reg=reg)
            print("R^2(Explained variance) for val_Resp: ", mean_val_EV)
            print("Pearson correlation for val_Resp: ", mean_val_R)

            if mean_val_R > Best_r:
                Best_EV = mean_val_EV
                Best_r = mean_val_R
                Best_model = model_name
                Best_layer = feature_layer

    # Second step: Find best regression type and parameter
    model = load_pretrained_model(Best_model)
    model.eval().to(device)
    Register_hook(model, Best_model, Best_layer, fn)

    Best_reg, Best_para = 'Ridge', 1000

    for reg_type in reg_dict.keys():
        for reg_para in reg_dict[reg_type]:

            print("Regression type: ", reg_type)
            print("Regression parameter: ", reg_para)
            reg = Register_regression(reg_type, reg_para)

            Extract_features(model, device, tr_dataloader)
            tr_feat = np.concatenate(result, axis=0)
            tr_feat = tr_feat.reshape(tr_feat.shape[0], -1)
            result.clear()
            
            _,mean_val_EV, mean_val_R = CrossValidationEncoding(CV=5, feature=tr_feat, resp=train_Resp, reg=reg)
            print("R^2(Explained variance) for val_Resp: ", mean_val_EV)
            print("Pearson correlation for val_Resp: ", mean_val_R)

            if mean_val_R > Best_r:
                Best_EV = mean_val_EV
                Best_r = mean_val_R
                Best_reg = reg_type
                Best_para = reg_para
                
    print("Best Pearson correlation:", Best_r)
    print("Best R^2(Explained variance):", Best_EV)

    # Third step, train the best model and layer with the best regression type and parameter and test on the test set
    model = load_pretrained_model(Best_model)
    model.eval().to(device)
    Register_hook(model, Best_model, Best_layer, fn)
    reg = Register_regression(Best_reg, Best_para)

    # Extract features
    Extract_features(model, device, tr_dataloader)
    tr_feat = np.concatenate(result, axis=0)
    tr_feat = tr_feat.reshape(tr_feat.shape[0], -1)
    result.clear()
    Extract_features(model,  device, te_dataloader)
    te_feat = np.concatenate(result, axis=0)
    te_feat = te_feat.reshape(te_feat.shape[0], -1)
    result.clear()

    # PCA
    # if pca_used:
    if True:
        pca = PCA(n_components=100)
        pca.fit(tr_feat)
        tr_feat = pca.transform(tr_feat)
        te_feat = pca.transform(te_feat)

    # Train model
    scaler = StandardScaler()
    scaler.fit(tr_feat)
    tr_feat = scaler.transform(tr_feat)
    reg.fit(tr_feat, train_Resp)
    tr_pred = reg.predict(tr_feat)

    # Test model
    te_feat = scaler.transform(te_feat)
    te_pred = reg.predict(te_feat)
    Test_EV = Explained_variance(te_Resp, te_pred)
    Test_R = Pearson_correlation(te_Resp, te_pred)

    print("Best model: ", Best_model)
    print("Best layer: ", Best_layer)
    print("Best regression type: ", Best_reg)
    print("Best regression parameter: ", Best_para)
    print("Best Test Pearson correlation:", Test_R)
    print("Best Test R^2(Explained variance):", Test_EV)

# Training encoding model with current data and select images with maximum and minimum response in the new dataset
def Select_Dataset(image_file, resp_file, dataset_file, para_dict):
    preprocess = para_dict['preprocess']
    model = para_dict['model']
    model_name = para_dict['model_name']
    feature_layer = para_dict['model_layer']
    device = para_dict['device']
    pca_used = para_dict['pca_used']
    reg = para_dict['reg']
    unit_num = para_dict['unit_num']
    num_extreme = para_dict['num_extreme']

    Register_hook(model, model_name, feature_layer, fn)
    tr_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)
    te_dataloader, _ = generate_dataloader(dataset_file, resp_file, batch_size=10, preprocess=preprocess)
    Extract_features(model, device, tr_dataloader)
    tr_feat = np.concatenate(result, axis=0)
    tr_feat = tr_feat.reshape(tr_feat.shape[0], -1)
    result.clear()
    Extract_features(model, device, te_dataloader)
    te_feat = np.concatenate(result, axis=0)
    te_feat = te_feat.reshape(te_feat.shape[0], -1)
    result.clear()

    if pca_used:
        pca = PCA(n_components=100)
        pca.fit(tr_feat)
        tr_feat = pca.transform(tr_feat)
        te_feat = pca.transform(te_feat)

    reg.fit(tr_feat, Resp)
    pred_response = reg.predict(te_feat)

    pred_response = pred_response[:,unit_num].reshape(-1)
    images = Load_data_from_file(dataset_file)
    pred_max_idx = np.argsort(pred_response)[-num_extreme:][::-1]
    pred_min_idx = np.argsort(pred_response)[:num_extreme]

    # Plot the images
    fig, ax = plt.subplots(2, num_extreme, figsize=(40, 40))
    for i in range(num_extreme):
        pred_max_img = images[pred_max_idx[i]]
        pred_min_img = images[pred_min_idx[i]]
        ax[0, i].imshow(pred_max_img)
        ax[0, i].set_title('Pred Max {}'.format(i+1))
        ax[0, i].axis('off')
        ax[1, i].imshow(pred_min_img)
        ax[1, i].set_title('Pred Min {}'.format(i+1))
        ax[1, i].axis('off')
    plt.show()

# Use RISE algorithm to visualize the saliency map in a specific image for a specific unit
def Visualization(image_file, resp_file, para_dict):
    preprocess = para_dict['preprocess']
    model = para_dict['model']
    model_name = para_dict['model_name']
    feature_layer = para_dict['feature_layer']
    device = para_dict['device']
    pca_used = para_dict['pca_used']
    reg = para_dict['reg']
    unit_num = para_dict['unit_num']
    img_idx = para_dict['img_idx']

    Register_hook(model, model_name, feature_layer, fn)
    image_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)
    Extract_features(model, device, image_dataloader)
    feat = np.concatenate(result, axis=0)
    feat = feat.reshape(feat.shape[0], -1)
    result.clear()

    images = Load_data_from_file(image_file)

    if pca_used:
        pca = PCA(n_components=100)
        pca.fit(feat)
        feat = pca.transform(feat)
    
    tr_feat = np.delete(feat, img_idx, axis=0)
    tr_Resp = np.delete(Resp, img_idx, axis=0)
    # Train model
    scaler = StandardScaler()
    scaler.fit(tr_feat)
    tr_feat = scaler.transform(tr_feat)
    reg.fit(tr_feat, tr_Resp)

    # Choose the unit you want to visualize
    vis_img = images[img_idx]

    # Generate and add masks to the image
    im_mask = np.random.rand(6,6,200) > 0.5
    im_ = im_mask.astype(float)
    zoom_factor = (224/6, 224/6, 1)
    im_ = zoom(im_, zoom_factor, order=3)
    im_new = np.zeros((200,224,224,3))
    for j in range(im_.shape[2]):
        imm = im_[:,:,j]
        for k in range(3):
            im_new[j,:,:,k] = np.clip(imm * vis_img[:,:,k] + (1-imm) * 127, 0, 255)

    # Extract features of masked images from fc7 layer
    im_new = im_new.astype(np.uint8)
    rise_dataset = ImageDataset(im_new, transform=preprocess)
    rise_dataloader = DataLoader(rise_dataset, batch_size=10, shuffle=False,pin_memory=True)

    Extract_features(model, device, rise_dataloader)
    feat_masked = np.vstack(result)
    result.clear()

    # Using encoding model to predict response for masked images, then compute the weight map
    feat_masked = pca.transform(feat_masked)
    pred_resp_rise = reg.predict(feat_masked)[:, unit_num]
    print(pred_resp_rise.shape)

    wmap = np.zeros((224,224))
    for j in range(im_.shape[2]):
        wmap += im_[:,:,j] * pred_resp_rise[j]
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Display the original image and weight map
    ax[0].imshow(vis_img.astype(np.uint8))
    ax[1].imshow(wmap, cmap='turbo')

    # Normalize the weight map and apply the colormap
    maxc = np.max(wmap)
    minc = np.min(wmap)
    weight_color = ((wmap - minc) / (maxc - minc) * 255 + 1).astype(int)
    cmap = plt.get_cmap('turbo', 256)
    ww = cmap(weight_color)
    ww = ww[:, :, :3]

    # Create the final map
    ww = ww * 255
    finalmap = ww * 0.7 + vis_img * 0.3
    # Display the final map
    ax[2].imshow(finalmap.astype(np.uint8))
    plt.show()


class ParameterDialog(QDialog):
    def __init__(self, parent=None, image_paths=None, response_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters Choosing")
        
        self._layout = QVBoxLayout()
        if os.path.exists('SoyoCry.jpg'):
            self.setWindowIcon(QIcon('SoyoCry.jpg'))
        self.image_path_label = QLabel("Image File:")
        self.image_path_combo = QComboBox()
        self.image_path_combo.addItems(image_paths)

        self.response_path_label = QLabel("Response File:")
        self.response_path_combo = QComboBox()
        self.response_path_combo.addItems(response_paths)

        self.model_name_label = QLabel("Model Name:")
        self.model_name_combo = QComboBox()
        self.model_name_combo.addItems(model_dict.keys())
        self.model_name_combo.currentIndexChanged.connect(self.update_layers)

        self.model_layer_label = QLabel("Model Layer:")
        self.model_layer_combo = QComboBox()

        self.regression_type_label = QLabel("Regression Type:")
        self.regression_type_combo = QComboBox()
        self.regression_type_combo.addItems(["Ridge", "Lasso", "PLS"])  # Example regression types

        self.regression_param_label = QLabel("Regression Parameter:")
        self.regression_param_edit = QLineEdit()

        self._layout.addWidget(self.image_path_label)
        self._layout.addWidget(self.image_path_combo)
        self._layout.addWidget(self.response_path_label)
        self._layout.addWidget(self.response_path_combo)
        self._layout.addWidget(self.model_name_label)
        self._layout.addWidget(self.model_name_combo)
        self._layout.addWidget(self.model_layer_label)
        self._layout.addWidget(self.model_layer_combo)
        self._layout.addWidget(self.regression_type_label)
        self._layout.addWidget(self.regression_type_combo)
        self._layout.addWidget(self.regression_param_label)
        self._layout.addWidget(self.regression_param_edit)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self._layout.addWidget(self.buttonBox)
    
        self.setLayout(self._layout)
        self.update_layers()

    def update_layers(self):
        model_name = self.model_name_combo.currentText()
        layers = model_dict.get(model_name, [])
        self.model_layer_combo.clear()
        self.model_layer_combo.addItems(layers)

    def get_parameters(self):
         return {
            "image_file": self.image_path_combo.currentText(),
            "response_file": self.response_path_combo.currentText(),
            "model_name": self.model_name_combo.currentText(),
            "model_layer": self.model_layer_combo.currentText(),
            "regression_type": self.regression_type_combo.currentText(),
            "regression_param": self.regression_param_edit.text()
        }

class ParameterDialog_select(ParameterDialog):
    def __init__(self, parent=None,image_paths=None, response_paths=None):
        super().__init__(parent,image_paths, response_paths)
        self.unit_idx_label = QLabel("Neural unit index:")
        self.unit_idx_edit = QLineEdit()
        self.num_extreme_label = QLabel("Number of extreme images:")
        self.num_extreme_edit = QLineEdit()

        self.new_buttonBox = self.buttonBox
        self._layout.removeWidget(self.buttonBox)
        self._layout.addWidget(self.unit_idx_label)
        self._layout.addWidget(self.unit_idx_edit)
        self._layout.addWidget(self.num_extreme_label)
        self._layout.addWidget(self.num_extreme_edit)
        self._layout.addWidget(self.new_buttonBox)
        
        self.setLayout(self._layout) 

    def get_parameters(self):
        return {
            "image_file": self.image_path_combo.currentText(),
            "response_file": self.response_path_combo.currentText(),
            "model_name": self.model_name_combo.currentText(),
            "model_layer": self.model_layer_combo.currentText(),
            "regression_type": self.regression_type_combo.currentText(),
            "regression_param": self.regression_param_edit.text(),
            "unit_idx": self.unit_idx_edit.text(),
            "num_extreme": self.num_extreme_edit.text()
        }
        
class ParameterDialog_Search(QDialog):
    def __init__(self, parent=None, image_paths=None, response_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Parameters Choosing")
        
        self._layout = QVBoxLayout()

        self.image_path_label = QLabel("Image File:")
        self.image_path_combo = QComboBox()
        self.image_path_combo.addItems(image_paths)

        self.response_path_label = QLabel("Response File:")
        self.response_path_combo = QComboBox()
        self.response_path_combo.addItems(response_paths)

        self.split_ratio_label = QLabel("Split Ratio:")
        self.split_ratio_edit = QLineEdit()

        self._layout.addWidget(self.image_path_label)
        self._layout.addWidget(self.image_path_combo)
        self._layout.addWidget(self.response_path_label)
        self._layout.addWidget(self.response_path_combo)
        self._layout.addWidget(self.split_ratio_label)
        self._layout.addWidget(self.split_ratio_edit)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self._layout.addWidget(self.buttonBox)
        self.setLayout(self._layout)

    def get_parameters(self):
         return {
            "image_file": self.image_path_combo.currentText(),
            "response_file": self.response_path_combo.currentText(),
            "split_ratio": self.split_ratio_edit.text()
        }

class ParameterDialog_dataset(ParameterDialog):
    def __init__(self, parent=None,image_paths=None, response_paths=None):

        super().__init__(parent,image_paths, response_paths)

        self.new_buttonBox = self.buttonBox
        self._layout.removeWidget(self.buttonBox)

        self.dataset_file_label = QLabel("Dataset File:")
        self.dataset_file_combo = QComboBox()
        self.button_1 = QPushButton('Add Dataset File', self)
        self.button_1.clicked.connect(self.addFile)
        self.unit_idx_label = QLabel("Neural unit index:")
        self.unit_idx_edit = QLineEdit()
        self.num_extreme_label = QLabel("Number of extreme images:")
        self.num_extreme_edit = QLineEdit()

        self._layout.addWidget(self.dataset_file_label)
        self._layout.addWidget(self.dataset_file_combo)
        self._layout.addWidget(self.button_1)
        self._layout.addWidget(self.unit_idx_label)
        self._layout.addWidget(self.unit_idx_edit)
        self._layout.addWidget(self.num_extreme_label)
        self._layout.addWidget(self.num_extreme_edit)
        self._layout.addWidget(self.new_buttonBox)

        self.setLayout(self._layout)
    
    def get_parameters(self):
         return {
            "image_file": self.image_path_combo.currentText(),
            "response_file": self.response_path_combo.currentText(),
            "dataset_file": self.dataset_file_combo.currentText(),
            "model_name": self.model_name_combo.currentText(),
            "model_layer": self.model_layer_combo.currentText(),
            "regression_type": self.regression_type_combo.currentText(),
            "regression_param": self.regression_param_edit.text(),
            "unit_idx": self.unit_idx_edit.text(),
            "num_extreme": self.num_extreme_edit.text()
        }
    
    def addFile(self):
        # Open a file dialog and get the selected file path
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)', options=options)
        if filePath:
            # Display the selected file path in the label
            self.dataset_file_combo.addItem(filePath)

class ParameterDialog_visualization(ParameterDialog):
    def __init__(self, parent=None,image_paths=None, response_paths=None):
        super().__init__(parent, image_paths, response_paths)
        self.unit_idx_label = QLabel("Neural unit index:")
        self.unit_idx_edit = QLineEdit()
        self.img_idx_label = QLabel("Image index:")
        self.img_idx_edit = QLineEdit()

        self.new_buttonBox = self.buttonBox
        self._layout.removeWidget(self.buttonBox)
        self._layout.addWidget(self.unit_idx_label)
        self._layout.addWidget(self.unit_idx_edit)
        self._layout.addWidget(self.img_idx_label)
        self._layout.addWidget(self.img_idx_edit)
        self._layout.addWidget(self.new_buttonBox)
        
        self.setLayout(self._layout) 

    def get_parameters(self):
        return {
            "image_file": self.image_path_combo.currentText(),
            "response_file": self.response_path_combo.currentText(),
            "model_name": self.model_name_combo.currentText(),
            "model_layer": self.model_layer_combo.currentText(),
            "regression_type": self.regression_type_combo.currentText(),
            "regression_param": self.regression_param_edit.text(),
            "unit_idx": self.unit_idx_edit.text(),
            "img_idx": self.img_idx_edit.text()
        }  

class Main_interface(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Interface")
        self.setGeometry(100, 100, 800, 600)
        self.image_path = []
        self.response_path = []
        self.init_UI()
    
    def init_UI(self):
        if os.path.exists('SoyoStare.jpg'):
            self.setWindowIcon(QIcon('SoyoStare.jpg'))
        self.table_1 = QTableWidget(self)
        self.table_1.setGeometry(175, 160, 450, 192)
        self.table_1.setColumnCount(2)
        self.table_1.setHorizontalHeaderLabels(['Image File', 'Response File'])
        self.table_1.setColumnWidth(0, 200)
        self.table_1.setColumnWidth(1, 200)
        self.table_1.setRowCount(5)

        self.button_1 = QPushButton('Add Image File', self)
        self.button_1.clicked.connect(self.addImageFile)
        self.button_1.setGeometry(210, 450, 141, 41)

        self.button_2 = QPushButton('Add Response File', self)
        self.button_2.clicked.connect(self.addResponseFile)
        self.button_2.setGeometry(450, 450, 141, 41)

        self.MenuBar = QMenuBar(self)
        self.Ana_menu = self.MenuBar.addMenu("Analysis")

        self.Encoding_action = self.Ana_menu.addAction("Encoding")
        self.Encoding_action.triggered.connect(self.showParameterDialog_Encoding)
        self.Select_action = self.Ana_menu.addAction("Select")
        self.Select_action.triggered.connect(self.showParameterDialog_Select)
        self.Search_action = self.Ana_menu.addAction("Search")
        self.Search_action.triggered.connect(self.showParameterDialog_Search)
        self.Dataset_action = self.Ana_menu.addAction("Dataset")
        self.Dataset_action.triggered.connect(self.showParameterDialog_Dataset)
        self.Vis_action = self.Ana_menu.addAction("Visualization")
        self.Vis_action.triggered.connect(self.showParameterDialog_Visualization)
        self.setMenuBar(self.MenuBar)
        
    def findNextEmptyRow(self, table, col):
        for row in range(table.rowCount()):
            if table.item(row, col) is None:
                return row
        return None
    
    def addImageFile(self):
        # Open a file dialog and get the selected file path
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)', options=options)
        if filePath:
            # Display the selected file path in the label
            self.image_path.append(filePath)
            rel_filePath = filePath.split('/')[-1]
            row_pos = self.findNextEmptyRow(self.table_1, 0)
            if row_pos is not None:
                self.table_1.setItem(row_pos, 0, QTableWidgetItem(rel_filePath))

    def addResponseFile(self):
        # Open a file dialog and get the selected file path
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filePath, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)', options=options)
        if filePath:
            # Display the selected file path in the label
            self.response_path.append(filePath)
            rel_filePath = filePath.split('/')[-1]
            row_pos = self.findNextEmptyRow(self.table_1,1)
            if row_pos is not None:
                self.table_1.setItem(row_pos, 1, QTableWidgetItem(rel_filePath))

    def showParameterDialog_Encoding(self):
        dialog = ParameterDialog(self, image_paths=self.image_path, response_paths=self.response_path)
        if dialog.exec() == QDialog.Accepted:
            parameters = dialog.get_parameters()
            self.runEncoding(parameters)

    def runEncoding(self, parameters):     
        para_dict = {
            'preprocess': default_preprocess, 
            'model': load_pretrained_model(parameters["model_name"]),
            'model_name': parameters["model_name"],
            'feature_layer': parameters["model_layer"],
            'device' :  device,  
            'pca_used' : True, 
            'reg' : Register_regression(parameters["regression_type"], float(parameters["regression_param"]))
        }

        image_file = parameters["image_file"]
        resp_file = parameters["response_file"]
        Encoding(image_file, resp_file, para_dict)
    
    def showParameterDialog_Select(self):
        dialog = ParameterDialog_select(image_paths=self.image_path, response_paths=self.response_path)
        if dialog.exec() == QDialog.Accepted:
            parameters = dialog.get_parameters()
            self.runSelect(parameters)
    
    def runSelect(self, parameters):
        para_dict = {
            'preprocess': default_preprocess,  
            'model': load_pretrained_model(parameters["model_name"]),
            'model_name': parameters["model_name"],
            'feature_layer': parameters["model_layer"],
            'device' :  device,  
            'pca_used' : True,  
            'reg' : Register_regression(parameters["regression_type"], float(parameters["regression_param"])),
            'unit_num': int(parameters["unit_idx"]),
            'num_extreme': int(parameters["num_extreme"])
        }
        image_file = parameters["image_file"]
        resp_file = parameters["response_file"]
        Select(image_file, resp_file, para_dict)
    
    def showParameterDialog_Search(self):
        dialog = ParameterDialog_Search(image_paths=self.image_path, response_paths=self.response_path)
        if dialog.exec() == QDialog.Accepted:
            parameters = dialog.get_parameters()
            self.runSearch(parameters)
    
    def runSearch(self, parameters):
        image_file = parameters["image_file"]
        resp_file = parameters["response_file"]
        split_ratio = float(parameters["split_ratio"])
        Search(image_file, resp_file, split_ratio)
    
    def showParameterDialog_Dataset(self):
        dialog = ParameterDialog_dataset(image_paths=self.image_path, response_paths=self.response_path)
        if dialog.exec() == QDialog.Accepted:
            parameters = dialog.get_parameters()
            self.runDataset(parameters)
     
    def runDataset(self, parameters):
        para_dict = {
            'preprocess': default_preprocess, 
            'model': load_pretrained_model(parameters["model_name"]),
            'model_name': parameters["model_name"],
            'model_layer': parameters["model_layer"],
            'reg': Register_regression(parameters["regression_type"], float(parameters["regression_param"])),
            'device' :  device,  
            'pca_used' : True, 
            'unit_num': int(parameters["unit_idx"]),
            'num_extreme': int(parameters["num_extreme"])
        }

        image_file = parameters["image_file"]
        resp_file = parameters["response_file"]
        dataset_file = parameters["dataset_file"]
        Select_Dataset(image_file, resp_file, dataset_file, para_dict)
    
    def showParameterDialog_Visualization(self):
        dialog = ParameterDialog_visualization(image_paths=self.image_path, response_paths=self.response_path)
        if dialog.exec() == QDialog.Accepted:
            parameters = dialog.get_parameters()
            self.runVis(parameters)
   
    def runVis(self, parameters):
        para_dict = {
            'preprocess': default_preprocess,  
            'model': load_pretrained_model(parameters["model_name"]),
            'model_name': parameters["model_name"],
            'feature_layer': parameters["model_layer"],
            'device' :  device,  
            'pca_used' : True,  
            'reg' : Register_regression(parameters["regression_type"], float(parameters["regression_param"])),
            'unit_num': int(parameters["unit_idx"]),
            'img_idx': int(parameters["img_idx"])
        }

        image_file = parameters["image_file"]
        resp_file = parameters["response_file"]
        Visualization(image_file, resp_file, para_dict)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main_interface()
    window.show()
    sys.exit(app.exec())
