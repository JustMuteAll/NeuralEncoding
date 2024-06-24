import os
import importlib
import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image_arr = self.images[index]
        if image_arr.shape[0] == 3:
            image = Image.fromarray(image_arr.transpose(2,1,0).astype('uint8'))
        else:
            image = Image.fromarray(image_arr.astype('uint8'))
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
    return sorted(np.array(image_paths), key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))


def Load_data_from_file(data_file):
    if data_file.endswith('.npy'):
        data = np.load(data_file)
    elif data_file.endswith('.mat'):
        try:
            data = sio.loadmat(data_file)
            key_name = list(data.keys())[3]
            data = data[key_name]
        except NotImplementedError:
            data = h5py.File(data_file)
            key_name = list(data.keys())[0]
            data = data[key_name][:]
    
    if len(data.shape) == 4 and data.shape[2] == 3:
        data = data.transpose(3,0,1,2)
    return data

def generate_dataloader(image_file, response_file, batch_size=10, preprocess=None):
    # Load images from file
    images = Load_data_from_file(image_file)
    resp = Load_data_from_file(response_file)
    image_dataset = ImageDataset(images,preprocess)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

    return image_dataloader, resp

def Register_hook(model, model_name, layer_name,fn):

    if model_name == 'alexnet':
        if layer_name == 'conv2':
            model.features[4].register_forward_hook(fn)
        elif layer_name == 'conv4':
            model.features[9].register_forward_hook(fn)
        elif layer_name == 'fc6':
            model.classifier[2].register_forward_hook(fn)

    elif model_name == 'densenet169':
        if layer_name == 'denseblock2':
            model.features.transition2.pool.register_forward_hook(fn)
        elif layer_name == 'denseblock3':
            model.features.transition3.pool.register_forward_hook(fn)
        elif layer_name == 'denseblock4':
            model.features.denseblock4.register_forward_hook(fn)
        elif layer_name == 'classifier':
            model.classifier.register_forward_hook(fn)
    
    elif model_name == 'inception_v3': 
        if layer_name == 'avgpool':
            model.avgpool.register_forward_hook(fn) 
        elif layer_name == 'lastconv':
            model.Mixed_7c.register_forward_hook(fn)
        elif layer_name == 'midconv':
            model.Mixed_6e.register_forward_hook(fn)

    elif model_name == 'resnet101':
        if layer_name == 'avgpool':
            model.avgpool.register_forward_hook(fn)
        elif layer_name == 'block4':
            model.layer4.register_forward_hook(fn)
        elif layer_name == 'block3':
            model.layer3.register_forward_hook(fn)
        elif layer_name == 'block2':
            model.layer2.register_forward_hook(fn)

    elif model_name == 'vgg19':
        if layer_name == 'classifier':
            model.classifier[1].register_forward_hook(fn)
        elif layer_name == 'lastconv':
            model.features[35].register_forward_hook(fn)
        elif layer_name == 'midconv':
            model.features[20].register_forward_hook(fn)

    elif model_name == 'vit_l_16':
        if layer_name == 'lastlayer':
            model.encoder.ln.register_forward_hook(fn)
        elif layer_name == 'midlayer':
            model.encoder.layers.encoder_layer_13.ln_1.register_forward_hook(fn)

def Register_regression(reg_type, reg_para):
    if reg_type == 'Ridge':
        reg = Ridge(alpha=reg_para)
    elif reg_type == 'Lasso':
        reg = Lasso(alpha=reg_para)
    elif reg_type == 'PLS':
        reg = PLSRegression(n_components=int(reg_para))
    return reg

def Extract_features(model,device, dataloader):
    model.eval()
    model.to(device)
    for batch in dataloader:
        batch = batch.to(device)
        _ = model(batch)

def Explained_variance(y_true, y_pred):
    total_var = np.var(y_true,axis=0)
    residual_var = np.sum((y_pred - y_true)**2,axis=0)/y_true.shape[0]
    ev = (total_var-residual_var)/total_var
    return np.mean(ev)

def Pearson_correlation(y_true, y_pred):
    mean_corr = 0
    for i in range(y_true.shape[1]):
        corr = np.corrcoef(y_true[:,i],y_pred[:,i])[0,1]
        mean_corr += corr
    mean_corr /= y_true.shape[1]
    return mean_corr

def convert_imgs_into_file(image_folder):
    image_files = os.listdir(image_folder)
    image_files.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
    image_files = [os.path.join(image_folder, f) for f in image_files]

    result = []
    for i, img_path in enumerate(tqdm(image_files)):
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = np.array(img)
        result.append(img)

    result = np.array(result)
    print(result.shape)
    return result

def CrossValidationEncoding(CV, feature, resp, pca_used=True, reg=None):

    n_samples = resp.shape[0]
    permuted_idx = np.random.permutation(n_samples)
    mean_val_EV, mean_val_R = 0, 0
    pred_resp = np.zeros_like(resp)
    for iter in range(CV):
        # Split the data into training and validation set
        val_idx = permuted_idx[int(iter*n_samples/CV):int((iter+1)*n_samples/CV)]
        tr_idx = np.setdiff1d(permuted_idx, val_idx)
        tr_feat, val_feat = feature[tr_idx], feature[val_idx]
        tr_Resp, val_Resp = resp[tr_idx], resp[val_idx]

        # PCA
        if pca_used:
            pca = PCA(n_components=100)
            pca.fit(tr_feat)
            tr_feat = pca.transform(tr_feat)
            val_feat = pca.transform(val_feat)

        # Train model
        scaler = StandardScaler()
        scaler.fit(tr_feat)
        tr_feat = scaler.transform(tr_feat)
        reg.fit(tr_feat, tr_Resp)
        tr_pred = reg.predict(tr_feat)

        # Test model
        val_feat = scaler.transform(val_feat)
        val_pred = reg.predict(val_feat)
        pred_resp[val_idx] = val_pred
        val_EV = Explained_variance(val_Resp, val_pred)
        mean_val_EV += val_EV
        val_R = Pearson_correlation(val_Resp, val_pred)
        mean_val_R += val_R

    mean_val_EV /= CV
    mean_val_R /= CV

    return pred_resp, mean_val_EV, mean_val_R
