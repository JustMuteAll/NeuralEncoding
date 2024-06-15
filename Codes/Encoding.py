import time
import argparse
from tqdm import tqdm

import torch
import numpy as np
import scipy.io as sio
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import *

def main():
    start_time = time.time()
    
    # Read command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', default='alexnet', type=str)
    args.add_argument('--feature_layer', default='fc6', type=str)
    args.add_argument('--weights_available', default=1, type=int)
    args.add_argument('--reg_type', default='LR', type=str)
    args.add_argument('--reg_para', default=0.001, type=float)

    # Set parameters
    Parser = args.parse_args()
    model_name = str(Parser.model_name)
    feature_layer = str(Parser.feature_layer)
    weights_available = Parser.weights_available
    reg_type = str(Parser.reg_type)
    reg_para = float(Parser.reg_para)
    weights_dir = r"D:\Downloads" # Model weights directory
    weights_path = os.path.join(weights_dir, '{}.pth'.format(model_name))
    
    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load pretrained model
    model = load_pretrained_model(model_name, weights_available, weights_path)
    model.eval().to(device)

    # Register hook
    global result
    result = []
    def fn(module, inputs, outputs):
        result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))
    Register_hook(model, model_name, feature_layer, fn)

    # Preprocess settings
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),             
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Determine the type of regression
    if reg_type == 'LR':
        reg = LinearRegression()
    elif reg_type == 'Ridge':
        reg = Ridge(alpha=reg_para)
    elif reg_type == 'Lasso':
        reg = Lasso(alpha=reg_para)
    elif reg_type == 'PLS':
        reg = PLSRegression(n_components=int(reg_para))

    # Determine the type of data split, 1 for train-test split, 0 for leave-one-out cross-validation
    Data_split = 1
    pca_used = 0

    if Data_split: 
        # Set image and response path, generate dataloader
        tr_image_file = r'C:\Users\DELL\Desktop\Pics\Pics'
        te_image_file = r'C:\Users\DELL\Desktop\valPics\valPics'
        tr_resp_path = r'C:\Users\DELL\Desktop\Diffusion Model\Data\Rsp_train.npy'
        te_resp_path = r'C:\Users\DELL\Desktop\Diffusion Model\Data\Rsp_val.npy'
        tr_dataloader, tr_Resp = generate_dataloader(tr_image_file, tr_resp_path, batch_size=10,preprocess=preprocess)
        te_dataloader, te_Resp = generate_dataloader(te_image_file, te_resp_path, batch_size=10,preprocess=preprocess)

        # Extract features
        Extract_features(model, device, tr_dataloader)
        tr_feat = np.concatenate(result, axis=0)
        tr_feat = tr_feat.reshape(tr_feat.shape[0],-1)
        result = []
        Extract_features(model,  device, te_dataloader)
        te_feat = np.concatenate(result, axis=0)
        te_feat = te_feat.reshape(te_feat.shape[0],-1)

        # PCA
        if pca_used:
            pca = PCA(n_components=50)
            pca.fit(tr_feat)
            tr_feat = pca.transform(tr_feat)
            te_feat = pca.transform(te_feat)

        # Train model
        scaler = StandardScaler()
        scaler.fit(tr_feat)
        tr_feat = scaler.transform(tr_feat)
        reg.fit(tr_feat, tr_Resp)
        tr_pred = reg.predict(tr_feat)
        tr_Resp, tr_pred = tr_Resp.reshape(tr_feat.shape[0],-1), tr_pred.reshape(tr_feat.shape[0],-1)
        Train_EV = Explained_variance(tr_Resp, tr_pred)
        Train_R = Pearson_correlation(tr_Resp, tr_pred)
        print("R^2(Explained variance) for Train_Resp: ", Train_EV)
        print("Pearson correlation for Train_Resp: ", Train_R)

        # Test model
        te_feat = scaler.transform(te_feat)
        te_pred = reg.predict(te_feat)
        te_Resp, te_pred = te_Resp.reshape(te_feat.shape[0],-1), te_pred.reshape(te_feat.shape[0],-1)
        Test_EV = Explained_variance(te_Resp, te_pred)
        Test_R = Pearson_correlation(te_Resp, te_pred)
        print("R^2(Explained variance) for Test_Resp: ", Test_EV)
        print("Pearson correlation for Test_Resp: ", Test_R)

        np.save("../Data/{}/pred_response.npy".format("YourResultFolder"), te_pred)

    else:
        # Set image and response path, generate dataloader
        image_file = r"C:\Users\DELL\Desktop\im96.mat"
        resp_file = r"C:\Users\DELL\Desktop\food_96_data.mat"
        image_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)

        # Extract features
        Extract_features(model, device, image_dataloader)
        feat = np.concatenate(result, axis=0)
        feat = feat.reshape(feat.shape[0],-1)
        result = []
        mean_corr, mean_EV = 0, 0
        pred_response = []

        # Leave-one-out cross-validation
        for idx in tqdm(range(Resp.shape[0])):
            te_feat = feat[idx,:].reshape(1,-1)
            te_Resp = Resp[idx,:]
            tr_feat = np.delete(feat, idx, axis=0)
            tr_Resp = np.delete(Resp, idx, axis=0)

            # PCA
            if pca_used:
                pca = PCA(n_components=25)
                pca.fit(tr_feat)
                tr_feat = pca.transform(tr_feat)
                te_feat = pca.transform(te_feat)

            # Train model
            scaler = StandardScaler()
            scaler.fit(tr_feat)
            tr_feat = scaler.transform(tr_feat)
            reg.fit(tr_feat, tr_Resp)
            tr_pred = reg.predict(tr_feat)
            
            # Test model
            te_feat = scaler.transform(te_feat)
            te_pred = reg.predict(te_feat).reshape(-1)
            pred_response.append(te_pred)
            te_Resp, te_pred = te_Resp.reshape(te_feat.shape[0],-1), te_pred.reshape(te_feat.shape[0],-1)
            Test_EV = Explained_variance(te_Resp, te_pred)
            Test_R = Pearson_correlation(te_Resp, te_pred)
            mean_corr += Test_R
            mean_EV += Test_EV

        print("Mean Pearson correlation for Test_Resp: ", mean_corr/Resp.shape[0])
        print("Mean R^2(Explained variance) for Test_Resp: ", mean_EV/Resp.shape[0])
        np.save("../Data/{}/pred_response.npy".format("YourResultFolder"), pred_response)

    end_time = time.time()
    print("Total Time:", end_time - start_time)

if __name__ == '__main__':
    main()
