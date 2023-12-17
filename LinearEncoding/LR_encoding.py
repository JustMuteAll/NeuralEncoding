'''
Predict neural responses using linear regression
'''
import torch
import argparse
import numpy as np
import scipy.io as sio
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from utils import *

def main():
    # Read command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', default='alexnet', type=str)
    args.add_argument('--feature_layer', default='fc6', type=str)
    args.add_argument('--weights_available', default=False, type=bool)
    args.add_argument('--reg_type', default='LR', type=str)
    args.add_argument('--reg_para', default=0.001, type=float)

    # Set parameters
    Parser = args.parse_args()
    model_name = Parser.model_name
    feature_layer = Parser.feature_layer
    weights_available = Parser.weights_available
    reg_type = Parser.reg_type
    reg_para = Parser.reg_para
    split_scale = 0.2
    
    # Set data path
    Image_path = r'C:\Users\DELL\Desktop\Predict\shared1000'
    Resp_path = r'C:\Users\DELL\Desktop\Predict\NSD_selected_response.npy'
    weights_dir = r"D:\Downloads"
    weights_path = os.path.join(weights_dir, '{}.pth'.format(model_name))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load pretrained model
    model = load_pretrained_model(model_name, weights_available, weights_path)
    model.eval().to(device)

    # Load images
    image_paths = Load_images(Image_path)
    print(image_paths[0:10])
    image_num = len(image_paths)

    # Preprocess setting
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),             
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract features
    dataloader = generate_dataloader(image_paths, batch_size=10,preprocess=preprocess)
    Features = Extract_features(model, device, dataloader, feature_layer)

    # Load neural responses
    if Resp_path.endswith('.npy'):
        Resp = np.load(Resp_path)
    elif Resp_path.endswith('.mat'):
        Resp = sio.loadmat(Resp_path)['Resp']

    # Determine the type of regression
    if reg_type == 'LR':
        reg = LinearRegression()
    elif reg_type == 'Ridge':
        reg = Ridge(alpha=reg_para)
    elif reg_type == 'Lasso':
        reg = Lasso(alpha=reg_para)
    elif reg_type == 'PLS':
        reg = PLSRegression(n_components=reg_para)

    # 5-fold Cross validation
    idx_random = np.random.permutation(image_num).astype(np.int32)
    Mean_EV = 0
    for fold_num in range(5):
        test_idx = idx_random[int(fold_num*split_scale*image_num):int((fold_num+1)*split_scale*image_num)]
        train_idx = np.setdiff1d(idx_random, test_idx)
        Train_features, Test_features = Features[train_idx], Features[test_idx]
        Train_Resp, Test_Resp = Resp[train_idx], Resp[test_idx]

        # Train model
        scaler = StandardScaler()
        scaler.fit(Train_features)
        Train_features = scaler.transform(Train_features)
        reg.fit(Train_features, Train_Resp)
        Train_Resp_pred = reg.predict(Train_features)
        Train_EV = Explained_variance(Train_Resp, Train_Resp_pred)
        print("R^2(Explained variance) for Train_Resp: ", Train_EV)

        # Test model
        Test_features = scaler.transform(Test_features)
        Test_Resp_pred = reg.predict(Test_features)
        Test_EV = Explained_variance(Test_Resp, Test_Resp_pred)
        print("R^2(Explained variance) for Test_Resp: ", Test_EV)
        Mean_EV += Test_EV
    Mean_EV /= 5
    print("Mean EV: ", Mean_EV)
if __name__ == '__main__':
    main()
