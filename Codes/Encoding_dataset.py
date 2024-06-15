import torch
import argparse
import numpy as np
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import *
import matplotlib.pyplot as plt

def main():
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


    # Set data path
    image_file = r"C:\Users\DELL\Desktop\im96.mat"
    dataset_file = r"C:\Users\DELL\Desktop\Predict\nsd_test_img.npy"
    resp_file = r"C:\Users\DELL\Desktop\food_96_data.mat"
    weights_dir = r"D:\Downloads"
    weights_path = os.path.join(weights_dir, '{}.pth'.format(model_name))

    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
    pca_used = 0

    # Load pretrained model
    model = load_pretrained_model(model_name, weights_available, weights_path)
    model.eval().to(device)

    global result
    result = []
    def fn(module, inputs, outputs):
        result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))
    Register_hook(model, model_name, feature_layer, fn)

    # Load images
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),             
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tr_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)
    te_dataloader, _ = generate_dataloader(dataset_file, resp_file, batch_size=10, preprocess=preprocess)
    # Extract features
    Extract_features(model, device, tr_dataloader)
    tr_feat = np.concatenate(result, axis=0)
    tr_feat = tr_feat.reshape(tr_feat.shape[0],-1)
    result = []
    Extract_features(model, device, te_dataloader)
    te_feat = np.concatenate(result, axis=0)
    te_feat = te_feat.reshape(te_feat.shape[0],-1)

    # Determine the type of regression
    if reg_type == 'LR':
        reg = LinearRegression()
    elif reg_type == 'Ridge':
        reg = Ridge(alpha=reg_para)
    elif reg_type == 'Lasso':
        reg = Lasso(alpha=reg_para)
    elif reg_type == 'PLS':
        reg = PLSRegression(n_components=int(reg_para))

    pred_response = []
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
    reg.fit(tr_feat, Resp)
    tr_pred = reg.predict(tr_feat)

    # Test model
    te_feat = scaler.transform(te_feat)
    te_pred = reg.predict(te_feat)
    pred_response = np.array(te_pred)


    num_extreme = 5
    unit_num = 0
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
    plt.savefig('../Data/{}/Pred_Max_Min_{}_{}.png'.format("YourResultFolder",model_name, unit_num))
    
if __name__ == '__main__':
    main()
