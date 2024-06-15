import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils import *
import time

def main():

    start_time = time.time()
    # Parameter dicts
    model_dict = {'alexnet':['conv2','conv4','fc6'], 
                  'densenet169':['denseblock2','denseblock3','denseblock4','classifier'],
                  'inception_v3':['avgpool','lastconv','midconv'],
                  'resnet101':['avgpool','block4','block3','block2'],
                  'vgg19':['classifier','lastconv','midconv'],
                  'vit_l_16':['lastlayer','midlayer']}
    reg_dict = {'Ridge':[1e-3,1e-2,1e2,1e3,1e4,1e5],
                'Lasso':[1e-3,1e-2,1e2,1e3,1e4,1e5],
                'PLS':[2,5,10,25,50,100]}
    weights_available = 0
    pca_used = 0

    # Set data path
    image_file = r"C:\Users\DELL\Desktop\im96.mat"
    resp_file = r"C:\Users\DELL\Desktop\food_96_data.mat"
    weights_dir = r"D:\Downloads"
    weights_path = os.path.join(weights_dir, '{}.pth'.format(model_name))

    # Set device and random seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),             
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_dataloader, Resp = generate_dataloader(image_file, resp_file, batch_size=10, preprocess=preprocess)

    # First step: Find best aligned model and layer
    Best_model, Best_layer = '', ''
    Best_EV, Best_r = 0,0

    for model_name in model_dict.keys():
        for feature_layer in model_dict[model]:

            # Load pretrained model
            print("Model: ", model_name)
            print("Layer: ", feature_layer)
            model = load_pretrained_model(model_name, weights_available, weights_path)
            model.eval().to(device)

            global result
            result = []
            def fn(module, inputs, outputs):
                result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))
            Register_hook(model, model_name, feature_layer, fn)

            # Extract features
            Extract_features(model, device, image_dataloader)
            feat = np.concatenate(result, axis=0)
            feat = feat.reshape(feat.shape[0],-1)
            result = []

            # Use Ridge(alpha=1e3) as default regression type
            reg = Ridge(alpha=1e3)

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
            if mean_EV/Resp.shape[0] > Best_EV:
                Best_EV = mean_EV/Resp.shape[0]
                Best_r = mean_corr/Resp.shape[0]
                Best_model = model_name
                Best_layer = feature_layer

    # Second step: Find best regression type and parameter
    model = load_pretrained_model(Best_model, weights_available, weights_path)
    model.eval().to(device)

    global result
    result = []
    def fn(module, inputs, outputs):
        result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))
    Register_hook(model, Best_model, Best_layer, fn)

    # Extract features
    Extract_features(model, device, image_dataloader)
    feat = np.concatenate(result, axis=0)
    feat = feat.reshape(feat.shape[0], -1)
    result = []

    Best_reg, Best_para = '', ''
    Best_pred = []
    for reg_type in reg_dict.keys():
        for reg_para in reg_dict[reg_type]:
            print("Regression type: ", reg_type)
            print("Regression parameter: ", reg_para)

            if reg_type == 'Ridge':
                reg = Ridge(alpha=reg_para)
            elif reg_type == 'Lasso':
                reg = Lasso(alpha=reg_para)
            elif reg_type == 'PLS':
                reg = PLSRegression(n_components=int(reg_para))

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
            if mean_EV/Resp.shape[0] > Best_EV:
                Best_EV = mean_EV/Resp.shape[0]
                Best_r = mean_corr/Resp.shape[0]
                Best_reg = reg_type
                Best_para = reg_para
                Best_pred = pred_response

    end_time = time.time()
    print("Total Time:", end_time - start_time)
    print("Best model: ", Best_model)
    print("Best layer: ", Best_layer)
    print("Best regression type: ", Best_reg)
    print("Best regression parameter: ", Best_para)
    print("Best Pearson correlation for Test_Resp: ", Best_r)
    print("Best R^2(Explained variance) for Test_Resp: ", Best_EV)

    pred_response = np.array(pred_response)
    np.save("../Data/{}/pred_response.npy".format("YourResultFolder"), Best_pred)


if __name__ == '__main__':
    main()
