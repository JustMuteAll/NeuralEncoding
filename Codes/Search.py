import torch
import numpy as np
from torchvision import transforms
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

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

if __name__ == '__main__':

    image_file = "your path"
    resp_file = "your path"
    split_ratio = 0.2

    # Set device and random seed
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    Search(image_file, resp_file, split_ratio)
