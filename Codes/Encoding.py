import torch
import numpy as np
import argparse
from torchvision import transforms
from utils import *

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))

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

if __name__ == '__main__':
    # Read command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', default='alexnet', type=str)
    args.add_argument('--feature_layer', default='fc6', type=str)
    args.add_argument('--reg_type', default='Ridge', type=str)
    args.add_argument('--reg_para', default=1000, type=float)

    # Set parameters
    Parser = args.parse_args()
    model_name = str(Parser.model_name)
    feature_layer = str(Parser.feature_layer)
    reg_type = str(Parser.reg_type)
    reg_para = float(Parser.reg_para)

    # Set device and random seed
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Preprocess settings
    default_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),             
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = load_pretrained_model(model_name).to(device).eval()
    preprocess = default_preprocess
    pca_used = True
    reg = Register_regression(reg_type, reg_para)
    para_dict = {'preprocess': preprocess, 'model': model, 'model_name': model_name, 'feature_layer': feature_layer, 'device': device, 'pca_used': pca_used, 'reg': reg}

    # Load image and response data
    image_file = "your path"
    resp_file = "your path"
    Encoding(image_file, resp_file, para_dict)
