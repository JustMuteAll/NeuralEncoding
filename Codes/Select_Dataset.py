import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import *

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))

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

if __name__ == '__main__':
     # Read command line arguments
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', default='alexnet', type=str)
    args.add_argument('--feature_layer', default='fc6', type=str)
    args.add_argument('--reg_type', default='Ridge', type=str)
    args.add_argument('--reg_para', default=1000, type=float)
    args.add_argument('--unit_num', default=0, type=int)
    args.add_argument('--num_extreme', default=5, type=int)

    # Set parameters
    Parser = args.parse_args()
    model_name = str(Parser.model_name)
    feature_layer = str(Parser.feature_layer)
    reg_type = str(Parser.reg_type)
    reg_para = float(Parser.reg_para)
    unit_num = int(Parser.unit_num)
    num_extreme = int(Parser.num_extreme)

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

    # Load model
    model = load_pretrained_model(model_name).to(device).eval()
    preprocess = default_preprocess
    pca_used = True
    reg = Register_regression(reg_type, reg_para)
    para_dict = {'preprocess': preprocess, 'model': model, 'model_name': model_name, 'model_layer': feature_layer, 'device': device, 'pca_used': pca_used, 'reg': reg, 'unit_num': unit_num, 'num_extreme': num_extreme}

    # Load image and response data
    image_file = 'your path'
    resp_file = 'your path'
    dataset_file = 'your path'
    Select_Dataset(image_file, resp_file, dataset_file, para_dict)
    
