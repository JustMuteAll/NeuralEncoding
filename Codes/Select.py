import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from utils import *

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))

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
    para_dict = {'preprocess': preprocess, 'model': model, 'model_name': model_name, 'feature_layer': feature_layer, 'device': device, 'pca_used': pca_used, 'reg': reg, 'unit_num': unit_num, 'num_extreme': num_extreme}

    # Load image and response data
    image_file = 'your path'
    resp_file = 'your path'
    Select(image_file, resp_file, para_dict)
