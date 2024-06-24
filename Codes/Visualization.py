import argparse
from torchvision import transforms
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from utils import *

# Register hook function
global result
result = []
def fn(module, inputs, outputs):
    result.append(outputs.cpu().detach().numpy().reshape(outputs.shape[0], -1))


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
    Register_hook(model, model_name, feature_layer, fn)
    preprocess = default_preprocess
    pca_used = True
    reg = Register_regression(reg_type, reg_para)
    para_dict = {'preprocess': preprocess, 'model': model, 'model_name': model_name, 'feature_layer': feature_layer, 'device': device, 'pca_used': pca_used, 'reg': reg, 'unit_num': unit_num, 'num_extreme': num_extreme}

    # Load image and response data
    image_file = 'your_image_file_path'
    resp_file = 'your_response_file_path'
    Visualization(image_file, resp_file, para_dict)
