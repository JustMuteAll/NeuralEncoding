import numpy as np
from utils import *
import matplotlib.pyplot as plt


def find_extreme_images(images_file, response_file, pred_file, unit_num=0, num_extreme=5):
    images = Load_data_from_file(images_file)
    responses = Load_data_from_file(response_file)[:,unit_num].reshape(-1)
    pred_response = Load_data_from_file(pred_file)[:,unit_num].reshape(-1)

    # Find the extreme response idx
    true_max_idx = np.argsort(responses)[-num_extreme:][::-1]
    true_min_idx = np.argsort(responses)[:num_extreme]
    pred_max_idx = np.argsort(pred_response)[-num_extreme:][::-1]
    pred_min_idx = np.argsort(pred_response)[:num_extreme]

    # Plot the images
    fig, ax = plt.subplots(4, num_extreme, figsize=(40, 40))
    for i in range(num_extreme):
        true_max_img = images[true_max_idx[i]]
        true_min_img = images[true_min_idx[i]]
        pred_max_img = images[pred_max_idx[i]]
        pred_min_img = images[pred_min_idx[i]]
        if images.shape[1] == 3:
            true_max_img = true_max_img.transpose(2, 1, 0)
            true_min_img = true_min_img.transpose(2, 1, 0)
            pred_max_img = pred_max_img.transpose(2, 1, 0)
            pred_min_img = pred_min_img.transpose(2, 1, 0)

        ax[0, i].imshow(true_max_img)
        ax[0, i].set_title('True Max {}'.format(i+1), fontdict={'fontsize': 40})
        ax[0, i].axis('off')
        ax[2, i].imshow(true_min_img)
        ax[2, i].set_title('True Min {}'.format(i+1), fontdict={'fontsize': 40})
        ax[2, i].axis('off')
        ax[1, i].imshow(pred_max_img)
        ax[1, i].set_title('Pred Max {}'.format(i+1),fontdict={'fontsize':40})
        ax[1, i].axis('off')
        ax[3, i].imshow(pred_min_img)
        ax[3, i].set_title('Pred Min {}'.format(i+1),fontdict={'fontsize':40})
        ax[3, i].axis('off')
    plt.savefig('../Data/{}/extreme_images.png'.format("YourDataFolder"))
    
    return

if __name__ == '__main__':
    
    images_file = r"C:\Users\DELL\Desktop\im96.mat"
    response_file = r"C:\Users\DELL\Desktop\food_96_data.mat"
    pred_file = r"C:\Users\DELL\Desktop\NeuralResponseEncoding-main\NeuralResponseEncoding-main\pred_response.npy"

    find_extreme_images(images_file, response_file, pred_file, 0)
