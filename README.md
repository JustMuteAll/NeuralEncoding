# NeuralResponseEncoding
This repository contains python codes for neural response encoding analysis. 

## Method of deep neural network features extraction
The feature extracting module used in this repository is torchlens, a new open-source Python package for extracting and characterizing hidden-layer activations in PyTorch models. 

The Github link of this module is:  https://github.com/johnmarktaylor91/torchlens 

## Environment setup
An anaconda environment with pytorch and basic data science packages is enough, here we list the main required packages:
- Numpy
- Pillow
- Scikit-learn
- Scipy.io
- Torch
- Torchlens == 0.1.2
- Torchvision


## Steps of encoding analysis
1. Prepare images and neural responses used in encoding analysis.
2. If you are working on a network-unreachable device, for example, on a hpc server, please download pretrained model weights to your local path and set weights path mannually in the function "load_pretrained_model".
3. To determine the name of layer you want to extract features from, run the code 'Visualize.py'. 
4. Run the encoding python code and set the command line parameter to get the result you want, for example:
   ```
   python LR_encoding.py --model_name 'alexnet' --feature_layer 'relu_6_19' --weights_available 'True' --reg_type 'Ridge' --reg_para '3000'
   ```

## Functions available & Plan to do
Now you can use pretrained models in the module 'torchvision.models' as image encoder, extract activations of the specific layer and train linear encoders to predict neural response of test images. 

Next I will try to make clip-trained model and ViT(Visual image transformer) available for encoding analysis. Besides, more kinds of readout layer will be added.
