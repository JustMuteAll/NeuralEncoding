# Encoding_NeuralResponse
This is a repository with encoding python codes of neural response for images. 
## Steps of encoding analysis
1. Prepare images and neural responses used in encoding analysis.
2. pip install -r requirements.txt
3. If you are working on a network-unreachable device, for example, on a hpc server, please download pretrained model weights to your local path and set weights path mannually in the function "load_pretrained_model".
4. run the encoding python code and set the command line parameter to get the result you want, for example:
   ```
   python LR_encoding.py
   ```
 ## Functions available & Plan to do
Now you can use pretrained models in the module 'torchvision.models' as image encoder, extract activations of the specific layer and train linear encoders to predict neural response of test images. 
Next I will try to make clip-trained model and ViT(Visual image transformer) available for encoding analysis. Besides, more kinds of readout layer will be added.
