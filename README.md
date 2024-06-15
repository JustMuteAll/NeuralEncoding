# NeuralEncoding
This repository contains python codes for neural response encoding analysis. 

## Environment setup
You can create conda environment using environment.yml in the main directory by entering conda env create -f environment.yml . It is an extensive environment and may include redundant libraries. You may also create environment by checking requirements yourself, an anaconda environment with pytorch and basic data science packages is enough.

## Data preparation
For encoding analysis, you should provide images and neural responses in your experiment.Images should be converted into one .mat or .npy file(Or two files, if there is explicit training/test set split in your experiment),and the index of images and neural responses file should be aligned.    

## Step of encoding analysis
For most following analysis, you need to set the data file path in the codes by yourself.

#### Basic encoding
For basic encoding analysis, using:
```
python Codes/Encoding.py --model_name 'alexnet' --feature_layer 'relu_6_19' --weights_available 'False' --reg_type 'Ridge' --reg_para '1000'
```
You can choose model and regresssion setting in the command line parameters, and different kinds of processing for a whole dataset or a training-test split are available.  

#### Searching for the best parameters combination
Find the best network & layer, regression type, regression parameters for encoding by:
```
python Codes/Encoding_search.py
```

#### Predicting neural response for a novel dataset
Construct encoding model by your dataset, and predict neural response for a novel dataset using:
'''
python Codes/Encoding_dataset.py
'''

#### Show images eliciting the maximal or the minimum response
After encoding analysis, show which images elicited the maximal or the minimum response of a neuron and the result predicted by encoding model using:
```
python Codes/Select.py
```


