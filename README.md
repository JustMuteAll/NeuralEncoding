# NeuralEncoding
This repository contains python codes for neural response encoding analysis. 

## Environment setup
You can create conda environment by ```conda env create -f environment.yaml``` using environment.yaml in the main directory. 

It is an extensive environment and may include redundant libraries. You may also create environment by checking requirements yourself, an anaconda environment with pytorch and basic data science packages is enough.

## Data preparation
For encoding analysis, you should provide images and neural responses in your experiment.Images and response should be converted into .mat or .npy format,and the index of images and neural responses file should be aligned.    

## GUI 
Use GUI to simplify procedures of analysis by ```python Codes/app.py```.

## Step of encoding analysis
You can also complete analysis by running separate codes when GUI is unavailable. For most following analysis, you need to set the data file path in the codes by yourself.

#### Basic encoding
For basic encoding analysis, using:
```
python Codes/Encoding.py --model_name 'alexnet' --feature_layer 'fc6' --weights_available 'False' --reg_type 'Ridge' --reg_para '1000'
```
You can choose model and regresssion setting in the command line parameters, network and layers available are listed in the function _Register_hook_ in the _utils.py_.
Besides, different kinds of processing for a whole dataset or a training-test split are available.  

#### Searching for the best parameters combination
Find the best network & layer, regression type, regression parameters for encoding by:
```
python Codes/Encoding_search.py
```

#### Predicting neural response for a novel dataset
Construct encoding model by your dataset, and predict neural response for a novel dataset using:
```
python Codes/Encoding_dataset.py
```

#### Show images eliciting the maximal or the minimum response
After encoding analysis, show which images elicited the maximal or the minimum response of a neuron and the result predicted by encoding model using:
```
python Codes/Select.py
```


