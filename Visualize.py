import torch
import torchvision.models as models
import torchlens as tl

# Load your model 
model = models.resnet50(pretrained=True)
input = torch.randn(1, 3, 224, 224)
# Visual the model using torchlens
tl.log_forward_pass(model, input, vis_opt="rolled")

