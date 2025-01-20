import torch
from classes.DQAgent_paper import model_param_path
# Load the .pth file
checkpoint = torch.load(model_param_path)

# View the contents of the checkpoint (e.g., model weights, parameters)
print(checkpoint)