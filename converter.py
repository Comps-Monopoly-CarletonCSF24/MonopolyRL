import torch
from classes.DQAgent.DQAgent import QNetwork
from classes.state import State_Size
from classes.DQAgent.action import Action_Size

def main():
    pytorch_model = QNetwork()
    checkpoint = torch.load('model_parameters.pth')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()

    dummy_input = torch.zeros(State_Size + Action_Size, dtype=torch.float32)

    torch.onnx.export(
        pytorch_model, 
        dummy_input, 
        "onnx_model.onnx", 
        verbose=True,
        input_names=["state_action_input"], 
        output_names=["q_value_output"],  
        opset_version=11 
    )

if __name__ == "__main__":
    main()
