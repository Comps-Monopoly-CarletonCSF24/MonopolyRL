import torch
from classes.DQAgent.DQAgent import QNetwork
from classes.state import State_Size
from classes.DQAgent.action import Action_Size

def main():
    pytorch_model = QNetwork()
    checkpoint = torch.load('./classes/DQAGent/model_parameters.pth')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()

    dummy_input = torch.zeros(State_Size + Action_Size, dtype=torch.float32)
    torch.onnx.export(pytorch_model, 
                      dummy_input, 
                      "./JavaScript Engine/model.onnx", 
                      opset_version=11, 
                      export_params=True,
                      input_names=["state_action_input"], 
                      output_names=["q_value_output"],  
        )


if __name__ == "__main__":
    main()
