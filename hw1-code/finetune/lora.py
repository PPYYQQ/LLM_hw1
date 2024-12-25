import torch
import torch.nn as nn
import transformers
import math

from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        # self.lora_right_weight = None
        # self.lora_left_weight = None
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        self.lora_right_weight = nn.Parameter(torch.empty(lora_dim, in_features))
        self.lora_left_weight = nn.Parameter(torch.empty(out_features, lora_dim))
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        #
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)
        # raise NotImplementedError
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        # origin
        original_output = nn.functional.linear(input, self.weight, self.bias)
        # LoRA 
        lora_output = input @ self.lora_right_weight.t()  # shape: (batch_size, lora_dim)
        lora_output = lora_output @ self.lora_left_weight.t()  # shape: (batch_size, output_dim)
        lora_output = lora_output * self.lora_scaling
        return original_output + lora_output
        # raise NotImplementedError
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    print('i am here')
    print(model)
    return model


def only_optimize_lora_parameters(model):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
    # raise NotImplementedError
    ##############################################################################

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora_' in k}
    return lora_state_dict
    # raise NotImplementedError
    ########################################################