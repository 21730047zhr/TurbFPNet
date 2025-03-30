from torch.nn import Module
import torch


def load_state_dict(
        model: Module,
        model_weights_path: str,
) -> Module:
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    pretrained_dict = checkpoint
    model_state_dict = model.state_dict()

    keys=[]
    for k,v in pretrained_dict.items():
      keys.append(k)
    i=0
    for k, v in model_state_dict.items():
      if v.size()== pretrained_dict[keys[i]].size():
        model_state_dict[k] = pretrained_dict[keys[i]]
        i = i+1
    model.load_state_dict(model_state_dict)

    return model