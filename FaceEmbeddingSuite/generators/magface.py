import sys
import os
import os

import torch
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import iresnet

class MagFace:
    def __init__(self, backbone_name, device):
        self.transforms = transforms.Compose([
                    transforms.Normalize(
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.]),
        ])
        self._device = device

        features_model = iresnet.iresnet100(
            pretrained=False,
            num_classes=512,
        )
        self.features_model = self._load_dict_inf(config, features_model).to(self.device)

    def _load_dict_inf(self, config, model):
        if os.path.isfile(config["magface_pretrained_path"]):
            
            if config["cpu_mode"]:
                checkpoint = torch.load(config["magface_pretrained_path"], map_location=torch.device("cpu"))
            else:
                checkpoint = torch.load(config["magface_pretrained_path"])
            _state_dict = self._clean_dict_inf(model, checkpoint['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(_state_dict)
            model.load_state_dict(model_dict)
            del checkpoint
            del _state_dict
            print("=> Magface pretrained model is loaded successfully")
        else:
            sys.exit(f"=> No checkpoint found at: {config['magface_pretrained_path']}")
        return model


    def _clean_dict_inf(self, model, state_dict):
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            
            new_k = '.'.join(k.split('.')[2:])
            if new_k in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_k].size():
                _state_dict[new_k] = v
        
            new_kk = '.'.join(k.split('.')[1:])
            if new_kk in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_kk].size():
                _state_dict[new_kk] = v
        num_model = len(model.state_dict().keys())
        num_ckpt = len(_state_dict.keys())
        if num_model != num_ckpt:
            sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
                num_model, num_ckpt))
        return _state_dict


    def _preprocess_data(self, img):
        img = np.array(img, dtype=np.float32)
        empty_tensor = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        img = np.array([img, empty_tensor])
        img_tensor = self.transforms(torch.from_numpy(img).permute(0, 3, 1, 2))

        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
        

    def generate(self, detection_data):
        


        if isinstance(type(img), np.ndarray):
            raise Exception("The type of data given to 'generate_embeddings' method is not a numpy array. \
You have probably forgotten to read the images as np.array")
        
        if not(img.shape[1] == 112 or img.shape[2] == 112):
            raise Exception("The size of images provided for this 'generate_embeddings' method must be 112*112")

        img = self._preprocess_data(img)
        generated_embs = self.features_model(img)
        generated_embs = generated_embs[0].detach().cpu()

        return generated_embs 

