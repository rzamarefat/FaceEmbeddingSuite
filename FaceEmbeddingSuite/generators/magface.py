import sys
import os
import os

import torch
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from .magface_utils import *
import gdown



class MagFace:
    def __init__(self, backbone_name, device):

        self._device = device
        self._backbone_name = backbone_name

        if self._backbone_name == "magface__iresnet100":
            ckpt_name = "magface_epoch_00025.pth"
            ckpt_url="https://drive.google.com/uc?id=1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H"
            arch_type = "iresnet100"
            features_model = iresnet100(
            pretrained=False,
            num_classes=512,
            )
        elif self._backbone_name == "magface__iresnet50":
            ckpt_name = "magface_iresnet50_MS1MV2_ddp_fp32.pth"
            ckpt_url="https://drive.google.com/uc?id=1QPNOviu_A8YDk9Rxe8hgMIXvDKzh6JMG"
            arch_type = "iresnet100"
            features_model = iresnet50(
            pretrained=False,
            num_classes=512,
            )
        elif self._backbone_name == "magface__iresnet18":
            ckpt_name = "magface_iresnet18_casia_dp.pth"
            ckpt_url="https://drive.google.com/uc?id=18pSIQOHRBQ-srrYfej20S5M8X8b_7zb9"
            arch_type = "iresnet100"
            features_model = iresnet18(
            pretrained=False,
            num_classes=512,
            )
        else:
            raise RuntimeError("")
        

        self._ckpt_path = os.path.join(os.getcwd(), "weights", ckpt_name)

        if not(os.path.isfile(self._ckpt_path)):
            gdown.download(ckpt_url, self._ckpt_path, quiet=False)



        self.features_model = self._load_dict_inf(features_model).to(self._device)


        self.transforms = transforms.Compose([
                    transforms.Normalize(
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.]),
        ])

    def _load_dict_inf(self, model):    
        checkpoint = torch.load(self._ckpt_path, map_location=self._device)
        _state_dict = self._clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        del checkpoint
        del _state_dict

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


    def _preprocess_data(self, faces):

        faces_data = []
        for face in faces:
            face = np.array(face, dtype=np.float32)
            empty_tensor = np.zeros((face.shape[0], face.shape[1], face.shape[2]), dtype=np.float32)
            face = np.array([face, empty_tensor])
            face_tensor = self.transforms(torch.from_numpy(face).permute(0, 3, 1, 2))
            # print("face_tensor.shape", face_tensor.shape)


        face_tensor = face_tensor.to(self._device)
        
        return face_tensor
        

    def generate(self, detection_data):
        
        faces_data = self._preprocess_data(detection_data["faces"])
        print("faces_data.shape======", faces_data.shape)
        generated_embs = self.features_model(faces_data)
        generated_embs = generated_embs[0].detach().cpu()

        return generated_embs 

