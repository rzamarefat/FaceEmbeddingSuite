from .adaface_utils import build_model
import torch
import gdown
import os

class AdaFace:
    def __init__(self, backbone_name, device):
        print("backbone_name", backbone_name)
        self._device = device
        self._backbone_name = backbone_name
        if self._backbone_name == "ir_101_webface_12m":
            ckpt_name = "adaface_ir101_webface12m.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT"
            arch_type = "ir_101"
        elif self._backbone_name == "ir_101_webface_4m":
            ckpt_name = "adaface_ir101_webface4m.ckpt"
            ckpt_url="https://drive.google.com/uc?id=18jQkqB0avFqWa0Pas52g54xNshUOQJpQ"
            arch_type = "ir_101"
        elif self._backbone_name == "ir_101_ms1_mv3":
            ckpt_name = "adaface_ir101_ms1mv3.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI"
            arch_type = "ir_101"
        elif self._backbone_name == "ir_101_ms1_mv2":
            ckpt_name = "adaface_ir101_ms1mv2.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1m757p4-tUU5xlSHLaO04sqnhvqankimN"
            arch_type = "ir_101"
        elif self._backbone_name == "ir_50_ms1_mv2":
            ckpt_name = "adaface_ir50_ms1mv2.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI"
            arch_type = "ir_50"
        elif self._backbone_name == "ir_50_webface_4m":
            ckpt_name = "adaface_ir50_webface4m.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN"
            arch_type = "ir_50"
        elif self._backbone_name == "ir_50_casia_webface":
            ckpt_name = "adaface_ir50_casia.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2"
            arch_type = "ir_50"
        elif self._backbone_name == "ir_18_webface_4m":
            ckpt_name = "adaface_ir18_webface4m.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y"
            arch_type = "ir_18"
        elif self._backbone_name == "ir_18_vggface2":
            ckpt_name = "adaface_ir18_vgg2.ckpt"
            ckpt_url="https://drive.google.com/uc?id=1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv"
            arch_type = "ir_18"
        elif self._backbone_name == "ir_18_casia_webface":
            ckpt_name = "adaface_ir18_casia"
            ckpt_url="https://drive.google.com/uc?id=1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe"
            arch_type = "ir_18"
        else:
            raise RuntimeError("")
        self._model = build_model(arch_type=arch_type)
        os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)

        
        self._ckpt_path = os.path.join(os.getcwd(), "weights", ckpt_name)

        if not(os.path.isfile(self._ckpt_path)):
            gdown.download(ckpt_url, self._ckpt_path, quiet=False)

        try:
            statedict = torch.load(self._ckpt_path, map_location=torch.device(self._device))['state_dict']
            model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
            self._model.load_state_dict(model_statedict)
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"An error occured: {e}")
        
        print("The model is built")
        

    def generate(self, data):
        print("Generated.")


