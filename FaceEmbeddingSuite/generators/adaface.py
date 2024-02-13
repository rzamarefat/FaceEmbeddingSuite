from .adaface_utils import build_model
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

class AdaFace:
    def __init__(self, backbone_name="ir_50", device="cpu"):
        self._device = device
        if not(backbone_name in ["ir_101", "ir_50", "is_se_50", "ir_34", "ir_18"]):
            raise RuntimeError("Please provide a valiud backbone name for the adaface model. The options are 'ir_101', 'ir_50', 'is_se_50', 'ir_34', 'ir_18'")

        self._backbone_name = backbone_name
        self._model = build_model(backbone_name=self._backbone_name)
        
        self._ckpt_path = os.path.join(os.getcwd(), "adaface_ir18_casia.ckpt")
        
        
        gdd.download_file_from_google_drive(file_id='1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe',
                                    dest_path=self._ckpt_path,
                                    unzip=True)
        
        

        statedict = torch.load(self._ckpt_path, map_location=torch.device(self._device))['state_dict']
        
        model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
        self._model.load_state_dict(model_statedict)
        self._model.to(self._device)
        self._model.eval()
        
        print("The model is built")
        

    def generate(self, data):
        print("Generated.")


