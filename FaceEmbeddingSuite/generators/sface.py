import torch
from .sface_utils import iresnet50
from torchvision import transforms
import os
import gdown
from PIL import Image

class SFace:
    def __init__(self, 
                backbone_name, 
                device
                ):

        self._backbone_name = backbone_name

        if self._backbone_name == "sface__casia_webface":
            ckpt_name = "3832backbone.pt"
            ckpt_url="https://drive.google.com/uc?id=1q9yaAYHX_PWEzrcCNPw0K-d_YXrFumFz"
            arch_type = "iresnet50"
        else:
            raise RuntimeError("")
        
        self._ckpt_path = os.path.join(os.getcwd(), "weights", ckpt_name)
        if not(os.path.isfile(self._ckpt_path)):
            gdown.download(ckpt_url, self._ckpt_path, quiet=False)

        self._ckpt_path = os.path.join(os.getcwd(), "weights", ckpt_name)
        self.device = device
        self._sface_model = self._build_sface_model()

        
        
    def _apply_transforms(self, faces):
        transformed_data = []
        sface_required_transforms = transforms.Compose([
                            transforms.Resize((112,112)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
        
        for face in faces:
            face = Image.fromarray(face)
            transformed_data.append(sface_required_transforms(face))
        
        return torch.stack(transformed_data)

    

    def _build_sface_model(self):
        model = iresnet50(num_features=512, use_se=False).to(self.device)
        model.load_state_dict(torch.load(self._ckpt_path, map_location=torch.device(self.device)))
        model.to(self.device)
        model.eval()
        return model

    def generate(self, detection_data):
        faces_data = self._apply_transforms(detection_data["faces"])

        if len(faces_data.shape) < 4:
            faces_data = torch.unsqueeze(faces_data, dim=0)

        faces_data = faces_data.to(self.device)
        emb_data = self._sface_model(faces_data)

        embs = []
        for index in range(len(emb_data)):
            emb = emb_data[index]
            embs.append(emb)

        detection_data.__setitem__("embs", embs)

        return detection_data

            
