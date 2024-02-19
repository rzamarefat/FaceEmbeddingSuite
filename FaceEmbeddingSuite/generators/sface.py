import torch
from .sface_utils import iresnet50
from torchvision import transforms

class SFace:
    def __init__(self, 
                path_to_sface_ckpt, 
                device='cpu'
                ):
        
        self.path_to_sface_ckpt = path_to_sface_ckpt
        self.device = device
        self._sface_model = self._build_sface_model()

        self._sface_required_transforms = transform = transforms.Compose([
                            transforms.Resize((112,112)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                            ])
    

    def _build_sface_model(self):
        model = iresnet50(num_features=512, use_se=False).to(self.device)
        model.load_state_dict(torch.load(self.path_to_sface_ckpt, map_location=torch.device(self.device)))
        model.to(self.device)
        model.eval()
        return model

    def generate_emb(self, face):

        face = self._sface_required_transforms(face)
        if len(face.shape) < 4:
            face = torch.unsqueeze(face, dim=0)

        face = face.to(self.device)
        emb = self._sface_model(face)

        return emb.detach().to('cpu').numpy()

if __name__ == "__main__":
    import cv2
    from glob import glob 
    from tqdm import tqdm


    sface_emb_gen = SFaceEmbGenerator(
        path_to_sface_ckpt = "/home/rmarefat/projects/face/deliver_to_dr/pretrained_weights/SFace.pth",
    )

    face_image = cv2.imread("/home/rmarefat/projects/face/DATA/authentication/id_photos_faces/1.jpg")
    emb = sface_emb_gen.generate_emb(face_image)

    print(emb.shape)
            
