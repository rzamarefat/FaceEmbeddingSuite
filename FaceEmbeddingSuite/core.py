from typing import Any
from .generators.adaface import AdaFace
from .generators.sface import SFace
from .generators.magface import MagFace
from .detectors.YOLOv8 import YOLOLandmark


class GeneratorManager:
    def __init__(self, 
                 generator_name="ir_101_webface_12m", 
                 device="cuda"
                 ):

        self._device = device

        self._adaface_model_name_list = ["ir_101_webface_12m", "ir_101_webface_4m", "ir_101_ms1_mv3", "ir_101_ms1_mv2", "ir_50_ms1_mv2", "ir_50_webface_4m", "ir_50_casia_webface", "ir_18_webface_4m", "ir_18_vggface2", "ir_18_casia_webface"]
        self._sface_model_name_list = ["sface__casia_webface"]
        self._magface_model_name_list = []

        self._all_allowed_model_name_list = self._adaface_model_name_list + self._sface_model_name_list + self._magface_model_name_list


        self._generator_name = generator_name

        if not(self._generator_name in self._all_allowed_model_name_list):
            raise RuntimeError("""Please provide a valid backbone name for the adaface model. The options are \n- 'ir_101_webface_12m' \n- 'ir_101_webface_4m'\n- 'ir_101_ms1_mv3'\n- 'ir_101_ms1_mv2'\n- 'ir_50_ms1_mv2'\n- 'ir_50_webface_4m'\n- 'ir_50_casia_webface'\n- 'ir_18_webface_4m'\n- 'ir_18_vggface2'\n- 'ir_18_casia_webface'""")
        
        if self._generator_name in self._adaface_model_name_list:
            self._generator = AdaFace(self._generator_name, self._device)
        elif self._generator_name in self._sface_model_name_list:
            self._generator = SFace(self._generator_name, self._device)
        elif self._generator_name in self._magface_model_name_list:
            self._generator = MagFace(self._generator_name, self._device)
        else:
            raise RuntimeError("Please provide a valid name for generator. The valid options are ['adaface', 'magface', 'poseface', 'sface', 'cosface', 'arcface']")

    def __call__(self, data):
        res = self._generator.generate(data)

        return res




class DetectorManager:
    def __init__(self,
                 detector_name="yolov8",
                 device="cuda"
                 ):
        
        self._detector_name = detector_name
        self._device = device


        if self._detector_name == "yolov8":
            self._detector = YOLOLandmark(device=self._device)
    

    def __call__(self, data, find_largest_box=False):
        res = self._detector.detect(data, find_largest_box=find_largest_box)

        return res
            

        