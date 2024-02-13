from .generators.adaface import AdaFace


class GeneratorManager:
    def __init__(self, 
                 generator_name="ir_101_webface_12m", 
                 device="cuda"
                 ):

        self._device = device
        self._adaface_model_name_list = ["ir_101_webface_12m", "ir_101_webface_4m", "ir_101_ms1_mv3", "ir_101_ms1_mv2", "ir_50_ms1_mv2", "ir_50_webface_4m", "ir_50_casia_webface", "ir_18_webface_4m", "ir_18_vggface2", "ir_18_casia_webface"]
        self._generator_name = generator_name

        if not(self._generator_name in self._adaface_model_name_list):
            raise RuntimeError("""Please provide a valid backbone name for the adaface model. The options are \n- 'ir_101_webface_12m' \n- 'ir_101_webface_4m'\n- 'ir_101_ms1_mv3'\n- 'ir_101_ms1_mv2'\n- 'ir_50_ms1_mv2'\n- 'ir_50_webface_4m'\n- 'ir_50_casia_webface'\n- 'ir_18_webface_4m'\n- 'ir_18_vggface2'\n- 'ir_18_casia_webface'""")
        
        if self._generator_name in self._adaface_model_name_list:
            self._generator = AdaFace(self._generator_name, self._device)

        else:
            raise RuntimeError("Please provide a valid name for generator. The valid options are ['adaface', 'magface', 'poseface', 'sface', 'cosface', 'arcface']")

    def __call__(self, data):
        self._generator.generate(data)

    