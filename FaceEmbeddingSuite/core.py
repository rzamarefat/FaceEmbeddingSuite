from .generators.adaface import AdaFace


class GeneratorManager:
    def __init__(self, generator_name="adaface"):
        self._generator_name = generator_name
        if self._generator_name == "adaface":
            self._generator = AdaFace()

        else:
            raise RuntimeError("Please provide a valid name for generator. The valid options are ['adaface', 'magface', 'poseface', 'sface', 'cosface', 'arcface']")

    def __call__(self, data):
        self._generator.generate(data)

    