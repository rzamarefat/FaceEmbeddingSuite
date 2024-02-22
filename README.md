# Face Embedding Suite

### Usage: Generators
```python
import FaceEmbeddingSuite as fes
import cv2

def main():
    data = cv2.imread("path/to/person.jpg")
    detector = fes.DetectorManager(detector_name="yolov8")

    # AdaFace    
    generator = fes.GeneratorManager(generator_name="ir_18_casia_webface")
    detection_data = detector(data)
    data = generator(detection_data)

    # SFace
    generator = fes.GeneratorManager(generator_name="sface__casia_webface")
    data = generator(detection_data)

    print(data)
```
