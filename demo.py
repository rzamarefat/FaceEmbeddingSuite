import FaceEmbeddingSuite as fes
import cv2

def main():
    data = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\FaceEmbeddingSuite\test_images\two_face.png")
    detector = fes.DetectorManager(detector_name="yolov8")

    detection_data = detector(data)

    # AdaFace
    # generator = fes.GeneratorManager(generator_name="ir_18_casia_webface")
    # data = generator(detection_data)
    # print(data)

    # # SFace
    # generator = fes.GeneratorManager(generator_name="sface__casia_webface")
    # data = generator(detection_data)
    # print(data)

    # MagFace
    generator = fes.GeneratorManager(generator_name="magface__iresnet100")
    data = generator(detection_data)
    # print(data)


if __name__ == "__main__":
    main()