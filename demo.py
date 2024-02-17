import FaceEmbeddingSuite as fes
import cv2

def main():
    data = cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\FaceEmbeddingSuite\test_images\two_face.png")
    detector = fes.DetectorManager(detector_name="yolov8")
    generator = fes.GeneratorManager(generator_name="ir_18_casia_webface")

    detection_data = detector(data)
    
    data = generator(detection_data)

    print(data)


if __name__ == "__main__":
    main()