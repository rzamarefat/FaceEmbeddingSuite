import FaceEmbeddingSuite as fes

def main():
    data = None
    detector = fes.DetectorManager(detector_name="yolov8")
    generator = fes.GeneratorManager(generator_name="ir_18_casia_webface")

    detector(data)
    generator(data)

if __name__ == "__main__":
    main()