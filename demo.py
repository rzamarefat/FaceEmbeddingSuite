import FaceEmbeddingSuite as fes

def main():
    data = None
    generator = fes.GeneratorManager(generator_name="adaface")
    generator(data)

if __name__ == "__main__":
    main()