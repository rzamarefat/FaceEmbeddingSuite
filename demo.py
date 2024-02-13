import FaceEmbeddingSuite as fes

def main():
    data = None
    generator = fes.GeneratorManager(generator_name="ir_18_casia_webface")
    generator(data)

if __name__ == "__main__":
    main()