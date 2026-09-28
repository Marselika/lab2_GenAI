"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 1: verificarea mediului de lucru.
"""

import sys
import platform


def main():
    print("=" * 60)
    print("VERIFICAREA MEDIULUI DE LUCRU")
    print("=" * 60)

    # 1. Informatii despre Python si sistemul de operare
    print(f"Python        : {sys.version.split()[0]}")
    print(f"Sistem        : {platform.system()} {platform.release()}")
    print(f"Arhitectura   : {platform.machine()}")
    print("-" * 60)

    # 2. Verificarea bibliotecilor obligatorii
    #    Fiecare import este intr-un try/except separat, ca sa vedem
    #    exact ce lipseste, nu doar prima eroare.
    try:
        import numpy as np
        print(f"NumPy         : {np.__version__}")
    except ImportError:
        print("NumPy         : LIPSESTE  -> pip install numpy")

    try:
        import matplotlib
        print(f"Matplotlib    : {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib    : LIPSESTE  -> pip install matplotlib")

    try:
        import sklearn
        print(f"scikit-learn  : {sklearn.__version__}")
    except ImportError:
        print("scikit-learn  : LIPSESTE  -> pip install scikit-learn")

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow    : LIPSESTE  -> pip install tensorflow")
        print("=" * 60)
        return

    print(f"TensorFlow    : {tf.__version__}")

    # 3. Versiunea de Keras. In TF >= 2.16 backend-ul implicit este Keras 3,
    #    ceea ce schimba modul in care implementam loss-ul VAE (Etapa 7).
    import keras
    print(f"Keras         : {keras.__version__}")
    print("-" * 60)

    # 4. Dispozitivele disponibile. Laboratorul este proiectat sa ruleze pe CPU;
    #    daca exista un GPU, TensorFlow il foloseste automat.
    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"CPU-uri vizibile : {len(cpus)}")
    print(f"GPU-uri vizibile : {len(gpus)}")
    if not gpus:
        print("  -> Antrenarea se va face pe CPU (este suficient pentru MNIST).")
    print("-" * 60)

    # 5. Test functional minim: construim si rulam un model banal.
    #    Daca acest pas trece, instalarea este realmente functionala,
    #    nu doar importabila.
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(2, activation="relu"),
    ])
    iesire = model(np.zeros((1, 4), dtype="float32"))
    print(f"Test model      : OK, forma iesirii = {tuple(iesire.shape)}")

    # 6. Test de acces la MNIST. La prima rulare descarca ~11 MB
    #    in ~/.keras/datasets/ si il pastreaza in cache.
    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    print(f"MNIST train     : {x_train.shape}")
    print(f"MNIST test      : {x_test.shape}")

    print("=" * 60)
    print("MEDIUL ESTE PREGATIT.")
    print("=" * 60)


if __name__ == "__main__":
    main()