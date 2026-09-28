"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 2: incarcarea si preprocesarea setului de date MNIST.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Directorul unde salvam toate rezultatele pentru raport.
RESULTS_DIR = "results"


def incarca_si_preproceseaza(verbose=True):
    """
    Incarca MNIST si il pregateste pentru retelele convolutionale.

    Returneaza:
        x_train : (60000, 28, 28, 1) float32 in [0, 1]
        x_test  : (10000, 28, 28, 1) float32 in [0, 1]
        y_test  : (10000,) uint8 - etichetele NU se folosesc la antrenare,
                  ci doar la Etapa 10, pentru a colora spatiul latent.
    """
    # Etichetele de antrenare nu ne sunt necesare deloc -> le ignoram cu "_".
    # Etichetele de test le pastram STRICT pentru vizualizarea spatiului latent.
    (x_train, _), (x_test, y_test) = mnist.load_data()

    if verbose:
        print("=" * 60)
        print("DATE BRUTE (asa cum vin din Keras)")
        print("=" * 60)
        print(f"x_train : {x_train.shape}, dtype={x_train.dtype}, "
              f"min={x_train.min()}, max={x_train.max()}")
        print(f"x_test  : {x_test.shape}, dtype={x_test.dtype}, "
              f"min={x_test.min()}, max={x_test.max()}")

    # --- PASUL 1: conversie la float32 --------------------------------------
    # Retelele neuronale lucreaza cu numere reale. uint8 (0-255) nu poate fi
    # folosit direct in calculul gradientilor.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # --- PASUL 2: normalizare in [0, 1] -------------------------------------
    # Impartim la 255 (valoarea maxima a unui pixel). Motivul este dublu:
    #   a) antrenarea converge mai repede si mai stabil cu intrari mici;
    #   b) stratul final al decoderului va folosi activarea 'sigmoid',
    #      care produce iesiri tot in [0, 1] -> intrarea si iesirea sunt
    #      pe aceeasi scara, iar 'binary_crossentropy' devine o functie
    #      de pierdere valida.
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # --- PASUL 3: adaugarea dimensiunii de canal ----------------------------
    # Conv2D asteapta tensori de forma (inaltime, latime, canale).
    # MNIST este grayscale -> 1 singur canal. Trecem de la
    # (60000, 28, 28) la (60000, 28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    if verbose:
        print()
        print("=" * 60)
        print("DATE DUPA PREPROCESARE")
        print("=" * 60)
        print(f"x_train : {x_train.shape}, dtype={x_train.dtype}, "
              f"min={x_train.min():.1f}, max={x_train.max():.1f}")
        print(f"x_test  : {x_test.shape}, dtype={x_test.dtype}, "
              f"min={x_test.min():.1f}, max={x_test.max():.1f}")
        print(f"Imagini de antrenare : {len(x_train)}")
        print(f"Imagini de test      : {len(x_test)}")
        print(f"Pixeli per imagine   : {28 * 28} "
              f"(dimensiunea spatiului de intrare)")
        print(f"Memorie x_train      : "
              f"{x_train.nbytes / (1024 ** 2):.1f} MB")
        print("=" * 60)

    return x_train, x_test, y_test


def afiseaza_esantioane(x, y=None, n=10, nume_fisier="mnist_samples.png"):
    """
    Afiseaza si salveaza primele n imagini din setul de date.
    Este o verificare vizuala: confirma ca preprocesarea nu a stricat datele.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(n * 1.2, 1.8))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        # squeeze() elimina dimensiunea de canal (28,28,1) -> (28,28),
        # pentru ca imshow asteapta o matrice 2D pentru imagini grayscale.
        plt.imshow(x[i].squeeze(), cmap="gray")
        if y is not None:
            ax.set_title(str(y[i]), fontsize=10)
        plt.axis("off")

    plt.suptitle("MNIST - esantioane dupa preprocesare (valori in [0, 1])",
                 fontsize=11)
    plt.tight_layout()

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    plt.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    x_train, x_test, y_test = incarca_si_preproceseaza()
    afiseaza_esantioane(x_test, y_test, n=10)