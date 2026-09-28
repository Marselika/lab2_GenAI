"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 3: construirea Autoencoder-ului clasic (convolutional).
"""

import keras
from keras import layers

# Dimensiunea spatiului latent. Aceeasi valoare va fi folosita si la VAE,
# ca cele doua modele sa fie comparabile in mod corect (Etapa 12).
LATENT_DIM = 16


def construieste_encoder(latent_dim=LATENT_DIM):
    """
    Encoderul: comprima o imagine 28x28x1 intr-un vector de 'latent_dim' numere.
    """
    intrare = keras.Input(shape=(28, 28, 1), name="imagine_intrare")

    # Bloc convolutional 1: extrage trasaturi simple (muchii, colturi).
    # padding="same" pastreaza dimensiunea spatiala: 28x28 -> 28x28.
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(intrare)
    # MaxPooling injumatateste rezolutia: 28x28 -> 14x14.
    x = layers.MaxPooling2D(2, padding="same")(x)

    # Bloc convolutional 2: trasaturi mai complexe (bucle, intersectii).
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 14x14 -> 7x7

    # Aplatizam harta 7x7x64 = 3136 de valori intr-un vector.
    x = layers.Flatten()(x)

    # BOTTLENECK-ul propriu-zis: 3136 -> latent_dim.
    # Fara activare (liniar), ca valorile latente sa poata fi si negative,
    # exact ca vectorul z_mean al VAE-ului de la Etapa 6.
    z = layers.Dense(latent_dim, name="vector_latent")(x)

    return keras.Model(intrare, z, name="encoder")


def construieste_decoder(latent_dim=LATENT_DIM):
    """
    Decoderul: reconstruieste o imagine 28x28x1 pornind de la vectorul latent.
    Este oglinda encoderului.
    """
    intrare_latenta = keras.Input(shape=(latent_dim,), name="vector_latent")

    # Expandam vectorul latent inapoi la volumul 7x7x64.
    x = layers.Dense(7 * 7 * 64, activation="relu")(intrare_latenta)
    x = layers.Reshape((7, 7, 64))(x)

    # Bloc de reconstructie 1: 7x7 -> 14x14
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    # Bloc de reconstructie 2: 14x14 -> 28x28
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    # Stratul final: 1 canal, activare sigmoid -> valori in [0, 1],
    # exact intervalul in care am normalizat imaginile de intrare.
    iesire = layers.Conv2D(1, 3, activation="sigmoid", padding="same",
                           name="imagine_reconstruita")(x)

    return keras.Model(intrare_latenta, iesire, name="decoder")


def construieste_autoencoder(latent_dim=LATENT_DIM):
    """
    Leaga encoderul si decoderul intr-un singur model antrenabil.
    Returneaza (autoencoder, encoder, decoder) - avem nevoie de encoder si
    decoder separat la Etapele 10-11.
    """
    encoder = construieste_encoder(latent_dim)
    decoder = construieste_decoder(latent_dim)

    intrare = keras.Input(shape=(28, 28, 1), name="imagine_intrare")
    z = encoder(intrare)
    reconstructie = decoder(z)

    autoencoder = keras.Model(intrare, reconstructie, name="autoencoder")
    return autoencoder, encoder, decoder


if __name__ == "__main__":
    autoencoder, encoder, decoder = construieste_autoencoder()

    print("\n" + "=" * 65)
    print("ENCODER")
    print("=" * 65)
    encoder.summary()

    print("\n" + "=" * 65)
    print("DECODER")
    print("=" * 65)
    decoder.summary()

    print("\n" + "=" * 65)
    print("AUTOENCODER COMPLET")
    print("=" * 65)
    autoencoder.summary()

    # Factorul de compresie: cat de mult am redus informatia.
    pixeli = 28 * 28
    print("\n" + "-" * 65)
    print(f"Dimensiune intrare      : {pixeli} valori (28 x 28 x 1)")
    print(f"Dimensiune spatiu latent: {LATENT_DIM} valori")
    print(f"Factor de compresie     : {pixeli / LATENT_DIM:.1f}x")
    print("-" * 65)