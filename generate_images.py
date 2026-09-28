"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 11: generarea de imagini noi prin esantionare din spatiul latent.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from vae import Sampling  # noqa: F401  - inregistreaza stratul personalizat

LATENT_DIM = 16
N_AFISATE = 10        # imagini desenate in figura
N_STATISTICA = 1000   # imagini folosite pentru metricile numerice
SEED = 42


def statistici_imagini(nume, imagini):
    """
    Cat de mult seamana imaginile generate, ca STATISTICA, cu MNIST real?

    MNIST este aproape binar: fundal negru (0) si trasatura alba (1), cu
    foarte putini pixeli intermediari. O generare reusita trebuie sa
    reproduca aceste proprietati:
      - intensitate medie apropiata de a datelor reale;
      - pixel maxim aproape de 1 (trasaturi clare, contrastate);
      - o fractiune similara de pixeli 'aprinsi'.
    O imagine spalacita, fara alb curat, esueaza la toate trei.
    """
    intensitate = imagini.mean()
    max_mediu = imagini.max(axis=(1, 2, 3)).mean()
    aprinsi = np.mean(imagini > 0.5)
    print(f"{nume:<22} intensitate={intensitate:.4f}  "
          f"max_mediu={max_mediu:.4f}  frac_pixeli>0.5={aprinsi:.4f}")
    return intensitate, max_mediu, aprinsi


def grafic_generare(img_vae, img_ae, nume_fisier="generated_images.png"):
    """
    Doua randuri, ACELEASI vectori latenti z, decodati de modele diferite.
    Este singura comparatie corecta: orice diferenta vine din model,
    nu din intrari diferite.
    """
    n = len(img_vae)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.6))

    for i in range(n):
        axes[0, i].imshow(img_vae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(img_ae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    fig.text(0.005, 0.70, "VAE", rotation=90, va="center", fontsize=10)
    fig.text(0.005, 0.25, "Autoencoder", rotation=90, va="center", fontsize=10)

    fig.suptitle("Imagini generate din ACELEASI vectori aleatori z ~ N(0, I)",
                 fontsize=11)
    fig.tight_layout(rect=[0.025, 0, 1, 0.93])

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def grafic_manifold(decoder_2d, n=15, nume_fisier="latent_manifold.png"):
    """
    Harta spatiului latent 2D al VAE: parcurgem o grila de puncte si
    decodam fiecare punct. Rezultatul arata cum se transforma continuu
    o cifra in alta - dovada directa a continuitatii spatiului latent.
    """
    # Grila NU este uniforma, ci urmeaza cuantilele distributiei normale.
    # Motivul: punctele sunt distribuite dupa N(0, I), deci o grila uniforma
    # ar supraesantiona marginile (unde nu exista date) si ar subesantiona
    # centrul (unde se afla majoritatea imaginilor). ppf = inversa functiei
    # de repartitie -> puncte echiprobabile.
    grila = norm.ppf(np.linspace(0.02, 0.98, n))

    # Construim toate punctele deodata si facem o singura predictie.
    puncte = np.array([[x, y] for y in grila[::-1] for x in grila],
                      dtype="float32")
    imagini = decoder_2d.predict(puncte, batch_size=256, verbose=0)

    figura = np.zeros((28 * n, 28 * n))
    for k in range(n * n):
        i, j = divmod(k, n)
        figura[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            imagini[k].squeeze()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(figura, cmap="gray", vmin=0, vmax=1)

    # Etichetam axele cu valorile latente reale, nu cu indici de pixeli.
    pozitii = np.arange(n) * 28 + 14
    ax.set_xticks(pozitii[::2])
    ax.set_xticklabels([f"{v:.1f}" for v in grila[::2]], fontsize=8)
    ax.set_yticks(pozitii[::2])
    ax.set_yticklabels([f"{v:.1f}" for v in grila[::-1][::2]], fontsize=8)
    ax.set_xlabel("Dimensiunea latenta 1")
    ax.set_ylabel("Dimensiunea latenta 2")
    ax.set_title(f"Harta spatiului latent al VAE (latent_dim=2), grila {n}x{n}\n"
                 "fiecare imagine este decodata dintr-un punct al planului latent",
                 fontsize=11)

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def grafic_interpolare(ae_encoder, ae_decoder, vae_encoder, vae_decoder,
                       x_test, idx_a=0, idx_b=1, pasi=10,
                       nume_fisier="interpolation.png"):
    """
    Interpolare liniara intre reprezentarile latente a doua imagini reale.
    La VAE tranzitia trebuie sa fie continua, trecand prin forme valide;
    la AE pot aparea imagini hibride sau degradate pe traseu.
    """
    a = x_test[idx_a:idx_a + 1]
    b = x_test[idx_b:idx_b + 1]

    # Coeficientii de amestec: 0 = imaginea A, 1 = imaginea B.
    alfa = np.linspace(0, 1, pasi).reshape(-1, 1)

    z_ae = ae_encoder.predict(np.concatenate([a, b]), verbose=0)
    drum_ae = (1 - alfa) * z_ae[0] + alfa * z_ae[1]
    img_ae = ae_decoder.predict(drum_ae, verbose=0)

    # [0] = z_mean; interpolam centrele distributiilor, nu esantioane.
    z_vae = vae_encoder.predict(np.concatenate([a, b]), verbose=0)[0]
    drum_vae = (1 - alfa) * z_vae[0] + alfa * z_vae[1]
    img_vae = vae_decoder.predict(drum_vae, verbose=0)

    fig, axes = plt.subplots(2, pasi, figsize=(pasi * 1.3, 3.4))
    for i in range(pasi):
        axes[0, i].imshow(img_vae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(img_ae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    fig.text(0.005, 0.70, "VAE", rotation=90, va="center", fontsize=10)
    fig.text(0.005, 0.25, "Autoencoder", rotation=90, va="center", fontsize=10)
    fig.suptitle("Interpolare liniara in spatiul latent, intre doua imagini "
                 "reale din setul de test", fontsize=11)
    fig.tight_layout(rect=[0.025, 0, 1, 0.93])

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def main():
    _, x_test, _ = incarca_si_preproceseaza(verbose=False)

    vae_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_encoder.keras"))
    vae_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_decoder.keras"))
    ae_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "ae_encoder.keras"))
    ae_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "ae_decoder.keras"))
    vae2_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae2d_decoder.keras"))
    print("[incarcat] toate modelele")

    # --- 1. Esantionare din N(0, I) -----------------------------------------
    # Testul central al capacitatii generative: extragem vectori complet
    # aleatori, fara nicio legatura cu vreo imagine reala.
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(N_STATISTICA, LATENT_DIM)).astype("float32")
    print(f"\nVectori latenti generati: {z.shape}, "
          f"medie={z.mean():+.3f}, std={z.std():.3f}")

    # ACELEASI z, trecuti prin decodere diferite.
    img_vae = vae_decoder.predict(z, batch_size=256, verbose=0)
    img_ae = ae_decoder.predict(z, batch_size=256, verbose=0)

    # --- 2. Metrici numerice -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"STATISTICA IMAGINILOR GENERATE ({N_STATISTICA} esantioane)")
    print("=" * 78)
    statistici_imagini("MNIST real (referinta)", x_test)
    statistici_imagini("VAE generat", img_vae)
    statistici_imagini("Autoencoder generat", img_ae)
    print("=" * 78)

    # --- 3. Graficele --------------------------------------------------------
    grafic_generare(img_vae[:N_AFISATE], img_ae[:N_AFISATE])
    grafic_manifold(vae2_decoder, n=15)
    grafic_interpolare(ae_encoder, ae_decoder, vae_encoder, vae_decoder,
                       x_test, idx_a=0, idx_b=1, pasi=10)


if __name__ == "__main__":
    main()