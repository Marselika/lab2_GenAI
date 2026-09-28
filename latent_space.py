"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 10: compararea spatiului latent al Autoencoder-ului cu cel al VAE.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from vae import Sampling  # noqa: F401  - inregistreaza stratul personalizat


def statistici_latente(nume, puncte):
    """Cat de 'regulat' este norul de puncte? Tinta VAE: medie 0, std 1."""
    print(f"{nume:<28} "
          f"medie={puncte.mean():+7.3f}  "
          f"std={puncte.std():7.3f}  "
          f"min={puncte.min():+8.2f}  "
          f"max={puncte.max():+8.2f}")


def deseneaza_panou(ax, puncte, etichete, titlu):
    """Un panou de tip scatter, colorat dupa cifra reala."""
    # tab10 = 10 culori discrete, cate una per cifra. Culoarea nu codifica
    # o marime, ci o identitate -> paleta calitativa, nu gradient.
    sc = ax.scatter(puncte[:, 0], puncte[:, 1], c=etichete,
                    cmap="tab10", s=2, alpha=0.6, vmin=-0.5, vmax=9.5)
    ax.set_title(titlu, fontsize=11)
    ax.set_xlabel("Dimensiunea latenta 1")
    ax.set_ylabel("Dimensiunea latenta 2")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    for margine in ("top", "right"):
        ax.spines[margine].set_visible(False)
    return sc


def main():
    _, x_test, y_test = incarca_si_preproceseaza(verbose=False)

    # --- 1. Spatiile latente de 16 dimensiuni (modelele principale) --------
    ae_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "ae_encoder.keras"))
    vae_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_encoder.keras"))

    lat_ae16 = ae_encoder.predict(x_test, batch_size=256, verbose=0)
    lat_vae16 = vae_encoder.predict(x_test, batch_size=256, verbose=0)[0]  # z_mean

    # 16 dimensiuni nu pot fi desenate direct -> proiectie PCA in plan.
    # PCA alege cele doua directii care pastreaza cea mai mare varianta.
    pca_ae = PCA(n_components=2).fit(lat_ae16)
    pca_vae = PCA(n_components=2).fit(lat_vae16)
    proj_ae16 = pca_ae.transform(lat_ae16)
    proj_vae16 = pca_vae.transform(lat_vae16)

    # --- 2. Spatiile latente de 2 dimensiuni (experimentul auxiliar) -------
    ae2_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "ae2d_encoder.keras"))
    vae2_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae2d_encoder.keras"))

    lat_ae2 = ae2_encoder.predict(x_test, batch_size=256, verbose=0)
    lat_vae2 = vae2_encoder.predict(x_test, batch_size=256, verbose=0)[0]

    # --- 3. Statistici numerice ---------------------------------------------
    print("=" * 78)
    print("STATISTICA SPATIILOR LATENTE")
    print("=" * 78)
    statistici_latente("AE  (16D, brut)", lat_ae16)
    statistici_latente("VAE (16D, z_mean)", lat_vae16)
    statistici_latente("AE  (2D, brut)", lat_ae2)
    statistici_latente("VAE (2D, z_mean)", lat_vae2)
    print("-" * 78)
    print(f"Varianta explicata de primele 2 componente PCA:")
    print(f"  AE  (16D) : {pca_ae.explained_variance_ratio_.sum() * 100:.1f}%")
    print(f"  VAE (16D) : {pca_vae.explained_variance_ratio_.sum() * 100:.1f}%")
    print("=" * 78)

    # --- 4. Figura cu 4 panouri ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    deseneaza_panou(axes[0, 0], proj_ae16, y_test,
                    "Autoencoder clasic - spatiu 16D, proiectie PCA")
    deseneaza_panou(axes[0, 1], proj_vae16, y_test,
                    "VAE - spatiu 16D (z_mean), proiectie PCA")
    deseneaza_panou(axes[1, 0], lat_ae2, y_test,
                    "Autoencoder clasic - spatiu 2D (direct)")
    sc = deseneaza_panou(axes[1, 1], lat_vae2, y_test,
                         "VAE - spatiu 2D (z_mean, direct)")

    # Pe panourile VAE marcam cercul de raza 2 sigma al distributiei N(0, I):
    # zona din care vom esantiona la Etapa 11.
    for ax in (axes[0, 1], axes[1, 1]):
        unghi = np.linspace(0, 2 * np.pi, 200)
        ax.plot(2 * np.cos(unghi), 2 * np.sin(unghi),
                color="#222222", linewidth=1.2, linestyle="--",
                label="raza 2σ a lui N(0, I)")
        ax.legend(frameon=False, loc="upper right", fontsize=9)

    # O singura legenda de culoare pentru toata figura.
    cbar = fig.colorbar(sc, ax=axes, ticks=range(10), fraction=0.03, pad=0.02)
    cbar.set_label("Cifra reala (folosita DOAR pentru colorare, "
                   "nu la antrenare)")

    fig.suptitle("Compararea spatiilor latente: Autoencoder clasic vs. VAE "
                 "(10.000 imagini din setul de test)", fontsize=13)

    cale = os.path.join(RESULTS_DIR, "latent_space.png")
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()