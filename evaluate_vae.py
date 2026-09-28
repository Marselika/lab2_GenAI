"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 9: evaluarea reconstructiei VAE + comparatie directa cu Autoencoder-ul.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR

# IMPORT OBLIGATORIU, chiar daca 'Sampling' nu apare explicit mai jos.
# Decoratorul @keras.saving.register_keras_serializable() se executa abia
# la importarea modulului 'vae'. Fara aceasta linie, load_model() nu stie
# ce este clasa 'Sampling' si esueaza cu:
#     TypeError: Could not locate class 'Sampling'.
from vae import Sampling  # noqa: F401


def metrici(originale, reconstruite):
    """Calculeaza setul standard de metrici pentru o reconstructie."""
    mse = float(np.mean((originale - reconstruite) ** 2))
    mae = float(np.mean(np.abs(originale - reconstruite)))
    psnr = float(10 * np.log10(1.0 / mse))
    mse_per_img = np.mean((originale - reconstruite) ** 2, axis=(1, 2, 3))
    return mse, mae, psnr, mse_per_img


def grafic_comparativ(originale, rec_ae, rec_vae, n=10,
                      nume_fisier="vae_reconstruction.png"):
    """
    Trei randuri, aceleasi imagini:
        1. original
        2. reconstructia Autoencoder-ului clasic
        3. reconstructia VAE
    Comparatia pe aceleasi exemple este singura corecta vizual.
    """
    randuri = [
        (originale, "Original"),
        (rec_ae, "Autoencoder"),
        (rec_vae, "VAE"),
    ]

    fig, axes = plt.subplots(3, n, figsize=(n * 1.3, 4.6))

    for r, (imagini, eticheta) in enumerate(randuri):
        for i in range(n):
            axes[r, i].imshow(imagini[i].squeeze(), cmap="gray",
                              vmin=0, vmax=1)
            axes[r, i].axis("off")

    # Etichetele de rand, in marginea figurii.
    for y, eticheta in zip([0.80, 0.52, 0.22],
                           ["Original", "Autoencoder", "VAE"]):
        fig.text(0.005, y, eticheta, rotation=90, va="center", fontsize=10)

    fig.suptitle("Reconstructie pe aceleasi imagini din setul de test: "
                 "original vs. Autoencoder clasic vs. VAE", fontsize=11)
    fig.tight_layout(rect=[0.025, 0, 1, 0.94])

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def main():
    _, x_test, _ = incarca_si_preproceseaza(verbose=False)

    # --- 1. Modelele antrenate ----------------------------------------------
    # Sampling se deserializeaza automat datorita decoratorului
    # @keras.saving.register_keras_serializable() din vae.py.
    vae_encoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_encoder.keras"))
    vae_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_decoder.keras"))
    autoencoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "autoencoder.keras"))
    print("[incarcat] vae_encoder.keras, vae_decoder.keras, autoencoder.keras")

    # --- 2. Reconstructiile VAE ---------------------------------------------
    z_mean, z_log_var, z = vae_encoder.predict(x_test, batch_size=256,
                                               verbose=0)

    # (a) Reconstructie DETERMINISTA: folosim direct z_mean, fara sampling.
    #     Este varianta standard de raportare - elimina zgomotul si masoara
    #     capacitatea reala de codare a modelului.
    rec_vae_det = vae_decoder.predict(z_mean, batch_size=256, verbose=0)

    # (b) Reconstructie STOCASTICA: folosim z esantionat, adica exact
    #     regimul in care modelul a fost antrenat. Va fi putin mai slaba.
    rec_vae_sto = vae_decoder.predict(z, batch_size=256, verbose=0)

    # --- 3. Reconstructia Autoencoder-ului (pentru comparatie) --------------
    rec_ae = autoencoder.predict(x_test, batch_size=256, verbose=0)

    # --- 4. Metricile --------------------------------------------------------
    mse_ae, mae_ae, psnr_ae, per_img_ae = metrici(x_test, rec_ae)
    mse_det, mae_det, psnr_det, per_img_det = metrici(x_test, rec_vae_det)
    mse_sto, mae_sto, psnr_sto, _ = metrici(x_test, rec_vae_sto)

    print("\n" + "=" * 72)
    print("RECONSTRUCTIE PE SETUL DE TEST (10.000 imagini)")
    print("=" * 72)
    print(f"{'Model':<34} {'MSE':>10} {'MAE':>10} {'PSNR (dB)':>11}")
    print("-" * 72)
    print(f"{'Autoencoder clasic':<34} {mse_ae:10.6f} {mae_ae:10.6f} "
          f"{psnr_ae:11.2f}")
    print(f"{'VAE (z = z_mean, determinist)':<34} {mse_det:10.6f} "
          f"{mae_det:10.6f} {psnr_det:11.2f}")
    print(f"{'VAE (z esantionat, stocastic)':<34} {mse_sto:10.6f} "
          f"{mae_sto:10.6f} {psnr_sto:11.2f}")
    print("=" * 72)
    print(f"Degradare VAE fata de AE (MSE determinist) : "
          f"{(mse_det / mse_ae - 1) * 100:+.1f}%")
    print(f"Cost al esantionarii (stocastic vs determinist) : "
          f"{(mse_sto / mse_det - 1) * 100:+.1f}%")
    print("=" * 72)

    # --- 5. Statistica spatiului latent -------------------------------------
    # Verificare importanta: cat de aproape este distributia latenta de N(0, I)?
    sigma = np.exp(0.5 * z_log_var)
    print("\n" + "-" * 72)
    print("STATISTICA SPATIULUI LATENT VAE (tinta: medie 0, sigma 1)")
    print("-" * 72)
    print(f"z_mean - media globala      : {z_mean.mean():+.4f}")
    print(f"z_mean - deviatia standard  : {z_mean.std():.4f}")
    print(f"sigma  - media globala      : {sigma.mean():.4f}")
    print(f"Dimensiuni active (std z_mean > 0.1) : "
          f"{int((z_mean.std(axis=0) > 0.1).sum())} din {z_mean.shape[1]}")
    print("-" * 72)

    # --- 6. Graficul comparativ ---------------------------------------------
    grafic_comparativ(x_test, rec_ae, rec_vae_det, n=10)

    # --- 7. Salvarea metricilor pentru Etapa 12 -----------------------------
    with open(os.path.join(RESULTS_DIR, "vae_history.json")) as f:
        istoric = json.load(f)

    rezultate = {
        "model": "vae",
        "latent_dim": istoric["latent_dim"],
        "epochs": istoric["epochs"],
        "batch_size": istoric["batch_size"],
        "training_time_sec": istoric["training_time_sec"],
        "final_loss": istoric["history"]["loss"][-1],
        "final_val_loss": istoric["history"]["val_loss"][-1],
        "final_reconstruction_loss": istoric["history"]["reconstruction_loss"][-1],
        "final_kl_loss": istoric["history"]["kl_loss"][-1],
        "test_mse": mse_det,
        "test_mse_sampled": mse_sto,
        "test_mae": mae_det,
        "test_psnr_db": psnr_det,
        "test_mse_std": float(per_img_det.std()),
        "latent_mean": float(z_mean.mean()),
        "latent_std": float(z_mean.std()),
        "latent_sigma_mean": float(sigma.mean()),
        "dimensiuni_active": int((z_mean.std(axis=0) > 0.1).sum()),
    }
    cale = os.path.join(RESULTS_DIR, "vae_metrics.json")
    with open(cale, "w") as f:
        json.dump(rezultate, f, indent=2)
    print(f"\n[salvat] {cale}")


if __name__ == "__main__":
    main()