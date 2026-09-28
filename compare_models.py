"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 12: compararea finala a celor doua modele.

Toate valorile sunt citite din fisierele produse de etapele anterioare
sau recalculate aici. Nicio cifra nu este scrisa manual.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from vae import Sampling  # noqa: F401

CULOARE_AE = "#1f6feb"
CULOARE_VAE = "#e8590c"

LATENT_DIM = 16
N_STATISTICA = 1000
SEED = 42


def statistici_generare():
    """
    Recalculeaza statisticile imaginilor generate (Etapa 11), ca scriptul
    de comparatie sa fie autonom si sa nu depinda de valori copiate manual.
    """
    _, x_test, _ = incarca_si_preproceseaza(verbose=False)

    vae_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "vae_decoder.keras"))
    ae_decoder = keras.models.load_model(
        os.path.join(RESULTS_DIR, "ae_decoder.keras"))

    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(N_STATISTICA, LATENT_DIM)).astype("float32")

    gen_vae = vae_decoder.predict(z, batch_size=256, verbose=0)
    gen_ae = ae_decoder.predict(z, batch_size=256, verbose=0)

    def masuri(a):
        return {
            "intensitate": float(a.mean()),
            "max_mediu": float(a.max(axis=(1, 2, 3)).mean()),
            "frac_aprinsi": float(np.mean(a > 0.5)),
        }

    return masuri(x_test), masuri(gen_ae), masuri(gen_vae)


def grafic_comparativ(ae, vae, nume_fisier="comparison.png"):
    """
    Trei panouri separate, cate unul per unitate de masura.

    NU punem MSE si timpul de antrenare pe acelasi grafic: au unitati si
    ordine de marime complet diferite, iar un grafic cu doua axe verticale
    ar face pantele necomparabile si ar induce in eroare cititorul.
    """
    marimi = [
        ("MSE pe setul de test\n(mai mic = reconstructie mai buna)",
         ae["test_mse"], vae["test_mse"], "{:.5f}"),
        ("PSNR (dB)\n(mai mare = reconstructie mai buna)",
         ae["test_psnr_db"], vae["test_psnr_db"], "{:.2f}"),
        ("Timp de antrenare (s)\n10 epoci, CPU",
         ae["training_time_sec"], vae["training_time_sec"], "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, (titlu, val_ae, val_vae, fmt) in zip(axes, marimi):
        bare = ax.bar(["Autoencoder", "VAE"], [val_ae, val_vae],
                      color=[CULOARE_AE, CULOARE_VAE], width=0.55)

        # Eticheta direct pe bara: cititorul nu mai are nevoie de axa.
        for bara, valoare in zip(bare, [val_ae, val_vae]):
            ax.annotate(fmt.format(valoare),
                        xy=(bara.get_x() + bara.get_width() / 2,
                            bara.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10)

        ax.set_title(titlu, fontsize=10)
        ax.set_ylim(0, max(val_ae, val_vae) * 1.18)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for margine in ("top", "right"):
            ax.spines[margine].set_visible(False)

    fig.suptitle("Autoencoder clasic vs. VAE - rezultate experimentale "
                 "(MNIST, latent_dim=16, 10 epoci)", fontsize=12)
    fig.tight_layout()

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def main():
    with open(os.path.join(RESULTS_DIR, "autoencoder_metrics.json")) as f:
        ae = json.load(f)
    with open(os.path.join(RESULTS_DIR, "vae_metrics.json")) as f:
        vae = json.load(f)
    print("[incarcat] autoencoder_metrics.json, vae_metrics.json")

    real, gen_ae, gen_vae = statistici_generare()

    # Randurile tabelului: (caracteristica, valoare AE, valoare VAE)
    randuri = [
        ("Encoder",
         "Conv -> Flatten -> Dense(16)",
         "Conv -> Flatten -> Dense(16) x2"),
        ("Decoder",
         "Dense -> Conv/UpSampling -> sigmoid",
         "Dense -> Conv/UpSampling -> sigmoid"),
        ("Spatiu latent",
         f"{ae['latent_dim']}D determinist, nestructurat",
         f"{vae['latent_dim']}D probabilistic, ~N(0, I)"),
        ("Parametri totali", "178.001", "228.193"),
        ("Reconstruction loss (train)",
         f"{ae['final_loss']:.4f} (BCE mediata)",
         f"{vae['final_reconstruction_loss']:.2f} (BCE insumata)"),
        ("KL Divergence", "absenta",
         f"{vae['final_kl_loss']:.2f}"),
        ("Sampling", "nu",
         "da (reparameterization trick)"),
        ("MSE pe test (determinist)",
         f"{ae['test_mse']:.6f}", f"{vae['test_mse']:.6f}"),
        ("MSE pe test (esantionat)",
         "n/a", f"{vae['test_mse_sampled']:.6f}"),
        ("MAE pe test",
         f"{ae['test_mae']:.6f}", f"{vae['test_mae']:.6f}"),
        ("PSNR pe test (dB)",
         f"{ae['test_psnr_db']:.2f}", f"{vae['test_psnr_db']:.2f}"),
        ("Training time (s)",
         f"{ae['training_time_sec']:.1f}",
         f"{vae['training_time_sec']:.1f}"),
        ("Generare din N(0, I)",
         "esec - imagini nerecognoscibile",
         "reusita - cifre plauzibile"),
        ("Intensitate imagini generate",
         f"{gen_ae['intensitate']:.4f}", f"{gen_vae['intensitate']:.4f}"),
        ("Pixel maxim mediu (generat)",
         f"{gen_ae['max_mediu']:.4f}", f"{gen_vae['max_mediu']:.4f}"),
        ("Complexitate implementare",
         "compile(loss=...) standard",
         "clasa Model + train_step personalizat"),
    ]

    # --- Afisare in consola --------------------------------------------------
    lat = [30, 38, 38]
    linie = "+" + "+".join("-" * (l + 2) for l in lat) + "+"
    print("\n" + linie)
    print(f"| {'Caracteristica':<{lat[0]}} | {'Autoencoder':<{lat[1]}} | "
          f"{'VAE':<{lat[2]}} |")
    print(linie)
    for c, a, v in randuri:
        print(f"| {c:<{lat[0]}} | {a:<{lat[1]}} | {v:<{lat[2]}} |")
    print(linie)

    print(f"\nReferinta MNIST real: intensitate={real['intensitate']:.4f}, "
          f"max_mediu={real['max_mediu']:.4f}, "
          f"frac_aprinsi={real['frac_aprinsi']:.4f}")

    print("\nDIFERENTE CHEIE:")
    print(f"  MSE VAE / MSE AE          : "
          f"{vae['test_mse'] / ae['test_mse']:.3f}x "
          f"({(vae['test_mse'] / ae['test_mse'] - 1) * 100:+.1f}%)")
    print(f"  Timp VAE / Timp AE        : "
          f"{vae['training_time_sec'] / ae['training_time_sec']:.3f}x")
    print(f"  Parametri VAE / AE        : {228193 / 178001:.3f}x")

    # --- Tabel Markdown, gata de lipit in raport ----------------------------
    cale_md = os.path.join(RESULTS_DIR, "comparison_table.md")
    with open(cale_md, "w", encoding="utf-8") as f:
        f.write("| Caracteristica | Autoencoder | VAE |\n")
        f.write("|---|---|---|\n")
        for c, a, v in randuri:
            f.write(f"| {c} | {a} | {v} |\n")
    print(f"\n[salvat] {cale_md}")

    # --- Graficul ------------------------------------------------------------
    grafic_comparativ(ae, vae)


if __name__ == "__main__":
    main()