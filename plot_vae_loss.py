"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 8 (partea de grafice): evolutia componentelor loss-ului VAE.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR

CULOARE_TRAIN = "#1f6feb"
CULOARE_VAL = "#e8590c"


def grafic_loss_vae(istoric, nume_fisier="vae_loss.png"):
    """
    Patru panouri, pentru ca loss-ul VAE are componente pe scari diferite:
      1. Total loss          = reconstruction + KL
      2. Reconstruction loss (componenta dominanta, ~75)
      3. KL divergence       (componenta de regularizare, ~24)
      4. MSE                 (metrica comparabila direct cu Autoencoder-ul)

    Nu le punem pe acelasi grafic: scarile difera de ~3x, iar un grafic cu
    doua axe verticale ar face pantele necomparabile si ar induce in eroare.
    """
    h = istoric["history"]
    epoci = np.arange(1, len(h["loss"]) + 1)

    panouri = [
        ("loss", "val_loss", "Total loss (reconstruction + KL)"),
        ("reconstruction_loss", "val_reconstruction_loss",
         "Reconstruction loss (binary crossentropy, insumata)"),
        ("kl_loss", "val_kl_loss", "KL divergence"),
        ("mse", "val_mse", "MSE (comparabil cu Autoencoder-ul clasic)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (cheie_train, cheie_val, titlu) in zip(axes.ravel(), panouri):
        ax.plot(epoci, h[cheie_train], color=CULOARE_TRAIN, linewidth=2,
                marker="o", markersize=4, label="Antrenare")
        ax.plot(epoci, h[cheie_val], color=CULOARE_VAL, linewidth=2,
                marker="s", markersize=4, label="Validare")

        # Etichete directe pe ultimul punct.
        zecimale = 5 if cheie_train == "mse" else 2
        ax.annotate(f"{h[cheie_train][-1]:.{zecimale}f}",
                    xy=(epoci[-1], h[cheie_train][-1]),
                    xytext=(-6, 10), textcoords="offset points",
                    color=CULOARE_TRAIN, fontsize=9, ha="right")
        ax.annotate(f"{h[cheie_val][-1]:.{zecimale}f}",
                    xy=(epoci[-1], h[cheie_val][-1]),
                    xytext=(-6, -14), textcoords="offset points",
                    color=CULOARE_VAL, fontsize=9, ha="right")

        ax.set_title(titlu, fontsize=11)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Valoare")
        ax.set_xticks(epoci)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)
        for margine in ("top", "right"):
            ax.spines[margine].set_visible(False)

    fig.suptitle("VAE - evolutia componentelor loss-ului "
                 f"({istoric['epochs']} epoci, latent_dim={istoric['latent_dim']})",
                 fontsize=12)
    fig.tight_layout()

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def main():
    with open(os.path.join(RESULTS_DIR, "vae_history.json")) as f:
        istoric = json.load(f)
    print("[incarcat] vae_history.json")

    grafic_loss_vae(istoric)

    # Tabelul pe epoci - util pentru raport.
    h = istoric["history"]
    print("\n" + "-" * 78)
    print(f"{'Ep':>2} {'loss':>8} {'rec':>8} {'kl':>7} {'mse':>8} | "
          f"{'val_loss':>9} {'val_rec':>8} {'val_kl':>7} {'val_mse':>8}")
    print("-" * 78)
    for i in range(len(h["loss"])):
        print(f"{i + 1:>2} {h['loss'][i]:8.2f} "
              f"{h['reconstruction_loss'][i]:8.2f} {h['kl_loss'][i]:7.2f} "
              f"{h['mse'][i]:8.5f} | {h['val_loss'][i]:9.2f} "
              f"{h['val_reconstruction_loss'][i]:8.2f} "
              f"{h['val_kl_loss'][i]:7.2f} {h['val_mse'][i]:8.5f}")
    print("-" * 78)


if __name__ == "__main__":
    main()