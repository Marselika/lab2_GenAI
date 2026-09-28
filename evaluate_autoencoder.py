"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 5: evaluarea Autoencoder-ului clasic.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR

# Culori alese astfel incat cele doua curbe sa ramana distincte si pentru
# cititorii cu deficiente de perceptie a culorii, si la tiparire alb-negru
# (difera si ca luminozitate, nu doar ca nuanta).
CULOARE_TRAIN = "#1f6feb"
CULOARE_VAL = "#e8590c"


def grafic_pierdere(istoric, nume_fisier="autoencoder_loss.png"):
    """
    Deseneaza doua panouri:
      stanga  - binary_crossentropy (functia efectiv minimizata)
      dreapta - MSE (metrica interpretabila a calitatii reconstructiei)
    """
    h = istoric["history"]
    epoci = np.arange(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    panouri = [
        (axes[0], "loss", "val_loss",
         "Binary Crossentropy (functia de pierdere)"),
        (axes[1], "mse", "val_mse",
         "MSE (metrica de calitate)"),
    ]

    for ax, cheie_train, cheie_val, titlu in panouri:
        ax.plot(epoci, h[cheie_train], color=CULOARE_TRAIN, linewidth=2,
                marker="o", markersize=4, label="Antrenare")
        ax.plot(epoci, h[cheie_val], color=CULOARE_VAL, linewidth=2,
                marker="s", markersize=4, label="Validare")

        # Eticheta directa pe ultimul punct: cititorul vede valoarea finala
        # fara sa citeasca de pe axa.
        ax.annotate(f"{h[cheie_train][-1]:.4f}",
                    xy=(epoci[-1], h[cheie_train][-1]),
                    xytext=(-6, 10), textcoords="offset points",
                    color=CULOARE_TRAIN, fontsize=9, ha="right")
        ax.annotate(f"{h[cheie_val][-1]:.4f}",
                    xy=(epoci[-1], h[cheie_val][-1]),
                    xytext=(-6, -14), textcoords="offset points",
                    color=CULOARE_VAL, fontsize=9, ha="right")

        ax.set_title(titlu, fontsize=11)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Valoare")
        ax.set_xticks(epoci)
        # Grila discreta: ajuta citirea fara sa concureze cu datele.
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)
        for margine in ("top", "right"):
            ax.spines[margine].set_visible(False)

    fig.suptitle("Autoencoder clasic - evolutia antrenarii "
                 f"({istoric['epochs']} epoci, latent_dim={istoric['latent_dim']})",
                 fontsize=12)
    fig.tight_layout()

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def grafic_reconstructie(originale, reconstruite, mse_per_imagine,
                         n=10, nume_fisier="autoencoder_reconstruction.png"):
    """
    Doua randuri: sus imaginile originale, jos reconstructiile.
    Intre randuri afisam MSE-ul individual al fiecarei reconstructii.
    """
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.4))

    for i in range(n):
        axes[0, i].imshow(originale[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstruite[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"{mse_per_imagine[i]:.4f}", fontsize=7, pad=2)

    # Etichetele de rand, puse in marginea figurii
    # (axis("off") ascunde ylabel-ul obisnuit).
    fig.text(0.005, 0.72, "Original", rotation=90, va="center", fontsize=10)
    fig.text(0.005, 0.28, "Reconstruit", rotation=90, va="center", fontsize=10)

    fig.suptitle("Autoencoder clasic - original vs. reconstructie "
                 "(numarul dintre randuri = MSE individual)", fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])

    cale = os.path.join(RESULTS_DIR, nume_fisier)
    fig.savefig(cale, dpi=150, bbox_inches="tight")
    print(f"[salvat] {cale}")
    plt.show()
    plt.close(fig)


def main():
    # --- 1. Datele si modelul antrenat --------------------------------------
    _, x_test, _ = incarca_si_preproceseaza(verbose=False)

    cale_model = os.path.join(RESULTS_DIR, "autoencoder.keras")
    autoencoder = keras.models.load_model(cale_model)
    print(f"[incarcat] {cale_model}")

    with open(os.path.join(RESULTS_DIR, "autoencoder_history.json")) as f:
        istoric = json.load(f)
    print("[incarcat] autoencoder_history.json")

    # --- 2. Graficele de pierdere -------------------------------------------
    grafic_pierdere(istoric)

    # --- 3. Reconstructia setului de test -----------------------------------
    reconstruite = autoencoder.predict(x_test, batch_size=256, verbose=0)

    # --- 4. Metrici numerice -------------------------------------------------
    # MSE global: media patratelor diferentelor peste TOTI pixelii din test.
    mse_global = float(np.mean((x_test - reconstruite) ** 2))

    # MSE per imagine: media pe cei 784 de pixeli ai fiecarei imagini.
    # axis=(1,2,3) colapseaza inaltimea, latimea si canalul.
    mse_per_imagine = np.mean((x_test - reconstruite) ** 2, axis=(1, 2, 3))

    # MAE - eroarea absoluta medie; mai putin sensibila la valori extreme.
    mae_global = float(np.mean(np.abs(x_test - reconstruite)))

    # PSNR - raportul semnal/zgomot de varf, in decibeli. Pentru imagini in
    # [0, 1], valoarea maxima a semnalului este 1, deci PSNR = 10*log10(1/MSE).
    # Valori mai mari inseamna reconstructie mai buna.
    psnr = float(10 * np.log10(1.0 / mse_global))

    print("\n" + "=" * 65)
    print("METRICI PE SETUL DE TEST (10.000 imagini)")
    print("=" * 65)
    print(f"MSE global              : {mse_global:.6f}")
    print(f"MAE global              : {mae_global:.6f}")
    print(f"PSNR                    : {psnr:.2f} dB")
    print(f"MSE per imagine - medie : {mse_per_imagine.mean():.6f}")
    print(f"MSE per imagine - std   : {mse_per_imagine.std():.6f}")
    print(f"MSE per imagine - min   : {mse_per_imagine.min():.6f} "
          f"(imaginea #{int(mse_per_imagine.argmin())})")
    print(f"MSE per imagine - max   : {mse_per_imagine.max():.6f} "
          f"(imaginea #{int(mse_per_imagine.argmax())})")
    print(f"Timp de antrenare       : {istoric['training_time_sec']:.1f} s")
    print("=" * 65)

    # --- 5. Comparatia vizuala ----------------------------------------------
    grafic_reconstructie(x_test, reconstruite, mse_per_imagine, n=10)

    # --- 6. Salvarea metricilor pentru Etapa 12 -----------------------------
    metrici = {
        "model": "autoencoder",
        "latent_dim": istoric["latent_dim"],
        "epochs": istoric["epochs"],
        "batch_size": istoric["batch_size"],
        "training_time_sec": istoric["training_time_sec"],
        "final_loss": istoric["history"]["loss"][-1],
        "final_val_loss": istoric["history"]["val_loss"][-1],
        "test_mse": mse_global,
        "test_mae": mae_global,
        "test_psnr_db": psnr,
        "test_mse_std": float(mse_per_imagine.std()),
    }
    cale_metrici = os.path.join(RESULTS_DIR, "autoencoder_metrics.json")
    with open(cale_metrici, "w") as f:
        json.dump(metrici, f, indent=2)
    print(f"[salvat] {cale_metrici}")


if __name__ == "__main__":
    main()