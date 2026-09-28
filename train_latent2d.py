"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 10 (pregatire): antreneaza variante cu latent_dim = 2, exclusiv
pentru vizualizarea directa a spatiului latent in plan.

Modelele principale raman cele cu latent_dim = 16 (Etapele 3-9).
Acestea sunt un experiment SEPARAT, auxiliar.
"""

import os
import json
import time

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from autoencoder import construieste_autoencoder
from vae import construieste_vae

EPOCHS = 10
BATCH_SIZE = 128
LATENT_2D = 2


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_train, x_test, _ = incarca_si_preproceseaza(verbose=False)

    timpi = {}

    # ---------------------------------------------------------------- AE 2D
    print("=" * 70)
    print("ANTRENARE AUTOENCODER cu latent_dim = 2")
    print("=" * 70)

    ae2, ae2_encoder, ae2_decoder = construieste_autoencoder(LATENT_2D)
    ae2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["mse"])

    start = time.time()
    ae2.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
            shuffle=True, validation_data=(x_test, x_test), verbose=2)
    timpi["ae2d_sec"] = time.time() - start

    ae2_encoder.save(os.path.join(RESULTS_DIR, "ae2d_encoder.keras"))
    ae2_decoder.save(os.path.join(RESULTS_DIR, "ae2d_decoder.keras"))
    print(f"[salvat] ae2d_encoder.keras, ae2d_decoder.keras "
          f"({timpi['ae2d_sec']:.1f} s)")

    # --------------------------------------------------------------- VAE 2D
    print("\n" + "=" * 70)
    print("ANTRENARE VAE cu latent_dim = 2")
    print("=" * 70)

    vae2, vae2_encoder, vae2_decoder = construieste_vae(LATENT_2D)
    vae2.compile(optimizer="adam")

    start = time.time()
    vae2.fit(x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
             validation_data=(x_test,), verbose=2)
    timpi["vae2d_sec"] = time.time() - start

    vae2_encoder.save(os.path.join(RESULTS_DIR, "vae2d_encoder.keras"))
    vae2_decoder.save(os.path.join(RESULTS_DIR, "vae2d_decoder.keras"))
    print(f"[salvat] vae2d_encoder.keras, vae2d_decoder.keras "
          f"({timpi['vae2d_sec']:.1f} s)")

    with open(os.path.join(RESULTS_DIR, "latent2d_times.json"), "w") as f:
        json.dump(timpi, f, indent=2)

    print("\n" + "-" * 70)
    print(f"Timp AE  (latent_dim=2) : {timpi['ae2d_sec']:.1f} s")
    print(f"Timp VAE (latent_dim=2) : {timpi['vae2d_sec']:.1f} s")
    print("-" * 70)


if __name__ == "__main__":
    main()