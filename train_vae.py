"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 8: antrenarea Variational Autoencoder-ului.
"""

import os
import json
import time

import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from vae import construieste_vae, LATENT_DIM

# Aceiasi hiperparametri ca la Autoencoder-ul clasic.
# Orice diferenta aici ar invalida comparatia de la Etapa 12.
EPOCHS = 10
BATCH_SIZE = 128


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Datele ----------------------------------------------------------
    x_train, x_test, _ = incarca_si_preproceseaza(verbose=False)
    print(f"Date incarcate: train={x_train.shape}, test={x_test.shape}")

    # --- 2. Modelul ---------------------------------------------------------
    vae, encoder, decoder = construieste_vae(LATENT_DIM)

    # --- 3. Compilarea ------------------------------------------------------
    # Observa: NU pasam niciun argument 'loss'. Functia de pierdere este
    # definita in interiorul lui train_step(), pentru ca depinde de tensori
    # interni (z_mean, z_log_var) inaccesibili unui loss standard.
    vae.compile(optimizer="adam")

    # --- 4. Antrenarea ------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"ANTRENARE VAE  |  epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
          f"latent_dim={LATENT_DIM}")
    print("=" * 70)

    start = time.time()

    # Observa: pasam DOAR x_train, fara tinta. Tinta este implicita in
    # train_step, unde comparam reconstructia cu intrarea.
    history = vae.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test,),
        verbose=2,
    )

    durata = time.time() - start

    print("=" * 70)
    print(f"Timp total de antrenare : {durata:.1f} s ({durata / 60:.2f} min)")
    print(f"Timp mediu per epoca    : {durata / EPOCHS:.1f} s")
    print("=" * 70)

    # --- 5. Salvarea modelelor ---------------------------------------------
    # Salvam encoderul si decoderul SEPARAT, nu modelul VAE complet.
    # Motivul: VAE este un model subclasat cu train_step personalizat;
    # salvarea lui integrala ar necesita get_config/from_config si ar
    # complica inutil laboratorul. Encoderul si decoderul sunt modele
    # functionale obisnuite si se salveaza fara probleme - iar ele sunt
    # tot ce ne trebuie la Etapele 9, 10 si 11.
    encoder.save(os.path.join(RESULTS_DIR, "vae_encoder.keras"))
    decoder.save(os.path.join(RESULTS_DIR, "vae_decoder.keras"))
    print(f"[salvat] {RESULTS_DIR}/vae_encoder.keras")
    print(f"[salvat] {RESULTS_DIR}/vae_decoder.keras")

    # --- 6. Salvarea istoricului -------------------------------------------
    istoric = {
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "training_time_sec": durata,
        "keras_version": keras.__version__,
    }
    cale_istoric = os.path.join(RESULTS_DIR, "vae_history.json")
    with open(cale_istoric, "w") as f:
        json.dump(istoric, f, indent=2)
    print(f"[salvat] {cale_istoric}")

    # --- 7. Rezumat pentru raport ------------------------------------------
    h = history.history
    print("\n" + "-" * 70)
    print("REZUMAT (valori de trecut in raport)")
    print("-" * 70)
    print(f"Total loss final (train)      : {h['loss'][-1]:.4f}")
    print(f"Total loss final (validation) : {h['val_loss'][-1]:.4f}")
    print(f"Reconstruction loss (train)   : {h['reconstruction_loss'][-1]:.4f}")
    print(f"KL loss (train)               : {h['kl_loss'][-1]:.4f}")
    print(f"MSE final (train)             : {h['mse'][-1]:.6f}")
    print(f"MSE final (validation)        : {h['val_mse'][-1]:.6f}")
    print(f"Timp de antrenare             : {durata:.1f} s")
    print("-" * 70)


if __name__ == "__main__":
    main()