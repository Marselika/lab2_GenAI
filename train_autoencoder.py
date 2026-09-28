"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapa 4: antrenarea Autoencoder-ului clasic.
"""

import os
import json
import time

import keras

from data_utils import incarca_si_preproceseaza, RESULTS_DIR
from autoencoder import construieste_autoencoder, LATENT_DIM

# Hiperparametrii ceruti in enunt.
EPOCHS = 10
BATCH_SIZE = 128


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Datele ----------------------------------------------------------
    x_train, x_test, _ = incarca_si_preproceseaza(verbose=False)
    print(f"Date incarcate: train={x_train.shape}, test={x_test.shape}")

    # --- 2. Modelul ---------------------------------------------------------
    autoencoder, encoder, decoder = construieste_autoencoder(LATENT_DIM)

    # --- 3. Compilarea ------------------------------------------------------
    # optimizer="adam" : rata de invatare adaptiva, alegerea standard;
    #                    nu necesita reglaj manual pentru acest laborator.
    # loss="binary_crossentropy" : compara pixel cu pixel imaginea de intrare
    #                    cu cea reconstruita. Ambele sunt in [0, 1], deci
    #                    fiecare pixel poate fi tratat ca o probabilitate.
    # metrics=["mse"]  : metrica suplimentara, doar pentru monitorizare;
    #                    NU influenteaza antrenarea. O raportam la Etapa 5.
    autoencoder.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["mse"],
    )

    # --- 4. Antrenarea ------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"ANTRENARE AUTOENCODER  |  epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    print("=" * 65)

    start = time.time()

    history = autoencoder.fit(
        x_train, x_train,             # intrare = tinta -> auto-supervizare
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,                 # amestecam ordinea la fiecare epoca
        validation_data=(x_test, x_test),
        verbose=2,                    # o linie per epoca (log curat)
    )

    durata = time.time() - start

    print("=" * 65)
    print(f"Timp total de antrenare : {durata:.1f} s "
          f"({durata / 60:.2f} min)")
    print(f"Timp mediu per epoca    : {durata / EPOCHS:.1f} s")
    print("=" * 65)

    # --- 5. Salvarea modelelor ---------------------------------------------
    # Salvam si subcomponentele: encoderul ne trebuie la Etapa 10
    # (spatiul latent), decoderul la Etapa 11 (generarea de imagini).
    autoencoder.save(os.path.join(RESULTS_DIR, "autoencoder.keras"))
    encoder.save(os.path.join(RESULTS_DIR, "ae_encoder.keras"))
    decoder.save(os.path.join(RESULTS_DIR, "ae_decoder.keras"))
    print(f"[salvat] {RESULTS_DIR}/autoencoder.keras")
    print(f"[salvat] {RESULTS_DIR}/ae_encoder.keras")
    print(f"[salvat] {RESULTS_DIR}/ae_decoder.keras")

    # --- 6. Salvarea istoricului -------------------------------------------
    # history.history este un dict cu listele de valori per epoca.
    # Il salvam in JSON ca sa putem reface graficele fara sa reantrenam.
    istoric = {
        "history": {k: [float(v) for v in vals]
                    for k, vals in history.history.items()},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "training_time_sec": durata,
        "keras_version": keras.__version__,
    }
    cale_istoric = os.path.join(RESULTS_DIR, "autoencoder_history.json")
    with open(cale_istoric, "w") as f:
        json.dump(istoric, f, indent=2)
    print(f"[salvat] {cale_istoric}")

    # --- 7. Rezumat pentru raport ------------------------------------------
    h = history.history
    print("\n" + "-" * 65)
    print("REZUMAT (valori de trecut in raport)")
    print("-" * 65)
    print(f"Loss final (train)      : {h['loss'][-1]:.4f}")
    print(f"Loss final (validation) : {h['val_loss'][-1]:.4f}")
    print(f"MSE final (train)       : {h['mse'][-1]:.6f}")
    print(f"MSE final (validation)  : {h['val_mse'][-1]:.6f}")
    print(f"Timp de antrenare       : {durata:.1f} s")
    print("-" * 65)


if __name__ == "__main__":
    main()