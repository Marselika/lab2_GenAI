"""
Laborator 2 - Autoencoder si VAE pe MNIST
Etapele 6-7: arhitectura Variational Autoencoder-ului si functia sa de loss.
"""

import keras
from keras import layers, ops
import tensorflow as tf

# Aceeasi dimensiune latenta ca la Autoencoder-ul clasic, ca sa fie
# comparabile corect la Etapa 12.
LATENT_DIM = 16


# ===========================================================================
# ETAPA 6a - Stratul de sampling (reparameterization trick)
# ===========================================================================
@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """
    Extrage un esantion z din distributia N(z_mean, sigma^2).

    Nu esantionam direct - aceasta operatie nu este derivabila si ar
    intrerupe propagarea gradientului. Folosim reparameterization trick:

        z = z_mean + sigma * epsilon,   epsilon ~ N(0, I)

    Aleatoriul este mutat in epsilon, care este o INTRARE independenta de
    parametrii retelei. Astfel z ramane o functie neteda de z_mean si
    z_log_var, iar gradientul poate curge inapoi spre encoder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generator de numere aleatoare propriu stratului - asigura
        # reproductibilitate si compatibilitate cu grafurile Keras 3.
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]

        # epsilon ~ N(0, I), generat proaspat la fiecare trecere.
        epsilon = keras.random.normal(shape=(batch, dim),
                                      seed=self.seed_generator)

        # Reteaua prezice log(sigma^2), nu sigma. Deci:
        #   sigma = exp(0.5 * log(sigma^2))
        # Motivul: log(sigma^2) poate lua orice valoare reala (inclusiv
        # negativa), pe cand sigma trebuie sa fie strict pozitiv. Predictia
        # in spatiul logaritmic elimina nevoia unei constrangeri.
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# ===========================================================================
# ETAPA 6b - Encoderul probabilistic
# ===========================================================================
def construieste_encoder_vae(latent_dim=LATENT_DIM):
    """
    Spre deosebire de encoderul clasic, care produce UN punct, acesta
    produce PARAMETRII UNEI DISTRIBUTII: media si log-varianta.
    """
    intrare = keras.Input(shape=(28, 28, 1), name="imagine_intrare")

    # Partea convolutionala este identica cu cea a Autoencoder-ului clasic,
    # tocmai pentru ca diferenta studiata sa fie doar natura spatiului latent.
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(intrare)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Flatten()(x)

    # Doua capete paralele, ambele pornind din aceleasi trasaturi:
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Esantionul propriu-zis.
    z = Sampling(name="z")([z_mean, z_log_var])

    # Returnam toate trei: z_mean si z_log_var sunt necesare pentru KL,
    # z pentru decoder. z_mean va fi folosit si la Etapa 10 (vizualizare).
    return keras.Model(intrare, [z_mean, z_log_var, z], name="encoder_vae")


# ===========================================================================
# ETAPA 6c - Decoderul
# ===========================================================================
def construieste_decoder_vae(latent_dim=LATENT_DIM):
    """
    Identic ca arhitectura cu decoderul Autoencoder-ului clasic.
    Diferenta nu este in decoder, ci in ce primeste la intrare.
    """
    intrare_latenta = keras.Input(shape=(latent_dim,), name="vector_latent")

    x = layers.Dense(7 * 7 * 64, activation="relu")(intrare_latenta)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    iesire = layers.Conv2D(1, 3, activation="sigmoid", padding="same",
                           name="imagine_reconstruita")(x)

    return keras.Model(intrare_latenta, iesire, name="decoder_vae")


# ===========================================================================
# ETAPA 7 - Modelul VAE cu functie de loss personalizata
# ===========================================================================
class VAE(keras.Model):
    """
    Model personalizat, necesar pentru ca loss-ul VAE NU poate fi exprimat
    ca o functie loss(y_true, y_pred) pasata lui compile().

    Motivul: componenta KL depinde de z_mean si z_log_var - tensori INTERNI
    ai encoderului, care nu apar nici in y_true, nici in y_pred. Keras nu
    are cum sa ii transmita unei functii de loss standard.

    Solutia este suprascrierea lui train_step(): avem acces la toti tensorii
    intermediari si construim loss-ul complet manual.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Trackere pentru cele trei componente ale loss-ului. Fara ele,
        # Keras nu ar sti ce sa afiseze in bara de progres si nu ar reseta
        # mediile la inceputul fiecarei epoci.
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # MSE pentru comparatia directa cu Autoencoder-ul clasic.
        self.mse_tracker = keras.metrics.Mean(name="mse")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.mse_tracker,
        ]

    def call(self, inputs):
        """Trecerea inainte - folosita la predict() si la reconstructii."""
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def calculeaza_loss(self, data):
        """
        Componentele loss-ului. Extrase intr-o metoda separata pentru a fi
        folosite identic la antrenare (train_step) si la validare (test_step).
        """
        z_mean, z_log_var, z = self.encoder(data)
        reconstructie = self.decoder(z)

        # --- Reconstruction loss ---------------------------------------
        # binary_crossentropy aplicata pe (28,28,1) reduce ultima axa si
        # returneaza (batch, 28, 28). Insumam peste pixeli (axele 1 si 2)
        # si abia apoi facem media pe batch.
        #
        # ATENTIE: insumarea, nu media, peste pixeli este esentiala.
        # Loss-ul VAE este suma a doi termeni care trebuie sa fie pe
        # aceeasi scara. KL se insumeaza peste cele 16 dimensiuni latente;
        # daca reconstructia ar fi mediata peste cei 784 de pixeli, ar fi
        # de ~784 de ori mai mica, iar KL ar domina complet -> modelul ar
        # colapsa si ar produce doar imagini neclare, identice.
        reconstruction_loss = ops.mean(
            ops.sum(
                keras.losses.binary_crossentropy(data, reconstructie),
                axis=(1, 2),
            )
        )

        # --- KL divergence ---------------------------------------------
        # Forma inchisa a divergentei KL intre N(mu, sigma^2) si N(0, I):
        #   KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Insumata peste dimensiunile latente, mediata peste batch.
        kl_loss = ops.mean(
            ops.sum(
                -0.5 * (1 + z_log_var
                        - ops.square(z_mean)
                        - ops.exp(z_log_var)),
                axis=1,
            )
        )

        total_loss = reconstruction_loss + kl_loss
        mse = ops.mean(ops.square(data - reconstructie))

        return total_loss, reconstruction_loss, kl_loss, mse

    def train_step(self, data):
        # Keras poate livra data ca tuplu (x, y); noi avem doar x.
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            total_loss, rec_loss, kl_loss, mse = self.calculeaza_loss(data)

        # Gradientii se calculeaza pentru TOTI parametrii (encoder + decoder)
        # simultan - este un singur model antrenat end-to-end.
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_tracker.update_state(mse)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Necesar pentru validation_data. Fara el, Keras ar incerca sa
        foloseasca loss-ul standard din compile(), care aici nu exista.
        Metricile rezultate apar automat prefixate cu 'val_'.
        """
        if isinstance(data, tuple):
            data = data[0]

        total_loss, rec_loss, kl_loss, mse = self.calculeaza_loss(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.mse_tracker.update_state(mse)

        return {m.name: m.result() for m in self.metrics}


def construieste_vae(latent_dim=LATENT_DIM):
    """Returneaza (vae, encoder, decoder)."""
    encoder = construieste_encoder_vae(latent_dim)
    decoder = construieste_decoder_vae(latent_dim)
    vae = VAE(encoder, decoder, name="vae")
    return vae, encoder, decoder


if __name__ == "__main__":
    vae, encoder, decoder = construieste_vae()

    print("\n" + "=" * 65)
    print("ENCODER VAE")
    print("=" * 65)
    encoder.summary()

    print("\n" + "=" * 65)
    print("DECODER VAE")
    print("=" * 65)
    decoder.summary()

    total = encoder.count_params() + decoder.count_params()
    print("\n" + "-" * 65)
    print(f"Parametri encoder : {encoder.count_params():,}")
    print(f"Parametri decoder : {decoder.count_params():,}")
    print(f"TOTAL VAE         : {total:,}")
    print(f"Spatiu latent     : {LATENT_DIM} dimensiuni "
          f"(z_mean + z_log_var = 2 x {LATENT_DIM} valori prezise)")
    print("-" * 65)