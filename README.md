# Laborator 2 — Autoencoder clasic și Variational Autoencoder pe MNIST

Implementare, antrenare și comparare a două arhitecturi encoder–decoder pe MNIST,
cu TensorFlow 2.20 / Keras 3.15 (CPU).

---

## 1. Configurația modelelor

Ambele modele au **partea convoluțională identică**, pentru ca singura variabilă
studiată să fie natura spațiului latent.

| | Autoencoder | VAE |
|---|---|---|
| Intrare | 28×28×1 (784 valori) | 28×28×1 |
| Encoder | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten | identic |
| Bottleneck | `Dense(16)` liniar | `Dense(16)` ×2: `z_mean`, `z_log_var` + strat `Sampling` |
| Decoder | Dense(3136) → Reshape(7,7,64) → Conv2D(64) → UpSampling → Conv2D(32) → UpSampling → Conv2D(1, sigmoid) | identic |
| Ieșire | 28×28×1 în [0,1] | 28×28×1 în [0,1] |
| Parametri | 178.001 | 228.193 |
| Factor de compresie | 784 → 16 = **49×** | 49× |

Cei 50.192 de parametri în plus la VAE sunt exclusiv capul `z_log_var`.

---

## 2. Justificarea hiperparametrilor

| Alegere | Valoare | Justificare |
|---|---|---|
| `latent_dim` | 16 | Compresie reală 49×. Valoare **identică la ambele modele** — altfel comparația e invalidă. |
| Bottleneck `Dense`, nu doar pooling | — | Fără el, „latentul" ar fi 7×7×64 = 3136 > 784, adică *expansiune*, nu compresie. |
| Activare bottleneck | liniară | `relu` ar anula valorile negative → unități moarte. Simetric cu `z_mean`, care e liniar. |
| Activare finală | `sigmoid` | Datele sunt normalizate în [0,1]; ieșirea trebuie pe aceeași scară. |
| Optimizator | Adam (lr implicit 0,001) | Rată adaptivă, fără reglaj manual; standard pentru acest tip de model. |
| Epoci | 10 | Cerință de laborator. Curbele erau încă în scădere — modelele nu au convergit complet. |
| Batch size | 128 | 469 pași/epocă; compromis stabilitate gradient ↔ viteză pe CPU. |
| `padding="same"` | — | Păstrează dimensiunea spațială; reducerea se face controlat, prin pooling. |

Toți hiperparametrii sunt **identici între AE și VAE**. Orice diferență ar fi
confundat efectul arhitecturii cu efectul reglajului.

---

## 3. Funcțiile de pierdere

### Autoencoder
L = BinaryCrossentropy(x, x̂) # mediată pe pixeli
Validă pentru că ambele imagini sunt în [0,1]; fiecare pixel e tratat ca
probabilitate. **BCE nu tinde spre 0** nici la reconstrucție perfectă — minimul ei
este entropia țintei. De aceea se stabilizează la ~0,083. Pentru calitate se
folosește MSE, care are 0 ca minim real.

### VAE
L = Reconstruction Loss + KL Divergence

Reconstruction = Σ_pixeli BCE(x, x̂) ≈ 75,7
KL = -0,5 · Σ_latent (1 + log σ² − μ² − σ²) ≈ 23,8

Cele două componente trag în direcții opuse:
- doar reconstrucție → AE clasic, spațiu latent fragmentat, fără generare;
- doar KL → *posterior collapse*: toate imaginile colapsează într-o pată identică;
- suma → spațiu simultan informativ și regulat.

**Detaliu critic de implementare:** reconstrucția se **însumează** pe pixeli, nu se
mediază. KL se însumează pe cele 16 dimensiuni latente; dacă reconstrucția ar fi
mediată pe 784 de pixeli, ar fi de ~784× mai mică, KL ar domina, iar modelul ar
colapsa.

**De ce `train_step` personalizat:** KL depinde de `z_mean` și `z_log_var`, tensori
*interni* ai encoderului, care nu apar nici în `y_true`, nici în `y_pred`. O funcție
`loss(y_true, y_pred)` pasată lui `compile()` nu îi poate accesa. (`add_loss` pe
tensori intermediari, din Keras 2, nu mai funcționează în Keras 3.)

---

## 4. Monitorizarea antrenării

Mărimi urmărite per epocă, salvate în `*_history.json`:

| Model | Metrici |
|---|---|
| AE | `loss`, `val_loss`, `mse`, `val_mse` |
| VAE | `loss`, `reconstruction_loss`, `kl_loss`, `mse` + variantele `val_` |

**Ce s-a observat:**

- **Fără supraînvățare** la niciun model: `val_loss ≈ loss` (diferență < 0,001).
  Bottleneck-ul de 16 valori acționează ca regularizator natural.
- **KL crește, apoi se stabilizează** (16,6 → 23,8). Comportament calitativ diferit
  de toate celelalte curbe și cea mai informativă observație a antrenării: inițial
  encoderul e aleator și deja aproape de `N(0,I)`; pe măsură ce învață să
  reconstruiască, trebuie să *separe* imaginile în spațiul latent, ceea ce crește KL.
  Platoul marchează echilibrul dintre cele două forțe.
- **Absența colapsului**: KL s-a stabilizat la ~23,8, nu a căzut spre 0 — confirmă
  că balansarea `sum`/`mean` este corectă.

⚠️ **Loss-urile brute nu sunt comparabile între modele** (0,083 mediat vs 99,5 însumat
+ KL). Singura metrică direct comparabilă este **MSE**.

---

## 5. Evaluarea calității generării

Test: se extrag vectori `z ~ N(0, I)` și se trec prin **același set de z** prin
ambele decodere.

MNIST este aproape binar (fundal 0, trăsătură 1). O generare reușită trebuie să
reproducă această statistică:

| Metrică | MNIST real | VAE | Autoencoder |
|---|---:|---:|---:|
| Intensitate medie | 0,1325 | **0,1287** (−3%) | 0,0491 (−63%) |
| Pixel maxim mediu | 0,9996 | **0,9600** | 0,5707 |
| Fracție pixeli > 0,5 | 0,1342 | **0,1277** | 0,0142 |

VAE-ul este statistic aproape indistinct de datele reale; AE-ul produce imagini
spălăcite, fără alb curat, nerecognoscibile ca cifre.

**Metrică respinsă:** distanța euclidiană până la cea mai apropiată cifră reală
indica AE-ul drept „mai bun" (3,71 vs 4,53) — pentru că o imagine aproape neagră
este aproape în L2 de o cifră subțire. Distanța în spațiul pixelilor recompensează
imaginile goale și nu este utilizabilă ca măsură de plauzibilitate.

Verificări suplimentare: harta spațiului latent 2D (`latent_manifold.png`) arată
tranziții continue între cifre; interpolarea liniară (`interpolation.png`) este
netedă la VAE și produce suprapuneri fantomatice la AE.

---

## 6. Interpretarea rezultatelor

### Reconstrucție

| Metrică | AE | VAE (determinist) | VAE (eșantionat) |
|---|---:|---:|---:|
| MSE | 0,007365 | 0,008481 | 0,011460 |
| MAE | 0,027101 | 0,031171 | 0,036412 |
| PSNR (dB) | 21,33 | 20,72 | 19,41 |
| Timp antrenare (s) | 257,8 | 256,6 | — |

AE-ul reconstruiește cu **~15% mai bine** (MSE). Raportarea se face pe varianta
**deterministă** (`z = z_mean`); varianta eșantionată include zgomotul din antrenare
și costă suplimentar ~35%.

### Spațiul latent

| | AE | VAE |
|---|---|---|
| Deviație standard | 2,82 | 0,956 |
| Interval | −11,58 … +13,58 | −4,39 … +4,03 |
| Varianță în primele 2 componente PCA | 31,1% | 19,8% |
| Dimensiuni active | — | 16 / 16 |

- **Scara**: AE se întinde pe ~25 de unități, VAE pe ~8, cu `std ≈ 0,96` — practic
  ținta teoretică de 1. Un `z ~ N(0,I)` cade în zona populată la VAE și în vid la AE.
- **PCA, aparent paradoxal**: 19,8% pare mai slab decât 31,1%, dar înseamnă inversul.
  KL împinge varianța să fie distribuită *uniform* pe toate cele 16 dimensiuni;
  izotropia perfectă ar da 2/16 = 12,5%. AE-ul concentrează varianța în câteva
  direcții și subutilizează restul.
- `σ` mediu = 0,255, nu 1: fiecare imagine are distribuție îngustă, dar **amestecul**
  tuturor aproximează `N(0, I)`. KL penalizează abaterea agregată, nu impune
  varianță unitară pe fiecare exemplu.

### Concluzia

VAE-ul pierde ~15% la MSE și 0,6 dB la PSNR, cu **+28% parametri** și **timp de
antrenare practic identic**. În schimb câștigă o capacitate pe care AE-ul nu o are
deloc: generarea de imagini noi.

Diferența nu este de grad, ci de natură. Pe reconstrucție modelele sunt comparabile;
pe generare unul reușește și celălalt eșuează. Costul real al VAE-ului nu este
computațional, ci de **complexitate a implementării**: `train_step` personalizat,
balansarea corectă a componentelor loss-ului, serializarea stratului `Sampling`.

---

## 7. Structura proiectului

```text
Laborator2_Autoencoder_MNIST/
├── check_env.py            # Etapa 1
├── data_utils.py           # Etapa 2
├── autoencoder.py          # Etapa 3
├── train_autoencoder.py    # Etapa 4
├── evaluate_autoencoder.py # Etapa 5
├── vae.py                  # Etapele 6-7
├── train_vae.py            # Etapa 8
├── plot_vae_loss.py        # Etapa 8 (grafice)
├── evaluate_vae.py         # Etapa 9
├── train_latent2d.py       # Etapa 10 (pregătire)
├── latent_space.py         # Etapa 10
├── generate_images.py      # Etapa 11
├── compare_models.py       # Etapa 12
├── requirements.txt
└── results/                # modele, istorice, metrici, figuri
```

### Descrierea fișierelor

| Fișier | Ce face |
|---|---|
| `check_env.py` | Verifică Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn; listează dispozitivele; rulează un model minimal și descarcă MNIST în cache. Confirmă că mediul e funcțional, nu doar importabil. |
| `data_utils.py` | Încarcă MNIST, convertește la `float32`, normalizează în [0,1], adaugă dimensiunea de canal → `(N,28,28,1)`. Returnează și `y_test`, folosit **exclusiv** la colorarea spațiului latent. Salvează `mnist_samples.png`. |
| `autoencoder.py` | Construiește encoderul, decoderul și Autoencoder-ul clasic (funcții parametrizate după `latent_dim`). Rulat direct, afișează cele trei `summary()` și factorul de compresie. |
| `train_autoencoder.py` | Compilează (`adam`, `binary_crossentropy`, metrică `mse`) și antrenează 10 epoci cu `x_train` ca intrare **și** țintă. Salvează modelul, encoderul, decoderul și `autoencoder_history.json` cu timpul de antrenare. |
| `evaluate_autoencoder.py` | Încarcă modelul salvat (nu reantrenează). Desenează curbele loss/MSE, reconstruiește `x_test`, calculează MSE / MAE / PSNR și statistica per imagine. Salvează două figuri și `autoencoder_metrics.json`. |
| `vae.py` | Stratul `Sampling` (reparameterization trick), encoderul probabilistic cu capete `z_mean` / `z_log_var`, decoderul și clasa `VAE(keras.Model)` cu `train_step` / `test_step` personalizate care implementează `reconstruction + KL`. |
| `train_vae.py` | Compilează fără argument `loss` și antrenează VAE-ul cu aceiași hiperparametri ca AE-ul. Salvează encoderul și decoderul **separat** (modelul subclasat nu se serializează direct) plus `vae_history.json`. |
| `plot_vae_loss.py` | Desenează cele patru componente ale loss-ului (total, reconstrucție, KL, MSE) în panouri separate — au scări diferite, deci nu pot împărți o axă. Afișează și tabelul pe epoci. |
| `evaluate_vae.py` | Reconstruiește `x_test` în ambele regimuri (`z_mean` determinist și `z` eșantionat), calculează metricile, compară direct cu AE-ul pe aceleași imagini și raportează statistica spațiului latent. Necesită `from vae import Sampling` pentru deserializare. |
| `train_latent2d.py` | Experiment auxiliar: antrenează variante AE și VAE cu `latent_dim = 2`, exclusiv pentru vizualizarea directă în plan. Nu înlocuiește modelele principale de 16D. |
| `latent_space.py` | Figură cu 4 panouri: spațiile latente 16D proiectate PCA și spațiile 2D directe, colorate după cifră. Afișează statistica fiecărui spațiu și varianța explicată de PCA. |
| `generate_images.py` | Eșantionează `z ~ N(0,I)` și decodează cu **ambele** decodere; calculează statisticile de plauzibilitate; generează harta spațiului latent 2D (grilă pe cuantile normale) și interpolarea liniară între două imagini reale. |
| `compare_models.py` | Asamblează comparația finală citind fișierele `*_metrics.json` și recalculând statisticile de generare — nicio valoare scrisă manual. Produce tabelul în consolă, `comparison_table.md` și `comparison.png`. |

### Fișiere generate în `results/`

| Fișier | Conținut |
|---|---|
| `*.keras` | Modele antrenate (AE, VAE, variantele 2D, encodere și decodere separate) |
| `*_history.json` | Metrici per epocă + timpul de antrenare |
| `*_metrics.json` | Metrici finale pe setul de test |
| `mnist_samples.png` | Eșantioane MNIST după preprocesare |
| `autoencoder_loss.png`, `vae_loss.png` | Curbele de antrenare |
| `autoencoder_reconstruction.png`, `vae_reconstruction.png` | Original vs. reconstrucție |
| `latent_space.png` | Comparația spațiilor latente |
| `generated_images.png`, `latent_manifold.png`, `interpolation.png` | Capacitatea generativă |
| `comparison.png`, `comparison_table.md` | Comparația finală |

### Ordinea de rulare

```bash
python check_env.py
python data_utils.py
python autoencoder.py
python train_autoencoder.py      # ~4-6 min
python evaluate_autoencoder.py
python vae.py
python train_vae.py              # ~4-7 min
python plot_vae_loss.py
python evaluate_vae.py
python train_latent2d.py         # ~5-10 min
python latent_space.py
python generate_images.py
python compare_models.py
```
