| Caracteristica | Autoencoder | VAE |
|---|---|---|
| Encoder | Conv -> Flatten -> Dense(16) | Conv -> Flatten -> Dense(16) x2 |
| Decoder | Dense -> Conv/UpSampling -> sigmoid | Dense -> Conv/UpSampling -> sigmoid |
| Spatiu latent | 16D determinist, nestructurat | 16D probabilistic, ~N(0, I) |
| Parametri totali | 178.001 | 228.193 |
| Reconstruction loss (train) | 0.0831 (BCE mediata) | 75.72 (BCE insumata) |
| KL Divergence | absenta | 23.78 |
| Sampling | nu | da (reparameterization trick) |
| MSE pe test (determinist) | 0.007365 | 0.008481 |
| MSE pe test (esantionat) | n/a | 0.011460 |
| MAE pe test | 0.027101 | 0.031171 |
| PSNR pe test (dB) | 21.33 | 20.72 |
| Training time (s) | 257.8 | 256.6 |
| Generare din N(0, I) | esec - imagini nerecognoscibile | reusita - cifre plauzibile |
| Intensitate imagini generate | 0.0491 | 0.1287 |
| Pixel maxim mediu (generat) | 0.5707 | 0.9600 |
| Complexitate implementare | compile(loss=...) standard | clasa Model + train_step personalizat |
