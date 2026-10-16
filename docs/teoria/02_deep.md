# Detección de anomalías con Deep Learning: autoencoders, denoising, modelos generativos y RBIG/flows

> **Estado:** versión revisada y definitiva del borrador `02_deep.md`. Se ha comprobado contra las 17 slides (imágenes), la transcripción y los notebooks con sus figuras. Las aportaciones propias van marcadas como **(nota del redactor)**.
>
> **Fuentes**
> - Slides `slides_deep/p01.png`–`p17.png` (AD_algoritmos_deep.pdf; en la portada pone "Valero Laparra"). Revisadas las 17 como imagen.
> - `transcripcion.txt`, líneas 2219–2534 ([03:50:01]–[04:22:16]): parte deep [03:50:01]–[04:14:01], guía de scripts [04:13:16]–[04:21:42] y cierre hasta [04:22:16]. Cruce con la transcripción completa en [02:21:15] y [02:55:47]–[02:55:55].
> - `slides_clasicas/p32.png`: "Teoría de la información – Miramos… *Information Theory Measures via Multidimensional Gaussianization*" (Laparra, Johnson, Camps-Valls, **Santos-Rodríguez**, Malo), arXiv 2010.03807.
> - Notebooks deep (`2026_Anomalias/scripts_deep/`): `AD_AE_simple(_clase)`, `AD_AE_convolucional(_clase)` y `UNET_convolutional_denoising`.
> - Notebooks RBIG: `AD_rbig_demo(_clase)`, `rbig_demo_val`, `RBIG_extra/information_theory_with_RBIG`, `RBIG_extra/rbig_demo_val_clase_CIFAR_cond` y `RBIG_extra/rbig_demo_val_clase_penguins`.
>
> **Quién habla:** la clase la da **Raúl Santos-Rodríguez** (Bristol), pero las slides y los notebooks son de **Valero Laparra**, autor de RBIG. Por eso el profesor habla de Valero en tercera persona: "Esto es valero, que es método favorito" [03:59:35], "a través de valero me mandáis un correo" [04:22:10]. Santos-Rodríguez también firma el artículo de RBIG sobre teoría de la información (slide clásica 32). La "imagen de Valero" que se menciona en la guía es la foto `https://www.uv.es/lapeva/Valero_2020_DQ.png` que usan los notebooks clásicos.
>
> **Errores fonéticos de la transcripción, ya interpretados:**
> - "otros códex", "otro en códem", "Codex", "noctur collers" → autoencoder(s)
> - "Guns", "huracán", "Khan", "lacan", "las llamas", "estas ganas" → GAN(s)
> - "Pdx", "T. N. X." → p(x)
> - "Arbic", "R pic" → RBIG; "marero" → Valero
> - "Outligers", "Applyars", "Outlife", "Yo, Layers" → outlier(s)
> - "dinoising", "de novici", "inoixing" → denoising
> - "Gausseno", "coleccionano", "coxiona", "auxiano" → gaussiano
> - "News" → NeurIPS
> - "queclopelitis", "ciclopelitis", "5 gritis", "quecolitis" → *capabilities*
> - "Pooly Layers" → pooling layers; "C, I, P" → seguramente MLP
> - "normales y Plus" → normalizing flows
> - "Edénse" → dudoso, quizá "en series" o texto

---

## 0. Mapa del tema

**Idea.** La slide 2 ("INTRO") organiza el bloque en tres familias:
1. **Predicción**: modelos supervisados, $A=f(x)$.
2. **Autoencoders**: no supervisados; se usa el error de reconstrucción o el espacio latente.
3. **Modelos generativos**, **explícitos** e **implícitos**. La slide 9 los detalla:
   - Explícitos: Flows · Autorregresivos (predictivos) · VAEs.
   - Implícitos: GANs · Diffusion.

**Mensaje global.**
- Al empezar: "Espero que no sea una sorpresa … vuelvan a salir [los autoencoders] aquí en medio. Al parecer valen para todos" [03:50:36]–[03:50:44].
- Resumen final: "Lo más importante es recordar que puedes utilizar una red neuronal igual que antes … para explicar anomalía contra no anomalía … No cambia mucho la cosa. Y en la parte generativa acordaos que cualquier cosa que estime la densidad de probabilidad o algo parecido, pues valdrá para seguir los mismos métodos que hacíamos en probabilidades … esto sí se utiliza mucho en la práctica cuando tengáis capacidad computacional y datos; son las mejores soluciones" [04:13:28]–[04:14:01].

*Fuente: slide 1 (portada "DETECCIÓN ANOMALIAS CON DEEP LEARNING – Valero Laparra", con una imagen decorativa de un chip en una sala de servidores), slides 2 y 9 · transcripción [03:50:36]–[03:50:48] y [04:13:23]–[04:14:01]*

---

## 1. Modelos predictivos supervisados: $A=f(x)$

**Idea.** "cualquier modelo de clasificación o de regresión nos valía para [detectar] anomalías si lo utilizamos bien, sobre todo en ese caso, que tengo muchas anomalías y puedo tener la clase anomalía y la clase normal" [03:50:55]–[03:51:00]. En ese caso todo es supervisado: se entrena "la criatura que más os guste" y se asume que "si he sido capaz de caracterizar las anomalías bien, seré capaz de predecirlas bien" [03:51:15]–[03:51:19].

**Detalle.**
- El score de anomalía es la salida directa de una red entrenada: $A=f(x)$. "hay una función que ha entrenado aquí en medio … me va a dar un score" [03:51:27].
- La frase sobre el signo del score queda cortada: "si es una anomalía, es bajo. Y si es lo correcto, es…" [03:51:33]. El signo es una convención: lo que importa es que la salida separe las dos clases.
- "con la arquitectura que más guste" [03:51:38].
- Ejemplo de la slide 3 ("Modelos predictivos – Supervisado – A = f(x)"): una red encoder–decoder de **segmentación** de un TAC craneal. Entra un corte en escala de grises y sale un mapa con fondo negro, una región rojo oscuro grande y una estructura verde pequeña. Leyenda: lila = *Convolution + Batch Normalization + ReLU*; amarillo = *Pooling*; rojo = *Deconvolution*; azul = *1×1 convolution*. Es una predicción supervisada píxel a píxel.
- En el cierre lo repite: "puedes utilizar una red neuronal igual que antes … Puedes utilizar una red para eso. No cambia mucho la cosa" [04:13:28]–[04:13:45].

*Fuente: slide 3 · transcripción [03:50:55]–[03:51:38] y [04:13:28]–[04:13:45]*

---

## 2. Autoencoders: definición, restricciones sobre $Z$ y score $|f(x)-x|$

**Idea.** "Entrada igual la salida, comprimimos en el medio" [03:51:55]. Es un método **no supervisado**. Si al código se le impone una distribución gaussiana se obtiene el VAE "que es lo que hablábamos ayer" [03:52:00].

**Fórmulas de la slide 4.**
- $Z=\mathrm{enc}(X)$ y $X=\mathrm{dec}(Z)$. En el diagrama la salida se llama $X'$: es la reconstrucción.
- **restricciones($Z$)**: el cuello de botella o la regularización del código. Es lo que impide que la red aprenda la identidad.
- "**No supervisados!**"
- Score: $A(x)=|f(x)-x|$, con $f=\mathrm{dec}\circ\mathrm{enc}$.
- Diagrama: input $x$ → capa → *code* $z$ (estrecho) → capa → output $x'$. Las cajas punteadas "encoder" y "decoder" se solapan en el código.

**Dos formas de usarlo para anomalías.** El profesor pregunta a clase "¿cómo pensáis que se puede utilizar esto para detectar anomalías?" [03:52:09] y un alumno responde "un espacio latente" [03:52:20].
1. **Error de reconstrucción**, "la forma típica" [03:52:32], [03:53:38]. Funciona igual que con PCA o con regresión: "yo he aprendido una compresión de los datos". PCA concentra lo importante en las componentes principales; "en las no principales se va a poner ruido, y hemos dicho que outliers es ruido" [03:52:55]. Lo que se reconstruye mal es anomalía.
2. **Espacio latente + modelo clásico**, en dos etapas: "Todo lo que hemos aprendido en las horas anteriores lo podéis hacer en espacio latente" [03:52:25]; "yo aprendo aquí todo lo que puedo en el medio, y ahora utilizo un modelo clásico" [03:53:45]–[03:53:51]. "la han utilizado en el pasado y con mucho éxito" [03:53:38]. Su ventaja es la **interpretabilidad**: "si yo pongo ahora un modelo sencillo sobre esto, el modelo sencillo lo puedes hacer todo lo interpretable que yo quiera, entonces puedo ser capaz de explicar más cosas que si intento explicar la red entera tal cual, a través del error de reconstrucción" [03:53:54]–[03:54:11].

**Relación con el denoising.** "siempre que yo haga, por ejemplo, denoising de imágenes, en realidad lo que estoy haciendo es entrenar un modelo que tendrá a la entrada una imagen y a la salida la imagen limpia. Y en el proceso estoy aprendiendo cómo quitar el ruido de esa imagen o cómo quitar los outliers de una base de datos" [03:53:09]–[03:53:28].

**Cruce con la sesión clásica.** En la parte de clustering ya mencionó "un artículo muy bonito" de cuantización vectorial "en el espacio latente del autoencoder" [02:55:47]–[02:55:55]. (nota del redactor: probablemente se refiere a VQ-VAE, centroides en el espacio latente.) Es la misma idea: aplicar un método clásico (distancia a centroides) sobre $Z$.

*Fuente: slide 4 · transcripción [03:51:46]–[03:54:11], [02:55:47]–[02:55:55]*

---

## 3. Deep AE (autoencoder denso con varias capas)

**Idea.** "si lo hacemos [MLP], le ponemos más capas, y seguimos haciendo lo mismo" [03:54:21].

**Slide 5.** Diagrama simétrico: Input (10, azul) → 8 (verde) → 6 (lila) → **Code (3, rojo)** → 6 → 8 → Output (10). Llaves "Encoder" y "Decoder".

### 3.1 Código del profesor: AE denso en Keras sobre MNIST (`AD_AE_simple_clase`)

Usa `keras`. La versión `_clase` corre con Keras 3, que imprime `keras.src.callbacks.history.History`; el enunciado se ejecutó con TF2. También `numpy` y `matplotlib`. Referencia de la cabecera: https://blog.keras.io/building-autoencoders-in-keras.html

```python
from keras.layers import Input, Dense
from keras.models import Model

# (a) "PCA casi-exacto": AE lineal -> subespacio de 32 dims (equivale a PCA)
encoding_dim = 32
input_img = Input(shape=(784,))                                   # MNIST 28x28 aplanado
encoded = Dense(encoding_dim, activation='linear')(input_img)     # ENCODER
decoded = Dense(784, activation='linear')(encoded)                # DECODER
pca = Model(input_img, decoded)        # se define pero no se entrena en el notebook

# (b) AE no lineal: ReLU en el código y sigmoide a la salida (píxeles en [0,1])
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# encoder y decoder por separado (para acceder al espacio latente)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder = Model(encoded_input, autoencoder.layers[-1](encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()            # se ignoran las etiquetas
x_train = x_train.astype('float32')/255.; x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), 784)); x_test = x_test.reshape((len(x_test), 784))

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))          # ENTRADA = SALIDA
# época 1: loss 0.3812 / val 0.1888 ... época 20: 0.0936 / 0.0922 (salida truncada); ~1 s/época
encoded_imgs = encoder.predict(x_test)                   # Z
decoded_imgs = decoder.predict(encoded_imgs)             # X'
```

**Limpieza de ruido (celdas 14–15).** `x_test_nois = x_test + 0.2*np.random.randn(...)`. La figura `celda15_2` muestra tres filas: los dígitos 7,2,1,0,4,1,4,9,5,9 limpios, las versiones ruidosas y sus reconstrucciones. El ruido desaparece aunque algunos dígitos se deforman: el 9 parece un 2 y el 5 sale emborronado. El AE no se entrenó para quitar ruido, pero proyecta sobre la "variedad de dígitos".

**Datos ruidosos frente a limpios (celdas 19–20).**
```python
error_nois = np.mean(np.abs(x_test_nois - x_test_nois_pred)**2, tuple(range(1, x_test_nois.ndim)))
error_tes  = np.mean(np.abs(x_test - autoencoder.predict(x_test))**2, tuple(range(1, x_test.ndim)))
```
Hay una errata en el notebook: escribe `ncoded_imgs = encoder.predict(x_test_nois)` y luego decodifica el `encoded_imgs` que ya tenía calculado. En la figura `celda20_3`, los ruidosos dan entre ~0,04 y 0,107 y los limpios entre ~0,001 y 0,044. Se separan casi del todo, con un pequeño solape hacia 0,04.

**Anomalías sintéticas inventadas (celda 22).** Solo se modifica la muestra 1000:
```python
X_test_anom = x_test.copy()
# Zero:          X_test_anom[1000,:] = 0
# Max:           X_test_anom[1000,:] = x_test.max()
# Negativo:      X_test_anom[1000,:] = 1 - X_test_anom[1000,:]
# Multiplicado:  X_test_anom[1000,:] = 1.5*X_test_anom[1000,:]
X_test_anom[1000,:] += 0.5*np.random.randn(1, 784)       # Ruido (la opción activa en _clase)
# Patch blanco (comentado): tramos del vector aplanado puestos a 1
error_tes = np.mean(np.abs(X_test_anom - autoencoder.predict(X_test_anom))**2, axis=1)
error_tes[1000]   # 0.29246962  (anomalía)
error_tes[1]      # 0.013239625 (normal)
```
- Figura `celda31_7`: un pico aislado de 0,29 en la muestra 1000 (punto rojo) sobre un fondo entre 0 y ~0,045. **Se detecta con claridad.**
- Figura `celda28_6`: la reconstrucción de la muestra ruidosa es un "dígito" deforme; el AE convierte lo que ve en algo parecido a un dígito.
- En el **enunciado** (`AD_AE_simple.md`) el ruido de la limpieza es 0,5 y la anomalía activa es "Negativo": error 0,86767 frente a 0,01122 de una muestra normal. La figura `celda25_5` muestra un único pico de ~0,87.

*Fuente: slide 5 · transcripción [03:54:21] · AD_AE_simple_clase celdas 1–31 (figuras 15, 20, 28, 31); AD_AE_simple celdas 2–25 (figura 25)*

---

## 4. CNN AE (autoencoder convolucional)

**Idea.** "Y si lo hacemos convolucional, pues todavía mucho mejor. Incluimos las capas con [filtros] convolucionales y los pooling layers … para imagen" [03:54:29]–[03:54:37].

**Slide 6, "CNN AE Convolutional".** El ejemplo son **frames de vídeo** de vigilancia:
- Encoder: *Input Frames* 10×227×227 → Conv 11×11 → 512×55×55 → Pool 2×2 → 512×27×27 → Conv 5×5 → 256×27×27 → Pool 2×2 → 256×13×13 → Conv 3×3 → 128×13×13.
- Decoder simétrico, con *Unpooling Layers* y *Deconvolutional Layers*, hasta las *Reconstructed Frames* de 10×227×227.
- CODE: https://www.tensorflow.org/tutorials/generative/autoencoder

(nota del redactor: el diagrama parece el de Hasan et al. 2016, *Learning temporal regularity in video sequences*. Probablemente es "lo mismo pero con vídeo" [04:20:39], aunque en el paquete no hay ningún notebook de vídeo.)

### 4.1 Código del profesor: AE convolucional mínimo sobre MNIST (`AD_AE_convolucional_clase`)

```python
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
encoding_dim = 1                                                   # canales del código
input_img = Input(shape=(28,28,1))
x1 = Conv2D(encoding_dim, 5, padding='SAME')(input_img)            # conv 5x5 lineal
encoded = MaxPooling2D(2)(x1)                                      # 14x14x1 = 196 valores
x2 = UpSampling2D(2, interpolation="bilinear")(encoded)
decoded = Conv2D(1, 5, padding='SAME', activation='relu')(x2)
autoencoder = Model(input_img, decoded)                            # 52 parámetros (26+26)
encoder = Model(input_img, encoded); decoder = Model(encoded, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
x_train = x_train.reshape((-1,28,28,1)); x_test = x_test.reshape((-1,28,28,1))   # datos 4D
hist = autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True, validation_split=0.1)
autoencoder.evaluate(x_test, x_test)        # [0.00944, 0.00944, 0.0390]   (~1 s/época)
```
La salida de la celda 5 incluye además el `summary` de una variante anterior de 1.106 parámetros cuyo código no se conserva: Conv2D de 10 filtros → AvgPool → Conv2D de 5 → AvgPool → UpSampling → Conv2D de 10 → …

- **Limpieza con σ=0,2** (celdas 18–19, figura `celda19_4`): las reconstrucciones salen suaves y algo borrosas, casi sin el ruido de fondo.
- **Ruidosos frente a limpios** (celdas 23–24, figura `celda24_5`): ruidosos entre ~0,036 y 0,079, limpios entre ~0,002 y 0,035. Se separan.

**Anomalía "Patch" (celdas 26–34). Verificado en las salidas: el parche NO se detecta de forma fiable.**
```python
X_test_anom[1000, 2:10, 2:25, 0] = 1                       # rectángulo blanco de 8x23 sobre un "7"
error_tes = np.mean(np.abs(X_test_anom - autoencoder.predict(X_test_anom))**2, axis=(1,2,3))
error_tes[1000]   # 0.034234814
error_tes[1]      # 0.0063956343
```
- Figura `celda30_7`: la reconstrucción **reproduce el parche** casi tal cual (una barra blanca sobre el 7).
- Figura `celda34_8`: el error de la muestra 1000 (≈0,0342, punto rojo) está en el nivel de los máximos de las muestras normales, varias de las cuales llegan a 0,034–0,0345. Queda empatado entre los primeros puestos del ranking, pero no destaca.

**Lección (nota del redactor, a partir de la salida):**
- El modelo tiene solo 52 parámetros, así que el problema no es la capacidad en número de pesos. Es que aprendió una operación **genérica y local**: una conv 5×5, pooling 2 y reescalado, casi un "suavizado + identidad". El cuello de botella apenas restringe (196 valores para 784) y vale para cualquier imagen. El AE reconstruye "demasiado bien" incluso lo que nunca vio, y el error deja de discriminar.
- El error que sí aparece viene sobre todo de la cantidad de "tinta" añadida, no de que el modelo reconozca algo raro.
- Es el mismo principio que las *capabilities* del profesor (sección 11): un detector por reconstrucción solo detecta las anomalías que su entrenamiento le ha dejado **incapaz de reproducir**. Una pérdida de reconstrucción baja no significa un buen detector. Las palancas son un código más estrecho, más regularización, un modelo más global (denso) y validar con anomalías, no con la pérdida.
- Contraste con el AE denso de la sección 3, cuyo código es de 32 sobre 784: allí la anomalía de ruido sí salía como un pico de 0,29.

**Enunciado (`AD_AE_convolucional.md`).** Misma estructura con `MaxPooling2D(4)`/`UpSampling2D(4)` (código de 7×7×1). En el notebook quedaron variantes de alumnos:
```python
# "Modelo de Luis": regularización L1 en los pesos + AveragePooling(6)
x1 = Conv2D(1,5,padding='SAME',kernel_regularizer=regularizers.l1(0.01))(input_img)
encoded = AveragePooling2D(6)(x1)
x2 = Conv2D(1,5,padding='SAME',kernel_regularizer=regularizers.l1(0.01))(encoded)
x3 = UpSampling2D(7, interpolation="bilinear")(x2)
decoded = Conv2D(1,5,padding='SAME',activation='relu')(x3)
# "Modelo de Samuel y Pablo y Jesus": 2 niveles, kernels 2x2, AvgPool + upsampling bicúbico (20 parámetros)
x1 = Conv2D(1,2,padding='SAME')(input_img); x2 = AveragePooling2D(2)(x1)
x3 = Conv2D(1,2,padding='SAME')(x2);        encoded = AveragePooling2D(2)(x3)       # 7x7x1
x5 = UpSampling2D(2, interpolation="bicubic")(encoded); x6 = Conv2D(1,2,padding='SAME')(x5)
x7 = UpSampling2D(2, interpolation="bicubic")(x6);      decoded = Conv2D(1,2,padding='SAME')(x7)
```
- Se entrenó el último modelo (20 parámetros): loss 0,0326 / val 0,0330 en la época 20 (~6–10 s/época) y test 0,0593.
- La anomalía activa es "Negativo" y el error se calcula con `axis=1`, que solo vale para datos aplanados; `_clase` lo corrige a `axis=(1,2,3)`.
- Las salidas 0,86767/0,01122 son idénticas a las del AE simple. (nota del redactor: copiadas, no recalculadas.)

*Fuente: slide 6 · transcripción [03:54:29]–[03:54:37] · AD_AE_convolucional_clase celdas 1–34 (figuras 19, 24, 27, 30, 34); AD_AE_convolucional celdas 2–31*

---

## 5. Denoising como detección de anomalías y tipos de ruido

**Idea.** "si mi aplicación es quitar ruido de imágenes, al final es lo mismo que intentar detectar anomalías" [03:54:47].

**Slide 7, "Denoising based".**
- $\hat{x}(y)=y+\sigma^2\nabla_y\log p(y)$. (nota del redactor: es la fórmula de Tweedie/Miyasawa, el denoiser óptimo en MSE para ruido gaussiano de varianza $\sigma^2$.)
- "**x-y ≈ Dp(x)**": el residuo del denoiser es proporcional al gradiente de la log-densidad. Un denoiser entrenado conoce implícitamente la forma de $p(x)$.
- Cuatro paneles muestran una densidad con contorno verde en forma de nube. Las flechas rojas (el campo $\nabla\log p$) apuntan a la zona de alta densidad, con la densidad de cada vez menos a cada vez más concentrada en el contorno. El último panel muestra trayectorias que convergen al contorno.

**Tipos de ruido, ejemplo en vivo con imágenes** [03:55:04]–[03:57:27]:
- Pregunta a clase: "¿sabes qué tipo de ruido es éste? ¿Parece gaussiano o no?" [03:55:04]–[03:55:11].
- **Sal y pimienta**: "un ruido muy reconocible, porque aparece de forma no correlada con la imagen en toda la superficie" [03:56:08]–[03:56:17]. Son "como puntitos blancos … en realidad no son puntitos blancos, sino que puede tener cualquier valor" [03:56:45]–[03:56:52]. "sí afecta mucho a la calidad de la imagen, más que el ruido gaussiano" [03:56:58].
- **Gaussiano**: "casi no lo deberíais ver, veis más o menos iguales estos 2" [03:56:17]. Dice que "suaviza los píxeles alrededor", que "acabáis teniendo algo más borroso" y que "a resoluciones pequeñas no afecta demasiado" [03:56:27]–[03:56:37]. (nota del redactor: lo que describe es más bien el efecto del *filtrado* gaussiano. El ruido gaussiano aditivo es independiente píxel a píxel. Se recoge tal como lo dijo.)
- Con el denoiser "puedo pasar a este tipo de reconstrucciones, que no es perfecta, ni mucho menos" [03:57:14]. "Outliers hizo lo mismo" [03:57:27]: con los outliers pasa lo mismo que con el ruido.

*Fuente: slide 7 · transcripción [03:54:43]–[03:57:27] (las imágenes mostradas en pantalla no están disponibles)*

---

## 6. CNN AE – U-Net para denoising

**Slide 8, "CNN AE – UNET. Para denoising (normalmente para segmentación)".**
- Entra una imagen ruidosa (un castillo con ruido de color). Bloques de $f=128$ → (Conv 2×2, stride 2, PReLU) → 256 → 512 → 1024 en el fondo.
- Subida con Transposed Conv + PReLU y **concatenaciones** con el encoder ($512+1024$, $256+512$, $128+256$), Conv 3×3 PReLU y bloque de salida.
- Al final hay una **suma** (flecha punteada, *Addition*) con la entrada ruidosa, que da la *Predicted Image*. Esa suma global es aprendizaje residual: la red predice la corrección o el ruido.

**Código del profesor (`UNET_convolutional_denoising.md`; solo existe esta versión, sin `_clase`).**
```python
def conv_block(inputs, filters, pool=True):
    # 2 x (Conv 3x3 -> BatchNorm -> ReLU -> Dropout 0.2), y MaxPool 2x2 opcional
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if pool: return x, tf.keras.layers.MaxPool2D((2, 2))(x)   # (skip, siguiente nivel)
    return x

def build_unet(shape, num_classes):
    inputs = tf.keras.Input(shape)
    x1, p1 = conv_block(inputs, 16); x2, p2 = conv_block(p1, 32)
    x3, p3 = conv_block(p2, 64);     x4, p4 = conv_block(p3, 128)
    b1 = conv_block(p4, 256, pool=False)                                   # bottleneck
    u1 = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(b1)
    x5 = conv_block(tf.keras.layers.Concatenate()([u1, x4]), 128, pool=False)
    u2 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x5)
    x6 = conv_block(tf.keras.layers.Concatenate()([u2, x3]), 64, pool=False)
    u3 = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x6)
    x7 = conv_block(tf.keras.layers.Concatenate()([u3, x2]), 32, pool=False)
    u4 = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x7)
    x8 = conv_block(tf.keras.layers.Concatenate()([u4, x1]), 16, pool=False)
    output = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='relu')(x8)  # 'softmax' para segmentar
    return tf.keras.Model(inputs, output)

model = build_unet([32,32,3], 3)                         # CIFAR-10 RGB
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
x_train_noise = x_train.copy() + 0.1*np.random.randn(*x_train.shape)     # ruido gaussiano σ=0.1
model.fit(x_train_noise, x_train, epochs=10, batch_size=32, shuffle=True,
          validation_data=(x_test, x_test))      # ojo: la validación es limpia->limpia
# loss 0.0201 -> 0.0069; val_loss 0.0039 -> 0.0021; 33-52 s/época (1563 pasos)
decoded_imgs = model.predict(x_test + 0.1*np.random.randn(*x_test.shape))
```
- A diferencia de la slide, este código no suma la entrada a la salida.
- Figura `celda12_2`: 10 imágenes CIFAR ruidosas (gato, barcos, avión, rana, coche, camión…) y sus versiones limpias, sin grano y algo borrosas.
- El notebook **no calcula ningún score de anomalía**. Para usarlo como detector se tomaría el residuo $|f(x)-x|$, que por la slide 7 se relaciona con $\nabla\log p$.

*Fuente: slide 8 · UNET_convolutional_denoising celdas 1–12*

---

## 7. Modelos generativos explícitos: $p(x)=f(x)$, baja probabilidad = anomalía

**Idea.** "no hay nada nuevo … tengo un modelo generativo explícito, quiero decir que ha aprendido la distribución de probabilidad … volvemos a hace 2 h o 3 h. Tienen una imagen, la paso por aquí, y me dice que la probabilidad es muy baja. Lo probable es que sea una anomalía" [03:57:46]–[03:58:06].

**Slide 10, "Modelos Generativos – Explícitos: p(x) = f(x)".** Muestra un mapa de densidad 2D (ejes ~96–106 × 0,04–0,16) con escala de color de blanco (0) a rojo oscuro (2,0). El máximo está hacia (100; 0,095) y la forma es elíptica e inclinada.

**Qué dijo:**
- Regla: anomalía si $p(x)<\tau$. "seguís necesitando el umbral que os diga si la probabilidad es menos de 0.1, esa es una anomalía" [03:58:16].
- Ventaja: "tiene mucho más poder que poner una gaussiana sobre unos datos que seguramente no han [sido] gaussianos … No hacemos suposiciones a la distribución y en principio, si tenemos muchos datos, podemos estimar algo muy bien" [03:58:28]–[03:58:36].
- "cualquier modelo generativo va a caer dentro de la categoría de los de probabilidad baja → anomalía alta" [03:58:41].
- Inconvenientes: "van a ser más caros. Y luego tendremos que tomar algunas decisiones" [03:58:53]–[03:58:55].
- Vale para cualquier dominio: "si tenéis cualquier modelo de probabilidad de imagen, de series temporales, de lo que queráis" [03:58:08]; "dependiendo de qué dominio estéis, cogéis el modelo que tengáis" [04:00:45].

*Fuente: slides 9 y 10 · transcripción [03:57:29]–[03:58:55], [04:00:45]–[04:00:48]*

---

## 8. Normalizing flows

**Idea.** "los modelos [de flows] van a ir de una distribución que tiene una forma rara a algo que yo conozco, y luego puedo volver hacia atrás … Puedo sacar muestras de aquí y ver [a dónde] han ido; entonces puedo seguir haciendo lo mismo que todos los métodos de la categoría de probabilidad" [03:59:01]–[03:59:17].

**Slide 11.** *Data Distribution* (densidad en forma de arco o banana, con sus marginales) $\mathcal{X}$ —$f_\theta$→ $\mathcal{Z}$ *Known Distribution* (gaussiana centrada). La flecha roja de vuelta es $f_\theta^{-1}$: sirve para generar muestreando en $\mathcal{Z}$ y deshaciendo la transformación.

(nota del redactor) Cambio de variable: $p_X(x)=p_Z(f_\theta(x))\,\left|\det\frac{\partial f_\theta(x)}{\partial x}\right|$.

**Requisitos.** "esto solo funcionará si tenéis ordenadores, servidores que valgan la pena, y tenéis muchos datos; si no, no. Pero en cuanto a principios, todo exactamente igual" [03:59:23].

*Fuente: slide 11 · transcripción [03:58:55]–[03:59:23]*

---

## 9. RBIG – Rotation-Based Iterative Gaussianization (el flow de Valero Laparra)

**Idea.** "Esto es [de] Valero, que es [su] método favorito … la idea es que yo no conozco la distribución original. La llevo a la gaussiana a través de hacer transformaciones, pero siempre puedo deshacerlas al final. Y puedes hacer lo mismo: puedes sacar muestras de aquí y ver a qué correspondían al otro lado, y puedes calcular la probabilidad de X, lo mismo que hemos visto por la mañana" [03:59:35]–[04:00:10]. También: "ya os he dicho que a Valero le gusta mucho RBIG" [04:20:46].

**Slide 12, "Flow: Rotation Based Iterative Gaussianization (RBIG)".** "From P(x) to P(y) (Gaussian)" mediante $\mathcal{G}$. A la izquierda, una densidad desconocida en forma de aspa con un gran "?"; a la derecha, contornos circulares concéntricos. La fórmula que aparece en la slide es:
$$p_{\mathbf{x}}(\mathbf{x})=p_{\mathbf{y}}(\mathcal{G}(\mathbf{x}))\left|\frac{d\mathcal{G}(\mathbf{x})}{d\mathbf{x}}\right|^{-1}$$

**Verificación del exponente −1. Conclusión: es una errata de la slide.** (nota del redactor)
- El exponente está en la imagen; no es un fallo del extractor.
- Con $\mathbf{y}=\mathcal{G}(\mathbf{x})$, el cambio de variable da $p_{\mathbf{x}}(\mathbf{x})=p_{\mathbf{y}}(\mathcal{G}(\mathbf{x}))\,|\det J_{\mathcal{G}}(\mathbf{x})|$, sin exponente. Es lo que publica el artículo de RBIG (Laparra, Camps-Valls y Malo, IEEE TNN 2011): $p_x(x)=p_y(\mathcal{G}(x))\,|\nabla_x\mathcal{G}(x)|$.
- Comprobación intuitiva: si $\mathcal{G}$ comprime una nube ancha en una gaussiana estándar, $|\det J|<1$, y $p_x$ tiene que ser *menor* que $p_y$ (la masa se reparte en más volumen). Eso exige multiplicar por $|\det J|$, no dividir.
- El $-1$ solo sería correcto con el jacobiano de la inversa: $|\det J_{\mathcal{G}^{-1}}(\mathbf{y})|^{-1}$.
- En RBIG, $\log|\det J_{\mathcal{G}}|$ es la suma por capas de los log-derivadas de las gaussianizaciones marginales; las rotaciones tienen determinante ±1.

**Mecanismo** (nota del redactor). Cada capa hace:
1. **Gaussianización marginal** de cada dimensión: CDF empírica seguida de la inversa de la CDF normal.
2. **Rotación** (PCA, ICA o aleatoria).

Se itera hasta que la reducción de información por capa se estabiliza. La suma de esas reducciones (`info_loss`) estima la *total correlation* de los datos. El artículo de la slide clásica 32 (arXiv 2010.03807, con Santos-Rodríguez como coautor) usa esto para estimar TC, entropía, información mutua y KL.

### 9.1 API de la librería `rbig` (IPL-UV) en los notebooks

```python
!pip install python-picard                               # dependencia (ICA Picard)
!pip install "git+https://github.com/IPL-UV/rbig.git"    # paquete py_rbig 0.0.1; funciona en Python 3.11 (Colab, AD_rbig_demo_clase)
```

| Elemento | Uso en los notebooks |
|---|---|
| `from rbig import RBIG` | Clase principal (también `from rbig import RBIG, entropy_marginal`). |
| `RBIG(rotation='pca', max_layers=1000)` | `rotation`: `'pca'`/`'PCA'`, `'ica'`, `'random'`. `max_layers`: 1000 (10000 en teoría de la información). |
| `RBIG(bound_ext=0.3, rotation=rot)` | `bound_ext`: cuánto se extiende el soporte de las marginales más allá del rango de train (0,3 en CIFAR, 0,01 en penguins). |
| `.fit(X)` / `.fit_transform(X)` | Ajusta; `fit_transform` devuelve además los datos gaussianizados. |
| `.inverse_transform(Z)` | Vuelve al espacio original: reconstrucción, o **generación** a partir de `randn`. |
| `.predict_proba(X)` | **Densidad $p(x)$** de cada muestra. Ojo: es la densidad, no su logaritmo (ver 9.7). |
| `.info_loss` | Reducción de información por capa; `np.cumsum` da la curva acumulada. |
| `.total_correlation()` | TC en bits; se multiplica por `np.log(2)` para pasar a nats. |
| `entropy_marginal(X)` | Entropías marginales por dimensión. |
| `from rbig._src.mutual_info import MutualInfoRBIG` | `MutualInfoRBIG(max_layers, rotation).fit(X, Y)` y luego `.mutual_info()`. |
| `rbig_model.det_jacobians` | Aparece solo en un `print` comentado. |
| `#from rbig.rbig import RBIGKLD, …` | API antigua con KL, comentada; **no se usa**. |

### 9.2 `AD_rbig_demo_clase`: RBIG para anomalías (toy y datos reales de ADBench)

```python
seed = 123; rng = np.random.RandomState(seed=seed); num_samples = 10000
x = np.abs(2*rng.randn(1, num_samples)); y = np.sin(x) + 0.25*rng.randn(1, num_samples)
data = np.vstack((x, y)).T                                   # (10000, 2): "seno con ruido"
x_tr, x_ts = train_test_split(data, test_size=0.2)
rbig_model = RBIG(rotation='pca', max_layers=1000)
data_trans = rbig_model.fit_transform(x_tr)                  # 3,6 s CPU (8000x2)
prob_input = rbig_model.predict_proba(x_ts)                  # 29 ms para 2000 puntos
# "se le aplica una no linealidad monotónica a la probabilidad para exagerar el efecto"
anomaly = 1 - np.log(prob_input + 1)**0.1
plt.hist(anomaly, 50); plt.scatter(x_ts[:,0], x_ts[:,1], s=1, c=anomaly, cmap='Reds')
```
- Figura `celda08_2`: la masa del score está entre 0,05 y 0,3 (moda ~0,1), con una cola larga hasta 1,0.
- Figura `celda09_3`, "Test Data Anomalies": la nube con forma de seno sale clara y los puntos de las colas (x>5, extremos de y) salen rojo oscuro. RBIG captura la forma no gaussiana.

**Con datos reales de ADBench** (celdas 11–20):
```python
!pip install adbench          # adbench 0.1.11 (instala pyod 1.0.0, etc.)
from adbench.myutils import Utils
Utils().download_datasets(repo='jihulab')            # CV_by_ResNet18, NLP_by_BERT y Classical
data = np.load(path + '6_cardio.npz', allow_pickle=True)     # alternativa comentada: '5_campaign.npz'
X, y = data['X'], data['y']
sns.pairplot(pd.DataFrame(X[:, :6]).assign(y=y), hue='y')    # etiquetas reales
rbig_model = RBIG(rotation='pca', max_layers=1000)
rbig_model.fit_transform(X)                    # se ajusta con TODO X, anomalías incluidas (no supervisado)
anomaly = 1 - np.log(rbig_model.predict_proba(X) + 1)**0.1
sns.pairplot(pd.DataFrame(X[:, :6]).assign(y=anomaly), hue='y')
```
- Figura `celda20_5`: pairplot de *cardio* coloreado por el score, en una escala de −0,25 a 0,75. El score puede ser negativo cuando $p>e-1$. Los puntos con score alto se concentran en los grupos alejados y en la periferia.
- **No se calcula ninguna métrica** (ni AUC ni AUPR) contra `y`.
- El **enunciado** `AD_rbig_demo.md` solo tiene la parte toy (celdas 1–9); aplicarlo a datos reales queda como ejercicio.

### 9.3 `rbig_demo_val`: gaussianizar, invertir, generar y estimar densidad

```python
rbig_model = RBIG(rotation='pca', max_layers=1000); data_trans = rbig_model.fit_transform(data)   # ~6 s CPU
data_approx = rbig_model.inverse_transform(data_trans)
np.testing.assert_array_almost_equal(data, data_approx)        # pasa: la inversión es exacta
plt.plot(np.cumsum(rbig_model.info_loss), 'o-')                # reducción de información
data_original_synthetic = rbig_model.inverse_transform(rng.randn(*data.shape))   # generación
prob_input = rbig_model.predict_proba(data)                    # 0,2 s
plt.hist(np.log(prob_input + 1), 50)
```
- `celda07_2`: los datos tras RBIG forman una nube aproximadamente gaussiana.
- `celda12_4`: la reducción acumulada pasa por 0,40 → 0,50 → 0,64 → 0,67 → 0,69 y se estabiliza hacia 0,71. En 2D converge en muy pocas capas.
- `celda16_6`: los datos sintéticos reproducen el seno con sus colas.
- `celda20_7`: el histograma de $\log(p+1)$ queda concentrado por debajo de 0,6.

### 9.4 `rbig_demo_val_clase_penguins`: datos tabulares reales pequeños

```python
penguins = sns.load_dataset("penguins").dropna()
data = penguins.values[:, 2:6].astype(float)       # 4 variables numéricas, (333, 4). El notebook usa np.float, que ya no existe en numpy>=1.24
rbig_model = RBIG(bound_ext=0.01, rotation='ica')
data_trans = rbig_model.fit_transform(data)        # "64/1000 capas, Total Info Red: 0.8318", 3 s
np.sum(np.abs(data - rbig_model.inverse_transform(data_trans)))   # 3,17e-10: invertible
data_original_synthetic = rbig_model.inverse_transform(np.random.randn(*data.shape))
prob_input = rbig_model.predict_proba(data)        # 0,28 s
```
Figuras abiertas en esta revisión:
- `celda13_3`: `info_loss` tiene 5 valores (≈0; 0,33; 0,38; 0; 0,13) que suman 0,83, el mismo total que la barra de progreso aunque esta hablara de 64 capas.
- `celda15_5`: dos marginales gaussianizadas, una nube redondeada.
- `celda26_9` y `celda28_10`: los datos **sintéticos** reproducen la estructura bimodal o en grupos de las especies (comparados con el pairplot real `celda29_11`), aunque con grupos menos nítidos.
- `celda33_12` y `celda34_13` son **salidas obsoletas** de otra ejecución: el histograma tiene ~10.000 muestras y el scatter es el seno, no los 333 pingüinos.

### 9.5 `rbig_demo_val_clase_CIFAR_cond`: densidades condicionadas por clase (imágenes)

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:, 4:-5:2, 4:-5:2, 0]                  # recorte, submuestreo y canal 0 -> (50000, 12, 12)
x_train_0 = x_train[y_train[:,0]==0].reshape(-1, 144)/255   # clase 0 (avión) y clase 1 (automóvil)
rbig_model_0 = RBIG(bound_ext=0.3, rotation='pca'); rbig_model_1 = RBIG(bound_ext=0.3, rotation='pca')
rbig_model_0.fit_transform(x_train_0)   # (5000,144): 62 capas, ~1 min (1 s/capa), Info Red 212,38
rbig_model_1.fit_transform(x_train_1)   # 61 capas, Info Red 214,20
data_synthetic_0 = rbig_model_0.inverse_transform(np.random.randn(100, 144))   # "imágenes" sintéticas
pX0_en_m0 = rbig_model_0.predict_proba(x_train_0[:100]); pX0_en_m1 = rbig_model_1.predict_proba(x_train_0[:100])
plt.plot(np.log(pX0_en_m0+1), np.log(pX0_en_m1+1), '.'); plt.plot([0,200],[0,200],'k-')
```
Figuras, todas abiertas. Eje x = $\log(p+1)$ bajo el modelo 0; eje y = bajo el modelo 1; un punto bajo la diagonal tiene más probabilidad en el modelo 0.
- `celda22_4` / `celda25_5`: datos gaussianizados y reducción de información (hasta ≈212–214).
- `celda31_8` / `celda32_9`: imágenes sintéticas de 12×12 de cada clase. Son texturas ruidosas, poco reconocibles: en 144D con 5.000 muestras el modelo no capta la semántica.
- **Train**:
  - Clase 0 (`celda38_11`): casi todos los puntos bajo la diagonal (correcto), pero 2 tienen y=0.
  - Clase 1 (`celda40_13`): casi todos sobre la diagonal, y muchos con x=0, es decir, probabilidad ≈0 bajo el modelo 0.
  - `celda37_10` y `celda39_12` muestran lo mismo como series.
- **Sintéticas**: separación perfecta. Las de clase 0 (`celda44_15`) quedan todas bajo la diagonal, con valores de hasta $\log p\approx370$ y varias con y=0. Las de clase 1 (`celda46_17`) quedan todas sobre la diagonal, con muchas x=0.
- **Test**:
  - Clase 0 (`celda50_19`): nube alrededor de la diagonal y varios puntos con x=0, es decir, **probabilidad ≈0 bajo su propio modelo**; alguno en (0,0).
  - Clase 1 (`celda52_21`): mayoría sobre la diagonal (correcto) y muchos con x=0.
  - Generaliza peor que en train: hay sobreajuste de la densidad con 144 dimensiones y 5.000 muestras.
- **Uso para anomalías**: una muestra con baja probabilidad bajo *todos* los modelos condicionados es anómala; si solo es baja bajo el modelo de su clase, es sospechosa.
- (nota del redactor) La celda 16 imprime `(1000, 32, 32, 3)`, que no cuadra con el reshape: se ejecutó fuera de orden. La cabecera de la celda 2 dice "Penguins Data" porque se copió de la plantilla.

### 9.6 `information_theory_with_RBIG`: TC, entropía e información mutua

```python
from rbig import RBIG, entropy_marginal
from rbig._src.mutual_info import MutualInfoRBIG
# Total Correlation: gaussiana correlada de 10D (10000 muestras)
data = rng.randn(10000, 10) @ A;  C = A.T @ A
tc_original = np.log(np.sqrt(np.diag(C))).sum() - 0.5*np.log(np.linalg.det(C))     # 9,9326
tc_rbig = RBIG(max_layers=10000, rotation='PCA').fit(data).total_correlation()*np.log(2)   # 6,8649 (14 s)
# Entropía (5000 muestras): H = sum H(x_i) - TC
H_rbig = entropy_marginal(data).sum() - ent_model.total_correlation()*np.log(2)    # 13,5708 frente a 10,3427 analítica
# Información mutua: X, Y de 10D cada una, gaussiana conjunta de 20D
mi_rbig = MutualInfoRBIG(max_layers=10000, rotation='PCA').fit(X, Y).mutual_info()*np.log(2)  # 6,5699 frente a 8,0713 (36 s; entrena 3 RBIG)
```
(El código está compactado; en el notebook `fit` y la consulta van en líneas separadas.)

Fórmulas (nota del redactor):
- $TC(\mathbf{x})=\sum_i H(x_i)-H(\mathbf{x})$; en una gaussiana, $TC=\sum_i\log\sigma_i-\tfrac12\log|C|$.
- $H(\mathbf{x})=\sum_i H(x_i)-TC$.
- $I(X;Y)=TC([X,Y])-TC(X)-TC(Y)$. Por eso `MutualInfoRBIG` entrena tres modelos.

Notas:
- La **KL no aparece** (solo el import comentado de `RBIGKLD`).
- Las estimaciones en 10D quedan lejos de las analíticas.
- (nota del redactor) Varias referencias "analíticas" ya están en nats y aun así se multiplican por `np.log(2)` (en MI y en entropía), así que la comparación es dudosa. En clase no se comentó.

### 9.7 Problema práctico: probabilidad ≈0, empates y AUPR (nota del redactor)

- **De dónde sale el 0.** `predict_proba` devuelve la densidad $p(x)$, no $\log p(x)$. Hay dos motivos para que salga 0:
  1. Si un punto cae fuera del soporte extendido de alguna marginal (`bound_ext`), la derivada de esa gaussianización marginal es 0 y $p$ es exactamente 0.
  2. En dimensión media o alta, $p$ abarca muchos órdenes de magnitud. En CIFAR hay valores de $\log p$ de hasta ~370, y los de baja densidad se van a un *underflow* a 0.0 en float64.
- **La transformación del profesor lo empeora.** `1 - log(p+1)**0.1` es monótona, así que en principio no cambia el orden. Pero $\log(p+1)\approx p$ cuando $p\ll1$, y en coma flotante $p+1=1$ para $p<10^{-16}$. **Todos los puntos poco probables, justo los candidatos a anomalía, acaban con el mismo score.** Los "x=0" de los gráficos de CIFAR son esa saturación: no significa que $p$ valga 0, sino que vale $p\lesssim0{,}01$, sea $10^{-3}$ o $10^{-300}$.
- **Por qué daña la AUPR.** La AUPR (`average_precision_score`) depende solo del orden. Un bloque grande de empates en la parte alta del ranking mezcla fraudes y normales al azar y hunde la precisión justo en los primeros puestos, que es donde más pesa.
- **Cómo evitarlo:**
  1. Usar **log-probabilidad**: score = $-\log p(x)$. Si la librería tiene un método de log-densidad (comprobar con `dir(modelo)`), usarlo. Si no, `np.log(np.clip(p, 1e-300, None))`. Nunca usar `log(p+1)` ni `p` directamente.
  2. Contar cuántos puntos quedan en el suelo (`(p==0).mean()`). Si son muchos, subir `bound_ext` o estandarizar antes.
  3. Deshacer los empates restantes con un segundo score: ordenar primero por $-\log p$ de RBIG y, dentro de cada empate, por el $-$`score_samples` del GMM o por la distancia de Mahalanobis.
  4. Ajustar el modelo solo con no-fraude y evaluar en datos que no se usaron para ajustar, porque en train no aparecen los ceros que sí aparecen en test (CIFAR).

*Fuente: slide 12, slide clásica 32 · transcripción [03:59:35]–[04:00:10], [04:20:46] · notebooks AD_rbig_demo(_clase), rbig_demo_val, rbig_demo_val_clase_penguins, rbig_demo_val_clase_CIFAR_cond, information_theory_with_RBIG (todas las celdas y figuras)*

---

## 10. Modelos autorregresivos y VAE

### 10.1 Autorregresivos ("predictivos")
- "Esto puede ser más útil, sobre todo con [texto/series], pero también con imágenes, que son los modelos generativos autorregresivos … en todos estos también tenemos acceso a las probabilidades. Muchas veces son condicionadas … pero nos siguen valiendo: si la probabilidad es muy muy baja, la probabilidad de que sea anómalo es muy muy alta" [04:00:17]–[04:00:41].
- **Slide 13, "Autoregresive Generative Models".**
  - Texto (LLMs): la secuencia "… problems turning **into** banking crises as …", con flechas desde *problems* y *turning* hacia *into*, que es el token predicho (en rosa). Una flecha "forward".
  - Imagen (PixelCNN): en una rejilla, los píxeles ya generados (azul) condicionan la distribución del píxel actual (rojo), con un histograma $p(x_i)$ entre 0 y 256.
- (nota del redactor) $p(x)=\prod_i p(x_i\mid x_{<i})$; el score sería $-\log p(x)$.

### 10.2 VAE
- **Slide 14, "Variational AE":** "El espacio latente es continuo – Modelo generativo". Un "0" de MNIST pasa por un MLP encoder, que da $\sigma$ (amarillo) y $\mu$ (azul); se muestrea (*Sample*) y un MLP decoder reconstruye el "0". CODE: https://www.tensorflow.org/tutorials/generative/cvae · JUGAR: http://leonchen.io/keras-js/#/mnist-vae
- Lo que dijo: "La única diferencia es que aquí tendremos algo que asumimos que es una variable gaussiana, y podemos [muestrear] … comparado con el tradicional … que no se podía" [04:00:57]–[04:01:04]. Y después: "si esto fuera un autoencoder [variacional] … tengo una variable aleatoria de la que puedo tomar muestras, y eso me permite tener acceso a p(x), básicamente … le pregunto a p(x) cuál es la probabilidad" [04:02:01]–[04:02:21].
- (nota del redactor) $\log p(x)\ge\mathrm{ELBO}=\mathbb{E}_{q(z|x)}[\log p(x|z)]-D_{KL}(q(z|x)\|p(z))$. Se usa el $-$ELBO (o el error de reconstrucción esperado) como score.

*Fuente: slides 13 y 14 · transcripción [04:00:17]–[04:01:09], [04:02:01]–[04:02:26]*

---

## 11. Generativos implícitos: GAN y el discriminador como score (ejercicio de clase)

**Slides.**
- Slide 15, "Modelos Generativos – Implícito: p(x) = g(f(x))", con los mismos paneles de campos de gradiente que la slide 7: la densidad no se obtiene directamente, sino a través de otra función.
- Slide 16, "GANs": *Noise Source* → *Generator* $F_G$ → *Generated Example*. Junto con un *Real Example*, entra en el *Discriminator* $F_D$, que decide *Real*/*Fake*.

**Planteamiento.** "aquí se pone ruido, aquí se genera algo desde el ruido, aquí se meten ejemplos, y aquí se clasifica si son reales o falsos" [04:01:29]–[04:01:39]. "El problema que tiene esto es que es generativo, pero yo no tengo acceso a p(x)" [04:01:49]–[04:01:57]. "Tengo una red, todo lo complicada que queráis, pero no tengo esa probabilidad. Tengo que meter ruido, saco muestras, pero no le puedo preguntar: esto, ¿es una anomalía o no?" [04:02:26]–[04:02:46].

**Ejercicio (5 min).** "¿qué haréis vosotros con una GAN? … eso es un artículo de los mejores de 2017. A alguien se le ocurrió cómo utilizar estas GAN para la detección de anomalías, y yo sólo necesito que penséis en 5 min lo mismo que pensaron ellos en no sé cuántos meses" [04:02:49]–[04:03:08]. Contexto: en 2017–2018 "no había difusión ni normalizing flows … con autoencoders no había acceso a p(x). Entonces el deep learning no se utilizaba para esto" [04:03:19]–[04:03:39].

**Solución, que sale de la clase y el profesor valida.** "una vez ya entrenado el discriminador …" [04:04:18]. "no puedo utilizar mi superred que convierte ruido en imágenes, pero he entrenado algo que, dada una entrada, me puede decir si es real o falso. Entonces yo simplemente pongo mi anomalía en la entrada del discriminador y me dará un score de real o falso; eso lo puedo utilizar directamente como una predicción de anomalía" [04:04:44]–[04:05:05].
- Score: $A(x)=1-F_D(x)$ (notación del redactor).
- "eso fue un artículo de NeurIPS … alguien dijo: ya que estamos entrenando estas cosas, igual el discriminador vale para lo nuestro. Y también hizo que se prestara más atención a la arquitectura del discriminador y se complicase más que originalmente, que simplemente era para contrabalancear el aprendizaje" [04:11:07]–[04:11:35].
- (nota del redactor: no nombra el artículo. AnoGAN, de Schlegl et al., es de IPMI 2017, no de NeurIPS, y combina el residuo del generador con features del discriminador. No se puede confirmar a cuál se refiere.)

**Cuándo falla: ruido frente a un león.** "va a depender de varias cosas, como está muy mal definido el problema … estoy intentando clasificar gatos y perros [y] meto mucho ruido a una de las imágenes. Pero puede haber otro caso: en vez de un gato o un perro, meto un león … las 2 pueden considerarse outliers o fuera de distribución. ¿En cuál creéis que tendría más problema el discriminador?" [04:05:25]–[04:06:04].
- La respuesta (en la transcripción todo sale como PROF, pero suena a un alumno) es que las "versiones ruidosas de lo que ya hemos tenido" las detecta ya como *fake* [04:06:14]–[04:06:18].
- Explicación: depende de lo que el discriminador vio como falso o verdadero durante el entrenamiento, y "eso está fuera de vuestro control" [04:06:39]–[04:06:46]. Mientras el generador aprende, "es bastante probable que en algún momento se conserve parte del ruido y parte de la imagen" [04:07:00]. El discriminador ya ha visto como *fake* imágenes con algo parecido a ruido gaussiano, así que **detecta bien las anomalías de tipo ruido** [04:07:00]–[04:07:19].
- En cambio, "sería raro que la GAN se hubiera puesto a generar caballos o leones muy realistas" (por ejemplo, una GAN condicional de perros y gatos), "entonces yo ahí no le he dicho nada al discriminador". **Falla con las anomalías semánticas** [04:07:23]–[04:07:49].

**Performance frente a capabilities** [04:07:55]–[04:11:06]. La evaluación "súper de moda": antes se medía solo la tarea (un clasificador perro/gato con un 90 % de aciertos por clase); ahora interesa "qué otras tareas o habilidades ha aprendido tu modelo por el camino". Pone como ejemplos los benchmarks de LLMs con unas 100 tareas, y probar un modelo de perros y gatos con lobos y leones *sin* transfer learning. Aplicado a anomalías: "es importante distinguir cuáles son las capabilities de tu modelo a la hora de saber qué tipo de anomalías vas a poder detectar" [04:10:38]–[04:10:54]. Las del tipo "añadir ruido, del estilo de la fuente de ruido que tienes" se detectan muy bien; con las de tipo fuera de distribución semántico "vamos a tener más problemas" [04:10:54]–[04:11:06].

**Alternativa.** "coger el generador, que en realidad sí que sabe cosas de la probabilidad, y modificarlo para que se convirtiera en un detector de anomalías" [04:11:46]–[04:11:53], "pero la idea buena es la otra" [04:11:57].

*Fuente: slides 15 y 16 · transcripción [04:01:14]–[04:11:57]*

---

## 12. Difusión

**Slide 17, "Diffusion".**
- *Forward SDE (data → noise)*: $\mathrm{d}\mathbf{x}=\mathbf{f}(\mathbf{x},t)\,\mathrm{d}t+g(t)\,\mathrm{d}\mathbf{w}$, de $\mathbf{x}(0)$ a $\mathbf{x}(T)$. Lo ilustran fotos de un perro cada vez más ruidosas.
- *Reverse SDE (noise → data)*: $\mathrm{d}\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})]\,\mathrm{d}t+g(t)\,\mathrm{d}\bar{\mathbf{w}}$.
- El término recuadrado es la **score function**, el mismo objeto que en el denoiser de la slide 7.

**Qué dijo.**
- "de forma alternativa a las GAN, lo que hace es controlar un poco el proceso de cómo se va aprendiendo o cómo se va desaprendiendo. Entonces eso nos puede venir bien si sabéis el tipo de anomalía a que queréis llegar" [04:12:09]–[04:12:18].
- "incluso ignorando difusión, una cosa que se puede hacer para **explicar anomalías** es intentar empezar desde una muestra que tenéis en vuestro conjunto de datos y deteriorarla hasta que llega a la anomalía, y luego volver" [04:12:23]–[04:12:34].
- "ahora sí que controlamos el proceso" [04:12:45].
- Límite: llegar a un lobo "no va a pasar", porque "mi objetivo es llegar a ruido". Las anomalías de otro tipo no las detectaría bien, "pero anomalías de tipo ruido es muy buena idea" [04:12:53]–[04:13:11].

*Fuente: slide 17 · transcripción [04:12:02]–[04:13:11]*

---

## 13. Requisitos de cómputo y datos

- Generativos explícitos: "van a ser más caros" [03:58:53].
- Flows: "sólo funcionará si tenéis ordenadores, servidores que valgan la pena, y tenéis muchos datos; si no, no" [03:59:23].
- Deep en general: "son las mejores soluciones" *cuando* hay cómputo y datos [04:13:55]–[04:14:01].
- Tiempos medidos en los notebooks (Colab):

| Modelo | Tiempo |
|---|---|
| AE denso MNIST | ~1 s/época |
| AE conv de 52 parámetros | ~1 s/época |
| AE conv de 20 parámetros (enunciado) | 6–10 s/época |
| U-Net CIFAR-10 | 33–52 s/época |
| RBIG 8000×2 | ≈4 s |
| RBIG 333×4 | 3 s |
| RBIG 10000×10 (hasta 10000 capas) | 14 s |
| MI con RBIG en 20D | 36 s |
| RBIG 5000×144 | ≈1 min por clase (1 s/capa) |

---

## Ejercicios de los notebooks (enunciado → solución)

| Notebook | Enunciado (sin `_clase`) | Solución (`_clase`) |
|---|---|---|
| AD_AE_simple | AE denso 784→32 (relu)→784 (sigmoid). Limpieza con σ=0,5. Anomalía activa: "Negativo", error 0,868 frente a 0,011. Opciones: Zero, Max, Negativo, ×1,5, Ruido. | Añade el AE lineal "PCA casi-exacto" como referencia, usa σ=0,2 y activa "Ruido 0,5": error 0,292 frente a 0,013, un pico claro. Añade el "Patch blanco" (comentado) y marca el punto anómalo en rojo. |
| AD_AE_convolucional | Pool/Upsampling ×4 y variantes de alumnos: "Luis" (L1 + AvgPool 6) y "Samuel, Pablo y Jesús" (2 niveles 2×2 bicúbico, 20 parámetros). Error con `axis=1`. Anomalía "Negativo". | Pool 2 (código de 14×14×1, 52 parámetros), mse/mae, ruido 4D, error con `axis=(1,2,3)`. Anomalía **Patch**: error 0,034 frente a 0,006, pero al nivel de los picos normales. **El AE reconstruye el parche**, lo que muestra el límite del método. |
| UNET_convolutional_denoising | Solo existe esta versión: U-Net 16-32-64-128-256 sobre CIFAR-10, ruido σ=0,1, entrenada de ruidosa a limpia. | — (ejercicio implícito: usarla como detector con el residuo y probar otros tipos de ruido). |
| AD_rbig_demo | Toy seno + ruido: RBIG, `predict_proba`, score `1-log(p+1)^0.1`, histograma y scatter. | Añade ADBench `6_cardio.npz` (alternativa `5_campaign`) y compara el pairplot con etiquetas reales con el coloreado por score. Sin métrica. |
| rbig_demo_val | fit_transform, inversión con assert, info_loss, generación sintética y densidad. | Variantes: **penguins** (tabular 333×4, `ica`, `bound_ext=0.01`) y **CIFAR_cond** (un RBIG por clase y log-probabilidades cruzadas en train, sintéticas y test). |
| information_theory_with_RBIG | TC, H y MI analíticas (gaussianas) frente a las estimadas con RBIG. | (ya ejecutado; no hay versión sin resolver). |

**Ejercicio oral (GAN)** [04:02:49]–[04:07:49]:
1. ¿Cómo usar una GAN para detectar anomalías si no da p(x)? → **Usar el discriminador entrenado como score real/fake.**
2. ¿Qué detecta peor, una imagen con mucho ruido o un león, si se entrenó con perros y gatos? → **El león.** El discriminador nunca vio nada parecido como *fake*; las salidas ruidosas del generador sí las vio.

**Notebooks que no existen.** En el paquete **no hay notebooks de GAN, VAE, difusión ni vídeo**, aunque el profesor los menciona:
- "el de la GAN puede estar bien, o se lo pedís a Claude" [04:21:08].
- "lo mismo pero con vídeo" [04:20:39].
- Las slides 14 y 6 solo enlazan los tutoriales de TensorFlow (`cvae`, `autoencoder`).

Tampoco hay notebook de "AE + espacio latente + modelo sencillo": el profesor dice explícitamente que "no está aquí" [04:21:20].

---

## Guía del profesor para el taller (qué scripts mirar y qué tocar)

**Contexto** [04:13:16]–[04:15:31]: "sólo nos falta enseñaros los scripts, de qué cosas mirar y cuáles no … espero que los podáis utilizar para el taller más que los de ayer". Valero "ha tenido un detalle muy bonito": "si tenéis tiempo, podéis coger los que están sin solucionar, y si no tenéis tiempo, cogéis los que están solucionados" [04:14:35]–[04:14:41]; los `_clase` son las soluciones. Hay una carpeta de clásicos (`scripts_classic`) y otra de deep (`scripts_deep`) [04:14:56].

**Clásicos** (se recogen aquí por completitud; el detalle está en los documentos 01A/01B):
- `AD_Tools_Pyod` y `AD_Tools_Sklearn`: "estos 2 podéis ignorarlos hacia el final, es más que nada repasar las herramientas que tienen por defecto las librerías" [04:15:04]–[04:15:09].
- Todos "empiezan con datos aleatorios y muy sencillos" [04:15:45] y siguen la estructura **1 dimensión → varias dimensiones → imagen de Valero** [04:20:02].
- `AD_Probabilidad`: jugar "con los **criterios**", 2 o 3 sigmas, y "probar que salen cosas diferentes" [04:16:32]–[04:16:51]. También el caso 2D con matriz de covarianza, "para ver qué ocurre cuando las cosas se vuelven menos ideales" [04:16:59]–[04:17:08].
- `AD_Distancias`: "lo más importante que necesitamos es el **umbral**. Entonces, jugar con él" [04:17:42]–[04:17:47].
- Imagen real: "os he dejado una foto suya [de Valero] … la idea es añadir ruido y luego detectar las anomalías" [04:17:57]–[04:18:14].
- `AD_reconstruccion`: "cambiar el modelo de regresión lineal por alguno un poco más interesante", por ejemplo cuadrático, y comprobar que "si el modelo está mal, resulta bastante mal" [04:18:59]–[04:19:32]. "siempre eso de **visualizar** y guardar" [04:19:38]–[04:19:51].
- En resumen: "distancia → modificar la distancia; probabilidad → el umbral; y en este el modelo" [04:20:09]–[04:20:13].

**Deep:**
- "lo otro que tenéis aquí, que no sé cómo de relevante será para el taller" [04:20:15]–[04:20:23]. "aquí tenéis lo mismo pero con vídeo" [04:20:39], aunque ese notebook no está en el paquete.
- RBIG: "a Valero le gusta mucho RBIG. Entonces tenéis todos ellos" [04:20:46]–[04:20:51].
- **Recomendación principal**: "Yo lo que miraría si fuera vosotros es el de **autoencoder**; no hace falta que sea el convolucional, simplemente para ver cómo hacer muestreo con una probabilidad que viene del modelo deep" [04:20:53]–[04:21:03]. Se refiere a `AD_AE_simple(_clase)`. La transcripción dice literalmente "muestreo"; por el contexto quiere decir obtener un score de anomalía de un modelo deep.
- "si tenéis muchas ganas y mucho tiempo, el de la **GAN** puede estar bien, o se lo pedís a Claude" [04:21:08].
- "el autoencoder con vuestra idea, de hecho, no está aquí … coges la parte latente y luego utilizas las cosas sencillas" [04:21:15]–[04:21:20]. Propone implementar **AE + latente + modelo sencillo**.
- "el discriminador de la GAN: intentad ver si lo podéis **hacer fallar** … Llevará un poquito más de tiempo, pero vale la pena" [04:21:28]–[04:21:31].
- "el resto son iguales a lo que hemos visto, pero con arquitectura [distinta]" [04:21:31].

---

## Lo que dijo el profesor (recomendaciones explícitas)

1. Con anomalías etiquetadas, cualquier clasificador o regresor sirve: "la criatura/arquitectura que más os guste" [03:50:55]–[03:51:38].
2. Un AE se usa de dos formas: error de reconstrucción (la típica) o espacio latente + modelo clásico. La segunda "funciona muy bien" y da **interpretabilidad** [03:52:25]–[03:54:11].
3. Outliers ≈ ruido; lo que queda fuera de las componentes principales es ruido [03:52:55].
4. Denoising = detección de anomalías [03:54:47]. El sal y pimienta degrada más que el gaussiano [03:56:58].
5. Cualquier modelo de p(x): probabilidad baja → anomalía alta, y **sigue haciendo falta un umbral** [03:58:16]–[03:58:41].
6. Los generativos son más potentes que una gaussiana y no hacen supuestos, pero son más caros y obligan a tomar decisiones [03:58:28]–[03:58:55].
7. Flows y RBIG solo con buenos servidores y muchos datos [03:59:23]. RBIG es invertible, permite muestrear y calcular p(x) [03:59:48]–[04:00:10].
8. Los autorregresivos dan probabilidades, a menudo condicionadas, y valen igual [04:00:29]–[04:00:41].
9. La GAN no da p(x): usar el **discriminador**. Va bien con anomalías de ruido y mal con las semánticas (el león) [04:04:44]–[04:07:49].
10. Evaluar las *capabilities*: qué tipos de anomalía puede detectar tu modelo [04:10:38]–[04:10:54].
11. Difusión: controla la degradación y es útil si conoces el tipo de anomalía. Para explicar una anomalía, degradar una muestra normal hasta llegar a ella [04:12:09]–[04:12:34].
12. Deep = las mejores soluciones si hay cómputo y datos [04:13:55].
13. Taller: mirar el AE (el convolucional no hace falta), la GAN como opcional (o pedírsela a Claude), implementar AE + latente + modelo sencillo e intentar hacer fallar el discriminador [04:20:53]–[04:21:31].
14. Clásicos: tocar el criterio σ (probabilidad), el umbral (distancias) y el modelo (reconstrucción), y visualizar siempre [04:16:32]–[04:19:51].

---

## Aplicación a nuestra práctica (fraude, AUPR, 0,17 %)

*Toda la sección es nota del redactor, salvo las citas del profesor. Contexto: 170.884 filas etiquetadas (X1..X20 anonimizadas tipo PCA, Amount en otra escala, 295 fraudes = 0,17 %, 4.167 duplicadas), 113.923 filas de producción sin etiqueta, métrica AUPR y línea base GMM(5) sobre no-fraude con AUPR de test 0,567.*

### A. Receta: autoencoder tabular para `02_baselines_no_supervisados`

Se adapta `AD_AE_simple_clase` (que trabaja con imágenes en [0,1]) a 21 variables continuas:

| Decisión | Notebook del profesor (MNIST) | Nuestro caso (tabular) | Por qué |
|---|---|---|---|
| Datos de entrenamiento | Todo x_train | **Solo Class=0**, sin duplicados | El AE tiene que aprender lo "normal", como el GMM de partida. |
| Escalado | /255 | `log1p(Amount)` (si Amount≥0; comprobar en EDA) + `StandardScaler` ajustado solo con train normal (o `RobustScaler`) | Si no, el MSE lo dominan las variables de mayor escala. |
| Arquitectura | 784→32→784 | **21→16→8→16→21** (ReLU dentro, código lineal de 8) | El cuello de botella es la "restricción(Z)" de la slide 4. |
| Salida y pérdida | sigmoid + binary_crossentropy | **lineal + MSE** | Los datos no están en [0,1]. |
| Parada | 50 épocas fijas | **EarlyStopping** sobre la pérdida de una validación de **no-fraude** (patience≈10, restore_best_weights) | Evita sobreajustar el ruido. |
| Score | `error_tes = mean((x-x̂)^2, axis=1)` | El mismo MSE por fila; mayor = más fraude | Sale ya en la dirección que pide el CSV. |
| Evaluación | Mirar el pico en un gráfico | `average_precision_score` en test estratificado y CV | Es la métrica de la práctica. |

```python
import numpy as np, keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

X = df.drop(columns='Class').copy(); y = df['Class'].values
X['Amount'] = np.log1p(X['Amount'])                       # comprobar antes que Amount >= 0
X_tr, X_te, y_tr, y_te = train_test_split(X.values, y, test_size=0.2, stratify=y, random_state=0)
X_norm = np.unique(X_tr[y_tr == 0], axis=0)               # solo no-fraude, sin duplicados
X_fit, X_val = train_test_split(X_norm, test_size=0.15, random_state=0)
sc = StandardScaler().fit(X_fit)

inp = keras.Input((X.shape[1],))
h = layers.Dense(16, activation='relu')(inp)
z = layers.Dense(8, name='z')(h)                          # bottleneck lineal
h = layers.Dense(16, activation='relu')(z)
out = layers.Dense(X.shape[1])(h)                         # salida lineal
ae = keras.Model(inp, out); encoder = keras.Model(inp, z)
ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
ae.fit(sc.transform(X_fit), sc.transform(X_fit), validation_data=(sc.transform(X_val),)*2,
       epochs=300, batch_size=256, shuffle=True, callbacks=[es], verbose=0)

def ae_feats(Xraw):
    Xs = sc.transform(Xraw); R = (Xs - ae.predict(Xs, verbose=0))**2
    return R.mean(axis=1), R, encoder.predict(Xs, verbose=0)  # score, error por variable, Z

s_te, R_te, Z_te = ae_feats(X_te)
print('AUPR', average_precision_score(y_te, s_te), 'ROC', roc_auc_score(y_te, s_te))
```

**Comprobaciones obligatorias:**
1. **Referencia lineal.** El "PCA casi-exacto" del profesor equivale aquí a `PCA(n_components=8)` ajustado con no-fraude, usando su error de reconstrucción como score. Si el AE no lineal no supera a PCA ni al GMM(5) (0,567) en AUPR, no compensa.
2. **Barrido del cuello de botella** (4, 6, 8, 12) **eligiendo por AUPR en validación con fraudes**, no por la pérdida. Es la lección del parche (sección 4): un AE que reconstruye "demasiado bien" generaliza también a los fraudes. Ojo: si X1..X20 son de verdad componentes PCA, ya están decorreladas, y un código pequeño perderá información de todas las filas por igual. Por eso el barrido es necesario.
3. **Varias semillas** (3–5): un AE pequeño varía de una ejecución a otra; conviene dar la media ± desviación de la AUPR o promediar los scores.
4. **Duplicados**: quitarlos (o separarlos en bloque) antes de partir en train/val/test, para que no haya copias de una misma fila a ambos lados.

### B. Espacio latente y error por variable como features del supervisado (`04_hibrido_y_ensemble`)

- Es justo la propuesta del profesor que "no está aquí": "coges la parte latente y luego utilizas las cosas sencillas" [04:21:15]–[04:21:20], y la que según él da interpretabilidad [03:53:54]–[03:54:11].
- **Variante no supervisada:** GMM, IsolationForest o Mahalanobis sobre $Z$ (8 dimensiones) en lugar de sobre $X$.
- **Variante híbrida:** al clasificador supervisado se le dan $[X,\ \text{error total},\ R_{1..21},\ Z_{1..8}]$: regresión logística si se busca interpretabilidad, o gradient boosting si se busca rendimiento. El vector $R_j=(x_j-\hat x_j)^2$ indica qué variables "no encajan" en cada transacción, y sirve tanto de feature como de explicación.
- **Sin fuga de información:** el AE (con su scaler) se ajusta **dentro de cada fold**, solo con los no-fraudes del train de ese fold. Las features de validación y test se calculan con ese AE (*out-of-fold*). Para producción se reentrena con todos los no-fraudes de construcción.
- Para el ensemble, mirar la **correlación de rangos** (Spearman) entre el score del AE y los demás. Si es baja y el AE tiene una AUPR razonable, aporta diversidad aunque por sí solo no gane.

### C. Supervisado $A=f(x)$ (`03_supervisados`)

El profesor dice que con anomalías etiquetadas vale cualquier arquitectura [03:51:15]–[03:51:38]. Con 295 positivos y 21 variables tabulares, una MLP con `class_weight` es viable, pero lo esperable es que no supere a gradient boosting. Tiene sentido como miembro más del ensemble.

### D. Cuándo NO compensa el deep aquí

- **Datos tabulares de pocas dimensiones** (21 variables, en buena parte ya transformadas tipo PCA). La ventaja del deep está en estructura espacial o secuencial (imágenes, vídeo, texto), que aquí no existe. En tabular, los modelos de árboles suelen ganar a las redes.
- **Pocas anomalías etiquetadas** (295): una red supervisada sobreajusta con facilidad; el boosting con validación cruzada suele ser más robusto.
- **El profesor lo advierte:** flows, GAN y difusión solo compensan "si tenéis … servidores que valgan la pena, y tenéis muchos datos" [03:59:23]. Además, el discriminador de una GAN solo detecta anomalías del tipo que vio como *fake* (el ruido), y el fraude se parece más al "león" que al ruido.
- **Regla de decisión:** incluir un modelo deep solo si (a) mejora la AUPR en CV por encima de la desviación entre folds respecto al mejor modelo no deep (GMM, IsolationForest, PCA, boosting), o (b) aporta diversidad demostrable al ensemble. En cualquier otro caso basta con mencionarlo en la reflexión final.
- **Prioridad orientativa por coste/beneficio:**
  1. AE denso (barato, minutos en CPU; para 02 y 04).
  2. RBIG (coste bajo-medio; alternativa directa al GMM, con el riesgo de los empates).
  3. MLP supervisada (barata, probablemente peor que boosting).
  4. VAE (coste medio).
  5. GAN y difusión (caras; no recomendadas).

### E. RBIG como estimador de densidad alternativo al GMM(5)

Es el mismo esquema que la línea base (ajustar una densidad con no-fraude y usar score = $-\log p$), pero sin suponer una mezcla de gaussianas. Se usa la API de los notebooks:
```python
from rbig import RBIG
Xs_norm = sc.transform(X_norm)
idx = np.random.RandomState(0).choice(len(Xs_norm), size=min(30000, len(Xs_norm)), replace=False)
rb = RBIG(rotation='pca', max_layers=1000)      # probar también 'ica' (penguins) y distintos bound_ext
rb.fit(Xs_norm[idx])                            # submuestreo: medir el tiempo antes de subir el tamaño
p = rb.predict_proba(sc.transform(X_te))
logp = np.log(np.clip(p, 1e-300, None))         # LOG-probabilidad; nunca log(p+1) (ver 9.7)
score_rbig = -logp
print('suelo:', (p <= 1e-300).mean())           # porcentaje de puntos empatados en el suelo
# deshacer empates con el GMM: ordenar por score_rbig y, dentro de cada empate, por -gmm.score_samples
from scipy.stats import rankdata
score = rankdata(score_rbig) + rankdata(-gmm.score_samples(sc.transform(X_te))) / (len(X_te) + 1)
print('AUPR RBIG', average_precision_score(y_te, score))
```
- Comparar en AUPR con el GMM con distintos `n_components` y con el mismo split. Si RBIG no mejora, se descarta y se documenta.
- Coste: 5000×144 tardó ~1 min y 10000×10 unos 14 s. Con ~170k×21 conviene empezar con 20–50k no-fraudes.
- Se puede añadir `-logp` de RBIG como feature del 04 (calculado *out-of-fold*).
- Como herramienta opcional de EDA (`01_EDA`): `total_correlation()` o `MutualInfoRBIG` entre bloques de variables, para justificar decisiones de selección.

### F. Advertencia de integridad académica

- 170.884 + 113.923 = 284.807 filas, el tamaño del conocido dataset público de fraude con tarjetas (ULB/Kaggle). Además, la descarga de ADBench del notebook de RBIG incluye `Classical/13_fraud.npz`.
- **Usar etiquetas públicas, o ese dataset, para puntuar o ajustar algo sobre producción sería hacer trampa y no se debe hacer.**
- El trabajo tiene que salir solo de los datos de construcción entregados; ADBench no se usa en esta práctica.

---

## Cobertura de fuentes

| Fuente | Rango | Cubierto | Notas |
|---|---|---|---|
| slides_deep/p01.png | Portada | Sí | Título, autor Valero Laparra e imagen decorativa. |
| p02.png | INTRO | Sí | Sección 0. |
| p03.png | Modelos predictivos | Sí | Sección 1, con la leyenda del diagrama de segmentación. |
| p04.png | Autoencoders | Sí | Sección 2. |
| p05.png | Deep AE | Sí | Sección 3 (10-8-6-3-6-8-10). |
| p06.png | CNN AE | Sí | Sección 4, con las dimensiones y la URL de TF. |
| p07.png | Denoising based | Sí | Sección 5 (Tweedie, x−y≈Dp(x)). |
| p08.png | CNN AE – UNET | Sí | Sección 6. |
| p09.png | Lista de generativos | Sí | Secciones 0 y 7. |
| p10.png | Explícitos | Sí | Sección 7. |
| p11.png | Normalizing Flows | Sí | Sección 8. |
| p12.png | RBIG | Sí | Sección 9: el exponente −1 está en la slide y **es una errata** (verificado con la teoría y el artículo de 2011). |
| p13.png | Autorregresivos | Sí | Sección 10.1. |
| p14.png | VAE | Sí | Sección 10.2, con las URLs. |
| p15.png | Implícito | Sí | Sección 11. |
| p16.png | GANs | Sí | Sección 11. |
| p17.png | Diffusion | Sí | Sección 12. |
| slides_clasicas/p32.png | Teoría de la información (RBIG) | Sí | Cabecera y sección 9 (arXiv 2010.03807, coautor Santos-Rodríguez). |
| transcripción 2219–2235 | [03:50:01]–[03:51:38] | Sí | Intro y supervisado; las 3 primeras líneas son ruido sin contenido técnico. |
| transcripción 2236–2262 | [03:51:46]–[03:54:37] | Sí | AE, latente, deep y conv. |
| transcripción 2263–2288 | [03:54:43]–[03:57:29] | Sí | Denoising y ruido (la parte visual en vivo no se puede recuperar). |
| transcripción 2289–2330 | [03:57:34]–[04:01:14] | Sí | Explícitos, flows, RBIG, autorregresivos y VAE. |
| transcripción 2331–2431 | [04:01:17]–[04:11:57] | Sí | GAN, ejercicio, ruido frente a león, capabilities. |
| transcripción 2432–2455 | [04:12:02]–[04:14:01] | Sí | Difusión y resumen. |
| transcripción 2456–2527 | [04:14:06]–[04:21:42] | Sí | Guía de scripts; [04:21:03] corregido a la cita literal. |
| transcripción 2528–2534 | [04:21:42]–[04:22:16] | Sí | Cierre, sin contenido técnico. |
| transcripción (cruce) | [02:21:15], [02:55:47]–[02:55:55] | Sí | Valero y la información; cuantización vectorial en el latente del AE. |
| AD_AE_simple.md | Celdas 0–26 | Sí | Figura 25. |
| AD_AE_simple_clase.md | Celdas 0–32 | Sí | Figuras 15, 20, 28 y 31 verificadas. |
| AD_AE_convolucional.md | Celdas 0–32 | Sí | Variantes de alumnos; salidas copiadas. |
| AD_AE_convolucional_clase.md | Celdas 0–35 | Sí | Figuras 24, 30 y 34 verificadas: el parche se reconstruye y no destaca. |
| UNET_convolutional_denoising.md | Celdas 0–13 | Sí | Figura 12. |
| AD_rbig_demo.md / _clase.md | 0–10 / 0–21 | Sí | Figuras 08, 09 y 20. |
| rbig_demo_val.md | Celdas 0–22 | Sí | Figuras 07, 12, 16 y 20. |
| information_theory_with_RBIG.md | Celdas 0–28 | Sí | TC, H y MI; sin KL. |
| rbig_demo_val_clase_CIFAR_cond.md | Celdas 0–53 | Sí | **Las 21 figuras revisadas** (22, 25, 30, 31, 32, 37–40, 44, 46, 50, 52 descritas). |
| rbig_demo_val_clase_penguins.md | Celdas 0–35 | Sí | Figuras 13, 15, 26, 28, 29 y 33 revisadas; 33 y 34 son obsoletas. |

**Dudas resueltas en esta revisión:**
- El exponente −1 de RBIG es una errata (sección 9).
- El parche del AE convolucional no se detecta, y se explica la lección (sección 4).
- Los empates por p≈0 en RBIG y cómo evitarlos (9.7 y E).
- Faltan los notebooks de GAN, VAE, difusión y vídeo (sección de ejercicios).
- Advertencia de integridad sobre ADBench y el dataset público (F).

**Dudas pendientes:**
- No se sabe qué artículo de GAN de 2017 cita el profesor, que dice "NeurIPS".
- No se ha comprobado si la librería `rbig` tiene un método de log-densidad (revisar con `dir()`).
- No se pueden recuperar las imágenes del ejemplo en vivo de tipos de ruido [03:55:35]–[03:56:45].
- La frase sobre el signo del score supervisado [03:51:33] está cortada.
- La receta tabular de la sección A no está ejecutada: sus hiperparámetros son un punto de partida que hay que validar con AUPR.
