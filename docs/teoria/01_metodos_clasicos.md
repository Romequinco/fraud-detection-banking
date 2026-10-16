# Métodos clásicos de detección de anomalías

Documento en dos partes, cada una con sus propias secciones de fuentes, ejercicios, recomendaciones
del profesor, aplicación a la práctica y cobertura:

- **Parte A: distancias, probabilidad y teoría de la información** (slides p10–p32, clase
  [01:02]–[02:33], notebooks `AD_Data`, `AD_Distancias`, `AD_Probabilidad`). Incluye las
  **baselines medidas en nuestros datos**.
- **Parte B: clasificación de una clase, regresión/reconstrucción y algoritmos clásicos** (K-means,
  VQ, DBSCAN, LOF, Isolation Forest, GMM/EM, PyOD) (slides p33–p48, clase [02:33]–[03:07],
  notebooks `AD_reconstruccion`, `AD_Tools_Sklearn`, `AD_Tools_Pyod`).

Las referencias "slide N", "[hh:mm:ss]" y "notebook X celda N" remiten a `docs/_fuentes/` (no
versionado; ver su `INVENTARIO.md`).

## Parte A — distancias, probabilidad y teoría de la información

> Fuentes:
> - Diapositivas `slides_clasicas/p10.png` … `p32.png` (Deteccion_de_anomalias.pdf), vistas una a una como imagen.
> - `transcripcion.txt` líneas 600–1384 ([01:02:03] – [02:33:02]). Nota: las líneas 1089–1187 ([01:52:22] – [02:15:46]) son el descanso (ruido de transcripción sin contenido).
> - `notebooks/AD_Data.md` y `AD_Data_clase.md` (+ figuras `AD_Data_clase_figs/celda11_1, celda12_2, celda14_3, celda15_4`).
> - `notebooks/AD_Distancias.md` y `AD_Distancias_clase.md` (+ figuras `AD_Distancias_clase_figs/celda05_1 … celda36_9`).
> - `notebooks/AD_Probabilidad.md` y `AD_Probabilidad_clase.md` (+ figuras `AD_Probabilidad_clase_figs/celda04_1 … celda40_14`).
>
> Convenciones: "PROF" = profesor de la sesión. Las slides y notebooks son de Valero Laparra. El ponente es Raúl Santos-Rodríguez (Bristol): dice que el paper de la slide 32 es "un artículo que también hice yo con él" [02:31:18], y él figura como coautor en esa slide. Además, cuenta su experiencia corrigiendo en Inglaterra. Los errores fonéticos se han interpretado: "una playa / off layer / outlife / offlider / aulayer" = outlier; "majadanovis / maharanovis / mahalhanovis / más a la nube" = Mahalanobis; "gaunciana / lousiana / louisiana / gausiana / gauskeana / bossiana / oxiana" = gaussiana; "callen / Curlback Lightner" = Kullback-Leibler; "Globe Shannon" = Claude Shannon; "Grid / Great World / create World" = grid world; "S.V.M / sevm" = SVM; "multilayer perceptual" = multilayer perceptron; "rainformea learning" = reinforcement learning; "crowcentropy" = cross-entropy; "pivalliu / ivario" = p-value; "aleatórica / aleatólica" = aleatoric; "camarero" (probablemente "con Valero").

---

### 0. Mapa del bloque (tres "frases para recordar")

**Idea.** El profesor estructura los métodos clásicos en tres familias, cada una con una frase guía y un "por defecto":

| Familia | Frase | Elección por defecto | Cómo se decide anomalía |
|---|---|---|---|
| Distancias | "A mayor distancia, más anómalo" (slide 11) | Euclídea | distancia (al centroide / vecino) + umbral |
| Probabilidad | "A menor probabilidad, más anómalo" (slide 19) | Gaussiana | nº de sigmas (95 % / 99 %) |
| Teoría de la información | "Información vs Anomalía" (slide 27) | Entropía de Shannon | entropía / información mutua / KL + umbral |

Resumen literal tras el descanso: "yo defino la distancia que yo quiero, umbral: la distancia al centro de la distribución y listo. Con probabilidad asumo que es gaussiana, decido cortar sigmas y listo, o con teoría de la información, que es donde las cosas se nos complican un poco" [02:20:33]–[02:20:46]. "Si en distancia se utiliza mucho la euclídea, en probabilidad se utiliza mucho la gaussiana, en entropía se utiliza sólo, el 99,9 % de los casos, esta entropía [Shannon]" [02:26:31]–[02:26:39].

Antes de empezar recalca que "casi cualquier clasificador de los que conocéis por el momento os valdrá para esto [...] con pequeñas modificaciones" [01:01:52]–[01:02:03].

*Fuente: slides 10, 11, 18, 19, 26, 27 · transcripción [01:02:03], [02:16:32]–[02:20:46], [02:26:31]*

---

### 1. "A mayor distancia, más anómalo" y sus excepciones

**Idea.** La regla básica de la familia de distancias: cuanto más lejos esté una muestra de los datos "normales", más probable es que sea outlier. Pero "no siempre es verdad a más distancia más anómalo" [01:02:22].

**Detalle / excepciones discutidas por el profesor.**

1. **Pocas muestras → clases mal caracterizadas** [01:02:46]–[01:04:39]. Ejemplo: el clasificador más sencillo posible con sólo 2 muestras, una de cada clase. "Cuando tengo muy pocas muestras, tengo muy mal caracterizadas mis clases" [01:02:58]. Remite al ejemplo del día anterior (una sola foto de una especie de nenúfar: "esa categoría la conozco regular [...] puedo extraer pocas conclusiones sobre sus propiedades" [01:03:07]). Llega una tercera muestra sin etiqueta ("muy habitual en outliers no saber a qué clase pertenecen y no estar etiquetados" [01:03:24]). Según la regla, la más lejana sería más outlier, pero como no se sabe qué forma tendrá la clase con más datos (dibuja dos posibles formas en la pizarra), puede que lo que parece outlier no lo sea o al revés. Conclusión: **"antes de empezar a pensar en distancias, tenéis que pensar si los datos que tenéis representan bien el concepto/categoría que queréis aprender. Si la respuesta es no, lo de la distancia es un poco irrelevante"** [01:04:29]–[01:04:39]. Si en cambio la estructura se entiende bien aunque haya pocos datos ("la cantidad de datos que necesito tendrá que ver con la complejidad de la estructura de datos" [01:04:47]), p. ej. todo bien caracterizado dentro de una pequeña esfera en 2D, entonces sí: "aunque la distancia sea pequeña [...] cuanto más lejos, más probable que sea un outlier" [01:05:04]–[01:05:19].

2. **Frontera vs lejanía (dos puntos a la misma distancia d)** [01:05:28]–[01:08:46]. Coloca adrede dos puntos a igual distancia $d$ de una clase: uno cae **entre las dos categorías (sobre la frontera)** y otro **lejos de ambas**. ¿Cuál es más anómalo? "Depende de vuestras asunciones" [01:06:07].
   - Argumento A: el que está entre categorías tiene "una pequeña probabilidad de que pertenezca a esta o a esa", está sobre la frontera ("estoy muy indeciso sobre él, pero creo que pertenece al problema que estoy tratando"); el otro está lejos de ambas clases → "tiene pinta de ser más claramente un outlier" [01:06:14]–[01:07:04].
   - Argumento B (**SVM / MLP**): "a más distancia de la frontera, más seguro estoy que pertenezco a la clase X"; "una SVM asume que, de hecho linealmente, cuanto más lejos estoy de la frontera menos me tengo que preocupar" [01:07:07]–[01:07:34]. Ejemplo gato/perro: un punto muy lejos de la frontera del lado perro "es muy poco gato, entonces tengo mucha confianza de que eso va a ser un perro" [01:07:41]–[01:07:49].
   - Moraleja: "he sido capaz de mencionar que seguramente es un outlier y seguramente no es un outlier dependiendo de qué es lo que he asumido" [01:07:53]. Modelos como el MLP o la SVM asumen "que todo lo que está aquí, aunque no tenga muestras, no son outliers, está todo bien. Pero eso nos puede llevar a problemas [...] si tengo una muestra de este lado que está muy lejos de la frontera, a lo mejor es un ejemplo de otra clase que no he visto nunca" y se clasificaría con mucha confianza como "perro" [01:08:23]–[01:08:46].

3. **Incertidumbre aleatoria (aleatoric) vs epistémica** [01:09:04]–[01:13:20] (lo introduce porque "está relacionado todo con lo mismo").
   - *Aleatoria*: en la frontera, $P(\text{círculo}\mid x) = 0{,}5$. "Por muchas muestras que yo siga cogiendo [...] pido el doble y seguiré teniendo la misma duda sobre la frontera" → no se reduce con más datos [01:10:19]–[01:11:44].
   - *Epistémica*: incertidumbre en zonas del espacio sin datos (lejos). "Es algo que en realidad sí puedo solucionar teniendo más muestras" (10×, 20×, 30×) [01:11:48]–[01:12:12]. Ejemplo de la gaussiana de alturas: una muestra en la cola "querría decir que esta persona mide 2 m y medio, pero si tengo miles de millones de personas en el mundo aparecerá alguien que llegará hasta ahí" [01:12:44]–[01:13:00].

4. **Problema de densidad** (lo retoma en probabilidad, ver §9): ni distancias ni probabilidades detectan una anomalía que esté rodeada por datos normales (nueva subclase en zona de alta densidad) [01:39:02]–[01:40:01].

*Fuente: slide 11 · transcripción [01:02:22]–[01:13:20]*

---

### 2. ¿Qué distancia? Euclídea (L2, RMSE)

**Idea.** "Qué definición de distancia, qué quiere decir distancia" [01:13:41]. Por defecto se usa la euclídea, "pero no tiene por qué" [01:13:50].

**Detalle / fórmulas (slide 12).** Diagrama de dos puntos $p=(p_1,p_2)$, $q=(q_1,q_2)$ con el triángulo rectángulo de catetos $q_1-p_1$ y $q_2-p_2$:
$$d(p,q)^2 = (q_1-p_1)^2 + (q_2-p_2)^2,\qquad d(p,q)=\sqrt{\textstyle\sum_i (q_i-p_i)^2}=\|p-q\|_2$$
Título de la slide: "Distancia Euclídea (L2, RMSE)".

**Ejemplos / comentarios del profesor.** "Cojo P, Q, sus 2 componentes, tomo la diferencia, elevo al cuadrado y tengo una distancia" [01:14:37]. Lo relaciona con el error cuadrático visto el día anterior: "tengo el valor que quiero estimar, el valor real, tomo la diferencia, elevo al cuadrado" [01:14:46]–[01:14:56]. Ventaja: "la euclídea es diferenciable, muy sencilla, y a lo mejor en la Manhattan tengo más problemas" [01:18:33]. "Si no sé nada, todo el mundo utiliza distancia euclídea para todo" [01:17:46].

*Fuente: slide 12 · transcripción [01:13:41]–[01:15:03], [01:17:46], [01:18:33]*

---

### 3. Distancia Manhattan (L1, MAE) — ejemplo Pac-Man

**Detalle / fórmulas (slide 13).** Figura de una cuadrícula (calles) con varios caminos entre dos esquinas: rojo, azul y amarillo (escalonados, todos de igual longitud L1) y verde (diagonal, euclídea).
$$d_1(\mathbf p,\mathbf q)=\|\mathbf p-\mathbf q\|_1=\sum_{i=1}^{n}|p_i-q_i|$$

**Cuándo usarla.** "Normalmente Manhattan, o cualquiera de las otras, se eligen cuando los datos en sí aparecen cuantizados, que quiere decir que tienes algo que no es continuo" [01:15:43]–[01:15:53]. "Si estoy hablando de distancias en un grid, no tiene mucho sentido calcular la distancia cuadrática a lo Euclides y tiene más sentido contar" [01:16:01]–[01:16:11].

**Ejemplo del profesor: Pac-Man** [01:16:19]–[01:18:48]. Pac-Man vive en un grid world con posiciones libres y bloqueadas (paredes), fantasmas y premios (cerezas). "Si yo quiero saber cuál es mi distancia a las cerezas, no me interesa calcular la distancia euclídea, sino el número de pasos que tengo que dar a través de este grid world para llegar hasta ahí, y en función de qué movimientos estén permitidos" [01:16:52]–[01:17:15]; las paredes "me requieren crear una distancia que esté ajustada a ella" [01:17:23].

Tres ideas que quiere que retengamos [01:17:31]–[01:17:46]:
1. Usar una distancia relacionada con mi problema.
2. Aunque no esté relacionada con el problema, a lo mejor me vale también.
3. Si no sé nada, todo el mundo usa la euclídea.

"Normalmente hay una correlación entre las 2: a mayor distancia euclídea, mayor distancia Manhattan, pero no tiene por qué" [01:18:10]. "Muchas veces por problemas de optimización se utiliza una de las 2 porque es más cómoda, sabiendo que estás pagando un precio si no estás optimizando lo que tienes que optimizar" [01:18:19]–[01:18:30]. En reinforcement learning "la gente utiliza Manhattan un montón porque son problemas tipo juegos [...] posiciones discretas entre las que me puedo mover" [01:23:42]–[01:24:05].

*Fuente: slide 13 · transcripción [01:15:03]–[01:18:53], [01:23:42]–[01:24:05]*

---

### 4. Distancia Lp (Minkowski), L∞, p<1 y "L0"

**Detalle / fórmulas (slide 14).**
$$\|x\|_p=\left(|x_1|^p+|x_2|^p+\cdots+|x_n|^p\right)^{1/p},\qquad d_p(\mathbf p,\mathbf q)=\|\mathbf p-\mathbf q\|_p$$
Figura: bolas unidad en 2D para $p=1$ (rombo, morado), $1{,}5$, $2$ (círculo, azul), $3$, $6$ y $\infty$ (cuadrado, rojo), en ejes $[-1,1]$.

**Slide 15.** Tres casos: "Distancia ∞ (L∞)", "Distancia p<1" (con dibujo de una bola unidad en forma de astroide/estrella de 4 puntas, cóncava) y "Distancia p=1 (L0)". (Revisado sobre la imagen: la slide pone literalmente "Distancia p=1 (L0)", que es una errata; la "L0" corresponde al límite $p\to 0$, que cuenta el número de componentes distintas de cero, $\|x\|_0=\#\{i: x_i\neq 0\}$. $L_\infty=\max_i |x_i|$.)

**Explicación del profesor** [01:19:01]–[01:23:28].
- Si $p=2$ → euclídea [01:19:39]. Si $p=1$ → "estoy simplemente sumando los valores absolutos de las componentes" (relacionado con el MAE "que veíamos ayer") [01:19:52]–[01:20:20].
- Si $p=\infty$: un alumno responde "máximo" y el profesor confirma [01:21:07]–[01:21:09]. "p puede tomar cualquier valor real entre 0 e infinito, pero sólo se usan 3 valores en la práctica [...] el 1, el 2 y el infinito" [01:21:11]–[01:21:23].
- "Si utilizáis el infinito va a escoger el valor máximo de todos [...] y el resto los va a llevar a 0. Eso es muy útil cuando queremos reducir la cantidad de componentes que guardamos en un vector, sólo nos quedamos con el más importante [...] puede ser una aproximación muy mala en muchos casos, pero a veces ayuda mucho" [01:21:31]–[01:21:46]. Ejemplo del día anterior: vector de películas vistas por una persona, dominado por 2–3 valores y el resto ceros; "este tipo de distancias [...] van a llevar ese comportamiento hasta el extremo" [01:21:54]–[01:22:06]. "Es bueno para simplificar cosas y es bueno desde el punto de vista de optimización. Tanto el 1 como el 2 como el infinito funcionan bastante bien" [01:22:27]. (Nota del redactor: la frase "el resto los lleva a 0" describe que L∞ sólo "mira" la componente máxima e ignora las demás; la promoción de esparsidad propiamente dicha se asocia a $p\le 1$.)
- **p como hiperparámetro**: "pocas veces en la vida, pero a veces se hace [...] hyperparameter tuning sobre esa p. Se suele empezar en 1 y se sabe que entre 1 y 2 puede haber algo interesante; a veces se utiliza p=1,2, p=1,3, pero es raro" [01:22:39]–[01:23:02]. En una red neuronal la función de pérdida (por defecto cross-entropy o cuadrática tipo euclídea) "es un parámetro que está ahí para que juegues con él [...] pero nadie en su sano juicio nunca o casi nunca lo tocaría"; según la que elijas "va a preferir unas soluciones" [01:23:02]–[01:23:28].
- Resumen: "fuera de eso [RL], veréis que distancia L2, L1 y el infinito son los que se usan en casi todo. Pero pequeño problema para nosotros en detección de anomalías es que ese tipo de distancias no representan todo lo que puede ser útil en nuestros casos" [01:24:07]–[01:24:29].

**Código del profesor** (AD_Distancias_clase, celdas 9 y 11): matriz de distancias Lp entre los 3 puntos inventados `A = [[1,2],[3,1],[-1,2]]`.
```python
# celda 9: prueba de una Lp con p<1 entre dos diferencias (2 y 0)
p= 0.1
(2**p + 0**p)**(1/p)
# salida mostrada: 1456.8762338020144

# celda 11: matriz 3x3 de distancias Lp entre los puntos de A para varios p
for p in [0.00001, 0.5, 1, 2, 3, 10000000]:   # p≈0 ("L0"), 0.5, L1, L2, L3, p≈inf
    print('p = ' + str(p))
    M = np.zeros((3,3))
    for n in range(A.shape[0]):
        for m in range(A.shape[0]):
            M[m,n] = (np.abs(A[n,0]-A[m,0])**p+np.abs(A[n,1]-A[m,1])**p)**(1/p)
    print(M)
```
Salidas: p=1e-5 → `[[0,inf,2],[inf,0,inf],[2,inf,0]]` (con RuntimeWarning de overflow); p=0.5 → `[[0,5.83,2],[5.83,0,9],[2,9,0]]`; p=1 → `[[0,3,2],[3,0,5],[2,5,0]]`; p=2 → `[[0,2.236,2],[2.236,0,4.123],[2,4.123,0]]`; p=3 → `[[0,2.080,2],[2.080,0,4.021],[2,4.021,0]]`; p=10000000 → `[[0,1,0],[1,0,1],[0,1,0]]`.

(Nota del redactor sobre estas salidas: (a) la salida de la celda 9 (1456.88) corresponde a $(2^{0.1}+1^{0.1})^{10}$, no a $(2^{0.1}+0^{0.1})^{10}=2$; probablemente el código se editó después de ejecutarse. (b) Con $p\to 0$ la fórmula diverge (∞) cuando las dos coordenadas difieren y da 2 cuando sólo difiere una: muestra que $p<1$ dispara las distancias cuando hay varias componentes distintas. (c) La salida con p=10000000 **es errónea por desbordamiento de enteros**: `A` es de enteros y `p` es un int, así que `2**10000000` en int64 da 0; la L∞ real sería `[[0,2,2],[2,0,4],[2,4,0]]`. Para aproximar L∞ con números usar `p=np.inf` → `np.max(np.abs(...))`, o convertir `A` a float.)

**Figura celda 32 (versión _clase, código no conservado)**: "Dibuja los valores en 2D que distan 1 del centro": cuadrado azul relleno de puntos en $[-2,2]^2$ y, en rojo, los puntos a distancia ≈1 del origen para un $p<1$, que forman la **estrella/astroide de 4 puntas** de la slide 15.

*Fuente: slides 14, 15 · transcripción [01:19:01]–[01:24:29] · AD_Distancias_clase celdas 9, 10, 11, 31, 32*

---

### 5. Distancia (o similitud) coseno — bag of words

**Detalle / fórmulas (slide 16).**
$$\cos(\theta)=\frac{\mathbf A\cdot\mathbf B}{\|\mathbf A\|\,\|\mathbf B\|}$$
Figura: vectores $A(x_1,y_1)$ y $B(x_2,y_2)$ desde el origen, ángulo $\theta$ entre ellos (azul) y distancia euclídea $d$ entre los extremos (rojo). (Nota del redactor: la distancia coseno habitual es $1-\cos\theta$.)

**Explicación del profesor** [01:24:29]–[01:27:20]. "Añade un aspecto que es muy útil en muchos casos: quiero mirar la distancia entre A y B; lo que puedo es estudiar cuál es este ángulo que los separa" [01:24:39]–[01:24:56]. Dos puntos que en euclídea/Manhattan están a distancia 1 pueden ser "mucho más cercanos" si el ángulo es muy pequeño; "incluso en el extremo, aunque hubiera otro punto aquí y esta distancia fuera también de 1, quiero tener alguna distancia que me diga que la distancia entre estos 2 puntos sea 0, porque los 2 vectores están alineados" [01:25:29]–[01:26:17].

**Ejemplo: texto / bag of words** [01:26:20]–[01:26:51]. Bag of words = contar cuántas veces aparece cada palabra y crear un vector de frecuencias. "Si 2 vectores son proporcionales porque tienen la misma frecuencia de ocurrencia de las palabras, pero simplemente uno es más largo que el otro [texto más largo], lo que voy a querer es que una distancia tipo coseno me diga que en realidad están muy cerca, mientras que una euclídea me diría que están muy lejos" [01:26:37]–[01:26:51]. Cálculo: producto escalar "A por B entre sus módulos me va a decir cuál es el coseno del ángulo que los separa" [01:27:01]–[01:27:11].

*Fuente: slide 16 · transcripción [01:24:29]–[01:27:20]*

---

### 6. Distancia de Mahalanobis — ejemplo altura/peso

**Detalle / fórmulas (slide 17).**
$$d_M(\vec x,\vec y;Q)=\sqrt{(\vec x-\vec y)^{\mathsf T} S^{-1}(\vec x-\vec y)}$$
con $S$ = **matriz de covarianza**. Figura: a la izquierda, espacio "nativo" con una nube de puntos correlacionada (negativamente) y elipses a $\sigma$, $2\sigma$, $3\sigma$ a lo largo de los ejes principales ($\sigma_1$, $\sigma_2$); flecha "$\times S^{-1}$"; a la derecha, "Euclidean space": los mismos datos transformados, con círculos concéntricos $\sigma_0$, $2\sigma_0$, $3\sigma_0$.

**Explicación del profesor** [01:27:20]–[01:30:30]. "Mahalanobis está guay cuando sabéis algo sobre las características de vuestros datos" [01:27:25].
- **Ejemplo altura/peso**: representar en 2D a todas las personas de clase por altura y peso. Altura ≈ 1,80 ± 10 (cm); peso ≈ 75 kg ± 20, "más variabilidad y en otra escala diferente" [01:27:39]–[01:28:04]. "Si simplemente tomase la distancia euclídea entre 2 personas, a lo mejor eso está dominado por la altura, porque está a una escala mucho más grande que el peso, o a lo mejor está dominado por el peso porque tiene una varianza mucho más grande" [01:28:15]–[01:28:29]. Mahalanobis tiene en cuenta ambos factores y "de alguna manera normaliza las variables" [01:28:33].
- "Si yo quito la S, lo que tengo es básicamente la distancia euclídea" (S = identidad ↔ misma varianza en las dos variables, círculos concéntricos). Si S no es la identidad tengo elipses "y los ejes de la elipse me están diciendo cuál es la varianza de mis datos" [01:28:40]–[01:29:17].
- **Tres opciones válidas** [01:29:23]–[01:29:46]: (1) euclídea en el espacio ya blanqueado (derecha); (2) Mahalanobis en el espacio nativo (izquierda); (3) **normalizar primero los datos y usar euclídea**. La cuarta (desnormalizar para luego usar Mahalanobis) "no tendría ningún sentido".
- Advertencia: "acordaos que la frase única era a mayor distancia más anómalo, pero no tengo ni idea de qué distancia estoy utilizando. **Si estoy utilizando la euclídea en un sitio donde debería utilizar Mahalanobis, estoy ordenando mis puntos mal en el espacio**" [01:30:09]–[01:30:24].
- Más adelante: Mahalanobis "de alguna manera es como un puente entre distancias y probabilidades, porque entiende información de la probabilidad [la varianza]" [01:48:21].

**Ejercicio relacionado (AD_Distancias, "Datos a partir de una imagen")**: calcular la distancia de Mahalanobis entre dos patches 8×8×3 (D=192) de una imagen (ver §Ejercicios). Sin solución en la versión _clase.

*Fuente: slide 17 · transcripción [01:27:20]–[01:30:30], [01:48:21] · AD_Distancias_clase celda 34*

---

### 7. Distancia "a qué": centroide, vecino más cercano, media de distancias

**Idea.** "Una última cosa sobre distancias: cómo aplicaríamos esto prácticamente en un algoritmo" [01:30:38]. Tenemos 3 puntos que representan nuestra clase y nos dan un punto nuevo; "el algoritmo de detección de novedad o de anomalía va a mirar distancias, pero lo que no os he dicho es la distancia ¿a dónde?" [01:30:58]–[01:31:15].

**Opciones enumeradas** [01:31:23]–[01:32:57]:
- **Distancia al centro (media/centroide) de la clase**: "si esto fuera clustering y ya tengo calculado el centro, es sólo calcular una distancia"; si no, "hacer la media de estos 3 puntos [...] y la distancia luego" [01:31:53]–[01:32:04].
- **Media de las distancias** a cada uno de los puntos [01:32:12].
- **Distancia más corta o más larga** ("serán igual de informativos, depende de cuáles sean mis datos") [01:32:22]–[01:32:27].
- **Vecino más cercano**: "una de las más típicas es intentar buscar el punto más cercano dentro de la clase y esa es la distancia que utilizo" [01:32:29]–[01:32:36].
- Cualquier combinación, con cualquiera de las distancias vistas: "es decisión vuestra cuando diseñáis este tipo de sistemas" [01:32:18], [01:32:40].
- "Normalmente, si a mí me dan estos 3 puntos, elijo **el centroide o el más cercano**. Es lo más típico" [01:32:44]–[01:32:49].

**Código del profesor** — el notebook usa la variante "distancia al vecino más cercano" (mínimo por filas de la matriz de distancias test×referencia):
```python
# AD_Distancias_clase celda 19-20 (1D): distancia L1 de cada valor test al ejemplo más cercano
p = 1
M = np.zeros((y.shape[0],y.shape[0]))
for n in range(y.shape[0]):          # n recorre los "valores ejemplo" (referencia)
    for m in range(y.shape[0]):      # m recorre los "valores test"
        M[m,n] = (np.abs(y[n,0]-y_hat[m,0])**p)**(1/p)
mmm = np.min(M,axis=1)               # score de anomalía = distancia al vecino más cercano
```

*Fuente: transcripción [01:30:38]–[01:32:57] · AD_Distancias_clase celdas 19–22, 27–30*

---

### 8. El umbral: fijo, por histograma, explicabilidad

**Idea.** "Y aún así me falta una cosa, que es un umbral" [01:32:57]. "Si estuviéramos en una clase de estadística en el año 60 [...] se solucionaría en 5 min": "voy a decir que mi umbral es esto y a partir de ahí todo es una novedad o una anomalía. Y eso se sigue haciendo a día de hoy" [01:33:01]–[01:33:21].

**Detalle.**
- **Umbral fijo**: p. ej., tomar "la distancia desde el punto más lejano al centro" como referencia $D$: "todo lo que pase de ahí es una anomalía" [01:33:24]–[01:33:42].
- **Consejo poco habitual pero útil**: "recordar cuál fue la distancia que te llevó a decir que esto era una anomalía, para luego categorizarlas más adelante" [01:33:48]–[01:33:51] (es decir, guardar el score, no sólo la etiqueta binaria).
- "Método más sencillo que podéis utilizar en la práctica: tengo un grupo de datos, elijo bien la distancia, elijo el umbral y a partir de ahí todo esto es raro" [01:34:09]–[01:34:13]. Repetido: "cojo los datos, decido qué distancia me gusta y cierro umbral. Lo [he] solucionado" [01:36:53].
- **Ejemplo de series temporales (figura proyectada que no está en ninguna de las 57 slides; por el contexto, son las series de actividad en Twitter de artistas del paper *Detecting trends in twitter time series*, ver `00_fundamentos.md` §12)** [01:34:34]–[01:36:32]: varias señales con outliers marcados con triángulos rojos; un pico no marcado está muy cerca de otros sí marcados → "en algún momento éstos se han quedado por debajo del umbral y los otros por encima, lo cual parece una decisión bastante arbitraria" [01:35:07]–[01:35:15].
- **Umbral por histograma**: "una forma más justa, si tenéis una base de datos buena, es hacer un histograma de todos los picos e intentar cortar por un punto donde estéis seguros de que por debajo se pertenece a una clase y por encima a otra, y a la vez tenéis pocos puntos que caigan entre medias o cerca del umbral" [01:35:18]–[01:35:39].
- **Explicabilidad**: en el ejemplo, un sistema que no funciona con umbrales deja sin marcar picos (8 o 10) que parecen igual de altos → "es difícil explicar luego en humano por qué eso que parece un pico, que está al lado de otro que es igual, no ha sido un outlier. Este problema no lo tenemos con distancias: es una cosa muy objetiva, a partir de aquí todos y por debajo ninguno" [01:36:07]–[01:36:16]. Ese sistema mira "otras propiedades temporales de la señal [...] cómo se ha llegado a ese pico". "**Lo bueno del umbral es que, una vez que está establecido, es muy fácil de explicar por qué**" [01:36:24]–[01:36:32].

*Fuente: transcripción [01:32:57]–[01:36:59]*

---

### 9. "A menor probabilidad, más anómalo"

**Idea (slide 19).** "Segunda cosa para recordar de esta hora: a más distancia, más anómalo; a menor probabilidad, más anómalo" [01:37:17].

**Detalle.**
- Concuerda con distancias en general, pero añade algo: si hay una muestra suelta "etiquetada y correcta" en una zona, "este que está en general más lejos de la media va a tener más probabilidad que su compañero, que está aquí en medio de la nada" [01:38:17]–[01:38:35]. "Este tipo de métodos nos va a dar [resultados] ligeramente diferentes porque entiende que no sólo la distancia [a] donde están casi todos los puntos importa, sino que **la probabilidad puede ser más importante que la distancia**" [01:38:40]–[01:38:50].
- **Limitación común**: si aparece un punto nuevo rodeado por la clase (una nueva subclase en zona densa), "esta definición nos va a decir que eso no es una novedad en ningún caso; la distancia también" [01:39:02]–[01:39:23]. "Cualquiera de los 2 va a estar limitado por eso: donde haya alta densidad es raro que haya una anomalía. Si de verdad queréis encontrar anomalías que estén muy cerca de vuestros datos, ninguno de estos métodos va a ser el mejor" [01:39:44]–[01:40:01].
- "**Si tenemos un buen modelo de probabilidad, que es difícil de tener, normalmente los métodos de probabilidad van a ser mejores que los de distancia y nos van a dar resultados más finos**" [01:39:33]–[01:39:36].
- Nunca buscar anomalías cerca de los picos de la distribución; buscarlas en las zonas de baja probabilidad (colas) [01:41:08]–[01:41:20].

*Fuente: slide 19 · transcripción [01:37:17]–[01:41:28]*

---

### 10. Repaso de probabilidad: discreta / continua, PDF, media y varianza

**Detalle / fórmulas.**
- **Slide 20**: "Discreta" (histograma de barras, panel "a) Discrete") y "Continua" (curva gaussiana, panel "b) Continuous"). Para continua:
$$\Pr[a\le X\le b]=\int_a^b f_X(x)\,dx$$
  Dibujo 3D: superficie "dP(s)/ds" (densidad) sobre un dominio S; el volumen bajo la superficie sobre una región A es "Volume = P(A)".
- **Slide 21** (tabla):

| | Discrete | Continuous |
|---|---|---|
| Probability distribution | $x_1..x_n$ con $p_1..p_n$ | pdf: $f(x)$ |
| F(x) (normalización) | $\sum_{i=1}^n p_i=1$ | $\int_{-\infty}^{\infty} f(x)\,dx=1$ |
| Mean $\mu$ | $\sum_{i=1}^n x_i p_i$ | $\int_{-\infty}^{\infty} x f(x)\,dx$ |
| Variance $\sigma^2$ | $\sum_{i=1}^n (x_i-\mu)^2 p_i$ | $\int_{-\infty}^{\infty}(x-\mu)^2 f(x)\,dx$ |

**Explicación del profesor** [01:40:18]–[01:48:21]. Discreta: "categorizo a personas de 1,80, de 1,85, cuántas hay en clase, y hago ese dibujo" [01:40:23]–[01:40:33]. Continua: "altura o peso si hiciéramos las cosas bien [...] voy hasta los milímetros [...] me refiero a infinitesimalmente pequeño; si no, milímetros sigue siendo discreta" [01:40:41]–[01:40:57]. En continua "para obtener la probabilidad de un intervalo integramos la PDF [...] porque asumimos que la probabilidad de un punto es 0" [01:47:11]–[01:47:22]. Sobre que integre a 1: "eso es algo que alguien se inventó [...] es una convención. Si todos decidimos que la probabilidad integra 5, pues también perfecto. Lo importante es que sea constante" [01:47:44]. La media: "x por la PDF, o la tradicional, la probabilidad por la muestra, y sumamos todas"; la varianza es "el momento que nos representa la dispersión sobre la media" [01:47:56]–[01:48:12].

*Fuente: slides 20, 21 · transcripción [01:40:18]–[01:41:05], [01:47:08]–[01:48:21]*

---

### 11. Distribución asumida vs real: ejemplo de las notas bimodales (Inglaterra)

**Idea.** "Una cosa es la distribución de vuestros datos, y otra es cuál es la que asumes que debería ser" [01:45:48].

**Ejemplo del profesor** [01:41:28]–[01:47:08]. En Inglaterra, al corregir exámenes de 0 a 10: es muy difícil suspender (aprobado en 5, se hace media incluso con asignaturas suspendidas, se compensa entre años) y luego se hacen percentiles por encima del 5 para clasificar (first class, second class, third class). Las notas reales que salen al corregir son **bimodales** ("muchos estudiantes con una nota razonable y los estudiantes que han ido más o menos a aprobar; está muy estudiado y suele pasar así" [01:44:16]–[01:44:36]). Pero lo que la universidad quiere es que "tus notas sean gaussianas, centradas en 6,5, para que luego sea todo perfecto a la hora de hacer un ranking" [01:44:50]–[01:44:56]. Si se alejan mucho, "cambian el número que te han puesto en el examen y te dicen qué número te habría salido sobre esta distribución [...] has sacado un 8, todo el mundo ha sacado un 8 [...] te bajan tu nota y en tu expediente puede poner 7 y medio" [01:45:14]–[01:45:31].

**Consecuencia para detección de anomalías** [01:45:55]–[01:46:46]: "para mí una anomalía en ésta [gaussiana asumida] va a estar por aquí [colas]; sin embargo una anomalía en ésta [bimodal real] puede ser algo que pase por aquí [entre los dos modos], o en una zona de baja densidad. **Entonces puedo estar enmascarando un montón de anomalías como algo normal porque he asumido que mi distribución no era la que tenía que ser.** [...] Si hubiera pintado el histograma lo vería, pero si he asumido que va a ser una gaussiana me voy a perder mucha información" [01:46:06]–[01:46:27]. "Eso pasa mucho con métodos de probabilidad: dependiendo de cómo de buenos seáis en estimar la probabilidad y de qué suposiciones habéis hecho vais a tener unos resultados u otros" [01:46:34]–[01:46:38]. "Se complica más cuantos más modos haya, cuanto menos gaussianas sean las distribuciones y cuanto más multidimensional sea todo" [01:46:53].

**Reflejo en el notebook**: AD_Probabilidad_clase celdas 37–39 (datos de 4 grupos con hueco central, evaluados con una gaussiana: el hueco de baja densidad no se marca como anómalo). Ver §Ejercicios.

*Fuente: transcripción [01:41:28]–[01:47:08]*

---

### 12. Umbral por número de sigmas (95 % / 99 %) y crítica al p-value

**Detalle (slide 22).** Título "Anomalía" / "La Distribución Normal" / "**Fijar un número de sigmas**". Figura de la normal con:
- Probabilidad por porciones: ≈0,0013 (más allá de ±3σ), ≈0,0214 (entre 2σ y 3σ), ≈0,1359 (1σ–2σ), ≈0,3413 (0–1σ) a cada lado.
- **95 % de valores entre −1,96σ y +1,96σ; 99 % entre −2,58σ y +2,58σ.**
- % acumulativo: −3σ: 0,1 %; −2σ: 2,3 %; −1σ: 15,9 %; 0: 50 %; +1σ: 84,1 %; +2σ: 97,7 %; +3σ: 99,9 %.
- Valores Z de −4 a +4; valores T: 20, 30, 40, 50, 60, 70, 80 (T = 50 + 10Z).

**Explicación del profesor** [01:48:39]–[01:51:57] y [02:16:54]–[02:17:23].
- "Opción 2, y la más frecuente de todas, que habréis visto desde el colegio [...] intervalos de confianza [...] esto es lo más sencillo que hay. Pequeño problema: voy a asumir que mis datos son gaussianos. Seguramente no lo sean, pero si lo son está muy bien establecido" [01:48:48]–[01:49:14].
- Diferencia con distancias: "en distancias no lo tenemos. Nadie ha dicho: cuando la distancia euclídea sea más de 2 queda todo bien, aunque tus datos estén normalizados. Pero aquí en probabilidad sí, porque sabemos que lo máximo va a ser 1 o 100 %, y consideramos que equivocarnos un 5 % de las veces no está tan mal, porque humanamente nos parece un umbral adecuado. Simplemente contamos el número de sigmas [...] de la media" [01:49:28]–[01:50:02]. Ejemplo: una persona que mida 2,30 o 2,40 cae fuera del intervalo del 95 % [01:50:02].
- **Crítica**: "funciona bien no porque tenga ninguna base matemática" [01:50:17]. Analogía con el **p-value** en tests de hipótesis: "todo el mundo ha decidido que si el p-value está por debajo de 0,05 está todo bien, pero eso no tiene ninguna validez"; ejemplo de estudios clínicos de si fumar mata: aceptamos que sí, "pero no precisamente porque sea justo 0,05 o 0,04 o 0,06" [01:50:27]–[01:51:17].
- "El que yo tenga un número de sigmas que me parezca adecuado para una anomalía no quiere decir que eso esté funcionando bien. **Si tenéis datos etiquetados de las anomalías, lo que sí podéis ver es cuántas de vuestras anomalías caen fuera de 2 sigma o de 3 sigma, y podéis ver si este sistema os funciona o no**" [01:51:24]–[01:51:38]. "Normalmente no tenemos etiquetas de anomalías, pero si las tuviésemos, **lo mejor que se puede hacer con ellas, más que entrenar un modelo, es verificar** que algo de este estilo está funcionando bien" [01:51:46]–[01:51:57].
- Recapitulación tras el descanso: "lo más típico es que se haga gaussiana y establecemos un umbral [...] Problemas que podemos tener: la distribución que sea incorrecta o el umbral que sea incorrecto [...] Esto es lo más típico del mundo, 95 %, 99 %, y aquí tenéis el valor correspondiente en desviaciones típicas" [02:16:54]–[02:17:23].

**Código del profesor** (AD_Probabilidad_clase celdas 15, 18, 20–22):
```python
# celda 15: modelo gaussiano 1D con media y desviación ESTIMADAS (incluyen el outlier)
u = np.mean(x) # media real
s = np.std(x) # desviación estandar real
y=(1/(np.sqrt(2*np.pi*s*s)))*np.exp(-(((x-u)**2)/(2*s*s)))   # pdf gaussiana en cada dato
plt.plot(x,y,'.')
plt.plot(R[0:-1],H,'ro-')            # comparada con el histograma

# celda 18: criterio de las 3 sigmas (versión unilateral: sólo cola derecha)
A = (x-u)>3*s

# celdas 20-22: criterio bilateral aplicado a una rejilla uniforme en [-10,10] para ver la región "normal"
xx = (np.random.rand(1000,1)-0.5)*20
AA = np.abs((xx-u))>3*s
plt.plot(x,1/np.log(pp)+1,'r.')      # score con histograma
plt.plot(x,1/np.log(y)+1,'.')        # score con gaussiana
plt.plot(xx,AA,'c.')                 # 1 = fuera de 3 sigmas
```
Resultado (figura celda22_8): la región "normal" de 3σ va aproximadamente de −4 a +4 (σ está inflada por el outlier en x≈9); el outlier queda fuera (AA=1). En celda19_7 sólo el punto x≈9 tiene A=1.

*Fuente: slide 22 · transcripción [01:48:39]–[01:51:57], [02:16:54]–[02:17:23] · AD_Probabilidad_clase celdas 15–22*

---

### 13. Probabilidad multidimensional y transformaciones

**Detalle (slide 23, "Multidimensional:").** Figura 3D: nube de puntos 2D de una gaussiana bivariante con elipse de contorno (verde) y sus dos marginales 1D proyectadas en las paredes (rojo y azul, con histogramas de barras). A la derecha, tres nubes: $\mathrm{cov}(X,Y)<0$ (elipse inclinada hacia abajo), $\mathrm{cov}(X,Y)\approx 0$ (círculo), $\mathrm{cov}(X,Y)>0$ (inclinada hacia arriba).

**Slide 24, "Probabilidad bajo transformaciones:".** Diagrama de una función monótona $y=g(x)$ (curva sigmoidal creciente); abajo la $\mathrm{PDF}(x)$, a la izquierda la $\mathrm{PDF}(y)$; un intervalo $[x, x+dx]$ se mapea a $[y, y+dy]$. (Nota del redactor: es la regla de cambio de variable $p_Y(y)=p_X(x)\,\left|\frac{dx}{dy}\right|$, que la slide ilustra pero no escribe.)

**Explicación del profesor** [02:17:23]–[02:19:31]. Lo anterior "nos vale para algunos problemas, pero no para demasiados, porque [es] unidimensional. Normalmente tenemos muchas características [...] todo en este caso es bidimensional y ya se complica todo bastante" [02:17:23]–[02:17:49]. Relación con Mahalanobis: "tenemos situaciones donde tenemos covarianza 0 entre 2 variables, una esfera perfecta o la matriz identidad [...] o elipses. Tenemos que intentar compensar por ello" [02:17:54]–[02:18:13]. Soluciones: "hacer una transformación de la variable y hacer lo mismo que acabamos de ver con nuestra gaussiana, o lógicamente usar Mahalanobis" [02:18:13]–[02:18:32]. "Si tengo algo que tiene una varianza diferente en cada dimensión, puedo intentar normalizarlo con una transformación y luego utilizar el mismo intervalo de confianza que tenía antes, o me puedo buscar la vida" [02:18:34]–[02:18:53].

**Código del profesor** (AD_Probabilidad_clase celdas 32–36): gaussiana multivariante con media y covarianza estimadas.
```python
N = 1000
X = (np.random.randn(N,2))-0.5          # gaussiana 2D centrada en (-0.5,-0.5)
mu = np.mean(X,axis=0)                   # media estimada
S = np.cov(X.T)                          # matriz de covarianza estimada
from scipy.stats import multivariate_normal
pX = multivariate_normal.pdf(X, mean=mu, cov=S)   # densidad de cada punto
plt.scatter(X[:,0],X[:,1], s=1, c=1/np.log(pX), cmap='Reds')   # score: más rojo = más anómalo
plt.axis('equal')
```
Figura celda36_11: nube circular; el centro casi blanco y los puntos de la periferia en rojo intenso → el score crece con la distancia (de Mahalanobis) al centro.

*Fuente: slides 23, 24 · transcripción [02:17:23]–[02:18:53] · AD_Probabilidad_clase celdas 32–36*

---

### 14. Modelos de distribución (log-normal, gamma…) o combinaciones

**Detalle (slide 25).** "Modelos:" "(o combinaciones)". Enlace: https://en.wikipedia.org/wiki/List_of_convolutions_of_probability_distributions. Figura: grafo enorme de relaciones entre distribuciones, separado en **Univariate** (Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Normal, LogNormal 1–7, Gamma, Generalized Gamma, Beta, ChiSquared, StudentT, Cauchy, Laplace, Weibull, Fréchet, Exponential, Pareto, Uniform, Logistic, Gumbel, Rayleigh, Rice, Lomax, Benford, Zipf, Triangular, Johnson SU/SB/SL, etc.), **Multivariate** (Wishart, Inverse Wishart, Dirichlet, Multivariate Student, Multinomial, Multivariate Normal, Multivariate Gaussian Process…) y **Empirical** (Random Sample, Systematic Sample, Unknown Sample, Mixture Distribution).

**Explicación del profesor** [02:19:13]–[02:20:24]. "Lo más importante a recordar en cuanto a probabilidad es que suele haber 2 alternativas: **una, asumo que todo es gaussiano o medio gaussiano** y utilizo lo que acabamos de ver; **o dos, trabajo con modelos que estiman la probabilidad** y me fío de que el modelo lo haya hecho bien, y sigo haciendo lo mismo: si la probabilidad es muy baja, [lo marco como anómalo]" [02:19:13]–[02:19:36]. "Es mucho más complicado [...] una log-normal, [...] una gamma o una [...]. Si veo que mis datos tienen mucha pinta de una de ellas y me conozco todas éstas de memoria, pues la utilizo. Si no, asumo que tiene una forma cualquiera y me quito de problemas" [02:19:36]–[02:20:00]. Comenta que nunca ha probado a subir datos a un LLM (ChatGPT, Claude) y preguntarle qué distribución tienen, "pero en principio debería poder hacerlo, incluso desde el dibujo" [02:20:04]–[02:20:19].

*Fuente: slide 25 · transcripción [02:19:13]–[02:20:24]*

---

### 15. Teoría de la información: información vs anomalía (Shannon)

**Detalle (slides 27–28).** Slide 27: "Información vs Anomalía". Slide 28: "Información — Claude E. Shannon (1948) 'A Mathematical Theory of Communication'" (foto de Shannon). "Información es proporcional a la inversa de la [probabilidad]":
$$I(x)\propto\frac{1}{p(x)}$$
(Nota del redactor: la definición estándar es $I(x)=-\log p(x)=\log\frac{1}{p(x)}$; en los notebooks se usa $1/p(x)$ y $1/\log p(x)$ como scores.)

**Explicación del profesor** [02:20:52]–[02:23:40].
- Pregunta a la clase: ¿"a más información, más anómalo" o "a menos información, más anómalo"? [02:20:55]. "Lo que hemos visto hasta ahora siempre es que a más información es más raro. Si os acordáis del ejemplo de las noticias: si yo os digo que mañana aparece un volcán en la Plaza Mayor, muy raro, nos da mucha información. Sería muy raro que pasase, entonces muy baja probabilidad. Según esa línea de razonamiento, **a más información, menos probabilidad y más anómalo**" [02:21:38]–[02:22:06].
- La paradoja: "lo que hemos dicho hasta ahora siempre ha sido más información mejor, porque el modelo puede aprender más; siempre quiero muestras que sean muy informativas. Pero eso lo decíamos asumiendo que esas muestras no son anómalas, es decir, siempre que las muestras sean correctas y pertenezcan a tu clase" [02:22:10]–[02:22:35].
- Física vs teoría de la información: "en física, si tienes una señal [...] de ruido blanco, se asume que es muy poco informativa. Sin embargo, [según Shannon], algo muy aleatorio como el ruido blanco, que no sabes dónde va a aparecer el siguiente píxel, es muy informativo, pero no es relevante para lo que queremos aprender [...] Entonces más información, menos probabilidad" [02:23:06]–[02:23:31].
- "Lo bueno que tiene la teoría de la información es que nos da medidas que podemos utilizar para evaluar si algo es anómalo o no, más allá de la probabilidad" [02:23:44].

*Fuente: slides 26, 27, 28 · transcripción [02:20:52]–[02:23:53]*

---

### 16. Entropía

**Detalle (slide 29).**
$$H(X)=-\sum_{i=1}^{n}P(x_i)\log P(x_i)$$
"Valor esperado de información de una fuente".

**Explicación del profesor** [02:23:53]–[02:26:51].
- "La entropía no es más que una forma de cuantificar ese 1 sobre la probabilidad, pero de una forma más bonita"; es para variables discretas [02:24:03]–[02:24:13].
- **Ejemplo del dado**: probabilidad de sacar un 1 (1/6) por el log de 1/6, sumado para los 6 casos → cuánta información hay al tirar un dado. "En física, tirar un dado es muy poco informativo porque es completamente aleatorio; para nosotros es muy informativo porque no sabemos lo que va a pasar" [02:24:13]–[02:24:31]. "Algo que tenga una distribución **uniforme** como un dado es la distribución más informativa que se puede tener" [02:24:40]–[02:24:49]. En una gaussiana, cerca de la media hay más probabilidad: "si alguien me dice que mide 1,80, vale, no me interesa; si mide 2,40 [...] ya me interesa más" [02:24:55]–[02:25:15].
- "La entropía nos puede valer como una forma de cuantificar cómo de anómalo es un punto" [02:25:22].
- **Problemas** (pregunta a la clase) [02:25:29]–[02:26:43]:
  1. "Necesitamos la probabilidad igualmente; todos los problemas que teníamos con medidas de probabilidad los seguimos teniendo" [02:25:56]–[02:26:08].
  2. Igual que se asumía gaussiana o euclídea, "aquí estoy asumiendo que la entropía es la entropía de Shannon, con logaritmo, que puede que no sea la mejor en todos los casos" [02:26:12]–[02:26:27]; "el 99,9 % de los casos [se usa] esta entropía, porque nadie sabe lo suficiente para elegir otras entropías, aunque existen" [02:26:39]–[02:26:43].

*Fuente: slide 29 · transcripción [02:23:53]–[02:26:51]*

---

### 17. Información mutua

**Detalle (slide 30).**
$$I(X;Y)=\sum_{y\in\mathcal Y}\sum_{x\in\mathcal X}p_{(X,Y)}(x,y)\log\left(\frac{p_{(X,Y)}(x,y)}{p_X(x)\,p_Y(y)}\right)$$
Diagrama de Venn con $\mathbf x=[x_1,x_2]$ (círculos rojos) e $\mathbf y=[y_1,y_2]$ (círculos azules) y leyenda: $H(x_1)$, $H(x_2)$, $H(\mathbf x)=H([x_1,x_2])$, $T(\mathbf x)=I(x_1,x_2)$ (correlación total dentro de $\mathbf x$), $I(\mathbf x,\mathbf y)$ (solape entre los dos grupos), $T([\mathbf x,\mathbf y])$ (correlación total de todas las variables juntas).

**Explicación del profesor** [02:26:51]–[02:27:43]. "Tenemos algo que es muy chulo, que son las medidas de información mutua [...] la idea de que tenemos 2 variables. Imaginad que en este caso tengo una variable que es mis datos que me gustan y otra es los posibles outliers. Lo que quiero saber es cuánta información comparten, porque si veo que no comparten ninguna, con cierta seguridad puedo decir que eso es un outlier" [02:26:57]–[02:27:14]. "Estamos intentando cuantificar cuánto solapan las 2 variables aleatorias" [02:27:19]. Lo relaciona con los **tests de hipótesis** (que no se comentan pero también servirían para detección de novedad): "no hace más que decirte cómo de probable es que una muestra venga de una distribución [gaussiana], que es justo lo que estamos mirando, aparte de poner un umbral" [02:27:28]–[02:27:43].

*Fuente: slide 30 · transcripción [02:26:51]–[02:27:48]*

---

### 18. Divergencia de Kullback-Leibler (entropía relativa)

**Detalle (slide 31).**
$$D_{\mathrm{KL}}(P\,\|\,Q)=\sum_i P(i)\ln\frac{P(i)}{Q(i)}$$
$$D_{\mathrm{KL}}(P\,\|\,Q)=-\sum_{x\in\mathcal X}p(x)\log q(x)+\sum_{x\in\mathcal X}p(x)\log p(x)=H(P,Q)-H(P)$$
(entropía cruzada menos entropía). Figuras: arriba, dos PDFs gaussianas $p(x)$ y $q(x)$ desplazadas ("Original Gaussian PDF's") y el integrando "KL Area to be Integrated" (área positiva a la izquierda y negativa a la derecha); abajo, cuatro gaussianas cada vez más desplazadas y sus integrandos KL, cuya área crece con la separación.

**Explicación del profesor** [02:27:48]–[02:31:13].
- "Esto es especialmente interesante porque a veces es fácil de calcular, incluso sin conocer las distribuciones. Es una **medida de distancia entre distribuciones de probabilidad**" [02:28:00]–[02:28:06].
- Visión unificada ("estamos haciendo lo mismo todo el rato") [02:28:24]–[02:30:22]:
  1. Distancias: una clase y un punto; si la distancia es mayor que un umbral, anomalía.
  2. Probabilidad: asumir y poner una gaussiana sobre los datos; umbral p. ej. en 2 sigmas; lo que está más allá es anomalía.
  3. KL: asumir también una gaussiana para la(s) muestra(s) nueva(s) "centrada ahí", y "en vez de mirar con respecto a esa distribución, lo que hago es mirar cuál es la distancia entre estas 2 [distribuciones]. Si la distancia entre esas 2 está más allá del umbral —ya no las muestras, sino las distribuciones— eso va a ser una anomalía" [02:29:34]–[02:29:55]. "Por una parte la más compleja, pero también la más realista" [02:30:01]–[02:30:05].
  - Simplificación: si son 2 gaussianas con la misma varianza ("que no la van a tener, porque tengo 3 muestras aquí y 1 aquí"), "podría mirar simplemente la distancia entre las medias y poner un umbral en eso" [02:30:09]–[02:30:22].
  - "En la vida real lo que se suele hacer es una estimación de probabilidad de esto, una estimación de probabilidad de esto, y la divergencia KL entre las 2" [02:30:29]–[02:30:36]. "Por eso en la práctica la gente en estadística hace mucho la del medio [probabilidad + umbral], un poco de ésta y nada de ésta" [02:30:43]–[02:30:48].
- "Se parece bastante a la entropía, con un logaritmo [del cociente] de las distribuciones [...] una distribución P y una Q, ver cómo de lejos está una de la otra. **Para gaussianas se puede hacer de forma cerrada y es muy sencilla; para otras distribuciones es más complicada**" [02:30:54]–[02:31:13].

*Fuente: slide 31 · transcripción [02:27:48]–[02:31:13]*

---

### 19. Toolbox RBIG — "Information Theory Measures via Multidimensional Gaussianization"

**Detalle (slide 32).** "Miramos… **Information Theory Measures via Multidimensional Gaussianization**. Valero Laparra, J. Emmanuel Johnson, Gustau Camps-Valls, Raul Santos-Rodríguez, Jesus Malo. https://arxiv.org/abs/2010.03807".

**Explicación del profesor** [02:31:18]–[02:33:02].
- "Aquí nos recomendaba [Valero] que miráramos un artículo que también hice yo con él, y yo recomendaría no mirarlo [...] yo no me atrevería ni a abrir el link"; creía que tenía 50–60 páginas, son 27 [02:31:18]–[02:31:56].
- "Ese artículo tenía una cosa bien [de antes de Claude]: viene con una **toolbox que os permite estimar todas estas medidas, tanto la entropía, como la divergencia KL, como la información mutua. Y eso en una línea soluciona todos estos problemas**, y es bastante eficiente y guay, **sobre todo en casos que sean de muy alta dimensión**" [02:32:04]–[02:32:21].
- "Para casos de baja dimensión, hagáis lo que hagáis, cualquiera de las [soluciones] que os estaban enseñando, todas estarán bien. Hay problemas cuando tenéis problemas de muy alta dimensión" [02:32:27]–[02:32:34]. "Si tenéis muy, muy alta dimensión esto se vuelve muy complicado, y no sólo muy costoso computacionalmente, sino muy complicado de que funcione bien. Lo que suelo hacer es esto directamente: una cosa [de] distancia lo más simple posible" [02:32:44]–[02:32:56].
- El uso práctico de RBIG (API, densidad, medidas de información) está en `02_deep.md` §9.

*Fuente: slide 32 · transcripción [02:31:18]–[02:33:02]*

---

### Ejercicios de los notebooks (enunciado → solución)

#### A. `AD_Data` (datos de ejemplo de anomalías — ADBench)

Cabecera: "Datos ejemplo de Anomalías", https://github.com/Minqi824/ADBench, autor Valero Laparra.

| Celda | Enunciado (AD_Data.md) | Solución / resultado (AD_Data_clase.md) |
|---|---|---|
| 3 (enun.) | Descarga directa con `requests` de `6_cardio.npz` desde GitHub; imprime claves `['X','y']` y muestra `X` | En _clase se elimina y se usa la librería |
| 3/5 | `!pip install adbench` y `utils.download_datasets(repo='jihulab')` (descarga CV_by_ResNet18, NLP_by_BERT, Classical; entre los clásicos está `13_fraud.npz`) | Igual |
| 5/6 | Carga `6_cardio.npz` → `X.shape=(1831, 21)`, `y.shape=(1831,)` | Igual; línea comentada `#data = np.load(path + '5_campaign.npz', ...)` |
| 8/9 | **EJERCICIO: Plotea los datos** (en el enunciado sólo hay `drive.mount`) | Pairplot de las 6 primeras columnas con `hue=y`, scatter de las 2 primeras columnas, y además LOF |

```python
# AD_Data_clase celdas 9-11: pairplot de 6 variables coloreado por etiqueta real
import seaborn as sns
import pandas as pd
df = pd.DataFrame(X[:,0:6])
df['y'] = y
sns.pairplot(df,hue='y')

# celda 12: scatter de las 2 primeras columnas coloreado por y
plt.scatter(X[:,0],X[:,1],c=y)

# celdas 13-14: Local Outlier Factor (no supervisado) con 20 vecinos
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)          # +1 = inlier, -1 = outlier
plt.scatter(X[:,0],X[:,1],c=y_pred)

# celda 15: pairplot coloreado por la predicción (yy = 1 si LOF dice outlier)
df = pd.DataFrame(X[:,0:6])
yy = (-y_pred+1)/2
df['y'] = yy
sns.pairplot(df,hue='y')

# celdas 16-20: comparación con la etiqueta real
np.abs(y_pred-y)
yy = (y_pred+1)/2                    # ahora yy = 1 inlier, 0 outlier (convención opuesta a y)
yy.min()                             # 0.0
np.sum(np.abs(y-yy))/len(y)          # 0.892408519934462
ii = y==1
np.sum(np.abs(y[ii]-yy[ii]))/len(y[ii])   # 0.05113636363636364
```
**Resultados observados.**
- Figura celda11_1 (pairplot por etiqueta real): las anomalías (naranja, y=1) son minoría; se concentran en valores bajos de la variable 1 (≈ −1), algunos grupos aislados en la variable 2 (≈6–8) y en la variable 5 (valores 13–20 cuando casi todo es 0). Muchas anomalías están solapadas con los normales en el resto de proyecciones.
- Figura celda12_2: scatter con valores en [0,1] y franjas horizontales; no coincide con el rango de cardio (estandarizado) → (nota del redactor) probablemente se ejecutó con otro dataset (p. ej. el `5_campaign` comentado).
- Figura celda14_3: LOF marca como outliers (morado) sólo unos pocos puntos aislados en los bordes de la nube.
- Figura celda15_4: con la predicción de LOF casi todo es "normal"; los puntos marcados son extremos aislados (p. ej. algunos de la variable 5 con valores 13–20).
- Métricas: como `y` (1=anomalía) y `yy` (1=inlier) tienen convenciones opuestas, `np.sum(|y-yy|)/len(y)=0.892` es en realidad la **tasa de acierto** (≈89 %), y sobre las anomalías reales `0.0511` es la **fracción de anomalías detectadas (recall ≈ 5 %)**, es decir, 9 de 176 (nota del redactor: 0.05114 × 176 = 9; cardio tiene 176 anomalías de 1831, ≈9,6 %). Conclusión: un detector no supervisado "por defecto" puede tener accuracy alta y recall bajísimo.

#### B. `AD_Distancias`

| # | Enunciado | Solución en _clase | Resultado |
|---|---|---|---|
| 1 | "EJERCICIO: Plotea los datos" (A = [[1,2],[3,1],[-1,2]]) | `plt.plot(A[:,0],A[:,1],'.')` | Figuras celda05_1/celda08_2: los 3 puntos (−1,2), (1,2), (3,1) |
| 2 | "Calcula las distancias Lp para p=[0, 0'5, 1, 2, 3, inf]" | Bucle doble con p ∈ {1e-5, 0.5, 1, 2, 3, 1e7} (código en §4) | Matrices en §4; ojo al overflow entero con p=1e7 |
| 3 | Muchos datos 1D: `y = rand(N,1)`, `y_hat = y + ruido` (enunciado σ=0.1; _clase σ=0.5). "Calcula las distancias Lp entre los valores reales y los predichos para distintas cantidades de ruido" | Matriz M (1000×1000) de |y[n] − y_hat[m]| con p=1 y `mmm = np.min(M,axis=1)` (distancia de cada test a su vecino más cercano) | Figura celda14_3: dispersión ejemplo vs test (ruido grande). Figura celda17_4: 10 ejemplos (azul) y 10 test (rojo) en la recta. Figura celda22_5: **forma de "V"/bañera**: los test dentro de [0,1] tienen distancia ≈0 (siempre hay un ejemplo cerca) y los de fuera crecen linealmente (hasta ≈1,3–1,5 en x≈−1,3 y x≈2,5) → la distancia al vecino más cercano sólo detecta lo que sale del soporte de los datos, no el ruido dentro del soporte |
| 4 | "Mete algún dato anómalo (outlier) y mira cómo afecta a cada una de las distancias" | Sin solución | — |
| 5 | Muchos datos 2D: parábola `y = x**2`, `x ∈ [-1,1]`, test con ruido σ=0.05 (en _clase se perturba también x: `x_hat`). "Calcula las distancias Lp entre cada punto 2D [...] ¿Qué punto tiene las mayores distancias para cada valor de p y de sigma?" | M[m,n] = L1 entre (x[n],y[n]) y (x_hat[m],y_hat[m]); `mmm = np.min(M,axis=1)` | Figura celda25_6: parábola azul y nube roja alrededor |
| 6 | "Busca puntos que se detecten como outlier sólo por algunas distancias" | Scatter de la nube test coloreada por `mmm` (Reds) sobre la parábola | Figura celda30_7: los puntos más alejados de la curva (en dirección normal) aparecen en rojo oscuro; los pegados a la curva, claros. No se compara entre distintos p |
| 7 | "Dibuja los valores en 2D que distan 1 del centro (0,0) para las distancias Lp" | Código no conservado; figura celda32_8 | Astroide rojo (p<1) dentro de un cuadrado azul [−2,2]² |
| 8 | Imagen (`https://www.uv.es/lapeva/Valero_2020_DQ.png`, retrato): convertir en datos (D,N) con patches 8×8×3 → D=192; calcular Lp entre dos patches; calcular **Mahalanobis** entre dos patches | Sólo carga y muestra la imagen (`skimage.io.imread`, figura celda36_9) | Sin solución |
| 9 | Meter un patch anómalo: ponerlo a negro, escalarlo (×un poco más de 1), cambiar contraste, meter ruido; analizar su distancia a los otros patches | Sin solución | — |

```python
# AD_Distancias_clase celdas 25-30 (2D, parábola): score = distancia L1 al punto de referencia más cercano
N = 1000
x = 2*((np.random.rand(N,1))-0.5)
y = x**2
sigma = 0.05
y_hat = y + sigma*np.random.randn(y.shape[0],y.shape[1])
x_hat = x + sigma*np.random.randn(x.shape[0],x.shape[1])
p = 1
M = np.zeros((y.shape[0],y.shape[0]))
for n in range(y.shape[0]):
    for m in range(y.shape[0]):
        M[m,n] = ((np.abs(x[n,0]-x_hat[m,0]))**p+(np.abs(y[n,0]-y_hat[m,0]))**p)**(1/p)
mmm = np.min(M,axis=1)
plt.plot(x,y,'.')
plt.scatter(x_hat,y_hat, s=1, c=mmm, cmap='Reds')
```

#### C. `AD_Probabilidad`

| # | Enunciado | Solución en _clase | Resultado |
|---|---|---|---|
| 0 | Datos: `x = randn(100,1)` | Se añade un outlier: `x[35] = x[35]*15` | Figura celda04_1: un punto en ≈9 |
| 1 | Histograma `np.histogram(x)` (en _clase con `density=True`) | `plt.plot(R[0:-1],H,'o-')` + datos sobre el eje | Figuras celda06_2/celda07_3: 10 bins; masa entre −2,5 y 2; bins vacíos entre 2 y 7; pequeño bin en ≈8 (el outlier) |
| 2 | "Plotea bien el histograma (sin usar plt.hist). En R tenemos el principio y el fin de los bins. Compáralo con plt.hist" | Sin solución explícita (se usa `R[0:-1]`, borde izquierdo) | — |
| 3 | "Calcula las probabilidades para cada dato" | `pp = np.interp(x,R[0:-1],H)` (interpola la densidad del histograma) | — |
| 4 | "Compara la cantidad de anomalía de cada dato (1/p(x)) con la anomalía calculada como distancia L2 y L1" | `plt.plot(x,1/pp,'.')` (no se calcula L1/L2) | Figura celda13_4: curva en U, ≈3 en el centro; el outlier x≈9 tiene 1/p≈115, y un punto en x≈1,9 llega a ≈43 (cae junto al bin vacío) |
| 5 | Modelo gaussiano (enunciado: u=0, s=1 "reales") | u, s **estimados** con `np.mean`, `np.std` (incluyen el outlier) y comparación con el histograma | Figura celda15_5: la gaussiana estimada (pico ≈0,30) es más ancha/baja que el histograma (pico ≈0,35) por culpa del outlier; el outlier tiene p≈0 |
| 6 | (score) | `plt.plot(x,1/np.log(pp),'r.')` vs `plt.plot(x,1/np.log(y),'.')` | Figura celda16_6: ambos en V (valores negativos; más cerca de 0 = más anómalo); el outlier: −0,04 (gaussiana) y −0,21 (histograma); el histograma es más "picudo" |
| 7 | "Encuentra los datos anómalos siguiendo el criterio de 3 sigmas" | `A = (x-u)>3*s` y después versión bilateral `AA = np.abs(xx-u)>3*s` sobre una rejilla en [−10,10] | Figuras celda19_7/celda22_8: sólo x≈9 es anómalo; zona normal ≈[−4,4] (σ inflada) |
| 8 | Comparar 1/p(x) con p(x) del modelo y del histograma | Sin solución adicional | — |
| 9 | "Compara p(x) cuando la media y la desviación estándar son las reales o son estimadas" | Implícito: se usan las estimadas (y el outlier las sesga) | — |
| 10 | Igual pero asumiendo un modelo con media 0,5 y sigma 2: "¿Qué pasa?" | Sin solución | (Nota del redactor: modelo mal especificado ⇒ ranking de anomalías desplazado/aplanado, como en el ejemplo de las notas) |
| 11 | "Calcula la probabilidad usando un modelo Gaussiano con Scipy" | Código no conservado; figura celda27_9 | Campana con máximo ≈0,40 (N(0,1)) y un punto en x≈3,4 con p≈0 |
| 12 | "Compara las anomalías de los siguientes datos usando distancias, histograma y modelos": `x = sign(randn)*|randn|**0.5` (distribución bimodal, con hueco en 0) | Sin solución | — |
| 13 | 2D: "Encuentra anomalías en los siguientes conjuntos de datos 2D" (1) gaussiana `randn(N,2)-0.5` | `mu`, `S = np.cov(X.T)`, `multivariate_normal.pdf`, score `1/np.log(pX)` | Figura celda32_10 (nube) y celda36_11 (periferia roja) |
| 14 | (2) `X2 = sign(randn)*|randn|**0.5` en 2D | `pX2 = multivariate_normal.pdf(X2, mean=mu, cov=S)` **con mu y S de la gaussiana anterior** | Figuras celda37_12/celda39_13: 4 grupos (uno por cuadrante) con un hueco en forma de cruz alrededor de los ejes. El score sólo depende de la distancia a (−0,5,−0,5): el cuadrante superior derecho aparece rojo y el inferior izquierdo blanco, y **el hueco de baja densidad (cruz central) no se marca como anómalo** → ilustra el problema de asumir la distribución equivocada (§11) |
| 15 | (3) `X = sign(randn)*|randn|**3` (colas muy pesadas) | Sólo se dibuja | Figura celda40_14: núcleo denso en el origen con puntos dispersos hasta ±60 a lo largo de los ejes |
| 16 | "¿Qué has hecho con los histogramas? ¿2 1D ó 1 2D?" | Sin solución | (Nota del redactor: 2 histogramas 1D asumen independencia entre variables; 1 histograma 2D captura la estructura conjunta pero requiere muchos más datos) |

```python
# AD_Probabilidad_clase celdas 3-13: outlier, histograma como estimador de densidad y score 1/p
x = np.random.randn(100,1)
x[35] = x[35]*15                     # outlier artificial
[H,R] = np.histogram(x,density=True) # H = densidad por bin, R = bordes de bins
plt.plot(R[0:-1],H,'o-')
plt.plot(x[:,0],np.zeros((x.shape[0],)),'bo')
pp = np.interp(x,R[0:-1],H)          # p(x) de cada dato interpolando el histograma
plt.plot(x,1/pp,'.')                 # cantidad de anomalía = 1/p(x)

# celda 16: scores en escala log (histograma vs gaussiana)
plt.plot(x,1/np.log(pp),'r.')
plt.plot(x,1/np.log(y),'.')

# celdas 37-39: datos no gaussianos evaluados con la gaussiana estimada antes
X2 = np.sign(np.random.randn(1000,2))*np.abs(np.random.randn(1000,2))**0.5
pX2 = multivariate_normal.pdf(X2, mean=mu, cov=S)
plt.scatter(X2[:,0],X2[:,1], s=1, c=1/np.log(pX2), cmap='Reds')

# celda 40: colas pesadas
X = np.sign(np.random.randn(1000,2))*np.abs(np.random.randn(1000,2))**3
```
(Nota del redactor: `1/np.log(p)` sólo es monótono creciente con la anomalía mientras $p<1$ (log negativo); si una densidad supera 1, el log es positivo y el score cambia de signo/se rompe. Para producción es más robusto usar $-\log p(x)$, que es la "información" de Shannon.)

---

### Lo que dijo el profesor (recomendaciones explícitas)

- "No siempre es verdad a más distancia más anómalo" [01:02:22].
- "Antes de empezar a pensar en distancias, tenéis que pensar si los datos que tenéis representan bien el concepto/categoría que queréis aprender. Si la respuesta es no, lo de la distancia es un poco irrelevante" [01:04:29]–[01:04:39].
- Un mismo punto puede ser "seguramente outlier" o "seguramente no outlier" según lo que asumas (frontera vs lejanía) [01:07:53]; SVM/MLP asumen que lejos de la frontera "está todo bien", lo que puede ocultar ejemplos de clases nunca vistas [01:08:23]–[01:08:46].
- La incertidumbre aleatoria (frontera) no se reduce con más datos; la epistémica (zonas sin datos) sí [01:10:46]–[01:13:13].
- Usar una distancia relacionada con el problema; si no, quizá otra valga; si no sabes nada, euclídea [01:17:31]–[01:17:46]. Elegir una distancia por comodidad de optimización tiene un precio [01:18:19].
- Manhattan para datos cuantizados/grids/juegos [01:15:43], [01:23:42].
- En la práctica sólo se usan p = 1, 2, ∞ [01:21:23]; tunear p es raro (1,2–1,3) [01:22:45].
- Coseno cuando importa la dirección y no el módulo (bag of words) [01:26:20]–[01:26:51].
- "Si estoy utilizando la euclídea en un sitio donde debería utilizar Mahalanobis, estoy ordenando mis puntos mal en el espacio" [01:30:19]. Alternativa equivalente: normalizar primero y usar euclídea [01:29:39].
- Distancia ¿a qué?: lo más típico, al centroide o al vecino más cercano [01:32:44].
- "Recordar cuál fue la distancia que te llevó a decir que esto era una anomalía para luego categorizarlas" [01:33:48].
- Umbral por histograma: cortar donde haya pocos puntos cerca del umbral [01:35:18]–[01:35:39]. Un umbral es fácil de explicar [01:36:32].
- "Si tenemos un buen modelo de probabilidad [...] normalmente los métodos de probabilidad van a ser mejores que los de distancia" [01:39:33].
- Ni distancia ni probabilidad detectan anomalías dentro de zonas de alta densidad [01:39:44]–[01:40:01].
- "Nunca voy a buscar anomalías cerca de los picos de la distribución" [01:41:20].
- Pintar el histograma antes de asumir gaussiana; asumir mal la distribución enmascara anomalías [01:46:15]–[01:46:27].
- El 95 %/99 % (sigmas) es una convención "humana", como el p < 0,05, sin base matemática [01:49:45], [01:50:17], [01:50:50].
- "Si tenéis datos etiquetados de las anomalías [...] ver cuántas caen fuera de 2 o 3 sigma [...] Lo mejor que se puede hacer con ellas, más que entrenar un modelo, es verificar" [01:51:24]–[01:51:57].
- Dos alternativas en probabilidad: asumir (medio) gaussiano, o un modelo que estime la densidad y fiarse de él [02:19:16]–[02:19:31]; si los datos tienen pinta de log-normal/gamma, usarla [02:19:49].
- La entropía arrastra todos los problemas de estimar la probabilidad [02:26:02].
- KL entre distribuciones es lo más realista pero lo más complejo; en la práctica se usa sobre todo probabilidad + umbral [02:30:01], [02:30:43].
- La toolbox RBIG estima entropía, KL e información mutua "en una línea", útil sobre todo en muy alta dimensión; en baja dimensión cualquier método vale; en muy alta dimensión él usa "una distancia lo más simple posible" [02:32:10]–[02:32:56].

---

### Aplicación a nuestra práctica (fraude, AUPR, 0,17 %)

(Sección de interpretación del redactor a partir de las fuentes; las ideas se atribuyen a la fuente que las justifica.)

1. **Escalas X1..X20 vs Amount → estandarizar o Mahalanobis** (§6, [01:28:15]–[01:30:24]). `Amount` tiene otra escala y gran varianza (como el "peso" del ejemplo): con euclídea dominaría la distancia. Opciones del profesor: estandarizar y usar euclídea, o Mahalanobis con la covarianza de los no-fraudes. Probable `log1p(Amount)` antes (§14: Amount tiene pinta log-normal/gamma; la slide 24 justifica transformar la variable). **Notebooks 01_EDA (comprobar escalas, correlaciones y forma de Amount) y 02_baselines_no_supervisados.**
2. **Baseline Mahalanobis = gaussiana única** (§6, §13): score $d_M(x,\mu_{\text{normal}})$ con $\mu$, $S$ estimados sólo en no-fraude (como en AD_Probabilidad celdas 33–35). Es el baseline "distancia + probabilidad" más barato y explicable. **02.**
3. **GMM(5) de partida del profesor** = alternativa 2 de [02:19:28] ("modelos que estiman la probabilidad") y respuesta al ejemplo de las notas bimodales (§11): una sola gaussiana enmascararía anomalías entre modos. `score_samples` devuelve $\log p(x)$, así que el score de entrega debe ser **$-\log p(x)$** (mayor = más fraude), coherente con $I(x)\propto 1/p(x)$ (slide 28) y evitando el `1/np.log(p)` de los notebooks (se rompe si $p>1$, frecuente en densidades de alta dimensión). Probar distintos nº de componentes / `covariance_type` comparando AUPR. **02 y 04.**
4. **Ajustar sólo con no-fraude, validar con las etiquetas** ([01:51:24]–[01:51:57]): el profesor dice que con etiquetas lo mejor es verificar; tenemos 295 fraudes → usarlos para validar (AUPR en CV estratificada) los detectores no supervisados, y además (fuera de este bloque) entrenar supervisados. **02, 03.**
5. **Umbral vs score**: AUPR es una métrica de ranking, no necesita umbral. Las sigmas (95 % / 99 %, slide 22) sólo sirven para explicar/umbralizar en el notebook (p. ej. "% de fraudes fuera de 3σ de Mahalanobis"), y el consejo de "recordar la distancia" [01:33:48] = **entregar el score continuo, no la etiqueta binaria**. Con 0,17 % de positivos, un corte a 2σ (≈5 % marcados) sería muy laxo. **99_ENTREGA.**
6. **Distancia a qué** (§7): kNN-distance al vecino más cercano no-fraude (como `np.min(M,axis=1)` en AD_Distancias) vs distancia al centroide. Con 170k filas la matriz completa N×N del notebook es inviable (≈2,9·10¹⁰ pares) → usar `sklearn.neighbors.NearestNeighbors` / submuestreo de no-fraudes (nota del redactor). **02.**
7. **Lección de la V de AD_Distancias** (celda22_5) y del LOF de AD_Data (recall ≈5 %): los métodos por distancia sólo ven lo que sale del soporte de los normales; fraudes "camuflados" en zonas densas no se detectan ([01:39:44]–[01:40:01]) → justifica pasar a supervisados/híbridos. **03, 04.** También: accuracy alta (0,89) no significa nada con clases desbalanceadas; por eso AUPR.
8. **Frontera vs lejanía** (§1): un clasificador supervisado (SVM/MLP/árboles) da alta confianza "no fraude" lejos de la frontera aunque allí no haya datos; en producción pueden aparecer fraudes de tipos nuevos lejos de todo → combinar score supervisado con un score de novedad (distancia/densidad) en el **ensemble híbrido (04)**.
9. **Distribuciones de las variables** (§11, §14): en 01_EDA pintar histogramas por variable y por clase (como el pairplot de AD_Data) para ver multimodalidad/colas pesadas (AD_Probabilidad celda 40) antes de asumir gaussiana; decidir transformaciones (log, rank-gauss, quantile). **01.**
10. **Histogramas 1D vs conjunto** (ejercicio 16 de AD_Probabilidad): un score tipo "suma de −log p de histogramas 1D por variable" (HBOS-like, nota del redactor) asume independencia; GMM/Mahalanobis capturan la estructura conjunta. Útil como baseline barato adicional. **02.**
11. **Producción vs construcción → KL** (§18): la idea de comparar distribuciones (no muestras) se puede usar para vigilar si la distribución de producción difiere de la de construcción (drift) por variable, p. ej. KL entre histogramas o test de hipótesis (nota del redactor). **01/99 (reflexión final).**
12. **Alta dimensión / RBIG** (§19): 21 variables es dimensión moderada; según el profesor "en baja dimensión cualquier método vale", y en muy alta dimensión una distancia simple. RBIG podría usarse como estimador de densidad alternativo (ver `02_deep.md` §9). **02/04 opcional.**


#### Baselines de distancia y probabilidad medidas en nuestros datos *(medido por el grupo)*

Mismo split común (70/30 estratificado, seed 42). Los modelos se ajustan **solo con los no-fraude de
train**, el escalado (`StandardScaler`) también, y el score es "mayor = más anómalo". Test: 51.266
filas, 89 fraudes. AUPR aleatorio = 0,0017.

| Método | Preproceso | AUPR test | ROC test |
|---|---|---|---|
| GMM 5 comp. (**partida del profesor**) | crudo | 0,567 | 0,948 |
| Euclídea al centroide | std | 0,225 | 0,959 |
| Euclídea al centroide | log1p(Amount) + std | 0,270 | 0,960 |
| kNN, media de 5 vecinos (referencia: 30k normales) | std | 0,336 | 0,965 |
| kNN, media de 5 vecinos | log1p(Amount) + std | 0,370 | 0,966 |
| Histogramas 1D, suma de −log p (tipo HBOS, 50 bins) | log1p(Amount) + std | 0,402 | 0,958 |
| Mahalanobis robusta (MinCovDet, 20k) | std | 0,283 | 0,930 |
| Mahalanobis empírica (= GMM 1 comp.) | std | 0,556 | 0,961 |
| Mahalanobis empírica (= GMM 1 comp.) | log1p(Amount) + std | 0,580 | 0,963 |
| GMM 5 comp. full | log1p(Amount) + std | 0,572 | 0,957 |
| **GMM 10 comp. full** | **log1p(Amount) + std** | **0,662** | 0,963 |

**Lecturas** (nota del redactor):
- **La covarianza importa:** en el mismo espacio estandarizado, la euclídea al centroide da 0,225 y
  Mahalanobis 0,556. Las X están estandarizadas pero no son independientes entre sí en los no-fraude
  (ni con `Amount`). Es literalmente el aviso del profesor: "si utilizo la euclídea donde debería
  utilizar Mahalanobis, ordeno mal mis puntos" [01:30:19].
- **log1p(Amount)** mejora casi todos los métodos (Amount tiene cola larga: §11 y §14).
- **La covarianza robusta empeora** (0,283). Al recortar la influencia de los normales extremos, la
  elipse se estrecha y muchos normales de cola pasan a parecer tan raros como los fraudes. Aquí la
  covarianza clásica funciona mejor.
- **Más componentes en el GMM** (densidad más flexible = "modelo que estima la probabilidad",
  [02:19:28]) es lo que más sube el AUPR, de 0,58 a 0,66. Hay que elegir el nº de componentes por CV
  sobre AUPR, no por BIC.
- La ROC es casi igual en todos (≈0,93–0,97) mientras la AUPR varía de 0,23 a 0,66: con un 0,17 % de
  positivos, **la ROC no discrimina entre métodos**. Ver `03_evaluacion_y_herramientas.md` §19.
- Ojo: es un único split con 89 fraudes, así que diferencias de ±0,05 no son concluyentes (ver
  `03_evaluacion_y_herramientas.md` §20). Hay que confirmarlo con CV en el notebook 02.

```python
# Reproducir (con la celda común ya ejecutada): ajustar con no-fraude, score mayor = más fraude
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors

def prep(X):
    X = X.copy(); X["Amount"] = np.log1p(X["Amount"]); return X.values

Xtr0 = prep(X_train[y_train == 0])
sc = StandardScaler().fit(Xtr0)
Z0, Zte = sc.transform(Xtr0), sc.transform(prep(X_test))
ref = Z0[np.random.default_rng(SEED).choice(len(Z0), 30000, replace=False)]

scores = {
    "mahalanobis": EmpiricalCovariance().fit(Z0).mahalanobis(Zte),
    "gmm10": -GaussianMixture(10, random_state=SEED).fit(Z0).score_samples(Zte),  # -log p(x)
    "knn5": NearestNeighbors(n_neighbors=5).fit(ref).kneighbors(Zte)[0].mean(axis=1),
}
print({k: metrics(y_test, s) for k, s in scores.items()})
plot_pr(y_test, scores)
```


---

### Cobertura de fuentes

| Fuente | Rango | Cubierto | Notas |
|---|---|---|---|
| slide p10 | portada "Distancias" | sí | §0 |
| slide p11 | "A mayor distancia, más anómalo" | sí | §1 |
| slide p12 | Euclídea (L2, RMSE) | sí | §2, fórmula y diagrama |
| slide p13 | Manhattan (L1, MAE) | sí | §3, fórmula y grid |
| slide p14 | Lp | sí | §4, fórmula y bolas unidad p=1…∞ |
| slide p15 | L∞, p<1, "p=1 (L0)" | sí | §4; posible errata "p=1 (L0)" señalada |
| slide p16 | Coseno | sí | §5 |
| slide p17 | Mahalanobis, matriz de covarianza | sí | §6, espacio nativo vs euclídeo |
| slide p18 | portada "Probabilidad" | sí | §0 |
| slide p19 | "A menor probabilidad, más anómalo" | sí | §9 |
| slide p20 | Discreta / continua, $\Pr[a\le X\le b]$, volumen = P(A) | sí | §10 |
| slide p21 | Tabla discreta/continua (normalización, media, varianza) | sí | §10 |
| slide p22 | Normal, fijar nº de sigmas (95 %/99 %, 1,96σ/2,58σ) | sí | §12, todos los valores de la figura |
| slide p23 | Multidimensional, cov<0, ≈0, >0 | sí | §13 |
| slide p24 | Probabilidad bajo transformaciones y=g(x) | sí | §13 |
| slide p25 | Modelos (o combinaciones), grafo de distribuciones, enlace Wikipedia | sí | §14 (listado parcial de nodos del grafo) |
| slide p26 | portada "Teoría de la información" | sí | §0 |
| slide p27 | Información vs Anomalía | sí | §15 |
| slide p28 | Shannon 1948, $I(x)\propto 1/p(x)$ | sí | §15 |
| slide p29 | Entropía | sí | §16 |
| slide p30 | Información mutua + diagrama de Venn (H, T, I) | sí | §17 |
| slide p31 | KL, $H(P,Q)-H(P)$, figuras | sí | §18 |
| slide p32 | Paper RBIG (arXiv 2010.03807) | sí | §19 |
| transcripción | 600–668 [01:02:03]–[01:08:46] | sí | §1 excepciones (pocas muestras, frontera, SVM/MLP) |
| transcripción | 668–713 [01:08:55]–[01:13:20] | sí | §1 incertidumbre aleatoria/epistémica |
| transcripción | 713–880 [01:13:24]–[01:30:30] | sí | §2–§6 tipos de distancia |
| transcripción | 881–943 [01:30:38]–[01:36:53] | sí | §7–§8 distancia a qué y umbral; ejemplo de series temporales (figura no disponible) |
| transcripción | 944–1043 [01:36:59]–[01:47:08] | sí | §9–§11 probabilidad, notas Inglaterra |
| transcripción | 1043–1086 [01:47:08]–[01:52:01] | sí | §10, §12 PDF, sigmas, p-value |
| transcripción | 1087–1188 [01:52:09]–[02:15:46] | sí (sin contenido) | Descanso; ruido de transcripción y charla sobre ciudades/fútbol inglés |
| transcripción | 1189–1245 [02:15:55]–[02:20:46] | sí | Recapitulación, §12–§14 |
| transcripción | 1246–1313 [02:20:52]–[02:26:43] | sí | §15–§16 |
| transcripción | 1314–1384 [02:26:51]–[02:33:02] | sí | §17–§19 |
| AD_Data.md | completo (celdas 0–10) | sí | Enunciado: descarga, carga cardio, "Plotea los datos" |
| AD_Data_clase.md | completo (celdas 0–21) + 4 figuras | sí | Pairplots, LOF, métricas (accuracy 0,89 / recall 0,05) |
| AD_Distancias.md | completo (celdas 0–25) + figuras | sí | Enunciados |
| AD_Distancias_clase.md | completo (celdas 0–38) + 9 figuras | sí | Celdas 5 y 32 sin código (sólo salida); ejercicios de imagen sin solución |
| AD_Probabilidad.md | completo (celdas 0–26) | sí | Enunciados (u=0, s=1 "reales") |
| AD_Probabilidad_clase.md | completo (celdas 0–42) + 14 figuras | sí | Celda 27 sin código; varios ejercicios sin solución |

**Huecos / dudas.**
- El ejemplo de series temporales con triángulos rojos ([01:34:34]–[01:36:32]) se apoya en una figura que no está en ninguna slide (revisadas p01–p57); probablemente es del paper de Twitter.
- Varias celdas de las versiones _clase no conservan el código (Distancias 5, 32; Probabilidad 27); se describen por la figura.
- Salidas de AD_Distancias celdas 9 y 11 (p=1e7) inconsistentes con el código (overflow entero / edición posterior), señalado como nota del redactor.
- Figura AD_Data celda12_2 no corresponde aparentemente al dataset cardio.


---

## Parte B — clasificación de una clase, regresión/reconstrucción, algoritmos clásicos y herramientas (sklearn, PyOD)

> **Fuentes**
> - Diapositivas `slides_clasicas/p33.png` – `p48.png` (Deteccion_de_anomalias.pdf), revisadas como imagen; cruzadas con `p07`–`p09` (enfoques, soluciones, librerías).
> - `transcripcion.txt` líneas 1384–1769 ([02:33:02] – [03:07:04]); cruces puntuales con líneas 650–668 [01:07:04]–[01:08:55] y 2140–2150 [03:42:57]–[03:43:51].
> - `notebooks/AD_reconstruccion.md` (enunciado) y `AD_reconstruccion_clase.md` (solución) + `AD_reconstruccion_clase_figs/`.
> - `notebooks/AD_Tools_Sklearn.md` (enunciado) y `AD_Tools_Sklearn_clase.md` (solución) + `AD_Tools_Sklearn_clase_figs/`.
> - `notebooks/AD_Tools_Pyod.md` (no hay versión `_clase`; trae el ejemplo ECOD ejecutado) + `AD_Tools_Pyod_figs/`.
> - Para la aplicación: `notebooks/taller_deteccion_anomalias_partida.md` (notebook de partida del profesor).
>
> **Convenciones.** Lo que no está en las fuentes y procede de conocimiento propio va marcado **(nota del redactor)**. Las afirmaciones de API de scikit-learn se han comprobado ejecutando código (sklearn 1.8.0); PyOD no está instalado en el equipo de revisión, así que su API se documenta según la documentación de PyOD 2.x y la salida del notebook (pyod 2.0.3).
> Transcripción automática interpretada: "una playa"/"aulayer"/"lyers"/"outlights" = outlier; "clausterling"/"clustream"/"clastream" = clustering; "viviescan"/"iviescan" = DBSCAN; "gremis"/"Kemilks"/"gamings" = k-means; "Smoke" = SMOTE; "cauce misión model" = Gaussian mixture model; "expectation a seversation"/"una semissation" = expectation maximization; "autorreversión"/"revisión lineal" = regresión / regresión lineal; "vector conductation" = vector quantization; "Oversompley" = oversampling; "S. V. M"/"sevm" = SVM; "One, clases, Bien. S. V: M" = One-Class SVM; "radio real" = (kernel) radial.

---

### 1. La última familia: plantearlo como clasificación (de una clase o binaria)

**Idea.** Para "completar nuestro proceso de soluciones" (tras distancia, probabilidad e información; slide p08 lista *Distancias · Probabilidad · Información · Regresión/Reconstrucción · Kernels · Redes*), la última alternativa es "no hacer nada de esto" y "no preocuparos ni de umbrales ni de nada por el estilo": quita complejidad al diseño de todo el proceso. Se recupera el nombre del inicio de la clase, **clasificación de una clase** (*one-class classification*). Se coge "vuestro clasificador favorito, el que sea" — "muy típico para esto es SVM" — y se entrena:
- **de una clase**, si solo hay datos normales, o
- **de dos clases**, "asumiendo que esta es otra clase", si "tenéis ya algunos outliers anotados" además de los datos normales: entonces "es una de las cosas más fáciles de hacer".

Con etiquetas, "puedo detectar outliers, incluso aunque los outliers tengan una distribución que parezca que no tienen nada en común entre ellos". En realidad se intenta **capturar las propiedades de los datos de verdad** y "me importa menos lo que esté pasando con los outliers".

**Detalle.**
- Es lo mismo que en los otros tres pasos: "Tengo una noción de distancia, y sé que hay una frontera", y todo lo que quede fuera son outliers.
- Problema: "normalmente no tengo etiquetas para esa clase nueva, pero si las tengo, es una de las cosas más fáciles de hacer".
- **Caso extremo, sin haber visto nunca un outlier**: un modelo de clasificación en producción. Para que el sistema sea robusto, "cualquier cosa que se aleje mucho va a ser un outlier y voy a tener una alerta". Se entrena un clasificador que **ponga todos mis puntos dentro de una esfera**; lo que quede fuera será outlier en el futuro.
- Problema de la esfera mínima: si llegan muestras justo en el borde puede haber problemas. Por eso, "con una SVM lo que se suele hacer es incluir un **margen** en el que acepto errores": la esfera más pequeña que contiene los datos es una, pero "voy a considerar una un poquito más grande", y "eso **depende del usuario**", es decir, cuánto de grande.
- **One-Class SVM**: "un problema de optimización un poco compleja": "dame la esfera en el espacio que mejor incluya todos mis datos", es decir, "calcular cuál es el volumen y la posición de esa esfera". El kernel puede ser lineal, radial (RBF) o el que se quiera.
- Generalización: "Las siguientes soluciones que vamos a ver son todas de este estilo. Yo elijo un modelo porque me ha gustado, creo que puede coger bien a mis datos". También vale el caso no supervisado: cualquier modelo de clustering "va a hacer algo de este estilo". Si encuentro uno o varios clusters, asumo que "todo lo que quede fuera es un problema".
- Cruce con [01:07:24]–[01:08:46]: con una SVM o un MLP binario, "cuanto más lejos estoy de la frontera, menos me tengo que preocupar". Pero un punto muy lejos de la frontera puede ser "un ejemplo de otra clase que no he visto nunca" y el modelo lo clasifica con plena confianza. Esto justifica la esfera alrededor de los datos (ver `00_fundamentos.md` §9).
- Cruce con [03:43:19]–[03:43:33]: antes de 2017 se usaba "cualquier clasificador": árboles de decisión ("es fácil separar un punto") y, en SVM, "normalmente será el de One-Class SVM".
- (nota del redactor) **Formulaciones.** SVDD (Tax & Duin), que es la "esfera" del profesor: $\min_{R,c,\xi} R^2 + \frac{1}{\nu n}\sum_i \xi_i$ s.a. $\|\phi(x_i)-c\|^2 \le R^2+\xi_i,\ \xi_i\ge 0$. La `OneClassSVM` de sklearn usa la formulación de Schölkopf: un hiperplano que separa los datos del origen en el espacio de características. Con kernel RBF, que tiene $k(x,x)$ constante, es equivalente a SVDD. $\nu\in(0,1]$ es **cota superior de la fracción de puntos de entrenamiento que quedan fuera** y cota inferior de la fracción de vectores soporte; es el "tamaño del margen que decide el usuario". Comprobado: con `nu=0.05` en Iris, `predict` marca el 5,3 % como −1.

**Ejemplos del profesor.** Dibujo en pizarra: puntos dentro de un círculo ("aunque no lo parezcan, se supone que es un círculo/esfera") con un segundo círculo algo mayor como margen; muestras nuevas en el borde.

**Código.** No hay OCSVM en ningún notebook ni slide (ver "Cobertura"). (nota del redactor) Equivalente correcto en sklearn:
```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ocsvm = make_pipeline(StandardScaler(),
                      OneClassSVM(kernel="rbf", gamma="scale", nu=0.01))  # nu ≈ fracción tolerada fuera
ocsvm.fit(X_normales)                       # solo datos normales (one-class)
score = -ocsvm.decision_function(X_eval)    # decision_function > 0 dentro (normal) → se niega
# equivalente para ranking: -ocsvm.score_samples(X_eval)  (decision_function = score_samples - offset_)

# Escalable a >100k filas: aproximación del kernel + SGD (lineal en n)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
ocsvm_big = make_pipeline(StandardScaler(),
                          Nystroem(kernel="rbf", gamma=1/21, n_components=300, random_state=0),
                          SGDOneClassSVM(nu=0.01, random_state=0))
score = -ocsvm_big.fit(X_normales).score_samples(X_eval)
```

*Fuente: transcripción [02:33:07]–[02:38:30] (líneas 1385–1447); cruces [01:07:24]–[01:08:46] y [03:43:19]–[03:43:33]; slide p08 ("Kernels"). No hay slide específica.*

---

### 2. Regresión / reconstrucción: "a mayor error de predicción, más anómalo"

**Idea.** Slide p34: **"Mayor error de predicción, más anómalo."** Con un modelo sobre los datos (clasificador, regresión, reconstrucción), "cuanto más se equivoca este modelo y más lejos esté de lo que incluía mis datos, más anómalo". Se ilustra en clasificación, pero "exactamente igual" en regresión: "cuanto más error tenga, más pinta va a tener eso de ser anómalo".

**Detalle / fórmulas.**
- Score: $a(x) = \| x - \hat{x} \|$ (reconstrucción) o $a(x,y) = |y - f(x)|$ (regresión).
- Frente a los métodos de distancia, en los que se calcula la distancia de cada punto a la nube azul y se pone un umbral, aquí "lo vamos a hacer más fácil todavía": se pone un modelo y se mira el error de cada punto respecto a él.
- **Ventaja**: "no nos hemos tenido que preocupar de decidir qué densidad de probabilidad es la adecuada para modelar estos datos, porque el modelo lo hacemos nosotros".
- **Pega: la complejidad del modelo.** "Si el modelo es muy sencillo, como por ejemplo una regresión lineal, puedo tener puntos aquí abajo que están muy lejos del modelo y en realidad no son anomalías. Entonces el modelo tiene que ser suficientemente complicado para capturar el comportamiento que nos interesa". "Vais a necesitar saber si vuestro modelo de verdad es bueno". Lo mismo pasa con la esfera de la OCSVM: si representa bien los datos, "lo hace perfecto"; si no, "no conseguís nada".
- **Procedimiento práctico** (slide p37): proyectar todos los puntos al modelo de regresión, que "a lo mejor no es el más adecuado del mundo para este tipo de datos", "calcular la distancia del punto al modelo de regresión o del punto a la frontera, y luego decidir qué va a ser el suficiente error para ser anomalía… depende de nosotros".
- **PCA como modelo de reconstrucción** (slide p36): las dos componentes principales hacen de "modelo"; "puntos que estén lejos de cualquiera de ellas en principio van a ser anomalías". Un punto alejado pero bien alineado con la dirección importante "no está claro que sea una anomalía"; en cualquier caso se decide con un umbral.
- (nota del redactor) Reconstrucción PCA con $k$ componentes: $\hat x = \mu + W_k W_k^\top (x-\mu)$; error $\|x-\hat x\|^2 = \sum_{j>k} z_j^2$, la suma de cuadrados de las componentes descartadas. El notebook usa $|z_2|$ (2D, k=1) y $|z_3|+|z_4|$ (Iris, k=2), que es la versión L1 de lo mismo.

**Incertidumbre: zona densa frente a zona sin datos (slide p38).** Dos puntos nuevos (rojo hueco) y su reconstrucción (rojo lleno) sobre la recta del PCA:
- Punto 1: la reconstrucción cae **en una zona de alta densidad**, pero el error es muy grande, así que con un umbral en el error saldrá outlier.
- Punto 2: la distancia al modelo es **muy pequeña**, pero la reconstrucción cae en un extremo de la recta **sin datos**.
- "¿Cuál pensáis que es más probable que sea un outlier?". "Es el ejemplo de antes de **incertidumbre epistémica y aleatoria**". "Si los 2 fueran outliers nadie tendría ningún problema con ello, pero depende de vuestra interpretación".
- (nota del redactor) El punto 2 es incertidumbre epistémica: no hay datos en esa región, y el error de reconstrucción no lo detecta. Combinar el error de reconstrucción con la densidad en el espacio latente (p. ej. una GMM sobre las componentes retenidas) cubre ambos casos. Ver `00_fundamentos.md` §10.

**Ejemplos (diapositivas, revisadas).**
- **p35, tres paneles.** (1) Nube oscura en forma de arco y dos grupitos de ~8 puntos amarillos sueltos, uno por encima del arco a la izquierda y otro por debajo, que son las anomalías. (2) Regresión lineal (recta naranja) con el color como error: los puntos anómalos salen **oscuros, con error bajo**, porque la recta pasa cerca de ellos, y en cambio los **extremos del arco** salen claros, azul a la izquierda y amarillo/naranja a la derecha: error alto en puntos normales porque el modelo es demasiado simple. (3) Curva no lineal que sigue el arco: ahora solo los puntos sueltos tienen error alto (verde/turquesa) y la nube queda oscura. El extremo inferior izquierdo sale algo azulado: "a lo mejor puede ser una anomalía, o a lo mejor no tenemos suficientes datos ahí".
- **p36**: nube de puntos rojos con los ejes PC1 (diagonal ascendente) y PC2 (perpendicular). "Lo bueno de este ejemplo es que aquí son todos rojos, no sabemos cuál es la anomalía".
- **p37**: recta verde (ejes 20–120 × 140–240). Los puntos originales (huecos, negros) se proyectan con líneas azules discontinuas a sus proyecciones (puntos azules sobre la recta). Hay un punto verde grande (la media) y un punto rojo con una flecha negra hacia su proyección. "Lo que tengo que hacer es calcular la distancia del punto al modelo".
- **p38**: "Original points" (azules), "New points" (rojo hueco 1 y 2), "Reconstruction" (rojo lleno); enlace https://setosa.io/ev/principal-component-analysis/.

**Código del profesor** (`AD_reconstruccion_clase.md`, celdas 2–14):
```python
import numpy as np
import matplotlib.pyplot as plt
N = 1000
x = np.random.rand(N,1)                              # valores originales
y = x + 0.1*np.random.randn(x.shape[0],x.shape[1])   # "predichos" (fake): y = x + ruido (σ=0.1)
y[0] = y[0]*2.5                                      # anomalía inyectada

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x,y)
y_hat = model.predict(x)

Err = np.abs(y-y_hat)          # score de anomalía = |residuo|
Anom = Err
plt.scatter(x,y, s=1, c=-Anom, cmap='Reds')   # OJO: c=-Anom → los MÁS anómalos salen MÁS CLAROS

Anom = Err>0.1                  # umbral fijo arbitrario
plt.scatter(x,y, s=1, c=Anom, cmap='Reds')

plt.hist(y-y_hat,bins=20)       # histograma del residuo

th = 3*np.std(y-y_hat)          # umbral 3σ del residuo
Anom = Err>th
plt.scatter(x,y, s=1, c=Anom, cmap='Reds')

# Lo mismo con PCA: la 2.ª componente es la distancia ortogonal a la recta (PC1)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = np.zeros((x.shape[0],2)); data[:,0] = x[:,0]; data[:,1] = y[:,0]
X_pca = pca.fit_transform(data)
Err = np.abs(X_pca[:,1])
Anom = Err>0.1
plt.scatter(x,y, s=1, c=Anom, cmap='Reds')
# celda 14: lo mismo coloreando con c=-Anom (mismo error de signo)
```
Figuras (`AD_reconstruccion_clase_figs`):
- `celda09_3.png`: con `c=-Anom` los puntos bien ajustados salen oscuros y la anomalía (≈(0.87, 1.83)) es casi invisible. Es un **error de signo** en la visualización.
- `celda10_4.png`: el umbral 0.1 marca bastantes puntos del borde de la banda de ruido (0.1 = 1σ).
- `celda11_5.png`: histograma del residuo: campana en [−0.3, 0.3] y una barra aislada en ≈0.95 (la anomalía).
- `celda12_6.png`: el umbral 3σ deja solo 2 puntos, la anomalía (≈0.87, 1.83) y otro (≈0.83, 1.17) en la cola del ruido.
- `celda13_7.png`: PCA con umbral 0.1 marca ~150 puntos a ambos lados de la banda (umbral demasiado bajo).
- `celda14_8.png`: PCA con `c=-Anom`: la nube sale roja/oscura y la anomalía blanca e invisible (mismo error de signo).

**Sobre el umbral 3σ** (duda del extractor, resuelta). `th = 3*np.std(y-y_hat)` con `Err = |y-y_hat| > th` es una regla **z-score bilateral sobre el residuo**: como la regresión con intercepto deja un residuo de media ≈0, equivale a $|r-\bar r|/\sigma_r > 3$. Bajo ruido gaussiano marca ~0,27 % de los puntos normales (≈2,7 de 1000) aunque no haya anomalías: es la regla de las "sigmas" de la parte de probabilidad (slide p22, "fijar un número de sigmas"). (nota del redactor) Matices: $\sigma$ se estima **incluyendo** las anomalías, lo que la infla, y lo robusto sería usar la MAD: $1{,}4826\cdot\text{mediana}(|r-\text{mediana}(r)|)$. Además, solo es válida si el residuo es aproximadamente gaussiano y homocedástico.

*Fuente: slides p33–p38 · transcripción [02:38:40]–[02:44:51] · notebook AD_reconstruccion_clase celdas 2–14.*

---

### 3. Regresión polinómica (complejidad del modelo)

**Idea.** Ejercicio que materializa la slide p35: datos parabólicos $y=x^2+\varepsilon$ en los que una recta fallaría en los extremos. Con `PolynomialFeatures(degree=2)` el modelo captura la forma y el residuo señala las anomalías.

**Código del profesor** (`AD_reconstruccion_clase.md`, celdas 16–22):
```python
N = 1000
sigma = 0.25
x = 2*((np.random.rand(N,1))-0.5)                    # x en [-1,1]
y = x**2 + sigma*np.random.randn(y.shape[0],y.shape[1])   # reutiliza la forma (1000,1) del y anterior
y[0] = y[0]+0.5                                      # dos anomalías inyectadas (±0.5 = ±2σ)
y[1] = y[1]-0.5

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)                  # "ajusta el grado según sea necesario"
X_train_poly = poly.fit_transform(x)

model2 = linear_model.LinearRegression()
model2.fit(X_train_poly,y)
y_hat2 = model2.predict(X_train_poly)
Err2 = np.abs(y-y_hat2)

th = 3*np.std(y-y_hat2)
plt.scatter(x,y, s=1, c=Err2, cmap='Reds')           # celda 21: color = error (signo correcto)
Anom = Err2>th
plt.plot(x,y,'.')
plt.plot(x[Anom,],y[Anom,],'ro')                     # celda 22: anomalías en rojo
```
Figuras: `celda16_9.png`, parábola ruidosa con una nube ancha; `celda19_10.png`, ajuste en forma de U que sigue la nube; `celda21_11.png`, color = error, con los puntos alejados de la U en rojo oscuro; `celda22_12.png`, el umbral 3σ deja **3 puntos rojos**: (≈−0.9, 0.06), (≈0.2, 1.02) y (≈0.63, −0.33).

**Interpretación** (revisada; nota del redactor). Con σ = 0,25, el umbral es ≈0,75 y las anomalías inyectadas de ±0,5 equivalen a 2σ. Cada una supera el umbral solo si el ruido la empuja ≥0,25 más en su dirección, lo que ocurre con una probabilidad de ~16 %. A la vez se esperan ~2,7 falsos positivos por pura cola gaussiana. Los residuos de los tres puntos marcados son aproximadamente +0,98 (≈3,9σ, compatible con `y[0]+0.5` más ruido positivo), −0,73 y −0,75 (≈3σ). Sin imprimir los índices no se puede saber cuáles son los inyectados. La lección del ejercicio es doble: el modelo tiene que tener la complejidad adecuada (la recta fallaría en los extremos, como en p35), y una anomalía de 2σ es indistinguible del ruido con un umbral de 3σ.

La parte "con regresión lineal" de este ejercicio no está resuelta (la celda 20 está vacía). Se haría con `LinearRegression().fit(x,y)` sobre `x` sin transformar: el residuo saldría alto en los extremos |x|≈1 y bajo en el centro, que es el panel 2 de p35. En el enunciado (`AD_reconstruccion.md`, celda 6) falta definir `sigma`; en la solución vale `sigma = 0.25`.

*Fuente: notebook AD_reconstruccion_clase celdas 15–22; slide p35.*

---

### 4. Reconstrucción con PCA en Iris (número de dimensiones)

**Idea.** Quedarse con 1, 2, 3 o 4 componentes y medir el error de reconstrucción. Con 4 componentes la reconstrucción es perfecta (error 0); con menos, aumenta el error de todos los puntos.

**Código del profesor** (celdas 25–31):
```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data                                        # (150,4)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Opción A (celda 29): score = magnitud en las componentes descartadas (3.ª y 4.ª)
A = np.abs(X_pca[:,3]) + np.abs(X_pca[:,2])

# Opción B (celda 30): reconstrucción explícita con 2 componentes (las otras a 0)
X_pca2 = np.zeros((X.shape[0],4))
X_pca2[:,0] = X_pca[:,0]
X_pca2[:,1] = X_pca[:,1]
A = np.sum(np.abs(X-pca.inverse_transform(X_pca2)),axis=1)   # error L1 de reconstrucción

plt.scatter(X_pca[:,0],X_pca[:,1], s=1, c=A, cmap='Reds')
```
Figuras: `celda26_13.png`, pares de dimensiones originales (setosa separada, versicolor y virginica solapadas); `celda28_14.png`, lo mismo en el espacio PCA; `celda29_15.png`/`celda30_16.png`, coloreado por error: en el plano PC1–PC2 el error no tiene estructura visible porque es ortogonal a ese plano, y en el plano PC3–PC4 los puntos alejados del origen son los rojos oscuros (el error *es* la distancia en esas componentes); `celda31_17.png`, PC1–PC2 coloreado por A.

(nota del redactor) Para el caso general de *k* componentes: `pca_k = PCA(n_components=k).fit(X)` y `A = ((X - pca_k.inverse_transform(pca_k.transform(X)))**2).sum(1)`. Variando k=1..3, cuanto menor es k, más error tiene todo el mundo y menos discriminante es el score.

*Fuente: notebook AD_reconstruccion_clase celdas 23–31.*

---

### 5. Uso de las etiquetas de anomalía: validar umbrales, no entrenar

**Idea.** Pregunta de un alumno [02:44:56]: si conocemos el tipo de anomalía y tenemos casos históricos, ¿se usarían como conjunto de validación para fijar el umbral? Respuesta [02:45:03]–[02:45:19]: "De hecho es lo que os decía antes, es como las utilizaría yo. **Si yo tuviera anomalías y son muy pocas, nunca las utilizaría para entrenar el modelo; las utilizaría para validar las decisiones que tomaba con los umbrales**".

**Detalle.**
- Excepción [02:45:26]: "si tenéis tantas anomalías como datos buenos, entonces sí las podéis utilizar para entrenar modelos que sean capaces de entender dónde están las anomalías, pero suele ser raro".
- Motivo [02:45:39]: "las anomalías, por definición, hasta que no tienen más densidad y se convierten en su propia clase, siempre es algo que puede aparecer en cualquier sitio del espacio, sin orden".
- [02:45:52]–[02:46:02]: "mientras estáis en esa situación donde el número de anomalías es tan pequeño con respecto al conjunto de datos… depende de lo que queráis hacer, pero yo las utilizaría [para validar] siempre".
- [02:51:55]–[02:52:06]: los algoritmos clásicos "suelen asumir que yo no he visto nunca los outliers antes" o que todo es no supervisado; "si las tenéis, las podéis utilizar para lo que queráis, o bien para validar, o bien para entrenar".

*Fuente: transcripción [02:44:56]–[02:46:02] (líneas 1512–1522) y [02:51:49]–[02:52:06].*

---

### 6. Desbalanceo de clases: undersampling, oversampling, SMOTE

**Idea.** Ya se vio en clasificación con clases muy desbalanceadas ("2 imágenes de perro y 5000 de gato"). La detección de anomalías puede verse como un **caso límite** de desbalanceo: una clase (gato) muy bien caracterizada y muchas otras (perro, caballo…) de las que hay muy pocas muestras.

**Detalle.**
- Se podría hacer un clasificador multiclase ("no hay nada que lo impida"), pero "estaría utilizando los mismos recursos [complejidad] para definir la frontera de algo que conocéis muy bien con respecto a algo que [apenas conocéis]", lo cual "no tiene mucho sentido".
- **Undersampling**: "quitar muestras [de la mayoritaria] hasta que me quedo con [tantas como la minoritaria]" y queda balanceado. Pega: "eliminar información que ya tengo no es bueno porque estoy tirando cosas a la basura".
- **Oversampling**: "añadir muestras con ruido, versiones sintéticas, hasta que tenga las mismas". Se puede hacer bien estimando la varianza de la clase y muestreando con esa varianza "para que tenga la misma pinta". Pega: los datos sintéticos tienen "asunciones muy fuertes sobre lo que estoy viendo".
- **SMOTE** ("Smoke" en la transcripción) [02:49:48]–[02:50:04]: "es una combinación de las 2": "coge puntos reales, los une e intenta muestrear en la zona que los une en función de la distancia que los separa"; "hace las cosas un poco más inteligente".
- Aviso [02:48:55]–[02:49:10]: "todo este tipo de soluciones valen también para anomaly detection, pero siempre asumir que la distribución de los outliers en principio va a ser una **combinación de un millón de clases que nunca habéis visto**". "No tiene por qué ser que todos los outliers vengan de una nueva clase muy bonita, en una zona del espacio".
- **"Estado del arte"** [02:49:21] (duda del extractor, resuelta). La frase "Luego tenéis estado del arte de esto que se llama… No sé si lo habréis visto" abre una enumeración interrumpida. Tras repasar las pegas del under y del oversampling, el profesor la cierra con "lo que suelo hacer es una cosa que se llama Smoke", y su descripción (interpolar entre puntos reales en función de la distancia) es la definición de SMOTE. Por contexto, el "estado del arte" del remuestreo al que se refiere es SMOTE.
- (nota del redactor) SMOTE (Chawla et al., 2002): $x_{new} = x_i + \lambda (x_{nn} - x_i)$, $\lambda\sim U(0,1)$, con $x_{nn}$ uno de los $k$ vecinos más cercanos **de la misma clase minoritaria**. En sentido estricto es oversampling, no una combinación de under y over. Lo que el profesor llama "combinación de las 2" es más bien que mezcla datos reales con sintéticos; las combinaciones literales existen (SMOTEENN, SMOTETomek en `imblearn.combine`). Implementación: `imblearn.over_sampling.SMOTE(k_neighbors=5)` (paquete `imbalanced-learn`). Alternativa sin remuestreo: `class_weight='balanced'` / `scale_pos_weight`. Ver «Aplicación» A.3 para usarlo sin fugas en CV.

*Fuente: transcripción [02:46:08]–[02:50:07] (líneas 1523–1569).*

---

### 7. Resumen: "en realidad todo es lo mismo"

Para cerrar regresión [02:50:11]–[02:51:35]:
- **Modelo (regresión/clasificación)**: "elegís un modelo y el modelo lo hace todo por vosotros; si el modelo está mal elegido, mal".
- **Probabilidad**: "elegís una distribución; si la distribución no es la de los datos, mal".
- **Distancia**: "elegís una distancia; si la distancia no es la que es, por ejemplo el Pac-Man, pues os van a matar" (ejemplo de la distancia en un espacio periódico tipo Pac-Man).
- **Teoría de la información**: "requiere que sepáis las distribuciones muy bien para que luego os podáis tirar [a calcular] la entropía o la información mutua".
- "En realidad todo es lo mismo. Y todos tienen ventajas e inconvenientes, pero siempre va a acabar dependiendo de vosotros decidir umbrales y decidir qué aproximación creéis que es mejor".
- Sobre los algoritmos clásicos, "específicamente diseñados" para outliers: "con todo lo que conocéis deberíais poder resolver estos problemas en la práctica sin tener que utilizar nada nuevo. Esto son simplemente ideas de gente que estuvo en el pasado, que les fue bien, pero que no tiene que ser por donde tiréis".

*Fuente: transcripción líneas 1570–1586.*

---

### 8. Algoritmos clásicos: índice (slides p39–p40)

Slide p39: portada "Algoritmos clásicos". Slide p40 "INDICE":
- K-means, VQ y GMM
- DBSCAN (1996)
- LOF (2000)
- Isolation Forest (2008)

El profesor da por conocidos k-means y DBSCAN ("son de los tradicionales de clustering, imagino que los conoceréis") y recuerda que estos métodos "suelen asumir que yo no he visto nunca los outliers antes de que me aparezcan en la práctica, o que es todo no supervisado" [02:51:39]–[02:52:00].

---

### 9. K-means / clustering (slide p41)

**Idea.** El clustering sirve como modelo de los datos normales: una vez caracterizadas "mis nubes en las que confío", cuando llegan muestras nuevas "puedo hacer medidas de distancia con cualquier muestra y saber qué pinta tiene". Mejora sobre la distancia pura: "no solo divide los datos en [anómalo] o no, sino que estamos buscando **estructura** dentro de nuestros datos".

**Detalle / problemas.**
- "Normalmente **no sé cuántos clusters busco**", y eso hace que "nada sea tan bonito como aquí"; además "el clustering te puede llevar a ver cosas que no existen, con lo cual se trata con cuidado".
- Ejemplo: "alguien ha dicho que tiene que haber 10 clusters" y k-means los dibuja como "mini nubes de puntos". K-means "nunca va a solapar las nubes" y todo queda "perfecto" (partición de Voronoi), aunque la estructura sea artificial.
- Al medir la distancia de una muestra nueva hay que decidir: ¿la media de las distancias a los 3 clusters o solo la más cercana? Son los "mismos problemas que teníamos antes con los métodos de distancia".
- Funcionaría mejor con selección automática del número de clusters, que existe en algunos casos (clustering **jerárquico**, que aglomera desde tantos clusters como muestras hasta uno; espectral). "En casi todos tenéis que decidir de antemano [k] y no hay ninguna regla para ello".
- Lectura recomendada: *Clustering: ciencia o arte* (en inglés), ~50 páginas, "muy bien escrito". La falta de objetivos claros del clustering obliga a mucho input del usuario. (nota del redactor: probablemente von Luxburg, Williamson y Guyon, "Clustering: Science or Art?", JMLR W&CP 2012.)
- (nota del redactor) Objetivo: $\min_{\{\mu_k\}} \sum_i \min_k \|x_i-\mu_k\|^2$; score de anomalía $a(x)=\min_k\|x-\mu_k\|$ (`KMeans.transform(X).min(axis=1)`).

**Slide p41 ("Clustering").** Tres figuras: "Ideal Clustering" (3 nubes bien separadas azul/verde/naranja con centroides negros); "Data in 5 Clusters" (5 grupos, varios solapados); una nube grande partida en 10 clusters (leyenda 0–9) a modo de teselado artificial.

*Fuente: slides p40–p41 · transcripción [02:51:39]–[02:55:37] (líneas 1587–1631).*

---

### 10. Vector Quantization (slides p42–p43)

**Idea.** Las líneas azules definen regiones del espacio (celdas de Voronoi) y los puntos rojos son centroides. "Lo que puedo hacer con clustering, que está muy bien, es que a partir de ahí puedo tirar los puntos a la basura y quedarme con los puntos rojos", "una representación bastante sintética de todo mi conjunto de datos". Después se calculan "las distancias de cualquier nuevo punto a esos elementos" ignorando todo lo demás. El profesor menciona "un artículo muy bonito" que hace VQ en el espacio latente de un autoencoder (nota del redactor: VQ-VAE, van den Oord et al., 2017).

**Slide p43, "En la práctica":**
- Distintas aproximaciones: i.e. **Lloyd** (1. asigna y 2. calcula la media).
- **Depende de la inicialización**: i.e. random.
- Enlaces: https://clustering-visualizer.web.app/kmeans · https://en.wikipedia.org/wiki/K-means_clustering

**Figura (p42).** Nube gaussiana gris 2D con ~16 centroides rojos y las fronteras de Voronoi en azul.

(nota del redactor) Ventaja práctica: con k centroides, puntuar una muestra nueva cuesta $O(kd)$ en vez de $O(nd)$ (kNN), lo que escala a 170k filas. `sklearn.cluster.KMeans(n_clusters=k, n_init=10)` o `MiniBatchKMeans`; en sklearn la inicialización por defecto es `k-means++`, no aleatoria.

*Fuente: slides p42–p43 · transcripción [02:55:47]–[02:56:34] (líneas 1632–1643).*

---

### 11. DBSCAN (slide p44)

**Idea.** *Density-based spatial clustering of applications with noise* (KDD 1996: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf), "uno de los métodos más elegantes de los años 90" en clustering. Se mira **cómo de conectados** están los puntos, "asumiendo que puntos que están muy conectados o muy cerca van a pertenecer a la misma estructura". La conexión se define como "densamente alcanzables" ("una frase que no sé si tiene sentido en ningún idioma").

**Slide.**
- **Punto núcleo (A)**: si n puntos están a distancia ε.
- **Puntos (densamente-)alcanzables (B y C)**: si no son punto núcleo pero están a una distancia menor de ε de un punto núcleo.
- **Ruido (N)**: si no es alcanzable.

**Explicación oral.** "Pongo una circunferencia alrededor de cada uno de los puntos y veo qué puntos han caído dentro"; luego miro "cuántas circunferencias solapan para cada punto: si son muchas, está dentro de los vecinos de muchos otros; si son pocas o ninguna, no tienen mucha conexión". Ventaja: B, aunque esté lejos, "es el final de uno de estos caminos, está bien conectado con los demás" y no es outlier; N no tiene vecinos y es outlier.
- **Hiperparámetro**: "vosotros tenéis que decidir cómo de grandes son las circunferencias", y de eso depende que N sea outlier: "si fuera un pelín más grande… hubiera dicho que es un punto dentro del mismo grupo sin ningún problema". "Siempre os falta algo que tenéis que poner vosotros; en este caso es el tamaño de la circunferencia" (ε). El otro parámetro, n, es `min_samples`. (nota del redactor) En sklearn, `min_samples` cuenta el propio punto.

**Figura (p44).** Puntos núcleo rojos (A entre ellos) con círculos de radio ε y flechas de alcanzabilidad; B y C amarillos (frontera) con círculos amarillos; N azul aislado, con un círculo que no contiene a nadie.

**Código del profesor** (`AD_Tools_Sklearn_clase.md`, celdas 14–17):
```python
from sklearn.cluster import DBSCAN
clf = DBSCAN(eps=0.5, min_samples=5)   # eps = radio, min_samples = n
clf.fit(X)                             # X = Iris (150,4), sin escalar
y = clf.labels_                        # etiqueta de cluster; -1 = RUIDO (anomalía)
plt.scatter(X[:,0],X[:,1],c=y)
```
Salida verificada: 2 clusters (0 = setosa, 1 = versicolor + virginica) y **17 puntos con etiqueta −1**. Uno está en setosa ((4.5, 2.3)); el resto, en los bordes del otro cluster ((7.7, 3.8), (7.9, 3.8), (5.0, 2.0), (7.7, 2.6)…). Figuras: `celda16_3.png` (2D) y `celda17_4.png` (todos los pares).

(nota del redactor) DBSCAN **no da score continuo**, solo la etiqueta −1/no −1, y **no tiene `predict`** para datos nuevos (verificado). Para AUPR un score binario es muy pobre. Un sustituto continuo en el mismo espíritu es la distancia al `min_samples`-ésimo vecino (`NearestNeighbors`), que es la "k-distance" usada para elegir ε.

*Fuente: slide p44 · transcripción [02:56:47]–[02:58:49] (líneas 1644–1668) · notebook AD_Tools_Sklearn_clase celdas 14–17.*

---

### 12. LOF: Local Outlier Factor (slides p45–p46)

**Idea (slide).** "Son anomalías los puntos que tienen una **densidad menor que sus vecinos**." Paper (2000): https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf; https://en.wikipedia.org/wiki/Local_outlier_factor. Es "de los que están diseñados específicamente para esto", con la misma idea de vecindad que DBSCAN.

**Explicación oral.** "Lo que puedo ir viendo es cuáles son mis vecinos para este punto A: si estoy mirando 3 vecinos más cercanos, cojo estos 3. Y ahora analizo para ellos su vecindad también". "Si todos mis vecinos tienen 3 vecinos cerca y yo los tengo muy lejos, es que algo raro pasa conmigo".
- Hiperparámetro: "el número de vecinos que quiero incluir dentro de mi circunferencia, y con eso defino el volumen" (k), frente al tamaño de la circunferencia (ε) en DBSCAN; "más o menos igual en motivación".
- (nota del redactor) $\text{reach-dist}_k(A,B)=\max(k\text{-dist}(B), d(A,B))$; $\text{lrd}_k(A)=1\Big/\Big(\frac{1}{|N_k(A)|}\sum_{B\in N_k(A)}\text{reach-dist}_k(A,B)\Big)$; $\text{LOF}_k(A)=\frac{1}{|N_k(A)|}\sum_{B\in N_k(A)}\frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}$. LOF≈1 es normal; LOF≫1, anómalo.

**Figuras.** p45: punto A a la izquierda con 3 líneas rojas a sus 3 vecinos, que están en un grupo denso con círculos discontinuos pequeños (su k-distancia), mientras que el círculo discontinuo de A es enorme. p46 añade un scatter 2D (Dim. 1 × Dim. 2, 0–100) con un cluster denso abajo a la izquierda y otro disperso arriba a la derecha, con círculos rojos de radio proporcional al LOF. Puntos aislados: 7.6259, 5.9645, 4.1668, 3.4745, 3.1361, 3.108, 2.7383, 2.5677. Puntos de borde de cluster: 1.9767, 1.4192, 1.3953, 1.358, 1.2764, 1.2681, 1.184.

**Código del profesor** (`AD_Tools_Sklearn_clase.md`, celdas 27–31):
```python
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20)   # k
y_pred = clf.fit_predict(X)                # -1 = outlier, 1 = inlier
plt.scatter(X[:,0],X[:,1],c=y_pred)
# se repite con n_neighbors = 4, 6 y otra vez 20
```
Figuras: `celda29_10.png` (k=4), ~8 outliers dispersos, algunos *dentro* de las nubes (muy locales, p. ej. (5.4, 3.4), (5.1, 3.3)); `celda30_11.png` (k=6); `celda28_9.png`/`celda31_12.png` (k=20), ~7 outliers en los extremos globales ((5.7, 4.4), (7.7, 3.8), (7.9, 3.8), (7.7, 2.6), (4.5, 2.3), (5.1, 2.5)). Conclusión: **k cambia qué se considera anómalo**; k pequeño es muy local y ruidoso, k grande más global.

(nota del redactor; API verificada en sklearn 1.8)
- `novelty=False` (por defecto): solo `fit_predict`, y el score de entrenamiento es `-clf.negative_outlier_factor_` (= LOF, mayor = más anómalo). **No existen** `score_samples`/`predict` para datos nuevos (AttributeError).
- `novelty=True`: `fit(X_train)` y luego `-clf.score_samples(X_new)` o `clf.predict(X_new)`; en este modo **no existe** `fit_predict`. No se debe puntuar con él el propio conjunto de entrenamiento.
- Con `contamination='auto'`, `offset_ = -1.5`, así que `predict` marca −1 cuando LOF > 1,5.
- Con muchos duplicados (nuestro caso: 4.167 filas) sklearn avisa de que los duplicados falsean el resultado si superan `n_neighbors`; conviene deduplicar el entrenamiento o subir k.

*Fuente: slides p45–p46 · transcripción [02:59:01]–[02:59:51] (líneas 1670–1679) · notebook AD_Tools_Sklearn_clase celdas 26–31.*

---

### 13. Escalabilidad de los métodos basados en vecindad

"Todos estos métodos que hemos visto hasta ahora, os daréis cuenta de que **todos escalan muy mal**. No es bueno cuando en un método tenéis que hacer distancias de todos contra todos, muchas veces, y todos estos se basan en eso. Y algunos, en repetir ese proceso todo lo que queráis". Además, hay que fijar el número de vecinos (LOF) o el tamaño de la circunferencia (DBSCAN).

(nota del redactor) El coste ingenuo de las distancias es $O(n^2 d)$. Con árboles KD/Ball baja a ~$O(n\log n)$ en dimensión baja, pero con 21 dimensiones se degradan hacia fuerza bruta. Una matriz completa de 170k×170k en float64 ocuparía ≈231 GB; sklearn busca vecinos por bloques sin materializarla, pero sigue siendo lento. La OCSVM con kernel RBF exacto escala entre $O(n^2)$ y $O(n^3)$, inviable con 170k filas sin submuestreo o sin `Nystroem` + `SGDOneClassSVM`.

*Fuente: transcripción [02:59:55]–[03:00:30] (líneas 1680–1688).*

---

### 14. Isolation Forest (slide p47)

**Idea (slide).** "Detecta anomalías usando árboles de decisión. **Un punto es anómalo si se le puede aislar con pocos cortes (decisiones).**" Paper (2008): https://ieeexplore.ieee.org/document/4781136/; https://en.wikipedia.org/wiki/Isolation_forest; https://www.cienciadedatos.net/documentos/py22-deteccion-anomalias-isolation-forest-python.html. Es "de los que están diseñados justo para anomaly detection".

**Explicación oral.**
- Dos maneras de usar árboles: (a) como antes, "tenéis un modelo sobre los datos [árbol de decisión, random forest o lo que queráis] y luego intentáis detectar outliers" por el error; (b) la definición de Isolation Forest: "si yo tengo un punto, voy a decir que es anómalo si lo puedo aislar con pocos cortes".
- "En realidad está diciendo lo mismo que todos los otros métodos". Un punto en medio de la nube es difícil de separar: "si intentas entrenar un modelo que clasifique rojo contra azul aquí, lo tendrías muy complicado, sería una frontera muy rara, tendría mucha complejidad; pues es lo que intentamos medir aquí". Difícil de separar significa que no es outlier; fácil de separar "con muy poca complejidad", que sí lo es.
- Generalización: empezar con un modelo lineal; si sigue siendo fácil separar, pasar a uno cuadrático, luego de orden 3…, hasta llegar a puntos que no se pueden separar de ninguna manera, que "seguro que no son outliers". La complejidad necesaria para aislar un punto funciona como score de normalidad.
- (nota del redactor) Score original: $s(x,n)=2^{-E[h(x)]/c(n)}$, con $h(x)$ la longitud del camino hasta aislar $x$ y $c(n)=2H(n-1)-2(n-1)/n$. Un $s$ próximo a 1 indica anomalía; $s$ muy por debajo de 0,5, normalidad; si todo ronda 0,5, no hay anomalías claras. En sklearn, `score_samples` devuelve $-s$ (verificado: `offset_ = -0.5` con `contamination='auto'`, así que `predict` marca −1 cuando $s>0{,}5$). Por eso los `-score_samples` del notebook están entre 0,38 y 0,67 y el umbral 0,55 de la versión de clase queda justo por encima de 0,5. El coste es casi lineal, $O(t\,\psi\log\psi)$ con `max_samples` ψ = min(256, n) por defecto: muy adecuado para 170k filas.

**Figura (p47).** "Isolating an anomalous point": un punto rojo arriba, separado del resto con **un solo corte**. "Isolating a normal point": un punto rojo en medio de la nube azul que necesita **varios cortes** (≈4, verticales y horizontales) para quedar aislado.

**Código del profesor** (`AD_Tools_Sklearn_clase.md`, celdas 5–10):
```python
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=10, warm_start=True)  # 10 árboles; warm_start permite añadir árboles después
clf.fit(X)                                               # X = Iris (150,4)

# "The lower, the more abnormal." (por eso el menos '-')
y_train_scores = -clf.score_samples(X)                   # ahora MAYOR = MÁS ANÓMALO

plt.plot(X[:,0],X[:,1],'.')
#ii = np.argmax(y_train_scores)        # versión enunciado: solo el más anómalo
ii = y_train_scores>0.55                # versión clase: varios, umbral 0.55
plt.plot(X[ii,0],X[ii,1],'r.')
```
Salida verificada: scores entre 0,384 y 0,670 (máximo 0,67007046 en el índice 131, el punto (7.9, 3.8)); **20 puntos** superan 0,55. Figura `celda09_1.png`: los 20 en rojo en la periferia ((4.3, 3.0), (5.7, 4.4), (7.9, 3.8), (5.0, 2.0)…); `celda10_2.png`, todos los pares. En el enunciado (`AD_Tools_Sklearn.md`) se marca solo `argmax`, y los valores son otros porque no hay `random_state`.

*Fuente: slide p47 · transcripción [03:00:36]–[03:02:15] (líneas 1689–1709) · notebook AD_Tools_Sklearn_clase celdas 5–10.*

---

### 15. GMM: Gaussian Mixture Model y EM (slide p48)

**Idea.** "Una combinación de cosas que hemos visto hasta ahora": asume que "una gaussiana sola no va a ser el mejor modelo para tus datos, entonces pongamos varias gaussianas". Se ajusta de forma iterativa: "pongo varias gaussianas, veo qué tal cubren mis datos y en función de eso hago una nueva iteración, como si fuera k-means, e intento colocar otra vez las gaussianas para ver si mejora un poco, y voy hasta que llego a un **mínimo local** en el que no mejora nada".
- Sobre el GMM: "es una cosa bastante básica, pero no se utiliza demasiado ya" [03:02:40]–[03:02:43]; "buena idea, pero se utiliza menos en la práctica" [03:07:04].
- Sobre EM, en cambio: "en la práctica se utiliza mucho porque es muy cómodo" [03:03:40].

**Slide p48, "En la práctica":** Expectation maximization (EM); Maximum likelihood. Enlace: https://www.cienciadedatos.net/documentos/py23-deteccion-anomalias-gmm-python.html. Figuras (puntos ×): a la izquierda, una sola gaussiana (elipses rojas grandes e inclinadas) sobre datos con dos grupos, un mal ajuste que cubre el hueco central vacío; a la derecha, dos gaussianas (círculos concéntricos), una por grupo, que ajustan bien. Quedan fuera como poco probables unos pocos puntos (≈(7.5, 6.9), (11, 2.2) y (5, 3.1) entre los grupos).

**EM explicado como "cuando no sé algo, me lo invento y veo si funciona"** [03:03:40]–[03:06:35]. Sin etiquetas, para cada punto se tira una moneda (cara = círculo, cruz = cruz), se entrena un modelo, se compara y "en la siguiente iteración arreglo los errores": dejo fijos los acertados y vuelvo a tirar la moneda en los erróneos; "el modelo va aprendiendo de sus errores". Con 4 puntos es fácil; con un millón "no es lo más sencillo" y "es mucho más fácil entrenar un modelo normalmente", pero "es una estrategia relativamente tonta que converge y que funciona en muchos problemas. Acordaos, porque es una idea útil en la vida". En el GMM: "ponemos unas gaussianas que nos inventamos en diferentes partes de los datos, vemos que son muy malas y las movemos hasta que vayan cuadrando". "Nos ha costado la vida entrenar, pero ahí lo tenemos".
- Uso para anomalías [03:06:42]–[03:06:59]: con una distribución fiable se usan "los intervalos de confianza, las sigmas, el 95 % de los puntos": "si ahora [un punto] es poco probable, quiere decir que es una anomalía".
- (nota del redactor) $p(x)=\sum_{k=1}^K \pi_k\,\mathcal N(x\mid\mu_k,\Sigma_k)$. E-step: $\gamma_{ik}=\frac{\pi_k\mathcal N(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j\mathcal N(x_i|\mu_j,\Sigma_j)}$. M-step: $\pi_k=\frac1n\sum_i\gamma_{ik}$, $\mu_k=\frac{\sum_i\gamma_{ik}x_i}{\sum_i\gamma_{ik}}$, $\Sigma_k=\frac{\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{\sum_i\gamma_{ik}}$. Score de anomalía: $-\log p(x)$. Coste por iteración $O(nKd^2)$ (con covarianza `full`): escala bien a 170k filas.

**Código del profesor** (`AD_Tools_Sklearn_clase.md`, celdas 18–25):
```python
from sklearn.mixture import GaussianMixture
# cov_type in ["spherical", "diag", "tied", "full"]
estimator = GaussianMixture(n_components=2, covariance_type="full", max_iter=20, random_state=0)
estimator.fit(X)

y = estimator.predict(X)        # componente asignada (clustering): 50 puntos → 0, 100 → 1
a = estimator.bic(X)            # BIC = 574.0178 → criterio para elegir n_components (menor es mejor)
plt.scatter(X[:,0],X[:,1],c=y)

py = estimator.score_samples(X) # log-verosimilitud por muestra, shape (150,)
plt.scatter(X[:,0],X[:,1],c=py) # MAYOR py = MÁS NORMAL → para anomalía usar -py
```
Figuras: `celda21_5.png`/`celda22_6.png`, 2 componentes (setosa frente al resto); `celda24_7.png`, color = log-verosimilitud (viridis): amarillo (alta densidad) en el centro de setosa y morado (baja) en (7.7, 2.6), (7.9, 3.8), (6.1, 2.6), etc.; `celda25_8.png`, ídem en todos los pares. (nota del redactor) `GaussianMixture` no tiene `decision_function`, y `predict` devuelve la componente, no si el punto es anómalo.

*Fuente: slide p48 · transcripción [03:02:17]–[03:07:04] (líneas 1710–1769) · notebook AD_Tools_Sklearn_clase celdas 18–25.*

---

### 16. PyOD: ECOD y la API común

**Idea.** La slide p09 cita tres librerías: **Scikit-learn, pycaret, PyOD** ("MÁS: datos, libros, ejemplos…"). PyOD es específica de detección de anomalías y tiene una API unificada: `fit`, `decision_function`, `decision_scores_`, `labels_`, `threshold_`, `predict`, `contamination`. Documentación: https://pyod.readthedocs.io/en/latest/pyod.models.html. pycaret no aparece en ningún notebook.

**Código del profesor** (`AD_Tools_Pyod.md`, celdas 2–12):
```python
!pip install pyod                    # se instaló pyod-2.0.3 (Colab, sklearn 1.6.1)

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)    # (150,4)

from pyod.models.ecod import ECOD
clf = ECOD()                         # repr: ECOD(contamination=0.1, n_jobs=1)
clf.fit(X)

y_train_scores = clf.decision_function(X)   # en PyOD: MAYOR = MÁS ANÓMALO (sin cambiar signo)
# En train se puede usar:
# y_train_scores = clf.decision_scores_

ii = np.argmax(y_train_scores)       # el más anómalo
plt.plot(X[:,0],X[:,1],'.')
plt.plot(X[ii,0],X[ii,1],'r.')
```
Salida: scores entre ≈2,86 y 13,32; el máximo, 13,32111548, está en el índice 13, el punto (4.3, 3.0, 1.1, 0.1), el de menor longitud de sépalo. Figura `celda11_1.png`: un único punto rojo en (4.3, 3.0); `celda12_2.png`, el mismo en todos los pares.

(nota del redactor)
- ECOD (*Empirical-Cumulative-distribution-based Outlier Detection*, Li et al., TKDE 2022): para cada dimensión estima la CDF empírica de las colas izquierda y derecha, suma $-\log$ de las probabilidades de cola de todas las dimensiones y se queda con el máximo entre cola izquierda, cola derecha y una versión que elige la cola según la asimetría de cada variable. No tiene hiperparámetros y cuesta $O(nd)$: muy rápido para 170k×21. Trata las dimensiones como **independientes**; las features tipo PCA están **incorreladas**, que no es lo mismo, pero es razonable.
- `contamination` (0,1 por defecto) **no cambia los scores**: solo fija `threshold_` y, con él, `labels_`/`predict` (1 = outlier, 0 = inlier). Para AUPR solo importa el score.

*Fuente: notebook AD_Tools_Pyod celdas 2–12; slide p09.*

---

### Herramientas

La chuleta de API de sklearn y PyOD (clase, parámetros clave, cómo sacar el score, signo y escalabilidad) está en `03_evaluacion_y_herramientas.md` §23.

### Ejercicios de los notebooks (enunciado → solución)

**AD_reconstruccion**
1. *"Detecta la anomalía calculando el error de una regresión lineal."* → **Resuelto** (celdas 5–14): `LinearRegression`, `Err = |y-y_hat|`, visualización, umbral fijo 0.1, histograma y umbral `3*np.std(y-y_hat)` (detecta la anomalía inyectada y un punto de la cola). Extra: lo mismo con `PCA(n_components=2)` y `|X_pca[:,1]|`.
2. *"Con los siguientes datos detecta las anomalías calculando el error de una regresión lineal. Haz lo mismo pero usando una regresión polinómica."* → **Resuelta la parte polinómica** (celdas 16–22, ver §3). La parte lineal no está (celda 20 vacía); ver la propuesta en §3.
3. *"Dados los siguientes datos calcula las anomalías quedándote con 1, 2, 3 o 4 dimensiones usando el PCA de sklearn."* (Iris) → **Resuelto solo para k=2** (celdas 27–31); la generalización a k está en §4.
4. *"Con la siguiente imagen: conviértela en datos (D,N) tomando patches de 8×8×3 → D=192; calcula las anomalías usando PCA variando el número de dimensiones en la reconstrucción."* (imagen `https://www.uv.es/lapeva/Valero_2020_DQ.png`, figura `celda34_18.png`) → **Sin solución** (solo se carga la imagen). Propuesta (nota del redactor):
   ```python
   from skimage import io
   from sklearn.decomposition import PCA
   im = io.imread('https://www.uv.es/lapeva/Valero_2020_DQ.png')[:, :, :3].astype(float) / 255  # quitar canal alfa si lo hay
   P = 8
   H, W = (im.shape[0] // P) * P, (im.shape[1] // P) * P
   patches = (im[:H, :W].reshape(H // P, P, W // P, P, 3)
                         .swapaxes(1, 2).reshape(-1, P * P * 3))   # (N, 192); el (D,N) del enunciado es su traspuesta
   for k in [1, 2, 5, 10, 20, 50]:
       pca = PCA(n_components=k).fit(patches)
       err = ((patches - pca.inverse_transform(pca.transform(patches)))**2).sum(1)
       plt.figure(); plt.imshow(err.reshape(H // P, W // P), cmap='Reds'); plt.title(f'k={k}')
   ```
   Lo esperable: con k pequeño destacan los patches con bordes o texturas finas (texto, contornos) y las zonas lisas tienen error ~0; al subir k el error baja en todos y solo quedan los patches realmente atípicos.
5. *"Mete un patch anómalo usando distintas estrategias (negro, escalar ×>1, cambiar contraste, meter ruido) y analiza su distancia a los otros patches."* → **Sin solución** (celda 36 vacía). Propuesta (nota del redactor):
   ```python
   from sklearn.neighbors import NearestNeighbors
   i = len(patches) // 2; p = patches[i].copy()
   variantes = {"negro": np.zeros_like(p), "x1.3": np.clip(p * 1.3, 0, 1),
                "contraste x2": np.clip((p - p.mean()) * 2 + p.mean(), 0, 1),
                "ruido": np.clip(p + 0.1 * np.random.randn(*p.shape), 0, 1)}
   pca = PCA(n_components=10).fit(np.delete(patches, i, axis=0))      # modelo entrenado sin el patch alterado
   nn = NearestNeighbors(n_neighbors=1).fit(np.delete(patches, i, axis=0))
   for nombre, q in variantes.items():
       e = ((q - pca.inverse_transform(pca.transform(q[None])))**2).sum()
       d = nn.kneighbors(q[None])[0][0, 0]
       print(nombre, "error recon:", e, "dist. al vecino más cercano:", d)
   ```
   Lo esperable: el patch negro está lejos en distancia, pero si la imagen tiene zonas oscuras puede tener un vecino casi idéntico y error de reconstrucción bajo (se reconstruye bien con la media). El escalado suave apenas cambia nada, porque queda en la dirección de la 1.ª componente (brillo). El ruido es lo que más dispara el error de reconstrucción, porque añade energía en las direcciones descartadas. Es otra versión de la discusión de p38: distancia al modelo frente a densidad.

**AD_Tools_Sklearn**
1. Ejemplo IsolationForest → **resuelto** (celdas 5–10); en `_clase` se marcan los 20 puntos con score > 0,55 en lugar del `argmax`.
2. *"¿Qué hace `predict`?"* → No resuelto explícitamente. Respuesta (nota del redactor, verificada): en IsolationForest, LOF (`novelty=True`) y OneClassSVM, `predict` devuelve −1 (outlier) o +1 (inlier) según `decision_function < 0`, es decir, binariza el score con el umbral `offset_` que fija `contamination` (IF con 'auto': −1 si $s>0{,}5$). En GaussianMixture devuelve el índice de la componente (clustering), no anomalía. DBSCAN no tiene `predict`.
3. *"Compara los algoritmos: GMM, DBSCAN, LOF, Isolation Forest… otros"* → **resuelto en `_clase`**: DBSCAN (celdas 14–17), GMM (18–25), LOF con k = 20/4/6/20 (26–31).
4. *"Cambia el código para mostrar varias anomalías y compara los métodos"* → **parcial** (umbral > 0,55 en IF; la celda 33 está vacía). Propuesta (nota del redactor): poner todos en la escala "mayor = más anómalo" y comparar el top-10:
   ```python
   from scipy.stats import spearmanr
   scores = {
       "IF":  -IsolationForest(random_state=0).fit(X).score_samples(X),
       "LOF": -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_,
       "GMM": -GaussianMixture(2, random_state=0).fit(X).score_samples(X),
       "DBSCAN": (DBSCAN(eps=0.5, min_samples=5).fit(X).labels_ == -1).astype(float),
   }
   top = {m: set(np.argsort(-s)[:10]) for m, s in scores.items()}
   print({(a, b): len(top[a] & top[b]) for a in top for b in top if a < b})
   print(spearmanr(np.column_stack(list(scores.values()))).statistic)
   ```

**AD_Tools_Pyod**
1. Ejemplo ECOD → **resuelto** (celdas 7–12).
2. *"Compara los algoritmos: GMM, DBSCAN, LOF, Isolation Forest… otros"* (con PyOD) → **sin solución** (celda 15 vacía; no hay versión `_clase`). Propuesta (nota del redactor):
   ```python
   from pyod.models.ecod import ECOD; from pyod.models.copod import COPOD
   from pyod.models.iforest import IForest; from pyod.models.lof import LOF
   from pyod.models.knn import KNN; from pyod.models.gmm import GMM
   from pyod.models.hbos import HBOS; from pyod.models.ocsvm import OCSVM
   from pyod.models.pca import PCA as PCAod
   modelos = {"ECOD": ECOD(), "COPOD": COPOD(), "IForest": IForest(random_state=0),
              "LOF": LOF(n_neighbors=20), "KNN": KNN(n_neighbors=5), "GMM": GMM(n_components=2),
              "HBOS": HBOS(), "OCSVM": OCSVM(), "PCA": PCAod()}
   S = {n: m.fit(X).decision_scores_ for n, m in modelos.items()}   # todos: mayor = más anómalo
   for n, s in S.items():
       print(n, np.argsort(-s)[:5])
   ```
   La API común hace trivial el bucle. DBSCAN no existe en PyOD; se compara aparte con sklearn.
3. *"Cambia el código para mostrar varias anomalías y compara los métodos"* → **sin solución**. Basta con `ii = np.argsort(-S[n])[:10]` en lugar del `argmax`, o con `ii = modelos[n].labels_ == 1` (el 10 % que marca `contamination=0.1`).

---

### Lo que dijo el profesor (recomendaciones explícitas)

- [02:33:30] La última alternativa es plantearlo como clasificación y "no preocuparos ni de umbrales ni de nada por el estilo": quita complejidad al diseño.
- [02:34:03] Clasificador favorito: "muy típico para esto es SVM".
- [02:34:49]–[02:34:56] Si hay etiquetas de anomalía, un clasificador "es una de las cosas más fáciles de hacer".
- [02:36:07] Para que un modelo en producción sea robusto: alerta para lo que "se aleje mucho" (esfera alrededor de los datos).
- [02:36:44]–[02:37:10] En la OCSVM, incluir un margen en el que se aceptan errores; su tamaño "depende del usuario".
- [02:38:49] / slide p34: "A mayor error de predicción, más anómalo".
- [02:41:01]–[02:41:13] "El modelo tiene que ser suficientemente complicado para capturar el comportamiento que nos interesa"; "vais a necesitar saber si vuestro modelo de verdad es bueno".
- [02:43:22] Ventaja de regresión/reconstrucción: no hay que elegir la densidad.
- [02:44:13] Zona densa frente a zona sin datos: depende de vuestra interpretación (incertidumbre epistémica frente a aleatoria).
- [02:45:03]–[02:45:19] **"Si yo tuviera anomalías y son muy pocas, nunca las utilizaría para entrenar el modelo; las utilizaría para validar las decisiones que tomaba con los umbrales."** [02:45:26] Solo con tantas anomalías como normales tiene sentido entrenar con ellas.
- [02:48:55]–[02:49:03] Las soluciones de desbalanceo valen, pero hay que asumir que los outliers son "una combinación de un millón de clases que nunca habéis visto".
- [02:49:40]–[02:50:04] El undersampling tira información, el oversampling sintético tiene asunciones fuertes y SMOTE "hace las cosas un poco más inteligente".
- [02:50:53]–[02:51:09] "En realidad todo es lo mismo… siempre va a acabar dependiendo de vosotros decidir umbrales y qué aproximación creéis que es mejor."
- [02:51:23]–[02:51:35] Con lo que sabéis podéis resolver los problemas "sin tener que utilizar nada nuevo"; los algoritmos clásicos "no tienen que ser por donde tiréis".
- [02:52:54]–[02:52:58] El clustering "te puede llevar a ver cosas que no existen, con lo cual se trata con cuidado"; [02:55:06] no hay regla para elegir el número de clusters. [02:55:08] Lectura: "Clustering: ciencia o arte".
- [02:58:43] "Siempre os falta algo que tenéis que poner vosotros": ε en DBSCAN; [03:00:18] k en LOF.
- [02:59:04] "Veréis luego en los códigos ejemplos de todos estos. Jugáis con ellos lo que queráis."
- [02:59:58] Los métodos de vecindad (DBSCAN, LOF…) "escalan muy mal": distancias de todos contra todos.
- [03:00:36] Isolation Forest es "de los diseñados justo para anomaly detection".
- [03:01:42] Idea de complejidad creciente (lineal → cuadrático → orden 3) para medir lo fácil que es aislar un punto.
- [03:02:43] / [03:07:04] GMM: idea básica, "se utiliza menos ya". [03:03:40] EM, en cambio, "en la práctica se utiliza mucho porque es muy cómodo".
- [03:06:12]–[03:06:17] EM: "estrategia relativamente tonta que converge y que funciona en muchos problemas; acordaos, es una idea útil en la vida".
- Notebook IF (celda 7): usar `-clf.score_samples(X)` porque "the lower, the more abnormal".

---

### Aplicación a nuestra práctica (fraude, AUPR, 0,17 %)

(Todo este apartado es **nota del redactor**, salvo las citas con marca de tiempo.)

**Datos de partida observados en `taller_deteccion_anomalias_partida.md`**: X1..X20 son enteros pares centrados en 0 cuya desviación típica **decrece con el índice** (X1 ≈ 391, X2 ≈ 331, X3 ≈ 302 … X16 ≈ 175), con colas muy pesadas (mín. X8 = −14.644, X2 = −14.544). Es el patrón de unas componentes PCA ordenadas por varianza y luego cuantizadas. `Amount` es ≥ 0 y muy asimétrica. El profesor parte con `train_test_split(test_size=0.3, stratify=Class, random_state=42)` → 119.618 / 51.266 filas, ajusta `GaussianMixture(5, random_state=42)` sobre `X_train0` (119.412 filas sin fraude, **sin escalar**) y usa `score_samples` (log p). El CSV de producción trae una columna `Class` vacía (NaN) que hay que quitar antes de puntuar.

#### A.1 Qué métodos probar en `02_baselines_no_supervisados` (170k × 21)

Protocolo común: **ajustar con los no-fraude de train** (como el baseline; es *novelty detection* y aprovecha las etiquetas como recomienda el profesor [02:45:03]), **evaluar AUPR en test** (que contiene fraudes) y usar los 295 fraudes solo para elegir hiperparámetros, nunca para entrenar el detector.

| Método | Coste esperado (≈119k filas de entrenamiento) | Hiperparámetros clave | ¿Con qué entrenar? | Escalado | Score "mayor = más fraude" |
|---|---|---|---|---|---|
| **GMM** (baseline 0,567) | Segundos a 1 min por ajuste (`full`, K ≤ 10) | `n_components` (1–20), `covariance_type`, `reg_covar`, `n_init`; elegir con BIC y validar con AP | Solo no-fraude | Con `full` es casi invariante a escalados lineales (salvo `reg_covar`); sí le afecta `log1p(Amount)`. Con `diag`/`spherical`, estandarizar | `-gm.score_samples(X)` |
| **Isolation Forest** | Segundos (300 árboles) | `n_estimators` 200–500, `max_samples` 256–4096, `max_features` <1, `random_state` | No-fraude o todo (0,17 % de contaminación apenas influye) | No necesita (los cortes son invariantes a transformaciones monótonas por variable) | `-iforest.score_samples(X)` |
| **PCA-reconstrucción** | < 1 s | k (nº de componentes); probar error en las **últimas** componentes | Solo no-fraude | Como X1..X20 ya parecen PCA, conviene dividir cada Xj por su σ (o usar Mahalanobis en las componentes descartadas). `log1p(Amount)` y escalar | `((Z - pca.inverse_transform(pca.transform(Z)))**2).sum(1)` |
| **K-means / VQ** | Segundos (`MiniBatchKMeans`, k = 20–200) | `n_clusters`, `n_init` | Solo no-fraude | **Imprescindible** (StandardScaler/RobustScaler + `log1p(Amount)`) | `km.transform(Z).min(axis=1)` |
| **LOF** (`novelty=True`) | Exacto con 119k: decenas de minutos o más; con una submuestra de 20–30k para `fit`: ~1–3 min para puntuar 165k filas | `n_neighbors` 20–50 (mayor que el nº máximo de duplicados) | Solo no-fraude (deduplicado / submuestreado) | **Imprescindible** | `-lof.score_samples(Z)` |
| **kNN** (distancia al k-ésimo vecino) | Igual que LOF | k | Solo no-fraude | **Imprescindible** | `NearestNeighbors(k).fit(Z0).kneighbors(Z)[0][:, -1]` o `pyod KNN.decision_function` |
| **OCSVM** | Exacta (RBF): horas e inviable con 119k; `Nystroem(300–1000)` + `SGDOneClassSVM`: segundos | `nu` (0,001–0,05), `gamma`, `n_components` | Solo no-fraude | **Imprescindible** | `-pipe.score_samples(Z)` |
| **ECOD / COPOD / HBOS** (PyOD) | Segundos ($O(nd)$) | ECOD/COPOD ninguno; HBOS `n_bins` | No-fraude o todo | No necesitan (ECDF e histogramas por variable); los empates por la cuantización afectan algo a HBOS | `det.decision_function(X)` (ya es mayor = más anómalo) |
| **Regresión cruzada** ("a mayor error de predicción…") | Segundos (20 regresiones lineales o árboles pequeños) | Modelo por feature | Solo no-fraude | Estandarizar | $\sum_j (x_j-\hat x_j)^2/\sigma_j^2$ |
| **DBSCAN** | Lento y sin score continuo | — | — | — | **Descartar** (o usar la distancia kNN, que es su versión continua) |

Notas transversales:
- **Escalado**: hay que ajustar el escalador **solo con los datos de entrenamiento** del detector, con `RobustScaler` mejor que `StandardScaler` por las colas pesadas, y aplicar `log1p` a `Amount`. Para IF, ECOD y GMM `full` el escalado lineal apenas importa; para los métodos de distancia (k-means, LOF, kNN, OCSVM) lo cambia todo.
- **Duplicados (4.167)**: en LOF/kNN provocan distancias 0 y LOF degenerados; deduplicar el conjunto de entrenamiento del detector. ADBench (p57) avisa de que "todos los no supervisados se ven muy afectados por anomalías duplicadas".
- **Coste**: fijar siempre `random_state`. El IF del notebook no lo fija y sus scores cambian entre ejecuciones.
- **Signo**: un AP cercano a la prevalencia (≈0,0017) o muy por debajo del baseline suele indicar el signo invertido. Comprobarlo también en los gráficos (el `c=-Anom` del notebook de reconstrucción).

#### A.2 Snippet: evaluar varios detectores con `average_precision_score`

Probado con datos sintéticos de la misma forma (sklearn 1.8):
```python
import numpy as np, pandas as pd, time
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem

feats = [f"X{i}" for i in range(1, 21)]
prep = ColumnTransformer([
    ("x", RobustScaler(), feats),
    ("amount", make_pipeline(FunctionTransformer(np.log1p), RobustScaler()), ["Amount"]),
])
X0 = X_train[y_train == 0].drop_duplicates()        # solo no-fraude, sin duplicados
prep.fit(X0)                                         # el escalador se ajusta solo con train
Z0, Zte = prep.transform(X0), prep.transform(X_test)
rng = np.random.default_rng(0)
Z0_sub = Z0[rng.choice(len(Z0), size=min(30_000, len(Z0)), replace=False)]  # para los métodos O(n²)

def pca_error(k):
    def f(Ztr, Zev):
        p = PCA(n_components=k).fit(Ztr)
        return ((Zev - p.inverse_transform(p.transform(Zev)))**2).sum(1)
    return f

def kmeans_dist(k):
    def f(Ztr, Zev):
        return MiniBatchKMeans(n_clusters=k, n_init=3, random_state=0).fit(Ztr).transform(Zev).min(axis=1)
    return f

# Cada detector devuelve un score con MAYOR = MÁS FRAUDE
detectores = {
    "GMM(5) full":   lambda Ztr, Zev: -GaussianMixture(5, covariance_type="full", random_state=42).fit(Ztr).score_samples(Zev),
    "IForest":       lambda Ztr, Zev: -IsolationForest(n_estimators=300, max_samples=1024, random_state=0, n_jobs=-1).fit(Ztr).score_samples(Zev),
    "PCA err k=10":  pca_error(10),
    "KMeans k=50":   kmeans_dist(50),
    "LOF nov (30k)": lambda Ztr, Zev: -LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(Z0_sub).score_samples(Zev),
    "OCSVM Nys (30k)": lambda Ztr, Zev: -make_pipeline(Nystroem(gamma=1/Ztr.shape[1], n_components=300, random_state=0),
                                         SGDOneClassSVM(nu=0.01, random_state=0)).fit(Z0_sub).score_samples(Zev),
}
# PyOD (si está instalado): ya devuelve mayor = más anómalo
try:
    from pyod.models.ecod import ECOD
    detectores["ECOD"] = lambda Ztr, Zev: ECOD().fit(Ztr).decision_function(Zev)
except ImportError:
    pass

res = []
for nombre, det in detectores.items():
    t0 = time.time(); s = det(Z0, Zte)
    res.append({"detector": nombre, "AUPR": average_precision_score(y_test, s),
                "ROC": roc_auc_score(y_test, s), "seg": round(time.time() - t0, 1)})
print(pd.DataFrame(res).sort_values("AUPR", ascending=False).to_string(index=False))
print("AP de un score aleatorio ≈ prevalencia:", y_test.mean())
```
Para comparar con el baseline (0,567 / 0,948), repetir el GMM sobre `X_train0` **sin** preprocesar. Así se separa el efecto del escalado y del `log1p` del efecto del modelo.

#### A.3 `03_supervisados`: `class_weight` frente a SMOTE dentro de la CV

- El profesor: con etiquetas, el clasificador "es una de las cosas más fáciles" [02:34:56], pero con muy pocas anomalías él las usaría para validar [02:45:03]. Además, los fraudes de producción pueden venir de "clases que nunca habéis visto" [02:49:03]. Conclusión: el supervisado debe validarse con CV estratificada y complementarse con detectores no supervisados (04).
- **`class_weight='balanced'`** (o `scale_pos_weight ≈ n_neg/n_pos ≈ 578` en XGBoost/LightGBM) no crea datos, no tiene riesgo de fuga y es la primera opción. Como AUPR solo depende del **orden** de los scores, el reponderado sirve sobre todo para que el modelo aprenda la clase minoritaria, no para calibrar.
- **SMOTE** hay que aplicarlo **solo al train de cada fold**, nunca antes de partir. Si se aplica antes, los sintéticos interpolados entre fraudes que acaban en validación filtran información y la AUPR de CV sale inflada. `imblearn.pipeline.Pipeline` aplica el remuestreo solo en `fit` (no en `predict`/`score`), lo que lo garantiza; el `Pipeline` de sklearn no admite samplers.
- **Duplicados**: con 4.167 filas repetidas, una fila y su copia pueden caer en folds distintos. Se evita agrupando por fila idéntica con `StratifiedGroupKFold`.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

grupos = pd.util.hash_pandas_object(X_train, index=False).values   # filas idénticas → mismo grupo
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

# A) class_weight (sin remuestreo)
pipe_cw = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
ap_cw = cross_val_score(pipe_cw, X_train, y_train, groups=grupos, cv=cv, scoring="average_precision")

# B) SMOTE dentro de cada fold (pip install imbalanced-learn)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
pipe_sm = ImbPipeline([
    ("scaler", StandardScaler()),                                   # SMOTE usa distancias → escalar antes
    ("smote", SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=0)),  # minoritaria hasta el 10 % de la mayoritaria
    ("clf", LogisticRegression(max_iter=2000)),
])
ap_sm = cross_val_score(pipe_sm, X_train, y_train, groups=grupos, cv=cv, scoring="average_precision")
print(ap_cw.mean(), ap_sm.mean())
```
(La variante A se ha ejecutado sobre datos sintéticos; la B requiere `imbalanced-learn`, que no está instalado en el equipo de revisión.) Para escoger entre A y B, decide la AUPR media de la CV y su dispersión entre folds, que con unos 59 fraudes por fold será grande.

#### A.4 `04_hibrido_y_ensemble` y entrega

- Usar los scores no supervisados (GMM, IF, PCA-error, ECOD, kNN) como **features** del supervisado, ajustándolos dentro de cada fold sobre los no-fraude de train para no filtrar, o combinar **rangos** (`scipy.stats.rankdata`) de varios detectores. "Todo es lo mismo" (modelo, distribución, distancia, información; [02:50:53]): cada familia falla en sitios distintos y sus errores son complementarios.
- `contamination` no influye en el score ni en la AUPR: no hace falta afinarlo.
- **Entrega**: CSV de una columna con el score (mayor = más fraude) para las 113.923 filas de producción, en el mismo orden y sin la columna `Class` vacía. Antes de exportar, comprobar el signo con la AP de test.

---

### Cobertura de fuentes

| Fuente | Rango | Cubierto | Notas |
|---|---|---|---|
| slide p07 | Enfoques: no / semi / supervisado | referencia | Desarrollado en `00_fundamentos.md` §7 |
| slide p08 | Soluciones: Distancias, Probabilidad, Información, Regresión/Reconstrucción, **Kernels**, Redes | sí | §1 (la OCSVM encaja en "Kernels", que no tiene sección propia en el PDF) |
| slide p09 | Librerías: Scikit-learn, pycaret, PyOD | sí | §16 (pycaret sin notebook) |
| slide p01–p32 | Revisadas por título (.txt): Introducción, Distancias, Probabilidad, Teoría de la información | sí (búsqueda) | **Ninguna trata OCSVM / one-class** |
| slide p33 | Portada "Regresión/reconstrucción" | sí | §2 |
| slide p34 | "Mayor error de predicción, más anómalo." | sí | §2 |
| slide p35 | 3 paneles: datos, recta, curva | sí | §2 (descripción corregida), §3 |
| slide p36 | PCA con ejes PC1/PC2 | sí | §2 |
| slide p37 | Proyección a la recta de regresión | sí | §2 (descripción corregida) |
| slide p38 | Original/new/reconstruction + setosa.io | sí | §2 |
| slide p39 | Portada "Algoritmos clásicos" | sí | §8 |
| slide p40 | Índice K-means/VQ/GMM, DBSCAN, LOF, IF | sí | §8 |
| slide p41 | K-means, clustering (3 figuras) | sí | §9 |
| slide p42 | Vector Quantization (Voronoi) | sí | §10 |
| slide p43 | VQ "En la práctica" (Lloyd, inicialización) | sí | §10 |
| slide p44 | DBSCAN (núcleo, alcanzables, ruido) | sí | §11 |
| slide p45 | LOF: definición + figura de vecinos | sí | §12 |
| slide p46 | LOF + scatter con valores | sí | §12 |
| slide p47 | Isolation Forest | sí | §14 |
| slide p48 | GMM, EM, ML | sí | §15 |
| transcripción | 1384–1447 [02:33:02–02:38:30] clasificación / OCSVM | sí | §1 |
| transcripción | 1448–1511 [02:38:40–02:44:51] regresión, PCA, incertidumbre | sí | §2 |
| transcripción | 1512–1522 [02:44:56–02:46:02] pregunta: validar umbrales | sí | §5 |
| transcripción | 1523–1569 [02:46:08–02:50:07] desbalanceo, SMOTE | sí | §6 |
| transcripción | 1570–1586 [02:50:11–02:51:35] "todo es lo mismo" | sí | §7 |
| transcripción | 1587–1631 [02:51:39–02:55:37] clustering / k-means | sí | §8, §9 |
| transcripción | 1632–1643 [02:55:47–02:56:34] VQ | sí | §10 |
| transcripción | 1644–1669 [02:56:47–02:58:57] DBSCAN | sí | §11 |
| transcripción | 1670–1688 [02:59:01–03:00:30] LOF y escalabilidad | sí | §12, §13 |
| transcripción | 1689–1709 [03:00:36–03:02:15] Isolation Forest | sí | §14 |
| transcripción | 1710–1769 [03:02:17–03:07:04] GMM y EM | sí | §15 (añadido "EM se usa mucho" [03:03:40]) |
| transcripción (cruce) | 650–668 [01:07:04–01:08:55]; 2140–2150 [03:42:57–03:43:51] | sí | §1 (SVM y distancia a la frontera; OCSVM pre-2017) |
| notebook | AD_reconstruccion.md (enunciado) | sí | Ejercicios |
| notebook | AD_reconstruccion_clase.md (celdas 0–36) + figs | sí | §2–4; figs 09, 10, 11, 12, 13, 14, 16, 19, 21, 22, 26, 28–31 descritas |
| notebook | AD_Tools_Sklearn.md (enunciado) | sí | Ejercicios, §14 |
| notebook | AD_Tools_Sklearn_clase.md (celdas 0–33) + figs | sí | §11, 12, 14, 15; salidas numéricas verificadas (17 ruido DBSCAN, 20 puntos IF > 0,55, máx. índice 131) |
| notebook | AD_Tools_Pyod.md (celdas 0–17) + figs | sí | §16 |
| notebook | taller_deteccion_anomalias_partida.md | parcial | Solo para la Aplicación (split, baseline GMM, escalas de X1..X20) |

**Dudas del extractor: resolución**
1. **OCSVM**: no aparece en ninguna slide (revisados los títulos de p01–p57; p08 solo nombra "Kernels") ni en ningún notebook. Sí aparece oralmente en [02:34:03], [02:36:44], [02:37:59] y [03:43:33]. El código de §1 es nota del redactor, verificado contra sklearn 1.8 (`OneClassSVM`, `SGDOneClassSVM` + `Nystroem`).
2. **"Estado del arte" [02:49:21]**: por el contexto se refiere a SMOTE (ver §6).
3. **Ejercicios sin solución**: propuestas en "Ejercicios" (patches de imagen, patch anómalo, comparación sklearn y PyOD, "¿qué hace predict?"), marcadas como nota del redactor.
4. **Umbral 3σ**: es un z-score bilateral sobre el residuo (≈0,27 % de falsos positivos bajo gaussianidad, con σ no robusta). En el ejercicio polinómico, las anomalías de ±2σ no están garantizadas entre los 3 puntos marcados (ver §2 y §3).

**Pendiente**: la API de PyOD no se ha podido ejecutar (no está instalado en el equipo de revisión); se documenta según PyOD 2.x y la salida del notebook. Tampoco se han ejecutado los snippets de PyOD ni de `imbalanced-learn`.
