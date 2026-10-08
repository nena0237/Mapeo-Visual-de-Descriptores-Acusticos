# ==============================================================================
# CÓDIGO 10 — GRÁFICOS Y TABLAS DE RESULTADOS
# ==============================================================================
# Este código no entrena nada. Lee los resultados que guardaron las tres líneas
# de clasificación y los convierte en imágenes listas para el artículo:
#   - 07_Linea_base_1_Clasificadores.py  (clasificadores tradicionales)
#   - 08_Linea_base_2_descriptores.py    (representación visual + EfficientNet-B0)
#   - 09_Linea_base_3_espectrogramas.py  (espectrogramas + EfficientNet-B0)
#
# Para cada línea guarda 3 imágenes en salidas/graficos:
#   1. GRÁFICAS DE LAS MEDIDAS
#      - Líneas 08 y 09: una imagen con 5 gráficas por época (Loss, AUC,
#        Accuracy, Recall y F1), con la curva de entrenamiento y la de prueba.
#        En la 08 hay una fila de gráficas por tabla (características e índices).
#      - Línea 07: los clasificadores no se entrenan por épocas, así que no hay
#        Loss. Se hace una imagen con barras de AUC, Accuracy, Recall y F1 de
#        cada clasificador, todas en el mismo eje (una gráfica por tabla).
#   2. MATRIZ DE CONFUSIÓN de los audios de prueba. Cada fila es la especie real
#      y cada columna la especie que predijo el modelo. Cada fila se divide por
#      el número de audios de esa especie, así la diagonal es el porcentaje de
#      audios bien reconocidos de cada especie.
#      - Línea 07: la del clasificador con mayor accuracy en prueba.
#      - Líneas 08 y 09: la de la última época.
#   3. TABLA con AUC, Accuracy, Recall y F1 en prueba. El mejor valor de cada
#      columna va en negrilla.
#
# IMPORTANTE:
# - Recall y F1 son macro: todas las especies pesan lo mismo.
# - Accuracy, Recall y F1 se muestran en porcentaje y el AUC de 0 a 1. En las
#   barras de la línea 07, para que todo quede en el mismo eje, el AUC se
#   muestra multiplicado por 100.
# - De cada línea se toma el resultado más reciente.
# - Si una línea todavía no tiene resultados, se avisa y se sigue con las demás.
# - Todo se lee por POSICIÓN de columna. No se pregunta nada por teclado.
# ==============================================================================

import os
import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==============================================================================
# BLOQUE 0 — CONEXIONES, POSICIONES Y ESTILO
# Es el único bloque que se modifica.
# ==============================================================================

# Nombre del archivo del código de colores, sin el ".py". De él salen las rutas.
ARCHIVO_COLORES = "05_colores"

# Tablas de las líneas 07 y 08, con el nombre que se muestra en las imágenes.
TABLAS = [("Caracteristicas", "Características"), ("Indices", "Índices acústicos")]

# Nombre de las medidas, en el orden en que se grafican.
MEDIDAS_CURVAS = ["Loss", "AUC", "Accuracy (%)", "Recall macro (%)", "F1 macro (%)"]
MEDIDAS_TABLA = ["AUC", "Accuracy (%)", "Recall macro (%)", "F1 macro (%)"]

# Posiciones en la tabla de resultados de la línea 07.
POS_CLASICOS_MODELO = 0
POS_CLASICOS_TEST = [5, 6, 7, 8]   # AUC, accuracy, recall, F1 en prueba
POS_CLASICOS_ACCURACY = 6          # para elegir el mejor clasificador
FACTOR_F1_CLASICOS = 100           # el F1 viene de 0 a 1

# Posiciones en las predicciones de la línea 07.
POS_PRED_REAL = 1                  # especie real
POS_PRED_PRIMER_CLASIFICADOR = 2   # desde aquí, una columna por clasificador

# Posiciones en el CSV de épocas de la línea 08, en el orden de MEDIDAS_CURVAS.
POS_DESCRIPTORES_TRAIN = [1, 5, 2, 3, 4]
POS_DESCRIPTORES_TEST = [6, 10, 7, 8, 9]
FACTOR_F1_DESCRIPTORES = 1         # el F1 ya viene en porcentaje

# Posiciones en el CSV de épocas de la línea 09, en el orden de MEDIDAS_CURVAS.
POS_ESPECTROGRAMAS_TRAIN = [1, 2, 3, 4, 5]
POS_ESPECTROGRAMAS_TEST = [6, 7, 8, 9, 10]
FACTOR_F1_ESPECTROGRAMAS = 100     # el F1 viene de 0 a 1

# Posiciones en las predicciones de las líneas 08 y 09.
POS_PRED_PREDICHA = 2

# Estilo de las imágenes.
DPI = 200
FONDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"
COLORES_MEDIDAS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLOR_ENCABEZADO = "#eceae4"
MAPA_MATRIZ = LinearSegmentedColormap.from_list("matriz", ["#fcfcfb", "#9ec5f0", "#2a78d6", "#0d3b73"])

# ==============================================================================
# BLOQUE 1 — CONEXIÓN CON EL CÓDIGO DE COLORES Y CARPETAS
# ==============================================================================
colores = importlib.import_module(ARCHIVO_COLORES)
CARPETA_SALIDAS = os.path.join(colores.CARPETA_BASE, "salidas")
CARPETA_CLASICOS = os.path.join(CARPETA_SALIDAS, "clasicos")
CARPETA_DESCRIPTORES = os.path.join(CARPETA_SALIDAS, "main")
CARPETA_ESPECTROGRAMAS = os.path.join(CARPETA_SALIDAS, "espectrogramas")
CARPETA_GRAFICOS = os.path.join(CARPETA_SALIDAS, "graficos")
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# Estilo general de matplotlib: fondo claro, sin bordes de arriba y derecha.
plt.rcParams["figure.facecolor"] = FONDO
plt.rcParams["axes.facecolor"] = FONDO
plt.rcParams["savefig.facecolor"] = FONDO
plt.rcParams["axes.edgecolor"] = TINTA_SECUNDARIA
plt.rcParams["axes.labelcolor"] = TINTA
plt.rcParams["text.color"] = TINTA
plt.rcParams["xtick.color"] = TINTA_SECUNDARIA
plt.rcParams["ytick.color"] = TINTA_SECUNDARIA
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 9

# ==============================================================================
# BLOQUE DE FUNCIONES — BÚSQUEDA Y LECTURA DE RESULTADOS
# ==============================================================================

# Devuelve el archivo más reciente de una carpeta que cumpla el patrón, sin
# contar los que empiezan por el texto de "excluir". Si no hay, devuelve "".
def buscar_reciente(carpeta, patron, excluir=""):
    archivos = []
    for ruta in glob.glob(os.path.join(carpeta, patron)):
        if excluir != "" and os.path.basename(ruta).startswith(excluir):
            continue
        archivos.append(ruta)
    if len(archivos) == 0:
        return ""
    return max(archivos, key=os.path.getmtime)


# Lee un CSV de épocas y devuelve: las épocas, las medidas de entrenamiento y
# las de prueba, cada una como matriz (épocas x 5 medidas) en el orden de
# MEDIDAS_CURVAS. El F1 se lleva a porcentaje con su factor.
def leer_epocas(ruta, posiciones_train, posiciones_test, factor_f1):
    epocas_csv = pd.read_csv(ruta)
    epocas = epocas_csv.iloc[:, 0].to_numpy()
    train = epocas_csv.iloc[:, posiciones_train].to_numpy(dtype=float)
    test = epocas_csv.iloc[:, posiciones_test].to_numpy(dtype=float)
    train[:, 4] = train[:, 4] * factor_f1
    test[:, 4] = test[:, 4] * factor_f1
    return epocas, train, test


# ==============================================================================
# BLOQUE DE FUNCIONES — DIBUJO
# ==============================================================================

# Guarda una figura en la carpeta de gráficos y la cierra.
def guardar_figura(figura, nombre_archivo):
    ruta = os.path.join(CARPETA_GRAFICOS, nombre_archivo)
    figura.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    print(f"Imagen guardada: {ruta}")


# Dibuja las curvas por época. Recibe una lista de filas; cada fila es:
# (nombre de la fila, épocas, matriz de train, matriz de prueba).
# Cada fila de la imagen tiene 5 gráficas, una por medida.
def dibujar_curvas(filas, titulo, nombre_archivo):
    figura, ejes = plt.subplots(len(filas), len(MEDIDAS_CURVAS), figsize=(3.4 * len(MEDIDAS_CURVAS), 3.0 * len(filas)), squeeze=False)
    numero_fila = 0
    for nombre_fila, epocas, train, test in filas:
        for numero_medida in range(len(MEDIDAS_CURVAS)):
            eje = ejes[numero_fila][numero_medida]
            eje.plot(epocas, train[:, numero_medida], color=COLOR_TRAIN, linewidth=2, marker="o", markersize=3, label="Entrenamiento")
            eje.plot(epocas, test[:, numero_medida], color=COLOR_TEST, linewidth=2, marker="o", markersize=3, label="Prueba")
            eje.set_title(f"{MEDIDAS_CURVAS[numero_medida]} — {nombre_fila}", fontsize=9)
            eje.set_xlabel("Época")
            eje.grid(color=REJILLA, linewidth=0.8)
            eje.set_xticks(epocas)
            eje.tick_params(axis="x", labelsize=7)
        numero_fila += 1
    ejes[0][0].legend(frameon=False, fontsize=8)
    figura.suptitle(titulo, fontsize=12, fontweight="bold")
    figura.tight_layout()
    guardar_figura(figura, nombre_archivo)


# Dibuja las barras de la línea 07. Recibe una lista de gráficas; cada una es:
# (nombre de la tabla, nombres de los clasificadores, matriz clasificadores x 4
# medidas de prueba). Las 4 medidas van en el mismo eje, en porcentaje.
def dibujar_barras(graficas, titulo, nombre_archivo):
    figura, ejes = plt.subplots(len(graficas), 1, figsize=(13, 4.2 * len(graficas)), squeeze=False)
    nombres_barras = ["AUC (×100)", "Accuracy", "Recall macro", "F1 macro"]
    ancho_barra = 0.2
    numero_grafica = 0
    for nombre_tabla, modelos, valores in graficas:
        eje = ejes[numero_grafica][0]
        posiciones = np.arange(len(modelos))
        for numero_medida in range(len(nombres_barras)):
            desplazamiento = (numero_medida - 1.5) * ancho_barra
            eje.bar(posiciones + desplazamiento, valores[:, numero_medida], width=ancho_barra * 0.9, color=COLORES_MEDIDAS[numero_medida], label=nombres_barras[numero_medida])
        eje.set_xticks(posiciones)
        eje.set_xticklabels(modelos, rotation=25, ha="right")
        eje.set_ylabel("Valor (%)")
        eje.set_ylim(0, 100)
        eje.set_title(f"Prueba — {nombre_tabla}", fontsize=10)
        eje.grid(axis="y", color=REJILLA, linewidth=0.8)
        eje.set_axisbelow(True)
        numero_grafica += 1
    ejes[0][0].legend(frameon=False, ncol=4, loc="upper right", fontsize=8)
    figura.suptitle(titulo, fontsize=12, fontweight="bold")
    figura.tight_layout()
    guardar_figura(figura, nombre_archivo)


# Calcula la matriz de confusión normalizada por fila. Las filas son las
# especies reales (en orden alfabético) y las columnas las mismas especies. Si
# el modelo predijo una especie que no está entre las reales, se cuenta en una
# columna extra "Otra especie". Devuelve la matriz y los nombres de las columnas.
def calcular_matriz(reales, predichas):
    especies = sorted(set(reales))
    posicion_especie = {especie: posicion for posicion, especie in enumerate(especies)}
    hay_otras = False
    for especie in predichas:
        if especie not in posicion_especie:
            hay_otras = True
    columnas = list(especies)
    if hay_otras:
        columnas.append("Otra especie")
    matriz = np.zeros((len(especies), len(columnas)))
    for real, predicha in zip(reales, predichas):
        columna = posicion_especie.get(predicha, len(columnas) - 1)
        matriz[posicion_especie[real], columna] += 1
    matriz = matriz / matriz.sum(axis=1, keepdims=True)
    return matriz, especies, columnas


# Dibuja una o varias matrices de confusión una al lado de la otra. Recibe una
# lista de matrices; cada una es: (título, especies reales, especies predichas).
def dibujar_matrices(matrices, titulo, nombre_archivo):
    figura, ejes = plt.subplots(1, len(matrices), figsize=(9 * len(matrices) + 1, 9.5), squeeze=False, layout="constrained")
    numero_matriz = 0
    imagen = None
    for titulo_matriz, reales, predichas in matrices:
        matriz, especies, columnas = calcular_matriz(reales, predichas)
        eje = ejes[0][numero_matriz]
        imagen = eje.imshow(matriz * 100, cmap=MAPA_MATRIZ, vmin=0, vmax=100)
        eje.set_xticks(range(len(columnas)))
        eje.set_xticklabels(columnas, rotation=90, fontsize=5.5)
        eje.set_yticks(range(len(especies)))
        eje.set_yticklabels(especies, fontsize=5.5)
        eje.set_xlabel("Especie predicha")
        # Si hay dos matrices, son de las mismas especies: los nombres de las
        # filas solo se escriben en la primera para que no se monten.
        if numero_matriz == 0:
            eje.set_ylabel("Especie real")
        else:
            eje.set_yticklabels([])
        eje.set_title(titulo_matriz, fontsize=10)
        eje.spines["top"].set_visible(True)
        eje.spines["right"].set_visible(True)
        numero_matriz += 1
    barra = figura.colorbar(imagen, ax=ejes[0].tolist(), fraction=0.02, pad=0.01)
    barra.set_label("Audios de la especie real (%)")
    figura.suptitle(titulo, fontsize=12, fontweight="bold")
    guardar_figura(figura, nombre_archivo)


# Convierte un valor de la tabla en texto: el AUC con 4 decimales y las demás
# medidas en porcentaje con 2 decimales.
def texto_valor(valor, numero_medida):
    if numero_medida == 0:
        return f"{valor:.4f}"
    return f"{valor:.2f}%"


# Dibuja una o varias tablas una debajo de la otra. Recibe una lista de tablas;
# cada una es: (título, nombre de la primera columna, nombres de las filas,
# matriz filas x 4 medidas en el orden de MEDIDAS_TABLA).
# El mejor valor de cada columna va en negrilla.
def dibujar_tablas(tablas, titulo, nombre_archivo):
    alturas = [len(nombres_filas) + 1 for _, _, nombres_filas, _ in tablas]
    figura, ejes = plt.subplots(len(tablas), 1, figsize=(9, 0.36 * sum(alturas) + 0.7 * len(tablas) + 0.4), squeeze=False, gridspec_kw={"height_ratios": alturas}, layout="constrained")
    numero_tabla = 0
    for titulo_tabla, nombre_columna, nombres_filas, valores in tablas:
        eje = ejes[numero_tabla][0]
        eje.axis("off")
        celdas = []
        for numero_fila in range(len(nombres_filas)):
            fila = [nombres_filas[numero_fila]]
            for numero_medida in range(len(MEDIDAS_TABLA)):
                fila.append(texto_valor(valores[numero_fila, numero_medida], numero_medida))
            celdas.append(fila)
        tabla = eje.table(cellText=celdas, colLabels=[nombre_columna] + MEDIDAS_TABLA, cellLoc="center", bbox=[0, 0, 1, 1])
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        for (fila_celda, columna_celda), celda in tabla.get_celld().items():
            celda.set_edgecolor(REJILLA)
            if fila_celda == 0:
                celda.set_facecolor(COLOR_ENCABEZADO)
                celda.set_text_props(fontweight="bold")
            else:
                celda.set_facecolor(FONDO)
        # El mejor valor de cada columna en negrilla (solo si hay más de una fila).
        if len(nombres_filas) > 1:
            for numero_medida in range(len(MEDIDAS_TABLA)):
                fila_mejor = int(np.nanargmax(valores[:, numero_medida]))
                tabla[(fila_mejor + 1, numero_medida + 1)].set_text_props(fontweight="bold")
        eje.set_title(titulo_tabla, fontsize=10, pad=4)
        numero_tabla += 1
    figura.suptitle(titulo, fontsize=12, fontweight="bold")
    guardar_figura(figura, nombre_archivo)


# ==============================================================================
# BLOQUE 2 — LÍNEA 07: CLASIFICADORES TRADICIONALES
# Para cada tabla se toma el resultado más reciente y sus predicciones.
# ==============================================================================
graficas_barras = []
matrices_clasicos = []
tablas_clasicos = []
for tipo_tabla, nombre_tabla in TABLAS:
    ruta_resultados = buscar_reciente(CARPETA_CLASICOS, f"Clasicos_{tipo_tabla}_*.csv")
    if ruta_resultados == "":
        continue
    resultados = pd.read_csv(ruta_resultados)
    modelos = resultados.iloc[:, POS_CLASICOS_MODELO].tolist()
    valores = resultados.iloc[:, POS_CLASICOS_TEST].to_numpy(dtype=float)
    valores[:, 3] = valores[:, 3] * FACTOR_F1_CLASICOS
    valores_barras = valores.copy()
    valores_barras[:, 0] = valores_barras[:, 0] * 100
    graficas_barras.append((nombre_tabla, modelos, valores_barras))
    tablas_clasicos.append((f"Prueba — {nombre_tabla}", "Clasificador", modelos, valores))
    ruta_predicciones = os.path.join(CARPETA_CLASICOS, f"Predicciones_{os.path.basename(ruta_resultados)}")
    if os.path.exists(ruta_predicciones):
        posicion_mejor = int(np.argmax(resultados.iloc[:, POS_CLASICOS_ACCURACY].to_numpy(dtype=float)))
        predicciones = pd.read_csv(ruta_predicciones)
        reales = predicciones.iloc[:, POS_PRED_REAL].tolist()
        predichas = predicciones.iloc[:, POS_PRED_PRIMER_CLASIFICADOR + posicion_mejor].tolist()
        matrices_clasicos.append((f"{nombre_tabla} — {modelos[posicion_mejor]} (mejor accuracy)", reales, predichas))
if len(graficas_barras) == 0:
    print("Línea 07: no hay resultados. Ejecute primero 07_Linea_base_1_Clasificadores.py.")
else:
    dibujar_barras(graficas_barras, "Línea base 1 — Clasificadores tradicionales", "07_Clasificadores_metricas.png")
    dibujar_tablas(tablas_clasicos, "Línea base 1 — Clasificadores tradicionales", "07_Clasificadores_tabla.png")
    if len(matrices_clasicos) > 0:
        dibujar_matrices(matrices_clasicos, "Línea base 1 — Matriz de confusión en prueba", "07_Clasificadores_matriz_confusion.png")

# ==============================================================================
# BLOQUE 3 — LÍNEA 08: REPRESENTACIÓN VISUAL + EFFICIENTNET-B0
# Para cada tabla se toman las épocas más recientes y sus predicciones.
# La tabla y la matriz son de la última época.
# ==============================================================================
filas_curvas = []
matrices_descriptores = []
nombres_descriptores = []
valores_descriptores = []
for tipo_tabla, nombre_tabla in TABLAS:
    ruta_epocas = buscar_reciente(CARPETA_DESCRIPTORES, f"Main_{tipo_tabla}_*_epocas.csv")
    if ruta_epocas == "":
        continue
    epocas, train, test = leer_epocas(ruta_epocas, POS_DESCRIPTORES_TRAIN, POS_DESCRIPTORES_TEST, FACTOR_F1_DESCRIPTORES)
    filas_curvas.append((nombre_tabla, epocas, train, test))
    nombres_descriptores.append(nombre_tabla)
    valores_descriptores.append(test[-1, 1:])
    ruta_predicciones = ruta_epocas.replace("_epocas.csv", "_predicciones.csv")
    if os.path.exists(ruta_predicciones):
        predicciones = pd.read_csv(ruta_predicciones)
        matrices_descriptores.append((f"{nombre_tabla} — época {int(epocas[-1])}", predicciones.iloc[:, POS_PRED_REAL].tolist(), predicciones.iloc[:, POS_PRED_PREDICHA].tolist()))
if len(filas_curvas) == 0:
    print("Línea 08: no hay resultados. Ejecute primero 08_Linea_base_2_descriptores.py.")
else:
    dibujar_curvas(filas_curvas, "Línea base 2 — Representación visual + EfficientNet-B0", "08_Descriptores_curvas.png")
    dibujar_tablas([("Prueba — última época", "Tabla", nombres_descriptores, np.array(valores_descriptores))], "Línea base 2 — Representación visual + EfficientNet-B0", "08_Descriptores_tabla.png")
    if len(matrices_descriptores) > 0:
        dibujar_matrices(matrices_descriptores, "Línea base 2 — Matriz de confusión en prueba (última época)", "08_Descriptores_matriz_confusion.png")

# ==============================================================================
# BLOQUE 4 — LÍNEA 09: ESPECTROGRAMAS + EFFICIENTNET-B0
# Se toman las épocas más recientes y sus predicciones. La tabla y la matriz
# son de la última época.
# ==============================================================================
ruta_epocas = buscar_reciente(CARPETA_ESPECTROGRAMAS, "Espectrogramas_*_epocas.csv")
if ruta_epocas == "":
    print("Línea 09: no hay resultados. Ejecute primero 09_Linea_base_3_espectrogramas.py.")
else:
    epocas, train, test = leer_epocas(ruta_epocas, POS_ESPECTROGRAMAS_TRAIN, POS_ESPECTROGRAMAS_TEST, FACTOR_F1_ESPECTROGRAMAS)
    dibujar_curvas([("Espectrogramas", epocas, train, test)], "Línea base 3 — Espectrogramas + EfficientNet-B0", "09_Espectrogramas_curvas.png")
    dibujar_tablas([("Prueba — última época", "Entrada", ["Espectrogramas"], test[-1:, 1:])], "Línea base 3 — Espectrogramas + EfficientNet-B0", "09_Espectrogramas_tabla.png")
    ruta_predicciones = ruta_epocas.replace("_epocas.csv", "_predicciones.csv")
    if os.path.exists(ruta_predicciones):
        predicciones = pd.read_csv(ruta_predicciones)
        dibujar_matrices([(f"Espectrogramas — época {int(epocas[-1])}", predicciones.iloc[:, POS_PRED_REAL].tolist(), predicciones.iloc[:, POS_PRED_PREDICHA].tolist())], "Línea base 3 — Matriz de confusión en prueba (última época)", "09_Espectrogramas_matriz_confusion.png")
