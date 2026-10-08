# ==============================================================================
# CÓDIGO 5 — COLORES
# ==============================================================================
# Este código pinta la plantilla con escala de grises según los valores de las
# características. Toma la tabla normalizada (CÓDIGO 3) y la plantilla
# (CÓDIGO 4), y genera la imagen de cada audio.
#
# Se pinta píxel por píxel, con numpy, todos los
# audios y en memoria. Las imágenes NO se guardan en disco: se generan en
# caliente cada vez que se necesitan. No se usa matplotlib para pintar, porque
# matplotlib suaviza los bordes de los bloques y la imagen deja de ser exacta.
#
# LA REGLA DEL COLOR:
#   - La imagen empieza completamente blanca (todos los píxeles en 1).
#   - Cada bloque se pinta con un solo gris: gris = 1 - valor normalizado.
#   - Valor 0 queda blanco (1) y valor 1 queda negro (0).
#   - El gris siempre queda entre 0 y 1.
#   - El tamaño y el lugar del bloque los define la plantilla (la importancia).
#     El color lo define el valor que la extracción le dio a ese audio.
#
# ESTE CÓDIGO SE USA DE DOS FORMAS:
#   1. El DataLoader lo importa y usa sus funciones para pintar en memoria las
#      imágenes con las que se entrena. Al importarlo no se ejecuta nada más.
#   2. Si se ejecuta directamente, pinta en memoria las imágenes de todos los
#      audios y guarda UNA sola imagen de muestra por tabla, con 5 audios uno
#      al lado del otro, para comparar que los colores sí cambian entre audios.
#
# IMPORTANTE:
# - La geometría no se calcula aquí: cada bloque se pinta en las filas y
#   columnas de píxeles que dice el CSV de la plantilla.
# - La imagen de muestra sale de las mismas imágenes pintadas en memoria. Es lo
#   único que este código guarda en disco.
# - No se pregunta nada por teclado y todo se lee por POSICIÓN de columna.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# BLOQUE 0 — RUTAS, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
CARPETA_PLANTILLA = os.path.join(CARPETA_BASE, "salidas", "plantilla")
CARPETA_TABLAS = os.path.join(CARPETA_BASE, "salidas", "importancias")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "colores")

# Plantilla que se va a leer. Si se deja vacío, el código toma sola la más
# reciente. Para fijar una se escribe el dataset, el método y el tamaño,
# ejemplo: "Aves_0-5s_XC_ReliefF_128px"
NOMBRE_PLANTILLA = ""

# Cuántos audios aparecen en la imagen de muestra de cada tabla.
N_MUESTRAS = 5

# Posiciones en la tabla normalizada del CÓDIGO 3.
POS_ARCHIVO = 0       # ruta del audio dentro de la carpeta de audios
POS_ESPECIE = 1       # nombre científico de la especie
N_COLUMNAS_META = 4   # las características empiezan después de estas columnas

# Posiciones en el CSV de la plantilla del CÓDIGO 4.
POS_TABLA = 1          # lugar de la característica dentro de la tabla normalizada
POS_COL_INICIO = 8     # primera columna de píxeles del bloque
POS_COL_FIN = 9        # columna de píxeles donde termina el bloque
POS_FILA_INICIO = 10   # primera fila de píxeles del bloque
POS_FILA_FIN = 11      # fila de píxeles donde termina el bloque

# ==============================================================================
# BLOQUE DE FUNCIONES
# Estas funciones son las que usa el DataLoader.
# ==============================================================================

# Devuelve el nombre de la plantilla más reciente que entregó el CÓDIGO 4.
# Los archivos se llaman: Plantilla_<tabla>_<dataset>_<método>_<tamaño>px.csv
# y el nombre que se devuelve es la parte <dataset>_<método>_<tamaño>px.
# Si se pasa una tabla ("Caracteristicas" o "Indices"), busca solo entre las
# plantillas de esa tabla.
def buscar_plantilla_reciente(tipo_tabla="*"):
    archivos_plantilla = glob.glob(os.path.join(CARPETA_PLANTILLA, f"Plantilla_{tipo_tabla}_*.csv"))
    if len(archivos_plantilla) == 0:
        raise SystemExit("No hay ninguna plantilla. Ejecute primero el CÓDIGO 4.")
    archivo_reciente = max(archivos_plantilla, key=os.path.getmtime)
    nombre_reciente = os.path.splitext(os.path.basename(archivo_reciente))[0]
    return nombre_reciente.split("_", 2)[2]


# Carga la plantilla de una tabla y la tabla normalizada que le corresponde.
# Del nombre de la plantilla salen el tamaño de la imagen y la ejecución del
# CÓDIGO 3, que es la que dice cuál es la tabla normalizada.
# Devuelve la plantilla, la tabla y el tamaño de la imagen en píxeles.
def cargar_plantilla_y_tabla(tipo_tabla, nombre_plantilla):
    nombre_ejecucion, texto_tamano = nombre_plantilla.rsplit("_", 1)
    tamano = int(texto_tamano.replace("px", ""))
    ruta_plantilla = os.path.join(CARPETA_PLANTILLA, f"Plantilla_{tipo_tabla}_{nombre_plantilla}.csv")
    ruta_tabla = os.path.join(CARPETA_TABLAS, f"{tipo_tabla}_{nombre_ejecucion}.csv")
    if not os.path.exists(ruta_plantilla):
        raise SystemExit(f"No se encontró la plantilla: {ruta_plantilla}")
    if not os.path.exists(ruta_tabla):
        raise SystemExit(f"No se encontró la tabla normalizada de la plantilla: {ruta_tabla}")
    plantilla = pd.read_csv(ruta_plantilla)
    tabla = pd.read_csv(ruta_tabla)
    return plantilla, tabla, tamano


# Pinta en memoria las imágenes de todos los audios de una tabla. Recibe la
# tabla normalizada, la plantilla y el tamaño de la imagen, y devuelve un
# arreglo de numpy con una imagen por audio: (audios, alto, ancho).
# Se recorre la plantilla bloque por bloque. Para cada bloque se toma la columna
# de su característica, que tiene un valor distinto para cada audio, y se pinta
# ese bloque en cada imagen con el gris de su propio audio.
# El resultado es idéntico a pintar audio por audio.
def pintar_imagenes(tabla, plantilla, tamano):
    imagenes = np.ones((len(tabla), tamano, tamano), dtype=np.float32)
    for bloque in plantilla.itertuples(index=False, name=None):
        valores = tabla.iloc[:, N_COLUMNAS_META + bloque[POS_TABLA]].to_numpy(dtype=float)
        grises = np.clip(1.0 - valores, 0.0, 1.0)
        imagenes[:, bloque[POS_FILA_INICIO]:bloque[POS_FILA_FIN], bloque[POS_COL_INICIO]:bloque[POS_COL_FIN]] = grises[:, None, None]
    return imagenes


# Guarda UNA imagen de muestra con varios audios, uno al lado del otro. Los
# audios se toman repartidos a lo largo de la tabla (el primero, el último y
# otros intermedios), para que sean de especies distintas. Encima de cada uno
# va el nombre de su especie. Cada píxel se dibuja tal cual, sin suavizar.
def guardar_muestra(imagenes, especies, cantidad, titulo, ruta_png):
    cantidad = min(cantidad, len(imagenes))
    posiciones = np.linspace(0, len(imagenes) - 1, cantidad).astype(int)
    figura, ejes = plt.subplots(1, cantidad, figsize=(3 * cantidad, 3.6), dpi=150, squeeze=False)
    numero_eje = 0
    for posicion in posiciones:
        eje = ejes[0][numero_eje]
        eje.imshow(imagenes[posicion], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        eje.set_title(f"audio {posicion}\n{especies[posicion]}", fontsize=8)
        eje.set_xticks([])
        eje.set_yticks([])
        numero_eje += 1
    figura.suptitle(titulo, fontsize=11)
    figura.savefig(ruta_png, bbox_inches="tight")
    plt.close(figura)


# ==============================================================================
# EJECUCIÓN DIRECTA
# Todo lo que sigue solo se ejecuta cuando este archivo se corre directamente.
# Cuando el DataLoader lo importa, esta parte no se ejecuta.
# ==============================================================================
if __name__ == "__main__":

    # ==========================================================================
    # BLOQUE 1 — BÚSQUEDA DE LAS PLANTILLAS
    # Si no se fijó una plantilla en el BLOQUE 0, se toma la más reciente.
    # Después se arma la lista de plantillas de esa ejecución: la de
    # características, la de índices o las dos.
    # ==========================================================================
    if NOMBRE_PLANTILLA == "":
        NOMBRE_PLANTILLA = buscar_plantilla_reciente()
    archivos_ejecucion = sorted(glob.glob(os.path.join(CARPETA_PLANTILLA, f"Plantilla_*_{NOMBRE_PLANTILLA}.csv")))
    archivos_ejecucion = [ruta for ruta in archivos_ejecucion if os.path.splitext(os.path.basename(ruta))[0].split("_", 2)[2] == NOMBRE_PLANTILLA]
    if len(archivos_ejecucion) == 0:
        raise SystemExit(f"No se encontró ninguna plantilla con el nombre: {NOMBRE_PLANTILLA}")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    print(f"Plantilla del CÓDIGO 4 : {NOMBRE_PLANTILLA}")

    # ==========================================================================
    # BLOQUE 2 — IMÁGENES DE CADA TABLA
    # Se repite lo mismo para cada plantilla:
    #   - Se carga la plantilla y su tabla normalizada.
    #   - Se pintan en memoria las imágenes de todos los audios.
    #   - Se guarda la imagen de muestra, que es lo único que queda en disco.
    # ==========================================================================
    for ruta_plantilla in archivos_ejecucion:
        nombre_plantilla = os.path.splitext(os.path.basename(ruta_plantilla))[0]
        tipo_tabla = nombre_plantilla.split("_", 2)[1]
        plantilla, tabla, TAMANO_PIXELES = cargar_plantilla_y_tabla(tipo_tabla, NOMBRE_PLANTILLA)
        imagenes = pintar_imagenes(tabla, plantilla, TAMANO_PIXELES)
        especies = tabla.iloc[:, POS_ESPECIE].tolist()
        RUTA_MUESTRA = os.path.join(CARPETA_SALIDA, f"Muestra_{tipo_tabla}_{NOMBRE_PLANTILLA}.png")
        guardar_muestra(imagenes, especies, N_MUESTRAS, f"Colores de {tipo_tabla} — {len(plantilla)} bloques, {TAMANO_PIXELES}x{TAMANO_PIXELES} px", RUTA_MUESTRA)
        print("")
        print(f"Tabla                : {tipo_tabla}")
        print(f"Imágenes pintadas    : {len(imagenes):,} (en memoria, no se guardan)")
        print(f"Imagen de muestra    : {RUTA_MUESTRA}")
