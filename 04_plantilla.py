# ==============================================================================
# CÓDIGO 4 — PLANTILLA GEOMÉTRICA
# ==============================================================================
# Este código crea la plantilla geométrica para todas las características.
# La plantilla es el "molde" de la imagen: dice en qué lugar va el bloque de
# cada característica y de qué tamaño es. Todavía no pinta ningún audio; eso lo
# hace el código de colores, que usa la plantilla que se guarda aquí.
#
# Recibe las tablas de importancias que entregó el CÓDIGO 3 y hace lo mismo con
# cada una: la de características y la de índices.
#
# CÓMO SE ARMA LA GEOMETRÍA:
#   - Las características se toman en el orden del archivo de importancias y se
#     agrupan en filas de 3 bloques.
#   - El alto de cada fila es la suma de las importancias de sus 3 bloques.
#   - El ancho de cada bloque es su importancia dividida entre el alto de la fila.
#   - Así el área de cada bloque es igual a su importancia: una característica
#     más importante ocupa más espacio en la imagen.
#   - La imagen completa mide 1 x 1, porque las importancias suman 1.
#
# Qué hace, en orden:
#   1. Pregunta por teclado el tamaño de la imagen en píxeles.
#   2. Busca las tablas de importancias de la última ejecución del CÓDIGO 3.
#   3. Crea la geometría de cada bloque en proporciones (de 0 a 1).
#   4. Convierte cada bloque a píxeles según el tamaño elegido.
#   5. Si algún bloque no cabe en ese tamaño, elimina las características de
#      menor importancia, vuelve a armar la plantilla y muestra una alerta.
#   6. Guarda la plantilla en un CSV y un dibujo de la plantilla en PNG.
#   7. Guarda un reporte con lo que pasó en cada paso.
#
# IMPORTANTE:
# - La geometría se calcula UNA sola vez, aquí. Los códigos siguientes leen el
#   CSV de la plantilla y no la vuelven a calcular.
# - El número de características debe ser múltiplo de 3. El CÓDIGO 3 ya lo
#   entrega así.
# - Un bloque "no cabe" cuando al pasarlo a píxeles mide menos de 1 píxel de
#   ancho o de alto: un píxel no se puede partir, así que ese bloque no se
#   podría dibujar. Pasa con muchas características en una imagen pequeña.
# - Las tablas del CÓDIGO 3 se leen solas y por POSICIÓN de columna.
# ==============================================================================

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==============================================================================
# BLOQUE 0 — RUTAS, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
CARPETA_ENTRADA = os.path.join(CARPETA_BASE, "salidas", "importancias")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "plantilla")

# Ejecución del CÓDIGO 3 que se va a leer. Si se deja vacío, el código toma
# sola la más reciente. Para fijar una se escribe el dataset y el método,
# ejemplo: "Aves_0-5s_XC_ReliefF"
NOMBRE_EJECUCION = ""

# Tamaños de imagen que aparecen en el menú, en píxeles (ancho y alto iguales).
TAMANOS_MENU = [64, 128]

# Cantidad de bloques que tiene cada fila de la plantilla.
BLOQUES_POR_FILA = 3

# Posiciones en la tabla de importancias del CÓDIGO 3.
POS_NOMBRE = 1        # nombre de la característica
POS_IMPORTANCIA = 2   # importancia normalizada (todas suman 1)

# ==============================================================================
# BLOQUE DE FUNCIONES
# ==============================================================================

# Lista donde se va guardando el reporte.
reporte = []


# Muestra un texto en pantalla y lo guarda en el reporte al mismo tiempo.
def anotar(texto):
    print(texto)
    reporte.append(texto)


# Pide un número por teclado y lo repite hasta que sea una opción del menú.
def pedir_numero(cantidad):
    while True:
        respuesta = input("Entrada por teclado: ").strip()
        if respuesta.isdigit() and 1 <= int(respuesta) <= cantidad:
            return int(respuesta)
        print(f"Opción no válida. Escriba un número entre 1 y {cantidad}.")


# Muestra el menú del tamaño de la imagen y devuelve el tamaño elegido en
# píxeles. En la opción "Otra" se escribe el número de píxeles.
def elegir_tamano():
    print("")
    print("TAMAÑO DE PÍXELES")
    numero_linea = 0
    for tamano in TAMANOS_MENU:
        numero_linea += 1
        print(f"    {numero_linea}. {tamano}")
    numero_otra = len(TAMANOS_MENU) + 1
    print(f"    {numero_otra}. Otra")
    numero = pedir_numero(numero_otra)
    if numero < numero_otra:
        return TAMANOS_MENU[numero - 1]
    while True:
        texto = input("Escriba el tamaño en píxeles (ejemplo: 224): ").strip()
        texto = texto.lower().replace("px", "").strip()
        if texto.isdigit() and int(texto) > 0:
            return int(texto)
        print("El tamaño debe ser un número entero mayor que 0.")


# Recibe la lista de importancias (áreas) y devuelve la geometría: un
# diccionario que para cada bloque guarda su esquina (x, y), su ancho (w) y su
# alto (h), todo en proporciones de 0 a 1.
# La "y" se mide desde abajo: la primera fila de bloques queda en la parte
# inferior de la imagen y las siguientes se van apilando hacia arriba.
def crear_geometria(areas):
    geometria = {}
    y = 0.0
    for inicio_fila in range(0, len(areas), BLOQUES_POR_FILA):
        fila = list(range(inicio_fila, inicio_fila + BLOQUES_POR_FILA))
        altura = sum(areas[i] for i in fila)
        x = 0.0
        for i in fila:
            ancho = areas[i] / altura
            geometria[i] = {"x": x, "y": y, "w": ancho, "h": altura}
            x += ancho
        y += altura
    return geometria


# Recibe la geometría en proporciones y devuelve, para cada bloque, las
# columnas y filas de píxeles que ocupa en la imagen final: cada borde se
# multiplica por el tamaño de la imagen y se le quita la parte decimal.
# En una imagen la fila 0 es la de arriba. Como la "y" de la geometría se mide
# desde abajo, las filas se cuentan al revés (tamaño menos y). Así la primera
# fila de bloques queda abajo en la imagen sin tener que voltearla después.
def convertir_a_pixeles(geometria, tamano):
    rectangulos = []
    for i, bloque in geometria.items():
        x1 = int(bloque["x"] * tamano)
        y1 = int(bloque["y"] * tamano)
        x2 = int((bloque["x"] + bloque["w"]) * tamano)
        y2 = int((bloque["y"] + bloque["h"]) * tamano)
        rectangulos.append({"col_inicio": x1, "col_fin": x2, "fila_inicio": tamano - y2, "fila_fin": tamano - y1})
    return rectangulos


# Arma la plantilla y revisa que todos los bloques quepan en el tamaño elegido.
# Recibe la tabla de bloques (posición 0 = lugar de la característica en la
# tabla normalizada, posición 1 = nombre, posición 2 = importancia) y el tamaño.
# Si algún bloque queda con 0 píxeles:
#   - Se eliminan las 3 características de menor importancia (una fila completa,
#     para que el total siga siendo múltiplo de 3).
#   - Las importancias que quedan se vuelven a normalizar para que sumen 1.
#   - Se arma de nuevo la plantilla y se revisa otra vez.
# Se repite hasta que todos los bloques tengan al menos 1 píxel.
# Devuelve la tabla de bloques final, la geometría, los rectángulos en píxeles
# y la lista de características eliminadas. Si todo cabe desde el principio, no
# se elimina nada y las importancias quedan exactamente como llegaron.
def ajustar_al_tamano(bloques, tamano):
    eliminadas = []
    while True:
        areas = bloques.iloc[:, 2].tolist()
        geometria = crear_geometria(areas)
        rectangulos = convertir_a_pixeles(geometria, tamano)
        pixeles = [(rectangulo["col_fin"] - rectangulo["col_inicio"]) * (rectangulo["fila_fin"] - rectangulo["fila_inicio"]) for rectangulo in rectangulos]
        if min(pixeles) > 0:
            return bloques, geometria, rectangulos, eliminadas
        if len(bloques) <= BLOQUES_POR_FILA:
            raise SystemExit(f"No es posible armar la plantilla en {tamano} píxeles. Elija un tamaño más grande.")
        filas_eliminar = bloques.iloc[:, 2].nsmallest(BLOQUES_POR_FILA).index
        eliminadas = eliminadas + bloques.loc[filas_eliminar].iloc[:, 1].tolist()
        bloques = bloques.drop(index=filas_eliminar).reset_index(drop=True)
        bloques.iloc[:, 2] = bloques.iloc[:, 2] / bloques.iloc[:, 2].sum()


# Dibuja la plantilla y la guarda como imagen PNG: cada bloque con su borde y
# con el número de su característica cuando el bloque es suficientemente grande.
# El dibujo usa la misma orientación de la imagen final (primera fila abajo).
def dibujar_plantilla(geometria, titulo, ruta_png):
    figura, eje = plt.subplots(figsize=(7, 7), dpi=150)
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1)
    eje.set_aspect("equal")
    eje.set_xticks([])
    eje.set_yticks([])
    eje.set_title(titulo, fontsize=11)
    for i, bloque in geometria.items():
        eje.add_patch(Rectangle((bloque["x"], bloque["y"]), bloque["w"], bloque["h"], facecolor="#e9eef5", edgecolor="#1f2a44", linewidth=0.8))
        if bloque["w"] > 0.035 and bloque["h"] > 0.025:
            eje.text(bloque["x"] + bloque["w"] / 2, bloque["y"] + bloque["h"] / 2, str(i), ha="center", va="center", fontsize=7, color="#1f2a44")
    figura.savefig(ruta_png, bbox_inches="tight")
    plt.close(figura)


# ==============================================================================
# BLOQUE 1 — PREGUNTA POR TECLADO
# Se pregunta el tamaño de la imagen. Con esa respuesta se ejecuta el resto.
# ==============================================================================
TAMANO_PIXELES = elegir_tamano()

# ==============================================================================
# BLOQUE 2 — BÚSQUEDA DE LAS TABLAS DE IMPORTANCIAS
# Los archivos del CÓDIGO 3 se llaman: Importancia_<tabla>_<dataset>_<método>.csv
# Si no se fijó una ejecución en el BLOQUE 0, se toma la del archivo más
# reciente. Después se arma la lista de tablas de esa ejecución: la de
# características, la de índices o las dos.
# ==============================================================================
if NOMBRE_EJECUCION == "":
    archivos_importancia = glob.glob(os.path.join(CARPETA_ENTRADA, "Importancia_*.csv"))
    if len(archivos_importancia) == 0:
        raise SystemExit("No hay ninguna tabla de importancias. Ejecute primero el CÓDIGO 3.")
    archivo_reciente = max(archivos_importancia, key=os.path.getmtime)
    nombre_reciente = os.path.splitext(os.path.basename(archivo_reciente))[0]
    NOMBRE_EJECUCION = nombre_reciente.split("_", 2)[2]
archivos_ejecucion = sorted(glob.glob(os.path.join(CARPETA_ENTRADA, f"Importancia_*_{NOMBRE_EJECUCION}.csv")))
archivos_ejecucion = [ruta for ruta in archivos_ejecucion if os.path.splitext(os.path.basename(ruta))[0].split("_", 2)[2] == NOMBRE_EJECUCION]
if len(archivos_ejecucion) == 0:
    raise SystemExit(f"No se encontró ninguna tabla de importancias para la ejecución: {NOMBRE_EJECUCION}")
os.makedirs(CARPETA_SALIDA, exist_ok=True)
RUTA_REPORTE = os.path.join(CARPETA_SALIDA, f"Reporte_Plantilla_{NOMBRE_EJECUCION}_{TAMANO_PIXELES}px.txt")
anotar("")
anotar("=" * 80)
anotar("REPORTE DE LA PLANTILLA GEOMÉTRICA")
anotar("=" * 80)
anotar("1. CONFIGURACIÓN")
anotar(f"Ejecución del CÓDIGO 3 : {NOMBRE_EJECUCION}")
anotar(f"Tamaño de la imagen    : {TAMANO_PIXELES}x{TAMANO_PIXELES} px")
anotar(f"Bloques por fila       : {BLOQUES_POR_FILA}")
anotar(f"Tablas encontradas     : {len(archivos_ejecucion)}")

# ==============================================================================
# BLOQUE 3 — PLANTILLA DE CADA TABLA
# Se repite lo mismo para cada tabla de importancias:
#   - Se leen el nombre y la importancia de cada característica.
#   - Geometría en proporciones y conversión a píxeles, revisando que todos los
#     bloques quepan en el tamaño elegido (función ajustar_al_tamano).
#   - Si hubo que eliminar características, se muestra la alerta.
#   - Guardado del CSV de la plantilla, con una fila por bloque, y del dibujo.
# ==============================================================================
numero_seccion = 1
for ruta_importancia in archivos_ejecucion:
    numero_seccion += 1
    nombre_importancia = os.path.splitext(os.path.basename(ruta_importancia))[0]
    tipo_tabla = nombre_importancia.split("_", 2)[1]
    df_imp = pd.read_csv(ruta_importancia)
    nombres = df_imp.iloc[:, POS_NOMBRE].tolist()
    areas = df_imp.iloc[:, POS_IMPORTANCIA].tolist()
    if len(areas) % BLOQUES_POR_FILA != 0:
        raise SystemExit(f"La plantilla espera un número de características múltiplo de {BLOQUES_POR_FILA} y {tipo_tabla} tiene {len(areas)}.")

    # Geometría en proporciones y en píxeles, ajustada al tamaño elegido.
    # posicion_tabla guarda en qué lugar está cada característica dentro de la
    # tabla normalizada del CÓDIGO 3, para saber de qué columna sale su valor.
    bloques = pd.DataFrame({"posicion_tabla": range(len(nombres)), "caracteristica": nombres, "importancia": areas})
    bloques, geometria, rectangulos, eliminadas = ajustar_al_tamano(bloques, TAMANO_PIXELES)

    # Tabla de la plantilla: una fila por bloque.
    plantilla = bloques.copy()
    plantilla.insert(0, "indice", range(len(plantilla)))
    plantilla["x"] = [geometria[i]["x"] for i in range(len(plantilla))]
    plantilla["y"] = [geometria[i]["y"] for i in range(len(plantilla))]
    plantilla["ancho"] = [geometria[i]["w"] for i in range(len(plantilla))]
    plantilla["alto"] = [geometria[i]["h"] for i in range(len(plantilla))]
    plantilla["col_inicio"] = [rectangulo["col_inicio"] for rectangulo in rectangulos]
    plantilla["col_fin"] = [rectangulo["col_fin"] for rectangulo in rectangulos]
    plantilla["fila_inicio"] = [rectangulo["fila_inicio"] for rectangulo in rectangulos]
    plantilla["fila_fin"] = [rectangulo["fila_fin"] for rectangulo in rectangulos]

    # Guardado del CSV y del dibujo.
    nombre_salida = f"Plantilla_{tipo_tabla}_{NOMBRE_EJECUCION}_{TAMANO_PIXELES}px"
    RUTA_CSV = os.path.join(CARPETA_SALIDA, f"{nombre_salida}.csv")
    RUTA_PNG = os.path.join(CARPETA_SALIDA, f"{nombre_salida}.png")
    plantilla.to_csv(RUTA_CSV, index=False)
    dibujar_plantilla(geometria, f"Plantilla de {tipo_tabla} — {len(plantilla)} bloques", RUTA_PNG)

    anotar("")
    anotar(f"{numero_seccion}. PLANTILLA DE {tipo_tabla.upper()}")
    anotar(f"Tabla de importancias    : {ruta_importancia}")
    anotar(f"Características recibidas: {len(nombres)}")
    anotar(f"Bloques de la plantilla  : {len(plantilla)}")
    anotar(f"Filas de bloques         : {len(plantilla) // BLOQUES_POR_FILA}")
    anotar(f"Suma de importancias     : {plantilla.iloc[:, 3].sum():.12f}")
    if len(eliminadas) > 0:
        anotar(f"ALERTA: por el tamaño de la imagen y el valor de las importancias no es posible ubicar las {len(nombres)} características de {tipo_tabla} en {TAMANO_PIXELES} píxeles.")
        anotar(f"Por eso se eliminaron las {len(eliminadas)} de menor importancia: {', '.join(eliminadas)}")
        anotar(f"La plantilla quedó con {len(plantilla)} bloques y sus importancias se normalizaron de nuevo para sumar 1.")
    anotar(f"Plantilla guardada (CSV) : {RUTA_CSV}")
    anotar(f"Dibujo guardado (PNG)    : {RUTA_PNG}")
    reporte.append("Bloques de la plantilla (índice, característica, importancia):")
    for fila in plantilla.itertuples(index=False, name=None):
        reporte.append(f"  [{fila[0]}] {fila[2]:<22} {fila[3]:.12f}")

# ==============================================================================
# BLOQUE 4 — REPORTE
# Guarda en un archivo de texto todo lo anotado durante la ejecución.
# Columnas del CSV de la plantilla, por posición:
#   0 índice | 1 posicion_tabla | 2 característica | 3 importancia
#   4 x | 5 y | 6 ancho | 7 alto
#   8 col_inicio | 9 col_fin | 10 fila_inicio | 11 fila_fin
# Para pintar un bloque en la imagen se usa:
#   imagen[fila_inicio:fila_fin, col_inicio:col_fin] = gris
# y el gris sale de la columna (columnas meta + posicion_tabla) de la tabla
# normalizada del CÓDIGO 3.
# ==============================================================================
anotar("")
anotar(f"Reporte guardado         : {RUTA_REPORTE}")
with open(RUTA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
    archivo_reporte.write("\n".join(reporte))
