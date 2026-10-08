# ==============================================================================
# CÓDIGO 2 — EXTRACCIÓN DE CARACTERÍSTICAS + NORMALIZACIÓN
# ==============================================================================
# Este código es la extracción de características. Es el segundo que se utiliza:
# recibe el CSV que entregó el filtrado (CÓDIGO 1), abre cada audio y le calcula
# 9 características acústicas.
#
# Siete de las características se calculan con librosa. Las frecuencias mínima
# y máxima se calculan con el método de Raven Pro (Cornell Lab of
# Ornithology), que las obtiene de la energía acumulada de la STFT.
#   freq_minima: frecuencia donde la energía acumulada cruza el 5 %  (Raven "Freq 5%")
#   freq_maxima: frecuencia donde la energía acumulada cruza el 95 % (Raven "Freq 95%")
# En Raven Pro esa energía se mide dentro de una selección que el analista dibuja
# a mano. Aquí no hay selección manual: se usa el audio completo como selección
# en el tiempo y todo el rango de frecuencia como selección en frecuencia.
# Se calcula UNA sola STFT por audio y se reutiliza para todas las características.
#
# Cada audio se carga en mono y remuestreado a 22.050 Hz (SR_OBJETIVO).
#
# Qué hace, en orden:
#   1. Pregunta por teclado el mínimo de audios por especie.
#   2. Busca y carga el CSV que entregó el filtrado.
#   3. Valida que cada especie tenga una sola etiqueta.
#   4. Quita las especies que tienen menos audios que el mínimo elegido.
#   5. Extrae las 9 características de cada audio.
#   6. Normaliza las características con MinMaxScaler al rango [0,1].
#   7. Valida la normalización.
#   8. Guarda el único CSV final normalizado.
#   9. Guarda un reporte con lo que pasó en cada paso.
#
# IMPORTANTE:
# - La etiqueta (label) ya viene creada desde el CÓDIGO 1 y aquí se conserva.
#   No se vuelve a codificar.
# - No se genera ni se guarda un archivo CSV bruto. Los valores originales se
#   conservan únicamente en memoria para calcular estadísticas y documentarlos
#   en el reporte.
# - Las especies que no llegan al mínimo se quitan ANTES de extraer: así la
#   normalización se hace una sola vez y solo con los audios con los que se
#   trabaja.
# - El archivo del filtrado se lee solo y sus columnas se leen por POSICIÓN,
#   nunca por nombre.
#
# Las cuatro primeras columnas del CSV final quedan en este orden:
#   archivo | especie | label | clase | características...
# ==============================================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# BLOQUE 0 — RUTAS, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
CARPETA_AUDIOS = os.path.join(CARPETA_BASE, "train_audio")
CARPETA_ENTRADA = os.path.join(CARPETA_BASE, "salidas", "filtrado")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "caracteristicas")

# Archivo del filtrado que se va a leer. Si se deja vacío, el código toma solo
# el archivo más reciente que entregó el filtrado.
# Para fijar uno se escribe su nombre, ejemplo: "Filtrado_Aves_0-5s_XC.csv"
ARCHIVO_FILTRADO = ""

# Opciones del menú del mínimo de audios por especie. El texto es lo que se
# muestra y el valor lo que se usa.
MINIMOS_MENU = [("5 muestras", 5), ("10 muestras", 10), ("15 muestras", 15)]

# Posiciones en el CSV del filtrado (el filtrado siempre las entrega así).
POS_ARCHIVO = 0   # ruta del audio dentro de la carpeta de audios
POS_ESPECIE = 1   # nombre científico de la especie
POS_LABEL = 2     # etiqueta numérica de la especie
POS_CLASE = 3     # clase taxonómica

# Parámetros de extracción.
SR_OBJETIVO = 22050         # frecuencia de muestreo a la que se carga cada audio
N_FFT = 2048                # tamaño de la ventana de la STFT
HOP_LENGTH = 512            # salto entre ventanas de la STFT
PERCENTIL_FREQ_MIN = 0.05   # energía acumulada para la frecuencia mínima (5 %)
PERCENTIL_FREQ_MAX = 0.95   # energía acumulada para la frecuencia máxima (95 %)

# Cada cuántos audios se muestra el avance en pantalla.
MOSTRAR_CADA = 500

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


# Pregunta cuántos audios debe tener como mínimo cada especie y devuelve el
# número. En "Otro" se escribe el número, solo o con texto (ejemplo: "20" o
# "20 muestras"). El mínimo es 2, porque el split necesita al menos un audio de
# cada especie en entrenamiento y otro en prueba.
def elegir_minimo():
    print("")
    print("CUÁNTAS MUESTRAS MÍNIMAS DESEA TENER POR ESPECIE")
    numero_linea = 0
    for texto, valor in MINIMOS_MENU:
        numero_linea += 1
        print(f"    {numero_linea}. {texto}")
    numero_otro = len(MINIMOS_MENU) + 1
    print(f"    {numero_otro}. Otro")
    numero = pedir_numero(numero_otro)
    if numero < numero_otro:
        return MINIMOS_MENU[numero - 1][1]
    while True:
        texto = input("Escriba el número mínimo de muestras (ejemplo: 20): ")
        numeros = re.findall(r"\d+", texto)
        if len(numeros) == 1 and int(numeros[0]) >= 2:
            return int(numeros[0])
        print("Escriba un solo número entero, mínimo 2.")


# Recibe la ruta de un audio y devuelve sus 9 características en un diccionario.
# Los nombres de las características se escriben únicamente aquí: el resto del
# código los toma de este diccionario.
def extraer_caracteristicas(ruta_audio):
    # Carga del audio en mono y a la frecuencia de muestreo SR_OBJETIVO.
    y, sr = librosa.load(ruta_audio, sr=SR_OBJETIVO, mono=True)
    # STFT única: se calcula una sola vez y se reutiliza.
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    frecuencias = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    # Energía por frecuencia y energía acumulada (método de Raven Pro).
    energia_por_frecuencia = np.sum(S ** 2, axis=1)
    energia_total = energia_por_frecuencia.sum()
    if energia_total == 0:
        raise ValueError("Audio sin energía útil para calcular características.")
    energia_acumulada = np.cumsum(energia_por_frecuencia) / energia_total
    # Posición donde la energía acumulada cruza el 5 % y el 95 %.
    indice_min = np.searchsorted(energia_acumulada, PERCENTIL_FREQ_MIN)
    indice_max = np.searchsorted(energia_acumulada, PERCENTIL_FREQ_MAX)
    indice_min = min(indice_min, len(frecuencias) - 1)
    indice_max = min(indice_max, len(frecuencias) - 1)
    # Las 9 características. Cada una es el promedio de todo el audio.
    caracteristicas = {}
    caracteristicas["freq_minima"] = float(frecuencias[indice_min])
    caracteristicas["freq_maxima"] = float(frecuencias[indice_max])
    caracteristicas["centroide"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    caracteristicas["bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    caracteristicas["rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    caracteristicas["spectral_contrast"] = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    caracteristicas["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    caracteristicas["rms"] = float(np.mean(librosa.feature.rms(S=S)))
    caracteristicas["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    return caracteristicas


# ==============================================================================
# BLOQUE 1 — PREGUNTA POR TECLADO Y CARGA DEL CSV DEL FILTRADO
# Primero se pregunta el mínimo de audios por especie.
# Si no se fijó un archivo en el BLOQUE 0, se busca en la carpeta del filtrado
# el CSV más reciente (el último que entregó el filtrado) y se lee ese.
# El nombre del dataset se toma del nombre del archivo, para que las salidas de
# este código queden con el mismo nombre que las del filtrado.
# ==============================================================================
minimo = elegir_minimo()
if ARCHIVO_FILTRADO == "":
    archivos_filtrado = glob.glob(os.path.join(CARPETA_ENTRADA, "Filtrado_*.csv"))
    if len(archivos_filtrado) == 0:
        raise SystemExit(f"No hay ningún CSV del filtrado en: {CARPETA_ENTRADA}")
    RUTA_CSV = max(archivos_filtrado, key=os.path.getmtime)
else:
    RUTA_CSV = os.path.join(CARPETA_ENTRADA, ARCHIVO_FILTRADO)
if not os.path.exists(RUTA_CSV):
    raise SystemExit(f"No se encontró el archivo de entrada: {RUTA_CSV}")
nombre_archivo = os.path.splitext(os.path.basename(RUTA_CSV))[0]
NOMBRE_DATASET = nombre_archivo.replace("Filtrado_", "", 1)
SALIDA_NORMALIZADO = os.path.join(CARPETA_SALIDA, f"Caracteristicas_{NOMBRE_DATASET}.csv")
SALIDA_REPORTE = os.path.join(CARPETA_SALIDA, f"Reporte_Caracteristicas_{NOMBRE_DATASET}.txt")
os.makedirs(CARPETA_SALIDA, exist_ok=True)
df = pd.read_csv(RUTA_CSV)
if len(df) == 0:
    raise SystemExit("El CSV del filtrado está vacío.")
anotar("=" * 80)
anotar("REPORTE DE EXTRACCIÓN Y NORMALIZACIÓN DE CARACTERÍSTICAS")
anotar("=" * 80)
anotar("1. CONFIGURACIÓN")
anotar(f"Archivo de entrada     : {RUTA_CSV}")
anotar(f"Dataset                : {NOMBRE_DATASET}")
anotar(f"Frecuencia de muestreo : {SR_OBJETIVO} Hz")
anotar(f"N_FFT                  : {N_FFT}")
anotar(f"Hop length             : {HOP_LENGTH}")
anotar(f"Frecuencia mínima      : energía acumulada {PERCENTIL_FREQ_MIN * 100:.0f}%")
anotar(f"Frecuencia máxima      : energía acumulada {PERCENTIL_FREQ_MAX * 100:.0f}%")

# ==============================================================================
# BLOQUE 2 — VALIDACIÓN DEL DATASET
# Se verifica que la relación especie ↔ label sea consistente antes de procesar
# los audios: una especie tiene un solo label y un label es de una sola especie.
# ==============================================================================
especie = df.iloc[:, POS_ESPECIE]
label = df.iloc[:, POS_LABEL]
labels_por_especie = label.groupby(especie).nunique()
especies_por_label = especie.groupby(label).nunique()
if labels_por_especie.max() != 1:
    raise SystemExit("Una especie está asociada a más de un label.")
if especies_por_label.max() != 1:
    raise SystemExit("Un mismo label está asociado a más de una especie.")
anotar("")
anotar("2. DATASET DE ENTRADA")
anotar(f"Audios recibidos       : {len(df):,}")
anotar(f"Especies               : {especie.nunique():,}")
anotar(f"Labels                 : {label.nunique():,}")
anotar("Relación especie-label : correcta")

# ==============================================================================
# BLOQUE 2.1 — MÍNIMO DE AUDIOS POR ESPECIE
# Se cuentan los audios de cada especie y se dejan solo las especies que llegan
# al mínimo elegido. Se hace ANTES de extraer: así la normalización se calcula
# una sola vez y solo con los audios con los que se trabaja. El orden de las
# filas del filtrado no cambia.
# ==============================================================================
audios_por_especie = especie.value_counts()
especies_conservadas = set(audios_por_especie[audios_por_especie >= minimo].index)
audios_antes = len(df)
df = df[df.iloc[:, POS_ESPECIE].isin(especies_conservadas)].reset_index(drop=True)
anotar("")
anotar("2.1 MÍNIMO DE AUDIOS POR ESPECIE")
anotar(f"Mínimo elegido         : {minimo} audios")
anotar(f"Especies antes         : {len(audios_por_especie):,}")
anotar(f"Especies eliminadas    : {len(audios_por_especie) - len(especies_conservadas):,}")
anotar(f"Especies que quedan    : {len(especies_conservadas):,}")
anotar(f"Audios antes           : {audios_antes:,}")
anotar(f"Audios que quedan      : {len(df):,}")
if len(especies_conservadas) < 2:
    raise SystemExit(f"Con un mínimo de {minimo} audios quedan menos de 2 especies. Elija un mínimo menor.")

# ==============================================================================
# BLOQUE 3 — EXTRACCIÓN DE CARACTERÍSTICAS
# Para cada audio:
#   - Se arma su ruta uniendo la carpeta de audios con la columna del archivo.
#   - Se extraen las 9 características con la función del bloque de funciones.
#   - Se guardan por separado las 4 columnas meta y las características.
# Si un audio falla, se anota el error y se continúa con el siguiente.
# ==============================================================================
filas_meta = []
filas_caracteristicas = []
errores = []
revisados = 0
print("")
print("Extrayendo características...")
for fila in df.itertuples(index=False, name=None):
    revisados += 1
    archivo_audio = str(fila[POS_ARCHIVO])
    ruta_audio = os.path.join(CARPETA_AUDIOS, archivo_audio.replace("/", os.sep))
    try:
        caracteristicas = extraer_caracteristicas(ruta_audio)
    except Exception as error:
        errores.append((archivo_audio, str(error)))
        print(f"ERROR: {archivo_audio} | {error}")
        continue
    filas_meta.append([fila[POS_ARCHIVO], fila[POS_ESPECIE], fila[POS_LABEL], fila[POS_CLASE]])
    filas_caracteristicas.append(caracteristicas)
    if revisados % MOSTRAR_CADA == 0:
        print(f"Audios revisados: {revisados:,} de {len(df):,}")
if len(filas_caracteristicas) == 0:
    raise SystemExit("No se pudo procesar ningún audio. Revise la carpeta de audios del BLOQUE 0.")

# ==============================================================================
# BLOQUE 4 — CONSTRUCCIÓN DE LAS TABLAS EN MEMORIA
# La tabla meta conserva los nombres de columna que traía el CSV del filtrado.
# La tabla bruta tiene una columna por característica, con los valores
# originales. Solo vive en memoria: no se guarda en disco.
# ==============================================================================
nombres_meta = [df.columns[POS_ARCHIVO], df.columns[POS_ESPECIE], df.columns[POS_LABEL], df.columns[POS_CLASE]]
df_meta = pd.DataFrame(filas_meta, columns=nombres_meta)
df_bruto = pd.DataFrame(filas_caracteristicas)
N_COLUMNAS_META = df_meta.shape[1]
anotar("")
anotar("3. EXTRACCIÓN")
anotar(f"Audios procesados      : {len(df_bruto):,}")
anotar(f"Errores                : {len(errores):,}")
anotar(f"Características        : {df_bruto.shape[1]}")
numero_caracteristica = 0
for caracteristica in df_bruto.columns:
    anotar(f"[{numero_caracteristica}] {caracteristica}")
    numero_caracteristica += 1

# ==============================================================================
# BLOQUE 5 — NORMALIZACIÓN MIN-MAX
# Se aplica MinMaxScaler únicamente a las características acústicas. Cada
# característica queda escalada al rango [0,1], manteniendo intactas las cuatro
# columnas meta. Esta normalización es la que posteriormente controla la
# intensidad de gris de cada bloque en la representación visual.
# ==============================================================================
scaler = MinMaxScaler()
valores_normalizados = scaler.fit_transform(df_bruto)
df_norm = pd.DataFrame(valores_normalizados, columns=df_bruto.columns)

# ==============================================================================
# BLOQUE 6 — VALIDACIÓN DE LA NORMALIZACIÓN
# Se verifica automáticamente que no existan NaN, que no existan infinitos, que
# todas las características estén dentro de [0,1] y que el número de registros
# no cambie. Si algo falla, el código se detiene.
# ==============================================================================
nan_total = int(np.isnan(valores_normalizados).sum())
inf_total = int(np.isinf(valores_normalizados).sum())
fuera_rango = int(((valores_normalizados < 0) | (valores_normalizados > 1)).sum())
if nan_total != 0:
    raise SystemExit(f"Se encontraron {nan_total} valores NaN después de normalizar.")
if inf_total != 0:
    raise SystemExit(f"Se encontraron {inf_total} valores infinitos después de normalizar.")
if fuera_rango != 0:
    raise SystemExit("Existen características fuera del rango [0,1].")
if len(df_norm) != len(df_meta):
    raise SystemExit("El número de registros cambió durante la normalización.")

# ==============================================================================
# BLOQUE 7 — GUARDADO DEL ÚNICO CSV FINAL NORMALIZADO
# Se unen las 4 columnas meta con las características normalizadas. Este es el
# único archivo de características que se guarda en disco.
# ==============================================================================
df_final = pd.concat([df_meta, df_norm], axis=1)
df_final.to_csv(SALIDA_NORMALIZADO, index=False)

# ==============================================================================
# BLOQUE 8 — REPORTE
# Se calculan estadísticas de los valores brutos y normalizados para documentar
# el proceso, se listan los audios con error y se guarda todo lo anotado.
# En el CSV final la especie está en la posición 1 y el label en la 2.
# ==============================================================================
estadisticas_brutas = df_bruto.agg(["min", "max", "mean", "std"]).T
estadisticas_normalizadas = df_norm.agg(["min", "max", "mean", "std"]).T
anotar("")
anotar("4. ESTADÍSTICAS DE CARACTERÍSTICAS BRUTAS")
for caracteristica, fila_est in estadisticas_brutas.iterrows():
    anotar(f"{caracteristica:<20} min={fila_est.iloc[0]:.10f} max={fila_est.iloc[1]:.10f} media={fila_est.iloc[2]:.10f} std={fila_est.iloc[3]:.10f}")
anotar("")
anotar("5. NORMALIZACIÓN")
anotar("Método                 : MinMaxScaler")
anotar("Rango objetivo         : [0,1]")
anotar(f"NaN finales            : {nan_total}")
anotar(f"Infinitos              : {inf_total}")
anotar(f"Fuera de rango         : {fuera_rango}")
anotar("")
anotar("6. ESTADÍSTICAS DE CARACTERÍSTICAS NORMALIZADAS")
for caracteristica, fila_est in estadisticas_normalizadas.iterrows():
    anotar(f"{caracteristica:<20} min={fila_est.iloc[0]:.10f} max={fila_est.iloc[1]:.10f} media={fila_est.iloc[2]:.10f} std={fila_est.iloc[3]:.10f}")
anotar("")
anotar("7. DATASET FINAL")
anotar(f"Registros              : {len(df_final):,}")
anotar(f"Especies               : {df_final.iloc[:, 1].nunique():,}")
anotar(f"Labels                 : {df_final.iloc[:, 2].nunique():,}")
anotar(f"Columnas meta          : {N_COLUMNAS_META}")
anotar(f"Columnas totales       : {df_final.shape[1]}")
anotar("Orden                  : archivo | especie | label | clase | características...")
anotar("")
anotar("8. AUDIOS CON ERROR")
if len(errores) == 0:
    anotar("No se registraron errores.")
for archivo_error, texto_error in errores:
    anotar(f"{archivo_error} | {texto_error}")
anotar("")
anotar("9. ARCHIVOS GENERADOS")
anotar("CSV bruto              : NO SE GUARDA; se usa únicamente en memoria para estadísticas")
anotar(f"CSV normalizado        : {SALIDA_NORMALIZADO}")
anotar(f"Reporte                : {SALIDA_REPORTE}")
with open(SALIDA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
    archivo_reporte.write("\n".join(reporte))
