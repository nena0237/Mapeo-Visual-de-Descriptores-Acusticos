# ==============================================================================
# CÓDIGO 2 — EXTRACCIÓN DE ÍNDICES ACÚSTICOS + NORMALIZACIÓN
# ==============================================================================
# Este código es la extracción de índices ecoacústicos con scikit-maad. Recibe
# el CSV que entregó el filtrado (CÓDIGO 1), abre cada audio y le calcula los
# índices acústicos temporales y espectrales.
#
# Qué hace, en orden:
#   1. Pregunta por teclado el mínimo de audios por especie.
#   2. Lee el CSV final del filtrado y valida que cada especie tenga una sola
#      etiqueta.
#   3. Quita las especies que tienen menos audios que el mínimo elegido.
#   4. Conserva archivo, especie, label y clase.
#   5. Calcula 16 índices temporales + 44 índices espectrales = 60 índices
#      escalares. Se excluye LEQt, así que quedan 59 en el dataset final.
#   6. No guarda un archivo bruto; los valores brutos solo permanecen en memoria
#      y se documentan en el reporte.
#   7. Normaliza todos los índices con MinMaxScaler al rango [0,1].
#   8. Guarda un único CSV final normalizado y un reporte TXT.
#
# IMPORTANTE:
# - El código usa librosa para cargar los audios porque BirdCLEF contiene
#   formatos como OGG y maad.sound.load solo admite WAVE.
# - Los índices se calculan con scikit-maad.
# - El label ya debe venir del CÓDIGO 1 y se conserva sin volverlo a crear.
# - Los audios se cargan con su frecuencia de muestreo original.
# - Las especies que no llegan al mínimo se quitan ANTES de extraer: así la
#   normalización se hace una sola vez y solo con los audios con los que se
#   trabaja.
# - El archivo del filtrado se lee solo y sus columnas se leen por POSICIÓN,
#   nunca por nombre.
# - La lista de índices no está escrita a mano: se toma de lo que devuelve
#   scikit-maad, en el mismo orden en que los devuelve.
#
# DECISIÓN SOBRE LEQt:
# LEQt se audita pero se excluye del dataset final por decisión metodológica.
# No se eliminan audios ni se imputan valores. LEQt produjo valores no válidos
# en una fracción de los audios y depende de parámetros de calibración del
# sistema de grabación que no son homogéneos en BirdCLEF.
#
# NOTA DE CALIBRACIÓN:
# BirdCLEF integra grabaciones provenientes de diferentes fuentes y equipos.
# Los parámetros de gain y sensibilidad se mantienen constantes para ejecutar el
# conjunto completo de índices de scikit-maad. Por tanto, LEQt y LEQf no deben
# interpretarse como mediciones SPL físicamente calibradas salvo que se disponga
# de metadatos reales del equipo.
#
# Las cuatro primeras columnas del CSV final quedan en este orden:
#   archivo | especie | label | clase | índices...
# ==============================================================================

import os
import re
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler

try:
    import maad
    from maad import sound, features
except ImportError as error_importacion:
    raise ImportError("No se encontró scikit-maad. Instala la librería con: pip install scikit-maad") from error_importacion

# ==============================================================================
# BLOQUE 0 — RUTAS, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
CARPETA_AUDIOS = os.path.join(CARPETA_BASE, "train_audio")
CARPETA_ENTRADA = os.path.join(CARPETA_BASE, "salidas", "filtrado")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "indices")

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

# Frecuencia de muestreo a la que se carga cada audio.
# None significa la frecuencia original del audio.
SR_OBJETIVO = None

# Parámetros de scikit-maad.
GAIN = 42                       # ganancia asumida del equipo, en dB
SENSITIVITY = -35               # sensibilidad asumida del micrófono, en dB/V
DB_THRESHOLD = 3                # umbral en dB para detectar actividad y eventos
REJECT_DURATION = 0.01          # duración mínima de un evento, en segundos
R_COMPATIBLE = "soundecology"   # NDSI y BI comparables con el paquete soundecology de R
FLIM_LOW = [0, 1500]            # banda de frecuencias bajas, en Hz
FLIM_MID = [1500, 8000]         # banda de frecuencias medias, en Hz
FLIM_HI = [8000, 20000]         # banda de frecuencias altas, en Hz
NPERSEG = 1024                  # tamaño de la ventana del espectrograma
NOVERLAP = NPERSEG // 2         # traslape entre ventanas
DETREND = True                  # se le resta al audio su promedio antes de calcular

# Índices que se calculan y se auditan, pero no entran al dataset final.
INDICES_EXCLUIDOS = ["LEQt"]

# Tolerancia para los errores de precisión de la normalización.
TOL_NORMALIZACION = 1e-9

# Cada cuántos audios se muestra el avance en pantalla.
MOSTRAR_CADA = 100

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


# Recibe la ruta de un audio y devuelve tres cosas: un diccionario con los
# índices temporales, un diccionario con los índices espectrales y la
# frecuencia de muestreo con la que se cargó el audio.
def extraer_indices(ruta_audio):
    # Carga del audio en mono.
    wave, fs = librosa.load(ruta_audio, sr=SR_OBJETIVO, mono=True)
    if wave.size == 0:
        raise ValueError("Audio vacío.")
    if not np.isfinite(wave).all():
        raise ValueError("El audio contiene valores NaN o infinitos.")
    # Detrend: se le resta al audio su promedio.
    if DETREND:
        wave = wave - np.mean(wave)
    if np.sum(np.square(wave, dtype=np.float64)) == 0:
        raise ValueError("Audio sin energía útil.")
    # Índices temporales: se calculan sobre la onda del audio.
    df_temporal = features.all_temporal_alpha_indices(s=wave, fs=fs, gain=GAIN, sensitivity=SENSITIVITY, dB_threshold=DB_THRESHOLD, rejectDuration=REJECT_DURATION, verbose=False, display=False)
    # Espectrograma de potencia del audio.
    Sxx_power, tn, fn, ext = sound.spectrogram(x=wave, fs=fs, window="hann", nperseg=NPERSEG, noverlap=NOVERLAP, verbose=False, display=False, savefig=None)
    # Índices espectrales: se calculan sobre el espectrograma.
    df_espectral, _ = features.all_spectral_alpha_indices(Sxx_power=Sxx_power, tn=tn, fn=fn, flim_low=FLIM_LOW, flim_mid=FLIM_MID, flim_hi=FLIM_HI, gain=GAIN, sensitivity=SENSITIVITY, verbose=False, R_compatible=R_COMPATIBLE, display=False)
    indices_temporales = df_temporal.iloc[0].to_dict()
    indices_espectrales = df_espectral.iloc[0].to_dict()
    return indices_temporales, indices_espectrales, int(fs)


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
SALIDA_NORMALIZADO = os.path.join(CARPETA_SALIDA, f"Indices_{NOMBRE_DATASET}.csv")
SALIDA_REPORTE = os.path.join(CARPETA_SALIDA, f"Reporte_Indices_{NOMBRE_DATASET}.txt")
os.makedirs(CARPETA_SALIDA, exist_ok=True)
df = pd.read_csv(RUTA_CSV)
if len(df) == 0:
    raise SystemExit("El CSV del filtrado está vacío.")
anotar("=" * 80)
anotar("REPORTE DE EXTRACCIÓN DE ÍNDICES ACÚSTICOS")
anotar("=" * 80)
anotar("1. CONFIGURACIÓN")
anotar(f"Archivo de entrada     : {RUTA_CSV}")
anotar(f"Dataset                : {NOMBRE_DATASET}")
anotar(f"Python                 : {sys.version.split()[0]}")
anotar(f"scikit-maad / maad     : {getattr(maad, '__version__', 'NO_DISPONIBLE')}")
anotar(f"Frecuencia de muestreo : {'original de cada audio' if SR_OBJETIVO is None else SR_OBJETIVO}")
anotar(f"Gain asumido           : {GAIN} dB")
anotar(f"Sensibilidad asumida   : {SENSITIVITY} dB/V")
anotar(f"dB threshold           : {DB_THRESHOLD}")
anotar(f"Reject duration        : {REJECT_DURATION} s")
anotar(f"R compatible           : {R_COMPATIBLE}")
anotar(f"Flim low               : {FLIM_LOW}")
anotar(f"Flim mid               : {FLIM_MID}")
anotar(f"Flim high              : {FLIM_HI}")
anotar(f"Nperseg                : {NPERSEG}")
anotar(f"Noverlap               : {NOVERLAP}")
anotar(f"Detrend                : {DETREND}")

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
# BLOQUE 3 — EXTRACCIÓN DE ÍNDICES
# Para cada audio:
#   - Se arma su ruta uniendo la carpeta de audios con la columna del archivo.
#   - Se calculan los índices con la función del bloque de funciones.
#   - Se guardan por separado las 4 columnas meta y los índices.
# Si un audio falla, se anota el error y se continúa con el siguiente.
# Los avisos repetidos de scikit-maad se silencian para que no llenen la pantalla.
# ==============================================================================
filas_meta = []
filas_indices = []
errores = []
frecuencias_muestreo = []
nombres_temporales = []
nombres_espectrales = []
revisados = 0
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
print("")
print("Extrayendo índices acústicos...")
for fila in df.itertuples(index=False, name=None):
    revisados += 1
    archivo_audio = str(fila[POS_ARCHIVO])
    ruta_audio = os.path.join(CARPETA_AUDIOS, archivo_audio.replace("/", os.sep))
    try:
        indices_temporales, indices_espectrales, fs = extraer_indices(ruta_audio)
        indices = {}
        indices.update(indices_temporales)
        indices.update(indices_espectrales)
        filas_meta.append([fila[POS_ARCHIVO], fila[POS_ESPECIE], fila[POS_LABEL], fila[POS_CLASE]])
        filas_indices.append(indices)
        frecuencias_muestreo.append(fs)
        nombres_temporales = list(indices_temporales.keys())
        nombres_espectrales = list(indices_espectrales.keys())
    except Exception as error:
        errores.append((archivo_audio, str(error)))
    if revisados % MOSTRAR_CADA == 0 or revisados == len(df):
        print(f"Revisados: {revisados:,}/{len(df):,} | Procesados: {len(filas_indices):,} | Errores: {len(errores):,}")
if len(filas_indices) == 0:
    raise SystemExit("No se pudo procesar ningún audio. Revise la carpeta de audios del BLOQUE 0.")

# ==============================================================================
# BLOQUE 4 — CONSTRUCCIÓN DE LAS TABLAS EN MEMORIA Y AUDITORÍA
# La tabla meta conserva los nombres de columna que traía el CSV del filtrado.
# La tabla de índices tiene una columna por índice, en el orden de scikit-maad.
# Los índices excluidos (LEQt) se auditan: se cuenta cuántos valores inválidos
# tuvieron y después se retiran. No se eliminan audios ni se imputan valores.
# Si algún índice conservado tiene NaN o infinitos, el código se detiene: no se
# imputa ni se elimina nada de forma automática para no modificar la metodología
# sin revisión.
# ==============================================================================
nombres_meta = [df.columns[POS_ARCHIVO], df.columns[POS_ESPECIE], df.columns[POS_LABEL], df.columns[POS_CLASE]]
df_meta = pd.DataFrame(filas_meta, columns=nombres_meta)
df_todos = pd.DataFrame(filas_indices)
df_todos = df_todos.apply(pd.to_numeric, errors="coerce")
N_COLUMNAS_META = df_meta.shape[1]
temporales_conservados = [indice for indice in nombres_temporales if indice not in INDICES_EXCLUIDOS]
espectrales_conservados = [indice for indice in nombres_espectrales if indice not in INDICES_EXCLUIDOS]
frecuencias_unicas = sorted(set(frecuencias_muestreo))
anotar("")
anotar("3. PROCESAMIENTO")
anotar(f"Audios procesados      : {len(df_todos):,}")
anotar(f"Audios con error       : {len(errores):,}")
anotar(f"Frecuencias muestreo   : {frecuencias_unicas}")
anotar(f"Nyquist mínimo         : {min(frecuencias_unicas) / 2:.2f} Hz")
anotar("")
anotar("4. ÍNDICES CALCULADOS Y DECISIÓN SOBRE LOS EXCLUIDOS")
anotar(f"Temporales conservados : {len(temporales_conservados)}")
anotar(f"Espectrales conservados: {len(espectrales_conservados)}")
anotar(f"Total escalares finales: {len(temporales_conservados) + len(espectrales_conservados)}")
for indice_excluido in INDICES_EXCLUIDOS:
    valores_excluido = df_todos[indice_excluido].to_numpy(dtype=float)
    invalidos_excluido = int((~np.isfinite(valores_excluido)).sum())
    anotar(f"{indice_excluido:<23}: EXCLUIDO | valores inválidos detectados: {invalidos_excluido}")
anotar("Decisión metodológica  : se eliminan como característica; no se eliminan audios y no se imputan valores.")
df_bruto = df_todos.drop(columns=INDICES_EXCLUIDOS)
valores_brutos = df_bruto.to_numpy(dtype=float)
invalidos_por_indice = pd.Series((~np.isfinite(valores_brutos)).sum(axis=0), index=df_bruto.columns)
indices_con_invalidos = invalidos_por_indice[invalidos_por_indice > 0]
if len(indices_con_invalidos) > 0:
    anotar("")
    anotar("PROCESO DETENIDO: se encontraron valores NaN o infinitos en los índices.")
    anotar("No se realizó imputación ni eliminación automática para no modificar la metodología sin revisión.")
    for indice_invalido, cantidad_invalidos in indices_con_invalidos.items():
        anotar(f"{indice_invalido}: {int(cantidad_invalidos):,}")
    with open(SALIDA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
        archivo_reporte.write("\n".join(reporte))
    raise SystemExit(f"Hay índices con valores NaN/inf. Revise el reporte: {SALIDA_REPORTE}")

# ==============================================================================
# BLOQUE 5 — NORMALIZACIÓN MIN-MAX
# MinMaxScaler lleva cada índice al rango [0,1]. Por aritmética de punto
# flotante pueden aparecer desviaciones microscópicas como 1.0000000000000002 o
# -2e-16. Se aceptan únicamente dentro de una tolerancia numérica estricta y
# luego se recortan exactamente a [0,1].
# ==============================================================================
scaler = MinMaxScaler()
valores_normalizados = scaler.fit_transform(df_bruto)
min_antes_clip = float(np.nanmin(valores_normalizados))
max_antes_clip = float(np.nanmax(valores_normalizados))
nan_final = int(np.isnan(valores_normalizados).sum())
inf_final = int(np.isinf(valores_normalizados).sum())
fuera_tolerancia = int(((valores_normalizados < -TOL_NORMALIZACION) | (valores_normalizados > 1 + TOL_NORMALIZACION)).sum())
ajustes_precision = int(((valores_normalizados < 0) | (valores_normalizados > 1)).sum())
valores_normalizados = np.clip(valores_normalizados, 0.0, 1.0)
df_norm = pd.DataFrame(valores_normalizados, columns=df_bruto.columns)

# ==============================================================================
# BLOQUE 6 — VALIDACIÓN DE LA NORMALIZACIÓN
# Solo se recortan desviaciones numéricas microscópicas dentro de la tolerancia.
# Valores no finitos o realmente fuera del rango detienen el proceso.
# ==============================================================================
fuera_rango = int(((valores_normalizados < 0) | (valores_normalizados > 1)).sum())
if nan_final != 0 or inf_final != 0:
    raise SystemExit(f"La normalización produjo valores no finitos. NaN={nan_final}, Inf={inf_final}.")
if fuera_tolerancia != 0:
    raise SystemExit(f"La normalización produjo {fuera_tolerancia} valores realmente fuera de [0,1]. Min={min_antes_clip}, Max={max_antes_clip}.")
if fuera_rango != 0:
    raise SystemExit("La validación final [0,1] falló después del ajuste de precisión.")
if len(df_norm) != len(df_meta):
    raise SystemExit("El número de registros cambió durante la normalización.")

# ==============================================================================
# BLOQUE 7 — GUARDADO DEL ÚNICO CSV FINAL NORMALIZADO
# Se unen las 4 columnas meta con los índices normalizados. Este es el único
# archivo de índices que se guarda en disco.
# ==============================================================================
df_final = pd.concat([df_meta, df_norm], axis=1)
df_final.to_csv(SALIDA_NORMALIZADO, index=False)

# ==============================================================================
# BLOQUE 8 — REPORTE
# Se documentan la normalización, el dataset final y los audios con error. La
# lista de índices y sus estadísticas brutas y normalizadas se escriben solo en
# el archivo del reporte, porque son muchas líneas para la pantalla.
# En el CSV final la especie está en la posición 1 y el label en la 2.
# ==============================================================================
estadisticas_brutas = df_bruto.agg(["min", "max", "mean", "std"]).T
estadisticas_normalizadas = df_norm.agg(["min", "max", "mean", "std"]).T
anotar("")
anotar("5. NORMALIZACIÓN")
anotar("Método                 : MinMaxScaler")
anotar("Rango objetivo         : [0,1]")
anotar(f"Tolerancia numérica    : {TOL_NORMALIZACION}")
anotar(f"Min antes de clip      : {min_antes_clip:.18f}")
anotar(f"Max antes de clip      : {max_antes_clip:.18f}")
anotar(f"Ajustes de precisión   : {ajustes_precision}")
anotar(f"Fuera de tolerancia    : {fuera_tolerancia}")
anotar(f"NaN finales            : {nan_final}")
anotar(f"Infinitos finales      : {inf_final}")
anotar(f"Fuera de rango final   : {fuera_rango}")
anotar("")
anotar("6. DATASET FINAL")
anotar(f"Registros              : {len(df_final):,}")
anotar(f"Especies               : {df_final.iloc[:, 1].nunique():,}")
anotar(f"Labels                 : {df_final.iloc[:, 2].nunique():,}")
anotar(f"Columnas meta          : {N_COLUMNAS_META}")
anotar(f"Índices finales        : {df_norm.shape[1]}")
anotar(f"Columnas totales       : {df_final.shape[1]}")
anotar("Orden                  : archivo | especie | label | clase | índices...")
anotar("")
anotar("7. AUDIOS CON ERROR")
if len(errores) == 0:
    anotar("No se registraron errores.")
for archivo_error, texto_error in errores:
    anotar(f"{archivo_error} | {texto_error}")
anotar("")
anotar("8. ARCHIVOS GENERADOS")
anotar("Archivo bruto          : NO SE GUARDA; solo se usa temporalmente en memoria y en el reporte.")
anotar(f"CSV normalizado        : {SALIDA_NORMALIZADO}")
anotar(f"Reporte                : {SALIDA_REPORTE}")
reporte.append("")
reporte.append("9. ÍNDICES TEMPORALES")
numero_indice = 0
for indice in temporales_conservados:
    reporte.append(f"[{numero_indice}] {indice}")
    numero_indice += 1
reporte.append("")
reporte.append("10. ÍNDICES ESPECTRALES")
numero_indice = 0
for indice in espectrales_conservados:
    reporte.append(f"[{numero_indice}] {indice}")
    numero_indice += 1
reporte.append("")
reporte.append("11. ESTADÍSTICAS BRUTAS")
for indice, fila_est in estadisticas_brutas.iterrows():
    reporte.append(f"{indice:<22} min={fila_est.iloc[0]:.10f} max={fila_est.iloc[1]:.10f} media={fila_est.iloc[2]:.10f} std={fila_est.iloc[3]:.10f}")
reporte.append("")
reporte.append("12. ESTADÍSTICAS NORMALIZADAS")
for indice, fila_est in estadisticas_normalizadas.iterrows():
    reporte.append(f"{indice:<22} min={fila_est.iloc[0]:.10f} max={fila_est.iloc[1]:.10f} media={fila_est.iloc[2]:.10f} std={fila_est.iloc[3]:.10f}")
with open(SALIDA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
    archivo_reporte.write("\n".join(reporte))
