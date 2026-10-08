# ==============================================================================
# CÓDIGO 8 — MAIN (NUEVA REPRESENTACION VISUAL + EFFICIENTNET-B0)
# ==============================================================================
# Este código entrena y evalúa la red con las imágenes de la representacion creada en 04_plantilla. Es el
# main del estudio que dio los resultados más altos (antes eran dos códigos, uno
# para las características y otro para los índices: Main_conAUC_70_30.py y
# Main_conAUC_indices57_70_30.py). Aquí quedan en uno solo: se hace lo mismo
# para cada tabla, una después de la otra.
#
# Este código NO repite nada de los códigos anteriores:
#   - El split (qué audios son de entrenamiento y cuáles de prueba) ya lo
#     entregó 03_metricas.py. Aquí solo se lee.
#   - La plantilla, los colores y las imágenes los entrega 06_Data_Loader.py.
#     Aquí solo se piden las imágenes ya pintadas y sus etiquetas.
#   - Las rutas del proyecto se toman del código de colores.
#
# EL MODELO (es el mismo del estudio, no se cambia):
#   - EfficientNet-B0 ya entrenada con ImageNet (transfer learning).
#   - La primera convolución se cambia de 3 canales (color) a 1 canal (grises),
#     porque las imágenes de la plantilla son en escala de grises. Sus pesos
#     iniciales son el promedio de los 3 canales originales.
#   - Toda la red queda congelada. Solo se entrenan la primera convolución y el
#     clasificador nuevo (Dropout + una capa con una salida por especie).
#   - Función de pérdida: CrossEntropyLoss. Optimizador: Adam.
#
# LO QUE SE MIDE EN CADA ÉPOCA:
#   - En entrenamiento (Train): pérdida, accuracy, recall, F1 y AUC.
#   - En prueba (Val): pérdida, accuracy, recall, F1 y AUC. No se entrena con
#     estos audios.
#   - Accuracy: porcentaje de audios en que la red acertó la especie.
#   - Recall y F1 son "macro": se calculan para cada especie y se promedian, así
#     todas las especies pesan lo mismo aunque tengan pocos audios. Una especie
#     que la red nunca predice cuenta como 0.
#   - El AUC es multiclase: se calcula el AUC de cada especie contra todas las
#     demás y se promedian.
#   - Estas medidas solo se calculan: no cambian el entrenamiento. La red
#     aprende solo con la pérdida (CrossEntropyLoss).
#   - Al final se muestra la MEJOR ÉPOCA, que es la de mayor AUC en prueba.
#
# IMPORTANTE:
# - La semilla se fija de nuevo antes de cada tabla. Así el resultado de cada
#   tabla es el mismo que si se entrenara sola y se puede repetir.
# - Para repetir el estudio, en 03_metricas.py el split debe ser 70/30 con
#   semilla 1024, y aquí la semilla, las épocas y los demás valores del BLOQUE 0
#   se dejan como están.
# - No se pregunta nada por teclado y todo se lee por POSICIÓN de columna.
# - Lo único que se guarda es un reporte de texto por tabla.
# ==============================================================================

import os
import glob
import random
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ==============================================================================
# BLOQUE 0 — CONEXIONES, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Nombre del archivo del DataLoader, sin el ".py". Debe estar en la misma
# carpeta que este archivo.
ARCHIVO_DATALOADER = "06_Data_Loader"

# Tablas con las que se entrena. Se entrena con cada una que tenga plantilla.
TABLAS = ["Caracteristicas", "Indices"]

# Plantilla que se va a usar. Si se deja vacío, se toma sola la más reciente de
# cada tabla. Para fijar una se escribe el dataset, el método y el tamaño,
# ejemplo: "Aves_0-5s_XC_ReliefF_128px"
NOMBRE_PLANTILLA = ""

# Split que se va a usar. Si se deja vacío, se toma solo el más reciente del
# mismo dataset de la plantilla. Para fijar uno se escribe la división y la
# semilla, ejemplo: "70-30_semilla1024"
NOMBRE_SPLIT = ""

# Parámetros del entrenamiento (los del estudio).
SEMILLA = 1024            # semilla del entrenamiento
NUM_EPOCAS = 15           # veces que la red ve todos los audios de entrenamiento
TAMANO_LOTE = 20          # audios que la red ve en cada paso
TASA_APRENDIZAJE = 1e-3   # tamaño del paso con el que Adam ajusta los pesos
DROPOUT = 0.3             # parte de las conexiones que se apagan en el clasificador

# Posiciones en el CSV del split de 03_metricas.py.
POS_SPLIT_ARCHIVO = 0   # ruta del audio dentro de la carpeta de audios
POS_SPLIT_SET = 3       # conjunto del audio: "train" o "test"

# ==============================================================================
# BLOQUE 1 — CONEXIÓN CON EL DATALOADER
# Se importa el DataLoader con importlib porque el nombre del archivo empieza
# por un número. Del DataLoader sale la clase PlantillaDataset, y del código de
# colores que él ya tiene cargado salen las rutas del proyecto.
# ==============================================================================
cargador_datos = importlib.import_module(ARCHIVO_DATALOADER)
PlantillaDataset = cargador_datos.PlantillaDataset
CARPETA_PLANTILLA = cargador_datos.colores.CARPETA_PLANTILLA
CARPETA_SPLIT = cargador_datos.colores.CARPETA_TABLAS
CARPETA_SALIDA = os.path.join(cargador_datos.colores.CARPETA_BASE, "salidas", "main")

# El entrenamiento se hace en la tarjeta gráfica si hay una; si no, en el procesador.
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lista donde se guarda cada línea del reporte de la tabla que se está entrenando.
reporte = []

# ==============================================================================
# BLOQUE DE FUNCIONES
# ==============================================================================

# Muestra un texto en pantalla y lo guarda para el reporte.
def anotar(texto):
    print(texto, flush=True)
    reporte.append(texto)


# Fija la semilla en todas las librerías que usan números al azar, para que el
# entrenamiento dé siempre el mismo resultado.
def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Devuelve la ruta del split que le corresponde a una plantilla.
# La plantilla se llama <dataset>_<método>_<tamaño>px y el split se llama
# Split_<dataset>_<división>_semilla<n>.csv, así que se buscan los splits que
# son del mismo dataset. Si no se fijó uno en el BLOQUE 0, se toma el más reciente.
def buscar_split(nombre_plantilla):
    nombre_dataset = nombre_plantilla.rsplit("_", 2)[0]
    if NOMBRE_SPLIT != "":
        ruta_fijada = os.path.join(CARPETA_SPLIT, f"Split_{nombre_dataset}_{NOMBRE_SPLIT}.csv")
        if not os.path.exists(ruta_fijada):
            raise SystemExit(f"No se encontró el split: {ruta_fijada}")
        return ruta_fijada
    archivos_dataset = []
    for ruta in glob.glob(os.path.join(CARPETA_SPLIT, f"Split_{nombre_dataset}_*.csv")):
        nombre_archivo = os.path.splitext(os.path.basename(ruta))[0]
        if nombre_archivo.rsplit("_", 2)[0] == f"Split_{nombre_dataset}":
            archivos_dataset.append(ruta)
    if len(archivos_dataset) == 0:
        raise SystemExit(f"No hay ningún split del dataset {nombre_dataset}. Ejecute primero 03_metricas.py.")
    return max(archivos_dataset, key=os.path.getmtime)


# Crea la red EfficientNet-B0 del estudio para un número de especies.
# El orden de los pasos no se cambia, porque de él depende que el resultado se
# pueda repetir con la misma semilla.
def crear_modelo(numero_clases):
    # Red ya entrenada con ImageNet.
    modelo = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Primera convolución de 1 canal en lugar de 3, con el promedio de los pesos originales.
    conv_original = modelo.features[0][0]
    modelo.features[0][0] = nn.Conv2d(in_channels=1, out_channels=conv_original.out_channels, kernel_size=conv_original.kernel_size, stride=conv_original.stride, padding=conv_original.padding, bias=False)
    with torch.no_grad():
        modelo.features[0][0].weight[:] = conv_original.weight.mean(dim=1, keepdim=True)
    # Se congela toda la red y se deja entrenable solo la primera convolución.
    for parametro in modelo.parameters():
        parametro.requires_grad = False
    for parametro in modelo.features[0][0].parameters():
        parametro.requires_grad = True
    # Clasificador nuevo, con una salida por especie. Como es nuevo, es entrenable.
    entradas_clasificador = modelo.classifier[1].in_features
    modelo.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(entradas_clasificador, numero_clases))
    return modelo.to(DISPOSITIVO)


# Calcula el AUC multiclase: el AUC de cada especie contra todas las demás, y
# después el promedio. Recibe la etiqueta real de cada audio y la probabilidad
# que la red le dio a cada especie.
def calcular_auc(etiquetas, probabilidades):
    aucs_por_clase = []
    for clase in np.unique(etiquetas):
        es_la_clase = (etiquetas == clase).astype(int)
        if es_la_clase.sum() == len(es_la_clase):
            continue
        aucs_por_clase.append(roc_auc_score(es_la_clase, probabilidades[:, clase]))
    if len(aucs_por_clase) == 0:
        return float("nan")
    return float(np.mean(aucs_por_clase))


# Junta lo que se fue guardando lote por lote en una época y devuelve las
# medidas en este orden: pérdida promedio, accuracy, recall, F1 (las tres en
# porcentaje) y AUC (de 0 a 1).
def resumir_epoca(suma_perdida, numero_lotes, lista_etiquetas, lista_predichas, lista_probabilidades):
    etiquetas = np.concatenate(lista_etiquetas, axis=0)
    predichas = np.concatenate(lista_predichas, axis=0)
    probabilidades = np.concatenate(lista_probabilidades, axis=0)
    especies = list(range(probabilidades.shape[1]))
    perdida = suma_perdida / numero_lotes
    exactitud = 100 * float((predichas == etiquetas).mean())
    recall = 100 * recall_score(etiquetas, predichas, labels=especies, average="macro", zero_division=0)
    f1 = 100 * f1_score(etiquetas, predichas, labels=especies, average="macro", zero_division=0)
    auc = calcular_auc(etiquetas, probabilidades)
    return perdida, exactitud, recall, f1, auc


# Convierte las medidas de una época en el texto que se muestra y se guarda.
def texto_medidas(nombre, medidas):
    perdida, exactitud, recall, f1, auc = medidas
    return f"{nombre} Loss: {perdida:.4f}, {nombre} Accuracy: {exactitud:.2f}%, {nombre} Recall: {recall:.2f}%, {nombre} F1: {f1:.2f}%, {nombre} AUC: {auc:.4f}"


# Entrena la red una época: recorre todos los lotes de entrenamiento y en cada
# uno calcula la pérdida y ajusta los pesos. Devuelve las medidas de la época.
def entrenar_epoca(modelo, cargador, criterio, optimizador):
    modelo.train()
    suma_perdida = 0.0
    lista_etiquetas = []
    lista_predichas = []
    lista_probabilidades = []
    for imagenes, etiquetas in cargador:
        imagenes = imagenes.to(DISPOSITIVO)
        etiquetas = etiquetas.to(DISPOSITIVO)
        optimizador.zero_grad()
        salidas = modelo(imagenes)
        perdida = criterio(salidas, etiquetas)
        perdida.backward()
        optimizador.step()
        suma_perdida += perdida.item()
        lista_etiquetas.append(etiquetas.cpu().numpy())
        lista_predichas.append(salidas.argmax(dim=1).cpu().numpy())
        lista_probabilidades.append(torch.softmax(salidas, dim=1).detach().cpu().numpy())
    return resumir_epoca(suma_perdida, len(cargador), lista_etiquetas, lista_predichas, lista_probabilidades)


# Evalúa la red con los audios de prueba, sin ajustar ningún peso.
# Devuelve las medidas de la época y la especie que predijo para cada audio.
def evaluar(modelo, cargador, criterio):
    modelo.eval()
    suma_perdida = 0.0
    lista_etiquetas = []
    lista_predichas = []
    lista_probabilidades = []
    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)
            salidas = modelo(imagenes)
            perdida = criterio(salidas, etiquetas)
            suma_perdida += perdida.item()
            lista_etiquetas.append(etiquetas.cpu().numpy())
            lista_predichas.append(salidas.argmax(dim=1).cpu().numpy())
            lista_probabilidades.append(torch.softmax(salidas, dim=1).cpu().numpy())
    medidas = resumir_epoca(suma_perdida, len(cargador), lista_etiquetas, lista_predichas, lista_probabilidades)
    predichas = np.concatenate(lista_predichas, axis=0)
    return medidas, predichas


# ==============================================================================
# BLOQUE 2 — TABLAS CON LAS QUE SE ENTRENA
# De la lista del BLOQUE 0 se dejan solo las tablas que tienen plantilla.
# ==============================================================================
tablas_con_plantilla = []
for tipo_tabla in TABLAS:
    if len(glob.glob(os.path.join(CARPETA_PLANTILLA, f"Plantilla_{tipo_tabla}_*.csv"))) > 0:
        tablas_con_plantilla.append(tipo_tabla)
if len(tablas_con_plantilla) == 0:
    raise SystemExit("No hay ninguna plantilla. Ejecute primero 04_plantilla.py.")
os.makedirs(CARPETA_SALIDA, exist_ok=True)
print(f"Dispositivo: {DISPOSITIVO}")

# ==============================================================================
# BLOQUE 3 — ENTRENAMIENTO Y EVALUACIÓN DE CADA TABLA
# Se repite lo mismo para cada tabla:
#   3.1 Se fija la semilla.
#   3.2 Se crean las imágenes y se reparten en entrenamiento y prueba con el split.
#   3.3 Se crea el modelo.
#   3.4 Se entrena y se evalúa en cada época.
#   3.5 Se busca la mejor época y se guarda el reporte.
# ==============================================================================
for tipo_tabla in tablas_con_plantilla:
    reporte.clear()

    # --------------------------------------------------------------------------
    # 3.1 SEMILLA
    # --------------------------------------------------------------------------
    fijar_semilla(SEMILLA)

    # --------------------------------------------------------------------------
    # 3.2 IMÁGENES Y SPLIT
    # El DataLoader entrega la imagen y la etiqueta de cada audio. El split dice
    # a qué conjunto pertenece cada audio y se une por la ruta del audio.
    # --------------------------------------------------------------------------
    dataset = PlantillaDataset(tipo_tabla, NOMBRE_PLANTILLA)
    ruta_split = buscar_split(dataset.nombre_plantilla)
    split = pd.read_csv(ruta_split)
    conjunto_por_audio = dict(zip(split.iloc[:, POS_SPLIT_ARCHIVO], split.iloc[:, POS_SPLIT_SET]))
    conjuntos = pd.Series(dataset.archivos).map(conjunto_por_audio).to_numpy()
    posiciones_train = np.where(conjuntos == "train")[0].tolist()
    posiciones_test = np.where(conjuntos == "test")[0].tolist()
    if len(posiciones_train) + len(posiciones_test) != len(dataset):
        raise SystemExit(f"Hay audios de la tabla {tipo_tabla} que no están en el split: {ruta_split}")
    cargador_train = DataLoader(Subset(dataset, posiciones_train), batch_size=TAMANO_LOTE, shuffle=True)
    cargador_test = DataLoader(Subset(dataset, posiciones_test), batch_size=TAMANO_LOTE)
    texto_split = os.path.splitext(os.path.basename(ruta_split))[0].rsplit("_", 2)
    RUTA_REPORTE = os.path.join(CARPETA_SALIDA, f"Main_{tipo_tabla}_{dataset.nombre_plantilla}_{texto_split[1]}_{texto_split[2]}.txt")
    anotar("=" * 70)
    anotar(f"RESULTADOS MAIN CON AUC - {tipo_tabla} ({len(dataset.plantilla)} bloques, {dataset.n_classes} especies)")
    anotar("=" * 70)
    anotar(f"Plantilla : {dataset.nombre_plantilla}")
    anotar(f"Split     : {os.path.basename(ruta_split)}")
    anotar(f"Train: {len(posiciones_train)}  Val: {len(posiciones_test)}")
    anotar(f"Semilla: {SEMILLA} | Épocas: {NUM_EPOCAS} | Lote: {TAMANO_LOTE} | Tasa de aprendizaje: {TASA_APRENDIZAJE} | Dropout: {DROPOUT}")
    anotar("")

    # --------------------------------------------------------------------------
    # 3.3 MODELO
    # Solo se le entregan al optimizador los pesos que quedaron entrenables.
    # --------------------------------------------------------------------------
    modelo = crear_modelo(dataset.n_classes)
    criterio = nn.CrossEntropyLoss()
    pesos_entrenables = [parametro for parametro in modelo.parameters() if parametro.requires_grad]
    optimizador = optim.Adam(pesos_entrenables, lr=TASA_APRENDIZAJE)

    # --------------------------------------------------------------------------
    # 3.4 ENTRENAMIENTO Y EVALUACIÓN POR ÉPOCA
    # En cada época primero se entrena con train y después se evalúa con test.
    # --------------------------------------------------------------------------
    # Cada fila del historial es: 0 época | 1 medidas de prueba, y las medidas
    # van en el orden de resumir_epoca: 0 pérdida, 1 accuracy, 2 recall, 3 F1, 4 AUC.
    historial = []
    filas_epocas = []
    for epoca in range(1, NUM_EPOCAS + 1):
        medidas_train = entrenar_epoca(modelo, cargador_train, criterio, optimizador)
        medidas_val, predichas_val = evaluar(modelo, cargador_test, criterio)
        historial.append((epoca, medidas_val))
        filas_epocas.append([epoca] + list(medidas_train) + list(medidas_val))
        anotar(f"Epoch {epoca}/{NUM_EPOCAS}")
        anotar(texto_medidas("Train", medidas_train))
        anotar(texto_medidas("Val", medidas_val))

    # --------------------------------------------------------------------------
    # 3.5 MEJOR ÉPOCA Y REPORTE
    # La mejor época es la de mayor AUC en prueba. Si dos épocas empatan, queda
    # la primera. También se muestra la última época, para comparar.
    # --------------------------------------------------------------------------
    mejor = historial[0]
    for fila in historial:
        if np.isnan(mejor[1][4]) or fila[1][4] > mejor[1][4]:
            mejor = fila
    ultima = historial[-1]
    anotar("")
    anotar("=" * 60)
    anotar(f"MEJOR EPOCA (por Val AUC): {mejor[0]}")
    anotar(texto_medidas("Val", mejor[1]))
    anotar(f"ULTIMA EPOCA: {ultima[0]}")
    anotar(texto_medidas("Val", ultima[1]))
    with open(RUTA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
        archivo_reporte.write("\n".join(reporte))
    print(f"Reporte guardado en: {RUTA_REPORTE}")

    # --------------------------------------------------------------------------
    # 3.6 RESULTADOS POR ÉPOCA Y PREDICCIONES (para 10_graficos.py)
    # Se guardan dos CSV junto al reporte:
    #   - Las medidas de cada época. Columnas: 0 época, 1 a 5 train (pérdida,
    #     accuracy, recall, F1, AUC) y 6 a 10 prueba en el mismo orden.
    #   - Las predicciones de la última época en los audios de prueba.
    #     Columnas: 0 archivo, 1 especie real, 2 especie predicha.
    # --------------------------------------------------------------------------
    columnas_epocas = ["epoca", "train_loss", "train_acc", "train_recall", "train_f1", "train_auc", "test_loss", "test_acc", "test_recall", "test_f1", "test_auc"]
    RUTA_EPOCAS = RUTA_REPORTE.replace(".txt", "_epocas.csv")
    pd.DataFrame(filas_epocas, columns=columnas_epocas).to_csv(RUTA_EPOCAS, index=False)
    especies_reales = dataset.classes[dataset.etiquetas.numpy()[posiciones_test]]
    especies_predichas = dataset.classes[predichas_val]
    predicciones = pd.DataFrame({"archivo": dataset.archivos[posiciones_test], "especie_real": np.asarray(especies_reales), "especie_predicha": np.asarray(especies_predichas)})
    RUTA_PREDICCIONES = RUTA_REPORTE.replace(".txt", "_predicciones.csv")
    predicciones.to_csv(RUTA_PREDICCIONES, index=False)
    print(f"Épocas guardadas en: {RUTA_EPOCAS}")
    print(f"Predicciones guardadas en: {RUTA_PREDICCIONES}")
    print("")
