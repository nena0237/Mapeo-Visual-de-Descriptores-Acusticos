# ==============================================================================
# CÓDIGO 7 — LÍNEA CLÁSICA (CLASIFICADORES TRADICIONALES)
# ==============================================================================
# Este código entrena y evalúa 11 clasificadores tradicionales de scikit-learn
# directamente con los valores de las características, sin imágenes. Es la
# línea clásica del estudio, la que se presentó en el congreso (antes era
# Basica_final.py, que se corría una vez para las características y otra para
# los índices). Aquí se hace lo mismo para cada tabla, una después de la otra.
#
# Este código NO repite nada de los códigos anteriores:
#   - Las tablas con las características seleccionadas y el split los entregó
#     03_metricas.py. Aquí solo se leen.
#   - Las rutas del proyecto y las posiciones de la tabla se toman del código
#     de colores.
#
# LOS 11 CLASIFICADORES (los mismos del estudio, con los mismos parámetros):
#   Regresión logística, KNN, SVM, Árbol de decisión, Random Forest,
#   Extra Trees, AdaBoost, Gradient Boosting, LDA, Naive Bayes y ANN (red
#   neuronal de una capa oculta de 100 neuronas).
#
# LO QUE SE MIDE (en entrenamiento y en prueba, igual en las tres líneas):
#   - AUC: el AUC de cada especie contra todas las demás, y después el
#     promedio. Se calcula con la probabilidad que el clasificador le da a cada
#     especie. La SVM del estudio no calcula probabilidades, así que con ella
#     se usa su puntaje de decisión; así la SVM no se cambia y su resultado
#     sigue siendo el mismo.
#   - Accuracy: porcentaje de audios bien clasificados.
#   - Recall macro: de los audios de cada especie, cuántos se reconocieron;
#     todas las especies pesan lo mismo.
#   - F1 macro: combina precisión y recall; todas las especies pesan lo mismo.
#
# IMPORTANTE:
# - La semilla no se escribe aquí: es la misma del split que se eligió en
#   03_metricas.py y se lee del nombre del archivo del split (..._semilla<n>.csv).
#   Con el split 70/30 y semilla 42 da el resultado del congreso.
# - Los clasificadores que tienen algo al azar usan la semilla, así que con la
#   misma versión de scikit-learn el resultado siempre es el mismo. Con otra
#   versión pueden cambiar unos decimales en algunos clasificadores (LDA y
#   Gradient Boosting).
# - No se pregunta nada por teclado y todo se lee por POSICIÓN de columna.
# - Por cada tabla se guardan dos CSV: la tabla de resultados y las
#   predicciones de cada clasificador en los audios de prueba (para la matriz
#   de confusión de 10_graficos.py).
# ==============================================================================

import os
import glob
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# BLOQUE 0 — CONEXIONES, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Nombre del archivo del código de colores, sin el ".py". Debe estar en la
# misma carpeta que este archivo.
ARCHIVO_COLORES = "05_colores"

# Tablas con las que se entrena. Se entrena con cada una que exista.
TABLAS = ["Caracteristicas", "Indices"]

# Ejecución de 03_metricas.py que se va a usar. Si se deja vacío, se toma sola
# la más reciente. Para fijar una se escribe el dataset y el método,
# ejemplo: "Aves_0-5s_XC_ReliefF"
NOMBRE_EJECUCION = ""

# Split que se va a usar. Si se deja vacío, se toma solo el más reciente del
# mismo dataset. Para fijar uno se escribe la división y la semilla,
# ejemplo: "70-30_semilla42"
NOMBRE_SPLIT = ""

# Posición de la etiqueta numérica de la especie en la tabla de 03_metricas.py.
POS_LABEL = 2

# Posiciones en el CSV del split de 03_metricas.py.
POS_SPLIT_ARCHIVO = 0   # ruta del audio dentro de la carpeta de audios
POS_SPLIT_SET = 3       # conjunto del audio: "train" o "test"

# ==============================================================================
# BLOQUE 1 — CONEXIÓN CON EL CÓDIGO DE COLORES
# Se importa con importlib porque el nombre del archivo empieza por un número.
# De él salen las rutas del proyecto y las posiciones de la tabla. Al
# importarlo no se pinta nada.
# ==============================================================================
colores = importlib.import_module(ARCHIVO_COLORES)
CARPETA_TABLAS = colores.CARPETA_TABLAS
CARPETA_SALIDA = os.path.join(colores.CARPETA_BASE, "salidas", "clasicos")

# Nombres de las medidas, en el orden en que salen en pantalla y en el CSV.
MEDIDAS = ["auc", "acc", "recall_m", "f1_m"]

# ==============================================================================
# BLOQUE DE FUNCIONES
# ==============================================================================

# Crea los 11 clasificadores con la semilla del split. Cada fila es: nombre del
# clasificador y el clasificador con sus parámetros. Son los mismos del
# estudio; si se cambia uno, cambia el resultado.
def crear_clasificadores(semilla):
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("KNN", KNeighborsClassifier()),
        ("SVM", SVC(kernel="rbf")),
        ("Decision Tree", DecisionTreeClassifier(random_state=semilla)),
        ("Random Forest", RandomForestClassifier(random_state=semilla)),
        ("Extra Trees", ExtraTreesClassifier(random_state=semilla)),
        ("AdaBoost", AdaBoostClassifier(random_state=semilla)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=semilla)),
        ("LDA", LinearDiscriminantAnalysis()),
        ("Naive Bayes", GaussianNB()),
        ("ANN", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=semilla)),
    ]


# Devuelve el nombre de la ejecución más reciente de 03_metricas.py que tenga
# la tabla pedida. Las tablas se llaman <tabla>_<dataset>_<método>.csv y el
# nombre que se devuelve es la parte <dataset>_<método>.
def buscar_ejecucion_reciente(tipo_tabla):
    archivos_tabla = glob.glob(os.path.join(CARPETA_TABLAS, f"{tipo_tabla}_*.csv"))
    if len(archivos_tabla) == 0:
        return ""
    archivo_reciente = max(archivos_tabla, key=os.path.getmtime)
    nombre_reciente = os.path.splitext(os.path.basename(archivo_reciente))[0]
    return nombre_reciente.split("_", 1)[1]


# Devuelve la ruta del split que le corresponde a una ejecución.
# La ejecución se llama <dataset>_<método> y el split se llama
# Split_<dataset>_<división>_semilla<n>.csv, así que se buscan los splits que
# son del mismo dataset. Si no se fijó uno en el BLOQUE 0, se toma el más reciente.
def buscar_split(nombre_ejecucion):
    nombre_dataset = nombre_ejecucion.rsplit("_", 1)[0]
    if NOMBRE_SPLIT != "":
        ruta_fijada = os.path.join(CARPETA_TABLAS, f"Split_{nombre_dataset}_{NOMBRE_SPLIT}.csv")
        if not os.path.exists(ruta_fijada):
            raise SystemExit(f"No se encontró el split: {ruta_fijada}")
        return ruta_fijada
    archivos_dataset = []
    for ruta in glob.glob(os.path.join(CARPETA_TABLAS, f"Split_{nombre_dataset}_*.csv")):
        nombre_archivo = os.path.splitext(os.path.basename(ruta))[0]
        if nombre_archivo.rsplit("_", 2)[0] == f"Split_{nombre_dataset}":
            archivos_dataset.append(ruta)
    if len(archivos_dataset) == 0:
        raise SystemExit(f"No hay ningún split del dataset {nombre_dataset}. Ejecute primero 03_metricas.py.")
    return max(archivos_dataset, key=os.path.getmtime)


# Devuelve el puntaje que el clasificador le da a cada especie para cada
# audio: la probabilidad si el clasificador la calcula, o si no (SVM) su
# puntaje de decisión. Cada columna es una especie, en el orden de clasificador.classes_.
def calcular_puntajes(clasificador, X):
    if hasattr(clasificador, "predict_proba"):
        return clasificador.predict_proba(X)
    return clasificador.decision_function(X)


# Calcula el AUC multiclase: el AUC de cada especie contra todas las demás, y
# después el promedio. Solo entran las especies que tienen audios en el conjunto.
def calcular_auc(reales, puntajes, clases_clasificador):
    reales = np.asarray(reales)
    aucs_por_clase = []
    for posicion_clase, clase in enumerate(clases_clasificador):
        es_la_clase = (reales == clase).astype(int)
        if es_la_clase.sum() == 0:
            continue
        if es_la_clase.sum() == len(es_la_clase):
            continue
        aucs_por_clase.append(roc_auc_score(es_la_clase, puntajes[:, posicion_clase]))
    if len(aucs_por_clase) == 0:
        return float("nan")
    return float(np.mean(aucs_por_clase))


# Calcula las 4 medidas de un conjunto (train o test). Se le pasan todas las
# especies para que las que no aparecen en el conjunto también cuenten (con 0).
# Devuelve las medidas en el orden de MEDIDAS.
def calcular_medidas(clasificador, X_conjunto, reales, especies):
    predichas = clasificador.predict(X_conjunto)
    auc = calcular_auc(reales, calcular_puntajes(clasificador, X_conjunto), clasificador.classes_)
    exactitud = accuracy_score(reales, predichas)
    _, recall_m, f1_m, _ = precision_recall_fscore_support(reales, predichas, labels=especies, average="macro", zero_division=0)
    return [auc, 100 * exactitud, 100 * recall_m, f1_m]


# Convierte las 4 medidas en el texto que se muestra en pantalla.
def texto_medidas(medidas):
    return f"AUC {medidas[0]:.4f} | Acc {medidas[1]:.2f}% | Recall(m) {medidas[2]:.2f}% | F1(m) {medidas[3]:.4f}"


# ==============================================================================
# BLOQUE 2 — ENTRENAMIENTO Y EVALUACIÓN DE CADA TABLA
# Se repite lo mismo para cada tabla:
#   3.1 Se cargan la tabla y el split, y se toma la semilla del split.
#   3.2 Se separan los valores (X) y la especie (y) en entrenamiento y prueba.
#   3.3 Se entrena y se evalúa cada clasificador.
#   3.4 Se guarda la tabla de resultados.
# ==============================================================================
os.makedirs(CARPETA_SALIDA, exist_ok=True)
tablas_entrenadas = 0
for tipo_tabla in TABLAS:

    # --------------------------------------------------------------------------
    # 3.1 TABLA, SPLIT Y SEMILLA
    # Si no se fijó una ejecución en el BLOQUE 0, se toma la más reciente.
    # El split se llama Split_<dataset>_<división>_semilla<n>.csv: de ahí
    # salen la división y la semilla.
    # --------------------------------------------------------------------------
    nombre_ejecucion = NOMBRE_EJECUCION
    if nombre_ejecucion == "":
        nombre_ejecucion = buscar_ejecucion_reciente(tipo_tabla)
    ruta_tabla = os.path.join(CARPETA_TABLAS, f"{tipo_tabla}_{nombre_ejecucion}.csv")
    if not os.path.exists(ruta_tabla):
        continue
    tabla = pd.read_csv(ruta_tabla)
    ruta_split = buscar_split(nombre_ejecucion)
    split = pd.read_csv(ruta_split)
    texto_split = os.path.splitext(os.path.basename(ruta_split))[0].rsplit("_", 2)
    semilla = int(texto_split[2].replace("semilla", ""))

    # --------------------------------------------------------------------------
    # 3.2 VALORES, ESPECIE Y CONJUNTOS
    # X son las columnas de características (después de las columnas de datos
    # del audio) e y es la etiqueta numérica de la especie. El split dice a qué
    # conjunto pertenece cada audio y se une por la ruta del audio.
    # --------------------------------------------------------------------------
    X = tabla.iloc[:, colores.N_COLUMNAS_META:]
    y = tabla.iloc[:, POS_LABEL]
    conjunto_por_audio = dict(zip(split.iloc[:, POS_SPLIT_ARCHIVO], split.iloc[:, POS_SPLIT_SET]))
    conjuntos = tabla.iloc[:, colores.POS_ARCHIVO].map(conjunto_por_audio)
    es_train = (conjuntos == "train").to_numpy()
    es_test = (conjuntos == "test").to_numpy()
    if int(es_train.sum()) + int(es_test.sum()) != len(tabla):
        raise SystemExit(f"Hay audios de la tabla {tipo_tabla} que no están en el split: {ruta_split}")
    X_train = X[es_train]
    y_train = y[es_train]
    X_test = X[es_test]
    y_test = y[es_test]
    especies = sorted(y.unique())
    RUTA_RESULTADOS = os.path.join(CARPETA_SALIDA, f"Clasicos_{tipo_tabla}_{nombre_ejecucion}_{texto_split[1]}_{texto_split[2]}.csv")
    print("=" * 70)
    print(f"Tabla: {tipo_tabla} | Ejecución: {nombre_ejecucion} | Split: {os.path.basename(ruta_split)}")
    print(f"Características: {X.shape[1]} | Clases: {len(especies)} | Train: {len(X_train)} | Test: {len(X_test)} | Semilla: {semilla}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 3.3 CLASIFICADORES
    # Cada clasificador se entrena con train y se mide en train y en test.
    # Cada fila de resultados es: nombre, 4 medidas de train y 4 medidas de test.
    # --------------------------------------------------------------------------
    filas_resultados = []
    predicciones_test = []
    for nombre_clasificador, clasificador in crear_clasificadores(semilla):
        clasificador.fit(X_train, y_train)
        medidas_train = calcular_medidas(clasificador, X_train, y_train, especies)
        medidas_test = calcular_medidas(clasificador, X_test, y_test, especies)
        filas_resultados.append([nombre_clasificador] + medidas_train + medidas_test)
        print(nombre_clasificador)
        print(f"  TRAIN | {texto_medidas(medidas_train)}")
        print(f"  TEST  | {texto_medidas(medidas_test)}")
        predicciones_test.append((nombre_clasificador, clasificador.predict(X_test)))

    # --------------------------------------------------------------------------
    # 3.4 TABLA DE RESULTADOS
    # Columnas: 0 modelo, 1 a 4 medidas de train, 5 a 8 medidas de test.
    # --------------------------------------------------------------------------
    columnas = ["modelo"] + [f"train_{medida}" for medida in MEDIDAS] + [f"test_{medida}" for medida in MEDIDAS]
    pd.DataFrame(filas_resultados, columns=columnas).to_csv(RUTA_RESULTADOS, index=False)
    print(f"Resultados guardados en: {RUTA_RESULTADOS}")

    # --------------------------------------------------------------------------
    # 3.5 PREDICCIONES EN PRUEBA
    # Para cada audio de prueba se guarda su especie real y la especie que
    # predijo cada clasificador (con el nombre de la especie, no con el número).
    # Columnas: 0 archivo, 1 especie real, y una columna por clasificador.
    # --------------------------------------------------------------------------
    especie_por_label = dict(zip(tabla.iloc[:, POS_LABEL], tabla.iloc[:, colores.POS_ESPECIE]))
    predicciones = pd.DataFrame({"archivo": tabla.iloc[:, colores.POS_ARCHIVO][es_test].to_numpy(), "especie_real": y_test.map(especie_por_label).to_numpy()})
    for nombre_clasificador, predichas in predicciones_test:
        predicciones[nombre_clasificador] = pd.Series(predichas).map(especie_por_label).to_numpy()
    RUTA_PREDICCIONES = os.path.join(CARPETA_SALIDA, f"Predicciones_{os.path.basename(RUTA_RESULTADOS)}")
    predicciones.to_csv(RUTA_PREDICCIONES, index=False)
    print(f"Predicciones guardadas en: {RUTA_PREDICCIONES}")
    print("")
    tablas_entrenadas += 1

if tablas_entrenadas == 0:
    raise SystemExit("No hay ninguna tabla de 03_metricas.py. Ejecute primero ese código.")
