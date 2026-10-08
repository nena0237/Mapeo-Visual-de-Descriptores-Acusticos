# ==============================================================================
# CÓDIGO 3 — IMPORTANCIAS + SPLIT
# ==============================================================================
# Recibe las tablas normalizadas que entregó la extracción (CÓDIGO 2: caracteristicas y índices)
# y hace lo mismo con cada una: la tabla de características y la de índices.
#
# Qué hace, en orden:
#   1. Pregunta por teclado el método de importancia, la división del split y
#      la semilla. Con esas tres respuestas se ejecuta todo lo demás.
#   2. Busca las tablas de la extracción.
#   3. Calcula la importancia de cada característica con el método elegido.
#   4. Normaliza las importancias y selecciona las características:
#        - Se eliminan las características con importancia = 0.
#        - Si el total no es múltiplo de 3, se eliminan las de menor importancia
#          hasta llegar al múltiplo de 3 inmediatamente inferior.
#        - Se renormalizan las importancias restantes para que sumen 1.
#        - Se conserva el orden original y se crea un índice consecutivo nuevo.
#   5. Guarda la tabla de importancias y la tabla normalizada con solo las
#      características seleccionadas.
#   6. Crea el split único estratificado, compartido por todas las líneas de
#      clasificación, y lo guarda.
#   7. Guarda un reporte con lo que pasó en cada paso.
#
# IMPORTANTE:
# - La importancia se calcula con todos los audios de la extracción y después
#   se hace el split.
# - El múltiplo de 3 es necesario porque la plantilla agrupa las
#   características en filas de 3 bloques.
# - Las importancias se guardan en el mismo orden de las columnas de la tabla,
#   porque la plantilla ubica cada bloque según ese orden.
# - No se modifican las tablas de la extracción: las tablas con las
#   características seleccionadas se guardan como archivos nuevos.
# - Todo se trabaja por POSICIÓN de columna, nunca por nombre de columna.
# ==============================================================================

import os
import re
import glob
import numpy as np
import pandas as pd
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

# ==============================================================================
# BLOQUE 0 — RUTAS, POSICIONES Y PARÁMETROS
# Es el único bloque que se modifica.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
CARPETA_CARACTERISTICAS = os.path.join(CARPETA_BASE, "salidas", "caracteristicas")
CARPETA_INDICES = os.path.join(CARPETA_BASE, "salidas", "indices")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "importancias")

# Dataset que se va a leer. Si se deja vacío, el código toma solo el más
# reciente que entregó la extracción. Para fijar uno se escribe su nombre,
# ejemplo: "Aves_0-5s_XC"
NOMBRE_DATASET = ""

# Posiciones en las tablas de la extracción (la extracción siempre las entrega así).
POS_ARCHIVO = 0       # ruta del audio dentro de la carpeta de audios
POS_ESPECIE = 1       # nombre científico de la especie
POS_LABEL = 2         # etiqueta numérica de la especie
N_COLUMNAS_META = 4   # las características empiezan después de estas columnas

# Parámetros de la importancia.
N_NEIGHBORS = 100       # vecinos de ReliefF
MULTIPLO_OBJETIVO = 3   # la plantilla usa filas de 3 bloques
N_ARBOLES = 500         # árboles de Random Forest y Extra Trees
SEMILLA_METODOS = 42    # semilla de los métodos que tienen azar (no es la del split)

# Opciones de los menús. El texto es lo que se muestra y el valor lo que se usa.
SPLITS_MENU = [("60/40", (60, 40)), ("70/30 (recomendado)", (70, 30)), ("80/20", (80, 20))]
SEMILLAS_MENU = [("1024 (recomendada)", 1024), ("42", 42)]

# ==============================================================================
# BLOQUE DE FUNCIONES — MÉTODOS DE IMPORTANCIA
# Todos los métodos funcionan igual que ReliefF: reciben la matriz X (una fila
# por audio y una columna por característica) y el vector y (la especie de cada
# audio), y devuelven UN número de importancia por característica.
# Un método que no devuelva eso no aplica para este código.
# ==============================================================================


# ReliefF: compara cada audio con sus vecinos más parecidos. Una característica
# es importante si separa a los audios vecinos que son de especies distintas.
def importancia_relieff(X, y):
    vecinos = min(N_NEIGHBORS, len(X) - 1)
    relieff = ReliefF(n_neighbors=vecinos)
    relieff.fit(X, y)
    return relieff.feature_importances_


# Random Forest: entrena muchos árboles de decisión. Una característica es
# importante si los árboles la usan mucho para separar las especies.
def importancia_random_forest(X, y):
    modelo = RandomForestClassifier(n_estimators=N_ARBOLES, random_state=SEMILLA_METODOS, n_jobs=-1)
    modelo.fit(X, y)
    return modelo.feature_importances_


# Extra Trees: igual que Random Forest, pero los cortes de cada árbol se eligen
# al azar, por eso es más rápido.
def importancia_extra_trees(X, y):
    modelo = ExtraTreesClassifier(n_estimators=N_ARBOLES, random_state=SEMILLA_METODOS, n_jobs=-1)
    modelo.fit(X, y)
    return modelo.feature_importances_


# Información mutua: mide cuánta información da la característica sobre la
# especie. Vale 0 cuando la característica no dice nada de la especie.
def importancia_informacion_mutua(X, y):
    return mutual_info_classif(X, y, random_state=SEMILLA_METODOS)


# ANOVA F: compara cuánto cambia la característica entre especies frente a
# cuánto cambia dentro de una misma especie.
def importancia_anova(X, y):
    valores_f, valores_p = f_classif(X, y)
    return valores_f


# Método principal (ReliefF) y lista cerrada de otros métodos de scikit-learn.
# Cada uno tiene: el texto del menú, el nombre para los archivos y su función.
# Para agregar un método nuevo se escribe su función arriba y una línea aquí.
METODO_ESTUDIO = ("ReliefF (aplicado en el estudio)", "ReliefF", importancia_relieff)
OTROS_METODOS = [("Random Forest", "RandomForest", importancia_random_forest), ("Extra Trees", "ExtraTrees", importancia_extra_trees), ("Información mutua", "InformacionMutua", importancia_informacion_mutua), ("ANOVA F", "AnovaF", importancia_anova)]

# ==============================================================================
# BLOQUE DE FUNCIONES — MENÚS POR TECLADO Y SELECCIÓN
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


# Muestra un menú numerado con los textos que recibe y devuelve el número elegido.
def elegir_opcion(titulo, textos):
    print("")
    print(titulo)
    numero_linea = 0
    for texto in textos:
        numero_linea += 1
        print(f"    {numero_linea}. {texto}")
    return pedir_numero(len(textos))


# Pregunta el método de importancia. La opción "Otro" abre la lista cerrada de
# métodos de scikit-learn. Devuelve el método elegido.
def elegir_metodo():
    numero = elegir_opcion("MÉTODO DE IMPORTANCIA", [METODO_ESTUDIO[0], "Otro"])
    if numero == 1:
        return METODO_ESTUDIO
    textos_otros = [metodo[0] for metodo in OTROS_METODOS]
    numero = elegir_opcion("OTROS MÉTODOS (scikit-learn)", textos_otros)
    return OTROS_METODOS[numero - 1]


# Analiza lo que se escribió en la opción "Otro" del split y devuelve los
# porcentajes (entrenamiento, prueba). Acepta dos números que sumen 100 o un
# solo número, que se toma como el porcentaje de entrenamiento.
# Ejemplos: "75/25" devuelve (75, 25), "85" devuelve (85, 15).
def interpretar_split(texto):
    numeros = re.findall(r"\d+(?:\.\d+)?", texto.replace(",", "."))
    valores = [float(numero) for numero in numeros]
    if len(valores) == 2 and valores[0] + valores[1] == 100 and valores[0] > 0 and valores[1] > 0:
        return (valores[0], valores[1])
    if len(valores) == 1 and 0 < valores[0] < 100:
        return (valores[0], 100 - valores[0])
    return None


# Pregunta la división del split y devuelve (entrenamiento, prueba) en porcentaje.
def elegir_split():
    textos = [opcion[0] for opcion in SPLITS_MENU] + ["Otro"]
    numero = elegir_opcion("DIVISIÓN DEL SPLIT (entrenamiento/prueba)", textos)
    if numero <= len(SPLITS_MENU):
        return SPLITS_MENU[numero - 1][1]
    while True:
        texto = input("Escriba la división (ejemplo: 75/25): ")
        division = interpretar_split(texto)
        if division is not None:
            return division
        print("No se reconoció la división. Los dos números deben sumar 100.")


# Pregunta la semilla del split y devuelve un número entero.
def elegir_semilla():
    textos = [opcion[0] for opcion in SEMILLAS_MENU] + ["Otra"]
    numero = elegir_opcion("SEMILLA A TRABAJAR", textos)
    if numero <= len(SEMILLAS_MENU):
        return SEMILLAS_MENU[numero - 1][1]
    while True:
        texto = input("Escriba la semilla (un número entero): ").strip()
        if texto.isdigit():
            return int(texto)
        print("La semilla debe ser un número entero.")


# Recibe los nombres de las características y sus importancias brutas, y aplica
# la selección. Devuelve tres tablas: las características
# seleccionadas, las eliminadas por importancia = 0 y las eliminadas para
# llegar al múltiplo de 3. En las tres tablas las columnas son:
# posición 0 = lugar de la característica en la tabla original, posición 1 =
# nombre, posición 2 = importancia normalizada.
def seleccionar_caracteristicas(nombres, importancias):
    importancias = np.asarray(importancias, dtype=float)
    importancias = np.nan_to_num(importancias, nan=0.0)
    # NORMALIZAR SOLO LAS IMPORTANCIAS: se resta el mínimo y se divide por la suma.
    importancias = importancias - importancias.min()
    if importancias.sum() != 0:
        importancias = importancias / importancias.sum()
    tabla = pd.DataFrame({"indice": range(len(nombres)), "caracteristica": nombres, "importancia_normalizada": importancias})
    # 1. ELIMINAR TODAS LAS CARACTERÍSTICAS CON IMPORTANCIA = 0
    eliminadas_cero = tabla[tabla.iloc[:, 2] == 0].copy()
    seleccionadas = tabla[tabla.iloc[:, 2] > 0].copy()
    # 2. SI EL TOTAL NO ES MÚLTIPLO DE 3, ELIMINAR LAS DE MENOR IMPORTANCIA
    #    HASTA LLEGAR AL MÚLTIPLO DE 3 INMEDIATAMENTE INFERIOR
    eliminadas_ajuste = seleccionadas.iloc[0:0].copy()
    resto = len(seleccionadas) % MULTIPLO_OBJETIVO
    if resto != 0:
        filas_eliminar = seleccionadas.iloc[:, 2].nsmallest(resto).index
        eliminadas_ajuste = seleccionadas.loc[filas_eliminar].copy()
        seleccionadas = seleccionadas.drop(index=filas_eliminar).copy()
    # 3. RENORMALIZAR LAS IMPORTANCIAS RESTANTES PARA QUE VUELVAN A SUMAR 1
    suma_restante = seleccionadas.iloc[:, 2].sum()
    if suma_restante != 0:
        seleccionadas.iloc[:, 2] = seleccionadas.iloc[:, 2] / suma_restante
    # 4. CONSERVAR EL ORDEN ORIGINAL
    seleccionadas = seleccionadas.sort_values(seleccionadas.columns[0]).reset_index(drop=True)
    return seleccionadas, eliminadas_cero, eliminadas_ajuste


# ==============================================================================
# BLOQUE 1 — PREGUNTAS POR TECLADO
# Primero el método de importancia, después la división del split y por último
# la semilla. Cuando están las tres respuestas se ejecuta el resto del código.
# ==============================================================================
texto_metodo, nombre_metodo, funcion_metodo = elegir_metodo()
porcentaje_train, porcentaje_test = elegir_split()
semilla = elegir_semilla()
texto_split = f"{porcentaje_train:g}-{porcentaje_test:g}"

# ==============================================================================
# BLOQUE 2 — BÚSQUEDA DE LAS TABLAS DE LA EXTRACCIÓN
# Si no se fijó un dataset en el BLOQUE 0, se toma el de la tabla más reciente
# que entregó la extracción. Después se arma la lista de tablas que existen
# para ese dataset: la de características, la de índices o las dos.
# El nombre con el que se guardan los archivos es el del dataset.
# ==============================================================================
if NOMBRE_DATASET == "":
    archivos_extraccion = glob.glob(os.path.join(CARPETA_CARACTERISTICAS, "Caracteristicas_*.csv")) + glob.glob(os.path.join(CARPETA_INDICES, "Indices_*.csv"))
    if len(archivos_extraccion) == 0:
        raise SystemExit("No hay ninguna tabla de la extracción. Ejecute primero los códigos de extracción (02).")
    archivo_reciente = max(archivos_extraccion, key=os.path.getmtime)
    nombre_reciente = os.path.splitext(os.path.basename(archivo_reciente))[0]
    NOMBRE_DATASET = nombre_reciente.split("_", 1)[1]
tablas = []
ruta_caracteristicas = os.path.join(CARPETA_CARACTERISTICAS, f"Caracteristicas_{NOMBRE_DATASET}.csv")
ruta_indices = os.path.join(CARPETA_INDICES, f"Indices_{NOMBRE_DATASET}.csv")
if os.path.exists(ruta_caracteristicas):
    tablas.append(("Caracteristicas", ruta_caracteristicas))
if os.path.exists(ruta_indices):
    tablas.append(("Indices", ruta_indices))
if len(tablas) == 0:
    raise SystemExit(f"No se encontró ninguna tabla de la extracción para el dataset: {NOMBRE_DATASET}")
os.makedirs(CARPETA_SALIDA, exist_ok=True)
RUTA_SPLIT = os.path.join(CARPETA_SALIDA, f"Split_{NOMBRE_DATASET}_{texto_split}_semilla{semilla}.csv")
RUTA_REPORTE = os.path.join(CARPETA_SALIDA, f"Reporte_Importancias_{NOMBRE_DATASET}_{nombre_metodo}_{texto_split}_semilla{semilla}.txt")
anotar("")
anotar("=" * 80)
anotar("REPORTE DE IMPORTANCIAS Y SPLIT")
anotar("=" * 80)
anotar("1. SELECCIÓN POR TECLADO")
anotar(f"Dataset                : {NOMBRE_DATASET}")
anotar(f"Método de importancia  : {texto_metodo}")
anotar(f"División del split     : {porcentaje_train:g}% entrenamiento / {porcentaje_test:g}% prueba")
anotar(f"Semilla                : {semilla}")

# ==============================================================================
# BLOQUE 3 — IMPORTANCIA Y SELECCIÓN DE CARACTERÍSTICAS
# Se repite lo mismo para cada tabla de la extracción:
#   - Se lee la tabla.
#   - X son las columnas de características (desde la posición N_COLUMNAS_META).
#   - y es la especie de cada audio. Las especies se numeran de 0 a N-1 en orden
#     alfabético, para que el resultado no dependa de los números que traiga la
#     columna label.
#   - Se calcula la importancia con el método elegido.
#   - Se seleccionan las características con la función de selección.
#   - Se guarda la tabla de importancias y la tabla normalizada que conserva las
#     columnas meta y solo las características seleccionadas.
# ==============================================================================
audios_por_tabla = []
tabla_base = None
numero_seccion = 1
for tipo_tabla, ruta_tabla in tablas:
    numero_seccion += 1
    df = pd.read_csv(ruta_tabla)
    nombres = list(df.columns[N_COLUMNAS_META:])
    X = df.iloc[:, N_COLUMNAS_META:].to_numpy(dtype=float)
    y, especies = pd.factorize(df.iloc[:, POS_ESPECIE], sort=True)
    print("")
    print(f"Calculando importancia de {tipo_tabla} con {nombre_metodo}...")
    importancias = funcion_metodo(X, y)
    seleccionadas, eliminadas_cero, eliminadas_ajuste = seleccionar_caracteristicas(nombres, importancias)
    if len(seleccionadas) == 0:
        raise SystemExit(f"No quedó ninguna característica seleccionada en la tabla de {tipo_tabla}.")
    posiciones_seleccionadas = seleccionadas.iloc[:, 0].tolist()
    seleccionadas.iloc[:, 0] = range(len(seleccionadas))
    columnas_finales = list(range(N_COLUMNAS_META)) + [N_COLUMNAS_META + posicion for posicion in posiciones_seleccionadas]
    df_seleccion = df.iloc[:, columnas_finales]
    RUTA_IMPORTANCIA = os.path.join(CARPETA_SALIDA, f"Importancia_{tipo_tabla}_{NOMBRE_DATASET}_{nombre_metodo}.csv")
    RUTA_TABLA = os.path.join(CARPETA_SALIDA, f"{tipo_tabla}_{NOMBRE_DATASET}_{nombre_metodo}.csv")
    seleccionadas.to_csv(RUTA_IMPORTANCIA, index=False)
    df_seleccion.to_csv(RUTA_TABLA, index=False)
    audios_por_tabla.append(set(df.iloc[:, POS_ARCHIVO]))
    if tabla_base is None:
        tabla_base = df.iloc[:, [POS_ARCHIVO, POS_ESPECIE, POS_LABEL]].copy()
    anotar("")
    anotar(f"{numero_seccion}. IMPORTANCIA DE {tipo_tabla.upper()}")
    anotar(f"Tabla de entrada                  : {ruta_tabla}")
    anotar(f"Audios                            : {len(df):,}")
    anotar(f"Especies                          : {len(especies):,}")
    anotar(f"Características originales        : {len(nombres)}")
    anotar(f"Eliminadas por importancia = 0    : {len(eliminadas_cero)}")
    for fila in eliminadas_cero.itertuples(index=False, name=None):
        anotar(f"  - {fila[1]}: {fila[2]:.12f}")
    anotar(f"Eliminadas para múltiplo de {MULTIPLO_OBJETIVO}     : {len(eliminadas_ajuste)}")
    for fila in eliminadas_ajuste.sort_values(eliminadas_ajuste.columns[2]).itertuples(index=False, name=None):
        anotar(f"  - {fila[1]}: {fila[2]:.12f}")
    anotar(f"Características finales           : {len(seleccionadas)}")
    anotar(f"Suma de importancias finales      : {seleccionadas.iloc[:, 2].sum():.12f}")
    anotar(f"Tabla de importancias             : {RUTA_IMPORTANCIA}")
    anotar(f"Tabla normalizada seleccionada    : {RUTA_TABLA}")
    reporte.append("Importancia final de cada característica:")
    for fila in seleccionadas.itertuples(index=False, name=None):
        reporte.append(f"  [{fila[0]}] {fila[1]:<22} {fila[2]:.12f}")

# ==============================================================================
# BLOQUE 4 — SPLIT ÚNICO ESTRATIFICADO
# Crea el split único que comparten todas las líneas de clasificación, para que
# se entrenen y se evalúen exactamente con los mismos audios.
# Estratificado significa que cada especie queda repartida en la misma
# proporción entre entrenamiento y prueba.
# Solo entran los audios que están en todas las tablas: si un audio falló en
# una de las extracciones, se quita del split.
# Cada especie necesita al menos 2 audios, uno para cada conjunto. Si no se
# puede hacer el split, el código se detiene y dice por qué.
# ==============================================================================
audios_comunes = set.intersection(*audios_por_tabla)
split = tabla_base[tabla_base.iloc[:, 0].isin(audios_comunes)].reset_index(drop=True)
posiciones = np.arange(len(split))
try:
    posiciones_train, posiciones_test = train_test_split(posiciones, test_size=porcentaje_test / 100, random_state=semilla, stratify=split.iloc[:, 1])
except ValueError as error_split:
    raise SystemExit(f"No se pudo hacer el split con esa división: {error_split}")
split["set"] = "train"
split.iloc[posiciones_test, 3] = "test"
split.to_csv(RUTA_SPLIT, index=False)
es_test = split.iloc[:, 3] == "test"
numero_seccion += 1
anotar("")
anotar(f"{numero_seccion}. SPLIT")
anotar(f"Audios en todas las tablas : {len(split):,}")
anotar(f"Entrenamiento (train)      : {int((~es_test).sum()):,}")
anotar(f"Prueba (test)              : {int(es_test.sum()):,}")
anotar(f"Especies en entrenamiento  : {split[~es_test].iloc[:, 1].nunique():,}")
anotar(f"Especies en prueba         : {split[es_test].iloc[:, 1].nunique():,}")
anotar("Columnas del split         : archivo | especie | label | set")
anotar(f"Split generado             : {RUTA_SPLIT}")
anotar(f"Reporte generado           : {RUTA_REPORTE}")

# ==============================================================================
# BLOQUE 5 — REPORTE
# Guarda en un archivo de texto todo lo anotado durante la ejecución.
# ==============================================================================
with open(RUTA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
    archivo_reporte.write("\n".join(reporte))
