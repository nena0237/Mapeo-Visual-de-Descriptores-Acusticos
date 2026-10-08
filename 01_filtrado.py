# ==============================================================================
# CÓDIGO 1 — FILTRADO
# ==============================================================================
# Este código es el filtrado. Es el primero que se utiliza: limpia los datos
# originales (train.csv y taxonomy.csv) y deja listo el CSV.
#
# Qué hace, en orden:
#   1. Carga train.csv y taxonomy.csv.
#   2. Une las dos tablas para que cada audio tenga su clase (Aves, Amphibia...).
#   3. Elimina los audios duplicados usando el identificador del archivo.
#   4. Agrega la etiqueta numérica (label) de cada especie.
#   5. Pregunta por teclado la clase, la duración y la colección.
#   6. Aplica los filtros de clase y de colección.
#   7. Mide la duración de los audios y aplica el filtro de duración.
#   8. Ordena las columnas y guarda el CSV filtrado.
#   9. Guarda un reporte con lo que pasó en cada paso.
#
# Todo se trabaja por POSICIÓN de columna, nunca por nombre de columna.
# Para usar este código en otro proyecto solo se cambia el BLOQUE 0.
# ==============================================================================

import os
import re
import itertools
import unicodedata
import pandas as pd
import soundfile as sf

# ==============================================================================
# BLOQUE 0 — RUTAS Y POSICIONES
# Es el único bloque que se modifica. Aquí están las carpetas del proyecto y el
# número de columna (posición, empezando en 0) donde está cada dato.
# ==============================================================================

# Carpeta principal del proyecto: todas las demás rutas salen de esta.
CARPETA_BASE = r"C:\Users\manue\Downloads\3ccbe"
RUTA_TRAIN = os.path.join(CARPETA_BASE, "DATOS", "train.csv")
RUTA_TAXONOMY = os.path.join(CARPETA_BASE, "DATOS", "taxonomy.csv")
CARPETA_AUDIOS = os.path.join(CARPETA_BASE, "train_audio")
CARPETA_SALIDA = os.path.join(CARPETA_BASE, "salidas", "filtrado")

# Posiciones en train.csv.
POS_TRAIN_CLAVE = 0       # código de la especie, sirve para unir con taxonomy
POS_TRAIN_ARCHIVO = 3     # ruta del audio, ejemplo: 1192948/CSA36373.ogg
POS_TRAIN_COLECCION = 4   # colección de donde viene el audio: XC, iNat o CSA
POS_TRAIN_ESPECIE = 9     # nombre científico de la especie

# Posiciones en taxonomy.csv.
POS_TAXO_CLAVE = 0        # código de la especie, el mismo de train
POS_TAXO_CLASE = 4        # clase taxonómica: Aves, Amphibia, Insecta, Mammalia

# Rangos de duración que aparecen en el menú, en segundos (mínimo, máximo).
DURACIONES_MENU = [(0, 5), (0, 10), (5, 10)]

# Palabras en español que no empiezan igual que el valor del dataset.
# Se guardan las tres primeras letras de la palabra y el valor al que equivale.
SINONIMOS = {"anf": "amphibia", "paj": "aves", "xen": "xc"}

# ==============================================================================
# BLOQUE DE FUNCIONES — MENÚS POR TECLADO
# Las tres preguntas (clase, duración y colección) usan estas mismas funciones,
# por eso el código de los menús está escrito una sola vez.
# ==============================================================================

# Lista donde se va guardando el reporte.
reporte = []


# Muestra un texto en pantalla y lo guarda en el reporte al mismo tiempo.
def anotar(texto):
    print(texto)
    reporte.append(texto)


# Pasa un texto a minúsculas y le quita las tildes, para poder compararlo.
def normalizar(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(letra for letra in texto if unicodedata.category(letra) != "Mn")
    return texto


# Pide un número por teclado y lo repite hasta que sea una opción del menú.
def pedir_numero(cantidad):
    while True:
        respuesta = input("Entrada por teclado: ").strip()
        if respuesta.isdigit() and 1 <= int(respuesta) <= cantidad:
            return int(respuesta)
        print(f"Opción no válida. Escriba un número entre 1 y {cantidad}.")


# Analiza lo que se escribió en la opción "Otra" y devuelve los valores que
# reconoce. Compara las tres primeras letras de cada palabra con las tres
# primeras letras de cada valor disponible. Ejemplo: "aves, insectos y
# mamiferos" devuelve ["Aves", "Insecta", "Mammalia"].
def interpretar_grupos(texto, valores):
    texto_normal = normalizar(texto)
    if "tod" in texto_normal:
        return list(valores)
    palabras = re.split(r"[^a-z0-9]+", texto_normal)
    elegidos = []
    for valor in valores:
        inicio_valor = normalizar(valor)[:3]
        for palabra in palabras:
            palabra = SINONIMOS.get(palabra[:3], palabra)
            if palabra[:3] == inicio_valor and valor not in elegidos:
                elegidos.append(valor)
    return elegidos


# Arma el menú de grupos de forma automática a partir de los valores que
# existen en los datos: cada valor solo (Individuales), todas las parejas
# (Combinados) y la opción "Otra". Devuelve la lista de valores elegidos.
# Sirve igual para las clases y para las colecciones.
def elegir_grupos(titulo, valores, incluir_todas):
    lineas = []
    opciones = []
    if incluir_todas:
        opciones.append(list(valores))
        lineas.append(f"  {len(opciones)}. Todas")
    lineas.append("  Individuales")
    for valor in valores:
        opciones.append([valor])
        lineas.append(f"    {len(opciones)}. {valor}")
    lineas.append("  Combinados")
    for pareja in itertools.combinations(valores, 2):
        opciones.append(list(pareja))
        lineas.append(f"    {len(opciones)}. {' + '.join(pareja)}")
    numero_otra = len(opciones) + 1
    lineas.append(f"    {numero_otra}. Otra")
    print("")
    print(titulo)
    print("\n".join(lineas))
    numero = pedir_numero(numero_otra)
    if numero < numero_otra:
        return opciones[numero - 1]
    while True:
        texto = input(f"Escriba cuáles quiere ({', '.join(valores)}): ")
        elegidos = interpretar_grupos(texto, valores)
        if len(elegidos) > 0:
            return elegidos
        print("No se reconoció ninguna. Intente de nuevo.")


# Analiza lo que se escribió en la opción "Otra" de la duración y devuelve
# (mínimo, máximo) en segundos. Un solo número significa de 0 a ese número.
# Si el texto dice "min" los números se toman como minutos.
# Ejemplos: "3 a 8" devuelve (3, 8), "15" devuelve (0, 15).
def interpretar_duracion(texto):
    texto_normal = normalizar(texto).replace(",", ".")
    numeros = re.findall(r"\d+(?:\.\d+)?", texto_normal)
    valores = [float(numero) for numero in numeros]
    if re.search(r"\bmin(utos?)?\b", texto_normal):
        valores = [valor * 60 for valor in valores]
    if len(valores) == 1 and valores[0] > 0:
        return (0.0, valores[0])
    if len(valores) == 2 and valores[0] != valores[1]:
        return (min(valores), max(valores))
    return None


# Muestra el menú de duración y devuelve el rango elegido (mínimo, máximo).
def elegir_duracion():
    print("")
    print("DURACIÓN")
    numero_linea = 0
    for minimo, maximo in DURACIONES_MENU:
        numero_linea += 1
        print(f"    {numero_linea}. {minimo} seg a {maximo} seg")
    numero_otra = len(DURACIONES_MENU) + 1
    print(f"    {numero_otra}. Otra")
    numero = pedir_numero(numero_otra)
    if numero < numero_otra:
        return DURACIONES_MENU[numero - 1]
    while True:
        texto = input("Escriba la duración en segundos (ejemplo: 3 a 8): ")
        rango = interpretar_duracion(texto)
        if rango is not None:
            return rango
        print("No se reconoció la duración. Intente de nuevo.")


# ==============================================================================
# BLOQUE 1 — CARGA
# Lee los dos archivos originales. Se leen como texto para no alterar ningún
# valor. Se guarda cuántas columnas tiene train porque las columnas nuevas
# (clase, label, duración) se agregan después de esas.
# ==============================================================================
train = pd.read_csv(RUTA_TRAIN, dtype=str)
taxonomy = pd.read_csv(RUTA_TAXONOMY, dtype=str)
columnas_train = train.shape[1]
os.makedirs(CARPETA_SALIDA, exist_ok=True)
anotar("=" * 70)
anotar("REPORTE DE FILTRADO")
anotar("=" * 70)
anotar("1. CARGA")
anotar(f"Registros en train    : {len(train):,}")
anotar(f"Registros en taxonomy : {len(taxonomy):,}")
anotar(f"Especies en train     : {train.iloc[:, POS_TRAIN_ESPECIE].nunique():,}")

# ==============================================================================
# BLOQUE 2 — UNIÓN DE TRAIN Y TAXONOMY
# De taxonomy solo se toman la clave y la clase. Se unen a train por la clave,
# así cada audio queda con su clase en una columna nueva al final (POS_CLASE).
# Los audios que no encuentran clase en taxonomy se eliminan.
# ==============================================================================
taxonomy_clase = taxonomy.iloc[:, [POS_TAXO_CLAVE, POS_TAXO_CLASE]].copy()
taxonomy_clase.columns = [train.columns[POS_TRAIN_CLAVE], taxonomy.columns[POS_TAXO_CLASE]]
taxonomy_clase = taxonomy_clase.drop_duplicates(subset=taxonomy_clase.columns[0])
df = train.merge(taxonomy_clase, on=train.columns[POS_TRAIN_CLAVE], how="left")
POS_CLASE = columnas_train
audios_sin_clase = int(df.iloc[:, POS_CLASE].isna().sum())
df = df[df.iloc[:, POS_CLASE].notna()].copy()
anotar("")
anotar("2. UNIÓN TRAIN + TAXONOMY")
anotar(f"Registros después de la unión : {len(df):,}")
anotar(f"Registros sin clase eliminados: {audios_sin_clase:,}")

# ==============================================================================
# BLOQUE 3 — ELIMINACIÓN DE DUPLICADOS
# La columna del archivo tiene este formato: 1192948/CSA36373.ogg
# El identificador del audio es el nombre sin carpeta y sin extensión: CSA36373
# (las letras de la colección más el número). Ese identificador debe ser único.
# Si se repite, se conserva la primera aparición y se eliminan las demás.
# Se usa letras + número y no solo el número, porque dos colecciones distintas
# pueden tener el mismo número (XC1234 e iNat1234 son audios diferentes).
# ==============================================================================
archivo = df.iloc[:, POS_TRAIN_ARCHIVO].astype(str)
archivo = archivo.str.replace("\\", "/", regex=False)
nombre_archivo = archivo.str.rsplit("/", n=1).str[-1]
identificador = nombre_archivo.str.rsplit(".", n=1).str[0]
es_repetido = identificador.duplicated(keep="first")
identificadores_repetidos = sorted(identificador[es_repetido].unique().tolist())
audios_antes_duplicados = len(df)
df = df[~es_repetido].copy()
anotar("")
anotar("3. DUPLICADOS")
anotar(f"Registros antes               : {audios_antes_duplicados:,}")
anotar(f"Identificadores repetidos     : {len(identificadores_repetidos):,}")
anotar(f"Registros eliminados          : {audios_antes_duplicados - len(df):,}")
anotar(f"Registros después             : {len(df):,}")
anotar(f"Identificadores               : {', '.join(identificadores_repetidos)}")

# ==============================================================================
# BLOQUE 4 — ETIQUETA (LABEL)
# A cada especie se le asigna un número, en orden alfabético y empezando en 0.
# La etiqueta queda en una columna nueva al final (POS_LABEL).
# ==============================================================================
codigos, especies = pd.factorize(df.iloc[:, POS_TRAIN_ESPECIE], sort=True)
df["label"] = codigos
POS_LABEL = df.shape[1] - 1
anotar("")
anotar("4. ETIQUETA")
anotar(f"Especies etiquetadas : {len(especies):,} (label de 0 a {len(especies) - 1})")

# ==============================================================================
# BLOQUE 5 — PREGUNTAS POR TECLADO
# Las opciones de clase y de colección no están escritas a mano: se leen de los
# datos, por eso el menú se adapta solo si cambian los datos.
# Las clases se ordenan por número de especies y las colecciones por número de
# audios, de mayor a menor.
# ==============================================================================
clases_disponibles = taxonomy.iloc[:, POS_TAXO_CLASE].value_counts().index.tolist()
colecciones_disponibles = df.iloc[:, POS_TRAIN_COLECCION].value_counts().index.tolist()
clases_elegidas = elegir_grupos("ESPECIE (clase)", clases_disponibles, False)
duracion_minima, duracion_maxima = elegir_duracion()
colecciones_elegidas = elegir_grupos("CATEGORÍA (colección)", colecciones_disponibles, True)
anotar("")
anotar("5. SELECCIÓN POR TECLADO")
anotar(f"Clase      : {' + '.join(clases_elegidas)}")
anotar(f"Duración   : más de {duracion_minima:g} seg y hasta {duracion_maxima:g} seg")
anotar(f"Colección  : {' + '.join(colecciones_elegidas)}")

# ==============================================================================
# BLOQUE 6 — FILTRO DE CLASE Y DE COLECCIÓN
# Se conservan solo los audios de las clases y de las colecciones elegidas.
# Estos dos filtros van antes que el de duración porque no necesitan abrir los
# audios; así en el bloque siguiente se miden menos archivos. El resultado es
# el mismo sin importar el orden en que se apliquen.
# ==============================================================================
df = df[df.iloc[:, POS_CLASE].isin(clases_elegidas)].copy()
audios_clase = len(df)
df = df[df.iloc[:, POS_TRAIN_COLECCION].isin(colecciones_elegidas)].copy()
audios_coleccion = len(df)
anotar("")
anotar("6. FILTRO DE CLASE Y COLECCIÓN")
anotar(f"Audios de la clase elegida     : {audios_clase:,}")
anotar(f"Audios de la colección elegida : {audios_coleccion:,}")

# ==============================================================================
# BLOQUE 7 — DURACIÓN
# La duración se mide como número de muestras dividido entre la frecuencia de
# muestreo del archivo. Las duraciones ya medidas se guardan en duraciones.csv
# para no volver a abrir los mismos audios en la siguiente ejecución.
# El filtro conserva los audios con duración mayor que el mínimo y menor o
# igual que el máximo. Así "0 a 5" y "5 a 10" no comparten ningún audio.
# ==============================================================================
RUTA_DURACIONES = os.path.join(CARPETA_SALIDA, "duraciones.csv")
duraciones = {}
if os.path.exists(RUTA_DURACIONES):
    tabla_duraciones = pd.read_csv(RUTA_DURACIONES)
    duraciones = dict(zip(tabla_duraciones.iloc[:, 0], tabla_duraciones.iloc[:, 1]))
audios_no_encontrados = 0
audios_con_error = 0
audios_medidos = 0
lista_duraciones = []
print("")
print("Midiendo la duración de los audios...")
for archivo_audio in df.iloc[:, POS_TRAIN_ARCHIVO]:
    if archivo_audio not in duraciones:
        ruta_audio = os.path.join(CARPETA_AUDIOS, str(archivo_audio).replace("/", os.sep))
        if not os.path.exists(ruta_audio):
            audios_no_encontrados += 1
            lista_duraciones.append(None)
            continue
        try:
            informacion = sf.info(ruta_audio)
        except Exception:
            audios_con_error += 1
            lista_duraciones.append(None)
            continue
        duraciones[archivo_audio] = informacion.frames / informacion.samplerate
        audios_medidos += 1
        if audios_medidos % 2000 == 0:
            print(f"  audios medidos: {audios_medidos:,}")
    lista_duraciones.append(duraciones[archivo_audio])
tabla_duraciones = pd.DataFrame({"archivo": list(duraciones.keys()), "duracion_seg": list(duraciones.values())})
tabla_duraciones.to_csv(RUTA_DURACIONES, index=False)
df["duracion_seg"] = lista_duraciones
POS_DURACION = df.shape[1] - 1
df = df[df.iloc[:, POS_DURACION].notna()].copy()
duracion = df.iloc[:, POS_DURACION].astype(float)
df = df[(duracion > duracion_minima) & (duracion <= duracion_maxima)].copy()
df.iloc[:, POS_DURACION] = df.iloc[:, POS_DURACION].astype(float).round(4)
anotar("")
anotar("7. FILTRO DE DURACIÓN")
anotar(f"Audios medidos en esta ejecución : {audios_medidos:,}")
anotar(f"Audios no encontrados            : {audios_no_encontrados:,}")
anotar(f"Audios con error de lectura      : {audios_con_error:,}")
anotar(f"Audios dentro de la duración     : {len(df):,}")

# ==============================================================================
# BLOQUE 8 — ORDEN DE COLUMNAS Y GUARDADO
# Las cinco primeras columnas del CSV final son siempre, en este orden:
# archivo, especie, label, clase y duración. Después van las demás columnas
# originales de train. El nombre del archivo se arma con lo que se eligió.
# ==============================================================================
if len(df) == 0:
    raise SystemExit("El filtrado no dejó ningún audio. Revise la clase, la duración y la colección.")
primeras = [POS_TRAIN_ARCHIVO, POS_TRAIN_ESPECIE, POS_LABEL, POS_CLASE, POS_DURACION]
restantes = [posicion for posicion in range(df.shape[1]) if posicion not in primeras]
df = df.iloc[:, primeras + restantes]
texto_clases = "+".join(clases_elegidas)
texto_duracion = f"{duracion_minima:g}-{duracion_maxima:g}s"
texto_colecciones = "+".join(colecciones_elegidas)
if len(colecciones_elegidas) == len(colecciones_disponibles):
    texto_colecciones = "Todas"
nombre_salida = f"{texto_clases}_{texto_duracion}_{texto_colecciones}"
RUTA_CSV = os.path.join(CARPETA_SALIDA, f"Filtrado_{nombre_salida}.csv")
RUTA_REPORTE = os.path.join(CARPETA_SALIDA, f"Reporte_Filtrado_{nombre_salida}.txt")
df.to_csv(RUTA_CSV, index=False)

# ==============================================================================
# BLOQUE 9 — REPORTE
# Resume el dataset final, lista cuántos audios quedaron por especie y guarda
# todo lo anotado en un archivo de texto.
# En el CSV final la especie está en la posición 1 y la duración en la 4.
# ==============================================================================
conteo_especies = df.iloc[:, 1].value_counts()
duracion_final = df.iloc[:, 4].astype(float)
anotar("")
anotar("8. DATASET FINAL")
anotar(f"Audios   : {len(df):,}")
anotar(f"Especies : {len(conteo_especies):,}")
anotar(f"Duración promedio : {duracion_final.mean():.4f} seg")
anotar(f"Duración mínima   : {duracion_final.min():.4f} seg")
anotar(f"Duración máxima   : {duracion_final.max():.4f} seg")
anotar(f"CSV generado      : {RUTA_CSV}")
anotar(f"Reporte generado  : {RUTA_REPORTE}")
reporte.append("")
reporte.append("9. AUDIOS POR ESPECIE")
for especie, cantidad in conteo_especies.items():
    reporte.append(f"{especie:<45}{cantidad:,}")
with open(RUTA_REPORTE, "w", encoding="utf-8") as archivo_reporte:
    archivo_reporte.write("\n".join(reporte))
