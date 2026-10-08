# ==============================================================================
# CÓDIGO 6 — DATALOADER FINAL
# ==============================================================================
# Este es el DataLoader final que se utiliza para entrenar los modelos.
# Contiene la clase PlantillaDataset, que le entrega a la red, para cada audio,
# su imagen y su etiqueta.
#
# Todo se hace en caliente, en memoria: las imágenes no se
# cargan de ningún archivo de imágenes. Se pintan en el momento en que se crea
# el dataset y solo existen mientras el código está corriendo.
#
# Conecta con los dos códigos anteriores:
#   - De la PLANTILLA usa el CSV con los bloques (dónde va cada característica).
#   - De COLORES usa sus funciones para pintar las imágenes en memoria.
#
# LO QUE ENTREGA:
#   - imagen   : tensor de forma (1, alto, ancho), tipo float32, valores de 0 a 1.
#                Es la imagen pintada píxel por píxel.
#   - etiqueta : tensor de tipo long con el número de la especie.
#   - len(dataset) es el número de audios y dataset.n_classes el de especies.
#
# CÓMO LO HACE:
#   - La geometría no se calcula aquí: la entrega la plantilla.
#   - El pintado no está escrito aquí: se usa el de colores, para que la regla
#     del color esté escrita una sola vez.
#   - La etiqueta va de 0 a N-1, con las especies en orden alfabético. No
#     depende de los números que traiga la columna label del filtrado.
#
# CÓMO SE USA desde el código de entrenamiento:
#   import importlib
#   data_loader = importlib.import_module("06_Data_Loader")
#   dataset = data_loader.PlantillaDataset("Caracteristicas")     o     ("Indices")
#   imagen, etiqueta = dataset[0]
#
# IMPORTANTE:
# - Como el nombre de este archivo empieza por un número, los otros códigos lo
#   importan con importlib, como se muestra arriba. La forma "from ... import"
#   no acepta nombres que empiecen por un número.
# - Las rutas del proyecto no se escriben aquí: se toman del código de colores.
# - Si se ejecuta directamente, hace una prueba: crea el dataset de cada tabla
#   que encuentre y muestra su resumen.
# ==============================================================================

import importlib
import pandas as pd
import torch
from torch.utils.data import Dataset

# ==============================================================================
# BLOQUE 0 — CONEXIÓN CON EL CÓDIGO DE COLORES
# Es el único bloque que se modifica.
# ==============================================================================

# Nombre del archivo del código de colores, sin el ".py". Debe estar en la
# misma carpeta que este archivo. Si se le cambia el nombre al archivo de
# colores, se cambia aquí.
ARCHIVO_COLORES = "05_colores"

# Se importa el código de colores. Se usa importlib porque el nombre del
# archivo empieza por un número. Al importarlo solo quedan disponibles sus
# funciones y sus posiciones; no se ejecuta el pintado de muestras.
colores = importlib.import_module(ARCHIVO_COLORES)


# ==============================================================================
# CLASE PlantillaDataset
# ==============================================================================
class PlantillaDataset(Dataset):

    # Crea el dataset de una tabla y pinta sus imágenes en memoria.
    #   tipo_tabla       : "Caracteristicas" o "Indices".
    #   nombre_plantilla : si se deja vacío, se toma sola la plantilla más
    #                      reciente de esa tabla. Para fijar una se escribe el
    #                      dataset, el método y el tamaño,
    #                      ejemplo: "Aves_0-5s_XC_ReliefF_128px"
    def __init__(self, tipo_tabla, nombre_plantilla=""):
        # Plantilla (bloques) y tabla normalizada (valores de cada audio).
        if nombre_plantilla == "":
            nombre_plantilla = colores.buscar_plantilla_reciente(tipo_tabla)
        self.tipo_tabla = tipo_tabla
        self.nombre_plantilla = nombre_plantilla
        self.plantilla, self.df, self.TAMANO_PIXELES = colores.cargar_plantilla_y_tabla(tipo_tabla, nombre_plantilla)

        # Datos de cada audio, por posición en la tabla normalizada.
        self.archivos = self.df.iloc[:, colores.POS_ARCHIVO].to_numpy()
        self.especies = self.df.iloc[:, colores.POS_ESPECIE].to_numpy()

        # Información de las clases. Las especies se ordenan alfabéticamente y
        # a cada una se le asigna un número de 0 a N-1: esa es la etiqueta.
        codigos, clases = pd.factorize(self.especies, sort=True)
        self.classes = clases
        self.n_classes = len(clases)

        # Pintado en memoria, píxel por píxel, con las funciones de colores.
        imagenes = colores.pintar_imagenes(self.df, self.plantilla, self.TAMANO_PIXELES)

        # Las imágenes y las etiquetas se dejan listas como tensores.
        # A cada imagen se le agrega la dimensión del canal: (1, alto, ancho).
        self.imagenes = torch.from_numpy(imagenes).unsqueeze(1).float()
        self.etiquetas = torch.tensor(codigos, dtype=torch.long)

        print("=" * 60)
        print(f"Tabla : {self.tipo_tabla}")
        print(f"Tamaño imagen : {self.TAMANO_PIXELES}x{self.TAMANO_PIXELES}")
        print(f"Total registros : {len(self.df)}")
        print(f"Número de clases : {self.n_classes}")
        print(f"Total rectángulos : {len(self.plantilla)}")
        print("Imágenes creadas en memoria (generación pixel a pixel).")
        print("=" * 60)

    # Número de audios del dataset.
    def __len__(self):
        return len(self.imagenes)

    # Entrega la imagen y la etiqueta del audio que está en la posición index.
    def __getitem__(self, index):
        imagen = self.imagenes[index]
        etiqueta = self.etiquetas[index]
        return imagen, etiqueta


# ==============================================================================
# PRUEBA
# Solo se ejecuta cuando este archivo se corre directamente, no cuando otro
# código lo importa. Crea el dataset de cada tabla que tenga plantilla y
# muestra la forma de la primera imagen y su etiqueta.
# ==============================================================================
if __name__ == "__main__":
    import os
    import glob
    for tipo_prueba in ["Caracteristicas", "Indices"]:
        if len(glob.glob(os.path.join(colores.CARPETA_PLANTILLA, f"Plantilla_{tipo_prueba}_*.csv"))) == 0:
            continue
        dataset = PlantillaDataset(tipo_prueba)
        imagen, etiqueta = dataset[0]
        print(f"Plantilla        : {dataset.nombre_plantilla}")
        print(f"Primera imagen   : forma {tuple(imagen.shape)}, tipo {imagen.dtype}")
        print(f"Primera etiqueta : {int(etiqueta)} ({dataset.classes[int(etiqueta)]})")
        print(f"Primer audio     : {dataset.archivos[0]}")
        print("")
