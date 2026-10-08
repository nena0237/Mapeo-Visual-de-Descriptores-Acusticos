# ==============================================================================
# CÓDIGO 9 — LÍNEA AVANZADA (ESPECTROGRAMAS + EFFICIENTNET-B0)
# ==============================================================================
# Este código es la LÍNEA BASE ORIGINAL de la competencia BirdCLEF 2025 de
# Kaggle (EfficientNet-B0 con mel-espectrogramas), usada en el estudio como
# línea avanzada para comparar contra la línea clásica y la plantilla.


# PAGINA OFICIAL DEL EVENTO
# https://www.kaggle.com/competitions/birdclef-2025

# CODIGO OFICIAL
# https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25

# CODIGO OFICIAL MEJORADO DESDE LA MISMA COMPETENCIA
# https://www.kaggle.com/code/nguyenquangthinhus/efficientnet-b0-pytorch-train-birdclef-25


# No se cambió la arquitectura, la función de pérdida, el scheduler ni la
# lógica de entrenamiento. Solo se adaptó para correr con los archivos del
# proyecto en el computador local, en lugar de las carpetas de Kaggle:
#   1. Rutas (OUTPUT_DIR, train_datadir, train_csv, taxonomy_csv).
#   2. train_csv es el split que entregó 03_metricas.py. Solo dice qué audios
#      y qué especies se usan; el modelo recibe el mel-espectrograma del audio
#      real, no las características.
#   3. taxonomy_csv es la taxonomy.csv original completa (206 especies).
#   4. LOAD_DATA = False: los espectrogramas se generan desde los audios en
#      lugar de cargar el .npy de 7,49 GB (el código original ya trae esa opción).
#   5. num_workers = 0 (Windows).
#   6. Se crean las columnas filepath y filename a partir de la ruta del audio
#      que trae el split.
#   7. El nombre científico se traduce al código de taxonomy.csv (primary_label);
#      sin esto todas las etiquetas quedan en cero.
#   8. En lugar del K-fold se usa el split único estratificado que entregó
#      03_metricas.py, el mismo de la línea clásica y de la plantilla.
#      La semilla no se escribe aquí: es la misma del split y se lee del nombre
#      del archivo (Split_<dataset>_<división>_semilla<n>.csv).
#   9. Se muestran las mismas medidas de las otras líneas: AUC, accuracy,
#      recall macro y F1 macro, en train y en test. El AUC es el que ya
#      calculaba el código original. El resultado que se reporta es el de la
#      última época.
#  10. Se guardan en salidas/espectrogramas las medidas por época (con la
#      pérdida) y las predicciones de la última época, para 10_graficos.py.
#
# Con el split 70/30 y semilla 42 da 13,04 % de accuracy en test (resultado
# del congreso).
# ==============================================================================

import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import timm
import sys
import glob
import importlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


# CAMBIO: medidas que se reportan en las tres líneas (accuracy, recall macro y
# F1 macro). Se calculan comparando la especie real con la que predijo la red.
# Se le pasan las especies del conjunto para que todas pesen lo mismo.
def calc_metricas(y_true, y_pred, labels):
    exactitud = accuracy_score(y_true, y_pred)
    _, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"acc": 100 * exactitud, "recall_m": 100 * recall_m, "f1_m": f1_m}


# CAMBIO: convierte el AUC y las 3 medidas en el texto que se muestra en pantalla.
def fmt(auc, medidas):
    return f"AUC {auc:.4f} | Acc {medidas['acc']:.2f}% | Recall(m) {medidas['recall_m']:.2f}% | F1(m) {medidas['f1_m']:.4f}"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ==============================================================================
# CAMBIO: CONEXIÓN CON 03_metricas.py
# El split y la semilla se eligieron en 03_metricas.py, así que aquí no se
# escriben. Se toma el split más reciente (o el que se fije en NOMBRE_SPLIT) y
# la semilla se lee de su nombre. La carpeta del split sale del código de colores.
# ==============================================================================
colores = importlib.import_module("05_colores")

# Split que se va a usar. Si se deja vacío, se toma el más reciente. Para fijar
# uno se escribe su nombre sin ".csv", ejemplo: "Split_Aves_0-5s_XC_70-30_semilla42"
NOMBRE_SPLIT = ""

# Posiciones en el CSV del split de 03_metricas.py.
POS_SPLIT_ARCHIVO = 0   # ruta del audio dentro de la carpeta de audios
POS_SPLIT_ESPECIE = 1   # nombre científico de la especie
POS_SPLIT_SET = 3       # conjunto del audio: "train" o "test"

if NOMBRE_SPLIT == "":
    archivos_split = glob.glob(os.path.join(colores.CARPETA_TABLAS, "Split_*.csv"))
    if len(archivos_split) == 0:
        raise SystemExit("No hay ningún split. Ejecute primero 03_metricas.py.")
    RUTA_SPLIT = max(archivos_split, key=os.path.getmtime)
else:
    RUTA_SPLIT = os.path.join(colores.CARPETA_TABLAS, f"{NOMBRE_SPLIT}.csv")
SEMILLA_SPLIT = int(os.path.splitext(os.path.basename(RUTA_SPLIT))[0].rsplit("semilla", 1)[1])
print(f"Split: {os.path.basename(RUTA_SPLIT)} | Semilla: {SEMILLA_SPLIT}")


class CFG:

    seed = SEMILLA_SPLIT  # CAMBIADO: sale del split de 03_metricas.py
    debug = False  # UNICO CAMBIO respecto a avanzada_LINK_debug.py: True -> False
    apex = False
    print_freq = 100
    num_workers = 0  # CAMBIADO (Windows): original = 2

    OUTPUT_DIR = r"C:\Users\manue\Downloads\3ccbe\3ccbe\base3_DL\avanzada_LINK_final"  # CAMBIADO

    train_datadir = r"C:\Users\manue\Downloads\3ccbe\train_audio"  # CAMBIADO
    train_csv = RUTA_SPLIT  # CAMBIADO: lista de audios y especies del split de 03_metricas.py
    test_soundscapes = ''
    submission_csv = ''
    taxonomy_csv = r"C:\Users\manue\Downloads\3ccbe\DATOS\taxonomy.csv"  # CAMBIADO: ruta local

    spectrogram_npy = r"C:\Users\manue\Downloads\3ccbe\birdclef2025_melspec_5sec_256_256.npy"

    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1

    LOAD_DATA = False  # CAMBIADO de True: riesgo de memoria con el .npy de 7.49GB completo
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 32
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5
    mixup_alpha = 0.5

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]


cfg = CFG()


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(cfg.seed)


def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX, power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_norm


def process_audio_file(audio_path, cfg):
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, (0, target_samples - len(center_audio)), mode='constant')

        mel_spec = audio2melspec(center_audio, cfg)

        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {'melspec': spec, 'target': torch.tensor(target, dtype=torch.float32), 'filename': row['filename']}

    def apply_spec_augmentations(self, spec):
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def encode_label(self, label):
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
    result = {key: [] for key in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    return result


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name, pretrained=cfg.pretrained, in_chans=cfg.in_channels,
            drop_rate=0.2, drop_path_rate=0.2
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        indices = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[indices]
        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")


def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=cfg.min_lr)
    elif cfg.scheduler == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=cfg.epochs // 3, gamma=0.5)
    elif cfg.scheduler == 'OneCycleLR':
        return None
    else:
        return None


def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    losses, all_targets, all_outputs = [], [], []
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:
        if isinstance(batch['melspec'], list):
            batch_outputs, batch_losses = [], []
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix({'train_loss': np.mean(losses[-10:]) if losses else 0, 'lr': optimizer.param_groups[0]['lr']})

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    y_true = np.argmax(all_targets, axis=1)
    y_pred = np.argmax(all_outputs, axis=1)
    metr = calc_metricas(y_true, y_pred, sorted(set(y_true.tolist())))
    return avg_loss, auc, metr


def validate(model, loader, criterion, device):
    model.eval()
    losses, all_targets, all_outputs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch['melspec'], list):
                batch_outputs, batch_losses = [], []
                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)
                    output = model(inputs)
                    loss = criterion(output, target)
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()
            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    y_true = np.argmax(all_targets, axis=1)
    y_pred = np.argmax(all_outputs, axis=1)
    metr = calc_metricas(y_true, y_pred, sorted(set(y_true.tolist())))
    # CAMBIO: también devuelve la especie real y la predicha de cada audio (para la matriz de confusión).
    return avg_loss, auc, metr, y_true, y_pred


def calculate_auc(targets, outputs):
    num_classes = targets.shape[1]
    aucs = []
    probs = 1 / (1 + np.exp(-outputs))
    for i in range(num_classes):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    return np.mean(aucs) if aucs else 0.0


def run_training(df, cfg):
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)

    if cfg.debug:
        cfg.update_debug_settings()

    spectrograms = None
    if cfg.LOAD_DATA:
        print("Loading pre-computed mel spectrograms from NPY file...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}")
            print("Will generate spectrograms on-the-fly instead.")
            cfg.LOAD_DATA = False

    if not cfg.LOAD_DATA:
        print("Will generate spectrograms on-the-fly during training.")
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    # CAMBIO: se elimina el K-fold. Se usa el split único de 03_metricas.py, el
    # MISMO de la línea clásica y de la plantilla, para que los tres métodos se
    # evalúen sobre exactamente los mismos audios. Como df es el mismo archivo
    # del split, el conjunto de cada audio está en su propia fila.
    conjunto = df.iloc[:, POS_SPLIT_SET]
    splits_una_vez = [(np.where(conjunto == "train")[0], np.where(conjunto == "test")[0])]

    best_scores = []

    for fold, (train_idx, val_idx) in enumerate(splits_una_vez):
        if fold not in cfg.selected_folds:
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')

        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)

        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader),
                                                 epochs=cfg.epochs, pct_start=0.1)
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_auc = 0
        best_epoch = 0
        filas_epocas = []

        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_auc, m_tr = train_one_epoch(
                model, train_loader, optimizer, criterion, cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            val_loss, val_auc, m_te, reales_te, predichas_te = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # CAMBIO: se muestran y se guardan AUC, accuracy, recall macro y F1 macro.
            print(f"  TRAIN | {fmt(train_auc, m_tr)}", flush=True)
            print(f"  TEST  | {fmt(val_auc, m_te)}", flush=True)
            filas_epocas.append({"epoca": epoch + 1, "train_loss": train_loss, "train_auc": train_auc, **{f"train_{k}": v for k, v in m_tr.items()}, "test_loss": val_loss, "test_auc": val_auc, **{f"test_{k}": v for k, v in m_te.items()}})

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch, 'val_auc': val_auc, 'train_auc': train_auc, 'cfg': cfg
                }, os.path.join(cfg.OUTPUT_DIR, f"model_fold{fold}.pth"))

        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")

        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*60)
    print("Resultados (split unico 70/30, sin K-fold):")
    print(f"Mejor AUC (test, mejor epoca): {best_scores[0]:.4f}")
    print("RESULTADO FINAL (ultima epoca, sin seleccionar):", fmt(val_auc, m_te), flush=True)
    # CAMBIO: los resultados por época y las predicciones de la última época se
    # guardan en salidas/espectrogramas, para 10_graficos.py.
    #   - Épocas. Columnas: 0 época, 1 a 5 train (pérdida, AUC, accuracy, recall,
    #     F1) y 6 a 10 prueba en el mismo orden.
    #   - Predicciones. Columnas: 0 archivo, 1 especie real, 2 especie predicha.
    carpeta_resultados = os.path.join(colores.CARPETA_BASE, "salidas", "espectrogramas")
    os.makedirs(carpeta_resultados, exist_ok=True)
    nombre_resultados = os.path.splitext(os.path.basename(RUTA_SPLIT))[0].replace("Split_", "Espectrogramas_", 1)
    ruta_epocas = os.path.join(carpeta_resultados, f"{nombre_resultados}_epocas.csv")
    pd.DataFrame(filas_epocas).to_csv(ruta_epocas, index=False)
    nombre_por_indice = taxonomy_df["scientific_name"].tolist()
    especies_reales = [nombre_por_indice[indice] for indice in reales_te]
    especies_predichas = [nombre_por_indice[indice] for indice in predichas_te]
    predicciones = pd.DataFrame({"archivo": val_df.iloc[:, POS_SPLIT_ARCHIVO].to_numpy(), "especie_real": especies_reales, "especie_predicha": especies_predichas})
    ruta_predicciones = os.path.join(carpeta_resultados, f"{nombre_resultados}_predicciones.csv")
    predicciones.to_csv(ruta_predicciones, index=False)
    print(f"Épocas guardadas en: {ruta_epocas}")
    print(f"Predicciones guardadas en: {ruta_predicciones}")
    print("="*60)


if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    # Adaptacion minima de columnas + FIX del mapeo de etiquetas (ver punto 7
    # del encabezado). taxonomy.csv usa codigos (bbwduc, etc.), no nombres
    # cientificos.
    sci_to_id = dict(zip(taxonomy_df["scientific_name"], taxonomy_df["primary_label"]))
    train_df["primary_label"] = train_df.iloc[:, POS_SPLIT_ESPECIE].map(sci_to_id)
    sin_match = train_df["primary_label"].isna().sum()
    if sin_match > 0:
        raise ValueError(f"{sin_match} filas no encontraron primary_label en taxonomy.csv")

    train_df["filepath"] = train_df.iloc[:, POS_SPLIT_ARCHIVO].apply(lambda p: os.path.join(cfg.train_datadir, p))
    train_df["filename"] = train_df["filepath"].apply(
        lambda p: os.path.relpath(p, cfg.train_datadir).replace("\\", "/")
    )

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")

    run_training(train_df, cfg)

    print("\nTraining complete!")
