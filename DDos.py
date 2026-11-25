import os
import sys
import tkinter as tk
from tkinter import filedialog
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Desactivar advertencias informativas de Matplotlib/Sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def seleccionar_archivo_csv():
  """Abre la ventana emergente del sistema para seleccionar el dataset CSV."""
  root = tk.Tk()
  root.withdraw()
  root.attributes("-topmost", True)

  print("📂 Selecciona tu dataset CSV desde la ventana emergente...")
  file_path = filedialog.askopenfilename(
      title="Selecciona el dataset preparado para entrenamiento",
      filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
  )
  return file_path


def evaluar_modelo(
    nombre_modelo, model, X_test, y_test, palette_colors, target_names
):
  """Calcula métricas y genera la matriz de confusión explicitando VN, FP, FN, VP."""
  print(f"\n==========================================")
  print(f"📊 EVALUACIÓN: {nombre_modelo}")
  print(f"==========================================")

  # Predicciones de clase y probabilidades
  y_pred = model.predict(X_test)

  # Obtener probabilidades (manejo por si el estimador no implementa predict_proba)
  if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
  else:
    auc = 0.0

  # Métricas principales
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, zero_division=0)
  rec = recall_score(y_test, y_pred, zero_division=0)
  f1 = f1_score(y_test, y_pred, zero_division=0)

  print(f"Accuracy : {acc:.4f}")
  print(f"Precision: {prec:.4f}")
  print(f"Recall   : {rec:.4f}")
  print(f"F1-Score : {f1:.4f}")
  print(f"ROC-AUC  : {auc:.4f}\n")

  print("--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
  print(
      classification_report(
          y_test, y_pred, target_names=target_names, zero_division=0
      )
  )

  # Matriz de Confusión 2x2
  cm = confusion_matrix(y_test, y_pred)
  vn, fp, fn, vp = cm.ravel()
  total = np.sum(cm)

  # Formato con siglas y porcentajes
  labels_matrix = np.array([
      [
          f"VN\n{vn:,}\n({vn/total:.1%})",
          f"FP (Alarma Falsa)\n{fp:,}\n({fp/total:.1%})",
      ],
      [
          f"FN (Ataque Evadido)\n{fn:,}\n({fn/total:.1%})",
          f"VP\n{vp:,}\n({vp/total:.1%})",
      ],
  ])

  # Visualización
  fig, ax = plt.subplots(figsize=(7, 5.5))
  sns.heatmap(
      cm,
      annot=labels_matrix,
      fmt="",
      cmap="Blues",
      xticklabels=target_names,
      yticklabels=target_names,
      cbar=False,
      annot_kws={"size": 11, "weight": "bold"},
      ax=ax,
  )

  ax.set_title(
      f"Matriz de Confusión Explicitada - {nombre_modelo}",
      fontsize=12,
      weight="bold",
      pad=15,
  )
  ax.set_xlabel("Clase Predicha por el Modelo", fontsize=10, weight="bold")
  ax.set_ylabel("Clase Real del Tráfico", fontsize=10, weight="bold")

  fig.tight_layout()
  filename = (
      f"matriz_confusion_explicita_{nombre_modelo.lower().replace(' ', '_')}.png"
  )
  plt.savefig(filename, dpi=300)
  plt.show()
  plt.close(fig)

  return {
      "Modelo": nombre_modelo,
      "Accuracy": acc,
      "Precision": prec,
      "Recall": rec,
      "F1-Score": f1,
      "ROC-AUC": auc,
  }


def main():
  # ==========================================
  # 1. CARGA DEL DATASET
  # ==========================================
  archivo_path = seleccionar_archivo_csv()

  if not archivo_path:
    print(
        "⚠️ No se seleccionó ningún archivo. Cancelando la ejecución del"
        " script."
    )
    sys.exit()

  nombre_archivo = os.path.basename(archivo_path)
  print(f"\n✅ Archivo seleccionado: '{nombre_archivo}'")
  print("Cargando datos en memoria...")

  try:
    df_raw = pd.read_csv(archivo_path, low_memory=False)
  except Exception as e:
    print(f"❌ Error al leer el archivo CSV: {e}")
    sys.exit()

  # ==========================================
  # 2. LIMPIEZA Y PREPARACIÓN
  # ==========================================
  df_raw.columns = df_raw.columns.str.strip()

  if "Label" not in df_raw.columns:
    print("❌ Error: No se encontró la columna 'Label' en el CSV.")
    sys.exit()

  # Binarizar Label (0: BENIGN, 1: DDoS)
  df_raw["Label"] = df_raw["Label"].apply(
      lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
  )

  selected_features = [
      "Flow Duration",
      "Flow Packets/s",
      "Flow Bytes/s",
      "Total Fwd Packets",
      "Total Backward Packets",
      "Fwd Packet Length Mean",
      "Fwd Packet Length Std",
      "Bwd Packet Length Mean",
      "Flow IAT Mean",
      "Flow IAT Std",
      "FIN Flag Count",
      "SYN Flag Count",
      "RST Flag Count",
      "ACK Flag Count",
  ]

  available_cols = [c for c in selected_features if c in df_raw.columns]
  X = df_raw[available_cols].copy()
  y = df_raw["Label"].copy()

  # Limpieza de nulos e infinitos
  X.replace([np.inf, -np.inf], np.nan, inplace=True)
  indices_validos = X.dropna().index
  X = X.loc[indices_validos]
  y = y.loc[indices_validos]

  print(
      f"Dimensiones limpias: {X.shape[0]} filas, {X.shape[1]} variables"
      " seleccionadas."
  )

  # ==========================================
  # 3. SEPARACIÓN ESTRATIFICADA Y ESCALAMIENTO
  # ==========================================
  print(
      "\nDividiendo conjunto de datos (70% Train, 15% Validation, 15% Test)..."
  )
  X_train, X_temp, y_train, y_temp = train_test_split(
      X, y, test_size=0.30, random_state=42, stratify=y
  )
  X_val, X_test, y_val, y_test = train_test_split(
      X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
  )

  # Para Redes Neuronales EL ESCALAMIENTO ES OBLIGATORIO
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  X_test_scaled = scaler.transform(X_test)

  target_names = ["BENIGN", "DDoS"]
  palette_colors = {"BENIGN": "#2ecc71", "DDoS": "#e74c3c"}
  resultados = []

  # ==========================================
  # 4. ENTRENAMIENTO DE RANDOM FOREST
  # ==========================================
  print("\n🌲 1/3 Entrenando Random Forest Classifier...")
  rf_model = RandomForestClassifier(
      n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
  )
  rf_model.fit(X_train_scaled, y_train)

  res_rf = evaluar_modelo(
      "Random Forest",
      rf_model,
      X_test_scaled,
      y_test,
      palette_colors,
      target_names,
  )
  resultados.append(res_rf)

  # ==========================================
  # 5. ENTRENAMIENTO DE XGBOOST
  # ==========================================
  print("\n⚡ 2/3 Entrenando XGBoost Classifier...")
  xgb_model = XGBClassifier(
      n_estimators=100,
      max_depth=6,
      learning_rate=0.1,
      random_state=42,
      n_jobs=-1,
      eval_metric="logloss",
  )
  xgb_model.fit(
      X_train_scaled,
      y_train,
      eval_set=[(X_val_scaled, y_val)],
      verbose=False,
  )

  res_xgb = evaluar_modelo(
      "XGBoost",
      xgb_model,
      X_test_scaled,
      y_test,
      palette_colors,
      target_names,
  )
  resultados.append(res_xgb)

  # ==========================================
  # 6. ENTRENAMIENTO DE PERCEPTRÓN MULTICAPA (MLP)
  # ==========================================
  print("\n🧠 3/3 Entrenando Red Neuronal (Perceptrón Multicapa - MLP)...")
  mlp_model = MLPClassifier(
      hidden_layer_sizes=(
          64,
          32,
      ),  # Capa 1: 64 neuronas, Capa 2: 32 neuronas
      activation="relu",  # Función de activación no lineal
      solver="adam",  # Optimizador Adam basado en gradiente estocástico
      max_iter=50,  # Máximo de iteraciones / épocas
      random_state=42,
      early_stopping=True,  # Detiene el entrenamiento si la validación deja de mejorar
  )
  mlp_model.fit(X_train_scaled, y_train)

  res_mlp = evaluar_modelo(
      "Red Neuronal (MLP)",
      mlp_model,
      X_test_scaled,
      y_test,
      palette_colors,
      target_names,
  )
  resultados.append(res_mlp)

  # ==========================================
  # 7. TABLA RESUMEN COMPARATIVA
  # ==========================================
  print("\n==========================================")
  print("🏆 RESUMEN COMPARATIVO FINAL DE MODELOS")
  print("==========================================")
  df_res = pd.DataFrame(resultados)
  print(df_res.to_string(index=False))


if __name__ == "__main__":
  main()