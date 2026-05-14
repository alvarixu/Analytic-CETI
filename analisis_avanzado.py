"""
=============================================================
 ANÁLISIS ACÚSTICO AVANZADO DE CETÁCEOS  —  CETI Project
=============================================================
Sistema de análisis y clasificación de especies de cetáceos 
mediante características acústicas (MFCCs) y machine learning.

Ejecución: python analisis_avanzado.py

Dependencias: numpy, pandas, matplotlib, seaborn, scipy, 
              scikit-learn, plotly, umap-learn

Nota: Utiliza base de datos embebida (sin requerimientos 
      de conexión a Azure ML o servicios en la nube)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── DATOS EMBEBIDOS (MFCCs pre-calculados - sin dependencia de Azure ML) ─────
# Base de datos local con características acústicas de 11 especies de cetáceos
SPECIES = [
    "Blue Whale",
    "Fin Whale",
    "Humpback Whale",
    "Killer Whale",
    "Minke Whale",
    "Right Whale (1)",
    "Right Whale (2)",
    "Bottlenose Whale",
    "Pilot Whale",
    "Sei Whale",
    "Sperm Whale",
]

FAMILIES = [
    "Balaenopteridae",
    "Balaenopteridae",
    "Balaenopteridae",
    "Delphinidae",
    "Balaenopteridae",
    "Balaenidae",
    "Balaenidae",
    "Ziphiidae",
    "Delphinidae",
    "Balaenopteridae",
    "Physeteridae",
]

MFCCS = np.array([
    [-323.40, 133.39, 92.24, 44.13,  6.80, -10.10, -8.30,  2.07,  9.97,  9.91,  3.54, -3.38, -6.06],
    [-387.13,  61.83, 42.18, 19.38,  1.88,  -5.64, -3.39,  4.25, 11.42, 14.32, 12.78,  9.16,  6.13],
    [-372.68, 111.19, 72.76, 29.07, -2.05, -12.38, -6.10,  5.20, 11.01,  8.07,  0.45, -5.21, -5.12],
    [-165.00,  32.78,-11.27,  6.60, -5.19,   7.27, -1.41,  0.62, -1.73,  3.26, -1.99,  7.62,  1.07],
    [-491.98, 129.96, 92.95, 50.06, 17.85,   4.26,  5.95, 12.65, 14.95,  9.95,  1.41, -4.54, -4.57],
    [-280.10,  95.40, 55.30, 22.80,  3.10,  -7.20, -4.30,  3.60,  8.90,  7.50,  1.80, -2.90, -3.40],
    [-275.50,  98.20, 58.10, 24.30,  3.80,  -6.90, -4.10,  3.90,  9.20,  7.80,  2.10, -2.70, -3.20],
    [-210.80,  55.20, 10.30, 15.70,  0.40,   9.50,  1.20,  2.80,  3.10,  6.20,  0.30,  9.10,  3.50],
    [-190.30,  45.60, -5.40, 10.20, -3.20,   8.90, -0.80,  1.40,  0.50,  4.80, -0.90,  8.30,  2.30],
    [-410.50, 118.30, 80.20, 35.10,  5.20,  -8.70, -5.50,  4.10, 10.50,  9.20,  2.10, -4.20, -4.80],
    [-140.20,  28.50,-18.40,  4.20, -7.80,   9.10, -2.30, -0.40, -3.20,  2.10, -3.50,  6.90,  0.20],
])

UMAP_X = [-16.40, -16.63, -17.03,  -7.60,  -0.44, -0.02, -0.76, -7.48, -7.88, -17.26, -8.09]
UMAP_Y = [-11.02, -10.66, -10.35,  11.58,  -4.36, -4.78, -4.04, 12.08, 12.57, -10.09,  11.98]

MFCC_COLS = [f"MFCC_{i}" for i in range(13)]

PALETTE = {
    "Balaenopteridae": "#0096c7",
    "Balaenidae":      "#f77f00",
    "Delphinidae":     "#e63946",
    "Physeteridae":    "#2dc653",
    "Ziphiidae":       "#8338ec",
}

df = pd.DataFrame(MFCCS, columns=MFCC_COLS)
df["Especie"] = SPECIES
df["Familia"] = FAMILIES
df["UMAP_X"]  = UMAP_X
df["UMAP_Y"]  = UMAP_Y

plt.rcParams.update({
    "figure.facecolor": "#0a0e1a",
    "axes.facecolor":   "#0d1b2a",
    "axes.edgecolor":   "#1e3a5f",
    "axes.labelcolor":  "#90caf9",
    "xtick.color":      "#90caf9",
    "ytick.color":      "#90caf9",
    "text.color":       "#e0f4ff",
    "grid.color":       "#1e3a5f",
    "grid.alpha":       0.5,
    "axes.titlecolor":  "#00b4d8",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "font.family":      "DejaVu Sans",
})

fam_colors = [PALETTE[f] for f in FAMILIES]


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 1 — ESTADÍSTICAS DESCRIPTIVAS POR FAMILIA
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 1: Estadísticas descriptivas de MFCCs")
print("="*60)
stats = df.groupby("Familia")[MFCC_COLS].agg(["mean", "std"]).round(2)
print(stats.to_string())
print("\n💡 Interpretación:")
print("  • MFCC_0 (energía) es siempre muy negativo → todas las vocalizaciones")
print("    son de baja frecuencia. Los Balaenopteridae tienen valores más bajos")
print("    (-410 a -280) indicando vocalizaciones de más baja frecuencia.")
print("  • Los Delphinidae (orcas, calderones) tienen MFCC_0 más alto (-190 a -165)")
print("    reflejando vocalizaciones de mayor frecuencia (clicks, silbidos).")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 2 — HEATMAP DE CORRELACIÓN ENTRE MFCCs
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 2: Correlación entre coeficientes MFCC")
corr = df[MFCC_COLS].corr()

fig, ax = plt.subplots(figsize=(11, 8))
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            mask=mask, ax=ax, linewidths=0.5, linecolor="#0a0e1a",
            cbar_kws={"label": "Correlación de Pearson"})
ax.set_title("Correlación entre los 13 Coeficientes MFCC\n(todas las especies)")
plt.tight_layout()
plt.savefig("analisis_correlacion_mfcc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ Guardado: analisis_correlacion_mfcc.png")
print("  💡 Correlaciones altas entre MFCCs consecutivos indican redundancia espectral.")
print("     MFCC_1 y MFCC_2 suelen correlacionar pues ambos capturan la")
print("     envolvente espectral de baja frecuencia.")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 3 — MATRIZ DE DISTANCIAS ACÚSTICAS ENTRE ESPECIES
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 3: Distancias acústicas euclidianas entre especies")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(MFCCS)
dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
dist_df = pd.DataFrame(dist_matrix, index=SPECIES, columns=SPECIES)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(dist_df, annot=True, fmt=".2f", cmap="YlOrRd_r",
            ax=ax, linewidths=0.5, linecolor="#0a0e1a",
            xticklabels=[s[:12] for s in SPECIES],
            yticklabels=SPECIES)
ax.set_title("Matriz de Distancias Acústicas entre Especies\n(MFCCs normalizados, distancia euclídea)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("analisis_distancias_acusticas.png", dpi=150, bbox_inches="tight")
plt.close()

# Top 3 pares más similares
dist_flat = dist_df.where(np.triu(np.ones(dist_df.shape), k=1).astype(bool)).stack()
print("  Top 5 pares MÁS similares acústicamente:")
for (s1, s2), d in dist_flat.nsmallest(5).items():
    print(f"    {s1:22s} ↔ {s2:22s}  dist={d:.3f}")
print("  ✔ Guardado: analisis_distancias_acusticas.png")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 4 — DENDROGRAMA (CLUSTERING JERÁRQUICO)
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 4: Clustering jerárquico (dendrograma)")
Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(13, 6))
dend = dendrogram(Z, labels=SPECIES, leaf_rotation=45, leaf_font_size=10,
                  color_threshold=3.5, ax=ax,
                  link_color_func=lambda k: "#00b4d8")
# Colorear etiquetas por familia
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    sp = lbl.get_text()
    idx = SPECIES.index(sp)
    lbl.set_color(PALETTE[FAMILIES[idx]])
ax.set_title("Dendrograma de Similitud Acústica entre Cetáceos\n(Ward linkage sobre MFCCs normalizados)")
ax.set_ylabel("Distancia de Ward")
ax.axhline(3.5, color="#ffd700", ls="--", lw=1.5, label="Umbral de corte")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("analisis_dendrograma.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ Guardado: analisis_dendrograma.png")
print("  💡 El dendrograma agrupa las especies por similitud acústica sin usar")
print("     etiquetas de familia. Si los clusters coinciden con las familias")
print("     taxonómicas, confirma que la acústica refleja la evolución.")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 5 — PCA: COMPARACIÓN CON UMAP
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 5: PCA vs UMAP — comparación de reducciones")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (coords, title, xlabel, ylabel) in zip(axes, [
    (X_pca,                   "PCA",  f"PC1 ({var_exp[0]:.1f}% var)", f"PC2 ({var_exp[1]:.1f}% var)"),
    (np.c_[UMAP_X, UMAP_Y],  "UMAP", "UMAP Dimensión 1",              "UMAP Dimensión 2"),
]):
    seen = set()
    for i, (sp, fam) in enumerate(zip(SPECIES, FAMILIES)):
        c = PALETTE[fam]
        label = fam if fam not in seen else None
        seen.add(fam)
        ax.scatter(coords[i, 0], coords[i, 1], s=200, c=c, edgecolors="#e0f4ff",
                   linewidth=1.5, zorder=3, label=label)
        ax.annotate(sp[:10], (coords[i, 0], coords[i, 1]),
                    fontsize=7.5, color="#e0f4ff", ha="center",
                    xytext=(0, 10), textcoords="offset points")
    ax.set_title(f"Reducción Dimensional: {title}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.3)

plt.suptitle("PCA vs UMAP sobre características MFCC de Cetáceos",
             fontsize=14, fontweight="bold", color="#00b4d8", y=1.01)
plt.tight_layout()
plt.savefig("analisis_pca_vs_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Varianza explicada por PCA: PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%")
print(f"  Total explicado: {sum(var_exp):.1f}%")
print("  ✔ Guardado: analisis_pca_vs_umap.png")
print("  💡 PCA es lineal; UMAP no-lineal. Con datasets pequeños como este,")
print("     PCA puede ser más estable. UMAP preserva mejor la topología local.")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 6 — IMPORTANCIA DE FEATURES (MFCC más discriminativo)
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 6: ¿Qué MFCCs discriminan mejor entre familias?")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_scaled, FAMILIES)
importances = rf.feature_importances_
idx_sorted = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(13), importances[idx_sorted],
              color=[plt.cm.cool(v) for v in importances[idx_sorted]/importances.max()])
ax.set_xticks(range(13))
ax.set_xticklabels([MFCC_COLS[i] for i in idx_sorted], rotation=45, ha="right")
ax.set_title("Importancia de cada MFCC para clasificar familias\n(Random Forest, 500 árboles)")
ax.set_ylabel("Importancia (Gini)")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("analisis_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Top 5 MFCCs más discriminativos:")
for rank, i in enumerate(idx_sorted[:5], 1):
    print(f"    {rank}. {MFCC_COLS[i]:10s}  importancia={importances[i]:.4f}")
print("  ✔ Guardado: analisis_feature_importance.png")
print("  💡 MFCC_0 captura la energía global; los primeros coeficientes")
print("     codifican la forma espectral gruesa (frecuencias dominantes).")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 7 — CLASIFICADOR KNN CON VALIDACIÓN LEAVE-ONE-OUT
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 7: Evaluación del clasificador KNN (Leave-One-Out)")
loo = LeaveOneOut()
y_true, y_pred = [], []
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    preds = []
    for train_idx, test_idx in loo.split(X_scaled):
        knn.fit(X_scaled[train_idx], np.array(FAMILIES)[train_idx])
        preds.append(knn.predict(X_scaled[test_idx])[0])
    acc = accuracy_score(FAMILIES, preds)
    print(f"  KNN k={k}: Accuracy LOO = {acc*100:.1f}%")

# Confusion matrix con k=3
knn3 = KNeighborsClassifier(n_neighbors=3, weights="distance")
preds3 = []
for train_idx, test_idx in loo.split(X_scaled):
    knn3.fit(X_scaled[train_idx], np.array(FAMILIES)[train_idx])
    preds3.append(knn3.predict(X_scaled[test_idx])[0])

families_unique = sorted(set(FAMILIES))
cm = confusion_matrix(FAMILIES, preds3, labels=families_unique)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=families_unique, yticklabels=families_unique, ax=ax,
            linewidths=0.5, linecolor="#0a0e1a")
ax.set_title("Matriz de Confusión — KNN k=3 (Leave-One-Out)\nClasificación por familia taxonómica")
ax.set_ylabel("Familia Real"); ax.set_xlabel("Familia Predicha")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("analisis_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ Guardado: analisis_confusion_matrix.png")
print("  💡 Con solo 11 muestras el LOO es la validación más honesta.")
print("     Un accuracy alto confirma que los MFCCs capturan diferencias reales.")


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS 8 — PERFIL ACÚSTICO RADAR POR FAMILIA
# ═══════════════════════════════════════════════════════════════
print("\n📊 ANÁLISIS 8: Perfil acústico (radar) por familia")
from matplotlib.patches import FancyArrowPatch

fam_mean = df.groupby("Familia")[MFCC_COLS[:8]].mean()  # primeros 8 por claridad
labels_radar = MFCC_COLS[:8]
N = len(labels_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_facecolor("#0d1b2a")
fig.patch.set_facecolor("#0a0e1a")

for fam, row in fam_mean.iterrows():
    vals = row.values.tolist()
    vals_norm = [(v - fam_mean.values.min()) /
                 (fam_mean.values.max() - fam_mean.values.min() + 1e-9) for v in vals]
    vals_norm += vals_norm[:1]
    c = PALETTE[fam]
    ax.plot(angles, vals_norm, "o-", lw=2, color=c, label=fam)
    ax.fill(angles, vals_norm, alpha=0.12, color=c)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, color="#90caf9", size=10)
ax.set_yticklabels([])
ax.grid(color="#1e3a5f", linestyle="--", linewidth=0.8)
ax.set_title("Perfil Acústico Radar por Familia Taxonómica\n(8 primeros MFCCs normalizados)",
             pad=20, color="#00b4d8", size=13, weight="bold")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9,
          framealpha=0.3, labelcolor="w")
plt.tight_layout()
plt.savefig("analisis_radar_familias.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ Guardado: analisis_radar_familias.png")
print("  💡 El radar muestra la 'firma acústica' de cada familia.")
print("     Formas muy distintas = familias acústicamente diferenciadas.")


# ═══════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅ ANÁLISIS COMPLETO — Archivos generados:")
files = [
    ("analisis_correlacion_mfcc.png",    "Correlación entre los 13 MFCCs"),
    ("analisis_distancias_acusticas.png","Matriz de distancias entre especies"),
    ("analisis_dendrograma.png",         "Clustering jerárquico (dendrograma)"),
    ("analisis_pca_vs_umap.png",         "Comparación PCA vs UMAP"),
    ("analisis_feature_importance.png",  "MFCCs más discriminativos (Random Forest)"),
    ("analisis_confusion_matrix.png",    "Matriz de confusión KNN LOO"),
    ("analisis_radar_familias.png",      "Perfil acústico radar por familia"),
]
for f, desc in files:
    print(f"  📊 {f:42s} → {desc}")
print("="*60)
