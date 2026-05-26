"""
================================================================================
PROJEKT DATA MINING – weatherAUS
Osoba 1: Business Understanding & Data Understanding (EDA)
================================================================================

Cel skryptu:
  1. Wczytać zbiór weatherAUS.csv
  2. Wygenerować statystyki opisowe i policzyć liczbę braków
  3. Wygenerować wykresy EDA jako PNG (do wklejenia do pliku Word)
  4. Zapisać tabelę statystyk do CSV (do wklejenia do dokumentu)

Wymagania:
  pip install pandas numpy matplotlib seaborn

Uruchomienie:
  1. Upewnij się, że weatherAUS.csv jest w tym samym folderze co skrypt
  2. python eda_osoba1.py
  3. Wszystkie wykresy zostaną zapisane w podfolderze ./figures/
  4. Tabele statystyk zostaną zapisane w podfolderze ./tables/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

DATA_PATH = "weatherAUS.csv"
FIG_DIR = "figures"
TAB_DIR = "tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# ============================================================
# 1. WCZYTANIE DANYCH
# ============================================================
print("=" * 70)
print("1. WCZYTANIE DANYCH")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

print(f"Liczba obserwacji:  {len(df):,}")
print(f"Liczba zmiennych:   {df.shape[1]}")
print(f"Zakres dat:         {df['Date'].min().date()}  →  {df['Date'].max().date()}")
print(f"Liczba lokalizacji: {df['Location'].nunique()}")


# ============================================================
# 2. ROZKŁAD ZMIENNEJ CELU
# ============================================================
print("\n" + "=" * 70)
print("2. ROZKŁAD ZMIENNEJ CELU (RainTomorrow)")
print("=" * 70)

target_counts = df["RainTomorrow"].value_counts(dropna=False)
target_pct = df["RainTomorrow"].value_counts(normalize=True, dropna=False) * 100
print(pd.concat([target_counts.rename("count"), target_pct.rename("%")], axis=1))

# Wykres 1: rozkład klasy celu
fig, ax = plt.subplots(figsize=(6, 4))
counts_clean = df["RainTomorrow"].value_counts()
colors = ["#5B9BD5", "#ED7D31"]
bars = ax.bar(counts_clean.index, counts_clean.values, color=colors, edgecolor="black")
ax.set_title("Rozkład zmiennej celu: RainTomorrow", fontsize=12, fontweight="bold")
ax.set_xlabel("RainTomorrow")
ax.set_ylabel("Liczba obserwacji")
for bar, val in zip(bars, counts_clean.values):
    pct = val / counts_clean.sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.margins(y=0.15)
plt.savefig(f"{FIG_DIR}/01_rozklad_celu.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/01_rozklad_celu.png")


# ============================================================
# 3. BRAKI DANYCH
# ============================================================
print("\n" + "=" * 70)
print("3. BRAKI DANYCH (% braków per zmienna)")
print("=" * 70)

missing = (df.isna().sum() / len(df) * 100).round(2).sort_values(ascending=False)
missing_df = missing[missing > 0].rename("% braków").to_frame()
print(missing_df)
missing_df.to_csv(f"{TAB_DIR}/braki_danych.csv")
print(f"  → Zapisano tabelę: {TAB_DIR}/braki_danych.csv")

# Wykres 2: braki danych
fig, ax = plt.subplots(figsize=(8, 6))
missing_plot = missing[missing > 0]
bars = ax.barh(range(len(missing_plot)), missing_plot.values,
               color=["#C00000" if x > 30 else "#ED7D31" if x > 10 else "#70AD47"
                      for x in missing_plot.values])
ax.set_yticks(range(len(missing_plot)))
ax.set_yticklabels(missing_plot.index)
ax.invert_yaxis()
ax.set_xlabel("% braków")
ax.set_title("Odsetek braków danych dla zmiennych", fontsize=12, fontweight="bold")
for i, val in enumerate(missing_plot.values):
    ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)
ax.axvline(30, color="red", linestyle="--", alpha=0.4, label="próg 30%")
ax.legend(loc="lower right")
plt.savefig(f"{FIG_DIR}/02_braki_danych.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/02_braki_danych.png")


# ============================================================
# 4. STATYSTYKI OPISOWE ZMIENNYCH NUMERYCZNYCH
# ============================================================
print("\n" + "=" * 70)
print("4. STATYSTYKI OPISOWE ZMIENNYCH NUMERYCZNYCH")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
stats = df[numeric_cols].describe().T.round(2)
stats["missing_%"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2).values
print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_%"]])
stats.to_csv(f"{TAB_DIR}/statystyki_opisowe.csv")
print(f"\n  → Zapisano tabelę: {TAB_DIR}/statystyki_opisowe.csv")


# ============================================================
# 5. ROZKŁADY ZMIENNYCH NUMERYCZNYCH (histogramy)
# ============================================================
print("\n" + "=" * 70)
print("5. HISTOGRAMY ZMIENNYCH NUMERYCZNYCH")
print("=" * 70)

n = len(numeric_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=40, color="#5B9BD5", edgecolor="black")
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
for j in range(n, len(axes)):
    axes[j].axis("off")
plt.suptitle("Histogramy zmiennych numerycznych", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_histogramy.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/03_histogramy.png")


# ============================================================
# 6. BOXPLOTY – DETEKCJA WARTOŚCI ODSTAJĄCYCH
# ============================================================
print("\n" + "=" * 70)
print("6. BOXPLOTY (outliers)")
print("=" * 70)

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#5B9BD5"))
    axes[i].set_title(col, fontsize=10, fontweight="bold")
for j in range(n, len(axes)):
    axes[j].axis("off")
plt.suptitle("Boxploty zmiennych numerycznych (detekcja outlierów)",
             fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_boxploty.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/04_boxploty.png")


# ============================================================
# 7. MACIERZ KORELACJI
# ============================================================
print("\n" + "=" * 70)
print("7. MACIERZ KORELACJI (zmienne numeryczne)")
print("=" * 70)

corr = df[numeric_cols].corr().round(2)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1, square=True, linewidths=0.5)
ax.set_title("Macierz korelacji – zmienne numeryczne", fontsize=12, fontweight="bold")
plt.savefig(f"{FIG_DIR}/05_korelacje.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/05_korelacje.png")
corr.to_csv(f"{TAB_DIR}/macierz_korelacji.csv")
print(f"  → Zapisano tabelę: {TAB_DIR}/macierz_korelacji.csv")


# ============================================================
# 8. ZWIĄZEK PREDYKTORÓW Z CELEM (boxploty per RainTomorrow)
# ============================================================
print("\n" + "=" * 70)
print("8. ZWIĄZEK PREDYKTORÓW Z CELEM")
print("=" * 70)

# Wybieramy najbardziej obiecujące zmienne (na podstawie literatury i intuicji)
key_predictors = ["Humidity3pm", "Pressure3pm", "Sunshine", "Cloud3pm",
                  "Rainfall", "WindGustSpeed", "Temp3pm", "MinTemp"]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()
df_clean = df.dropna(subset=["RainTomorrow"])
for i, col in enumerate(key_predictors):
    sns.boxplot(data=df_clean, x="RainTomorrow", y=col,
                ax=axes[i], palette=["#5B9BD5", "#ED7D31"])
    axes[i].set_title(f"{col} vs RainTomorrow", fontsize=10, fontweight="bold")
    axes[i].set_xlabel("")
plt.suptitle("Rozkład kluczowych predyktorów w grupach RainTomorrow",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_predyktor_vs_cel.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/06_predyktor_vs_cel.png")


# ============================================================
# 9. ROZKŁAD GEOGRAFICZNY (liczba obserwacji per lokalizacja)
# ============================================================
print("\n" + "=" * 70)
print("9. LICZBA OBSERWACJI PER LOKALIZACJA")
print("=" * 70)

loc_counts = df["Location"].value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
loc_counts.plot(kind="barh", ax=ax, color="#5B9BD5", edgecolor="black")
ax.invert_yaxis()
ax.set_xlabel("Liczba obserwacji")
ax.set_title("Liczba obserwacji per lokalizacja (49 stacji)",
             fontsize=12, fontweight="bold")
plt.savefig(f"{FIG_DIR}/07_lokalizacje.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/07_lokalizacje.png")


# ============================================================
# 10. ROZKŁAD CZASOWY (liczba obserwacji per rok)
# ============================================================
print("\n" + "=" * 70)
print("10. ROZKŁAD CZASOWY OBSERWACJI")
print("=" * 70)

df["Year"] = df["Date"].dt.year
year_counts = df["Year"].value_counts().sort_index()
print(year_counts)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(year_counts.index.astype(str), year_counts.values,
              color="#5B9BD5", edgecolor="black")
ax.set_title("Liczba obserwacji per rok", fontsize=12, fontweight="bold")
ax.set_xlabel("Rok")
ax.set_ylabel("Liczba obserwacji")
for bar, val in zip(bars, year_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:,}",
            ha="center", va="bottom", fontsize=9)
ax.margins(y=0.1)

# Dodajemy linie pionowe pokazujące planowany split
ax.axvline(x=year_counts.index.tolist().index(2014) + 0.5, color="green",
           linestyle="--", alpha=0.7, label="Granica train/val")
ax.axvline(x=year_counts.index.tolist().index(2015) + 0.5, color="orange",
           linestyle="--", alpha=0.7, label="Granica val/test")
ax.axvline(x=year_counts.index.tolist().index(2016) + 0.5, color="red",
           linestyle="--", alpha=0.7, label="Wyłączone (2017 niepełny)")
ax.legend(loc="upper left", fontsize=9)
plt.savefig(f"{FIG_DIR}/08_rozklad_lat.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/08_rozklad_lat.png")


# ============================================================
# 11. SEZONOWOŚĆ OPADÓW (% RainTomorrow=Yes per miesiąc)
# ============================================================
print("\n" + "=" * 70)
print("11. SEZONOWOŚĆ OPADÓW")
print("=" * 70)

df["Month"] = df["Date"].dt.month
df_clean = df.dropna(subset=["RainTomorrow"])
monthly_rain = df_clean.groupby("Month").apply(
    lambda x: (x["RainTomorrow"] == "Yes").sum() / len(x) * 100
)
print(monthly_rain.round(2))

fig, ax = plt.subplots(figsize=(9, 4.5))
months_pl = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze",
             "Lip", "Sie", "Wrz", "Paź", "Lis", "Gru"]
bars = ax.bar(months_pl, monthly_rain.values, color="#5B9BD5", edgecolor="black")
ax.set_title("Sezonowość opadów: % dni z RainTomorrow=Yes per miesiąc (półkula południowa)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Miesiąc")
ax.set_ylabel("% RainTomorrow = Yes")
ax.axhline(y=monthly_rain.mean(), color="red", linestyle="--", alpha=0.6,
           label=f"Średnia: {monthly_rain.mean():.1f}%")
ax.legend()
for bar, val in zip(bars, monthly_rain.values):
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8)
ax.margins(y=0.1)
plt.savefig(f"{FIG_DIR}/09_sezonowosc.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/09_sezonowosc.png")


# ============================================================
# PODSUMOWANIE
# ============================================================
print("\n" + "=" * 70)
print("EDA ZAKOŃCZONE")
print("=" * 70)
print(f"\nWygenerowano {len(os.listdir(FIG_DIR))} plików PNG w folderze ./{FIG_DIR}/")
print(f"Wygenerowano {len(os.listdir(TAB_DIR))} tabel CSV w folderze ./{TAB_DIR}/")
print("\nGotowe do wklejenia do dokumentu Word.\n")
