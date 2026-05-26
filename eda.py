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
# 3b. ANALIZA BRAKÓW STRUKTURALNYCH (per Location)
# ============================================================
print("\n" + "=" * 70)
print("3b. BRAKI STRUKTURALNE — które stacje w ogóle nie mają pomiarów?")
print("=" * 70)

problem_cols = ['Sunshine', 'Cloud9am', 'Cloud3pm', 'Evaporation']
struct_data = []
for col in problem_cols:
    no_data = df.groupby('Location')[col].apply(lambda x: x.notna().sum() == 0).sum()
    total_locs = df['Location'].nunique()
    rows_in_no_data = df[df['Location'].isin(
        df.groupby('Location').filter(lambda g: g[col].notna().sum() == 0)['Location'].unique()
    )].shape[0]
    pct_rows = rows_in_no_data / len(df) * 100
    struct_data.append({
        'Zmienna': col,
        'Stacji bez ŻADNYCH danych': f"{no_data}/{total_locs}",
        '% wierszy ze stacji bez danych': f"{pct_rows:.1f}%",
        '% braków ogółem': f"{df[col].isna().sum()/len(df)*100:.1f}%"
    })

struct_df = pd.DataFrame(struct_data)
print(struct_df.to_string(index=False))
struct_df.to_csv(f"{TAB_DIR}/braki_strukturalne.csv", index=False)
print(f"\n  → Zapisano tabelę: {TAB_DIR}/braki_strukturalne.csv")

# Wykres 2b: braki strukturalne
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(problem_cols))
vals_struct = []
vals_random = []
for col in problem_cols:
    locs_no_data = df.groupby('Location').filter(
        lambda g: g[col].notna().sum() == 0)['Location'].unique()
    struct_missing = df[df['Location'].isin(locs_no_data)].shape[0]
    total_missing = df[col].isna().sum()
    random_missing = total_missing - struct_missing
    vals_struct.append(struct_missing / len(df) * 100)
    vals_random.append(random_missing / len(df) * 100)

ax.bar(x_pos, vals_struct, color='#C00000', label='braki strukturalne (stacja w ogóle nie mierzy)')
ax.bar(x_pos, vals_random, bottom=vals_struct, color='#ED7D31',
       label='braki losowe (stacja mierzy, ale ten dzień brak)')
ax.set_xticks(x_pos)
ax.set_xticklabels(problem_cols)
ax.set_ylabel("% obserwacji z brakiem")
ax.set_title("Rozkład braków: strukturalne vs losowe (kluczowy obraz!)",
             fontsize=12, fontweight="bold")
for i, (s, r) in enumerate(zip(vals_struct, vals_random)):
    if s > 0:
        ax.text(i, s/2, f"{s:.1f}%", ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
    if r > 0:
        ax.text(i, s + r/2, f"{r:.1f}%", ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02b_braki_strukturalne.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/02b_braki_strukturalne.png")


# ============================================================
# 3c. STRATEGIA OBSŁUGI BRAKÓW (porównanie wariantów)
# ============================================================
print("\n" + "=" * 70)
print("3c. STRATEGIA OBSŁUGI BRAKÓW — porównanie empiryczne")
print("=" * 70)
print("Eksperyment: Random Forest na 4 wariantach obsługi braków")
print("Train: lata ≤2014, Test: rok 2015")
print()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df_exp = df.dropna(subset=['RainTomorrow']).copy()
df_exp['target'] = (df_exp['RainTomorrow'] == 'Yes').astype(int)
df_exp['Date_dt'] = pd.to_datetime(df_exp['Date'])
df_exp['RainToday_b'] = (df_exp['RainToday'] == 'Yes').astype(int)
num_cols_exp = ['MinTemp','MaxTemp','Rainfall','Sunshine','WindGustSpeed',
                'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm',
                'Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']
tr = df_exp[df_exp['Date_dt'].dt.year <= 2014].copy()
te = df_exp[df_exp['Date_dt'].dt.year == 2015].copy()

global_med_exp = tr[num_cols_exp].median()
loc_med_exp = tr.groupby('Location')[num_cols_exp].median()

def eval_rf(X_tr, X_te):
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_tr, tr['target'])
    return roc_auc_score(te['target'], rf.predict_proba(X_te)[:, 1])

# Wariant 1: globalna mediana
tr1 = tr.copy(); te1 = te.copy()
tr1[num_cols_exp] = tr1[num_cols_exp].fillna(global_med_exp)
te1[num_cols_exp] = te1[num_cols_exp].fillna(global_med_exp)
auc1 = eval_rf(tr1[num_cols_exp + ['RainToday_b']].fillna(0),
                te1[num_cols_exp + ['RainToday_b']].fillna(0))

# Wariant 2: per Location wszędzie
def imp_per_loc(d):
    d = d.copy()
    for loc in d['Location'].unique():
        mask = d['Location'] == loc
        for col in num_cols_exp:
            med = loc_med_exp.loc[loc, col] if loc in loc_med_exp.index else np.nan
            if pd.isna(med):
                med = global_med_exp[col]
            d.loc[mask, col] = d.loc[mask, col].fillna(med)
    return d
tr2 = imp_per_loc(tr); te2 = imp_per_loc(te)
auc2 = eval_rf(tr2[num_cols_exp + ['RainToday_b']].fillna(0),
                te2[num_cols_exp + ['RainToday_b']].fillna(0))

# Wariant 3: HYBRYDA (Sunshine globalnie, reszta per Location) - WYBRANA STRATEGIA
def imp_hybrid(d):
    d = d.copy()
    d['Sunshine'] = d['Sunshine'].fillna(global_med_exp['Sunshine'])
    other = [c for c in num_cols_exp if c != 'Sunshine']
    for loc in d['Location'].unique():
        mask = d['Location'] == loc
        for col in other:
            med = loc_med_exp.loc[loc, col] if loc in loc_med_exp.index else np.nan
            if pd.isna(med):
                med = global_med_exp[col]
            d.loc[mask, col] = d.loc[mask, col].fillna(med)
    return d
tr3 = imp_hybrid(tr); te3 = imp_hybrid(te)
auc3 = eval_rf(tr3[num_cols_exp + ['RainToday_b']].fillna(0),
                te3[num_cols_exp + ['RainToday_b']].fillna(0))

# Wariant 4: usunięcie 4 zmiennych z dużymi brakami
safe_cols = [c for c in num_cols_exp if c not in ['Sunshine','Cloud9am','Cloud3pm','Evaporation']]
tr4 = tr.copy(); te4 = te.copy()
tr4[safe_cols] = tr4[safe_cols].fillna(global_med_exp[safe_cols])
te4[safe_cols] = te4[safe_cols].fillna(global_med_exp[safe_cols])
auc4 = eval_rf(tr4[safe_cols + ['RainToday_b']].fillna(0),
                te4[safe_cols + ['RainToday_b']].fillna(0))

results = [
    ("1. Globalna mediana wszędzie", auc1, "baseline"),
    ("2. Per Location wszędzie", auc2, f"{(auc2-auc1)*100:+.2f} pp"),
    ("3. HYBRYDA: Sunshine globalnie, reszta per Loc", auc3, f"{(auc3-auc1)*100:+.2f} pp"),
    ("4. Usunięcie zmiennych z brakami >30%", auc4, f"{(auc4-auc1)*100:+.2f} pp"),
]
print(f"{'Strategia':<55} {'AUC':>7}   Różnica")
print("-" * 80)
best_auc = max(r[1] for r in results)
for name, auc, diff in results:
    marker = "  ← WYBRANA" if name.startswith("3.") else ("  ← najlepsza empirycznie" if auc == best_auc and not name.startswith("3.") else "")
    print(f"{name:<55} {auc:.4f}   {diff}{marker}")

# Zapisz tabelę
strategies_df = pd.DataFrame(
    [{'Strategia': r[0], 'AUC': r[1], 'Różnica vs baseline': r[2]} for r in results]
)
strategies_df.to_csv(f"{TAB_DIR}/strategie_imputacji.csv", index=False)
print(f"\n  → Zapisano tabelę: {TAB_DIR}/strategie_imputacji.csv")

# Wykres porównawczy
fig, ax = plt.subplots(figsize=(10, 5))
names_short = ["Globalna\nmediana", "Per Location\nwszędzie", "Hybryda\n(Sunshine globalnie,\nreszta per Loc)", "Usunięcie\nzmiennych"]
aucs = [auc1, auc2, auc3, auc4]
colors = ['#A6A6A6', '#5B9BD5', '#70AD47', '#ED7D31']
bars = ax.bar(names_short, aucs, color=colors, edgecolor='black')
ax.set_ylabel("AUC")
ax.set_ylim(min(aucs) - 0.005, max(aucs) + 0.005)
ax.set_title("Porównanie strategii obsługi braków (Random Forest, walidacja 2015)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=auc1, color='gray', linestyle='--', alpha=0.5, label='baseline (globalna)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02c_strategie_imputacji.png")
plt.close()
print(f"  → Zapisano: {FIG_DIR}/02c_strategie_imputacji.png")


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
