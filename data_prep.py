"""
================================================================================
PROJEKT DATA MINING – weatherAUS
Osoba 2: Data Preparation & Feature Engineering
================================================================================

Zadania:
  1. Czyszczenie i obsługa braków (imputacja sezonowa, winsoryzacja)
  2. Inżynieria cech (sin/cos wiatrów, sezony, różnice ciśnień i wilgotności)
  3. Kodowanie kategorii (One-Hot Location, binaryzacja Yes/No)
  4. Eksperyment własny: 3 warianty obsługi Sunshine/Evaporation/Cloud (XGBoost)
  5. Podział czasowy train / val / test
  6. Eksport: X_train, X_val, X_test, y_train, y_val, y_test + pipeline

Wymagania:
  pip install pandas numpy scikit-learn xgboost joblib matplotlib

Uruchomienie:
  python data_prep.py
  Dane wyjściowe → ./prepared/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

# ─────────────────────────────────────────────────
DATA_PATH   = "weatherAUS.csv"
OUT_DIR     = "prepared"
FIG_DIR     = "figures"
TAB_DIR     = "tables"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(TAB_DIR,  exist_ok=True)

WIND_COLS   = ["WindGustDir", "WindDir9am", "WindDir3pm"]
CAT_COLS_NO_WIND = ["Location"]          # One-Hot
BIN_COLS    = ["RainToday"]              # Yes/No → 1/0
TARGET_COL  = "RainTomorrow"

HIGH_MISS   = ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"]

# Kolejność kierunków (co 22.5°)
WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

# Sezon na półkuli południowej (odwrócony)
def month_to_season(m):
    if m in [12, 1, 2]:   return "Summer"
    elif m in [3, 4, 5]:  return "Autumn"
    elif m in [6, 7, 8]:  return "Winter"
    else:                  return "Spring"


# ============================================================
# KROK 1 – WCZYTANIE DANYCH
# ============================================================
print("=" * 70)
print("1. WCZYTANIE DANYCH")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
print(f"Wczytano: {df.shape[0]:,} wierszy × {df.shape[1]} kolumn")


# ============================================================
# KROK 2 – USUNIĘCIE WIERSZY BEZ ZMIENNEJ CELU
# ============================================================
print("\n" + "=" * 70)
print("2. USUNIĘCIE WIERSZY BEZ RainTomorrow")
print("=" * 70)

before = len(df)
df = df.dropna(subset=[TARGET_COL]).copy()
removed = before - len(df)
print(f"Usunięto: {removed:,} wierszy ({removed/before*100:.2f}%)")
print(f"Pozostało: {len(df):,} wierszy")

# Binaryzacja celu
df["target"] = (df[TARGET_COL] == "Yes").astype(int)


# ============================================================
# KROK 3 – INŻYNIERIA CECH
# ============================================================
print("\n" + "=" * 70)
print("3. INŻYNIERIA CECH")
print("=" * 70)

# 3a. Cechy z daty
df["Year"]   = df["Date"].dt.year
df["Month"]  = df["Date"].dt.month
df["Season"] = df["Month"].apply(month_to_season)
print("  ✓ Year, Month, Season")

# 3b. Kierunki wiatru → sin / cos
for col in WIND_COLS:
    deg = df[col].map(WIND_DIR_MAP)          # NaN dla braków
    rad = np.deg2rad(deg)
    df[col + "_sin"] = np.sin(rad)
    df[col + "_cos"] = np.cos(rad)
    df.drop(columns=[col], inplace=True)
print(f"  ✓ sin/cos dla: {WIND_COLS}")

# 3c. Cechy syntetyczne
df["TempRange"]      = df["MaxTemp"]     - df["MinTemp"]
df["PressureChange"] = df["Pressure3pm"] - df["Pressure9am"]
df["HumidityChange"] = df["Humidity3pm"] - df["Humidity9am"]
print("  ✓ TempRange, PressureChange, HumidityChange")

# 3d. Binaryzacja RainToday
df["RainToday"] = (df["RainToday"] == "Yes").astype(float)   # float → NaN zachowane
print("  ✓ RainToday → 0/1")


# ============================================================
# KROK 4 – PODZIAŁ CZASOWY TRAIN / VAL / TEST
# ============================================================
print("\n" + "=" * 70)
print("4. PODZIAŁ CZASOWY")
print("=" * 70)

train_df = df[df["Year"] <= 2014].copy()
val_df   = df[df["Year"] == 2015].copy()
test_df  = df[df["Year"] == 2016].copy()
# Rok 2017 wyłączony z modelowania (niepełny)

for name, d in [("Train (≤2014)", train_df), ("Val (2015)", val_df), ("Test (2016)", test_df)]:
    yes_pct = d["target"].mean() * 100
    print(f"  {name:18s}: {len(d):>7,} obs.  |  RainTomorrow=Yes: {yes_pct:.1f}%")


# ============================================================
# KROK 5 – DEFINICJA KOLUMN DO PIPELINE
# ============================================================

# Kolumny numeryczne (po feature engineering, bez id-like)
DROP_COLS = ["Date", "RainTomorrow", "target", "Year", "Location", "Season"]
num_cols = [c for c in train_df.columns
            if c not in DROP_COLS
            and train_df[c].dtype in [np.float64, np.int64, float, int]]
cat_cols_ohe  = ["Location"]
cat_cols_ord  = ["Season"]    # Ordinal / OneHot – użyjemy OHE też tutaj


# ============================================================
# KROK 6 – IMPUTACJA SEZONOWA (per Location + Month)
# ============================================================
print("\n" + "=" * 70)
print("5. IMPUTACJA BRAKÓW")
print("=" * 70)

class SeasonalLocationImputer(BaseEstimator, TransformerMixin):
    """
    Imputacja mediany per (Location, Month).
    Fallback: globalna mediana z train.
    Działa tylko na kolumnach numerycznych przekazanych jako feature_names.
    """
    def __init__(self, num_cols):
        self.num_cols = num_cols

    def fit(self, X, y=None):
        # X to DataFrame z kolumnami Location, Month + num_cols
        self.global_med_  = X[self.num_cols].median()
        self.local_med_   = (
            X.groupby(["Location", "Month"])[self.num_cols]
            .median()
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.num_cols:
            mask = X[col].isna()
            if not mask.any():
                continue
            for idx in X[mask].index:
                loc = X.at[idx, "Location"]
                mon = X.at[idx, "Month"]
                try:
                    val = self.local_med_.loc[(loc, mon), col]
                    if pd.isna(val):
                        val = self.global_med_[col]
                except KeyError:
                    val = self.global_med_[col]
                X.at[idx, col] = val
        return X


class CategoricalModeImputer(BaseEstimator, TransformerMixin):
    """Imputacja mody per Location dla kolumn kategorycznych."""
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        self.global_mode_ = {c: X[c].mode()[0] for c in self.cat_cols}
        self.local_mode_  = {}
        for c in self.cat_cols:
            self.local_mode_[c] = X.groupby("Location")[c].agg(
                lambda x: x.mode()[0] if x.notna().any() else np.nan
            )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.cat_cols:
            mask = X[col].isna()
            if not mask.any():
                continue
            for idx in X[mask].index:
                loc = X.at[idx, "Location"]
                try:
                    val = self.local_mode_[col].loc[loc]
                    if pd.isna(val):
                        val = self.global_mode_[col]
                except KeyError:
                    val = self.global_mode_[col]
                X.at[idx, col] = val
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clipping na percentylach p_low i p_high (obliczonych na train)."""
    def __init__(self, num_cols, p_low=1, p_high=99):
        self.num_cols = num_cols
        self.p_low    = p_low
        self.p_high   = p_high

    def fit(self, X, y=None):
        self.lower_ = X[self.num_cols].quantile(self.p_low / 100)
        self.upper_ = X[self.num_cols].quantile(self.p_high / 100)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.num_cols:
            X[col] = X[col].clip(lower=self.lower_[col], upper=self.upper_[col])
        return X


# ─── Fit imputerów wyłącznie na train ───────────────────────
# Kolumny kategoryczne do imputacji (Season nie ma braków, RainToday już binarny)
cat_impute_cols = []   # sin/cos nie mają kategorii do imputacji

# Fit imputer numeryczny
num_imputer = SeasonalLocationImputer(num_cols)
num_imputer.fit(train_df)

winsorizer = Winsorizer(num_cols)
# Fit winsoryzera na danych po imputacji (używamy train)
train_imp = num_imputer.transform(train_df)
winsorizer.fit(train_imp)

# Transformacja wszystkich trzech zbiorów
train_imp = winsorizer.transform(train_imp)
val_imp   = winsorizer.transform(num_imputer.transform(val_df))
test_imp  = winsorizer.transform(num_imputer.transform(test_df))

# Raport braków po imputacji
still_missing = train_imp[num_cols].isna().sum().sum()
print(f"  Braki w train po imputacji numerycznej: {still_missing}")
print(f"  ✓ Imputacja sezonowa (Location + Month) + winsoryzacja 1–99 percentyl")


# ============================================================
# KROK 7 – KODOWANIE I STANDARYZACJA (scikit-learn pipeline)
# ============================================================
print("\n" + "=" * 70)
print("6. KODOWANIE I STANDARYZACJA")
print("=" * 70)

# Wybieramy kolumny wejściowe (bez metadanych i celu)
FEATURE_COLS = (
    num_cols
    + ["Location", "Season"]
    + [c for c in train_imp.columns if c.endswith("_sin") or c.endswith("_cos")]
)
# Usuwamy duplikaty (sin/cos mogą już być w num_cols jeśli dtype float)
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))
# Upewniamy się że Location i Season nie są w num_cols
FEATURE_COLS = [c for c in FEATURE_COLS if c in train_imp.columns]

# Podział na typy
final_num_cols = [c for c in FEATURE_COLS
                  if c not in ["Location", "Season"]
                  and train_imp[c].dtype in [np.float64, float, np.int64, int]]
final_cat_cols = ["Location", "Season"]

print(f"  Cechy numeryczne: {len(final_num_cols)}")
print(f"  Cechy kategoryczne: {final_cat_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), final_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), final_cat_cols),
    ],
    remainder="drop"
)

preprocessor.fit(train_imp[FEATURE_COLS])

def transform_set(df_imp):
    arr = preprocessor.transform(df_imp[FEATURE_COLS])
    # Nazwy kolumn
    ohe_names = preprocessor.named_transformers_["cat"] \
                             .get_feature_names_out(final_cat_cols).tolist()
    col_names = final_num_cols + ohe_names
    return pd.DataFrame(arr, columns=col_names, index=df_imp.index)

X_train = transform_set(train_imp)
X_val   = transform_set(val_imp)
X_test  = transform_set(test_imp)

y_train = train_imp["target"].values
y_val   = val_imp["target"].values
y_test  = test_imp["target"].values

print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}  |  y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}  |  y_test:  {y_test.shape}")
print(f"  ✓ StandardScaler + OneHotEncoder (ColumnTransformer)")


# ============================================================
# KROK 8 – EKSPERYMENT WŁASNY: 3 WARIANTY OBSŁUGI BRAKÓW
#          dla Sunshine / Evaporation / Cloud9am / Cloud3pm
# ============================================================
print("\n" + "=" * 70)
print("7. EKSPERYMENT WŁASNY – OBSŁUGA BRAKÓW W ZMIENNYCH Z HIGH MISS")
print("   Zmienne: Sunshine, Evaporation, Cloud9am, Cloud3pm")
print("   Model: XGBoost (domyślne hiperparametry)")
print("   Walidacja: rok 2015  |  Train: lata ≤ 2014")
print("=" * 70)

# Przygotowanie surowych zbiorów (bez pipeline'u OHE Location)
# Używamy uproszczonego zestawu cech żeby eksperyment był szybki i czytelny
SAFE_NUM = [c for c in num_cols if c not in HIGH_MISS]
ALL_NUM  = num_cols  # zawiera HIGH_MISS

def quick_impute(df_raw, fit_df, cols):
    """Globalna mediana z fit_df (szybka wersja na potrzeby eksperymentu)."""
    med = fit_df[cols].median()
    d = df_raw[cols].copy()
    for c in cols:
        d[c] = d[c].fillna(med[c])
    return d

def build_xgb_features(df_raw, fit_df, strategy):
    """
    Buduje macierz cech dla danego wariantu.
    strategy: 'impute' | 'native' | 'drop'
    sin/cos kolumny są już w df_raw (powstały w kroku inżynierii cech).
    """
    # Wszystkie numeryczne kolumny z df_raw (bez meta i celu)
    # SAFE_NUM już zawiera sin/cos jeśli ich dtype to float64
    all_avail = [c for c in df_raw.columns
                 if c not in ["Date", "RainTomorrow", "target", "Year",
                               "Location", "Season", "Month", "WindGustDir",
                               "WindDir9am", "WindDir3pm"]
                 and df_raw[c].dtype in [np.float64, float, np.int64, int]]

    safe_cols = [c for c in all_avail if c not in HIGH_MISS]
    high_cols  = [c for c in HIGH_MISS if c in df_raw.columns]

    if strategy == "drop":
        cols_use = safe_cols
    else:  # impute lub native
        cols_use = safe_cols + high_cols

    # Deduplikacja z zachowaniem kolejności
    seen = set(); cols_use = [c for c in cols_use if not (c in seen or seen.add(c))]

    X = df_raw[cols_use].copy().astype(float)
    med = fit_df[cols_use].median()

    if strategy == "impute":
        X = X.fillna(med)
    elif strategy == "drop":
        X = X.fillna(med)
    elif strategy == "native":
        # Imputujemy tylko safe (nie HIGH_MISS) – XGB sam obsłuży NaN
        X[safe_cols] = X[safe_cols].fillna(med[safe_cols])

    return X

results_exp = []
xgb_params = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
    eval_metric="auc",
    verbosity=0,
)

for strategy, label in [
    ("impute", "a) Imputacja medianą (globalna)"),
    ("native", "b) Natywna obsługa braków przez XGBoost"),
    ("drop",   "c) Usunięcie zmiennych z wysokim % braków"),
]:
    X_tr_exp = build_xgb_features(train_df, train_df, strategy)
    X_va_exp = build_xgb_features(val_df,   train_df, strategy)

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr_exp, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_va_exp)[:, 1])
    results_exp.append({
        "Wariant": label,
        "Cechy z HIGH_MISS": "Tak" if strategy != "drop" else "Nie",
        "Imputacja HIGH_MISS": "Tak" if strategy == "impute" else ("Natywna XGB" if strategy == "native" else "—"),
        "Liczba cech": X_tr_exp.shape[1],
        "AUC (val 2015)": round(auc, 4),
    })
    print(f"  [{strategy:6s}]  AUC = {auc:.4f}  ({label})")

exp_df = pd.DataFrame(results_exp)
exp_df.to_csv(f"{TAB_DIR}/eksperyment_braki_xgb.csv", index=False)
print(f"\n  → Tabela zapisana: {TAB_DIR}/eksperyment_braki_xgb.csv")

# Wybór wariantu
best_row = exp_df.loc[exp_df["AUC (val 2015)"].idxmax()]
print(f"\n  Najlepszy wariant: {best_row['Wariant']}  (AUC = {best_row['AUC (val 2015)']})")
print("  → Do użycia przez Osoby 3 i 4 jako punkt startowy.")

# Wykres eksperymentu
fig, ax = plt.subplots(figsize=(9, 4.5))
colors_exp = ["#5B9BD5", "#70AD47", "#ED7D31"]
aucs_exp   = exp_df["AUC (val 2015)"].tolist()
labels_exp = [r["Wariant"] for r in results_exp]
bars = ax.bar(labels_exp, aucs_exp, color=colors_exp, edgecolor="black")
ax.set_ylim(min(aucs_exp) - 0.005, max(aucs_exp) + 0.006)
ax.set_ylabel("AUC (zbiór walidacyjny 2015)")
ax.set_title("Eksperyment: strategie obsługi zmiennych z wysokim % braków\n"
             "(Sunshine, Evaporation, Cloud9am, Cloud3pm) – XGBoost",
             fontsize=11, fontweight="bold")
for bar, val in zip(bars, aucs_exp):
    ax.text(bar.get_x() + bar.get_width()/2, val,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(y=aucs_exp[0], color="gray", linestyle="--", alpha=0.5, label="baseline (imputacja)")
ax.legend()
ax.tick_params(axis="x", labelsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_eksperyment_braki_xgb.png")
plt.close()
print(f"  → Wykres: {FIG_DIR}/10_eksperyment_braki_xgb.png")


# ============================================================
# KROK 9 – EKSPORT ZBIORÓW
# ============================================================
print("\n" + "=" * 70)
print("8. EKSPORT ZBIORÓW")
print("=" * 70)

# CSV
X_train.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
X_val.to_csv(  f"{OUT_DIR}/X_val.csv",   index=False)
X_test.to_csv( f"{OUT_DIR}/X_test.csv",  index=False)
pd.Series(y_train, name="target").to_csv(f"{OUT_DIR}/y_train.csv", index=False)
pd.Series(y_val,   name="target").to_csv(f"{OUT_DIR}/y_val.csv",   index=False)
pd.Series(y_test,  name="target").to_csv(f"{OUT_DIR}/y_test.csv",  index=False)

# Pickle (szybszy import dla Osób 3 i 4)
joblib.dump(X_train,      f"{OUT_DIR}/X_train.pkl")
joblib.dump(X_val,        f"{OUT_DIR}/X_val.pkl")
joblib.dump(X_test,       f"{OUT_DIR}/X_test.pkl")
joblib.dump(y_train,      f"{OUT_DIR}/y_train.pkl")
joblib.dump(y_val,        f"{OUT_DIR}/y_val.pkl")
joblib.dump(y_test,       f"{OUT_DIR}/y_test.pkl")
joblib.dump(preprocessor, f"{OUT_DIR}/preprocessor.pkl")

print(f"  ✓ X_train / X_val / X_test  →  {OUT_DIR}/")
print(f"  ✓ y_train / y_val / y_test  →  {OUT_DIR}/")
print(f"  ✓ preprocessor.pkl          →  {OUT_DIR}/")
print(f"\n  Kolumny wyjściowe ({X_train.shape[1]}): {list(X_train.columns[:5])} ... (OHE Location + Season)")

# Podsumowanie cech
feat_summary = pd.DataFrame({
    "Cecha": X_train.columns,
    "Typ": ["numeryczna" if c in final_num_cols else "OHE" for c in X_train.columns],
    "Przykład_train_mean": X_train.mean().round(4).values,
})
feat_summary.to_csv(f"{TAB_DIR}/cechy_po_przetworzeniu.csv", index=False)
print(f"  ✓ Opis cech →  {TAB_DIR}/cechy_po_przetworzeniu.csv")


# ============================================================
# PODSUMOWANIE
# ============================================================
print("\n" + "=" * 70)
print("DATA PREPARATION ZAKOŃCZONE")
print("=" * 70)
print(f"""
Gotowe zbiory dla Osób 3 i 4:
  prepared/X_train.pkl  |  {X_train.shape[0]:,} × {X_train.shape[1]}
  prepared/X_val.pkl    |  {X_val.shape[0]:,} × {X_val.shape[1]}
  prepared/X_test.pkl   |  {X_test.shape[0]:,} × {X_test.shape[1]}

Wczytanie w kolejnych skryptach:
  import joblib
  X_train = joblib.load('prepared/X_train.pkl')
  y_train = joblib.load('prepared/y_train.pkl')
  # ... analogicznie val / test

Eksperyment braków → tables/eksperyment_braki_xgb.csv
Wykres eksperymentu → figures/10_eksperyment_braki_xgb.png
""")
