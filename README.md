# Spotify Churn — EDA & Baseline ML

A simple, beginner-friendly project to predict user churn (`is_churned`) on a Spotify-like dataset.
---

## Where to get the data

> **Note:** The dataset is **not** stored in this repo (it’s ignored by `.gitignore`).

- **Data source:** https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data

- **Expected local paths (adjust in notebooks if needed):**
      data/raw/...
      data/processed/spotify_churn_dataset_clean.parquet
---
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
---

## Data description (columns)

- **user_id** - user identifier (ID).
- **age** - age (years).
- **listening_time** - average daily listening time (minutes/day).
- **songs_played_per_day** - number of songs played per day.
- **skip_rate** - share of skipped tracks (0–1).
- **ads_listened_per_week** - number of ads per week.
- **offline_listening** - offline playback available (True/False).
- **subscription_type** - plan type (e.g., Free, Premium, Family…).
- **device_type** - device category (Desktop / Mobile / Web).
- **country** - country code.
- **gender** - categorical gender field.
- **is_churned** - **target**: whether the user churned (0/1).

> For modeling we don’t use **`user_id`**.  
> **`offline_listening`** strongly depends on **`subscription_type`** / ads.

---
### 1) EDA - `notebook/01_eda.ipynb`
- Quick sanity & dtypes: preview dataset (shape/head/info), fix types (numeric/boolean/category), handle missing values & duplicates
- Basic stats (mean/median/std, min–max, quartiles, IQR).
- Histograms & boxplots; outliers via Tukey’s rule.
- Categorical summaries: counts & percentages (`value_counts`), **churn %** per category.
- Dependence checks:
- numeric↔numeric correlations (heatmap),

### 2) Baseline ML - `notebook/02_ml.ipynb`
- Preprocessing pipeline with `ColumnTransformer`:
- `StandardScaler` for numeric features,
- `OneHotEncoder(drop='first')` for categorical features.
- Validation: `StratifiedKFold(n_splits=5)`.
- Models: `LogisticRegression`, `KNeighborsClassifier`, `SVC (rbf)`,  
`GaussianNB`, `DecisionTree`, `RandomForest`, `XGBClassifier`.
- Metrics: `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (CV + test).
- Visuals: confusion matrices per model; optional Yellowbrick (ROC/PR, classification report).

---

## How to run

```bash
# in the project folder
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Open the notebooks:
jupyter lab    # or use VS Code
