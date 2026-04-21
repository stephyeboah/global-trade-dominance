# 🌍 Global Trade Dominance — Data Science Lab Project

> **Clustering & Time Series Forecasting Project** | Master's in Data Science  
> Analyzing and forecasting the evolution of global trade dominance using World Bank data (2006–2024)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-4B8BBE?style=for-the-badge)
![Prophet](https://img.shields.io/badge/Facebook%20Prophet-0668E1?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

In 2000, the USA and the EU dominated global trade. By 2024, China had overtaken both to become the largest trading partner for nearly all of Asia, Africa, and South America. This project investigates the **evolution of trade dominance** among leading economies and forecasts future trade trajectories.

The pipeline combines **unsupervised clustering** to group countries by macroeconomic profile, followed by **time series forecasting** on two representative economies — Ghana and Brazil — using SARIMAX and Facebook Prophet.

**Research question:** How will economies sustain or lose their trade dominance in the next decade, and which economic factors will prove decisive?

---

## 👥 Team & Roles

| Name | Role |
|------|------|
| **Stephen Adu Poku Yeboah** | Data preprocessing (GEM dataset), SARIMAX modelling & forecasting |
| Emmanuel Ampah | Data preprocessing (Global Economy Indicators), EDA, Prophet modelling,  SARIMAX modelling |
| Jan Zanini | Clustering analysis, results interpretation |

---

## 🗂️ Data Sources

| Source | Description |
|--------|-------------|
| [World Bank Global Economic Monitor (GEM)](https://datacatalog.worldbank.org) | Monthly/quarterly trade indicators for 200+ countries (2006–2024): exports, imports, CPI, exchange rates, reserves |
| [Kaggle Global Economy Indicators](https://www.kaggle.com) | Annual macroeconomic indicators for 180+ countries: GDP, trade flows, inflation, FDI, unemployment |

---

## 🔧 Pipeline Architecture

```
┌──────────────────────────────────┐
│  Global Economy Indicators       │  ← Clustering features
│  (Kaggle - annual)               │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  Feature Construction            │
│  (scale, dynamics, structure,    │
│   trade openness, FX metrics)    │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  Preprocessing & Clustering      │
│  KNN Imputer + RobustScaler +    │
│  Winsorizing → K-Means (K=2)     │
└────────────────┬─────────────────┘
                 │
          Select representatives:
          Ghana (Cluster 0)
          Brazil (Cluster 1)
                 │
                 ▼
┌──────────────────────────────────┐
│  GEM Time Series Data            │  ← Forecasting features
│  (World Bank - monthly)          │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  GEM Preprocessing               │
│  IterativeImputer + Box-Cox +    │
│  Outlier detection & treatment   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  SARIMAX + Facebook Prophet Forecasting       │
│  Train: 2006–2020  |  Test: 2021–2024        │
│  Forecast horizon: 24 months                 │
└──────────────────────────────────────────────┘
```

---

## 🔍 My Contribution

### 1. GEM Dataset Preprocessing
The World Bank GEM dataset required a careful preprocessing pipeline before any forecasting could be applied. I was responsible for the full preprocessing of this time series dataset:

**Missing Value Imputation — IterativeImputer**
- Used `IterativeImputer` (scikit-learn) with a Random Forest estimator — a multivariate approach that predicts each missing feature using all other features iteratively until convergence
- Chosen over simple mean/median imputation to preserve relationships between variables (e.g. between exports and reserves)
- Validated by checking that distributions remained stable after imputation

**Outlier Detection & Treatment**
Used three complementary approaches to identify anomalous values in the export time series:
- **Z-score** — statistical deviation from the mean
- **IQR (Interquartile Range)** — boxplot-based detection
- **Isolation Forest** — machine learning-based anomaly detection

Treatment strategy depended on outlier density:
- `< 2%` outliers → interpolation
- `< 5%` outliers → rolling median
- `≥ 5%` outliers → winsorization

**Series Transformation**
- Applied **Box-Cox transformation** to stabilize variance, normalize skewed distributions, and linearize exponential trends in export/import data — essential for meeting SARIMAX residual assumptions

---

### 2. SARIMAX Modelling & Forecasting

I built and tuned the **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) models for both Ghana and Brazil.

**Parameter Selection — Grid Search**
- Tested 64 parameter combinations across ARIMA(p,d,q) and seasonal SARIMA(P,D,Q,s=12) orders
- Selected best model using **AIC (Akaike Information Criterion)**
- Best model after outlier treatment and Box-Cox: `order=(1,1,1)`, `seasonal_order=(0,1,1,12)`, AIC = −537.04

**Model Diagnostics**
- Verified residuals fluctuate randomly around zero (no systematic patterns)
- Checked correlogram for absence of significant autocorrelation
- Q-Q plot and Jarque-Bera test confirmed near-normality with minor tail deviations

**Monte Carlo Simulation**
- Generated probabilistic forecast ranges using Monte Carlo simulation on SARIMAX parameters to provide more stable confidence intervals

---

## 📊 Key Results

### Clustering
- **K = 2** produced the highest silhouette score (> 0.5), clearly separating developed/large emerging economies from smaller developing ones
- K-Means selected over K-Medoids (ARI = 0.022 between them — very different partitions; K-Means offered better interpretability)
- **Cluster 1 (orange):** dominant trade powers — USA, Germany, China, Japan, etc.
- **Cluster 0 (blue):** developing/emerging economies — Ghana, most of Africa and parts of Asia

### Forecasting — Ghana
| Model | Projection |
|-------|-----------|
| SARIMAX | Continued volatility, median ~1.25–1.5B USD, wide confidence interval |
| Monte Carlo | More stable: median ~1.5B USD, tighter interval |
| Prophet | Clear upward trend: ~1.5B → ~1.7B USD by end of 2027 |

### Forecasting — Brazil
| Model | Projection |
|-------|-----------|
| SARIMAX | Sharp decline from recent highs, stabilizing at 15–17B USD |
| Monte Carlo | Confirms decline but with constrained range |
| Prophet | Optimistic: gradual rise from ~25B → 30B+ USD by 2028 |

> 💡 **Key insight:** Ghana shows a clear, resilient upward trajectory; Brazil faces high uncertainty and volatility — highlighting the shifting balance of global trade.

---

## 📁 Repository Structure

```
global-trade-dominance/
├── README.md
├── clustering.ipynb                  # K-Means, K-Medoids, silhouette, dendrogram, world map
├── Copia_di_Project.ipynb            # GEM preprocessing, SARIMAX, Prophet forecasting
├── Copia_di_creazioneDataset         # Dataset creation and merging script
└── GLOBAL_TRADE_DOMINANCE_Report.pdf # Full project report
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn statsmodels prophet matplotlib seaborn scipy
```

### Clustering
```bash
jupyter notebook clustering.ipynb
```

### Forecasting (SARIMAX + Prophet)
```bash
jupyter notebook Copia_di_Project.ipynb
```

---

## 🛠️ Technologies

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | KNN Imputer, IterativeImputer, RobustScaler, K-Means, K-Medoids |
| statsmodels | SARIMAX modelling, residual diagnostics |
| Facebook Prophet | Trend + seasonal decomposition forecasting |
| matplotlib / seaborn | Visualizations, dendrograms, world map, ROC/lift |
| scipy | Box-Cox transformation, statistical tests |

---

## ⚠️ Known Limitations

- Many countries had significant missing data in the GEM dataset, requiring careful imputation
- SARIMAX performed well in-sample but showed wide confidence intervals for volatile economies like Brazil
- Prophet projects optimistic long-term trends but is less sensitive to recent extreme fluctuations
- FIA regulation-style unpredictable external events (trade wars, pandemics) cannot be modelled

---

## 📄 License

Academic project — Università degli Studi di Milano-Bicocca, Data Science MSc. Data sourced from the World Bank GEM and Kaggle Global Economy Indicators.
