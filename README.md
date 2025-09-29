# Data Analysis for Network Traffic (CICIDS-2017)

**Purpose**  
Build an end-to-end **intrusion detection analytics pipeline** that transforms raw network traffic into **clean features**, evaluates multiple **ML classifiers** (LogReg, DT, RF, XGBoost, GNB), performs **dimensionality reduction** via PCA (accuracy vs. speed trade-offs), and **simulates real-time** monitoring by streaming predictions to **Power BI** dashboards.

**What this showcases**  
- **Data Engineering**: robust preprocessing (null/∞ handling, label encoding, leakage-safe scaling)  
- **Data Analysis**: EDA, correlation analysis, attack distributions, protocol analysis  
- **Machine Learning**: multi-model training & evaluation; confusion matrices; feature importance; metrics comparison  
- **Optimization**: **PCA** variance thresholds → **speed vs. accuracy** decisions for real-time scenarios  
- **Visualization**: Power BI **live dashboards** from a Python streaming pipeline  
- **Reproducibility**: notebook-driven workflow with saved artifacts (scalers/models)

**Key Outcomes**  
- Clear **model comparison** with XGBoost as a top performer (see Results)  
- **PCA@0.99** yields a strong latency/accuracy balance for “real-time” simulation  
- **Operational flow** from raw CSVs → features → model → streaming → dashboard

----
</br>

<!-- Hero visuals (from your repo's /images folder) -->
<p align="center">
  <img src="images/Distribution of Attack Type in CICIDS 2017 Dataset.png" alt="Attack distribution" width="48%" />
  <img src="images/Correlation Heatmap between attributes.png" alt="Correlation heatmap" width="48%" />
</p>

----

## System Overview

**Flow**
1) **Ingest** CICIDS-2017 daily CSVs  
2) **Preprocess**: drop IDs/IPs/timestamps → encode label → handle null/∞ → train/test split → fit **StandardScaler on train only**  
3) **Model**: train LR, DT, RF, XGB, GNB → evaluate (Accuracy, Precision, Recall, F1, MCC) → confusion matrices  
4) **Optimize**: evaluate **PCA** at variance thresholds (0.90/0.95/0.99/0.999) → select trade-off for real-time  
5) **Simulate real-time**: Python loop → preprocess + predict → POST to **Power BI** Streaming Dataset  
6) **Visualize**: live tiles & reports (daily trends, flagged events, heatmaps, KPIs)

### onceptual Diagram

```mermaid
flowchart LR
  A[CICIDS-2017 CSVs] --> B[Preprocessing: drop IDs and IPs, encode label, handle null and infinite, split, scale]
  B --> C[Model Training: LR, DT, RF, XGB, GNB]
  C --> D[Evaluation: metrics, confusion matrices, feature importance]
  D --> E[PCA Trade-offs: 0.90, 0.95, 0.99, 0.999]
  E --> F[Real-Time Simulation: Python streaming predictions]
  F --> G[Power BI Streaming Dataset]
  G --> H[Live Dashboards: KPIs, trends, heatmaps]
```

### Why This Design?

- Keeps **data leakage** out by scaling *after* the split  
- Compares **classical ML models** for interpretable baselines and production-friendly performance  
- Uses **PCA** only where it helps latency, not blindly everywhere  
- Separates **batch evaluation** from **streaming simulation** to mirror real ops  

---

<!-- Optional second visual for architecture section -->
<p align="center">
  <img src="images/Accuracy of Different Model.png" alt="Accuracy across models" width="60%" />
</p>

---

## 2) Dataset & Challenges

**Dataset: [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)**  
- One of the most widely used **Intrusion Detection System (IDS)** benchmarks.  
- Traffic captured over 5 days in 2017 at the University of New Brunswick.  
- **Mix of benign and malicious flows**: DDoS, PortScan, Botnet, Infiltration, Heartbleed, etc.  
- **80+ features**: packet counts, byte rates, flags, durations, header stats.  

### Key Properties
- **Multi-class labels** (`BENIGN` + multiple attack types)  
- **Highly imbalanced**: e.g., some attack types dominate, others are rare  
- **Mixed data types**: integers, floats, categorical (protocols)  
- **Dirty fields**: nulls, “Infinity” in rate features, non-informative IDs (Flow ID, IP addresses)  

---

### Distribution Examples

**Attack distribution**
<p align="center">
  <img src="images/Distribution of Attack Type in CICIDS 2017 Dataset.png" alt="Distribution of attack types" width="70%" />
</p>

**Protocol counts**
<p align="center">
  <img src="images/Count of Protocol.png" alt="Protocol counts" width="45%" />
  <img src="images/Count of Label across various Protocol.png" alt="Label counts across protocols" width="45%" />
</p>

---

### Data Cleaning Challenges

1. **Irrelevant IDs**  
   - `Flow ID`, `Source IP`, `Destination IP`, `Timestamp` → removed  
   - Why? They don’t generalize for detection, and may cause leakage.  

2. **Missing Values**  
   - Features like `Flow Bytes/s` had **nulls** → imputed (safe defaults / median).  

3. **Infinite Values**  
   - Rate-based features sometimes yielded **∞** (e.g., division by zero).  
   - Replaced with capped values / imputed safely.  

4. **Skewed Classes**  
   - Some attacks extremely under-represented.  
   - Approached as a **binary detection problem (Benign vs Malicious)** first.  

5. **Mixed Data Types**  
   - Encoded labels (`BENIGN=0`, `ATTACK=1`).  
   - Continuous features scaled using **StandardScaler** (fit only on train split).  

---

### Column Type Mix

<p align="center">
  <img src="images/Distribution of Column Types.png" alt="Distribution of column types" width="60%" />
</p>


---

## 3) Data Engineering Pipeline

Turning CICIDS-2017 raw CSVs into **ML-ready features** required a robust preprocessing pipeline.  
Every transformation was chosen to **improve generalization, avoid leakage, and stabilize training**.

---

### Step-by-Step Transformations

1. **Concatenate Daily Files**  
   - CICIDS-2017 is split across multiple CSVs (per day).  
   - Merged into a **single unified dataset** for consistency.  

2. **Drop Identifiers & Non-Informative Fields**  
   - Removed: `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`.  
   - *Reason*: These carry unique/session-specific values, not predictive patterns. Keeping them risks **data leakage**.

3. **Label Encoding**  
   - Converted `Label` → **binary classification** (`0 = Benign`, `1 = Attack`).  
   - *Reason*: Models perform better with numeric encoding, and binary framing avoids extreme class imbalance.

4. **Missing Value Handling**  
   - Columns like `Flow Bytes/s`, `Flow Packets/s` contained **NaNs**.  
   - Imputed using **median values** (robust against skew).  

5. **Infinity Replacement**  
   - Some flow features (rates, divisions) produced **∞** values.  
   - Strategy: Replace with column maximum or safe constant.  
   - *Reason*: Avoids math errors and ensures consistency.

6. **Train-Test Split (80/20)**  
   - Ensured **random shuffling** while preserving label balance.  
   - *Reason*: Avoid contamination of train/test.

7. **Scaling with StandardScaler**  
   - Fit **only on training data**, then applied to test.  
   - *Reason*: Prevents data leakage (test info influencing scaling).  
   - Scaling critical since models like Logistic Regression and XGBoost are sensitive to feature magnitudes.

---

### Preprocessing Pipeline (Conceptual)

```mermaid
flowchart TD
  A[Raw CICIDS-2017 CSVs] --> B[Concatenate Files]
  B --> C[Drop IDs and IPs]
  C --> D[Encode Labels]
  D --> E[Handle NaN and Infinite Values]
  E --> F[Train Test Split]
  F --> G[Fit StandardScaler on Train]
  G --> H[Transform Train and Test Features]
```

### Why This Matters

- **Leakage-safe scaling** → ensures real-world robustness  
- **Consistent handling of ∞ & NaN** → prevents crashes during streaming simulation  
- **Binary framing** → provides a strong first baseline; later extensible to multi-class attacks  
- **Reproducibility** → scaler & models saved under `Packages/` and `Models/` for reuse  

---

## 4) Exploratory Data Analysis (EDA)

Before modeling, we **interrogated the dataset** to uncover signal, bias, and anomalies.  
This ensured that models were trained on **meaningful, stable features** — not noise.

---

### 4.1 Distribution of Attack Types
- Dataset is **heavily imbalanced** — DDoS and PortScan dominate.  
- Rare classes (e.g., Heartbleed, Infiltration) appear only a handful of times.  
- Insight: Without rebalancing, models risk **bias towards majority classes**.  
- Solution: Start with **binary detection** (Benign vs. Attack) → reduces skew.

<p align="center">
  <img src="images/Distribution of Attack Type in CICIDS 2017 Dataset.png" alt="Distribution of Attack Types" width="70%" />
</p>

---

### 4.2 Flow Duration Patterns
- **Box plots** show how attack types differ in session length.  
- Example: **DDoS flows** are very short; **Infiltration flows** are long-lived.  
- *Interpretation*: Flow duration can be a **strong feature** for discriminating attacks.

<p align="center">
  <img src="images/Box Plot of Flow Duration for Various Attack Types.png" alt="Flow Duration Box Plot" width="70%" />
</p>

---

### 4.3 Daily Trends
- Averaged flow duration per day shows **temporal variability**.  
- Malicious traffic bursts occur on specific days → simulates **real-world attack windows**.

<p align="center">
  <img src="images/Average Flow Duration for Each Day.png" alt="Average Flow Duration Per Day" width="70%" />
</p>

---

### 4.4 Protocol & Flag Analysis
- Protocol usage skews: TCP dominates, UDP less frequent.  
- Attack traffic often manipulates flag distributions (SYN floods, abnormal flag counts).  
- **Flag counts by label** → reveals systematic **signature differences**.

<p align="center">
  <img src="images/Average of Various Flag Count for each Label.png" alt="Flag Count vs Label" width="70%" />
</p>

---

### 4.5 Feature Correlations
- High correlation among rate-based features → risk of **multicollinearity**.  
- Correlation heatmap confirms redundancies that PCA later exploits.  

<p align="center">
  <img src="images/Correlation Matrix between Attributes.png" alt="Correlation Matrix" width="45%" />
  <img src="images/Correlation Heatmap between attributes.png" alt="Correlation Heatmap" width="45%" />
</p>

---

### EDA Takeaways
- **Severe imbalance** → justify binary framing & careful evaluation metrics  
- **Certain features (duration, flags, byte rates)** carry strong predictive signal  
- **Redundancies in features** → motivates PCA for dimensionality reduction  
- **Temporal attack bursts** → simulation of “real-world-like” traffic feasible

---

## 5) Modeling Methodology

We designed a **systematic ML benchmarking study** on CICIDS-2017.  
The goal: compare **classical ML algorithms** with different **bias-variance trade-offs, interpretability, and computational efficiency**.

---

### 5.1 Models Selected & Rationale

| Model                   | Why It Was Chosen                                                   | Strengths                         | Weaknesses                           |
|--------------------------|---------------------------------------------------------------------|-----------------------------------|---------------------------------------|
| **Logistic Regression** | Simple linear baseline, interpretable coefficients                 | Transparent, fast, probabilistic  | Limited to linear decision boundaries |
| **Decision Tree**       | Nonlinear splits, interpretable rules                               | Easy to visualize, no scaling req | High variance (overfitting risk)      |
| **Random Forest**       | Ensemble of trees → reduces variance                                | Strong generalization, robust     | Slower training, less interpretable   |
| **XGBoost**             | Gradient boosting, regularized, highly competitive for tabular data | Excellent accuracy, handles skew  | More tuning required, less transparent|
| **Gaussian Naive Bayes**| Probabilistic baseline, fast                                        | Handles categorical-like features | Assumes feature independence          |

📌 **Key Idea:** Cover the spectrum — from interpretable (LogReg, DT) to ensemble (RF, XGB) to lightweight baseline (GNB).

---

### 5.2 Experimental Protocol

1. **Split:** 80/20 train-test  
2. **Scaling:** StandardScaler applied where needed (LR, GNB, XGB)  
3. **Metrics:** Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC)  
   - *Reason for MCC*: handles class imbalance better than accuracy alone  
4. **Confusion Matrices:** to visualize **false positives (FP)** vs **false negatives (FN)** trade-offs  
5. **Repeatability:** Each model + scaler object saved under `Packages and Models/`

---

### 5.3 Results: Accuracy Across Models

<p align="center">
  <img src="images/Accuracy of Different Model.png" alt="Accuracy comparison across models" width="60%" />
</p>

- **XGBoost dominates (~99% accuracy)** → robust, handles nonlinearity, strong against imbalance  
- **Random Forest close second (~97%)** → strong ensemble baseline  
- **Logistic Regression still competitive (~94%)** → interpretable linear model  
- **Decision Tree (~96%)** → useful, but prone to overfitting  
- **Naive Bayes (~88%)** → weakest, due to violated independence assumptions

---

### 5.4 Confusion Matrices (Error Profiles)

Confusion matrices reveal *how* errors are distributed — critical in IDS, where **false negatives (missed attacks)** are far more dangerous than false positives.

<p align="center">
  <img src="images/Confusion Matrix of Logistic Regression.png" alt="LogReg Confusion" width="45%" />
  <img src="images/Confusion Matrix of Decision Tree Classifier.png" alt="DT Confusion" width="45%" />
</p>
<p align="center">
  <img src="images/Confusion Matrix of Random Forest Classifier.png" alt="RF Confusion" width="45%" />
  <img src="images/Confusion Matrix of XGBoost Classifier.png" alt="XGB Confusion" width="45%" />
</p>
<p align="center">
  <img src="images/Confusion Matrix of Gaussian Naive Bayes.png" alt="GNB Confusion" width="45%" />
</p>

**Observations**
- **Logistic Regression**: high FP (flags benign traffic incorrectly)  
- **Decision Tree**: overfits some classes → inconsistent FP/FN balance  
- **Random Forest**: strong across all classes, lower variance than single tree  
- **XGBoost**: near-perfect separation; lowest FN (best at catching attacks)  
- **Naive Bayes**: many misclassifications due to independence assumption

---

### 5.5 Takeaways
- **XGBoost chosen as primary model** for deployment  
- **RF** provides a simpler ensemble fallback  
- **LogReg** provides interpretable insights (coefficients) for feature-level reasoning  
- **DT/GNB** used mainly for baselines and comparative study

---

## 6) Feature Importance & Interpretability

High predictive accuracy alone isn’t enough in **network intrusion detection**.  
We need to know **which features drive predictions** — for transparency, debugging, and trust in deployment.

---

### 6.1 Logistic Regression Coefficients
- Linear model → coefficients show **direction & magnitude** of feature impact.  
- Example: Longer **Flow Duration** increases probability of malicious label.  
- Useful for **interpretable baselines** and feature sanity-checks.  

<p align="center">
  <img src="images/Feature Importance from Logistic Regression.png" alt="LogReg Coefficients" width="60%" />
</p>

---

### 6.2 Decision Tree Importance
- Splits highlight **which features best separate benign vs attack traffic**.  
- Provides **rule-based intuition** (e.g., “if flow duration < threshold → benign”).  

<p align="center">
  <img src="images/Feature Importance from Decision Tree.png" alt="DT Importance" width="60%" />
</p>

---

### 6.3 Random Forest Importance
- Aggregates many trees → robust measure of feature relevance.  
- Helps reduce overfitting bias from single trees.  
- Confirms consistency: flow duration, byte rates, flag counts rank highly.  

<p align="center">
  <img src="images/Feature Importance from Random Forest.png" alt="RF Importance" width="60%" />
</p>

---

### 6.4 XGBoost Importance (Winner Model)
- Provides **multiple importance metrics** (Gain, Cover, Frequency).  
- **Gain** shows which features most improve split quality.  
- Here, **Flow Duration, Destination Port, Packet Length Mean** dominate.  
- Security Insight: Attack flows often exploit **specific ports** and **abnormal packet lengths**.

<p align="center">
  <img src="images/Feature Importance from XGBoost.png" alt="XGB Importance" width="60%" />
</p>

---

### Takeaways
- **Flow Duration** is consistently critical → many attacks differ in session length.  
- **Destination Port** patterns emerge in XGBoost → port-based exploits common.  
- **Packet Length Mean / Byte Rates** strongly influence detection → consistent with flooding attacks.  
- Feature importance analysis validates that models are not just memorizing noise — they’re leveraging meaningful network properties.
---

## 7) PCA & Performance Optimization

With >80 features, many of them highly correlated (see heatmaps), we explored **Principal Component Analysis (PCA)** to reduce dimensionality.

---

### 7.1 Why PCA?
- Many CICIDS-2017 features are **redundant** (correlated rates, byte counts).  
- High dimensionality increases:
  - **Training time** (slower iteration cycles)  
  - **Risk of overfitting**  
  - **Deployment latency** (bad for real-time IDS)  
- PCA can compress features into a smaller set of **orthogonal components** that capture most variance.

---

### 7.2 Variance Threshold Experiments
We tested PCA with thresholds:

- **0.90 (90%) variance**  
- **0.95 (95%) variance**  
- **0.99 (99%) variance**  
- **0.999 (99.9%) variance**

📊 Impact on metrics across thresholds:

<p align="center">
  <img src="images/Impact of Variance Threshold on Metrics.png" alt="PCA Variance Threshold Impact" width="70%" />
</p>

---

### 7.3 Performance Metrics Across Thresholds
- Accuracy, Precision, Recall, F1 measured for each PCA configuration  
- Runtime performance also tracked

<p align="center">
  <img src="images/PERFORMANCE METRICS FOR VARIANCE THRESHOLD.png" alt="PCA Performance Metrics" width="70%" />
</p>

---

### 7.4 With vs Without PCA
Direct comparison:

- **Without PCA:** Slightly higher accuracy, but slower training/inference  
- **With PCA (0.99):**  
  - Accuracy drop = **~0.18% only**  
  - Training speed-up = **~27% faster**  
  - Much lighter model for real-time simulation

<p align="center">
  <img src="images/Peformance Metrics Comparison with and without PCA.png" alt="PCA vs No PCA Comparison" width="70%" />
</p>

---

### Takeaway
- **PCA @ 0.99 variance** chosen for real-time simulation → optimal trade-off  
- Achieved **27% faster training** while keeping accuracy nearly identical  
- This mirrors real-world ML trade-offs:
  - Batch analysis (offline) → full features  
  - Real-time IDS (online) → PCA-compressed features for low latency

---

## 8) Real-Time Simulation Design

After training and validating models, we built a **simulation pipeline** to mimic how an Intrusion Detection System (IDS) would run in production.  
Instead of just reporting static metrics, the system generates **real-time predictions** and pushes them into a **Power BI Streaming Dataset** for live dashboards.

---

### 8.1 Operational Flow

1. **Sample Traffic Flow**
   - Randomly selects records from the CICIDS-2017 dataset
   - Mimics live arrival of new traffic events

2. **Preprocessing**
   - Apply the same pipeline: drop IDs → impute → scale → (PCA at 0.99 variance)

3. **Prediction**
   - Run through **XGBoost model** (winner from benchmarking)
   - Classify as `Benign (0)` or `Attack (1)`

4. **Stream to Power BI**
   - Use Python script to **POST results to Power BI REST API**
   - Power BI stores records in a **Streaming Dataset**

5. **Live Dashboards**
   - Visual tiles auto-refresh with near real-time latency
   - KPIs, daily trends, attack heatmaps, and alert counts visible instantly

---

### 8.2 Flow Diagram

```mermaid
flowchart TD
    A[Sampled Flow from CICIDS Dataset] --> B[Preprocessing Pipeline]
    B --> C[XGBoost Prediction]
    C --> D[POST JSON Payload to Power BI API]
    D --> E[Power BI Streaming Dataset]
    E --> F[Real-Time Dashboard: KPIs, Trends, Alerts]
```

### 8.3 Example JSON Payload (to Power BI API)
```json
{
  "flow_id": "123456",
  "duration": 2.13,
  "protocol": "TCP",
  "prediction": "Attack",
  "timestamp": "2025-09-29T12:34:56Z"
}
```
---

### 8.4 Dashboard Screens

**Realtime Streaming Flow**

---
<p align="center"> <img src="images/Average of Various Flag Count for each Label.png" alt="Realtime Flow Simulation" width="65%" /> </p>

---

**Daily Trends of Attack Traffic**

---

<p align="center"> <img src="images/Average Flow Duration for Each Day.png" alt="Daily Trends Dashboard" width="70%" /> </p>

---

**Power BI Live Dashboard**

---

<p align="center"> <img src="images/Count of Protocol.png" alt="Power BI Dashboard Example" width="45%" /> <img src="images/Count of Label across various Protocol.png" alt="Power BI Dashboard Example 2" width="45%" /> </p>

---

### Takeaway

- Built a true end-to-end pipeline: from offline ML → real-time prediction → streaming visualization.

- Demonstrates skills in data engineering + ML + visualization integration.

- Mirrors production-grade IDS workflows: detection → alerting → monitoring.
