# Drift Taxonomy Engine — Complete Explanation

## What Is This Project?

This project builds a **Drift Taxonomy Engine** — an automated system that detects when a machine learning model's data changes in production, figures out **what type** of change it is, and tells you **exactly what to do about it**.

We use a **credit card fraud detection** model as the real-world case study.

---

## The Problem We Are Solving

### The Real-World Scenario

You train a fraud detection model. It works great (95% recall). You deploy it to production. Three months later, it's catching only 40% of fraud. Millions of dollars are being stolen.

**Why did this happen?** The data changed. But HOW did it change?

### Why This Matters

Every ML model in production eventually faces **data drift** — the data it receives starts looking different from the data it was trained on. This is the #1 reason production ML models fail silently.

The problem is: **not all drift is the same**, and **not all drift requires the same response**.

If you treat all drift the same way, you either:
- Waste money retraining models that don't need it
- Miss dangerous drift that requires immediate action
- Retrain on corrupted data (making things worse)

---

## What Is Data Drift?

**Data drift** = the statistical properties of the data your model receives in production change over time, compared to what the model was trained on.

Think of it like this: You trained your model to recognize "normal" transactions. But the world changed — what's "normal" now looks different from what was "normal" 6 months ago.

---

## The Three Types of Drift (Our Taxonomy)

### 1. Covariate Drift (Input Distribution Shift)

**What it means:** The input features changed, but the relationship between features and the target is still the same.

**Real example:** Average transaction amounts went from $50 to $150 due to inflation. The model sees bigger numbers, but the RULE for detecting fraud hasn't changed — fraud is still "unusually large transactions relative to the user's history."

**Analogy:** Imagine a thermometer that's calibrated for Celsius. If you move from a cold country to a hot country, the temperature readings are higher (input shifted), but the thermometer's LOGIC for measuring temperature is still correct. It just needs a slight recalibration.

**Detection method:** Statistical tests (KS test, PSI) on individual features. If the distributions shifted significantly, it's covariate drift.

**Key property:** Model performance might drop slightly, but the model's learned patterns are still fundamentally correct.

---

### 2. Concept Drift (The Relationship Changed)

**What it means:** The relationship between features and the target variable has changed. What used to be fraud is no longer fraud, or new fraud patterns emerged that look nothing like the old ones.

**Real example:** Fraudsters evolved. They used to steal credit cards and make huge purchases. Now they use sophisticated social engineering to make many small, normal-looking transactions. The old model's rules (look for big unusual purchases) are now WRONG — they'll miss this new fraud entirely.

**Analogy:** Imagine you trained a model to predict "is this email spam?" based on patterns from 2020. In 2020, spam had misspellings and suspicious links. By 2024, spam uses perfect AI-generated text. The CONCEPT of "what spam looks like" fundamentally changed. Your old rules are useless now.

**Detection method:** Monitor model PERFORMANCE (recall, precision, F1). If performance drops significantly while input data looks normal, concept drift is happening. The model is confidently making wrong predictions.

**Key property:** This is the MOST DANGEROUS type because:
- Feature monitoring tools like Evidently **cannot detect it** (inputs may look fine)
- The model doesn't "know" it's wrong — it still outputs confident predictions
- It's invisible until you compare predictions against ground truth labels

---

### 3. Pipeline Drift (Data Quality Degradation)

**What it means:** Something broke in the data pipeline. Features are corrupted, missing, or garbage — not because the world changed, but because an engineer broke something upstream.

**Real example:** A new hire accidentally pushed code that broke the feature extraction pipeline. Now the "transaction_frequency" feature returns NULL for 30% of users, and another feature is stuck at 0.0 because the API it depends on is down.

**Analogy:** Your thermometer isn't wrong about the temperature, and the temperature didn't change. Someone just poured coffee on the thermometer and now it's giving random readings. The fix isn't to "retrain" the thermometer — it's to clean it.

**Detection method:** Check for anomalies in data quality — sudden spikes in missing values, features stuck at constant values, values outside the expected range, sign changes in features that should always be positive.

**Key property:** The model isn't wrong, and the world didn't change. The DATA is just broken. **DO NOT retrain on this data** — you'd be teaching the model to work with garbage inputs.

---

## Why Existing Tools Don't Solve This

| Tool | What It Does | What It MISSES |
|------|-------------|----------------|
| **Evidently AI** | Detects feature distribution shifts (covariate drift) | Cannot detect concept drift. Flags too many false positives at scale. |
| **NannyML** | Estimates performance without labels | Still requires careful threshold tuning. No automated response. |
| **Great Expectations** | Data quality checks (pipeline drift) | No ML awareness. Doesn't know about feature importance. |
| **Basic monitoring** | Alert when metric drops | No classification of WHY. No specific remediation guidance. |

**The gap:** No existing tool classifies the drift type AND recommends the specific correct action. They just say "drift: yes/no" and leave you to figure out the rest.

---

## Our Solution: The Drift Taxonomy Engine

### What It Does (Step by Step)

```
INPUT:                                     OUTPUT:
───────                                    ───────
• Current batch of data        ──────→     • Drift Type: CONCEPT
• Reference data (training)    ──────→     • Severity: CRITICAL
• Model performance metrics    ──────→     • Action: FULL_RETRAIN
                                           • Confidence: 0.92
                                           • Top signals: [...]
```

### The Engine's Internal Logic

#### Step 1: Collect Signals

The engine runs multiple checks in parallel:

```
Check pipeline health    → missing values? range violations? constant features?
Check feature shifts     → KS test + Cohen's d on each feature distribution
Check target shift       → chi-squared on target variable distribution
Check model performance  → compare current recall/precision to baseline
```

#### Step 2: Classify Drift Type

```python
# Simplified decision logic:

if performance_dropped_significantly (>5%):
    drift_type = "CONCEPT"
    # The model is WRONG — old rules don't work anymore

elif missing_values_spiked OR range_violations OR constant_features:
    drift_type = "PIPELINE"
    # Data is BROKEN — something upstream failed

elif statistical_test_significant AND effect_size_large:
    drift_type = "COVARIATE"
    # Inputs shifted — model is still mostly correct but needs tuning

elif multiple_strong_signals:
    drift_type = "MIXED"
    # Multiple things happening at once
```

#### Step 3: Score Severity (0 to 1)

Not all drift is equally dangerous. The engine weighs drift by **feature importance**:

- Drift in `V14` (most important feature for fraud detection) = HIGH severity
- Drift in `V28` (barely used by model) = LOW severity

```
severity_score = feature_importance × drift_magnitude

Then map to levels:
< 0.05  → NONE
< 0.15  → LOW  
< 0.40  → MEDIUM
< 0.70  → HIGH
≥ 0.70  → CRITICAL
```

Concept drift is automatically elevated to HIGH+ because it means the model is actively wrong.

#### Step 4: Recommend Action (The Research Contribution)

This is the novel part. Based on drift type × severity, the engine recommends a **specific** action:

| Drift Type \ Severity | LOW | MEDIUM | HIGH | CRITICAL |
|---|---|---|---|---|
| **Covariate** | monitor | alert | investigate | **incremental_retrain** |
| **Concept** | alert | investigate | **full_retrain** | **full_retrain** |
| **Pipeline** | alert | investigate | **block** | **block** |
| **Mixed** | alert | investigate | **full_retrain** | **block** |

---

## The Key Insight: Incremental vs. Full Retraining

This is the core research contribution. Instead of a generic "retrain" action, we distinguish:

### Incremental Retraining (for Covariate Drift)

**When:** Input distributions shifted, but the model's learned rules are still valid.

**What to do:** Fine-tune the existing model on recent data. Keep the learned weights/patterns and just adjust the decision boundaries slightly.

**Why this works:** The model already knows "what fraud looks like." It just needs to learn "what fraud looks like at the NEW normal transaction amounts."

**Cost:** Fast (minutes). Low risk. Preserves all existing knowledge.

**Example:** Inflation caused all amounts to increase 2x. Fine-tune the model on last month's data so it learns the new baseline.

---

### Full Retraining (for Concept Drift)

**When:** The fundamental relationship between features and the target changed. The model's learned rules are now WRONG.

**What to do:** Throw away the old model entirely. Collect fresh labeled data. Train a new model from scratch.

**Why fine-tuning WON'T work:** If the concept changed, fine-tuning would try to preserve old patterns that are now wrong. You'd end up with a model that's confused between old and new rules.

**Cost:** Slow (hours/days). Expensive. Requires fresh labeled data.

**Example:** Fraudsters completely changed their attack strategy. The old patterns are useless. You need new labeled examples of the NEW type of fraud to learn from scratch.

---

### Block (for Pipeline Drift)

**When:** Data is corrupted or broken.

**What to do:** STOP the pipeline. Don't make predictions. Don't retrain. Fix the data source first.

**Why retraining is DANGEROUS here:** If you retrain on corrupted data, you teach the model that garbage inputs are normal. This poisons the model permanently.

**Cost:** Immediate operational disruption, but prevents permanent damage.

**Example:** A feature is returning NULL for 30% of users due to a broken API. If you retrain, the model learns "NULL is normal" and will make terrible predictions even after the API is fixed.

---

## How We Validate This Works

### Test Scenarios in the Notebook

We simulate 4 real-world scenarios and run the engine on each:

| Scenario | What We Simulate | Expected Engine Output |
|----------|-----------------|----------------------|
| **Clean data** | No drift at all | Type: NONE, Action: NONE |
| **Covariate drift** | Increase amounts 3x, shift V1/V14 | Type: COVARIATE, Action: INCREMENTAL_RETRAIN |
| **Concept drift** | Degrade model performance to 45% recall | Type: CONCEPT, Action: FULL_RETRAIN |
| **Pipeline drift** | Inject 15% NaN, constant columns, range violations | Type: PIPELINE, Action: BLOCK |

### Streaming Simulation (Production-Like)

We simulate 10 time windows where drift gradually increases:
- Windows 0-2: Normal operations (no drift)
- Windows 3-5: Gradual covariate drift (amounts increasing)
- Windows 6-7: Concept drift kicks in (model starts failing)
- Windows 8-9: Pipeline corruption on top of everything

The engine correctly escalates its recommendations over time: MONITOR → ALERT → INVESTIGATE → INCREMENTAL_RETRAIN → FULL_RETRAIN → BLOCK.

---

## Project Architecture

```
drift-taxonomy-engine/
├── data/
│   └── creditcard.csv              # 284,807 credit card transactions
├── notebooks/
│   ├── 01_baseline_fraud_detection.ipynb   # EDA + understand the data
│   ├── 02_baseline_model_training.ipynb    # Train Random Forest (95%+ recall)
│   └── 03_drift_simulation_detection.ipynb # THE MAIN NOTEBOOK:
│       │                                     Part I: Simulate & detect drift with Evidently
│       │                                     Part II: Build the Drift Taxonomy Engine
│       │                                     Part III: Streaming monitor + production API
│   └── notebook_artifacts/          # Saved results (JSON, CSV, PKL)
├── src/                             # (Future: extract engine as importable module)
└── requirements.txt
```

---

## The Full Pipeline (What Each Notebook Does)

### Notebook 01: Baseline Fraud Detection (EDA)
- Load credit card dataset (284K transactions, 492 fraud cases)
- Understand class imbalance (0.17% fraud rate)
- Feature analysis (PCA-transformed V1-V28 + Amount + Time)
- Identify which features are most correlated with fraud

### Notebook 02: Model Training
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Handle class imbalance with SMOTE
- Select best model (Random Forest, ~95% recall)
- Extract feature importances (used by the engine for severity weighting)

### Notebook 03: Drift Simulation & Taxonomy Engine
**Part I — Understand Drift Detection:**
- Simulate covariate drift (shift feature distributions)
- Simulate concept drift (degrade model performance)
- Simulate pipeline drift (inject missing values, corrupt data)
- Use Evidently to detect drift (show its strengths and limitations)
- Compare gradual vs. sudden drift

**Part II — Build the Taxonomy Engine (Our Contribution):**
- Define data structures (DriftType, Severity, Action enums)
- Build `DriftTaxonomyEngine` class with:
  - `_check_pipeline_drift()` — missing values, range violations, constant features
  - `_check_covariate_drift()` — KS test + Cohen's d (requires BOTH statistical AND practical significance)
  - `_check_target_drift()` — chi-squared on target distribution
  - `_check_concept_drift()` — performance decay detection
  - `_classify_drift_type()` — priority-based classification (concept > pipeline > covariate)
  - `_score_severity()` — feature-importance-weighted severity scoring
  - `_recommend_action()` — the novel action matrix with granular retraining

**Part III — Production Readiness:**
- Streaming drift monitor (10 time windows simulating production)
- Automated response playbooks (step-by-step for each action level)
- Production API design (how this would be deployed as a service)
- Configuration export (JSON configs for deployment)

---

## What Makes This a Research Contribution

### The Gap in Existing Literature

| What Exists | What's Missing |
|---|---|
| Papers on drift detection algorithms | No systematic classification framework |
| Tools that detect drift (Evidently, NannyML) | No automated action recommendation |
| Generic "retrain when drift detected" advice | No distinction between retrain strategies |
| Individual drift type studies | No unified engine handling all types |

### Our Contribution

1. **Unified taxonomy** — One engine that classifies covariate, concept, and pipeline drift
2. **Severity scoring weighted by feature importance** — A drift in an important feature matters more
3. **Granular action matrix** — Specifically distinguishes incremental_retrain vs. full_retrain vs. block
4. **Automated response playbooks** — Step-by-step remediation for each action level
5. **Production-ready design** — Streaming monitor, API design, configuration export

### Why This Matters for Industry

In production ML systems at companies like Google, Meta, Netflix:
- Models serve millions of predictions per day
- A broken model costs millions in revenue/fraud losses
- Manual investigation of every drift alert is impossible
- You NEED an automated system that classifies drift and tells you what to do

This engine is that system.

---

## Key Terminology

| Term | Meaning |
|------|---------|
| **Reference data** | The data the model was trained on (the "ground truth" distribution) |
| **Current data** | New data arriving in production |
| **KS test** | Kolmogorov-Smirnov test — checks if two distributions are different |
| **Cohen's d** | Effect size — how PRACTICALLY significant a difference is (not just statistically) |
| **PSI** | Population Stability Index — measures how much a distribution shifted |
| **Feature importance** | How much the model relies on each feature for predictions |
| **Recall** | % of actual fraud cases the model catches |
| **Precision** | % of model's fraud alerts that are actually fraud |
| **F1 score** | Harmonic mean of precision and recall |
| **SMOTE** | Technique to handle class imbalance by creating synthetic minority samples |
| **Evidently** | Open-source library for ML monitoring and drift detection |

---

## How to Run This Project

```bash
# 1. Create virtual environment
python -m venv .drift
.drift\Scripts\Activate.ps1  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
#    01 → EDA and data understanding
#    02 → Model training (produces feature importances)
#    03 → Drift simulation + Taxonomy Engine (the main contribution)
```

---

## Summary (One Paragraph)

We built a **Drift Taxonomy Engine** that solves the problem of "my ML model is failing in production and I don't know why or what to do." The engine automatically classifies data drift into three types (covariate, concept, pipeline), scores severity using feature-importance weighting, and recommends specific actions — crucially distinguishing between **incremental retraining** (when inputs shifted but the model logic is still valid) and **full retraining** (when the world changed and the model's rules are now wrong) and **blocking** (when data is corrupt and retraining would poison the model). This is validated on a credit card fraud detection system across multiple simulated drift scenarios.
