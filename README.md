# 🚦 Predicting Traffic Accident Severity in Bahrain: A Comparative Study

A statistical learning and machine learning research project analyzing **35,999 historical traffic accident records in Bahrain** to predict accident severity levels (`Fatal`, `Serious`, and `Slight`)[cite: 8]. This study compares four classical classification algorithms and provides an in-depth analysis of class imbalance failure modes in real-world traffic safety datasets[cite: 8].

---

## 📌 Project Overview

Traffic accidents are a critical public safety challenge in Bahrain[cite: 8]. This study explores predicting accident outcomes using statistical learning models[cite: 8]. While all implemented classical algorithms achieved an overall accuracy of **79.95%** matching the majority-class baseline, the study reveals how severe class imbalance (16:1 ratio) causes standard models to fail at detecting critical minority classes (`Fatal` and `Serious` accidents)[cite: 8].

### Key Findings
* **Severe Class Imbalance:** The dataset consists of **79.95% Slight**, **15.08% Serious**, and **4.97% Fatal** accidents[cite: 8].
* **Majority Class Bias:** Standard algorithms optimized for global accuracy defaulted to predicting `Slight` for 100% of test cases[cite: 8].
* **Zero Sensitivity:** All four baseline models yielded **0.00 sensitivity** for `Fatal` and `Serious` categories, resulting in a **Cohen's Kappa statistic of 0.0000**[cite: 8].
* **Core Takeaway:** High overall accuracy can mask complete model failure when predicting critical, rare events in public health and safety domains[cite: 8].

---

## 🛠️ Models & Methodologies

The project evaluates four classical supervised learning techniques implemented in **R**[cite: 8]:

1. **Multinomial Logistic Regression (`nnet`):** Probabilistic linear model using maximum likelihood estimation[cite: 8].
2. **Linear Discriminant Analysis - LDA (`MASS`):** Discriminant analysis assuming multivariate normality and shared covariance across classes[cite: 8].
3. **Quadratic Discriminant Analysis - QDA (`MASS`):** Flexible discriminant method allowing class-specific covariance matrices[cite: 8].
4. **K-Nearest Neighbors - KNN (`class`):** Non-parametric distance-based instance classifier using standardized numeric features[cite: 8].

---

## 📊 Dataset & Preprocessing

* **Dataset Size:** 35,999 records (Partitioned 70/30 into 25,200 training and 10,799 test samples)[cite: 8].
* **Target Variable:** `Accident_Severity` (`Fatal`, `Serious`, `Slight`)[cite: 8].
* **Key Predictors:** Year, Weather, Road Type, Light Conditions, Speed Limit, Vehicles Involved, Casualties, Hour, Latitude, Longitude, and engineered feature `Is_Night`[cite: 8].
* **Preprocessing:** Standardized numeric features for distance-based calculations and applied stratified train-test splitting to preserve class ratios[cite: 8].

---

## 📈 Model Performance Summary

| Model | Overall Accuracy | 95% CI | Kappa Statistic | Fatal Sensitivity | Serious Sensitivity | Slight Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **79.95%** | (0.7953, 0.8036) | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| **Linear Discriminant Analysis (LDA)** | **79.95%** | (0.7953, 0.8036) | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| **Quadratic Discriminant Analysis (QDA)** | **79.95%** | (0.7953, 0.8036) | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| **K-Nearest Neighbors (KNN)** | **79.95%** | (0.7953, 0.8036) | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

*No Information Rate (NIR) Baseline: **79.95%**[cite: 8]*

---

## 💡 Recommendations for Future Improvements

To overcome class imbalance and build deployable accident severity models, the report outlines several concrete strategies[cite: 8]:

* **Synthetic Oversampling:** Apply **SMOTE** or hybrid sampling methods (`ROSE`) to generate synthetic minority class examples during training[cite: 8].
* **Class Weighting:** Assign inverse frequency weights (e.g., ~16:1 penalty ratio for misclassifying `Fatal` vs `Slight` accidents)[cite: 8].
* **Advanced Ensembles:** Utilize **Balanced Random Forests** or **XGBoost** with `scale_pos_weight` adjustments[cite: 8].
* **Alternative Evaluation Metrics:** Evaluate performance using **Macro F1-Score**, **Class-Specific Recall/Sensitivity**, and **Precision-Recall AUC** instead of raw accuracy[cite: 8].
* **Cost-Sensitive Learning:** Incorporate real-world risk matrices penalizing severe false negatives[cite: 8].

---

## 🏛️ Academic Context

* **Institution:** University of Bahrain — College of Science, Post Graduate Programs[cite: 8]
* **Course/Program:** Statistical Learning / Data Science Graduate Research Project Report[cite: 8]
* **Supervision:** Dr. Sawsan Hilal[cite: 8]
