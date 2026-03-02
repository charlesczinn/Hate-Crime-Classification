# Predicting Violent Hate Crime Classification Using Machine Learning
A Random Forest and Logistic Regression Comparison

# 1.0 Overview
This project analyzes hate crimes in Chicago from 2012–2025 to identify spatial patterns and factors associated with violent hate crimes. Using Chicago Police Department open data, the goal is to predict whether a hate crime is violent and to interpret which features drive violence.
The project implements:
- Logistic Regression (baseline)
- Random Forest (tuned)
- Feature importance and partial dependence analysis
This repository includes all code, processed data, and results necessary for reproducibility.

# 2.0 Data
Source: Chicago Police Department (public hate crime dataset)
Temporal Coverage: 2012–2025
Unit of Analysis: Individual incident
Outcome Variable: violent_crime (binary)
    1 = Physical violence (assault, battery, robbery, arson, homicide)
    0 = Non-violent offense (e.g., vandalism, harassment)
Violence is operationalized as direct physical harm, consistent with established criminological definitions.

Predictor Variables:
- Motivations: Bias type (e.g., ANTI-JEWISH, ANTI-GAY)
- CPD.Area: Police reporting area (North, Central, South)
- DISPOSITION: Determination of bias (Bona fide, Undetermined, Unfounded)
- DATEOCC: Date of occurrence (aggregated to season)

# 3.0 Methodology
Data Cleaning & Feature Engineering
- Converted DATEOCC to a season variable (Winter, Spring, Summer, Fall)
- Encoded categorical variables with pd.get_dummies

Modeling
- Split data: 75% train / 25% test
- Baseline model: Logistic Regression (balanced classes)
- Tuned model: Random Forest with hyperparameter search for:
        n_estimators = 100, 200, 300
        max_features = 4, 6, 8
        min_samples_leaf = 1, 2, 4

Evaluation Metrics
- Accuracy
- ROC-AUC
- Confusion matrices
- Feature importance & partial dependence plots

# 4.0 Model Evaluation & Results
Model Performance Summary:

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.688    | 0.759   |
| Tuned Random Forest | 0.688    | 0.749   |

- Logistic Regression slightly outperforms Random Forest in ROC-AUC.
- Both models demonstrate moderate predictive performance for violent hate crimes.

Random Forest Feature Importance:
| Rank | Feature                                 | Importance |
| ---- | --------------------------------------- | ---------- |
| 1    | Motivations_ANTI-JEWISH                 | 0.297      |
| 2    | Motivations_ANTI-GAY (MALE)             | 0.127      |
| 3    | Motivations_ANTI-ASIAN                  | 0.057      |
| 4    | Motivations_ANTI-HISPANIC/LATINO        | 0.048      |
| 5    | CPD.Area_South                          | 0.041      |
| 6    | CPD.Area_North                          | 0.039      |
| 7    | Motivations_ANTI-WHITE                  | 0.037      |
| 8    | Motivations_ANTI-TRANSGENDER            | 0.037      |
| 9    | Motivations_ANTI-MULTIPLE RACES/GROUP   | 0.036      |
| 10   | Motivations_ANTI-BLACK/AFRICAN-AMERICAN | 0.034      |

Partial Dependencies:
| Feature                          | Effect on `violent_crime=1` |
| -------------------------------- | --------------------------- |
| Motivations_ANTI-JEWISH          | decreases likelihood        |
| Motivations_ANTI-GAY (MALE)      | increases likelihood        |
| Motivations_ANTI-ASIAN           | increases likelihood        |
| Motivations_ANTI-HISPANIC/LATINO | increases likelihood        |
| CPD.Area_South                   | increases likelihood        |

- Anti-Jewish bias reduces the probability of violence.
- Anti-Gay, Anti-Asian, and Anti-Hispanic biases increase the likelihood.
- Violent incidents are more likely in the Southern CPD reporting area.

Other Findings:
- Other anti-Queer biases increased predicted probability.
- Violent incidents are less likely in the Northern CPD reporting area.
- Spring and Summer slightly increased predicted probability relative to Fall/Winter.

Separate exploratory spatial analysis (not included in model training) revealed:
- Northern Chicago exhibited higher overall hate crime counts.
- Southern Chicago exhibited a higher proportion of violent incidents.
These findings suggest potential geographic reporting or contextual differences, though spatial covariates were not included in the classifier.

# 5.0 Limitations & Next Steps
Geography: Analysis is restricted to Chicago, results may differ in other cities.
Demographics: Socioeconomic and population features were not included. Incorporating ward-level demographics could improve predictive power.
Bias Reporting: Variability in reporting may affect model accuracy, particularly in affluent vs. marginalized areas.
Future Work: Expand analysis to other cities, incorporate covariates, and explore temporal trends more deeply.
