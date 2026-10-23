# Predicting Violent Hate Crimes in Chicago
Logistic Regression and Random Forest Comparison

## 1. Overview
This project analyzes Chicago Police Department hate crime records (2012–2025) to predict whether a hate crime involves physical violence and to identify which incident characteristics are most associated with violence.

The notebook (`Hate_Crime_Classification.ipynb`) covers:
- Data cleaning and exploratory analysis
- A logistic regression baseline and a tuned random forest, evaluated with 5-fold cross-validation and a held-out test set
- A leakage check on a post-investigation variable
- Model interpretation with permutation importance and partial dependence

All code, data, and outputs needed to reproduce the analysis are included.

## 2. Data
- **Source:** Chicago Police Department hate crime open data (`data/crimes.csv`)
- **Coverage:** 2012–2025; unit of analysis is the individual incident
- **Outcome:** `violent_crime` (1 = physical violence: assault, battery, robbery, arson, homicide; 0 = non-violent offenses such as vandalism or harassment)
- **Predictors** (available at the time of report):
  - `Motivations`: bias type (e.g., ANTI-JEWISH, ANTI-GAY (MALE))
  - `CPD.Area`: police reporting area (North, Central, South)
  - `season`: derived from the date of occurrence

## 3. Methods
**Cleaning.** Stripped whitespace from bias categories and removed 5 records marked `DUPLICATE` or `PENDING INVESTIGATION`, leaving 1,599 incidents (52.9% violent).

**Modeling.** Stratified 75/25 train/test split. Logistic regression (balanced class weights) as a baseline; random forest (balanced class weights) tuned by grid search over `n_estimators`, `max_features`, and `min_samples_leaf` using 5-fold cross-validated ROC-AUC.

**Leakage check.** `DISPOSITION` (bona fide, undetermined, unfounded) is determined after investigation, so it is excluded from the main models and tested separately.

**Interpretation.** Permutation importance on the test set (drop in ROC-AUC) and partial dependence plots for the top features.

## 4. Results

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall (violent) | Precision (violent) |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.772 | 0.760 | 0.695 | 0.670 | 0.732 |
| Random forest (tuned) | 0.759 | 0.740 | 0.680 | 0.679 | 0.706 |

- Logistic regression performs as well as or slightly better than the random forest, suggesting largely additive relationships; the simpler model is sufficient.
- Adding `DISPOSITION` changes test ROC-AUC by less than 0.01, so results are not driven by post-investigation information.

**Most important features (permutation importance, random forest):**

| Rank | Feature | Mean drop in ROC-AUC |
| --- | --- | --- |
| 1 | Motivations_ANTI-JEWISH | 0.071 |
| 2 | Motivations_ANTI-GAY (MALE) | 0.046 |
| 3 | Motivations_ANTI-TRANSGENDER | 0.011 |
| 4 | Motivations_ANTI-BLACK/AFRICAN-AMERICAN | 0.008 |
| 5 | Motivations_ANTI-WHITE | 0.007 |

**Direction of effects:**
- Anti-Jewish incidents are far less likely to be violent (16% vs. 61% of other incidents), consistent with a high share of property offenses.
- Anti-gay (male) (78% violent), anti-transgender (86%, n = 51), and anti-white (76%) incidents are more likely to involve physical violence.
- Incidents in the South reporting area are more often violent (75%) than in the North (47%), though the South has few incidents (177) and adds little predictive value once motivation is known.

Figures are saved in `outputs/`: exploratory summaries, ROC curves and confusion matrices, permutation importance, and partial dependence.

## 5. Limitations and Next Steps
- **Moderate predictive power:** report-time features cannot capture the circumstances that drive violence; results describe associations, not causes.
- **Reporting differences:** hate crime reporting varies across neighborhoods and communities.
- **Geography:** results apply to Chicago and may not generalize.
- **Next steps:** add ward- or community-level demographic covariates, model temporal trends, and compare with other cities.

## 6. Reproducing
Requires Python 3 with pandas, numpy, scikit-learn, matplotlib, and seaborn. Run `Hate_Crime_Classification.ipynb` from the repository root; figures are written to `outputs/`.
