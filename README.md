# Predicting Violent Hate Crime Classification Using Machine Learning
A Random Forest and Logistic Regression Comparison

# 1.0 Overview
This project develops and evaluates supervised machine learning models to classify whether a reported hate crime incident is violent or non-violent. Using hate crime data from the Chicago Police Department (2012–2025), the analysis implements logistic regression and random forest classifiers to assess the predictive utility of categorical incident characteristics.
The primary objective is to determine: Can structured incident-level features predict whether a hate crime involves physical violence?
Understanding predictors of violent incidents can help inform analytical approaches to incident risk classification and structured decision-support tools.

# 2.0 Data
Source: Chicago Police Department (public hate crime dataset)
Temporal Coverage: 2012–2025
Unit of Analysis: Individual incident
Outcome Variable: violent_crime (binary)
    1 = Physical violence (assault, battery, robbery, arson, homicide)
    0 = Non-violent offense (e.g., vandalism, harassment)
Violence is operationalized as direct physical harm, consistent with established criminological definitions.
Predictor Variables
Bias Type (e.g., anti-Black, anti-Jewish, anti-Gay, etc.)
Police Reporting Area (North, Central, South)
Disposition
    Bona fide
    Undetermined
    Unfounded
Season of occurrence (derived from date)
Categorical predictors were one-hot encoded using pandas.get_dummies().

# 3.0 Methodology
3.1 Data Preprocessing
Converted date field to datetime format
Derived seasonal feature from incident month
Encoded categorical variables via one-hot encoding
Split dataset into training (75%) and testing (25%) sets
Random seed fixed for reproducibility

3.2 Models
Logistic Regression (Baseline Model)
Purpose: Establish linear baseline performance
Evaluation metric: ROC-AUC
Random Forest Classifier
Ensemble of decision trees
Non-linear, non-parametric model
Few distributional assumptions
Hyperparameters tuned using GridSearchCV:
    n_estimators: 100–500
    max_features: 2–8
    min_samples_leaf: 2–6
    5-fold cross-validation
    Scoring metric: ROC-AUC

# 4.0 Model Evaluation
Evaluation metrics: ROC-AUC, Accuracy, Feature importance, Partial dependence analysis
Performance Summary

| Model                 | Accuracy | ROC-AUC |
| --------------------- | -------- | ------- |
| Untuned Random Forest | 0.661    | 0.717   |
| Tuned Random Forest   | 0.678    | 0.749   |

# 5.0 Key Findings
5.1 Feature Importance
The most influential predictors included:
    Anti-Jewish bias
    Anti-Gay bias
    Seasonal indicators
    Disposition category
Notably:
Anti-Jewish bias decreased predicted probability of violence.
Anti-Gay and other anti-Queer biases increased predicted probability.
Spring and Summer slightly increased predicted probability relative to Fall/Winter.
Bona fide designation increased predicted probability of violence relative to undetermined cases.

5.2 Spatial Observations (Exploratory)
Separate exploratory spatial analysis (not included in model training) revealed:
    Northern Chicago exhibited higher overall hate crime counts.
    Southern Chicago exhibited a higher proportion of violent incidents.
These findings suggest potential geographic reporting or contextual differences, though spatial covariates were not included in the classifier.

# 6.0 Limitations
Several constraints affect interpretation:
No demographic or socioeconomic covariates included in classification model.
Potential reporting bias across neighborhoods.
No class imbalance correction applied.
No temporal cross-validation.
Single-city case study limits generalizability.

Future extensions could incorporate:
Ward-level demographic integration
Class imbalance handling (e.g., SMOTE or class weights)
Out-of-time validation
Multi-city comparative modeling
