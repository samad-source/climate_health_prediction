# Climate Risk and Health Prediction

## Detailed Technical and Research Report

**Competition:** Zindi Climate Risk and Health Prediction Challenge\
**Project account:** `am_unscripted`\
**Project type:** Supervised binary classification\
**Recorded public leaderboard position:** 2nd place\
**Recorded public score:** 0.835549156

------------------------------------------------------------------------

## Abstract

This project investigated supervised machine-learning methods for
predicting whether a record is climate-sensitive using demographic,
geographic, temporal, and climate-related variables. The work was
completed as part of the Zindi Climate Risk and Health Prediction
Challenge. The workflow included data inspection, joining additional
climate features, date-based feature engineering, preprocessing,
exploratory analysis, model comparison, hyperparameter tuning, threshold
selection, ensemble learning, and competition submission generation.

A range of classification algorithms was evaluated. AdaBoost, Logistic
Regression, and HistGradientBoosting were combined through probability
averaging. A manually weighted blend assigning 50% weight to AdaBoost,
30% to HistGradientBoosting, and 20% to Logistic Regression achieved the
strongest recorded local validation combined score of 0.8377, using a
threshold of 0.48. A four-model blend that added CatBoost did not
improve on this result. The competition leaderboard snapshot showed a
public score of 0.835549156 and rank 2.

The experiments also identified important limitations. Age and year were
highly influential in permutation and feature-ablation experiments,
while a chronological diagnostic produced materially lower performance
than random validation. The chronological split used in that diagnostic
had an overlapping boundary year and should be rerun with a strict year
boundary. Consequently, the report distinguishes local validation
results from public leaderboard results and does not interpret
predictive associations as causal effects.

## Table of Contents

1.  Introduction
2.  Background and Problem Definition
3.  Aim and Objectives
4.  Dataset Description
5.  Data Inspection and Quality Checks
6.  Exploratory Data Analysis
7.  Data Preprocessing
8.  Feature Engineering
9.  Model Development
10. Model Evaluation
11. Hyperparameter Tuning
12. Ensemble Learning
13. Feature Importance and Ablation Experiments
14. Temporal Generalisation Diagnostic
15. Competition Submission and Leaderboard Results
16. Discussion
17. Limitations and Risks
18. Recommendations and Future Work
19. Conclusion
20. References
21. Appendix: Reproducibility and Submission Schema

------------------------------------------------------------------------

## 1. Introduction

Climate conditions can affect human health and living conditions in
different ways. Understanding which observations are labelled
climate-sensitive may help researchers explore relationships among
environmental conditions, demographic characteristics, geography, and
time.

Machine learning provides methods for identifying statistical patterns
in historical data and using those patterns to predict labels for unseen
records. This project applies supervised classification to the Climate
Risk and Health Prediction Challenge hosted on Zindi.

The project is an educational competition exercise. Its model outputs
should not be interpreted as medical diagnoses, causal explanations, or
direct public-health recommendations.

## 2. Background and Problem Definition

### 2.1 Machine-learning task

The task is **supervised binary classification**.

-   Supervised: the training dataset contains a known target label.
-   Binary: the target has two classes, represented as 0 and 1.
-   Classification: the model predicts which class an observation
    belongs to.

The target column is `is_climate_sensitive`.

### 2.2 Problem statement

Given the demographic, geographic, date-related, and climate variables
available for each record, train a model that predicts the probability
that the record belongs to the climate-sensitive class.

### 2.3 Prediction outputs

The competition submission used two prediction columns: - `TargetF1`: a
binary class prediction derived by applying a threshold to the
positive-class probability. - `TargetRAUC`: the positive-class
probability, which preserves ranking information for ROC-AUC evaluation.

A threshold changes the binary predictions and therefore affects F1.
ROC-AUC is calculated from probability scores and does not require
choosing a single classification threshold.

## 3. Aim and Objectives

### 3.1 Aim

To develop and evaluate machine-learning classification models for the
Zindi Climate Risk and Health Prediction Challenge.

### 3.2 Objectives

1.  Inspect the supplied training, test, and climate feature datasets.
2.  Check data quality, missing values, duplicates, and identifier
    consistency.
3.  Combine the supplied datasets using the record identifier.
4.  Engineer useful temporal and temperature-derived features.
5.  Explore the relationship between predictors and the target.
6.  Build reproducible preprocessing and modelling pipelines.
7.  Compare several classification algorithms.
8.  Tune selected model hyperparameters.
9.  Evaluate probability-based ensemble methods.
10. Generate valid competition submissions.
11. Examine feature importance and possible temporal distribution shift.
12. Document results, limitations, and opportunities for further work.

## 4. Dataset Description

Three principal CSV datasets were used.

  ---------------------------------------------------------------------------
  File                              Recorded dimensions Purpose
  ------------------------ ---------------------------- ---------------------
  `Train.csv`                                3,146 × 13 Labelled
                                                        model-training data

  `Test.csv`                                 1,030 × 12 Unlabelled records
                                                        for competition
                                                        prediction

  `climate_features.csv`                     4,176 × 18 Additional climate
                                                        and geographic
                                                        features
  ---------------------------------------------------------------------------

The original training columns included `ID`, `zone`, `gender`,
`deathdate`, `age`, average/maximum/minimum temperature, precipitation,
latitude, longitude, location, and `is_climate_sensitive`.

The test dataset contained the corresponding predictors but not the
target.

The climate feature dataset included variables such as elevation, hot
days, rainfall summaries, vegetation indices, slope, rolling average
temperatures, rolling maximum and minimum temperatures, and temperature
range summaries.

### 4.1 Target distribution

The training target distribution was:

  Class           Records   Percentage
  ----------- ----------- ------------
  0                 1,099       34.93%
  1                 2,047       65.07%
  **Total**     **3,146**     **100%**

The target was moderately imbalanced, with class 1 representing
approximately 65.07% of the training observations.

### 4.2 Identifier-based merge

The climate features were joined to the training and test tables using
`ID`. The merge used a one-to-one validation check to ensure that each
ID matched one climate feature record. The climate feature table's
duplicate `deathdate` field was removed after merging.

The feature `hot_days_30d` was dropped because it was zero for every
training and test record in the inspected data. A constant feature
cannot distinguish between observations.

## 5. Data Inspection and Quality Checks

The initial inspection included: - Dataset shape and column names. -
Data types. - Missing-value counts. - Duplicate-row checks. -
Duplicate-ID checks. - Train/test identifier overlap. - Climate-feature
ID matching. - Target class counts.

The recorded checks found no missing values, duplicate rows, or
duplicate IDs in the three inspected datasets. Train and test
identifiers matched climate-feature records, and train/test IDs did not
overlap.

These checks describe the inspected data versions and should be rerun if
the source files change.

## 6. Exploratory Data Analysis

Exploratory data analysis (EDA) was used to understand the target
distribution, feature distributions, possible outliers, and differences
between target classes.

### 6.1 Age

Age showed a substantial difference between target classes: - Mean age
for class 0: approximately 43.25. - Mean age for class 1: approximately
14.04.

The overall median age was approximately 3, while the mean was
approximately 24.24, indicating a strongly skewed age distribution.

Age was later identified as the most influential feature in the recorded
permutation-importance experiment. This is a predictive association; it
does not establish that age causes climate sensitivity or explain how
the target label was defined.

### 6.2 Gender

The observed class-1 rate was approximately: - Female: 67.16%. - Male:
63.16%.

This difference is descriptive. It should not be treated as a causal or
universal relationship.

### 6.3 Climate and geographic features

The recorded class-wise means for many climate variables were relatively
close. Examples included average temperature, maximum temperature,
minimum temperature, precipitation, rolling rainfall, vegetation
indices, elevation, slope, and rolling temperature summaries.

Small differences in group averages do not prove that a variable has no
predictive value. A model can use nonlinear patterns and interactions
that are not obvious from comparing means alone.

### 6.4 Outliers

Outliers were not automatically removed. Unusual climate or demographic
observations may be valid. Removing them without domain justification
could discard useful information or distort the distribution.

## 7. Data Preprocessing

### 7.1 Date parsing

The date column was parsed using pandas date conversion with
mixed-format support and invalid values coerced to missing during
parsing. Calendar fields were then extracted.

### 7.2 Numerical and categorical features

Categorical features: - `zone` - `gender`

Numerical features included: - `age` - temperature and precipitation
measurements - elevation and slope - vegetation indices - rainfall
summaries - rolling temperature summaries - temperature range - month,
day of year, and year

### 7.3 Scaling

`StandardScaler` was used for numerical columns. Scaling places
numerical features on comparable standardized scales and is particularly
useful for models such as Logistic Regression and SVM.

### 7.4 One-hot encoding

`OneHotEncoder(handle_unknown="ignore")` was used for categorical
columns. This converts categories into numerical indicator columns and
allows the pipeline to handle categories not encountered during fitting.

### 7.5 ColumnTransformer and Pipeline

A `ColumnTransformer` applied numerical scaling and categorical encoding
to the appropriate columns. The transformations and classifier were
combined in scikit-learn `Pipeline` objects.

This approach helps keep preprocessing consistent between training,
validation, and test prediction. Transformations are fitted as part of
the training pipeline rather than manually fitted on the complete
dataset before splitting.

### 7.6 Train-validation split

The primary local evaluation used: - 80% training subset. - 20%
validation subset. - `random_state=42`. - Stratification by the target
label.

Stratification maintained a similar class proportion in both subsets.

## 8. Feature Engineering

### 8.1 Temperature range

A temperature-range feature was created from maximum and minimum
temperature:

`temperature_range = max_temperature - min_temperature`

This represents the difference between the recorded maximum and minimum
temperatures.

### 8.2 Calendar features

The date field was used to derive: - `year` - `month` - `day_of_year`

These fields allow models to learn patterns associated with calendar
time. Their predictive value must be checked against the competition's
prediction-time rules and the way the target was generated.

### 8.3 Rolling climate features

The additional climate dataset supplied summaries over different
windows, including 7-day, 30-day, and 90-day rainfall and temperature
measures. These features were merged by ID rather than independently
calculated in the notebook.

## 9. Model Development

Several algorithms were evaluated to compare different ways of learning
patterns.

### 9.1 Logistic Regression

Logistic Regression is a linear classification model that estimates the
probability of belonging to a class. It provided a useful baseline and
performed competitively in the local validation experiments.

### 9.2 Random Forest

Random Forest combines many decision trees trained using different
samples and feature subsets. Its predictions are aggregated to reduce
the instability of a single tree.

### 9.3 HistGradientBoosting

HistGradientBoosting builds decision trees sequentially, with later
trees attempting to improve on errors made by earlier trees. A tuned
configuration was retained for ensemble experiments.

Recorded best parameters: - `max_leaf_nodes`: 10 - `max_iter`: 200 -
`learning_rate`: 0.02 - `l2_regularization`: 2

### 9.4 Extra Trees

Extra Trees is a tree ensemble that introduces additional randomisation
when selecting splits. It was evaluated as another tree-based approach.

### 9.5 Support Vector Machine

SVM attempts to identify a decision boundary that separates classes. The
pipeline used preprocessing so that numerical scaling and categorical
encoding were applied consistently.

### 9.6 AdaBoost

AdaBoost combines weak learners sequentially, giving more attention to
observations that previous learners handled poorly. It achieved the
strongest individual-model combined validation result in the recorded
eight-model comparison.

A scikit-learn FutureWarning appeared for the default `SAMME.R`
algorithm. It was a deprecation warning rather than a training failure.
The warning does not by itself invalidate the completed experiment.

### 9.7 XGBoost and LightGBM

These are gradient-boosting frameworks that build ensembles of decision
trees. Both were evaluated and compared with the other candidate
algorithms.

### 9.8 CatBoost

CatBoost was also tested. A randomized hyperparameter search over 20
candidates with 3-fold cross-validation selected: - `random_strength`:
0.5 - `learning_rate`: 0.01 - `l2_leaf_reg`: 7 - `iterations`: 400 -
`depth`: 6

The best cross-validation ROC-AUC reported by that search was
approximately 0.8103. On the separate random validation split, the tuned
CatBoost model achieved F1 0.8347, ROC-AUC 0.8285, and combined score
0.8322 at threshold 0.35.

## 10. Model Evaluation

### 10.1 F1 score

F1 is the harmonic mean of precision and recall. It balances
false-positive and false-negative behaviour and is useful when class
distribution is not perfectly balanced.

### 10.2 ROC-AUC

ROC-AUC measures how well the model ranks positive examples above
negative examples across possible classification thresholds. It uses
probability or score outputs rather than a single binary threshold.

### 10.3 Combined competition-oriented score

The experiments used:

`Combined Score = 0.6 × F1 + 0.4 × ROC-AUC`

The exact competition instructions should be checked before every final
submission in case the metric or column requirements change.

### 10.4 Threshold selection

A threshold converts probabilities into class labels. For example, with
threshold 0.48: - Probability at least 0.48 → class 1. - Probability
below 0.48 → class 0.

Threshold selection affects F1 but does not change ROC-AUC when the
underlying probability scores remain the same.

Thresholds were searched on validation predictions. Since the same
validation set was used for selection and reporting, the resulting F1
and combined scores may be optimistic. A separate threshold-selection
set or nested/repeated cross-validation would provide a stronger
estimate.

## 11. Hyperparameter Tuning

RandomizedSearchCV was used for selected model tuning. It samples
parameter combinations rather than exhaustively evaluating every
possible combination.

The tuned HistGradientBoosting experiment used: - 20 parameter
samples. - 3-fold cross-validation. - F1 scoring. - A fixed random seed.

CatBoost tuning used 20 parameter candidates and 3-fold
cross-validation, with ROC-AUC as the search metric.

Cross-validation scores from different searches should only be compared
when the data folds, scoring metric, preprocessing, and experimental
setup are comparable.

## 12. Ensemble Learning

### 12.1 Why ensemble learning was used

An ensemble combines predictions from multiple models. Different
algorithms may learn different patterns, and combining their
probabilities can sometimes improve ranking and classification
performance.

### 12.2 Equal-weight soft voting

The first successful ensemble combined: - AdaBoost - Logistic
Regression - Tuned HistGradientBoosting

Soft voting averages predicted class probabilities. The ensemble was
evaluated on the validation subset and then fitted on the full labelled
training data for test prediction.

Recorded validation result: - Threshold: 0.46 - F1: 0.8307 - ROC-AUC:
0.8361 - Combined score: 0.8328

The corresponding competition submission was associated with the
recorded public score of 0.835549156 in the leaderboard history.

### 12.3 Manually weighted probability blend

A second experiment tested different weights for the three models. The
strongest tested blend assigned: - AdaBoost: 50% - HistGradientBoosting:
30% - Logistic Regression: 20%

Its validation results were: - Threshold: 0.48 - F1: 0.8320 - ROC-AUC:
0.8462 - Combined score: 0.8377

This was the strongest recorded local validation combined score among
the tested blends.

### 12.4 Adding CatBoost

CatBoost was added to several four-model blends. The best tested
four-model combination assigned: - AdaBoost: 45% - HistGradientBoosting:
25% - Logistic Regression: 15% - CatBoost: 15%

Its validation results were: - Threshold: 0.45 - F1: 0.8326 - ROC-AUC:
0.8432 - Combined score: 0.8368

This was below the three-model weighted blend's 0.8377. Therefore, the
four-model combination was not selected as the best local validation
blend.

## 13. Feature Importance and Ablation Experiments

### 13.1 Permutation importance

Permutation importance measures how much a model's score changes when
the values of one feature are randomly shuffled. A larger drop suggests
the model relies more heavily on that feature for the selected metric
and evaluation data.

Recorded AdaBoost permutation importance included:

  Feature               Importance
  ------------------- ------------
  age                     0.286849
  year                    0.078659
  tavg_7d                 0.008166
  temperature_range       0.005651
  tmax_30d                0.005373
  rain_sum_30d            0.004205
  rain_sum_90d            0.003956
  tmin_30d                0.003707
  tavg_30d                0.003415
  min_temperature         0.002780

The values are specific to the model, validation set, scoring metric,
and permutation procedure. They do not establish causation.

### 13.2 Feature-ablation experiment

A feature-ablation experiment compared feature groups on the random
validation split:

  Feature group            ROC-AUC
  ---------------------- ---------
  All features              0.8325
  Without year              0.7951
  Without age               0.5839
  Without age and year      0.5438

This experiment suggests that age and year carried substantial
predictive information in that split. It does not prove that either
feature is a causal driver or that the same performance relationship
will hold in future periods.

### 13.3 Year-wise target rates

The observed target rate varied across years. Examples: - 2008: 79.6%
climate-sensitive. - 2013: 56.3%. - 2017: 57.3%.

Some years had relatively few observations. Rates from small groups are
less stable and should be interpreted cautiously.

The relationship between year and target should be checked against the
challenge's label-generation process and feature-availability rules to
rule out leakage or an unintended proxy.

## 14. Temporal Generalisation Diagnostic

A chronological diagnostic sorted the training data by year and split
the rows at 80%.

Recorded results: - Training years printed: 2007--2019. - Validation
years printed: 2019--2022. - ROC-AUC: 0.6248. - F1 at threshold 0.48:
0.7028.

The result was substantially lower than the random-split ROC-AUC. This
raises the possibility of temporal distribution shift or a random-split
estimate that is too optimistic for future-year prediction.

**Important implementation limitation:** the printed year ranges
overlapped at 2019. The split was made by row position after sorting,
not by holding out a strictly later set of complete years. It should
therefore be treated as an initial diagnostic, not a definitive
time-based evaluation.

A stronger follow-up would select a specific year boundary, train only
on years before that boundary, and validate only on years after it. The
boundary should be chosen based on the competition's actual train/test
time structure.

## 15. Competition Submission and Leaderboard Results

### 15.1 Submission format

The generated CSV had 1,030 rows and three columns: - `ID` -
`TargetF1` - `TargetRAUC`

The weighted blend submission used threshold 0.48. The file was checked
for: - Expected row count. - Expected columns. - Missing IDs. - Missing
binary predictions. - Missing probabilities.

No missing values were reported in the inspected submission file.

### 15.2 Leaderboard snapshot

The leaderboard supplied during the project showed:

    Rank Account            Public score
  ------ ---------------- --------------
       1 abayomiabiodun      0.839356094
       2 am_unscripted       0.835549156
       3 Deysholey           0.833121403

The score gap between first and second was approximately 0.003806938.

The weighted submission experiment did not produce a higher displayed
public score in the leaderboard snapshot supplied later. The current
best recorded public result therefore remained 0.835549156 at that
point.

Public leaderboard scores are based on the competition's public
evaluation subset and can differ from local validation results. The
leaderboard is not a substitute for a carefully designed validation
strategy.

## 16. Discussion

The project demonstrates a complete applied classification workflow:
data joining, preprocessing, feature engineering, model comparison,
tuning, threshold selection, ensemble learning, and submission
generation.

The strongest local validation combined score came from a weighted blend
of AdaBoost, HistGradientBoosting, and Logistic Regression. Adding
CatBoost did not improve the tested blend. This illustrates that a more
complex ensemble is not automatically better.

Age and year were much more influential than most individual climate
variables in the recorded feature-importance experiment. However,
importance is model- and dataset-dependent. It may reflect genuine
predictive signal, target construction, data collection practices,
correlations, or leakage. The available experiments cannot distinguish
these explanations conclusively.

The lower chronological diagnostic score is particularly important. It
suggests that a random split may not fully represent performance on
observations from later periods. The diagnostic also needs to be
corrected to use a strict non-overlapping year boundary.

The competition score and local validation score answer different
questions. Local validation estimates performance on a held-out portion
of the supplied training data. The public leaderboard measures
performance on the competition's public test subset. Differences between
them are expected.

## 17. Limitations and Risks

1.  **Validation optimism:** thresholds were selected and evaluated on
    the same validation predictions.
2.  **Temporal split issue:** the chronological diagnostic had a shared
    boundary year.
3.  **Potential temporal shift:** the chronological result was much
    lower than the random-split result.
4.  **Feature availability:** the project must verify that all features,
    particularly date-derived fields and climate summaries, are
    available at the intended prediction time.
5.  **Feature interpretation:** importance values are not causal
    effects.
6.  **Limited sample size:** the labelled dataset contained 3,146
    records.
7.  **Leaderboard uncertainty:** public leaderboard results may not
    match private leaderboard or future performance.
8.  **Competition scope:** the model was built for a challenge dataset
    and has not been validated for operational health or policy
    decisions.
9.  **Reproducibility:** package versions, exact source data versions,
    notebook execution order, and all model settings should be recorded
    for a fully reproducible run.

## 18. Recommendations and Future Work

1.  Re-run chronological validation using a strict year boundary with no
    overlapping years.
2.  Compare random, grouped, and temporal validation according to the
    actual competition data structure.
3.  Check the data dictionary and challenge rules for feature
    availability and label-generation timing.
4.  Use repeated or nested cross-validation for more reliable model and
    threshold selection.
5.  Evaluate calibration as well as discrimination, especially if
    probabilities will be interpreted.
6.  Inspect errors by year, zone, gender, and age groups where permitted
    and statistically meaningful.
7.  Investigate whether age and year remain useful and legitimate under
    the challenge's prediction-time rules.
8.  Record exact package versions, random seeds, and data checksums.
9.  Preserve the best known submission and compare new experiments
    against it before replacing it.
10. Avoid treating the model as a clinical or policy tool without
    independent domain validation, fairness assessment, and expert
    oversight.

## 19. Conclusion

This project applied multiple machine-learning classification algorithms
to the Zindi Climate Risk and Health Prediction Challenge. The workflow
covered data inspection, feature merging, preprocessing, exploratory
analysis, feature engineering, model training, hyperparameter tuning,
model comparison, ensemble learning, and competition submission
creation.

The best recorded local validation blend used AdaBoost (50%),
HistGradientBoosting (30%), and Logistic Regression (20%), with a
threshold of 0.48 and a combined validation score of 0.8377. The
four-model blend that included CatBoost scored 0.8368 and did not
improve on the three-model blend.

The recorded public leaderboard score was 0.835549156, placing the
project in second position in the supplied leaderboard snapshot. A
chronological diagnostic produced lower performance than random
validation, highlighting the importance of validation design and
temporal generalisation.

Overall, the project demonstrates that successful machine learning
involves more than choosing an algorithm. Data quality, feature
availability, preprocessing, evaluation design, threshold selection,
model combination, and careful interpretation all influence whether a
model's results are reliable.

## 20. References

1.  Zindi. *Climate Risk and Health Prediction Challenge*.
    https://zindi.world/competitions/climate-risk-health-prediction-challenge-miva
2.  scikit-learn documentation. *Pipeline and composite estimators*.
    https://scikit-learn.org/stable/modules/compose.html
3.  scikit-learn documentation. *Model evaluation: quantifying the
    quality of predictions*.
    https://scikit-learn.org/stable/modules/model_evaluation.html
4.  scikit-learn documentation. *Cross-validation: evaluating estimator
    performance*.
    https://scikit-learn.org/stable/modules/cross_validation.html
5.  scikit-learn documentation. *Permutation feature importance*.
    https://scikit-learn.org/stable/modules/permutation_importance.html
6.  CatBoost documentation. https://catboost.ai/docs/
7.  XGBoost documentation. https://xgboost.readthedocs.io/
8.  LightGBM documentation. https://lightgbm.readthedocs.io/

## 21. Appendix: Reproducibility and Submission Schema

### A. Main software libraries

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   CatBoost
-   XGBoost
-   LightGBM
-   Jupyter Notebook

### B. Submission validation checklist

Before submitting: - Confirm the CSV contains exactly the expected
competition columns. - Confirm the row count matches the competition
test set. - Confirm IDs are present and unique. - Confirm `TargetF1`
contains only valid binary values. - Confirm `TargetRAUC` contains
finite probability values between 0 and 1. - Confirm there are no
missing values. - Confirm the submission file corresponds to the
intended model and threshold.

### C. Example submission schema

  ID                 TargetF1   TargetRAUC
  ---------------- ---------- ------------
  ID_example_001            0         0.42
  ID_example_002            1         0.63

The example values above are illustrative only and are not competition
predictions.
