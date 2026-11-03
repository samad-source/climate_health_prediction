# Climate Risk and Health Prediction

A machine-learning classification project developed for the **Zindi
Climate Risk and Health Prediction Challenge**. The project combines
demographic, geographic, date-derived, and climate-related features to
predict the target `is_climate_sensitive`.

> **Competition result recorded during this project:** the public
> leaderboard displayed a score of **0.835549156**, placing the project
> in **2nd place** at the time of the recorded leaderboard snapshot.
> Leaderboard positions and scores can change.

## Project overview

The task is binary classification. Each record represents an observation
with demographic and environmental information. The model estimates
whether the observation belongs to the climate-sensitive class.

The competition submission contains: - `ID`: record identifier. -
`TargetF1`: binary prediction produced using a selected probability
threshold. - `TargetRAUC`: predicted probability for the positive class,
used for ranking-based evaluation.

The competition score was treated during experimentation as:

\[ `\text{Combined Score}`{=tex} = 0.6 `\times `{=tex}F1 + 0.4
`\times `{=tex}ROC`\text{-}`{=tex}AUC \]

Always verify the current competition metric and submission instructions
on the official Zindi page before submitting.

## Dataset

The working data files were: - `Train.csv` --- labelled training
records. - `Test.csv` --- unlabelled competition test records. -
`climate_features.csv` --- additional climate and geographic features
joined by `ID`. - A data dictionary was also consulted.

Recorded dataset dimensions: - Train: **3,146 rows × 13 columns** before
feature engineering. - Test: **1,030 rows × 12 columns** before feature
engineering. - Climate features: **4,176 rows × 18 columns**.

The climate feature table was merged with train and test using `ID` with
a one-to-one validation check. The extra climate-table `deathdate` was
removed after merging to avoid a duplicate date field. `hot_days_30d`
was dropped because it was zero for every train and test record.

Do not commit competition data unless the competition's data licence and
rules explicitly allow redistribution. The repository can instead
explain how to obtain the files from Zindi.

## Workflow

1.  Load and inspect the supplied datasets.
2.  Check row counts, column names, missing values, duplicate records,
    and duplicate IDs.
3.  Join climate features to train and test using `ID`.
4.  Parse dates and derive calendar features, including `year`, `month`,
    and `day_of_year`.
5.  Create `temperature_range` from maximum and minimum temperature.
6.  Separate numeric and categorical features.
7.  Split the labelled data into stratified training and validation
    subsets using `random_state=42`.
8.  Build preprocessing and model pipelines.
9.  Train and compare classification algorithms.
10. Tune selected models and test ensemble combinations.
11. Generate probabilities and binary predictions for the competition
    test set.
12. Save a CSV with the required submission columns.

## Feature engineering and preprocessing

The numerical feature set included age, temperature measurements,
precipitation, elevation, slope, vegetation indices, rainfall summaries,
rolling temperature summaries, temperature range, and date-derived
fields. The categorical features were `zone` and `gender`.

Preprocessing used: - `StandardScaler` for numerical columns. -
`OneHotEncoder(handle_unknown="ignore")` for categorical columns. -
`ColumnTransformer` to apply the correct transformation to each feature
type. - `Pipeline` to keep preprocessing and model fitting together.

The target distribution in the training data was: - Class 0: **1,099
(34.93%)** - Class 1: **2,047 (65.07%)**

## Models explored

The experiments included: - Logistic Regression - Random Forest -
HistGradientBoosting - Extra Trees - Support Vector Machine - AdaBoost -
XGBoost - LightGBM - CatBoost - Soft-voting and manually weighted
probability ensembles

### Recorded validation results

The following are results from the project's validation experiments.
Thresholds were selected by searching validation predictions, so these
values may be optimistic and should not be interpreted as unbiased final
estimates.

  ----------------------------------------------------------------------------------
  Model / experiment          Threshold             F1        ROC-AUC       Combined
  ---------------------- -------------- -------------- -------------- --------------
  Logistic Regression              0.35         0.8281         0.8227         0.8259

  Random Forest                    0.50         0.8078         0.7956         0.8029

  Tuned                            0.46         0.8199         0.8283         0.8233
  HistGradientBoosting                                                

  AdaBoost                         0.50         0.8239         0.8325         0.8274

  XGBoost                          0.37         0.8194         0.8157         0.8179

  LightGBM                         0.30         0.8103         0.8154         0.8124

  SVM                              0.42         0.8186         0.7956         0.8094

  Extra Trees                      0.48         0.8018         0.7833         0.7944

  CatBoost tuned                   0.35         0.8347         0.8285         0.8322

  Equal-weight soft                0.46         0.8307         0.8361         0.8328
  ensemble                                                            

  Weighted 3-model                 0.48         0.8320         0.8462     **0.8377**
  ensemble (Ada 50%,                                                  
  Hist 30%, LR 20%)                                                   

  Best tested 4-model              0.45         0.8326         0.8432         0.8368
  blend (Ada 45%, Hist                                                
  25%, LR 15%, Cat 15%)                                               
  ----------------------------------------------------------------------------------

The manually weighted three-model blend had the strongest recorded local
validation combined score among these experiments. The four-model blend
that included CatBoost did not improve on it.

## Feature importance and temporal checks

Permutation importance for AdaBoost ranked `age` first (approximately
0.2868), followed by `year` (approximately 0.0787). The remaining
individual features had much smaller measured importance in that
experiment.

A feature-ablation experiment showed that removing age substantially
reduced ROC-AUC on the random validation split. Removing year also
reduced it. These results show predictive association, not causation,
and do not establish that the features will generalise equally well to
other time periods.

A chronological diagnostic produced ROC-AUC **0.6248** and F1
**0.7028**, lower than the random-split results. The diagnostic split
was created by sorting records by year and splitting at 80%; the printed
year ranges overlapped at 2019, so it was not a strict year-held-out
split. This is an important limitation and should be corrected before
drawing firm conclusions about future-year performance.

## Competition result

The leaderboard snapshot recorded: - Project account: `am_unscripted` -
Displayed rank: **2** - Displayed public score: **0.835549156** -
First-place score in that snapshot: **0.839356094**

The public score gap was approximately **0.003806938**. This is a
historical snapshot, not a guarantee of current rank.

## Repository structure

A suggested structure is:

``` text
climate-risk-health-prediction/
├── README.md
├── notebooks/
│   └── climate_features.ipynb
├── reports/
│   └── technical_report.md
├── submissions/
│   ├── submission.csv
│   ├── submission_ensemble.csv
│   └── submission_weighted_ensemble.csv
├── requirements.txt
└── .gitignore
```

Keep the original data files out of Git unless redistribution is
explicitly permitted. Also check that submission files do not violate
competition rules before publishing them.

## Environment and installation

The project was developed in Python using a Jupyter notebook. Main
libraries used include:

-   pandas
-   NumPy
-   scikit-learn
-   Matplotlib / Seaborn (for exploratory plots, if used in the
    notebook)
-   XGBoost
-   LightGBM
-   CatBoost

A starting `requirements.txt` can contain:

``` text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
jupyter
ipykernel
```

Install with:

``` bash
python -m pip install -r requirements.txt
```

Open the notebook in Jupyter or VS Code and update the data paths to the
location where the permitted competition files have been downloaded.

## Reproducibility notes

-   A random seed of `42` was used for the main split and several model
    experiments.
-   The main random validation split was stratified.
-   Preprocessing was encapsulated in pipelines.
-   Threshold selection was performed using validation predictions, not
    the competition test labels.
-   The test set was used only to generate submission predictions.
-   Some experiments used the same validation split for threshold
    selection and reporting. Their reported scores are therefore
    model-selection estimates, not independent final performance
    estimates.

## Limitations and responsible interpretation

-   Predictive importance does not imply causality.
-   The target definition and the timing at which each feature becomes
    available should be checked against the competition documentation.
-   The strong influence of age and year warrants domain review and
    leakage checks.
-   The chronological diagnostic suggests possible temporal distribution
    shift.
-   Performance should be assessed across relevant subgroups and time
    periods before any real-world use.
-   This competition model is an educational project and is not a
    clinical or public-health decision system.

## Acknowledgements

Dataset and challenge: [Zindi --- Climate Risk and Health Prediction
Challenge](https://zindi.world/competitions/climate-risk-health-prediction-challenge-miva).

## Author

Project participant: `am_unscripted` (Zindi leaderboard account).
