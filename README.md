# Fitness & Health Tracking Data Analysis

## Project Purpose

This project performs exploratory data analysis (EDA) and machine learning on a synthetic fitness and health tracking dataset. The goal is to uncover patterns and relationships between physical activity, physiological metrics, and subjective wellness indicators using Python data science libraries.

The project investigates four core research questions:

1. **How does activity time affect mood?**
2. **How does workout type affect calories burned?**
3. **What correlations exist among all numeric variables?**
4. **Can we predict calories burned from activity features?**

## Dataset

**File:** `Fitness_Health_Tracking_Dataset_with_Missing_Values.xlsx`

The dataset contains per-session fitness records for multiple users. Each row represents one workout session logged by a user. After cleaning and aggregating, each user is represented by a single row (average of all their sessions).

### Columns

| Column | Type | Description |
| `User_ID` | Integer | Unique identifier for each user |
| `Age` | Float | Age of the user in years |
| `Height (cm)` | Float | User's height in centimetres |
| `Weight (kg)` | Float | User's weight in kilograms |
| `Steps_Taken` | Float | Number of steps taken during the session |
| `Calories_Burned` | Float | Estimated calories burned during the session |
| `Hours_Slept` | Float | Hours of sleep the night before |
| `Water_Intake (Liters)` | Float | Water consumed in litres |
| `Active_Minutes` | Float | Total active minutes during the session |
| `Heart_Rate (bpm)` | Float | Average heart rate in beats per minute |
| `Stress_Level (1-10)` | Float | Self-reported stress level on a 1–10 scale |
| `Workout_Type` | String | Type of workout performed (e.g., Cardio, Strength) |
| `Mood` | String | Self-reported mood (Sad, Neutral, Stressed, Happy) |

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, aggregation, and manipulation |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Base plotting and chart customisation |
| `seaborn` | Statistical visualisations (box plots, violin plots, heatmaps) |
| `scipy.stats` | ANOVA tests for group comparisons |
| `sklearn.linear_model` | Linear regression modelling |
| `sklearn.model_selection` | Train/test splitting |
| `sklearn.metrics` | Model evaluation (RMSE, R²) |

## Implementation Design

The notebook is structured as a sequential pipeline. Rather than using class-based design, it follows a **functional and procedural** approach where each code section handles one analytical step. The one formal function defined is `buildCalorieModel()`.

## Code Sections

### 1. Data Loading and Preprocessing

**What it does:**

- Loads the Excel file using `pandas.read_excel()`
- Drops rows with missing values using `dropna()`
- Aggregates multiple sessions per user into one row per user using `groupby('User_ID').agg()`
  - Numeric columns: averaged with `'mean'`
  - Categorical columns: mode selected with `lambda x: x.mode()[0]`

**Why per-user aggregation matters:** Without this step, users with more recorded sessions would be over-represented in correlation and regression analyses, introducing bias.

### 2. Visualisation — Active Minutes by Mood (Box Plot)

**Chart type:** `sns.boxplot`

**Purpose:** Answers **Question 1** — whether more active minutes correlates with a better mood.

**Design choices:**
- Mood categories are ordered meaningfully: `['Sad', 'Neutral', 'Stressed', 'Happy']`
- Box plots are ideal here because they show the median, interquartile range, and outliers for each group simultaneously

**Finding:** The box plots show largely overlapping distributions across mood groups. This is consistent with the near-zero correlations found later, confirming the dataset is synthetically generated without embedded real-world relationships.

### 3. Visualisation — Calories Burned by Workout Type (Box Plot)

**Chart type:** `sns.boxplot`

**Purpose:** Answers **Question 2** — whether different workout types lead to different calorie expenditure.

**Design choices:**
- `palette='Set2'` for visually distinct but accessible group colours
- Box plots again chosen for their ability to compare distributions across multiple groups

**Finding:** No statistically significant difference in calories burned across workout types, confirmed by the ANOVA results (all p-values > 0.05).

### 4. Visualisation — Heart Rate by Workout Type (Violin Plot)

**Chart type:** `sns.violinplot`

**Purpose:** Extends Question 2 by examining whether workout type influences average heart rate.

**Design choices:**
- Violin plots are used here instead of box plots because they reveal the full shape of the distribution, not just summary statistics
- `inner='box'` adds a mini box plot inside each violin for median reference

**Finding:** Heart rate distributions are nearly identical across workout types — again confirming synthetic data with no embedded domain logic.

### 5. Histograms — Active Minutes and Steps Taken Distributions

**Chart type:** `plt.hist`

**Purpose:** Shows the overall spread of two key activity metrics, with mean and median reference lines overlaid.

**Design choices:**
- Mean line in red (dashed), median line in orange (solid) for easy differentiation
- `bins=15` balances granularity and readability

**Insight:** Both distributions appear roughly uniform or slightly right-skewed — typical of randomly generated synthetic data rather than real-world bimodal fitness distributions.

### 6. Correlation Matrix

**Method:** `numeric_df.corr()` — Pearson correlation coefficients between all numeric columns.

**Output:**
- A `pandas` correlation DataFrame (printed in the notebook)
- A lower-triangle heatmap using `sns.heatmap` with `mask=np.triu(...)` to avoid redundancy
- A ranked table of the top 10 absolute pairwise correlations
- A horizontal bar chart of those top correlations, colour-coded by direction (green = positive, red = negative)

**Answers Question 3.**

**Key finding:** Out of 45 numeric variable pairs, zero showed strong correlations (|r| ≥ 0.3) and only one showed moderate correlation (0.1 ≤ |r| < 0.3). This conclusively confirms the dataset is synthetically generated with no embedded real-world relationships.

### 7. ANOVA Tests

**Method:** `scipy.stats.f_oneway()`

Two separate one-way ANOVAs are performed:

- **Workout Type vs all numeric variables** — tests whether any numeric metric differs significantly between workout type groups
- **Mood vs all numeric variables** — tests whether any numeric metric differs significantly between mood groups

**Significance threshold:** p < 0.05

**Findings:**
- Workout Type: no significant differences found across any numeric variable
- Mood: one borderline significant result — `Height (cm)` by Mood group (p = 0.0012), almost certainly a spurious correlation in random data

**Limitation:** ANOVA assumes approximately normal distributions and equal variances across groups (homoscedasticity). These assumptions are not formally tested here.

### 8. `buildCalorieModel()` — Linear Regression

**Purpose:** Answers **Question 4** — can we predict calories burned from activity features?

#### Function Signature
```python
def buildCalorieModel(dataframe):
```

#### Parameters
| Parameter | Type | Description |
|---|---|---|
| `dataframe` | `pandas.DataFrame` | The cleaned, per-user aggregated DataFrame |

#### Attributes / Internal Variables

| Variable | Description |
|---|---|
| `features` | NumPy array of input features: `Active_Minutes` and `Steps_Taken` |
| `target` | NumPy array of the target variable: `Calories_Burned` |
| `X_train`, `X_test` | 80% / 20% split of feature data |
| `y_train`, `y_test` | 80% / 20% split of target data |
| `model` | `sklearn.linear_model.LinearRegression` instance |
| `y_pred` | Predicted calorie values on the test set |
| `rmse` | Root Mean Square Error — measures average prediction error in calorie units |
| `r2` | R² Score — proportion of variance in calories explained by the model (0 = worst, 1 = perfect) |

#### Methods Called Internally

| Method | Purpose |
|---|---|
| `train_test_split(..., test_size=0.2, random_state=42)` | Reproducible 80/20 split |
| `model.fit(X_train, y_train)` | Trains the linear regression on training data |
| `model.predict(X_test)` | Generates predictions on held-out test data |
| `mean_squared_error(y_test, y_pred) ** 0.5` | Computes RMSE |
| `r2_score(y_test, y_pred)` | Computes R² |

#### Output
- Prints model coefficients, intercept, RMSE, and R² to the console
- Produces a scatter plot of actual vs predicted calories, with a red dashed line representing a perfect prediction (slope = 1)

#### Finding

The R^2 score is approximately **-0.0008** which indicates that the model explains almost no variance in calories burned. This is expected: since the dataset was synthetically generated, there is no embedded relationship between activity minutes, steps and calories, so linear regression will not be able to find a meaningful pattern.

#### Limitation

- It uses only two features (`Active_Minutes`, `Steps_Taken`). The real world model would have additional predictors of weight, age and heart rate.
- Assumes a linear relationship between features and calories in the model. Non-linear models (e.g. Random Forest, Gradient Boosting) might be able to find latent structure even in synthetic data.
- No feature scaling (standardization or normalization) used. Although we don't strictly need to scale our data when we use linear regression, it is a good idea when your features are on very different scales and units.

## Key Limitations of the Dataset

1. **Synthetic data**: The dataset was synthetically generated and no real-world correlations exist. All analysis findings must be interpreted in this context – the project shows the *methodology*, not real health insights.

2. **Dropped missing values:** Drop rows which have any missing values in any column (`dropna()`) A more robust approach would preserve more of the data by imputing missing values with the median or a KNN imputer.

3. **Session aggregation loses detail:** Averaging all sessions per user collapses within-person variation.  A different structure would be needed for longitudinal or repeated-measures analyses.

4. **No feature engineering**: No derived features (e.g. BMI from height and weight, or activity intensity categories), which reduces the richness of regression models.

5. **No binary/categorical encoding.** Mood and Workout Type are used only for grouping in visualizations and ANOVA. They are not encoded and fed into the regression model.

## How to Run

1. Ensure the dataset file `Fitness_Health_Tracking_Dataset_with_Missing_Values.xlsx` is in the same directory as the notebook.
2. Install required libraries if not already present:
   ```
   pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
   ```
3. Run all cells in order. Each section builds on the cleaned, aggregated DataFrame produced at the top of the notebook.

## Summary of Findings

| Research Question | Finding |
|---|---|
| Does activity time affect mood? | No significant relationship found. Active minutes distributions are nearly identical across all four mood groups. |
| Does workout type affect calories burned? | No significant difference. ANOVA p-values > 0.05 for all variables across workout types. |
| What correlations exist among numeric variables? | Virtually none. 44 of 45 variable pairs have \|r\| < 0.1. Dataset is confirmed synthetic. |
| Can we predict calories burned from activity features? | No. R² ≈ −0.0008. Active minutes and steps have no predictive power for calories in this dataset. |

All findings are consistent with the dataset being synthetically generated without embedded domain relationships.
