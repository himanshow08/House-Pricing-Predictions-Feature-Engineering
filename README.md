#  Feature Engineering for Ames House Prices

This repository contains a Jupyter Notebook (`house_prices_dataset_feature_eng_notebook_1100027.ipynb`) detailing an end-to-end feature engineering pipeline for the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

The primary focus of this project is not on model accuracy, but on the systematic process of data cleaning, transformation, and feature creation. Every decision is accompanied by a clear justification and supporting evidence (visualizations or metrics), as required by the assignment.

---

##  Requirements

To run this notebook, you will need:
* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (for `StandardScaler`, `PCA`)
* scipy (for `boxcox`)

---

##  How to Use

1.  Download the `train.csv` file from the [Kaggle competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2.  Place the `train.csv` file in the same directory as the notebook.
3.  Run the Jupyter Notebook **`house_prices_dataset_feature_eng_notebook_1100027.ipynb`** from top to bottom. The cells are designed to be run in sequential order.

---

##  Feature Engineering Pipeline Summary

The notebook follows a systematic pipeline, with each step justified and verified:

1.  **Target Variable Analysis:** The `SalePrice` target variable is analyzed, found to be highly right-skewed (skewness: 1.88), and corrected using a `log1p` transformation (new skewness: 0.12).

2.  **Domain-Aware Imputation:** Missing values are handled based on the data dictionary, not arbitrary rules.
    * **`NA` as "None":** Categorical features (like `Alley`, `PoolQC`, `GarageType`) have their `NA` values correctly imputed with the string `"None"`, as `NA` signifies the *absence* of the feature.
    * **`NA` as 0:** Corresponding numeric features (like `GarageArea`) are filled with `0`.
    * **Grouped Median:** `LotFrontage` is imputed using the median of its `Neighborhood` to improve accuracy.

3.  **Outlier Removal:** Two severe anomalies in `GrLivArea` (identified via scatter plot and documentation) are surgically removed to prevent model skew.

4.  **Feature Creation:** New, high-value features are engineered, including:
    * Aggregated features: `TotalSF`, `TotalBath`, `TotalPorchSF`.
    * Age-based features: `HouseAge`, `RemodAge`, `GarageAge`.
    * `Neighborhood_Tier`: A binned feature that converts the high-cardinality `Neighborhood` variable into a 5-tier ordinal feature based on median sale price.

5.  **Transformation & Encoding:**
    * **Skewness:** `Box-Cox` transformation is applied to 15 highly skewed numeric predictors.
    * **Ordinal Features:** Features with inherent order (e.g., `ExterQual`) are manually mapped to numerical values.
    * **Nominal Features:** All remaining categorical features are one-hot encoded.

6.  **Student-Specific Feature:** A `student_random_feature` is generated (ID: 1100027) and included in the analysis as a control variable. It is correctly shown to have no correlation and no significant loading on any principal components, validating the pipeline.

7.  **Dimensionality Reduction:** `StandardScaler` is applied, followed by `PCA` (Principal Component Analysis) to manage multicollinearity. The final dataset retains 95% of the original variance using **123 components** (down from 211).

---

##  Final Output

The notebook's final outputs are the model-ready datasets:

* `X_pca`: The scaled and reduced feature matrix (123 components).
* `y_train_log`: The log-transformed target variable.
