# screen_time_py

This project aims to predict the `Work_Productivity_Score` based on smartphone usage and other related factors. We investigate various regression models to determine their effectiveness and identify the most influential predictors of productivity.

## Data Source
The dataset used for this analysis is the "Smartphone usage productivity dataset", extracted from Kaggle.

## Problem Statement
Can different variables reliably predict the productivity of people? What are the most influential predictors of productivity based on the collected data?

## Methodology

### 1. Data Import & Initial Exploration
- The dataset was loaded from a CSV file.
- Initial checks were performed to understand data structure (`df.info()`), identify missing values (`df.isnull().sum()`), and get descriptive statistics for numerical columns (`df.describe()`).
- Value counts for categorical columns (`Gender`, `Occupation`, `Device_Type`) were also examined.

### 2. Data Preprocessing
- **Feature Identification**: Categorical features (`Gender`, `Occupation`, `Device_Type`) and numerical features were identified.
- **One-Hot Encoding**: Categorical features were transformed into numerical format using one-hot encoding (`pd.get_dummies`) to prepare them for model training. The original categorical columns were dropped.

### 3. Model Training & Evaluation
- **Data Splitting**: The preprocessed dataset was divided into training (80%) and testing (20%) sets, with `Work_Productivity_Score` as the target variable.
- **Regression Models**: The following regression models were trained and evaluated:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - RandomForestRegressor
    - GradientBoostingRegressor
- **Evaluation Metrics**: Models were evaluated using Mean Squared Error (MSE) and R-squared ($R^2$) on the test set.

### 4. Model Predictions Visualization
- A scatter plot comparing actual vs. predicted `Work_Productivity_Score` values from the best-performing model (Lasso) was generated to visually assess its performance.

### 5. Feature Importance Analysis
- **Linear Models (Coefficients)**: Coefficients were extracted from Linear Regression, Ridge, and Lasso models to understand the linear impact of each feature.
- **Tree-based Models (Feature Importances)**: Feature importance scores were extracted from RandomForestRegressor and GradientBoostingRegressor models to identify non-linear relationships.

## Key Findings

### Model Performance
All models, including the best-performing Lasso model, showed very low or negative R-squared values (close to -0.0000 to -0.0295), indicating that none of the models were effectively explaining the variance in `Work_Productivity_Score` with the current features. The MSE values were consistently around 8.3-8.5.

### Feature Importance
- **Linear Models**: Coefficients for Linear Regression and Ridge were very small, suggesting minimal linear impact. Lasso drove all coefficients to zero, implying no strong linear correlation with the target variable after regularization.
- **Tree-based Models**: These models identified a consistent set of top influential features:
    - `Daily_Phone_Hours` (most important)
    - `Social_Media_Hours`
    - `Age`
    - `Weekend_Screen_Time_Hours`
    - `App_Usage_Count`
    - `Sleep_Hours`
- Categorical features (Gender, Occupation, Device_Type) showed much lower importance compared to numerical features in tree-based models.

## Conclusion & Next Steps

The overall low R-squared values suggest that the current features are insufficient to predict `Work_Productivity_Score` effectively, or the relationship is highly non-linear and not captured by these models. 

To improve predictive performance, the following steps are recommended:
- **Feature Engineering**: Explore creating new features or transforming existing ones (e.g., interaction terms, polynomial features) to uncover hidden relationships.
- **Advanced Models**: Consider more complex non-linear models like neural networks.
- **Data Quality/Collection**: Re-evaluate the `Work_Productivity_Score` metric's reliability and consider gathering additional, potentially more relevant, data.
- **Domain Expertise**: Consult domain experts to understand other factors that might influence work productivity.

