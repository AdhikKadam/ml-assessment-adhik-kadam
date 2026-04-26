# Scenario: Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### (a) Machine Learning Formulation
To determine the optimal promotion strategy for individual store locations, the scenario is formulated as a **Supervised Learning: Regression** problem, specifically geared toward continuous count estimation. Alternatively, it can be framed as an **Uplift Modeling** problem to measure incremental causal impact.

* **Target Variable ($y$):** Sales Volume (number of items sold per store per month). 
* **Candidate Input Features ($X$):**
    * **Store Meta-Features:** Floor space (sq. ft.), location category (Urban, Semi-urban, Rural), local competition density index.
    * **Demographic Context:** Local median income, average age, loyalty program penetration rate.
    * **Temporal & Contextual Features:** Seasonality index, holiday/festival flags, historical baseline footfall.
    * **Treatment Variable:** Promotion Type (Categorical: Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Bonus).
* **Justification:** The primary business objective is to maximize a quantitative output (items sold). A regression model allows for the estimation of expected sales volume for any given promotion type. In a production environment, the model will simulate all five promotional treatments for a given store-month, and the business will select the treatment yielding the highest predicted volume (Prescriptive Analytics).

### (b) Target Variable: Sales Volume vs. Total Revenue
Measuring performance via Total Sales Revenue introduces critical statistical noise due to price variance. **Sales Volume (Items Sold)** is the superior target variable for the following reasons:

1. **Price Isolation:** Revenue is heavily skewed by item price points. A highly successful promotion moving hundreds of low-cost items (e.g., socks) might show a lower revenue spike than a poorly performing promotion moving a few high-cost items (e.g., leather jackets). Volume directly measures the core objective: consumer engagement and transaction generation.
2. **Margin and Cannibalization:** Certain promotions (like Flat Discounts) artificially depress revenue per item. Evaluating by revenue penalizes the promotion for doing exactly what it was designed to do (lower price to drive volume).
3. **Broader ML Principle (Objective Alignment):** This illustrates the principle of **Proxy Variable Alignment**. In applied machine learning, the mathematical target variable must align directly with the specific behavior the algorithm is meant to optimize. If the model optimizes for revenue, it will bias toward stores that sell luxury goods, regardless of promotional effectiveness.

### (c) Alternative Modelling Strategy
A single global model suffers from underfitting local nuances, while training 50 independent models risks high variance and overfitting due to limited sample sizes per store. 

**Proposed Alternative: Hierarchical (Mixed-Effects) Modeling or Clustered Segmentation**
1. **Cluster-Based Strategy:** Apply an unsupervised clustering algorithm (e.g., K-Means or DBSCAN) to group stores based on multidimensional similarities (e.g., high-income rural, high-competition urban) rather than geography alone. Train separate LightGBM or XGBoost models for each cluster.
2. **Hierarchical Modeling:** Alternatively, utilize an architecture that learns global patterns (fixed effects) while allowing coefficients to vary by store or region (random effects). 
3. **Justification:** This approach effectively captures non-linear feature interactions (e.g., Rural customers reacting exponentially better to Flat Discounts) while sharing statistical strength across statistically similar stores, preventing the "average of everything fits nothing" dilemma of a single global model.

---

## B2. Data and EDA Strategy

### (a) Data Integration and Grain
The transition from transactional logs to a model-ready dataset requires careful joins to prevent temporal data leakage. 

* **Join Architecture:**
    1. Base: `calendar` table filtered to the target historical window.
    2. Cross Join with `store_attributes` to create a complete matrix of all stores for all dates.
    3. Left Join `promotion_details` on `store_id` and `date`.
    4. Left Join aggregated `transactions` on `store_id` and `date`.
* **Modelling Grain:** **One row = One Store-Month.**
* **Aggregations Performed (Pre-Modelling):**
    * **Target:** `SUM(items_sold)` partitioned by store and month.
    * **Temporal:** `COUNT(festival_days)` and `COUNT(weekend_days)` per month.
    * **Behavioral Lags:** `AVG(items_sold_previous_3_months)` to establish a rolling baseline.
    * **Contextual:** `AVG(monthly_footfall)`.

### (b) Exploratory Data Analysis (EDA)
A robust EDA phase dictates feature engineering priorities. Key analyses include:

| Analytical Technique | Diagnostic Purpose | Impact on Modelling Decisions |
| :--- | :--- | :--- |
| **Distribution Analysis (Violin Plots)**<br>*(Items Sold by Promo Type)* | Identify median performance and variance spreads across different promotions. | If distributions are highly skewed (e.g., rare massive sales spikes), the target variable may require a log transformation (e.g., $log(1 + y)$). |
| **Multicollinearity Check (VIF / Correlation Matrix)** | Identify independent variables that provide redundant information (e.g., Store Size and Baseline Footfall). | Highly correlated features will be pruned or subjected to Principal Component Analysis (PCA) to stabilize model coefficients. |
| **Time-Series Decomposition** | Separate the dataset into Trend, Seasonality, and Residual components. | Informs the creation of temporal features (e.g., month-of-year indicators) to ensure the model attributes sales spikes to the promotion, not just organic holiday lifts. |
| **Interaction Plots**<br>*(Promo Type vs. Location Category)* | Visualize if the slopes of sales lift intersect when comparing promotion types across different store geographies. | Verifies the hypothesis from B1(c). If lines intersect sharply, it mathematically confirms that location-based clustering or explicit interaction terms are strictly necessary. |

### (c) Mitigating Class Imbalance (80% No-Promotion Baseline)
If 80% of the training data represents the "No Promotion" control state, standard regression models will bias heavily toward baseline averages, treating the 20% promotional lifts as statistical outliers to be smoothed over.

**Corrective Strategies:**
1. **Target Transformation (Lift Prediction):** Instead of predicting absolute sales, predict the **Incremental Lift**. Calculate the historical rolling average for a store without promotions, subtract it from the actual sales, and train the model exclusively on the delta ($\Delta y$). This normalizes the 80% baseline to near zero.
2. **Causal Inference (Meta-Learners):** Deploy an Uplift Modeling framework (such as a T-Learner). Train Model A strictly on the 80% "No Promotion" data to establish a highly accurate baseline. Train Model B on the 20% "Promotion" data. The true predicted effect is the difference between Model B and Model A.
3. **Sample Weighting:** If maintaining a single model, apply Cost-Sensitive Learning by utilizing inverse frequency weighting in the algorithm's loss function, penalizing the model more heavily for errors made on the minority (promotional) class rows.



## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Evaluation Strategy

**The Splitting Strategy: Temporal (Chronological) Split**
With three years of monthly data, a **Time-Series / Chronological Split** is strictly required. For example:
* **Training Set:** Years 1 and 2 (Months 1-24).
* **Validation/Test Set:** Year 3 (Months 25-36) or an expanding window forward-chaining cross-validation.

**Why a Random Split is Inappropriate:**
A random split (like `train_test_split` in scikit-learn) would cause **Temporal Data Leakage**. If you randomly shuffle the data, the model might train on December 2025 data and then be asked to "predict" November 2025. In the real world, you cannot use future knowledge to predict the past. Retail is highly seasonal; the model must prove it can learn from past cycles to predict *future, unseen* cycles.

**Evaluation Metrics & Interpretation:**
To evaluate regression performance in a business context, I would use:

1.  **Mean Absolute Error (MAE):** * *What it is:* The absolute average difference between predicted and actual items sold.
    * *Business Interpretation:* "On average, our model's volume prediction is off by $\pm 150$ items per store." It is highly intuitive for stakeholders.
2.  **Root Mean Squared Error (RMSE):**
    * *What it is:* Similar to MAE, but squares the errors before averaging, penalizing larger mistakes heavily.
    * *Business Interpretation:* If under-stocking a highly successful promotion by 1,000 items is disproportionately more damaging to the brand than missing by 100 items ten times, RMSE will highlight these severe misses.
3.  **Weighted Mean Absolute Percentage Error (WMAPE):**
    * *What it is:* Sum of absolute errors divided by the sum of actual values.
    * *Business Interpretation:* "The model has an overall error rate of 8% across total sales volume." It prevents large stores (which naturally have larger absolute errors) from entirely skewing a standard MAPE metric.

### (b) Investigating and Communicating Time-Varying Recommendations

If the model recommends different promotions for Store 12 in December vs. March, it demonstrates that the model is correctly utilizing dynamic features rather than just memorizing static store attributes. 

**Investigation Strategy: Local Feature Importance (SHAP)**
Global feature importance tells us what drives the model overall, but to explain specific, individual predictions, I would use **SHAP (SHapley Additive exPlanations) values**. SHAP breaks down exactly how much each feature contributed to pushing a specific prediction higher or lower than the baseline average.

**Communication to the Marketing Team:**
I would present the team with two **SHAP Waterfall Charts** (one for December, one for March) and explain the narrative the data tells:

* **The December Story:** "In December, the SHAP chart shows that the `Holiday_Season` flag and `Historical_High_Footfall` features pushed the predicted sales for the 'Loyalty Points' promotion exceptionally high. Customers are already buying gifts (footfall is naturally high); rewarding them with points capitalizes on this volume without sacrificing the immediate margin like a Flat Discount would."
* **The March Story:** "In March, the SHAP chart shows a negative impact from `Off_Peak_Season` and `Low_Footfall`. However, the 'Flat Discount' prediction was pushed up because historical data shows this specific location responds elastically to direct price cuts during slow months. The model is using the discount to artificially generate footfall when organic traffic is low."

### (c) End-to-End Deployment and Monitoring

To operationalize the model for monthly zero-touch inference, we need a robust MLOps pipeline.

**1. Model Serialization:**
After final training, the model artifact (e.g., an XGBoost or LightGBM object) and the entire preprocessing pipeline (imputers, scalers, one-hot encoders) are serialized together using tools like `joblib`, `pickle`, or converted to an `ONNX` format. This artifact is stored in a model registry (e.g., MLflow, AWS SageMaker).

**2. The Monthly Inference Pipeline (Batch Processing):**
A job scheduler (like Apache Airflow or an AWS EventBridge CRON job) triggers at the end of every month:
* **Data Prep:** An automated SQL script queries the data warehouse for the upcoming month's calendar features, calculates the latest rolling footfall baselines, and pulls current store attributes.
* **The Matrix:** The script creates a "scoring grid." For each of the 50 stores, it creates 5 identical rows. It then injects one of the 5 promotion types into each row, resulting in 250 rows total.
* **Prediction:** The saved model is loaded into memory, processes the 250 rows, and outputs predicted `items_sold` for each.
* **Optimization:** A script groups the results by `store_id`, applies an `argmax()` function to select the promotion with the highest predicted volume, and exports the final 50 recommendations to a dashboard (like Tableau) or directly to the marketing team's CRM.

**3. Monitoring and Degradation Detection:**
Models degrade over time as consumer behavior changes (e.g., a recession alters spending habits).
* **Data Drift Monitoring:** Track the distributions of incoming monthly features against the training data. If a store's average footfall drops by 40% due to local construction, an alert is triggered indicating the model is operating in an unseen data space.
* **Performance Monitoring (Concept Drift):** At the end of every month, join the actual sales data back to the predictions. Calculate the rolling MAE and WMAPE. 
* **Retraining Trigger:** Set a hard threshold (e.g., "If WMAPE exceeds 12% for two consecutive months"). When breached, an alert is sent to the data science team to investigate, append the newest data, and retrain the model.