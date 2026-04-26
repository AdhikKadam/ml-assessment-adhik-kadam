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