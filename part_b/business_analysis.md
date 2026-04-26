# Scenario: Promotion Effectiveness at a Fashion Retail Chain
 ## B1. Problem Formulation(a)
 **(a) Machine Learning Formulation** 
 To maximize sales, we need to predict the outcome of various promotional "treatments" based on store-specific contexts.
 - **Target Variable (y)**: The number of items sold (Sales Volume) per store per month.
 - **Candidate Input Features (X)**:
    - **Store Characteristics:** Location type (Urban, Semi-urban, Rural), store size (sq. ft), and local competition density.
    - **Demographics:** Average income level, age distribution, and loyalty member density of the local area.
    - **External Factors:** Month/Seasonality, footfall trends, and holiday markers.
    - **Actionable Variable:** The Promotion Type (the 5 categories).
- **Problem Type:** This is a **Supervised Learning: Regression** problem.
    - **Justification:** The target (items sold) is a continuous numerical value. While the promotions are categorical, the model's goal is to estimate a quantity. Once the model is trained, the retailer can perform Prescriptive Analytics by "plugging in" all five promotion types for a specific store and selecting the one that yields the highest predicted volume.

**(b) Target Variable: Sales Volume vs. Revenue**
While revenue keeps the lights on, Items Sold is often a cleaner signal for "promotion effectiveness" for the following reasons:
1. **Price Neutrality:** Revenue can be skewed by high-ticket items. Selling one $500 coat generates more revenue than ten $30 t-shirts, but the t-shirt sale might indicate a much more successful "BOGO" campaign
2. **Inventory Impact:** Promotions are often used to clear stock or increase market penetration. Volume directly measures consumer engagement and "basket size," whereas revenue can be heavily influenced by price changes or inflation.
3. **The Broader Principle:** This illustrates Alignment with Objective. In ML, the target variable must be a direct proxy for the behavior you wish to influence. If you want to measure customer response, count the responses (items sold); if you want to measure financial margin, use profit. Choosing a variable that is too "noisy" (like revenue, which fluctuates with price) leads to poor model generalization.

**(c) Alternative Modeling Strategy**
A "Global Model" risks washing out the nuances of rural vs. urban behavior. Instead, I propose a **Clustered Modeling Approach** (or Segmented Models).

**The Strategy:**
1. **Segment the Stores:** Group the 50 stores into clusters (e.g., using K-Means) based on location type and demographics rather than just geography.
2. **Train Segment-Specific Models:** Create three distinct models—one for Urban, one for Semi-urban, and one for Rural.
3. **Justification:** * **Feature Interaction:** Rural customers might be highly sensitive to "Flat Discounts" due to price consciousness, while Urban customers might prefer "Loyalty Points" or "Free Gifts."
    - **Variance Reduction:** A global model might try to find an "average" that fits no one. Segmented models allow the weights of the input features to differ significantly. For instance, "Competition Density" might be a massive predictor in Urban centers but statistically irrelevant in isolated Rural areas.

## B2. Data and EDA Strategy
**(a) Data Integration and Grain**
To build the model, we must move from raw transactional logs to a structured "Snapshot" of store performance.
- **The Join Logic:**
    - Start with the **Transactions** table.
    - Left Join **Store Attributes** on store_id.
    - Left Join **Calendar** on transaction_date.
    - Join **Promotion Details** based on the promotion active in that specific store during that date range.
- **Final Modeling Grain: One row = One Store-Month**
     - Since the team runs one promotion per store per month, the model needs to see the cumulative effect of that decision.
- **Aggregations Required:**
    - **Sum** of items_sold (Target variable).
    - **Average** monthly_footfall.
    - **Count** of festival_days and weekends in that month.
    - **Mode** of customer_demographic (to capture the primary shopper type for that month).

