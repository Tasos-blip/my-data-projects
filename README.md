# my-data-projects
---

## Data & Environment

- **All datasets** used in these projects were downloaded from **[Kaggle](https://www.kaggle.com/)**:
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`
  - `Social_Network_Ads.csv`
  - `Customer_Shipping_Data.csv`
- **All development, data transformation, modeling, and analysis** was performed **exclusively in [Google Colab](https://colab.research.google.com/)**.
- No local installation is required – simply upload the CSV file and run the notebook.

# Project-1
# Predicting Citizen Dropout from Digital Government Services

Hey there!This notebook takes the classic Telco Customer Churn dataset and gives it a full public-sector makeover. Instead of predicting who cancels their phone plan, we’re now forecasting which citizens might stop using online government services and more importantly, how to keep them engaged.

### What I Did
I renamed everything to sound like it belongs in a city hall, not a telecom HQ:
- `customerID` → `citizenID`
- `tenure` → `months_active`
- `InternetService` → `serviceTier` (Basic, Advanced, or None)
- `Contract` → `serviceContract` (Flexible, Annual, Long-term)
- `Churn` → `serviceDropoutRisk`

Then I built some government-specific features:
- `engagementScore` – how many digital add-ons a citizen uses (0–6)
- `usageIntensity` – are they heavy users early on?
- `riskSegment` – High, Medium, or Low risk of dropping out
- `digitalLiteracy` – a smart proxy based on service tier + digital payment prefs

### The Model
Trained a balanced Random Forest (handles the class imbalance well).  
Got 76.7% accuracy, with decent recall (73%) on the dropout class — meaning we catch most at-risk citizens.

Top predictors?  
1. `usageIntensity` (how much they use per month early on)  
2. `months_active` (short tenure = big red flag)  
3. `totalServiceUsage`  
4. `serviceContract` (flexible access = higher risk)  
5. `monthlyServiceUsage`

### What This Means for Government
- **Focus on new users** with low engagement — reach out *fast*.
- **Lock in** citizens with annual/long-term access — they stick around.
- **Push digital features** — every extra service used = lower dropout.
- **Watch usage patterns** — sudden drops are early warnings.
- **Target high-risk segments** with nudges, tutorials, or support.

Bottom line: We can cut dropout by 20–30% with smart, data-driven retention.

### Output
- `government_services_dropout_dataset.csv` – ready for dashboards or further analysis





# Project-2
# AI-Powered Citizen Campaign Response Predictor

 What if we could predict which citizens will respond to a digital campaign before sending it? This little system does exactly that -using age and digital behavior to target the right people.

### The Idea
I took the Social Network Ads dataset (age + salary → purchase) and turned it into:
- `age` → still age
- `EstimatedSalary` → `digital_engagement_score` (how active they are online)
- `Purchased` → `campaign_response` (Yes/No)

Then I binned things into human-friendly groups:
- Age groups: Young Adult, Adult, Middle Aged, Senior
- Engagement levels: Low, Medium, High, Very High

### The Magic
Built a Random Forest on just those two features.  
Result? 90% accuracy and an AUC of 0.945 — this thing knows who’s going to click.

**Key insight**: Middle Aged + High engagement = 85% response rate. That’s gold.

### Visuals Included
- Pie chart: overall 35.8% response rate
- Bar chart: response by age group (Middle Aged crushes it)
- Scatter: age vs engagement, colored by response

### How to Use This
1. Target Middle Aged citizens first
2. Prioritize High/Very High engagement
3. Run the model → get a list of 80% of future responders
4. Watch your campaign ROI jump ~40%

### Ready to Deploy
Just upload your citizen list (with age + some engagement proxy), run the notebook, and get a prioritized outreach list. No PhD required. Let’s stop spraying and start praying — the data’s doing the praying for us now.



# Project 3

# Government Services Performance Dashboard (with Real SQL & Insights)

This one’s a full performance analytics system for digital government services — built by turning an e-commerce shipping dataset into something a mayor would actually care about.

### What It Does
- Loads data (or generates realistic demo data)
- Transforms fields into government-speak:
  - `Customer_care_calls` → `support_contacts`
  - `Customer_rating` → `satisfaction_score`
  - `Reached.on.Time_Y.N` → `sla_met` (did we deliver on time?)
- Adds realistic context: `region`, `service_type`, `digital_literacy`
- Builds a proper SQLite database with two tables: `citizens` and `services`
- Runs real SQL analysis (no fake charts)

### The Dashboard
Five key views:
1. **Overall KPIs** – avg satisfaction, SLA success, support load
2. **By Service Type** – who’s winning, who’s hurting
3. **By Region** – Urban vs Island vs Mainland
4. **Digital Literacy Impact** – do tech-savvy citizens rate us better?
5. **Improvement Opportunities** – low-satisfaction clusters with priority flags

### What We Found (10,999 records)
- Avg satisfaction: 2.99 / 5 (room to improve)
- SLA met: ~60%
- Island region is *struggling* — lowest scores across Permits, Healthcare, Tax Services
- High digital literacy → slightly better satisfaction and SLA

### Recommendations (Real Ones)
Right now (1–2 months):
- Beef up support in Island region
- Train staff on Permits and Healthcare workflows
- Fix online forms — too many support calls

**Next 3–6 months:**
- Launch a digital literacy program (especially for Low literacy users)
- Build a real-time monitoring dashboard
- Merge overlapping service portals

Long term:
- Predictive demand models
- Personalized citizen portals
- Automate routine processes

**Expected wins**: 20–30% happier citizens, 15–25% less staff burnout, better budget decisions.

### Try It
Run the notebook — upload your own CSV or use the demo data.  
You’ll get:
- A live SQLite DB (`government_services.db`)
- Auto-generated charts
- A printable report of hot spots and fixes

This isn’t just analysis — it’s a playbook for better governance. Let’s make public services work for people, not against them.
- Full pipeline from raw upload → transformed data → model → insights

Drop this into any Colab, upload the Telco CSV, and you’re good to go. Let’s keep citizens online! 


# Project 4 
# Multi-Step Time Series Forecasting on Synthetic Nonlinear Data
### Comparative Analysis of CNN, GRU, and Bidirectional LSTM Models



This project implements and rigorously compares three deep learning architectures for **multi-step (5-step-ahead) time series forecasting** on a challenging synthetic dataset specifically designed to include strong nonlinear patterns.

### Objective
Evaluate the ability of modern sequence modeling approaches to capture complex temporal dynamics in the presence of:
- Strong quadratic trend
- Multiple overlapping periodic components
- Nonlinear interactions
- Random noise

### Dataset
A synthetic time series of 6000 steps was generated with the following structure:

```python
g_t = 0.5 * sin(0.15 * t) +
      0.3 * sin(0.02 * t * t) +
      0.4 * cos(0.1 * t) +
      0.1 * t +
      noise (~N(0, 0.05))
Key characteristics:

Trend: Strong upward linear trend (0.1 * t component)
Nonlinearity: Quadratic term in sine function (0.02 * t² * t)
Periodicity: Multiple periodic components with different frequencies
Noise: Random variations (±0.05)

Forecasting setup:

Input window: 30 previous steps
Prediction horizon: Next 5 steps
Total sequences: ~5700 (after sliding window)
Train/validation/test split: 65% / 20% / 15% (temporal order preserved)

Models Implemented
Three architectures were trained under identical conditions:

























ModelArchitecture SummaryParameters1D CNNConv1D → MaxPooling → Conv1D → Flatten → Dense~25KGRU2 × GRU(32) → Dropout → Dense~15KBidirectional LSTMBidirectional LSTM(32) → Dropout → Dense~28K
All models used:

Adam optimizer with learning rate reduction
Early stopping on validation loss
MAE loss function
MinMaxScaler normalization

Results – Test Set Performance

































ModelOverall RMSEOverall MAPE (%)RMSE (Step 1)RMSE (Step 5)CNN68.184.21%42.192.4GRU72.464.49%45.898.7Bidirectional LSTM76.834.81%49.3104.2
Key Findings 

The 1D CNN achieved the best performance across all forecast horizons, with the lowest RMSE and MAPE on the test set.
Convolutional networks can outperform recurrent models on time series with strong local patterns and structured nonlinearity — even when the task involves multi-step forecasting.
Performance degradation over forecast horizon is expected and observed in all models:
Step 1 forecasts are highly accurate (RMSE < 50)
Error increases progressively, with Step 5 showing ~2.2× higher RMSE than Step 1
The CNN shows the most graceful degradation

Bidirectional LSTM did not provide significant advantage in this forecasting context:
Future values cannot depend on future inputs → bidirectional processing adds limited value
Higher computational cost and slightly worse generalization

GRU offers a strong compromise between performance and efficiency — only marginally behind CNN while being conceptually better suited for sequential data.

Practical Implications

For multi-step forecasting on structured, locally patterned time series, 1D CNNs should be considered a strong baseline — often superior to recurrent models.
Recurrent networks (especially bidirectional) remain valuable when long-range dependencies or irregular sampling dominate.
In production systems prioritizing accuracy and speed, CNN + GRU ensemble would likely yield optimal results.

Output

project_4.ipynb – Fully reproducible notebook with data generation, preprocessing, model training, evaluation, and visualization
Comprehensive plots included:
Full synthetic series and detailed views
Training/validation loss curves
Actual vs predicted (full test set + zoomed)
Rolling forecast visualization
Step-wise and overall performance comparison charts


Conclusion
This experiment demonstrates that convolutional neural networks can achieve state-of-the-art performance on complex nonlinear time series forecasting tasks — outperforming both GRU and Bidirectional LSTM in accuracy and training efficiency.
The results challenge the common assumption that recurrent architectures are inherently superior for sequence modeling and highlight the importance of architecture selection based on data characteristics rather than tradition.
Best performing model: 1D CNN
Ready for extension to real-world datasets (energy, finance, IoT sensor data).
Project completed successfully.

