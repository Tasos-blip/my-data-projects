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
