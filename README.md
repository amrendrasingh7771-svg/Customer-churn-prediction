Customer Churn Prediction (IBM Telco Dataset)
A machine learning project that predicts whether a telecom customer will leave (churn), using customer demographics, subscribed services, contract type, and billing information.
I compared three classifiers (Logistic Regression, Decision Tree, Random Forest), evaluated them with precision, recall, and confusion matrices, and diagnosed and reduced overfitting in the tree-based models.
Results at a glance
Best model: Logistic Regression with 80.8% test accuracy
Churn recall: 57.6% of customers who actually churned were caught, at 65.1% precision
Baseline to beat: only 26.5% of customers churn, so always predicting "stays" would score about 73.5%
Overfitting finding: the Decision Tree and the 500-tree Random Forest score about 99.8% on training data but much lower on test data. Limiting tree depth and leaf size cut the Random Forest train-test gap from about 20 points to about 5.
Dataset
Source: IBM Telco Customer Churn dataset
Size: 7,043 customers, 20 predictor features, 1 target column (Churn)
Feature groups:
Demographics: gender, SeniorCitizen, Partner, Dependents
Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Account and billing: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Class balance: 73.5% stayed, 26.5% churned
Approach
Data checks: looked for null values and duplicates and reviewed summary statistics. There are no null values, and 22 duplicate rows appear once the customer ID is removed.
Cleaning TotalCharges: the column loads as text because 11 values are blank. I converted it to numbers (blanks become missing) and filled them with the median.
Removing customerID: it is a unique label with no predictive value, so it was dropped.
Encoding: categorical columns were converted to numbers with Label Encoding.
Split: 80% train / 20% test (random_state=43).
Models: Logistic Regression, Decision Tree, and Random Forest (500 trees), plus a regularized Random Forest (max_depth=10, min_samples_split=5, min_samples_leaf=5, 100 trees).
Evaluation: accuracy on train and test sets, precision, recall, and confusion matrices.
Results
Test set of 1,409 customers:
Model	Train accuracy	Test accuracy	Churn precision	Churn recall
Logistic Regression	80.3%	80.8%	65.1%	57.6%
Random Forest (500 trees)	99.8%	79.6%	64.0%	50.5%
Random Forest (regularized)	85.2%	80.4%	66.5%	51.1%
Decision Tree	99.8%	74.6%	51.5%	55.1%
Logistic Regression confusion matrix (rows are actual, columns are predicted):
Predicted stays	Predicted churns
Actually stayed	925	114
Actually churned	157	213
What the results show
Logistic Regression is the most stable model: its train and test accuracy are within 0.5 points.
The unpruned Decision Tree and the 500-tree Random Forest memorize the training data (about 99.8% train accuracy) and generalize worse (74.6% and 79.6% on test).
Constraining tree depth and leaf size cuts the Random Forest gap from about 20 points to about 5 (85.2% train vs. 80.4% test).
Accuracy alone is not enough here. Because most customers stay, the churn-class recall (how many churners are caught) is the more useful number for a retention team.
Results move by about one point depending on the random seed.
Predicting for a new customer
After training, the model can score a new customer profile (values must use the same encoding as the training data):
python
new_customer = pd.DataFrame({
    "gender": [1], "SeniorCitizen": [0], "Partner": [0], "Dependents": [0],
    "tenure": [2], "PhoneService": [1], "MultipleLines": [0],
    "InternetService": [0], "OnlineSecurity": [2], "OnlineBackup": [2],
    "DeviceProtection": [0], "TechSupport": [0], "StreamingTV": [0],
    "StreamingMovies": [0], "Contract": [0], "PaperlessBilling": [1],
    "PaymentMethod": [3], "MonthlyCharges": [53.85], "TotalCharges": [108.15],
})
print("Predicted churn:", model.predict(new_customer)[0])  # 1 = churn, 0 = stays
Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
How to run
bash
# 1. Clone the repository
git clone https://github.com/amrendrasingh7771-svg/<repository-name>.git
cd <repository-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 3. Open the notebook
jupyter notebook Telecom_churn_.ipynb
Place the Telco dataset CSV in the project folder and update the file path in the first data-loading cell.
Limitations and next steps
Label Encoding on categories: it assigns an arbitrary order to unordered categories such as PaymentMethod. One-Hot Encoding would suit Logistic Regression better.
Feature scaling: Logistic Regression shows a convergence warning on unscaled data. Adding StandardScaler should fix it.
Class imbalance: recall for churners is under 60%. Class weights or tuning the decision threshold could catch more churners.
Single split: results come from one train-test split. Cross-validation would give a more reliable estimate.
Model tuning: hyperparameters were set by hand. A grid or randomized search could improve the tree models.
Author
Amrendra Singh – Data Analyst | Data Science & AI/ML Fresher Lucknow, India · LinkedIn · GitHub · amrendrasingh7771@gmail.com

