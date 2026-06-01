MONTHLY_DATA_PATH = "new_valid_users_spend.csv"
WEEKLY_DATA_PATH = "new_valid_users_spend_weekly.csv"

#Column identifiers
TARGET_COL = "new_valid_users_cnt"
EXOG_COL = "spend_amt_usd"

#Validation parameters
MONTHLY_TEST_HORIZON = 12 #for SARIMAX
WEEKLY_TEST_HORIZON = 52 #for XGBoost/LSTM

#Random state seed for reproducibility
RANDOM_STATE = 42