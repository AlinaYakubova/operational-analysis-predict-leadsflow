import pandas as pd

def main():
    df = pd.read_csv("data/new_valid_users_spend_monthly.csv")
    print(df.head())

if __name__ == "__main__":
    main()