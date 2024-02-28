import pandas as pd
from sklearn.model_selection import train_test_split

# Constants for test and validation set sizes
TEST_SIZE = 0.2
VAL_SIZE = 0.25

def load_data(excel_file):
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print("Error: File not found. Please make sure you've entered the correct file name.")
        exit()
    return df

def preprocess_data(df):
    # For demonstration, fill missing values with a placeholder (-1)
    df = df.fillna(-1)
    return df

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=42)
    return train_df, val_df, test_df

def save_to_csv(train_df, val_df, test_df):
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

if __name__ == "__main__":
    # Prompt user for Excel file name
    excel_file = input("Please enter the filename of the Excel file (including extension): ")

    # Load data
    df = load_data(excel_file)

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Save data to CSV files
    save_to_csv(train_df, val_df, test_df)

    print("Data has been processed and split into training, validation, and test sets.")
