import pandas as pd
from main import main

# Load_data
# test = pd.read_pickle('data/input/test_data_4300_rows.pkl')
test = pd.read_parquet('data/task1_test_for_user.parquet')
# Predict
main(test)
