import pandas as pd
from main import main


test = pd.read_parquet('data/task1_test_for_user.parquet')
main(test)
