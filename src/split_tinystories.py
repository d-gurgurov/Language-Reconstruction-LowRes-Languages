import pandas as pd

# loading the CSV file
df = pd.read_csv('/netscratch/dgurgurov/thesis/mt_lrls/results/tinystories_goodlrl.csv')

# determining the midpoint
midpoint = len(df) // 2

# splitting the DataFrame into two halves
first_half = df.iloc[:midpoint]
second_half = df.iloc[midpoint:]

# saving the two halves into separate CSV files
first_half.to_csv('tinystories_goodlrl_1.csv', index=False)
second_half.to_csv('tinystories_goodlrl_2.csv', index=False)

print("Done!")
