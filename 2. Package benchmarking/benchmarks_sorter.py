import pandas as pd
from glob import glob

df = pd.DataFrame()

files = glob("pip_benchmarks_*")

for file in files:
    temp_df = pd.read_csv(file, index_col=0)
    df = df.append(temp_df)

# Grouping repeated rows
df = df.groupby(["library"]).mean()

print(df)
df.to_csv("combined_times.csv")