import pandas as pd
from glob import glob
import numpy as np

data = pd.DataFrame()

files = glob("ppip_benchmarks_*")

for file in files:
    temp_df = pd.read_csv(file, index_col=0)
    data = data.append(temp_df)

# Grouping repeated rows
df_real = data.groupby(["library"]).agg({
    'real_time_full':[np.median, np.average, np.std, max],
    'real_time_solo':[len, np.median, np.average, np.std, max],
    'real_time_constant':[np.median, np.average, np.std, max]
})

df_user = data.groupby(["library"]).agg({
    'user_time_full':[np.median, np.average, np.std, max],
    'user_time_solo':[len, np.median, np.average, np.std, max],
    'user_time_constant':[np.median, np.average, np.std, max]
})

df_sys = data.groupby(["library"]).agg({
    'sys_time_full':[np.median, np.average, np.std, max],
    'sys_time_solo':[len, np.median, np.average, np.std, max],
    'sys_time_constant':[np.median, np.average, np.std, max]
})

df_real.columns = df_real.columns.map('_'.join)
df_user.columns = df_user.columns.map('_'.join)
df_sys.columns = df_sys.columns.map('_'.join)

print(df_real, df_user, df_sys)
df_real.to_csv("combined_times_real.csv")
df_user.to_csv("combined_times_user.csv")
df_sys.to_csv("combined_times_sys.csv")