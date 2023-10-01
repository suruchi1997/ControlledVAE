import os
import pandas as pd


df_csv_1=pd.DataFrame()
# add seed values you utilized
for file in os.listdir("models"):
    # add the path to your csv files
    df3=pd.read_csv(f"models/{file}/eval.csv")
    df_csv_1 = df_csv_1._append(df3, ignore_index=False)  #

metrics = ["success","lat_cost_mean","lat_cost_std","ctrl_cost","ctrl_cost_std","diff_cost","diff_cost_std","log_min_svd","log_max_svd"]
grouped = df_csv_1.groupby('beta').mean()[metrics]
df_csv_1.drop(metrics, axis=1, inplace=True)
df_csv_1.drop_duplicates(subset=['beta'], keep='last', inplace=True)
df_combined = df_csv_1.merge(right=grouped, right_index=True, left_on='beta', how='right')
df_combined.to_html("Tab.html")
df_combined.to_csv("out.csv")