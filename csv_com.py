import os
import pandas as pd


df_csv_1=pd.DataFrame()
# add seed values you utilized
for file in os.listdir("models"):
    # add the path to your csv files
    df3=pd.read_csv(f"models/{file}/eval.csv")
    df_csv_1 = df_csv_1._append(df3, ignore_index=False)  #

metrics = ["beta","success","lat_cost_mean","lat_cost_std","ctrl_cost","ctrl_cost_std","diff_cost","diff_cost_std","log_min_svd","log_max_svd"]
grouped = df_csv_1.groupby(['beta'])[metrics].mean()
grouped.to_html("Tab.html")
grouped.to_csv("out.csv")