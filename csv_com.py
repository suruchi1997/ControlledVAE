import pandas as pd
import scipy.stats as stats

df_csv_1=pd.DataFrame()
# add seed values you utilized
csv_files=[1,2,3]

for file in csv_files:
    # add the path to your csv files
    df3=pd.read_csv("conv_mul/"+str(file)+".csv")

    df_csv_1 = df_csv_1.append(df3, ignore_index=True)
df_combined= df_csv_1.groupby(['beta'])["beta","success","lat_cost_mean","lat_cost_std","ctrl_cost","ctrl_cost_std","diff_cost","diff_cost_std","log_min_svd","log_max_svd"].mean()
df_combined.to_html("Tab.html")
df_combined.to_csv("out.csv")
