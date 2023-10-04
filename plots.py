import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd


# Assuming df1 is your DataFrame
# out.csv is the combined csv file you csn use the path of your csv files
df1=pd.read_csv("out.csv")
df1["total_cost"]=df1["ctrl_cost"]+df1["diff_cost"]
df1=df1.sort_values(by=['beta'])
cost_0=float(df1[df1["beta"]==0]["total_cost"])
df1["per_change"] = ((df1["total_cost"]-cost_0)/cost_0)*100
df1["expsq_cost"] = np.exp(df1["total_cost"]**2)
df1['expsq_cost_smoothed'] = df1["expsq_cost"].rolling(window=3, min_periods=1).mean()

x = np.arange(0, len(df1["beta"]), 1)
y = df1["log_min_svd"]
X_Y_Spline = make_interp_spline(x, y)
X_ = np.linspace(x.min(), x.max(), 500)
Y_ = X_Y_Spline(X_)
y = df1["per_change"]
interpolated_curve = interp1d(x, y, kind='cubic')
x_smooth = np.linspace(x.min(), x.max(), 1000)
y_smooth = interpolated_curve(x_smooth)

fig, ax1 = plt.subplots()
# plot in red
color = 'tab:red'
ax1.set_xlabel(r"$Controllability\ Tradeoff,\ \beta$",fontsize=30)
ax1.set_xticks(x,df1["beta"],fontsize=21)
ax1.plot(x_smooth,y_smooth,color=color,linewidth=4,label=r"$Change\ in\ Control\ Cost\ in\ \%$")
ax1.tick_params(axis="y",labelcolor=color,labelsize=25)
ax1.set_ylabel(r"$Change\ in\ Control\ Cost\ in\ \%$",fontsize=30,color=color)
#plot in blue
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.plot(X_,Y_,color=color,linewidth=4, label=r"$Degree\ of\ Controllability$")
ax2.tick_params(axis="y",labelcolor=color,labelsize=25)
ax2.set_ylabel(r"$Degree\ of\ Controllability$",fontsize=30,color=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# create a legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines += lines2
labels += labels2
fig.legend(lines, labels, loc="upper left",bbox_to_anchor=(0.2, 0.92), fontsize=22)
plt.savefig("fig.png",dpi=300)
plt.show()
