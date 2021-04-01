"""
check ratings
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ramp_intensity.csv')
# convert to long form
dfl = pd.wide_to_long(df, stubnames='OGP-', i=['subject', 'cream'], j='timepoint').reset_index()
dfl.rename(columns={'OGP-':'rating'}, inplace=True)
# print(dfl)
# plot
plt.subplots(figsize=(4,4))
sns.lineplot(data=dfl, x='timepoint', y='rating', hue='cream', markers=True, err_style="bars", ci=68)
plt.savefig('./figs/ratings.png')