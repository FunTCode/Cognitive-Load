import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Reading CSV file
df = pd.read_csv(r'./exp_2/release_data_predicted.csv')

# 将Difficulty和Predicted列的值映射为对应的标签
df['Difficulty'] = df['Difficulty'].map({1: 'Low', 2: 'High'})
df['Predicted'] = df['Predicted'].map({0: 'Baseline', 1: 'Low', 2: 'High'})

# Group according to Difficulty and Predicted, and count the number of each group
grouped = df.groupby(['Difficulty', 'Predicted']).size().unstack()

# Calculate the proportion and keep two decimal places
grouped['Low_ratio'] = (grouped['Low'] / grouped.sum(axis=1))
grouped['High_ratio'] = (grouped['High'] / grouped.sum(axis=1))
grouped['Baseline_ratio'] = (grouped['Baseline'] / grouped.sum(axis=1))

# Draw bar charts
ax = grouped[['Low_ratio', 'High_ratio', 'Baseline_ratio']].plot(kind='bar', stacked=True, figsize=(10, 6))

# Add a proportion number to each bar
for i, (index, row) in enumerate(grouped.iterrows()):
    ax.annotate(f"{row['Low_ratio']*100:.2f}%", xy=(i, row['Low_ratio'] / 2), ha='center', va='center', color='black', fontsize=12)
    ax.annotate(f"{row['High_ratio']*100:.2f}%", xy=(i, row['Low_ratio'] + row['High_ratio'] / 2), ha='center', va='center', color='black',fontsize=12)
    ax.annotate(f"{row['Baseline_ratio']*100:.2f}%", xy=(i, row['Low_ratio'] + row['High_ratio'] + row['Baseline_ratio'] / 2), ha='center', va='center', color='black',fontsize=12)

# Sets the text size of the axis name and label
ax.xaxis.label.set_fontsize(14)
ax.yaxis.label.set_fontsize(14)
ax.tick_params(axis='both', labelsize=12)

# Set the Y-axis scale to percentage
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.title('Predicted Difficulty Ratios by Actual Difficulty', fontsize=16)
plt.xlabel('Actual Difficulty Levels')
plt.ylabel('Predicted Difficulty Ratio (%)')
plt.xticks(rotation=0)
plt.legend(['Low', 'High', 'Baseline'], fontsize=12)
plt.show()
