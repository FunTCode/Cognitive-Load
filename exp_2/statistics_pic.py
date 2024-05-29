import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Reading CSV file
df = pd.read_csv(r'./exp_2/release_data_predicted.csv')

# Map the values of the Difficulty and Predicted columns to the corresponding labels
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

#########################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt

# Read processed data
df = pd.read_csv(r'./exp_2/release_data_predicted.csv')

# Group the questions according to difficulty and Seq, and calculate the average Predicted values of all participants under each Seq
grouped = df.groupby(['Difficulty', 'Seq'])['Predicted'].mean().reset_index()

# Filter out the data with difficulty 1
difficulty_1_data = grouped[grouped['Difficulty'] == 1]
# Filter out the data with problem difficulty of 2
difficulty_2_data = grouped[grouped['Difficulty'] == 2]

# Smooth data using moving averages
window_size = 5  # Window size
difficulty_1_data_smooth = difficulty_1_data['Predicted'].rolling(window=window_size, min_periods=1).mean()
difficulty_2_data_smooth = difficulty_2_data['Predicted'].rolling(window=window_size, min_periods=1).mean()

# Draw a line chart
plt.figure(figsize=(10, 6))

# Draw a line chart with Difficulty of 1 (smoothing)
plt.plot(difficulty_1_data['Seq'], difficulty_1_data_smooth, label='Low Difficulty')

# Draw a line chart with Difficulty 2 (smoothing)
plt.plot(difficulty_2_data['Seq'], difficulty_2_data_smooth, label='High Difficulty')

# Add titles and labels
plt.title('Average Cognitive Load (Predicted) over Time by Difficulty Level',fontsize=16)
plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Cognitive Load (Predicted)',fontsize=14)
plt.yticks([0, 1, 2], ['Baseline', 'Low', 'High'],fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()


