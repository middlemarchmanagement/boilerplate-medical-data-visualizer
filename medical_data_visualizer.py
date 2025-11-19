import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
# Calculate BMI: weight (kg) / (height (m))^2
# height is in cm, so divide by 100 to get meters.
bmi = df['weight'] / (df['height'] / 100)**2
# If BMI > 25, set to 1 (overweight), else set to 0 (not overweight)
df['overweight'] = (bmi > 25).astype(int)

# 3
# Normalize data by making 0 always good and 1 always bad.
# If the value of cholesterol or gluc is 1, set the value to 0.
# If the value is more than 1 (i.e., 2 or 3), set the value to 1.
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# 4
def draw_cat_plot():
    # 5
    # Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    # Group and reformat the data to split it by cardio and show counts
    # Rename 'value' column to 'total' for proper counting in catplot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7
    # Convert the data into long format and create a chart
    # Use 'kind="bar"' to make it a bar plot
    # Split by 'cardio' using 'col' and use 'variable' for x-axis categories
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    
    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # Clean the data in the df_heat variable by filtering out incorrect data segments
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    # Plot the correlation matrix using seaborn's heatmap
    sns.heatmap(corr, linewidths=1, annot=True, square=True, mask=mask, fmt='.1f', center=0.08, cbar_kws={'shrink': 0.5})

    # 16
    fig.savefig('heatmap.png')
    return fig
