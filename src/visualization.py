"""visualization.py

Plotting utilities for PHRS results.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style='whitegrid')

def save_fig(fig, out_path, dpi=150):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)

def plot_top_risk(df, top_n=20, out_path='results/figures/top_risk_districts.png'):
    df_sorted = df.sort_values('Overall_Avg_Risk', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(df_sorted))))
    sns.barplot(x='Overall_Avg_Risk', y='District', data=df_sorted, ax=ax, palette='Reds_r')
    ax.set_title(f'Top {top_n} High-Risk Districts (PHRS)')
    ax.set_xlabel('Average PHRS')
    ax.set_ylabel('District')
    save_fig(fig, out_path)

def plot_low_risk(df, top_n=20, out_path='results/figures/low_risk_districts.png'):
    df_sorted = df.sort_values('Overall_Avg_Risk', ascending=True).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(df_sorted))))
    sns.barplot(x='Overall_Avg_Risk', y='District', data=df_sorted, ax=ax, palette='Greens_r')
    ax.set_title(f'Top {top_n} Lowest-Risk Districts (PHRS)')
    ax.set_xlabel('Average PHRS')
    ax.set_ylabel('District')
    save_fig(fig, out_path)

def plot_statewise_box(df, out_path='results/figures/statewise_boxplot.png'):
    if 'State' not in df.columns:
        raise ValueError('Dataframe must contain a "State" column for statewise boxplot.')
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.boxplot(y='State', x='Overall_Avg_Risk', data=df, ax=ax, palette='coolwarm')
    ax.set_title('Statewise Distribution of Overall Average PHRS')
    ax.set_xlabel('PHRS')
    ax.set_ylabel('State')
    save_fig(fig, out_path)
