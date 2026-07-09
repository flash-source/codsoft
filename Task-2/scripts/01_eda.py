"""
Exploratory Data Analysis for the credit card fraud dataset.
Generates summary stats + diagnostic plots saved to outputs/.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'fraudTrain.csv')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')

cols_needed = ['trans_date_trans_time', 'category', 'amt', 'is_fraud', 'city_pop', 'gender']
df = pd.read_csv(TRAIN_PATH, usecols=cols_needed)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour

n = len(df)
n_fraud = df['is_fraud'].sum()
print(f"Total transactions: {n:,}")
print(f"Fraudulent transactions: {n_fraud:,} ({n_fraud/n*100:.3f}%)")
print(f"Legitimate transactions: {n - n_fraud:,} ({(n-n_fraud)/n*100:.3f}%)")
print()
print("Amount stats by class:")
print(df.groupby('is_fraud')['amt'].describe())

# --- Plot 1: class imbalance ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df['is_fraud'].value_counts()
axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=['#4C72B0', '#C44E52'])
axes[0].set_title('Class Distribution (raw counts)')
axes[0].set_ylabel('Number of transactions')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')

axes[1].bar(['Legitimate', 'Fraud'], counts.values, color=['#4C72B0', '#C44E52'])
axes[1].set_yscale('log')
axes[1].set_title('Class Distribution (log scale)')
axes[1].set_ylabel('Number of transactions (log)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/01_class_imbalance.png', dpi=120)
plt.close()

# --- Plot 2: fraud rate by category ---
cat_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(10, 6))
cat_fraud.plot(kind='barh', color='#C44E52')
plt.xlabel('Fraud rate (%)')
plt.title('Fraud Rate by Merchant Category')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/02_fraud_rate_by_category.png', dpi=120)
plt.close()

# --- Plot 3: fraud rate by hour ---
hour_fraud = df.groupby('hour')['is_fraud'].mean() * 100
plt.figure(figsize=(10, 5))
plt.plot(hour_fraud.index, hour_fraud.values, marker='o', color='#C44E52')
plt.xlabel('Hour of day')
plt.ylabel('Fraud rate (%)')
plt.title('Fraud Rate by Hour of Day')
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/03_fraud_rate_by_hour.png', dpi=120)
plt.close()

# --- Plot 4: amount distribution fraud vs legit (log scale) ---
plt.figure(figsize=(10, 5))
import numpy as np
plt.hist(np.log1p(df[df.is_fraud == 0]['amt']), bins=60, alpha=0.6, label='Legitimate', density=True, color='#4C72B0')
plt.hist(np.log1p(df[df.is_fraud == 1]['amt']), bins=60, alpha=0.6, label='Fraud', density=True, color='#C44E52')
plt.xlabel('log(1 + amount)')
plt.ylabel('Density')
plt.title('Transaction Amount Distribution by Class')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/04_amount_distribution.png', dpi=120)
plt.close()

print("\nEDA plots saved to outputs/")
