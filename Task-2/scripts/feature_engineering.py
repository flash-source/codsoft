"""
Feature engineering for credit card fraud detection.
Transforms raw transaction records into a clean numeric feature matrix.
"""
import numpy as np
import pandas as pd

CATEGORY_COLS = None  # set at fit time, so train/test share identical dummy columns


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between customer location and merchant location."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def engineer_features(df):
    """Build model-ready features from a raw fraudTrain/fraudTest dataframe."""
    df = df.copy()

    # --- datetime features ---
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month

    # cyclical encoding for hour (captures midnight-adjacency)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # --- age at time of transaction ---
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    # --- geo distance between cardholder and merchant ---
    df['distance_km'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )

    # --- gender as binary ---
    df['gender_M'] = (df['gender'] == 'M').astype(int)

    # --- log transform amount (heavily right-skewed) ---
    df['amt_log'] = np.log1p(df['amt'])

    feature_cols = [
        'amt', 'amt_log', 'city_pop', 'age', 'distance_km',
        'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'month', 'gender_M'
    ]

    out = df[feature_cols].copy()

    # one-hot encode merchant category (fixed, low-cardinality: ~14 values)
    category_dummies = pd.get_dummies(df['category'], prefix='cat')
    out = pd.concat([out, category_dummies], axis=1)

    return out


ALL_CATEGORIES = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 'grocery_pos',
    'health_fitness', 'home', 'kids_pets', 'misc_net', 'misc_pos',
    'personal_care', 'shopping_net', 'shopping_pos', 'travel'
]


def align_columns(df, reference_cols):
    """Ensure a feature dataframe has exactly reference_cols (fills missing dummy cols with 0)."""
    for c in reference_cols:
        if c not in df.columns:
            df[c] = 0
    return df[reference_cols]
