"""Trains the laptop price model and saves pipe.pkl + df.pkl.

Run:  python train.py
Re-run this whenever you change your scikit-learn version, so the pickle
files always match the installed version.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent

# Column order used by both training and the Streamlit app
FEATURES = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
            'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']


def fetch_processor(text):
    if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
        return text
    if text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    return 'AMD Processor'


def cat_os(inp):
    if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
        return 'Windows'
    if inp in ('macOS', 'Mac OS X'):
        return 'Mac'
    return 'Others/No OS/Linux'


def parse_memory(mem):
    """Return (HDD, SSD) in GB from strings like '128GB SSD +  1TB HDD'."""
    hdd = ssd = 0
    mem = str(mem).replace('.0', '').replace('GB', '').replace('TB', '000')
    for part in mem.split('+'):
        part = part.strip()
        digits = ''.join(ch for ch in part if ch.isdigit())
        size = int(digits) if digits else 0
        if 'HDD' in part:
            hdd += size
        elif 'SSD' in part:
            ssd += size
        # Hybrid / Flash Storage are ignored (same as the notebook)
    return hdd, ssd


def load_and_clean(csv_path=BASE / 'laptop_data.csv'):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

    res = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(int)
    df['ppi'] = (((res[0] ** 2 + res[1] ** 2) ** 0.5) / df['Inches']).astype(float)

    df['Cpu brand'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3])).apply(fetch_processor)

    mem = df['Memory'].apply(parse_memory)
    df['HDD'] = mem.apply(lambda t: t[0])
    df['SSD'] = mem.apply(lambda t: t[1])

    df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df[df['Gpu brand'] != 'ARM']

    df['os'] = df['OpSys'].apply(cat_os)

    df = df[FEATURES + ['Price']].reset_index(drop=True)
    return df


def build_pipeline():
    step1 = ColumnTransformer(
        transformers=[('col_tnf',
                       OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'),
                       [0, 1, 7, 10, 11])],
        remainder='passthrough')

    estimators = [
        ('rf', RandomForestRegressor(n_estimators=350, random_state=3,
                                     max_samples=0.5, max_features=0.75, max_depth=15)),
        ('gbdt', GradientBoostingRegressor(n_estimators=100, max_features=0.5, random_state=3)),
        ('et', ExtraTreesRegressor(n_estimators=100, random_state=3,
                                   max_samples=0.5, max_features=0.75, max_depth=10,
                                   bootstrap=True)),
    ]
    step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))
    return Pipeline([('step1', step1), ('step2', step2)])


def train_and_save(verbose=True):
    df = load_and_clean()
    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    if verbose:
        pred = pipe.predict(X_test)
        print('R2 score:', round(r2_score(y_test, pred), 4))
        print('MAE     :', round(mean_absolute_error(y_test, pred), 4))

    with open(BASE / 'df.pkl', 'wb') as f:
        pickle.dump(df, f)
    with open(BASE / 'pipe.pkl', 'wb') as f:
        pickle.dump(pipe, f)
    if verbose:
        print('Saved df.pkl and pipe.pkl')
    return pipe, df


if __name__ == '__main__':
    train_and_save()
