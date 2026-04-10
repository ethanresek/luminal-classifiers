import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

""" Include the below in any model files:
CSV = r'PATH/NAME/HERE'
DF = pd.read_csv(CSV)
Y_OLD_NAME = 'pam50_+_claudin-low_subtype'
KEEP = list(DF.columns[31:520]) + [Y_OLD_NAME]

Then preprocess:
pre_split = preprocess(DF, KEEP)
train, val, test = split_data(pre_split)
"""

def preprocess(df, keep, y_old_name='pam50_+_claudin-low_subtype', y_new_name='subtype',
               allowed=['LumA', 'LumB'], transform={'LumA': 1, 'LumB': 0}):
    df = df[keep].copy()
    df = df[df[y_old_name].isin(allowed)]
    df[y_old_name] = df[y_old_name].map(transform)
    df= df.rename(columns={y_old_name: y_new_name})
    return df

def split_data(df, test_size=0.15, val_size=0.176):
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)
    return train, val, test
