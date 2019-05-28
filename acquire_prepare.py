import pandas as pd
import numpy as np

def acquire_prep():
    df = pd.read_excel('CapstoneData(5-28-2019).xlsx')

    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(index=str, columns{'majorphase':'major_phase'}, inplace=True)

    return df