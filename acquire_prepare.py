import pandas as pd
import numpy as np

def acquire_prep():
    df = pd.read_excel('CapstoneData(5-28-2019).xlsx').infer_objects()

    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(index=str, columns={'majorphase':'major_phase'}, inplace=True)

    df.drop(columns=['client_id'], inplace=True)

    df.dropna(subset=['oil_eur', 'gas_eur'], inplace=True)

    df.type = df.type.astype('category')
    df.status = df.status.astype('category')
    df.major_phase = df.major_phase.astype('category')
    df.entity_reserve_category = df.entity_reserve_category.astype('category')
    df.prod_method = df.prod_method.astype('category')
    df.frac_fluid_type = df.frac_fluid_type.astype('category')

    df = df[df.status != 'Injection']
    df = df[df.major_phase != 'INJ']
    df = df[df.major_phase != 'SWD']
    df = df[df.lateral_len != 0]

    return df