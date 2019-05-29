import pandas as pd
import numpy as np

def acquire_oil():
    '''
    Selected Data from Excel File
    '''
    return pd.read_excel('CapstoneData(5-28-2019).xlsx').infer_objects()

def prep_oil(df):

    '''
    Lowercase all column headers, replace spaces with _'s, and add _ to majorphase
    '''

    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    df.rename(index=str, columns={'majorphase':'major_phase'}, inplace=True)

    '''
    Drop erc column, drop na's from oil_eur and gas_eur
    '''

    df.drop(columns=['entity_reserve_category'], inplace=True)
    df.dropna(subset=['oil_eur', 'gas_eur'], inplace=True)

    '''
    Change particular columns to category type
    '''

    df.type = df.type.astype('category')
    df.status = df.status.astype('category')
    df.major_phase = df.major_phase.astype('category')
    df.prod_method = df.prod_method.astype('category')
    df.frac_fluid_type = df.frac_fluid_type.astype('category')

    '''
    Remove particular types that we are not wanting to focus on
    '''

    df = df[df.status != 'Injection']
    df = df[df.major_phase != 'INJ']
    df = df[df.major_phase != 'SWD']
    df = df[df.lateral_len != 0]

    '''
    Remove really old data as there appears to be some date integrity issues
    '''

    df = df[df['first_prod'].dt.year > 1940]
    df = df[df['last_prod'].dt.year > 1940]

    '''
    Create new columns to be able to compare recovery equivalents based on type, then drop originals
    '''

    df['recovery'] = df.oil_eur + df.gas_eur/6
    df['recovery_per_foot'] = df['recovery']/df['lateral_len'] * 1000
    df.drop(columns=['oil_eur', 'gas_eur'], inplace=True)

    '''
    Remove data integrity observation
    '''

    df = df[df.lateral_len < 14000]

    '''
    Create new column for months active, then create new column for a ratio of recovery to months active
    '''

    df[['months_active']] = df[['last_prod']]
    df[['months_active']] = df.last_prod.dt.to_period('M') - df.first_prod.dt.to_period('M')
    df['months_active'].astype(str).astype(int)
    df['months_active'].replace('0', '1', inplace=True)

    df[['recovery_per_month']] = df[['recovery']]
    df[['recovery_per_month']] = df['recovery'] / df['months_active']

    df.to_csv('cleaned_oil_df.xlsx')

    return df