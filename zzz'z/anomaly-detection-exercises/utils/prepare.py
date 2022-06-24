import pandas as pd

def prepare_curriculum(df):
    """ Prepare the curriculum access logs dataframe """
    df = df.copy()
    
    # Only select specific columns
    df = df[['date','path','user_id','cohort_id','ip']]
    
    # Convert date to datetime
    df.date = pd.to_datetime(df.date)
    
    df = df.set_index(df.date)
    
    return df