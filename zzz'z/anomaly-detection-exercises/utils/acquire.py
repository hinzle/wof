import pandas as pd
import os
from env import get_db_url

def acquire():
    
    filename = 'curriculum_logs.csv'
    
    if os.path.exists('/Users/hinzlehome/codeup-data-science/anomaly-detection-exercises/csv'+filename):
        
        return pd.read_csv('/Users/hinzlehome/codeup-data-science/anomaly-detection-exercises/csv'+filename)
    
    else:
        query = """SELECT * FROM logs LEFT JOIN cohorts ON logs.user_id = cohorts.id"""
        df = pd.read_sql(query, get_db_url('curriculum_logs'))
        
        df.to_csv('/Users/hinzlehome/codeup-data-science/anomaly-detection-exercises/csv'+filename, index=False)
        
        return df



