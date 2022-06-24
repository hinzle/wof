from imports import *

def get_titanic_data(use_cache=True):
    # guard clause
    if os.path.exists('titanic.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('titanic.csv')
    print('Acquiring data from SQL database')
    df=pd.read_sql('SELECT * FROM passengers',get_db_url('titanic_db'))
    df.to_csv('titanic.csv', index=False)
    return df

def get_iris_data(use_cache=True):
    if os.path.exists('iris.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('iris.csv')
    print('Acquiring data from SQL database')
    df=pd.read_sql(
        '''
        SELECT * FROM measurements
        LEFT JOIN species USING (species_id)
        ''',
        get_db_url('iris_db'))    
    df.to_csv('iris.csv', index=False)
    return df

def get_telco_data(use_cache=True):
    if os.path.exists('telco.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('telco.csv')
    print('Acquiring data from SQL database')
    df=pd.read_sql(
        '''
        SELECT *
        FROM customers
        LEFT JOIN internet_service_types USING (internet_service_type_id)
        LEFT JOIN contract_types USING (contract_type_id)
        LEFT JOIN payment_types USING (payment_type_id)
        ''',
        get_db_url('telco_churn'))     
    df.to_csv('telco.csv', index=False)
    return df

