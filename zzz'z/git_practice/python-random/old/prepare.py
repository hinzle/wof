from imports import *
import acquire as acq

def prep_iris():
	'''
	pulls and prepares the iris dataframe for analysis.
	-> cleaned iris dataframe
	'''
	print('prepping iris')
	iris=acq.get_iris_data()
	iris.drop_duplicates()
	iris2=iris.drop(['species_id','measurement_id'],axis=1)
	iris3=iris2.rename({'species_name':'species'},axis=1)
	dummy_df=pd.get_dummies(iris3.species, dummy_na=False, drop_first=True)
	iris4=pd.concat([iris3, dummy_df], axis=1)
	iris5=iris4.drop(['species'],axis=1)
	return iris5

def prep_titanic():
	'''
	pulls and prepares the titanic dataframe for analysis.
	-> cleaned titanic dataframe
	'''
	print('prepping titanic')	
	titanic=acq.get_titanic_data()
	titanic.drop_duplicates()
	titanic2=titanic.drop(['passenger_id','pclass','deck','embark_town','alone'],axis=1)
	dummy_df=pd.get_dummies(titanic2[['sex','embarked','class']], dummy_na=False, drop_first=[True,True,True])
	titanic3=pd.concat([titanic2, dummy_df], axis=1)
	titanic4=titanic3.drop(['sex','embarked','class'],axis=1)
	return titanic4

def prep_telco():
	'''
	pulls and prepares the telco dataframe for analysis.
	-> cleaned telco dataframe
	'''
	print('prepping telco')
	telco=acq.get_telco_data()
	telco.drop_duplicates(inplace=True)
	telco.total_charges.replace(to_replace=[' ',''],value=np.nan,inplace=True)
	telco['total_charges']=telco.total_charges.astype('float')
	telco.drop(['payment_type_id','contract_type_id','internet_service_type_id'],axis=1,inplace=True)
	dummy_list=[
		'gender',
		'partner',
		'dependents',
		'phone_service',
		'multiple_lines',
		'online_security',
		'online_backup',
		'device_protection',
		'tech_support',
		'streaming_tv',
		'streaming_movies',
		'paperless_billing',
		'churn',
		'internet_service_type',
		'contract_type',
		'payment_type'
		]
	dummy_df=pd.get_dummies(telco[dummy_list], dummy_na=False, drop_first=True)
	telco=pd.concat([telco, dummy_df], axis=1)
	telco.drop(dummy_list,axis=1)
	return telco

