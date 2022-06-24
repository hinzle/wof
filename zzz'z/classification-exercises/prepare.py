from imports import *
import acquire as acq

def prep_iris():
	'''
	pulls and prepares the iris dataframe for analysis.
	
	<-: cleaned iris dataframe
	'''
	print('prepping iris')
	iris=acq.get_iris_data()
	iris.drop_duplicates()
	iris2=iris.drop(['species_id','measurement_id'],axis=1)
	iris3=iris2.rename({'species_name':'species'},axis=1)
	dummy_df=pd.get_dummies(iris3.species, dummy_na=False, drop_first=True)
	iris4=pd.concat([iris3, dummy_df], axis=1)
	#iris5=iris4.drop(['species'],axis=1)
	return iris4

def prep_titanic():
	'''
	pulls and prepares the titanic dataframe for analysis.
	
	<-: cleaned titanic dataframe
	'''
	print('prepping titanic')	
	titanic=acq.get_titanic_data()
	titanic.drop_duplicates()
	titanic.age.replace(to_replace=[' ',''],value=np.nan,inplace=True)
	titanic2=titanic.drop(['passenger_id','pclass','deck','embark_town','alone'],axis=1)
	dummy_df=pd.get_dummies(titanic2[['sex','embarked','class']], dummy_na=False, drop_first=[True,True,True])
	titanic3=pd.concat([titanic2, dummy_df], axis=1)
	titanic4=titanic3.drop(['sex','embarked','class'],axis=1)
	return titanic4

def prep_telco():
	'''
	pulls and prepares the telco dataframe for analysis.
	
	<-: cleaned telco dataframe
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
	telco.drop(dummy_list,axis=1,inplace=True)
	return telco

def tralidest(df,target_column=list):
	'''
	takes in a dataframe and a target name, outputs three dataframes: 'train', 'validate', 'test', each stratified on the named target. 
	
	->: str e.g. 'df.target_column'
	<-: 3 x pandas.DataFrame ; 'train', 'validate', 'test'

	training set is 60% of total sample
	validate set is 23% of total sample
	test set is 17% of total sample

	'''
	train, _ = train_test_split(df, train_size=.6, random_state=123, stratify=df[target_column])
	validate, test = train_test_split(_, test_size=(3/7), random_state=123, stratify=_[target_column])
	train.reset_index(drop=True, inplace=True)
	validate.reset_index(drop=True, inplace=True)
	test.reset_index(drop=True, inplace=True)
	return train, validate, test

def impute_mode(train, validate, test, col=list):
	'''
	take in train, validate, and test DataFrames, impute mode for 'col',
	and return train, validate, and test DataFrames
	'''
	imputer = SimpleImputer(missing_values = np.NAN, strategy='most_frequent')
	train[col] = imputer.fit_transform(train[col])
	validate[col] = imputer.transform(validate[col])
	test[col] = imputer.transform(test[col])
	return train, validate, test

def ml_data(train, validate, test, target=list):
	'''
	->: train, validate, test 
	<-: X_train, y_train, X_validate, y_validate, X_test, y_test
	'''
	X_train = train.drop(columns=target)
	y_train = train[target]
	X_validate = validate.drop(columns=target)
	y_validate = validate[target]
	X_test = test.drop(columns=target)
	y_test = test[target]
	return [X_train, y_train, X_validate, y_validate, X_test, y_test]

def decision_tree_predict(X_train, y_train,max_depth=int):
	clf = DecisionTreeClassifier(max_depth=max_depth, random_state=123)
	clf=clf.fit(X_train, y_train)
	y_pred=clf.predict(X_train)
	y_pred_prob=clf.predict_proba(X_train)
	labels=[str(i) for i in clf.classes_]
	conf=pd.DataFrame(confusion_matrix(y_train,y_pred))
	classi_report=(pd.DataFrame(classification_report(y_train, y_pred,output_dict=True)))
	print('_________________________\n')
	print('Accuracy of Decision Tree classifier on training set: {:.2%}\n'
      .format(clf.score(X_train, y_train)))
	plt.figure(figsize=(16, 9), dpi=300)
	tree=plot_tree(clf, feature_names=X_train.columns, class_names=labels, rounded=True)
	plt.show()
	print('_________________________\n')
	print('Confusion Matrix\n')
	print(conf,'\n')
	print('_________________________\n')
	print('Classification Report\n')
	print(classi_report)
	return [clf, tree, y_pred, y_pred_prob, conf, classi_report]