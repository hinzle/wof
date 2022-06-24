from imports import *
import acquire as acq

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