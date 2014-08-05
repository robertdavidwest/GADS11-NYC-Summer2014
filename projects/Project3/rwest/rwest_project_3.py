import pandas 
import numpy
import sklearn
import sklearn.svm
from sklearn.svm import SVR
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pdb
import matplotlib.pyplot as plt
import timeit

train_data = pandas.io.parsers.read_csv('lemon_training.csv')
#test_data = pandas.io.parsers.read_csv('lemon_test.csv')

# features contains all continuous variables and IsBadBuy
#features = [	'VehYear',	'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',	'MMRAcquisitionRetailAveragePrice',	'MMRAcquisitonRetailCleanPrice',	'MMRCurrentAuctionAveragePrice',  'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',	'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost', 'BYRNO']
features = [	'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'VehBCost', 'WarrantyCost', 'BYRNO']


y_label = 'IsBadBuy'

train_data = train_data[features + [y_label]]

# REPLACE INF VALUES WITH NAN #
#train_data = train_data.replace([numpy.inf, -numpy.inf], numpy.nan)
# DROP NAN #
train_data = train_data.dropna()

numpy.random.seed(seed=1)
train_data['random'] = [numpy.random.random() for i in range(len(train_data))]
x_train = train_data[train_data.random > .4][features]
y_train = train_data[train_data.random > .4][y_label]
x_test = train_data[train_data.random <= .4][features]
y_test = train_data[train_data.random <= .4][y_label]

print 'len x_train = ' + str(len(x_train))
print 'len x_test = ' + str(len(x_test))

model = []
var1s = []
var2s = []
r2s_insample = []
f1s_insample = []
r2s_outsample = []
f1s_outsample = []

n_estimators = 30

#model_names = ['RandomForestClassifier(n_estimators=30)', 'ExtraTreesClassifier(n_estimators=30)', 'AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=30)','SVR()']
model_names = ['adaboost 30']


#for current_model in (RandomForestClassifier(n_estimators=n_estimators), ExtraTreesClassifier(n_estimators=n_estimators), AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=n_estimators),SVR() ):
for current_model in ( AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=n_estimators) ):
	current_model_name = model_names.pop(0)
	variables_used = []
	for var1 in x_train.columns:
		variables_used.append(var1)
		for var2 in x_train.columns.drop(variables_used):
			x_train_subset = x_train[[var1,var2]]
			x_test_subset = x_test[[var1,var2]]
			
			clf = current_model.fit(x_train_subset, y_train)
			
			### In sample results
			y_hat_train = clf.predict(x_train_subset)
			f1_in = sklearn.metrics.f1_score(y_train,y_hat_train)
			r2_in = clf.score(x_train_subset, y_train)
			
			y_hat_test = clf.predict(x_test_subset)
			
			f1_out = sklearn.metrics.f1_score(y_test,y_hat_test)
			r2_out = clf.score(x_test_subset, y_test)			
			
			### Apend results
			model.append(current_model_name)
			var1s.append(var1)
			var2s.append(var2)
			r2s_insample.append(r2_in)
			f1s_insample.append(f1_in)
			r2s_outsample.append(r2_out)
			f1s_outsample.append(f1_out)
			
			print current_model_name + ' 1st var: ' + var1 + ', 2nd var: ' + var2
	
results = pandas.DataFrame({'model': model,'var1': var1s,'var2': var2s,'r2_insample': r2s_insample, 'f1_insample': f1s_insample, 'r2_outsample': r2s_outsample, 'f1_outsample': f1s_outsample} )
results.to_csv('run_results.csv')
