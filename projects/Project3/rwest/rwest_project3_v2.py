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


# 
train_data['price_to_avg_ratio'] =  numpy.divide(train_data['VehBCost'],train_data['MMRAcquisitionAuctionAveragePrice'])
train_data['annual_mileage'] = numpy.divide(train_data['VehOdo'],train_data['VehicleAge'])
train_data['model_and_year'] = train_data['Model'] + str(train_data['VehYear'])

# dummy variables
train_data['region'] = map(lambda x: numpy.floor(x/100), train_data['VNZIP1'])
region_dummies = pandas.get_dummies(train_data['region'])
modelandyear_dummies = pandas.get_dummies(train_data['model_and_year'])
train_data['isGreen'] = train_data['AUCGUART'] == "GREEN"
train_data['isManheim'] = train_data['Auction'] == "MANHEIM"

train_data = pandas.concat([train_data,region_dummies],axis=0)

features = ['price_to_avg_ratio', 'annual_mileage',	'VehYear',	'VehicleAge',	'IsOnlineSale',	'WarrantyCost','isGreen','isManheim'] + region_dummies.columns


[332]: Index([u'RefId', u'IsBadBuy', u'PurchDate', u'Auction', u'VehYear', u'VehicleAge', u'Make', u'Model', u'Trim', u'SubModel', u'Color', u'Transmission', u'WheelTypeID', u'WheelType', u'VehOdo', u'Nationality', u'Size', u'TopThreeAmericanName', u'MMRAcquisitionAuctionAveragePrice', u'', u'', u'', u'', u'', u'', u'', u'PRIMEUNIT', u'AUCGUART', u'BYRNO', u'VNZIP1', u'VNST', u'VehBCost', u'IsOnlineSale', u'WarrantyCost'], dtype='object')


y_label = 'IsBadBuy'

train_data = train_data[features + [y_label]]

# REPLACE INF VALUES WITH NAN #
train_data = train_data.replace([numpy.inf, -numpy.inf], numpy.nan)
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
r2s_insample = []
f1s_insample = []
r2s_outsample = []
f1s_outsample = []

n_estimators = 100

model_names = ['RandomForestClassifier(n_estimators=30)'] #, 'ExtraTreesClassifier(n_estimators=30)', 'AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=30)','SVR()']

#for current_model in (RandomForestClassifier(n_estimators=n_estimators), ExtraTreesClassifier(n_estimators=n_estimators), AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=n_estimators),SVR() ):
current_model = RandomForestClassifier(n_estimators=n_estimators)

current_model_name = model_names.pop(0)

clf = current_model.fit(x_train, y_train)

### In sample results
y_hat_train = clf.predict(x_train)

f1_in = sklearn.metrics.f1_score(y_train,y_hat_train)
r2_in = clf.score(x_train, y_train)

y_hat_test = clf.predict(x_test)

f1_out = sklearn.metrics.f1_score(y_test,y_hat_test)
r2_out = clf.score(x_test, y_test)			

### Apend results
model.append(current_model_name)
r2s_insample.append(r2_in)
f1s_insample.append(f1_in)
r2s_outsample.append(r2_out)
f1s_outsample.append(f1_out)

print 'current_model_name :' + current_model_name
print 'r2_insample : ' + str(r2_in )
print 'f1_insample : ' + str(f1_in)
print 'r2_outsample: ' + str(r2_out)
print 'f1_outsample: ' + str(f1_out)

#results = pandas.DataFrame({'model': model,'r2_insample': r2s_insample, 'f1_insample': f1s_insample, 'r2_outsample': r2s_outsample, 'f1_outsample': f1s_outsample} )
#results.to_csv('run_results_multi_var.csv')
	