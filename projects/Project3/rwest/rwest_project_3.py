import pandas 
import numpy
import sklearn.svm
import matplotlib.pyplot as plt

train_data = pandas.io.parsers.read_csv('lemon_training.csv')
test_data = pandas.io.parsers.read_csv('lemon_test.csv')

print 'train data len before drops = ' + str(len(train_data))

train_data = train_data[['MMRCurrentRetailCleanPrice','VehicleAge', 'IsBadBuy']]
# REPLACE INF VALUES WITH NAN #
train_data = train_data.replace([numpy.inf, -numpy.inf], numpy.nan)
# DROP NAN #
train_data = train_data.dropna()
print 'train data len after drops = ' + str(len(train_data))

t = train_data[['MMRCurrentRetailCleanPrice','VehicleAge']]
clf = sklearn.svm.SVR()
clf.fit(t,train_data['IsBadBuy'])

### In sample results
prob = clf.predict(t)
y_hat = prob >= .5
in_sample_accuracy = float(sum(y_hat == train_data['IsBadBuy']))/len(y_hat)

true_pos = sum((y_hat == True) & (True == train_data['IsBadBuy']))
false_pos = sum((y_hat == True) & (False == train_data['IsBadBuy']))
false_neg = sum((y_hat == False) & (True == train_data['IsBadBuy']))
precision = float(true_pos) / (true_pos + false_pos)
recall = float(true_pos) / (true_pos + false_neg)
f1 = 2*precision*recall/(precision+recall)

# Out of sample results
### we do not have leman list for test_data
# out_of_sample_accuracy = float(sum(y_hat == test_data['IsBadBuy']))/len(y_hat)

print 'In sample accuracy = %.2f%%' % (in_sample_accuracy*100)
print 'In sample precision = %.2f%%' % (precision*100)
print 'In sample recall = %.2f%%' % (recall*100)
print 'In sample F1  = %.2f%%' % (f1*100)
print ''


# print 'Out of sample accuracy = %.2f%%' % (out_of_sample_accuracy*100)


