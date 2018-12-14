from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from scipy.stats import sem
from numpy import mean


def mean_score(scores):
    return ("Mean score: {} (+/-{})".format(mean(scores), sem(scores)))


iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# Get dataset with only thhe first two atributes
X, y = X_iris[:, :2], y_iris

# Split the dataset into a training and a testing set
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# For each feature, calculater the average, subtract the mean value from feature value, and divide
# the result by their standard deviation (To avoid that features with large values may weight too
# much)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = SGDClassifier()
clf.fit(X_train, y_train)

print(clf.predict(scaler.transform([[4.7, 3.1]])))
print(clf.decision_function(scaler.transform([[4.7, 3.1]])))

# Overfitting
y_train_pred = clf.predict(X_train)
print('Accuracy: ', metrics.accuracy_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

# BINARY PROBLEMS
# Precision: Proportion of instances predicted as positives that were correctly evaluated
# Relcall: Proportion of positive instances that were correctly evaluated
# F1-score: Mean of precision and recall
print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))


# MULTI-CLASS PROBLEMS
print(metrics.confusion_matrix(y_test, y_pred))


# Create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([('scaler', preprocessing.StandardScaler()), ('linear_model', SGDClassifier())])

# Create a k-fold cross validation iterator of k=5 folds
# X.shape[0] = number of instances
cv = KFold(n_splits=5, shuffle=False, random_state=33)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)

print(mean_score(scores))
