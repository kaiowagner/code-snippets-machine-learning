from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from numpy import mean


def evaluate_cross_validation(clf, X, y, K):
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method
    # of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {} (+/-{})".format(mean(scores), sem(scores)))


news = fetch_20newsgroups(subset='all')

print(type(news.data))
print(type(news.target))
print(type(news.target_names))
print(news.data[0])
print(news.target[0])

# The loaded data is already in a random order

SPLIT_PERC = 0.75
split_size = int(len(news.data) * SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
y_train = news.data[:split_size]
y_test = news.data[split_size:]

clf_1 = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
clf_2 = Pipeline([('vect', HashingVectorizer(non_negative=True)), ('clf', MultinomialNB())])
clf_3 = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])

clfs = [clf_1, clf_2, clf_3]


for clf in clfs:
    evaluate_cross_validation(clf, news.data, news.target, 5)
