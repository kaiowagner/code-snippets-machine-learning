from sklearn import datasets

# If True, returns (data, target) instead of a Bunch object
# Type: numpy.ndarray
X_iris, y_iris = datasets.load_iris(return_X_y=True)

# or Bunch object
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# (150, 4) (150,)
print(X_iris.shape, y_iris.shape)

# ['setosa' 'versicolor' 'virginica']
print(iris.target_names)
