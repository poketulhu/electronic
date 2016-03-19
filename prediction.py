import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split, KFold
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


def get_prediction(data):
  target = data['SPECIAL_PART']
  data = data.drop("SPECIAL_PART", 1)

  classifiers = {
                "Logistic regression": LogisticRegression(),
                "Naive bayes": GaussianNB(),
                "K nearest neughbours": KNeighborsClassifier(),
                # "SVM": svm.SVC(),
                "Decision tree": DecisionTreeClassifier(),
                "Extra tree": ExtraTreeClassifier(),
                "Ada boost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "Extra trees": ExtraTreesClassifier(),
                "Gradient boosting": GradientBoostingClassifier(),
                "Random forest": RandomForestClassifier()
              }

  train_X, test_X, train_y, test_y = train_test_split(data, target, train_size=.60)
  train_X = train_X.values
  train_y = train_y.values
  accuracies = []
  for key, value in classifiers.items():
    kf = KFold(train_X.shape[0], n_folds=4)
    print("Classifier: " + key)
    for train, test in kf:
      X_train, X_test = train_X[train], train_X[test]
      y_train, y_test = train_y[train], train_y[test]
      model = value.fit(X_train, y_train)
      output = model.predict(X_test)
      accuracies.append(accuracy_score(y_test, output))
    print("For " + key + " accuracy = " + str(sum(accuracies) / float(len(accuracies))))
    accuracies = []
  pass

data = pd.read_csv("processed_data.csv")
get_prediction(data)
