import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from time import time
import LoadData


# Function to run the decision tree
def runSVM(X, Y):
    kernels = ['rbf', 'poly']
    gammas = {'gamma': [0.01, 0.05, 0.1, 0.5, 1]}
    exponents = {'degree': [1, 2, 3, 4, 5]}
    for i in kernels:
        clf = SVC(kernel=i)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.3, random_state=100)
        if i == 'rbf':
            param_grid = gammas
        else:
            param_grid = exponents
        grid = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy")
        grid.fit(X_train, Y_train)
        best_score = grid.best_score_
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        testing_x_and_y = [X_test, Y_test]
        print("The best parameter was: {0}".format(best_params))
    return best_model, best_score, testing_x_and_y


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    start = time()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    end = time() - start
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    axes = plt.gca()
    axes.set_ylim([0,1.1])
    plt.legend(loc="best")
    plt.tight_layout()
    return plt, end

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

    # print("Report : \n",      classification_report(y_test, y_pred))

    return accuracy_score(y_test, y_pred)

# Driver code
def main():
    title = "SVM Learning Curves (Contraceptive)"
    contracept_X, contracept_Y = LoadData.contraceptiveData()
    contraceptX_train, contraceptX_test, contraceptY_train, contraceptY_test = train_test_split(
        contracept_X, contracept_Y, test_size=0.30, random_state=100)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    # change the kernel here
    estimator = SVC(gamma=.001, C=1000.0, kernel='poly')
    plt, contracept_elapsed_time = plot_learning_curve(estimator, title, contraceptX_train, contraceptY_train,
                                                       (0.1, 0.5), cv=cv, n_jobs=4)
    print("It took SVM (Contraceptive) {0}s to train".format(contracept_elapsed_time))
    estimator.fit(contraceptX_train, contraceptY_train)
    print(estimator.score(contraceptX_train, contraceptY_train))
    t0 = time()
    y_pred = estimator.predict(contraceptX_test)
    print("SVM (Contraceptive) Took {0}s to test".format(time() - t0))
    print("SVM Accuracy Score (Contraceptive) was {0}%".format(accuracy_score(contraceptY_test, y_pred) * 100))
    plt.show()

    title = "SVM Learning Curves (Wine)"
    wine_X, wine_Y = LoadData.wineData()
    wineX_train, wineX_test, wineY_train, wineY_test = train_test_split(
        wine_X, wine_Y, test_size=0.30, random_state=100)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    # change the kernel here
    estimator = SVC(gamma=.001, C=1000.0, kernel='rbf')
    plt, wine_elapsed_time = plot_learning_curve(estimator, title, wineX_train, wineY_train,
                                                 (0.1, 1.01), cv=cv, n_jobs=4)
    print("It took SVM (Wine) {0}s to train".format(wine_elapsed_time))
    estimator.fit(wineX_train, wineY_train)
    print(estimator.score(wineX_train, wineY_train))
    t0 = time()
    y_pred = estimator.predict(wineX_test)
    print("It took SVM (Wine) {0}s to test".format((time() - t0)))
    print("SVM Accuracy Score (Wine) was {0}%".format(accuracy_score(wineY_test, y_pred) * 100))
    plt.show()

main()