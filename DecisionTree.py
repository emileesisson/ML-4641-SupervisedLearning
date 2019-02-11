import time
from termcolor import cprint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import LoadData
from SupportVectorMachines import plot_learning_curve


# Function to run the decision tree
def analyze(X, Y):
    # Spliting the dataset into train and test
    clf = DecisionTreeClassifier(criterion="gini")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state = 100)

    # using randomized search and k-fold to figure out the best set of parameters and training data
    max_depth = [i for i in range(1, 30)]
    min_samples_leaf = [i for i in range(1, 20)]
    param_grid = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    start_time = time.time()
    grid = RandomizedSearchCV(clf, param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, Y_train)
    cprint("Total training time: {0}".format(time.time() - start_time), 'blue')
    min_samples_graph = [result.mean_validation_score for result in grid.grid_scores_][1:20]
    training_x_and_y = [X_train, Y_train]
    testing_x_and_y = [X_test, Y_test]
    graph_data = [min_samples_leaf, min_samples_graph]
    best_params = grid.best_params_
    cprint("The best parameters were: {0}".format(best_params), 'red')
    return grid.best_estimator_, grid.best_score_, training_x_and_y, testing_x_and_y, graph_data


# Driver code
def main():
    LoadData.warning()

    # Building Phase
    first_X, first_Y = LoadData.contraceptiveData()
    clf_first, first_training_score, first_training_data, first_testing_data, first_graph_data = analyze(first_X,
                                                                                                         first_Y)
    print("Decision Tree Training Score (Contraceptive) After Cross Validation: {0:.2f}%".format(first_training_score * 100))
    LoadData.calc_accuracy(first_training_data[1], clf_first.predict(first_training_data[0]), first_testing_data[1],
                           clf_first.predict(first_testing_data[0]))
    title = "Decision Tree Learning Curves (Contraceptive)"
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    plt, contracept_elapsed_time = plot_learning_curve(clf_first, title, first_training_data[0], first_training_data[1],
                                                       (0.1, 0.5), cv=cv, n_jobs=4)
    plt.show()

    second_X, second_Y = LoadData.wineData()
    clf_second, second_training_score, second_training_data, second_testing_data, second_graph_data = analyze(second_X,
                                                                                                              second_Y)
    print("Decision Tree Training Score (Wine) After Cross Validation: {0:.2f}%".format(second_training_score * 100))
    LoadData.calc_accuracy(second_training_data[1], clf_second.predict(second_training_data[0]),
                                  second_testing_data[1], clf_second.predict(second_testing_data[0]))
    title = "Decision Tree Learning Curves (Wine)"
    plt, contracept_elapsed_time = plot_learning_curve(clf_second, title, second_training_data[0], second_training_data[1],
                                                       (0.1, 0.5), cv=cv, n_jobs=4)
    plt.show()


if __name__ == "__main__":
    main()
