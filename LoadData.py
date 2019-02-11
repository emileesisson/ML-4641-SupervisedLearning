import pandas as pd
import time
from termcolor import cprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings


def contraceptiveData():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data', sep=',', header=None,
                       names=['wifeAge', 'wifeEducation', 'husbandEducation', 'numChildrenBorn', 'wifeReligion',
                              'wifeWorking', 'husbandJob', 'standardOfLiving', 'mediaExposure', 'contraceptiveMethod'])
    X, Y = data.values[:, 0:9], data.values[:, 9]
    return X, Y


def wineData():
    balance_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                               sep=';', header=0)
    X, Y = balance_data.values[:, 0:11], balance_data.values[:, 11]
    return X, Y


def calc_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_test_pred))
    print("Report: \n", classification_report(y_test, y_test_pred))
    start_time = time.time()
    print("Train Accuracy Score: ", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy Score: ", accuracy_score(y_test, y_test_pred))
    cprint("Testing time: {0} \n".format(time.time() - start_time), 'blue')


def warning():
    warnings.filterwarnings("ignore")


# For exporting data to a clearly formatted CSV for report
def exportData(filename, columns, data):
    with open(filename, 'a') as f:
        f.write(",".join(columns))
        f.write("\n")
        for line in data:
            f.write(",".join(columns))
            f.write("\n")

