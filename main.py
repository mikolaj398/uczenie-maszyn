from numpy import NaN
import pandas
import csv
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from selection_methods import get_selection_forward, get_selection_backward, get_random_forest, get_relieF

NUMBER_OF_FEATURES = [10, 25, 50, 70]


def import_data(path):
    df = pandas.read_csv(path)
    negative_class = df[df['Bankrupt?'] == 0].sample(n = 150)
    positive_class = df[df['Bankrupt?'] == 1].sample(n = 50)

    df = pandas.concat([negative_class, positive_class])
    df = shuffle(df)
    y = df['Bankrupt?']
    del df['Bankrupt?']

    return df, y

if __name__ == '__main__':

    X, y = import_data("data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    NUMBER_OF_FEATURES.append(X_train.shape[1]-1)

    csv_header = ['metoda', 'estymator', 'klasyfikator', 'ilość cech', 'f1', 'dokładność', 'precyzja', 'recall', 'czas uczenia']
    with open('results.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for number_of_features in NUMBER_OF_FEATURES:
            print(number_of_features)

            print("forward")
            for row in get_selection_forward(number_of_features, X_train, y_train, X_test, y_test):
                writer.writerow(row)
            
            print("backward")
            for row in get_selection_backward(number_of_features, X_train, y_train, X_test, y_test):
                writer.writerow(row)

            print("random forest")
            for row in get_random_forest(number_of_features, X_train, y_train, X_test, y_test):
                writer.writerow(row)
            
            print("ReliefF")
            for row in get_relieF(number_of_features, X_train, y_train, X_test, y_test):
                writer.writerow(row)
