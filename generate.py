import pandas
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from skrebate import ReliefF


SCORING = ['f1', 'accuracy', 'precision', 'recall']
NUMBER_OF_FEATURES = [10, 25, 50]

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=3),
    RandomForestRegressor(n_estimators = 250),
    LogisticRegression(),
]

def import_data(path):
    df = pandas.read_csv(path)
    negative_class = df[df['Bankrupt?'] == 0].sample(n = 150)
    positive_class = df[df['Bankrupt?'] == 1].sample(n = 50)

    df = pandas.concat([negative_class, positive_class])
    df = shuffle(df)
    y = df['Bankrupt?']
    del df['Bankrupt?']

    return df, y

def check_score(fitted_data, y_train, X_test, y_test):
    local_score = {}
    for classifier in CLASSIFIERS:
        print(f'uczenie {str(classifier)}')
        classifier.fit(fitted_data, y_train)
        cv_results = cross_validate(classifier, X_test, y_test, cv=2, scoring=SCORING)
        local_score[str(classifier)] = {}
        for k,v in cv_results.items():
            local_score[str(classifier)][k] = list(v)

    return local_score


if __name__ == '__main__':

    X, y = import_data("data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    NUMBER_OF_FEATURES.append(X_train.shape[1]-1)

    scores = {
        "ForwardSequentialFeatureSelector": {},
        "BackwardSequentialFeatureSelector": {},
        "RandomForest": {},
        "ReliefF": {}
    }


    for number_of_features in NUMBER_OF_FEATURES:
        print(number_of_features)

        print("forward")
        scores["ForwardSequentialFeatureSelector"][str(number_of_features)] = {}
        for classifier in CLASSIFIERS:
            print(f'wybieranie {str(classifier)}')
            fitted_data = None
            if number_of_features != X_train.shape[1] - 1:
                fitted_data = SequentialFeatureSelector(classifier, n_features_to_select=number_of_features, cv=2, direction="forward").fit_transform(X_train, y_train)
            else:
                fitted_data = X_train
            scores["ForwardSequentialFeatureSelector"][str(number_of_features)][str(classifier)] = check_score(fitted_data, y_train, X_test, y_test)
        
        with open('ForwardSequentialFeatureSelector.json', 'w') as f:
            json.dump(scores["ForwardSequentialFeatureSelector"], f)
        
# ==================================================================================================================================================

        print("backward")
        scores["BackwardSequentialFeatureSelector"][str(number_of_features)] = {}
        for classifier in CLASSIFIERS:
            print(f'wybieranie {str(classifier)}')
            fitted_data = None
            if number_of_features != X_train.shape[1] - 1:
                fitted_data = SequentialFeatureSelector(classifier, n_features_to_select=number_of_features, cv=2, direction="backward").fit_transform(X_train, y_train)
            else:
                fitted_data = X_train
            scores["BackwardSequentialFeatureSelector"][str(number_of_features)][str(classifier)] = check_score(fitted_data, y_train, X_test, y_test)
        
        with open('BackwardSequentialFeatureSelector.json', 'w') as f:
            json.dump(scores["BackwardSequentialFeatureSelector"], f)
        
# ==================================================================================================================================================

        print("random forest")
        scores["RandomForest"][str(number_of_features)] = {}

        fitted_data = None
        if number_of_features != X_train.shape[1] - 1:
            fitted_data = SelectFromModel(RandomForestRegressor(n_estimators = 250), max_features=number_of_features).fit_transform(X_train, y_train)
        else:
            fitted_data = X_train
        scores["RandomForest"][str(number_of_features)] = check_score(fitted_data, y_train, X_test, y_test)

        with open('RandomForest.json', 'w') as f:
            json.dump(scores["RandomForest"], f)

# ==================================================================================================================================================

        print("ReliefF")
        if number_of_features != X_train.shape[1] - 1:
            fitted_data = ReliefF(n_features_to_select=number_of_features).fit_transform(X_train.values, y_train.values)
        else:
            fitted_data = X_train
        scores["ReliefF"][str(number_of_features)] = check_score(fitted_data, y_train, X_test, y_test)
        
        with open('ReliefF.json', 'w') as f:
            json.dump(scores["ReliefF"], f)
