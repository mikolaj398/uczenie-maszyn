
SCORING = ['f1', 'accuracy', 'precision', 'recall']
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from skrebate import ReliefF
import numpy

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=3),
    RandomForestRegressor(n_estimators = 250),
    LogisticRegression(),
]


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

def get_selection_forward(number_of_features, X_train, y_train, X_test, y_test):
    rows = []
    for classifier in CLASSIFIERS:
        print(f'wybieranie {str(classifier)}')
        fitted_data = None
        if number_of_features != X_train.shape[1] - 1:
            fitted_data = SequentialFeatureSelector(classifier, n_features_to_select=number_of_features, cv=2, direction="forward").fit_transform(X_train, y_train)
        else:
            fitted_data = X_train
            
        classifiers_fit_scores = check_score(fitted_data, y_train, X_test, y_test)
        for k,v in classifiers_fit_scores.items():
            row = [
                    'Sekwencyjne przeszukiwanie w przód',
                    classifier,
                    k,
                    number_of_features,
            ]
            scorring = []
            for score in SCORING:
                if v[f'test_{score}'][0] != numpy.NaN:
                    scorring.append((v[f'test_{score}'][0]+v[f'test_{score}'][0])/2)
                else:
                    scorring.append('null')
            row.append((v['score_time'][0]+v['score_time'][0])/2)
            row.extend(scorring)
            rows.append(row)
    return rows

def get_selection_backward(number_of_features, X_train, y_train, X_test, y_test):
    rows = []
    for classifier in CLASSIFIERS:
        print(f'wybieranie {str(classifier)}')
        fitted_data = None
        if number_of_features != X_train.shape[1] - 1:
            fitted_data = SequentialFeatureSelector(classifier, n_features_to_select=number_of_features, cv=2, direction="backward").fit_transform(X_train, y_train)
        else:
            fitted_data = X_train
            
        classifiers_fit_scores = check_score(fitted_data, y_train, X_test, y_test)
        for k,v in classifiers_fit_scores.items():
            row = [
                    'Sekwencyjne przeszukiwanie w tył',
                    classifier,
                    k,
                    number_of_features,
            ]
            scorring = []
            for score in SCORING:
                if v[f'test_{score}'][0] != numpy.NaN:
                    scorring.append((v[f'test_{score}'][0]+v[f'test_{score}'][0])/2)
                else:
                    scorring.append('null')
            row.append((v['score_time'][0]+v['score_time'][0])/2)
            row.extend(scorring)
            rows.append(row)
    return rows

def get_random_forest(number_of_features, X_train, y_train, X_test, y_test):
    rows = []
    fitted_data = None
    if number_of_features != X_train.shape[1] - 1:
        fitted_data = SelectFromModel(RandomForestRegressor(n_estimators = 250), max_features=number_of_features).fit_transform(X_train, y_train)
    else:
        fitted_data = X_train
    classifiers_fit_scores = check_score(fitted_data, y_train, X_test, y_test)
    
    for k,v in classifiers_fit_scores.items():
        row = [
                'Las losowy',
                'Las losowy',
                k,
                number_of_features,
        ]
        scorring = []
        for score in SCORING:
            if v[f'test_{score}'][0] != numpy.NaN:
                scorring.append((v[f'test_{score}'][0]+v[f'test_{score}'][0])/2)
            else:
                scorring.append('null')
        row.append((v['score_time'][0]+v['score_time'][0])/2)
        row.extend(scorring)
        rows.append(row)
    return rows

def get_relieF(number_of_features, X_train, y_train, X_test, y_test):
    rows = []
    fitted_data = None
    if number_of_features != X_train.shape[1] - 1:
        fitted_data = ReliefF(n_features_to_select=number_of_features).fit_transform(X_train.values, y_train.values)
    else:
        fitted_data = X_train
    classifiers_fit_scores = check_score(fitted_data, y_train, X_test, y_test)
    
    for k,v in classifiers_fit_scores.items():
        row = [
                'ReliefF',
                'ReliefF',
                k,
                number_of_features,
        ]
        scorring = []
        for score in SCORING:
            if v[f'test_{score}'][0] != numpy.NaN:
                scorring.append((v[f'test_{score}'][0]+v[f'test_{score}'][0])/2)
            else:
                scorring.append('null')
        row.append((v['score_time'][0]+v['score_time'][0])/2)
        row.extend(scorring)
        rows.append(row)
    return rows