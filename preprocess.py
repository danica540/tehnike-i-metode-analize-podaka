#!/usr/bin/env python3
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
import csv
import pandas as pd
import numpy as np
from pprint import pprint
from csv import reader
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def _load_csv(filename):
    df = pd.read_csv(filename, na_values=['#NAN'])
    return df


def _divide_dataframe_into_features_and_outcome(df):
    X = df.drop('income', 1)
    y = df.income
    return X, y


def _replace_outcome_values(y):
    for i in range(len(y)):
        val = y[i].strip()
        if val == ">50K":
            y[i] = 1
        else:
            y[i] = 0
    return y


def _print_unique_values_of_features(X):
    for col_name in X.columns:
        if X[col_name].dtypes == 'object':
            unique_cat = len(X[col_name].unique())
            print("Atribut '{col_name}' ima {unique_cat} jedinstvenih vrednosti".format(
                col_name=col_name, unique_cat=unique_cat))


def _replace_low_frequency_countries_with_other(X):
    X['native_country'] = ['United-States ' if x ==
                           'United-States' else 'Other' for x in X['native_country']]
    return X



def _dummy_dataframe(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


def _impute_missing_values(X):
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    si.fit(X)
    X = pd.DataFrame(data=si.transform(X), columns=X.columns)
    return X


def _find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1
    limit_min = q1 - 1.5*iqr
    limit_max = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < limit_min) | (x > limit_max)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values


# Prikaz distribucija atributa po income-u
def _plot_histogram(x, y):
    plt.hist(list(x[y == 0]), alpha=0.5, label='Outcome=>50K')
    plt.hist(list(x[y == 1]), alpha=0.5, label='Outcome=<=50K')
    plt.title("Histogram of '{var_name}' by Outcome Category".format(
        var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()


def _add_interactions(df):
    # Izvlacenje imena kolona
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # Nalazenje interakcija
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df, columns=colnames)

    # Brisanje interakcija ciji termini imaju sve nule
    all_zero_values = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[all_zero_values], axis=1)

    return df


def _pca(X):
    pca = PCA(n_components=10)
    X_pca = pd.DataFrame(pca.fit_transform(X))
    return X_pca


# Kreiranje modela i sracunavanje performansi

def _find_model_performance_naive_bayes(X_train, y_train, X_test, y_test):
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_hat = [x[1] for x in classifier.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc


def _find_model_performance_svm(X_train, y_train, X_test, y_test):
    classifier = SVC()
    classifier.fit(X_train, y_train)
    y_hat = [x[1] for x in classifier.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc


def _find_model_performance_tree(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_hat = [x[1] for x in classifier.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc


def _find_model_performance_k_neighbors(X_train, y_train, X_test, y_test):
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_hat = [x[1] for x in classifier.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc


def _print_model_comparison(model_name, performanse_processed, performanse_unprocessed):
    pprint(
        f"-------------------------- {model_name} -----------------------------------")
    pprint(f"AUC modela sa preprocesiranjem: {performanse_processed}")
    pprint(f"AUC modela bez preprocesiranja: {performanse_unprocessed}")
    performanse_improve = ((performanse_processed-performanse_unprocessed)/performanse_unprocessed)*100
    pprint(f"Poboljsanje modela: {performanse_improve}%")
    pprint("-------------------------------------------------------------------")


if __name__ == "__main__":

    filename = 'adults.csv'
    df = _load_csv(filename)

    pprint("Income column value count:")
    pprint(df['income'].value_counts())

    # Podela DataFrame-a na dataframe sa atributima (karakteristikama) i na kolonu sa vrednostima atributa income, za koji vrsimo predikciju
    X, y = _divide_dataframe_into_features_and_outcome(df)
    # Zamena vrednosti y-a sa 0 i 1
    y = _replace_outcome_values(y)

    # Kategoricki atributi - unique values
    _print_unique_values_of_features(X)

    # Prikaz distribucije vrednosi kolona
    pprint(X['native_country'].value_counts(
    ).sort_values(ascending=False).head(10))
    pprint(X['workclass'].value_counts().sort_values(ascending=False).head(10))
    pprint(X['education'].value_counts().sort_values(ascending=False).head(10))
    pprint(X['marital_status'].value_counts(
    ).sort_values(ascending=False).head(10))
    pprint(X['occupation'].value_counts().sort_values(ascending=False).head(10))
    pprint(X['relationship'].value_counts(
    ).sort_values(ascending=False).head(10))
    pprint(X['race'].value_counts().sort_values(ascending=False).head(10))
    pprint(X['sex'].value_counts().sort_values(ascending=False).head(10))

    X = _replace_low_frequency_countries_with_other(X)

    # Prikaz kako radi dummies metoda u pandasu
    pprint(pd.get_dummies(X['race']).head(5))
    pprint(" ")

    # Izdvajanje atributa koji imaju nenumericke vrednosti
    todummy_list = ['workclass', 'education', 'marital_status',
                    'occupation', 'relationship', 'race', 'sex', 'native_country']

    X = _dummy_dataframe(X, todummy_list)

    # Zamena nedostajucih vrednosti srednjom vrednoscu
    X = _impute_missing_values(X)


    # Detekcija outlier-a
    tukey_indices, tukey_values = _find_outliers_tukey(X['age'])
    pprint("Outliers tukey")
    pprint(np.sort(tukey_values))


    # Prikaz outlier-a na chart-u
    tmp=X['age']
    plt.scatter(x=tmp, y=[0] * len(tmp),c="blue")
    plt.scatter(x=tukey_values, y=[0] * len(tukey_values),c="red")
    plt.ylim(0,0)
    plt.show()

    #_plot_histogram(X['age'], y)
    #_plot_histogram(X['education_num'], y)
    #_plot_histogram(X['hours_per_week'], y)

    X = _add_interactions(X)
    pprint("'''''''''''''''''''Interakcije'''''''''''''''''''")
    pprint(X.head(5))

    # PCA za redukciju dimenzionalnosti, ubaceno samo kao razlog zbog cega sam koristila selectKBest
    # PCA mi je voma nerazumljiv za citanje
    X_pca = _pca(X)

    pprint("PCA")
    pprint(X_pca)

    # Podela na train i test skupove

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, random_state=1)

    # Promena broja kolona nakon dummy metode i dodavanja interakcija
    pprint("Promena broja kolona pre i nakon dodavanja interakcija i primenjivanja dummy metode:")
    pprint(df.shape)
    pprint(X.shape)

    # Kod velikih datasetova moze doci do overfitting-a i sporog sracunavanja, pa je izvrsena selekcija kolona
    # Selekcija K najvaznijih kolona

    select = SelectKBest(f_classif,k=16)
    selected_features = select.fit(X_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    X_train_selected = X_train[colnames_selected]
    X_test_selected = X_test[colnames_selected]

    pprint(f"Selected columns: {len(colnames_selected)}")
    pprint(colnames_selected)

    X_train_selected.to_csv("./X_train_selected.csv", index=False, header=True)


    # Performanse Tree

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    performanse_processed = _find_model_performance_tree(
        X_train_selected, y_train, X_test_selected, y_test)

    df = df.drop('income', 1)
    df["income"] = y
    # IZBACIVANJE NEDOSTAJUĆIH VREDNOSTI iz neprocesiranog dataframe-a
    df_unprocessed = df
    df_unprocessed = df_unprocessed.dropna(axis=0, how='any')
    pprint(df.shape)
    pprint(df_unprocessed.shape)

    # Izbacivanje nenumerickih atributa
    for col_name in df_unprocessed.columns:
        if df_unprocessed[col_name].dtypes not in ['int32', 'int64', 'float32', 'float64']:
            df_unprocessed = df_unprocessed.drop(col_name, 1)

    df_unprocessed['income'] = y
    # Podela na skup karakteristika i ciljnih atributa
    X_unprocessed = df_unprocessed.drop('income', 1)
    y_unprocessed = df_unprocessed.income

    # Kako neprocesirani podaci izgledaju
    pprint(X_unprocessed.head(5))

    # Podela na train i test akupove
    X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
        X_unprocessed, y_unprocessed, train_size=0.9, random_state=1)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    performanse_unprocessed = _find_model_performance_tree(
        X_train_unprocessed, y_train, X_test_unprocessed, y_test)

    _print_model_comparison("TREE", performanse_processed, performanse_unprocessed)

    # Performanse naive Bayes
    performanse_processed = _find_model_performance_naive_bayes(
        X_train_selected, y_train, X_test_selected, y_test)

    performanse_unprocessed = _find_model_performance_naive_bayes(
        X_train_unprocessed, y_train, X_test_unprocessed, y_test)

    _print_model_comparison("BAYES", performanse_processed, performanse_unprocessed)

    # Performanse K najblizih suseda
    performanse_processed = _find_model_performance_k_neighbors(
        X_train_selected, y_train, X_test_selected, y_test)

    performanse_unprocessed = _find_model_performance_k_neighbors(
        X_train_unprocessed, y_train, X_test_unprocessed, y_test)

    _print_model_comparison("K NEIGHBORS", performanse_processed, performanse_unprocessed)
