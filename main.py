# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from CombinedAttributesAdder import CombinedAttributesAdder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Press the green button in the gutter to run the script.

def load_housing_data(housing_path=HOUSING_PATH):
    # if housing.csv exists?
    if not os.path.exists(os.path.join(housing_path, "housing.csv")):
        fetch_housing_data()
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print(f'Fetching {housing_url}')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    print(f'Extracting {tgz_path} to {housing_path}')
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def plot_housing_data(housing):
    housing.hist(bins=50, figsize=(20,15))
    plt.show()

if __name__ == '__main__':
    housing = load_housing_data()
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
                 s=housing["population"]/100, label="population", figsize=(10,7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    plt.show()
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # median = housing["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(median, inplace=True)

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    prediction = lin_reg.predict(some_data_prepared)
    print(f'Prediction: {prediction}\nActual: {list(some_labels)}')

    # housing_prediction = lin_reg.predict(housing_prepared)
    # lin_mse = mean_squared_error(housing_labels, housing_prediction)
    # lin_rmse = np.sqrt(lin_mse)
    # print(lin_rmse)
    #
    # tree_reg = DecisionTreeRegressor()
    # tree_reg.fit(housing_prepared, housing_labels)
    # housing_prediction = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_labels, housing_prediction)
    # tree_rmse = np.sqrt(tree_mse)
    #
    # print(tree_rmse)

    # forest_reg = RandomForestRegressor()
    # forest_reg.fit(housing_prepared, housing_labels)
    # housing_prediction = forest_reg.predict(housing_prepared)

    # forest_mse = mean_squared_error(housing_labels,housing_prediction)
    # forest_rmse = np.sqrt(forest_mse)
    #
    # print(forest_rmse)



    # tree_scores = cross_val_score(tree_reg, housing_prepared,
    #                          housing_labels, scoring="neg_mean_squared_error",
    #                          cv=10)
    # lin_scores = cross_val_score(lin_reg,housing_prepared,
    #                              housing_labels, scoring="neg_mean_squared_error",
    #                              cv=10)

    # forest_scores = cross_val_score(forest_reg, housing_prepared,
    #                                 housing_labels, scoring="neg_mean_squared_error",
    #                                 cv=10)

    # tree_rmse_scores = np.sqrt(-tree_scores)
    # lin_rmse_scores = np.sqrt(-lin_scores)
    # forest_rmse_scores = np.sqrt(-forest_scores)

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()

    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)

    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_estimator_)
    # print(f'Decision Tree Regressor \n{tree_rmse_scores.mean()} +/- {tree_rmse_scores.std()}')
    # print(f'Linear Regressor \n{lin_rmse_scores.mean()} +/- {lin_rmse_scores.std()}')
    # print(f'Random Forest Regressor \n{forest_rmse_scores.mean()} +/- {forest_rmse_scores.std()}')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
