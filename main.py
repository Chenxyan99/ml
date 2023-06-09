import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, reciprocal
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from transformer import CombinedAttributesAdder, DataFrameSelector, indices_of_top_k, TopFeatureSelector

HOUSING_PATH = "datasets/housing"


#### 加载数据
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
#### 创建训练集/测试集
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#### 去除income_cat列
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

#### 准备数据
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#### 数据清洗
# housing.dropna(subset=["total_bedrooms"])  # 选项1(去行)
# housing.drop("total_bedrooms", axis=1)  # 选项2(去列)
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)  # 选项3(赋值)

# imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
# X = imputer.fit_transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#### 处理文本和类别属性
# housing_cat = housing["ocean_proximity"]
# encoder_LB = LabelBinarizer()
# housing_cat_1hot = encoder_LB.fit_transform(housing_cat)

#### 自定义转换器(测试)
# attr_adder = CombinedAttributesAdder()
# housing_extra_attribs = attr_adder.transform(housing.values)

#### 转换流水线
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),
    ("cat_encoder", OneHotEncoder(sparse_output=False))
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)


#### 选择训练模型
#### 交叉验证
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


#### 线性回归(误差大)
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)

#### 决策树回归(过拟合->误差大)
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)

####  随机森林
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                 scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# lin_rmse_scores = np.sqrt(-lin_scores)
# forest_rmse_scores = np.sqrt(-forest_scores)

# display_scores(forest_rmse_scores)


#### 模型微调(网格搜索)

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
#### 练习1
# param_grid = [
#     {"kernel": ["linear"], "C": [300, 1000, 3000, 10000, 30000]}
# ]
# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
#                            verbose=2, n_jobs=4)
grid_search.fit(housing_prepared, housing_labels)
# print(np.sqrt(-grid_search.best_score_))
# print(grid_search.best_params_)

#### 练习2
# params_distribs = {
#     "kernel": ["rbf"],
#     "C": reciprocal(157055, 157056),
#     "gamma": expon(scale=1)
# }
# svm_reg = SVR()
# rnd_search = RandomizedSearchCV(svm_reg, param_distributions=params_distribs,
#                                 n_iter=1, cv=5, scoring="neg_mean_squared_error",
#                                 verbose=2, n_jobs=4, random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
# print(np.sqrt(-rnd_search.best_score_))
# print(rnd_search.best_params_)

#### 分析最佳模型
feature_importances = grid_search.best_estimator_.feature_importances_
#
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers["cat_pipeline"]["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# print(sorted(zip(feature_importances, attributes), reverse=True))

#### 用测试集评估系统
# final_model = grid_search.best_estimator_
# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()
#
# X_test_prepared = full_pipeline.transform(X_test)
# final_predictions = final_model.predict(X_test_prepared)
#
# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
# print(final_rmse)

top_k_feature_indices = indices_of_top_k(feature_importances, 5)
print(top_k_feature_indices)
print(np.array(attributes)[top_k_feature_indices])
print(sorted(zip(feature_importances, attributes), reverse=True)[:5])
preparation_and_feature_selection_pipeline = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances, 5))
])
housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
print(housing_prepared_top_k_features[0:3])
print(housing_prepared[0:3, top_k_feature_indices])
