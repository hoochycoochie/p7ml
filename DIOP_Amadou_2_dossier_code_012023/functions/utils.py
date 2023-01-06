import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import timeit


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 5) * bar_width + bar_width / 5

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(
                x + x_offset,
                y,
                width=bar_width * single_width,
                color=colors[i % len(colors)],
            )

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), prop={"size": 20})


def draw_completion(df, head_size=100):

    data_nan = df.isna().sum().sort_values(ascending=False).head(head_size)
    plt.title("Proportion de NaN par variable (%)")
    sns.barplot(x=data_nan.values / df.shape[0] * 100, y=data_nan.index)


## fonction renvoyant les colonnes et le pourcentage de valeurs nulles pour chacune d'elle
def columns_na_percentage(df):
    na_df = (
        (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).reset_index()
    )
    na_df.columns = ["Column", "na_rate_percent"]
    return na_df


## fonction qui trace la matrice de corrélation
def show_correlation_matrix(df, relevant_numeric_columns):
    corr_matrix = df[relevant_numeric_columns].corr()
    fig = plt.figure(1, figsize=(14, 14))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_matrix, mask=mask, square=True, linewidths=0.1, annot=True)
    plt.xlim(0, corr_matrix.shape[1])
    plt.ylim(0, corr_matrix.shape[0])
    plt.show()


def corr_matrix(df, relevant_numeric_columns, threshold=0):
    corr_matrix = df[relevant_numeric_columns].corr().abs()
    sol = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        .stack()
        .sort_values(ascending=False)
    )
    sol = sol[sol >= threshold]
    print(sol)
    return sol


def model_func(
    df,
    model,
    target_col,
    feature_cols,
    test_size,
    random_state,
    X_train,
    X_test,
    y_train,
    y_test,
):
    coefs = dict()
    # X=df[feature_cols].values
    # y=df[target_col].values
    start_time = timeit.default_timer()
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = timeit.default_timer() - start_time
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20, 5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of " + target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of " + target_col)
    plt.legend(loc="best", fancybox=True, shadow=True)

    plt.show()

    # return linear_model.score(X_test,y_test),rmse,linear_model.coef_
    return {
        "best_params": None,
        "R2": model.score(X_test, y_test),
        "rmse": rmse,
        "model": model.__class__.__name__,
        "time_elapsed": elapsed,
    }


def grid_search_cv_func(
    df,
    target_col,
    feature_cols,
    param_grid,
    scoring,
    model,
    test_size,
    random_state,
    cv,
    X_train,
    X_test,
    y_train,
    y_test,
):
    # X_train,X_test,y_train,y_test=train_test_split(df[feature_cols].values,df[target_col].values,test_size=test_size,random_state=random_state)
    ridge = GridSearchCV(model, param_grid, scoring=scoring, cv=5)
    start_time = timeit.default_timer()
    ridge.fit(X_train, y_train)
    y_pred = ridge.best_estimator_.predict(X_test)
    elapsed = timeit.default_timer() - start_time
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20, 5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of " + target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of " + target_col)
    plt.legend(loc="best", fancybox=True, shadow=True)

    plt.show()
    # ,best_score,rmse,model_name,time_elapsed
    return {
        "best_params": ridge.best_params_,
        "R2": ridge.best_score_,
        "rmse": rmse,
        "model": model.__class__.__name__,
        "time_elapsed": elapsed,
    }


def grid_search_cv_class_func(
    df,
    target_col,
    feature_cols,
    param_grid,
    scoring,
    model,
    test_size,
    random_state,
    cv,
    X_train,
    X_test,
    y_train,
    y_test,
):
    # X_train,X_test,y_train,y_test=train_test_split(df[feature_cols].values,df[target_col].values,test_size=test_size,random_state=random_state)
    grid = GridSearchCV(model, param_grid, scoring, cv)
    start_time = timeit.default_timer()
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    elapsed = timeit.default_timer() - start_time
    # rmse= np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20, 5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of " + target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of " + target_col)
    plt.legend(loc="best", fancybox=True, shadow=True)

    plt.show()
    # ,best_score,rmse,model_name,time_elapsed
    return {
        "best_params": grid.best_params_,
        "R2": grid.best_score_,
        "rmse": "0",
        "model": model.__class__.__name__,
        "time_elapsed": elapsed,
    }


def random_search_cv_func(
    df,
    target_col,
    feature_cols,
    param_grid,
    scoring,
    model,
    test_size,
    random_state,
    cv,
    X_train,
    X_test,
    y_train,
    y_test,
):
    # X_train,X_test,y_train,y_test=train_test_split(df[feature_cols].values,df[target_col].values,test_size=test_size,random_state=random_state)
    ridge = RandomizedSearchCV(model, param_grid, scoring=scoring, cv=5)
    start_time = timeit.default_timer()
    ridge.fit(X_train, y_train)
    y_pred = ridge.best_estimator_.predict(X_test)
    elapsed = timeit.default_timer() - start_time
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20, 5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of " + target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of " + target_col)
    plt.legend(loc="best", fancybox=True, shadow=True)

    plt.show()
    return {
        "best_params": ridge.best_params_,
        "R2": ridge.best_score_,
        "rmse": rmse,
        "model": model.__class__.__name__,
        "time_elapsed": elapsed,
    }


def find_outliers(data, col, name):

    sorted_data = np.sort(data[col])
    Q3 = np.quantile(sorted_data, 0.75)
    Q1 = np.quantile(sorted_data, 0.25)
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    outlier_free_list = [
        x for x in sorted_data if ((x < lower_range) | (x > upper_range))
    ]

    outliers = data.loc[data[col].isin(outlier_free_list)]

    if len(outliers) > 0:
        outliers = outliers[name].values.tolist()
        return list(map(lambda x: (x, col), outliers))
    else:
        return []
