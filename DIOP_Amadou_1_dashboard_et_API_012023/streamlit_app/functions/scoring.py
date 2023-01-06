import numpy as np
import pandas as pd


def columns_na_percentage(df):
    na_df = (
        (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).reset_index()
    )
    na_df.columns = ["Column", "na_rate_percent"]
    return na_df


def one_hot_encoding_dataframe(df):
    """
    one hot encoding
    """
    original_columns = list(df.columns)
    cat_columns = [x for x in df.columns if df[x].dtype == "object"]
    df = pd.get_dummies(df, columns=cat_columns, dummy_na=False)
    new_added_columns = list(set(df.columns).difference(set(original_columns)))
    return df, new_added_columns, df.columns


def numeric_vars_agg(df_var, groupby_var, agg, suffix, cols_to_remove):
    df_agg = df_var
    if set(cols_to_remove).issubset(df_var.columns.tolist()):
        df_agg = df_var.drop(columns=cols_to_remove)

    df_agg = df_agg.groupby(groupby_var, as_index=False).agg(agg).reset_index()

    # List of column names
    columns = [groupby_var]

    # Iterate through the variables names
    for var in df_agg.columns.levels[0]:
        # Skip the id name
        if var != groupby_var:

            # Iterate through the stat names
            for stat in df_agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append("%s_%s_%s" % (suffix, var, stat))

    # Assign the list of columns names as the dataframe column names
    df_agg.columns = columns
    # df_train_test=df_train_test.merge(df_agg,on=groupby_var,how='left')
    return df_agg
    # df_bureau_agg.head()


def manage_all_df(params):
    df_train = params["df_train"]
    df_bureau = params["df_bureau"]
    df_pos_cash_balance = params["df_pos_cash_balance"]
    df_installments_payments = params["df_installments_payments"]
    df_credit_card_balance = params["df_credit_card_balance"]
    df_previous_application = params["df_previous_application"]
    df_train = manage_train_df(df_train)
    df_bureau = manage_bureau_df(df_bureau)

    df_pos_cash_balance = manage_pos_cash_balance_df(df_pos_cash_balance)
    df_installments_payments = manage_installments_payments_df(df_installments_payments)
    df_previous_application = manage_previous_application_df(df_previous_application)
    df_credit_card_balance = manage_credit_card_balance_df(df_credit_card_balance)

    df_train = df_train.merge(df_bureau, on="SK_ID_CURR", how="left")
    df_train = df_train.merge(df_pos_cash_balance, on="SK_ID_CURR", how="left")
    df_train = df_train.merge(df_installments_payments, on="SK_ID_CURR", how="left")
    df_train = df_train.merge(df_credit_card_balance, on="SK_ID_CURR", how="left")
    df = df_train.merge(df_previous_application, on="SK_ID_CURR", how="left")
    na_df = columns_na_percentage(df)
    na_columns = na_df[
        na_df["na_rate_percent"] > 30
    ]  ## 49 colonnes ont  des valeurs nulles
    cols_to_remove = na_columns.Column.tolist()

    if set(cols_to_remove).issubset(df.columns.tolist()):
        df.drop(columns=cols_to_remove, axis=1, inplace=True)

    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in string_cols:
        df[col] = df[col].fillna("unknown")

    for col in numeric_cols:
        if col not in ["SK_ID_CURR", "SK_ID_BUREAU", "index", "TARGET"]:
            df[col] = df[col].fillna(df[col].median())
            max_value = np.nanmax(df[col][df[col] != np.inf])
            df[col].replace([np.inf, -np.inf], max_value, inplace=True)

    df["INTEREST_RATE"] = (
        (df["CREDIT_TERM"] * df["AMT_ANNUITY"] / df["AMT_CREDIT"]) - 1
    ) * 100

    return df


def manage_bureau_df(df_bureau):
    df_bureau = clean_columns(df_bureau, threshold=30)
    numeric_cols = df_bureau.select_dtypes(exclude=["object"]).columns.tolist()
    string_cols = df_bureau.select_dtypes(include=["object"]).columns.tolist()
    nb_past_loans = (
        df_bureau.groupby("SK_ID_CURR", as_index=False)["SK_ID_BUREAU"]
        .count()
        .rename(columns={"SK_ID_BUREAU": "COUNT_PAST_LOANS"})
    )

    cols_to_remove = ["CREDIT_CURRENCY"]

    if set(cols_to_remove).issubset(string_cols):
        string_cols.remove(cols_to_remove[0])

    if set(cols_to_remove).issubset(df_bureau.columns.tolist()):
        df_bureau.drop(columns=cols_to_remove, axis=1, inplace=True)

    for col in string_cols:
        df_bureau[col] = df_bureau[col].fillna("unknown")

    for col in numeric_cols:
        if col not in ["SK_ID_CURR", "SK_ID_BUREAU", "index"]:
            df_bureau[col] = df_bureau[col].fillna(df_bureau[col].median())

    df_bureau_agg_numeric = numeric_vars_agg(
        df_bureau,
        "SK_ID_CURR",
        ["mean", "min", "max", "std"],
        "BUREAU",
        ["SK_ID_BUREAU"],
    )
    df_bureau_agg_str = most_frequent_str_var(
        df_bureau, string_cols, "SK_ID_CURR", "BUREAU"
    )
    df_bureau_final = pd.merge(
        pd.merge(df_bureau_agg_numeric, df_bureau_agg_str, on="SK_ID_CURR"),
        nb_past_loans,
        on="SK_ID_CURR",
    )
    return df_bureau_final


def clean_columns(df, threshold=30):
    na_df = columns_na_percentage(df)
    na_columns = na_df[na_df["na_rate_percent"] >= threshold]
    df.drop(columns=na_columns["Column"].tolist(), axis=1, inplace=True)
    return df


def most_frequent_str_var(df_var, str_cols, groupby_var, suffix):
    related_cols = str_cols + [groupby_var]
    agg = dict()
    agg_rename_cols = dict()
    for col in str_cols:
        agg[col] = lambda x: pd.Series.mode(x)[0]
        agg_rename_cols[col] = "frequent-{}-{}".format(suffix, col)

    df_agg = (
        df_var[related_cols]
        .groupby(groupby_var)
        .agg(agg)
        .reset_index()
        .rename(columns=agg_rename_cols)
    )

    return df_agg


def manage_pos_cash_balance_df(df_pos_cash_balance):
    df_pos_cash_balance = clean_columns(df_pos_cash_balance, threshold=30)
    numeric_cols = df_pos_cash_balance.select_dtypes(
        exclude=["object"]
    ).columns.tolist()
    string_cols = df_pos_cash_balance.select_dtypes(include=["object"]).columns.tolist()

    for col in string_cols:
        df_pos_cash_balance[col] = df_pos_cash_balance[col].fillna("unknown")

    for col in numeric_cols:
        df_pos_cash_balance[col] = df_pos_cash_balance[col].fillna(
            df_pos_cash_balance[col].median()
        )

    df_pos_cash_balance_agg_numeric = numeric_vars_agg(
        df_pos_cash_balance,
        "SK_ID_CURR",
        ["mean", "min", "max", "var", "sum"],
        "POS_CASH_BALANCE",
        ["SK_ID_PREV"],
    )
    str_cols = df_pos_cash_balance.select_dtypes(include=["object"]).columns.tolist()
    df_pos_cash_balance_agg_str = most_frequent_str_var(
        df_pos_cash_balance, str_cols, "SK_ID_CURR", "POS_CASH_BALANCE"
    )
    nb_pos_sale = (
        df_pos_cash_balance.groupby("SK_ID_CURR", as_index=False)["SK_ID_PREV"]
        .count()
        .rename(columns={"SK_ID_PREV": "COUNT_POS_SALE"})
    )
    df_pos_cash_balance_final = pd.merge(
        pd.merge(
            df_pos_cash_balance_agg_numeric,
            df_pos_cash_balance_agg_str,
            on="SK_ID_CURR",
        ),
        nb_pos_sale,
        on="SK_ID_CURR",
    )
    return df_pos_cash_balance_final


def manage_installments_payments_df(df_installments_payments):
    df_installments_payments = clean_columns(df_installments_payments, 30)
    cols_to_remove = ["NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]

    if set(cols_to_remove).issubset(df_installments_payments.columns.tolist()):
        df_installments_payments.drop(columns=cols_to_remove, axis=1, inplace=True)

    nb_installment = (
        df_installments_payments.groupby("SK_ID_CURR", as_index=False)["SK_ID_PREV"]
        .count()
        .rename(columns={"SK_ID_PREV": "COUNT_PAST_INSTALLMENT"})
    )
    df_installments_payments_agg_numeric = numeric_vars_agg(
        df_installments_payments,
        "SK_ID_CURR",
        ["mean", "min", "max", "var", "sum"],
        "INSTALLMENTS_PAYMENTS",
        ["SK_ID_PREV"],
    )
    df_installments_payments_final = pd.merge(
        df_installments_payments_agg_numeric, nb_installment, on="SK_ID_CURR"
    )
    return df_installments_payments_final


def manage_credit_card_balance_df(df_credit_card_balance):
    df_credit_card_balance = clean_columns(df_credit_card_balance, threshold=30)
    numeric_cols = df_credit_card_balance.select_dtypes(
        exclude=["object"]
    ).columns.tolist()
    string_cols = df_credit_card_balance.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for col in string_cols:
        df_credit_card_balance[col] = df_credit_card_balance[col].fillna("unknown")

    for col in numeric_cols:
        df_credit_card_balance[col] = df_credit_card_balance[col].fillna(
            df_credit_card_balance[col].median()
        )

    nb_credit_sale = (
        df_credit_card_balance.groupby("SK_ID_CURR", as_index=False)["SK_ID_PREV"]
        .count()
        .rename(columns={"SK_ID_PREV": "COUNT_CREDIT_SALE"})
    )
    str_cols = df_credit_card_balance.select_dtypes(include=["object"]).columns.tolist()
    str_cols = [col for col in str_cols if col not in ["SK_ID_CURR"]]

    df_credit_card_balance_agg_str = most_frequent_str_var(
        df_credit_card_balance, str_cols, "SK_ID_CURR", "CREDIT_CARD_BALANCE"
    )
    df_credit_card_balance_agg_numeric = numeric_vars_agg(
        df_credit_card_balance,
        "SK_ID_CURR",
        ["mean", "min", "max", "var", "sum"],
        "CREDIT_CARD_BALANCE",
        ["SK_ID_PREV"],
    )
    df_credit_card_balance_final = pd.merge(
        pd.merge(
            df_credit_card_balance_agg_numeric,
            df_credit_card_balance_agg_str,
            on="SK_ID_CURR",
        ),
        nb_credit_sale,
        on="SK_ID_CURR",
    )
    return df_credit_card_balance_final


def manage_previous_application_df(df_previous_application):
    regex = "PHONE|EMAIL|DOCUMENT|MOBIL|CITY|REGION|SOCIAL|DAYS|REQ|START"
    REGEX_COLS = [
        x
        for x in df_previous_application.columns[
            df_previous_application.columns.str.contains(regex)
        ]
    ]

    df_previous_application = df_previous_application.drop(columns=REGEX_COLS)
    cols_to_remove = [
        "NFLAG_LAST_APPL_IN_DAY",
        "FLAG_LAST_APPL_PER_CONTRACT",
        "NAME_GOODS_CATEGORY",
        "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE",
        "SELLERPLACE_AREA",
        "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP",
        "PRODUCT_COMBINATION",
        "NAME_TYPE_SUITE",
    ]

    if set(cols_to_remove).issubset(df_previous_application.columns.tolist()):
        df_previous_application.drop(columns=cols_to_remove, axis=1, inplace=True)

    numeric_cols = df_previous_application.select_dtypes(
        exclude=["object"]
    ).columns.tolist()
    string_cols = df_previous_application.select_dtypes(
        include=["object"]
    ).columns.tolist()

    for col in string_cols:
        df_previous_application[col] = df_previous_application[col].fillna("unknown")

    for col in numeric_cols:
        df_previous_application[col] = df_previous_application[col].fillna(
            df_previous_application[col].median()
        )

    nb_past_home_loans = (
        df_previous_application.groupby("SK_ID_CURR", as_index=False)["SK_ID_PREV"]
        .count()
        .rename(columns={"SK_ID_PREV": "COUNT_PAST_HOME_LOANS"})
    )
    df_previous_application_agg_numeric = numeric_vars_agg(
        df_previous_application,
        "SK_ID_CURR",
        ["mean", "min", "max", "var", "sum"],
        "PREVIOUS_APPLICATION",
        ["SK_ID_PREV"],
    )
    str_cols = df_previous_application.select_dtypes(
        include=["object"]
    ).columns.tolist()
    df_previous_application_agg_str = most_frequent_str_var(
        df_previous_application, str_cols, "SK_ID_CURR", "PREVIOUS_APPLICATION"
    )
    df_previous_application_final = pd.merge(
        pd.merge(
            df_previous_application_agg_numeric,
            df_previous_application_agg_str,
            on="SK_ID_CURR",
        ),
        nb_past_home_loans,
        on="SK_ID_CURR",
    )

    return df_previous_application_final


def manage_train_df(df_train):
    currency = 86
    df_train["AMT_GOODS_PRICE"] = df_train["AMT_GOODS_PRICE"] / currency
    df_train["AMT_CREDIT"] = df_train["AMT_CREDIT"] / currency
    df_train["AMT_ANNUITY"] = df_train["AMT_ANNUITY"] / currency
    df_train["AMT_INCOME_TOTAL"] = df_train["AMT_INCOME_TOTAL"] / currency

    df_train["AMT_INCOME_TOTAL"] = df_train["AMT_INCOME_TOTAL"].apply(np.ceil)
    df_train["AMT_CREDIT"] = df_train["AMT_CREDIT"].apply(np.ceil)
    df_train["AMT_ANNUITY"] = df_train["AMT_ANNUITY"].apply(np.ceil)
    df_train["AGE"] = df_train["DAYS_BIRTH"] / (-365)
    df_train["AGE"] = df_train["AGE"].apply(np.ceil)

    df_train["PRO_SENIORITY"] = df_train["DAYS_EMPLOYED"] / (-365)
    df_train["PRO_SENIORITY"] = df_train["PRO_SENIORITY"].apply(np.ceil)

    df_train["ACCOUNT_SENIORITY"] = df_train["DAYS_REGISTRATION"] / (-365)
    df_train["ACCOUNT_SENIORITY"] = df_train["ACCOUNT_SENIORITY"].apply(np.ceil)

    cols_to_remove = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION"]

    if set(cols_to_remove).issubset(df_train.columns.tolist()):
        df_train.drop(columns=cols_to_remove, axis=1, inplace=True)

    def age_(x):
        if x < 0 or x == -0:
            return np.nan
        return x

    df_train["PRO_SENIORITY"] = df_train["PRO_SENIORITY"].map(age_)
    df_train["ACCOUNT_SENIORITY"] = df_train["ACCOUNT_SENIORITY"].map(age_)
    df_train[["AGE", "ACCOUNT_SENIORITY", "PRO_SENIORITY"]].describe()

    na_df = columns_na_percentage(df_train)
    na_columns = na_df[
        na_df["na_rate_percent"] > 40
    ]  ## 49 colonnes ont  des valeurs nulles
    cols_to_remove = na_columns.Column.tolist()

    if set(cols_to_remove).issubset(df_train.columns.tolist()):
        df_train.drop(columns=cols_to_remove, axis=1, inplace=True)

    cols_to_remove = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "HOUR_APPR_PROCESS_START",
        "DAYS_LAST_PHONE_CHANGE",
        "WEEKDAY_APPR_PROCESS_START",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "FLAG_WORK_PHONE",
        "DAYS_ID_PUBLISH",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]

    if set(cols_to_remove).issubset(df_train.columns.tolist()):
        df_train.drop(columns=cols_to_remove, axis=1, inplace=True)

    cols = [x for x in df_train.columns.tolist() if x not in ["TARGET", "SK_ID_CURR"]]
    numeric_cols = df_train[cols].select_dtypes(exclude=["object"]).columns.tolist()
    string_cols = df_train[cols].select_dtypes(include=["object"]).columns.tolist()
    cols_to_remove = ["ORGANIZATION_TYPE"]
    if set(cols_to_remove).issubset(df_train.columns.tolist()):
        df_train.drop(columns=cols_to_remove, axis=1, inplace=True)

    if set(cols_to_remove).issubset(string_cols):
        string_cols.remove(cols_to_remove[0])

    for col in string_cols:
        df_train[col] = df_train[col].fillna("unknown")

    na_annuity = df_train[(pd.isna(df_train["AMT_ANNUITY"]))]
    df_train.drop(na_annuity.index, axis=0, inplace=True)
    df_train["CNT_FAM_MEMBERS"] = df_train["CNT_FAM_MEMBERS"].fillna(0)
    na_df = columns_na_percentage(df_train[numeric_cols])
    na_columns = na_df[na_df["na_rate_percent"] > 0]
    for col in na_columns["Column"].tolist():
        df_train[col] = df_train[col].fillna(df_train[col].median())

    # * pourcentage de l'annuité par rapport au crédit: rapport entre le montant de l'annuité et celui du crédit dû
    df_train["ANNUITY_AMT_CREDIT_PERCENT"] = (
        df_train["AMT_ANNUITY"] / (df_train["AMT_CREDIT"])
    ) * 100
    df_train["ANNUITY_AMT_CREDIT_PERCENT"] = df_train[
        "ANNUITY_AMT_CREDIT_PERCENT"
    ].apply(np.ceil)
    if "ANNUITY_AMT_CREDIT_PERCENT" not in numeric_cols:
        numeric_cols.append("ANNUITY_AMT_CREDIT_PERCENT")

    # * rapport entre le montant de l'annuité et le revenu annuel du client
    df_train["ANNUITY_INCOME_PERCENT"] = (
        df_train["AMT_ANNUITY"] / df_train["AMT_INCOME_TOTAL"]
    ) * 100
    df_train["ANNUITY_INCOME_PERCENT"] = df_train["ANNUITY_INCOME_PERCENT"].apply(
        np.ceil
    )
    if "ANNUITY_INCOME_PERCENT" not in numeric_cols:
        numeric_cols.append("ANNUITY_INCOME_PERCENT")

    # * la durée du prêt: arrport entre le montant du crédit et l'annuité

    df_train["CREDIT_TERM"] = df_train["AMT_CREDIT"] / df_train["AMT_ANNUITY"]
    df_train["CREDIT_TERM"] = df_train["CREDIT_TERM"].apply(np.ceil)
    # * l'âge de l'emprunteur lorsque le crédit arriverait à terme
    df_train["AGE_LOAN_FINISH"] = df_train["AGE"] + df_train["CREDIT_TERM"]
    df_train["AGE_LOAN_FINISH"] = df_train["AGE_LOAN_FINISH"].apply(np.ceil)

    # * pourcentage du prêt alloué à l'achat de biens et ou services
    df_train["GOODS_LOAN_PERCENT"] = (
        df_train["AMT_GOODS_PRICE"] / df_train["AMT_CREDIT"]
    ) * 100

    df_train["GOODS_LOAN_PERCENT"] = df_train["GOODS_LOAN_PERCENT"].apply(np.ceil)

    # * revenu net du client après paiement de l'annuité: différence entre le revenu annuel et le montant de l'annuité
    df_train["INCOME_AFTER_ANNUITY"] = (
        df_train["AMT_INCOME_TOTAL"] - df_train["AMT_ANNUITY"]
    )
    df_train["INCOME_AFTER_ANNUITY"] = df_train["INCOME_AFTER_ANNUITY"].apply(np.ceil)

    # * revenu net par tête du foyer familial après paiement de l'annuité:
    # rapport le nombre de personnes composant la famille entre le revenu net

    df_train["NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"] = (
        df_train["INCOME_AFTER_ANNUITY"] / df_train["CNT_FAM_MEMBERS"]
    )
    df_train["NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"] = df_train[
        "NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"
    ].apply(np.ceil)
    # NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY    NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY
    # nombre de documents fournis
    regex = "FLAG_DOCUMENT"
    DOCUMENT_COLS = [x for x in df_train.columns[df_train.columns.str.contains(regex)]]
    DOCUMENT_COLS
    df_train["DOCUMENT_COUNT"] = (df_train[DOCUMENT_COLS] == 1).sum(axis=1)
    if set(DOCUMENT_COLS).issubset(df_train.columns.tolist()):
        df_train.drop(columns=DOCUMENT_COLS, axis=1, inplace=True)

    # rapport entr
    df_train["DAYS_WORKING_PERCENT"] = df_train["PRO_SENIORITY"] / df_train["AGE"]

    df_train["DAYS_UNEMPLOYED"] = abs(df_train["AGE"]) - abs(df_train["PRO_SENIORITY"])

    df_train["EXT_SOURCE_MEAN"] = (df_train[["EXT_SOURCE_2", "EXT_SOURCE_3"]]).mean(
        axis=1
    )

    df_train["EXT_SOURCE_MEDIAN"] = (df_train[["EXT_SOURCE_2", "EXT_SOURCE_3"]]).median(
        axis=1
    )

    df_train["EXT_SOURCE_MIN"] = (df_train[["EXT_SOURCE_2", "EXT_SOURCE_3"]]).min(
        axis=1
    )

    df_train["EXT_SOURCE_MAX"] = (df_train[["EXT_SOURCE_2", "EXT_SOURCE_3"]]).max(
        axis=1
    )
    for col in DOCUMENT_COLS:
        if col in numeric_cols:
            numeric_cols.remove(col)
    for col in [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_MEDIAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
    ]:
        if col not in numeric_cols:
            numeric_cols.append(col)

    # find max value of column
    max_value = np.nanmax(
        df_train["NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"][
            df_train["NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"] != np.inf
        ]
    )

    # replace inf and -inf in column with max value of column
    df_train["NET_INCOME_PER_FAMILY_HEAD_AFTER_ANNUITY"].replace(
        [np.inf, -np.inf], max_value, inplace=True
    )

    cols_to_remove = [
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "EXT_SOURCE_MEDIAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
        "AMT_GOODS_PRICE",
        "REGION_RATING_CLIENT",
    ]
    for col in cols_to_remove:
        if col in numeric_cols:
            numeric_cols.remove(col)
    if set(cols_to_remove).issubset(df_train.columns.tolist()):
        df_train.drop(columns=cols_to_remove, axis=1, inplace=True)

    return df_train
