import pandas as pd
import streamlit as st
import joblib

import matplotlib.pyplot as plt
import math
import requests
import plotly.express as px
import json
from functions import scoring as utils

# st.title("modèle de scoring")
URL_API = "http://api:8000"
# URL_API = "http://localhost:8000"

PREDICT_URL = URL_API + "/predict"
CLUSTER_URL = URL_API + "/cluster"

headers = {"Content-type": "application/json"}
features = joblib.load("features.sav")
reference_df = pd.read_csv("data/reference_df.csv")
important_features = joblib.load("important_features.sav")
COLS_GROUP = [
    "AGE",
    "EXT_SOURCE_MEAN",
    "CODE_GENDER",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "CNT_CHILDREN",
    "ANNUITY_AMT_CREDIT_PERCENT",
    "PRO_SENIORITY",
    "ANNUITY_INCOME_PERCENT",
    "CREDIT_TERM",
    "AGE_LOAN_FINISH",
    "label",
]
KEY_COLUMN = "SK_ID_CURR"


# @st.cache
# def load_users():
#     users = pd.read_csv("data/df_test.csv")

#     target_0 = users[users["TARGET"] == 0].head(5)
#     target_1 = users[users["TARGET"] == 1].head(5)

#     # users = users[users["TARGET"] == "1"]
#     users = target_1.append(target_0)
#     ids = users[KEY_COLUMN].tolist()

#     return users, ids


def one_hot_encoding_dataframe(df):
    """
    one hot encoding
    """
    original_columns = list(df.columns)
    cat_columns = [x for x in df.columns if df[x].dtype == "object"]
    df = pd.get_dummies(df, columns=cat_columns, dummy_na=False)
    new_added_columns = list(set(df.columns).difference(set(original_columns)))
    return df, new_added_columns, df.columns


def make_request_prediction(payload):

    y_pred_proba = requests.post(PREDICT_URL, data=payload, headers=headers)
    y_pred_proba = y_pred_proba.json()

    result = {0: y_pred_proba["0"], 1: y_pred_proba["1"]}
    return result


def make_request_cluster(payload):

    y_pred_proba = requests.post(CLUSTER_URL, data=payload, headers=headers)
    y_pred_proba = y_pred_proba.json()

    return y_pred_proba


def populate_payload(user):
    payload = {
        "REGION_POPULATION_RELATIVE": user[0],
        "REGION_RATING_CLIENT_W_CITY": user[1],
        "AGE": user[2],
        "PRO_SENIORITY": user[3],
        "ACCOUNT_SENIORITY": user[4],
        "CREDIT_TERM": user[5],
        "AGE_LOAN_FINISH": user[6],
        "GOODS_LOAN_PERCENT": user[7],
        "DAYS_WORKING_PERCENT": user[8],
        "DAYS_UNEMPLOYED": user[9],
        "EXT_SOURCE_MEAN": user[10],
        "BUREAU_DAYS_CREDIT_mean": user[11],
        "BUREAU_DAYS_CREDIT_min": user[12],
        "BUREAU_DAYS_CREDIT_max": user[13],
        "BUREAU_DAYS_CREDIT_std": user[14],
        "BUREAU_DAYS_CREDIT_ENDDATE_mean": user[15],
        "BUREAU_DAYS_CREDIT_UPDATE_mean": user[16],
        "BUREAU_DAYS_CREDIT_UPDATE_min": user[17],
        "BUREAU_DAYS_CREDIT_UPDATE_std": user[18],
        "POS_CASH_BALANCE_MONTHS_BALANCE_mean": user[19],
        "POS_CASH_BALANCE_MONTHS_BALANCE_min": user[20],
        "POS_CASH_BALANCE_MONTHS_BALANCE_var": user[21],
        "POS_CASH_BALANCE_MONTHS_BALANCE_sum": user[22],
        "COUNT_POS_SALE": user[23],
        "INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_mean": user[24],
        "INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_min": user[25],
        "INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_var": user[26],
        "INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_mean": user[27],
        "INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_min": user[28],
        # "INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_var": user[29],
        # "PREVIOUS_APPLICATION_AMT_ANNUITY_mean": user[30],
        # "PREVIOUS_APPLICATION_RATE_DOWN_PAYMENT_mean": user[31],
        # "PREVIOUS_APPLICATION_RATE_DOWN_PAYMENT_max": user[32],
        # "PREVIOUS_APPLICATION_CNT_PAYMENT_max": user[33],
        # "PREVIOUS_APPLICATION_CNT_PAYMENT_sum": user[34],
        # "frequent_BUREAU_CREDIT_ACTIVE_Closed": user[35],
        # "frequent_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Refused": user[36],
        # "NAME_INCOME_TYPE_Working": user[37],
        # "NAME_EDUCATION_TYPE_Higher_education": user[38],
        # "CODE_GENDER_M": user[39],
        # "CODE_GENDER_F": user[40],
        # "frequent_BUREAU_CREDIT_ACTIVE_Active": user[41],
        # "NAME_EDUCATION_TYPE_Secondary_secondary_special": user[42],
        # "NAME_INCOME_TYPE_Pensioner": user[43],
        # "frequent_PREVIOUS_APPLICATION_CODE_REJECT_REASON_HC": user[44],
        # "frequent_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Approved": user[45],
        # "frequent_PREVIOUS_APPLICATION_NAME_PAYMENT_TYPE_XNA": user[46],
        # "NAME_CONTRACT_TYPE_Cash_loans": user[47],
        # "NAME_CONTRACT_TYPE_Revolving_loans": user[48],
    }
    payload = json.dumps(payload)
    return payload


# @st.cache
def load_group(label):

    users = pd.read_csv("data/aggregated_df.csv")

    users = users[users.label == label][COLS_GROUP]

    return users


# users, ids = load_users()
upload = st.sidebar.checkbox("Charger les données d'un client", True, "show_sample")
# user_id = st.sidebar.selectbox("Choisir un client", ids, disabled=upload)


def display_user(user, selected_user_raw):
    payload = populate_payload(user)
    y_pred_proba = make_request_prediction(payload)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    label_0 = "Solvable " + str(round(y_pred_proba[0] * 100)) + "%"
    label_1 = "Non solvable " + str(round(y_pred_proba[1] * 100)) + "%"
    labels = (label_0, label_1, " ")
    sizes = [y_pred_proba[0], y_pred_proba[1]]
    explode = (0, 1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    colors = ["lightgreen", "red", "white"]
    val = [y_pred_proba[0], y_pred_proba[1]]
    val.append(sum(val))
    fig = plt.figure(figsize=(1, 2), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.pie(
        val,
        labels=labels,
        colors=colors,
        textprops={"fontsize": 5, "fontstyle": "italic"},
    )
    ax.add_artist(plt.Circle((0, 0), 0.6, color="white"))
    st.pyplot(fig)
    rf_user = selected_user_raw[important_features]

    st.markdown("valeurs des features importance des clients solvables")
    st.dataframe(reference_df)

    st.markdown("valeurs des features importance du client actuel")
    st.dataframe(rf_user)
    label = make_request_cluster(payload)

    if label:
        st.markdown("Label group:" + str(label))

        group_df = load_group(label)

        st.markdown("statistiques descriptive du groupe " + str(label))
        st.dataframe(group_df.describe())
        relevant_cols = [
            "AGE",
            "EXT_SOURCE_MEAN",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "ANNUITY_AMT_CREDIT_PERCENT",
            "ANNUITY_INCOME_PERCENT",
            "AGE_LOAN_FINISH",
        ]
        for col in relevant_cols:
            arr = group_df[col]
            bins = round(1 + 3.322 * math.log(len(group_df)))
            # fig, ax = plt.subplots()
            # ax.hist(arr, bins=bins)
            # ax.set_title("Histogramme  de la variable " + col)
            fig = px.histogram(
                arr,
                x=col,
                nbins=bins,
                title="Histogramme  de la variable " + col + ", groupe " + str(label),
                opacity=0.8,
                color_discrete_sequence=["indianred"],  # color of histogram bars
            )
            st.plotly_chart(fig)


def main():

    if upload == True:
        main_upload = st.sidebar.file_uploader(
            label="df_main csv",
            type="csv",
        )

        bureau_upload = st.sidebar.file_uploader(
            label="df_bureau csv",
            type="csv",
        )
        pos_cash_balance_upload = st.sidebar.file_uploader(
            label="df_pos_cash_balance csv",
            type="csv",
        )
        installments_payments_upload = st.sidebar.file_uploader(
            label="df_installments_payments csv",
            type="csv",
        )
        credit_card_balance_upload = st.sidebar.file_uploader(
            label="df_credit_card_balance csv",
            type="csv",
        )

        previous_application_upload = st.sidebar.file_uploader(
            label="df_previous_application csv",
            type="csv",
        )

        if (
            main_upload is not None
            and bureau_upload is not None
            and pos_cash_balance_upload is not None
            and installments_payments_upload is not None
            and credit_card_balance_upload is not None
            and previous_application_upload is not None
        ):

            df_train = pd.read_csv(main_upload)
            df_bureau = pd.read_csv(bureau_upload)
            df_pos_cash_balance = pd.read_csv(pos_cash_balance_upload)
            df_installments_payments = pd.read_csv(installments_payments_upload)
            df_credit_card_balance = pd.read_csv(credit_card_balance_upload)
            df_previous_application = pd.read_csv(previous_application_upload)
            params = {
                "df_train": df_train,
                "df_bureau": df_bureau,
                "df_pos_cash_balance": df_pos_cash_balance,
                "df_installments_payments": df_installments_payments,
                "df_credit_card_balance": df_credit_card_balance,
                "df_previous_application": df_previous_application,
            }
            df = utils.manage_all_df(params)
            df = df[[col for col in df.columns.tolist() if col not in ["label"]]]
            df_all, new_added_columns, columns = one_hot_encoding_dataframe(df)
            selected_user_raw = df_all
            for col in features:
                if col not in selected_user_raw.columns.tolist():
                    selected_user_raw[col] = 0

            selected_user_raw = selected_user_raw[features]

            cols = [
                col
                for col in selected_user_raw.columns.tolist()
                if col not in ["SK_ID_CURR", "SK_ID_BUREAU", "index", "TARGET"]
            ]
            selected_user_raw = selected_user_raw[cols]

            user = selected_user_raw.iloc[0].tolist()
            display_user(user, selected_user_raw)


main()
