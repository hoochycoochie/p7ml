import uvicorn
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# st.title("modèle de scoring")

# loaded_model = joblib.load("model_xgboost.joblib")

loaded_model = joblib.load("model_linear_regression.joblib")

scaler = joblib.load("standardscaler.bin")
pca = joblib.load("pca.bin")
kmeans = joblib.load("kmeans.bin")
features = joblib.load("features.sav")
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


# init app
app = FastAPI()


class request_body(BaseModel):

    REGION_POPULATION_RELATIVE: float
    REGION_RATING_CLIENT_W_CITY: float
    AGE: float
    PRO_SENIORITY: float
    ACCOUNT_SENIORITY: float
    CREDIT_TERM: float
    AGE_LOAN_FINISH: float
    GOODS_LOAN_PERCENT: float
    DAYS_WORKING_PERCENT: float
    DAYS_UNEMPLOYED: float
    EXT_SOURCE_MEAN: float
    BUREAU_DAYS_CREDIT_mean: float
    BUREAU_DAYS_CREDIT_min: float
    BUREAU_DAYS_CREDIT_max: float
    BUREAU_DAYS_CREDIT_std: float
    BUREAU_DAYS_CREDIT_ENDDATE_mean: float
    BUREAU_DAYS_CREDIT_UPDATE_mean: float
    BUREAU_DAYS_CREDIT_UPDATE_min: float
    BUREAU_DAYS_CREDIT_UPDATE_std: float
    POS_CASH_BALANCE_MONTHS_BALANCE_mean: float
    POS_CASH_BALANCE_MONTHS_BALANCE_min: float
    POS_CASH_BALANCE_MONTHS_BALANCE_var: float
    POS_CASH_BALANCE_MONTHS_BALANCE_sum: float
    COUNT_POS_SALE: int
    INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_mean: float
    INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_min: float
    INSTALLMENTS_PAYMENTS_DAYS_INSTALMENT_var: float
    INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_mean: float
    INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_min: float

    # INSTALLMENTS_PAYMENTS_DAYS_ENTRY_PAYMENT_var: float
    # PREVIOUS_APPLICATION_AMT_ANNUITY_mean: float
    # PREVIOUS_APPLICATION_RATE_DOWN_PAYMENT_mean: float
    # PREVIOUS_APPLICATION_RATE_DOWN_PAYMENT_max: float
    # PREVIOUS_APPLICATION_CNT_PAYMENT_max: float
    # PREVIOUS_APPLICATION_CNT_PAYMENT_sum: float
    # frequent_BUREAU_CREDIT_ACTIVE_Closed: float
    # frequent_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Refused: float
    # NAME_INCOME_TYPE_Working: int
    # NAME_EDUCATION_TYPE_Higher_education: int
    # CODE_GENDER_M: int
    # CODE_GENDER_F: int
    # frequent_BUREAU_CREDIT_ACTIVE_Active: int
    # NAME_EDUCATION_TYPE_Secondary_secondary_special: int
    # NAME_INCOME_TYPE_Pensioner: int
    # frequent_PREVIOUS_APPLICATION_CODE_REJECT_REASON_HC: int
    # frequent_PREVIOUS_APPLICATION_NAME_CONTRACT_STATUS_Approved: int
    # frequent_PREVIOUS_APPLICATION_NAME_PAYMENT_TYPE_XNA: int
    # NAME_CONTRACT_TYPE_Cash_loans: int
    # NAME_CONTRACT_TYPE_Revolving_loans: int


@app.post("/predict")
async def classify(data: request_body):
    dist_data = data.dict()
    df = pd.DataFrame([dist_data])

    scaled_data = scaler.transform(df.values)

    pca_data = pca.transform(scaled_data)

    y_pred_proba = loaded_model.best_estimator_.predict_proba(pca_data)[0]
    y_pred_proba = y_pred_proba.tolist()

    result = {0: y_pred_proba[0], 1: y_pred_proba[1]}
    return result


@app.post("/cluster")
async def cluster(data: request_body):

    dist_data = data.dict()

    df = pd.DataFrame([dist_data])

    scaled_data = scaler.transform(df.values)

    pca_data = pca.transform(scaled_data)

    label = int(kmeans.predict(pca_data)[0])

    return label


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
