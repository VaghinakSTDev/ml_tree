import pandas as pd
import numpy as np

from sklearn import preprocessing

INPUT_COLUMNS = [
                 'LastAnnualPremiumAmt',
                 "CopayPct",
                 "TotalClaims",
                 "TotalClaimsDenied",
                 "InitialPremiumAmtPriorYr",
                 "PolicyForm",
                 "Age",
                 "DurationPet",
                 "InitialWrittenPremiumAmt",
                 "TotalClaimsPaid",
                 "LastAnnualPremiumAmtPriorYr",
                 "BreedName",
                 "ControllingStateCd",
                 "DurationPolicy",
                 ]


def normlize_df(df):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = min_max_scaler.fit_transform(df)
    normalized_X = preprocessing.normalize(x_scaled)
    return pd.DataFrame(preprocessing.StandardScaler().fit_transform(normalized_X))


def churned_and_not_churned_data(df, size=1000):
    churned_df = df.loc[df['Churn'] == 1]
    not_churned_df = df.loc[df['Churn'] == 0].sample(n=size)
    return not_churned_df.append(churned_df.sample(n=size))


def handle_data(df, bread_names, state_names):

    df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000

    df = pd.DataFrame(df).fillna(method='ffill')
    bread_names_column = df['BreedName'].tolist()
    state_names_column = df['ControllingStateCd'].tolist()

    breads_encoder = preprocessing.LabelEncoder().fit(bread_names)
    states_encoder = preprocessing.LabelEncoder().fit(state_names)
    import pdb
    df['BreedName'] = breads_encoder.transform(bread_names_column)
    df['ControllingStateCd'] = states_encoder.transform(state_names_column)

    # imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)

    # df = imputer.fit_transform(df)

    balance_data = normlize_df(df)

    data = balance_data.values.tolist()

    return np.array(data)