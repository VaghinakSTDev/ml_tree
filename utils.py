import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

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
    from sklearn import preprocessing
    x = df.values
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_X = preprocessing.normalize(x_scaled)
    return pd.DataFrame(preprocessing.StandardScaler().fit_transform(normalized_X))


def churned_and_not_churned_data(df, size=1000):
    churned_df = df.loc[df['Churn'] == 1]
    not_churned_df = df.loc[df['Churn'] == 0].sample(n=size)
    return not_churned_df.append(churned_df.sample(n=size))


def handle_data(df, bread_names, state_names):
    df = df
    df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000

    df = pd.DataFrame(df).fillna(method='ffill')
    bread_names_column = df['BreedName'].tolist()
    state_names_column = df['ControllingStateCd'].tolist()


    breads_encoder = LabelEncoder().fit(bread_names)
    states_encoder = LabelEncoder().fit(state_names)
    df['BreedName'] = breads_encoder.transform(bread_names_column)
    import pdb
    # pdb.set_trace()
    df['ControllingStateCd'] = states_encoder.transform(state_names_column)
    balance_data = normlize_df(df)
    X = balance_data.values.tolist()

    return np.array(X)