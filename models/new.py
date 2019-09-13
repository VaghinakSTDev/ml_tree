import pandas as pd
import numpy as np
import pickle
import progressbar
import pdb
from tensorflow.keras.callbacks import TensorBoard

import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, classification_report,
                             confusion_matrix
                             )


tensorboard = TensorBoard(log_dir='/var/www/tree/logs/', write_graph=True)

K.set_floatx('float32')
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))


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
                 "PetId",
                 "Churn"
                 ]

EXCLUDE_REASONS = [
    "StateChange",
    "TestPolicy",
    "UnderwritingReasons",
    "PolicyReplaced",
    "DuplicatePolicy",
    "OwnershipTerminated",
    "DataFix",
    "CustomerMovedOutOfTheCountry",
    "Last Day of Policy, Will Reactivate",
]

def balancing_data(df):
    caceled_df = df.loc[df['Churn'] == 1]
    not_caceled_df_count = df.loc[df['Churn'] == 0].shape[0]
    caceled_df_count = caceled_df.shape[0]
    for i in range(int(not_caceled_df_count/caceled_df_count - 1)):
        df = df.append(caceled_df, ignore_index=True)
    return df

def normlize_df(df):
    from sklearn import preprocessing
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_X = preprocessing.normalize(x_scaled)
    return pd.DataFrame(normalized_X)

def handle_data(path):
    df = pd.read_excel(path, usecols=INPUT_COLUMNS)
    df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000
    churned_ids = df.loc[df['Churn'] == 1]['PetId'].values.tolist()
    canceled_df = df[df["PetId"].isin(churned_ids)]

    df = df.drop(canceled_df.loc[canceled_df['Churn'] == 0].index)

    # balance_data[balance_data["PetId"].isin(churned_ids) & balance_data['Churn'] == 0]
    # balance_data = balance_data.drop(
    #     balance_data[balance_data["PetId"].isin(churned_ids) & balance_data['Churn'] == 0]
    # )

    df = pd.DataFrame(df).fillna(method='ffill').drop_duplicates()
    # balance_data = self.balancing_data(df)
    balance_data = df
    # balance_data = pd.read_csv('data/imbalanced_df.scv')

    Y = balance_data['Churn'].values
    # balance_data = balance_data.drop(["Churn", ], axis=1)
    train, test = train_test_split(df, test_size=0.3)

    test.to_excel('data/test_data.xlsx')

    bread_names_column = balance_data['BreedName'].tolist()
    state_names_column = balance_data['ControllingStateCd'].tolist()
    balance_data = balance_data.drop(["BreedName", "Churn", "ControllingStateCd", 'PetId'], axis=1)

    BREED_NAMES = set()
    STATES_NAMES = set()

    [BREED_NAMES.add(breed_name) for breed_name in bread_names_column]
    [STATES_NAMES.add(state) for state in state_names_column]

    BREED_NAMES = list(BREED_NAMES)
    STATES_NAMES = list(STATES_NAMES)

    balance_data = normlize_df(balance_data)

    X = balance_data.values.tolist()

    index = 0
    bar = progressbar.ProgressBar(
        maxval=len(X), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    )
    bar.start()
    bar_index = 0
    for row in X:
        for i in range(len(BREED_NAMES)):
            if bread_names_column[index] == BREED_NAMES[i]:
                row.append(1)
            else:
                row.append(0)
        for j in range(len(STATES_NAMES)):
            if state_names_column[index] == STATES_NAMES[j]:
                row.append(1)
            else:
                row.append(0)
        index += 1
        bar_index += 1
    bar.finish()

    with open('data/binary_property.pickle', 'wb') as f:
        pickle.dump((balance_data.columns, BREED_NAMES, STATES_NAMES), f)
    for i in X[0]:
        print(type(i), i)
    return train_test_split(np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), test_size=0.3, shuffle=True)


PATH = "data/10k.xlsx"
# PATH = "data/STdevChurnData.xlsx"

X_train, X_test, Y_train, Y_test = handle_data(PATH)

model = Sequential()
model.add(Dense(len(X_train[0]), kernel_initializer='uniform',   activation='relu'))
model.add(Dense(int(len(X_train[0])/2), kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    X_train, Y_train, epochs=10, batch_size=128, use_multiprocessing=True,
    validation_data=(X_test, Y_test), callbacks=[tensorboard]
)
model.save('data/binary_model.h5')

y_pred = model.predict(X_test)
print(y_pred)
print("accuracy_score\n", accuracy_score(Y_test, y_pred.round()))
print("classification_report\n", classification_report(Y_test, y_pred.round()))
pdb.set_trace()

