import pdb
import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, Imputer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, classification_report,
                             confusion_matrix
                             )

K.set_floatx('float16')

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
    "ReasonChurn",
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


INPUTS_COUNT = 0
breads_encoder, states_encoder = None, None


def balancing_data(df):
    caceled_df = df.loc[df['Churn'] == 1]
    not_caceled_df_count = df.loc[df['Churn'] == 0].shape[0]
    caceled_df_count = caceled_df.shape[0]
    for i in range(int(not_caceled_df_count / caceled_df_count - 1)):
        df = df.append(caceled_df, ignore_index=True)
    return df


def normalize(data):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = min_max_scaler.fit_transform(data)
    normalized_X = preprocessing.normalize(x_scaled)
    return pd.DataFrame(preprocessing.StandardScaler().fit_transform(normalized_X))


def get_test_train_dataset(df):
    df = handle_data(df)

    return train_test_split(df, test_size=0.3, shuffle=True)


def handle_data(df):
    df = df[INPUT_COLUMNS]
    df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000

    df = pd.DataFrame(df).drop_duplicates(keep='last')
    df = df[~df.ReasonChurn.isin(EXCLUDE_REASONS)]
    df = pd.DataFrame(df).fillna(method='ffill')
    df = pd.DataFrame(df).drop_duplicates(subset=['PetId', 'Churn'], keep='last')

    bread_names_column = df['BreedName'].tolist()
    state_names_column = df['ControllingStateCd'].tolist()

    df = df.drop(['ReasonChurn', 'PetId'], axis=1)

    # imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    #
    # df = imputer.fit_transform(df)

    breed_names = set()
    state_names = set()

    [breed_names.add(breed_name) for breed_name in bread_names_column]
    [state_names.add(state) for state in state_names_column]

    breed_names = list(breed_names)
    state_names = list(state_names)
    global breads_encoder
    global states_encoder

    breads_encoder = LabelEncoder().fit(breed_names)
    states_encoder = LabelEncoder().fit(state_names)

    np.save('data/breads_encoder.npy', breads_encoder.classes_)
    np.save('data/states_encoder.npy', states_encoder.classes_)

    return df


def load_properties():
    breads_encoder = LabelEncoder().fit(np.load('model_data/breads_encoder.npy'))
    states_encoder = LabelEncoder().fit(np.load('model_data/states_encoder.npy'))


def save_model(model, path):
    model.save(path + '_binary_model.h5')


def initial_fit(self):
    X_train, X_test, Y_train, Y_test = self.get_train_test_data(self.X, self.Y)
    self.model.fit(X_train, Y_train, epochs=1, batch_size=16, validation_data=(X_test, Y_test))
    self.evaluate(X_test, Y_test)
    # self.save_model()


def evaluate(model, data, target):
    y_pred = model.predict(data)
    print("precision_score", precision_score(target, y_pred.round(), average='macro'))
    print("recall_score", recall_score(target, y_pred.round(), average='macro'))
    print("f1_score", f1_score(target, y_pred.round(), average='macro'))
    print("classification_report\n", classification_report(target, y_pred.round()))
    print("accuracy_score", accuracy_score(target, y_pred.round()))


def build_model(path):
    df = pd.read_excel(path)
    train_data, test_data = get_test_train_dataset(df)

    test_data.to_excel('data/test_data.xlsx')
    print(train_data)
    train_data['BreedName'] = breads_encoder.transform(train_data['BreedName'])
    train_data['ControllingStateCd'] = states_encoder.transform(train_data['ControllingStateCd'])
    test_data['BreedName'] = breads_encoder.transform(test_data['BreedName'])
    test_data['ControllingStateCd'] = states_encoder.transform(test_data['ControllingStateCd'])

    Y_train = train_data['Churn'].copy().values
    Y_test = test_data['Churn'].copy().values

    train_data.drop("Churn", axis=1, inplace=True)
    test_data.drop("Churn", axis=1, inplace=True)
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

    train_data = imputer.fit_transform(train_data)
    test_data = imputer.fit_transform(test_data)

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    X_train = np.array(train_data, dtype=np.float16)
    X_test = np.array(test_data, dtype=np.float16)

    input_layer_size = len(X_train[0])

    K.set_floatx('float32')

    # sess_gpu = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(tf.test.is_gpu_available())
    from tensorflow.python.client import device_lib

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    adam = Adam(learning_rate=0.001)
    model = Sequential()

    model.add(Dense(input_layer_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))
    save_model(model, path)
    y_pred = model.predict(X_test)

    print("accuracy_score\n", accuracy_score(Y_test, y_pred.round()))

    evaluate(model, X_test, Y_test)


# export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64"


# PATH = "data/10k.xlsx"
PATH = "data/STdevChurnData.xlsx"

if __name__ == "__main__":
    # NUM_GPU = 1

    build_model(PATH)



