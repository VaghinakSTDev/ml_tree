import pandas as pd
import numpy as np
import pickle
import progressbar
import pdb

import tensorflow as tf
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


class Network:
    def __init__(self, path=None):
        # self.num_gpu = 1
        self.path = path
        self.inputs_count = None
        self.model = None
        self.X = None
        self.Y = None
        self.__init_model()

    def __init_model(self):
        """
        The function will try to load the model with the name with the first part of the argument,
         otherwise it will create a new model and train
        :return:
        """
        if self.path:
            path_name = self.path.split('.')[0]
            self.path_name = path_name
            try:
                print('Loading existing model')
                self.load_model()
                with open(path_name + "_binary_property.pickle", "rb") as f:
                    self.columns, self.bread_names, self.state_names = pickle.load(f)
            except Exception:
                print("Loading failed")
                print('Creating a new model')
                self.handle_data()
                self.build_model(self.inputs_count)
        else:
            print("Specify file path")

    def balancing_data(seln, df):
        caceled_df = df.loc[df['Churn'] == 1]
        not_caceled_df_count = df.loc[df['Churn'] == 0].shape[0]
        caceled_df_count = caceled_df.shape[0]
        for i in range(int(not_caceled_df_count/caceled_df_count - 1)):
            df = df.append(caceled_df, ignore_index=True)
        return df

    def normlize_df(self, df):
        from sklearn import preprocessing
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_X = preprocessing.normalize(x_scaled)
        return pd.DataFrame(normalized_X)

    def handle_data(self):
        # df = pd.read_excel(self.path, usecols=INPUT_COLUMNS)
        # df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000
        # churned_ids = df.loc[df['Churn'] == 1]['PetId'].values.tolist()
        # canceled_df = df[df["PetId"].isin(churned_ids)]
        # df = df.drop(canceled_df.loc[canceled_df['Churn'] == 0].index)
        # # balance_data[balance_data["PetId"].isin(churned_ids) & balance_data['Churn'] == 0]
        # # balance_data = balance_data.drop(
        # #     balance_data[balance_data["PetId"].isin(churned_ids) & balance_data['Churn'] == 0]
        # # )
        #
        # df = pd.DataFrame(df).fillna(method='ffill').drop_duplicates()
        # df = df[~df.ReasonChurn.isin(EXCLUDE_REASONS)]
        # balance_data = self.balancing_data(df)
        # train, test = train_test_split(df, test_size=0.3)
        # #
        # test.to_excel('data/test_data.xlsx')
        # balance_data = pd.DataFrame(df).fillna(method='ffill')
        # print("Dataset Shape: ", df.shape)
        # print("canceled", balance_data.loc[balance_data['Churn'] == 1].shape)
        # print("not canceled", balance_data.loc[balance_data['Churn'] == 0].shape)
        # # Printing the dataset obseravtions
        # print("Dataset: ", df.head())
        #
        # balance_data.to_csv('data/balanced_df.scv', encoding='utf-8')
        balance_data = pd.read_csv('data/imbalanced_df.scv')
        print("Dataset Shape: ", balance_data.shape)
        print("canceled", balance_data.loc[balance_data['Churn'] == 1].shape)
        print("not canceled", balance_data.loc[balance_data['Churn'] == 0].shape)

        Y = balance_data['Churn'].values

        bread_names_column = balance_data['BreedName'].tolist()

        state_names_column = balance_data['ControllingStateCd'].tolist()

        balance_data = balance_data.drop(["BreedName", "Churn", "ControllingStateCd", 'PetId'], axis=1)

        BREED_NAMES = set()
        STATES_NAMES = set()

        [BREED_NAMES.add(breed_name) for breed_name in bread_names_column]
        [STATES_NAMES.add(state) for state in state_names_column]

        BREED_NAMES = list(BREED_NAMES)
        STATES_NAMES = list(STATES_NAMES)

        self.columns = balance_data.columns
        self.bread_names = BREED_NAMES
        self.state_names = STATES_NAMES

        balance_data = self.normlize_df(balance_data)

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

        self.X = np.array(X, dtype=np.float16)
        self.Y = np.array(Y, dtype=np.float16)
        self.inputs_count = len(X[0])
        with open(self.path_name + '_binary_property.pickle', 'wb') as f:
            pickle.dump((self.columns, self.bread_names, self.state_names), f)
        # with open(self.path_name + '_binary_data.pickle', 'wb') as f:
        #     pickle.dump((X, Y,), f)

    def get_train_test_data(self, X=None, Y=None, test_size=0.3):
        """
        splitting data for training and testing
        :param X: samples
        :param Y: targets
        :return: X_train, X_test, Y_train, Y_test
        """
        if X is None or Y is None:
            if self.X is None:
                self.handle_data()
            return train_test_split(self.X, self.Y, test_size=0.3, random_state=100, shuffle=True)
        return train_test_split(X, Y, test_size=test_size, random_state=100, shuffle=True)

    def load_model(self):
        self.model = load_model(self.path_name + '_binary_model.h5')

    def save_model(self):
        self.model.save(self.path_name + '_binary_model.h5')

    def initial_fit(self):
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(self.X, self.Y)
        self.model.fit(X_train, Y_train, epochs=1, batch_size=128, validation_data=(X_test, Y_test))
        self.evaluate(X_test, Y_test)
        self.save_model()

    def evaluate(self, X, Y):
        y_pred = self.model.predict_classes(X)
        print("precision_score", precision_score(Y, y_pred.round(), average='macro'))
        print("recall_score", recall_score(Y, y_pred.round(), average='macro'))
        print("f1_score", f1_score(Y, y_pred.round(), average='macro'))
        print("classification_report\n", classification_report(Y, y_pred.round()))
        print("accuracy_score", accuracy_score(Y, y_pred.round()))

    def build_model(self, input_layer_size):
        K.set_floatx('float32')

        # sess_gpu = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(tf.test.is_gpu_available())
        from tensorflow.python.client import device_lib

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        adam = Adam(learning_rate=0.000007)
        model = Sequential()


        model.add(Dense(input_layer_size, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(int(input_layer_size / 2), kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model = model
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(self.X, self.Y)




        history = self.model.fit(
            X_train, Y_train, epochs=20, batch_size=64
            , validation_data=(X_test, Y_test)
        )
        self.save_model()

        print(X_test[0])
        y_pred = self.model.predict(X_test)
        print(history.history)
        print("classification_report\n", classification_report(Y_test, y_pred.round()))
        print("accuracy_score\n", accuracy_score(Y_test, y_pred.round()))
        # print(confusion_matrix(X_test, Y_test))
        self.model.evaluate(X_test, Y_test, verbose=1)
        # self.initial_fit()

    def predict(self, X):
        return self.model.predict_classes(np.array(X, dtype=np.float16), verbose=1)
# export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64"


# PATH = "data/10k.xlsx"
PATH = "data/STdevChurnData.xlsx"


if __name__ == "__main__":
    # NUM_GPU = 1
    network = Network(PATH)
    network.handle_data()
    network.build_model(network.inputs_count)




# pip install TensorFlow==1.13.1


