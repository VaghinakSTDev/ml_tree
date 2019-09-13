import pandas as pd
import numpy as np
import pickle
import progressbar
import logging

import tensorflow
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

logging.basicConfig(
    filename='logs/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG
)
print(device_lib.list_local_devices())


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
                 # "DurationPolicy",
                 "target"
                 ]


class Network:
    def __init__(self, path=None):
        self.num_gpu = 1
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
                with open(path_name + "_property.pickle", "rb") as f:
                    self.X, self.Y = pickle.load(f)
            except FileNotFoundError:
                print('Creating a new model')
                self.handle_data()
                self.build_model(self.inputs_count)
        else:
            logging.error("Specify file path")

    def balancing_data(self):
        """
        balancing data in by three categories
        :return:
        """
        print("Starting data balancing")
        to_low = self.balance_data.loc[self.balance_data['target'] == -1]
        to_high = self.balance_data.loc[self.balance_data['target'] == 1]
        normal = self.balance_data.loc[self.balance_data['target'] == 0]
        len_to_low = to_low.shape[0] if to_low.shape[0] else 1
        len_to_high = to_high.shape[0] if to_high.shape[0] else 1
        len_to_same = normal.shape[0] if normal.shape[0] else 1
        balanced_df = pd.DataFrame()

        balanced_df = balanced_df.append(pd.concat([to_low]*int(len_to_same/len_to_low + 0.5)))
        balanced_df = balanced_df.append(pd.concat([to_high]*int(len_to_same/len_to_high + 0.5)))

        balanced_df = balanced_df.append(normal)

        print("End balancing")
        self.balance_data = balanced_df

    def normlize_df(self, df):
        """
        Normalize the data to scale the variable to have values ​​from 0 to 1
        :param df:
        :return:
        """
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        normalized_X = preprocessing.normalize(x_scaled)
        return pd.DataFrame(normalized_X)

    def handle_data(self):
        """
        obtaining data from CSV, normalizing and separating samples and targets
        :return:
        """
        self.balance_data = pd.read_csv(str(self.path), usecols=INPUT_COLUMNS)
        self.balance_data.loc[self.balance_data['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000
        self.balance_data = pd.DataFrame(self.balance_data).fillna(method='ffill')

        print("Dataset Shape: ", self.balance_data.shape)
        print("Dataset: ", self.balance_data.head())

        self.balancing_data()
        Y = self.balance_data['target'].values

        bread_names_column = self.balance_data['BreedName'].tolist()
        state_names_column = self.balance_data['ControllingStateCd'].tolist()

        self.balance_data = self.balance_data.drop(["BreedName", "target", "ControllingStateCd"], axis=1)

        BREED_NAMES = set()
        STATES_NAMES = set()
        [BREED_NAMES.add(breed_name) for breed_name in bread_names_column]
        [STATES_NAMES.add(state) for state in state_names_column]

        BREED_NAMES = list(BREED_NAMES)
        STATES_NAMES = list(STATES_NAMES)

        self.columns = self.balance_data.columns
        self.bread_names = BREED_NAMES
        self.state_names = STATES_NAMES

        balance_data = self.normlize_df(self.balance_data)

        X = balance_data.values.tolist()

        index = 0
        bar = progressbar.ProgressBar(
            maxval=len(X), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        )
        bar.start()
        bar_index = 0

        for row in X:
            bar.update(bar_index + 1)
            for i in range(len(BREED_NAMES) - 1):
                if bread_names_column[index] == BREED_NAMES[i]:
                    row.append(1)
                else:
                    row.append(0)
            for j in range(len(STATES_NAMES) - 8):
                if state_names_column[index] == STATES_NAMES[j]:
                    row.append(1)
                else:
                    row.append(0)
            index += 1
            bar_index += 1
        bar.finish()

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        dummy_y = to_categorical(encoded_Y)

        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(dummy_y, dtype=np.float)
        self.inputs_count = len(X[0])

        with open(self.path_name + '_property.pickle', 'wb') as f:
            pickle.dump((self.X, self.Y), f)

    def get_train_test_data(self, X=None, Y=None, test_size=0.3):
        """
        splitting data for training and testing
        :param X: samples
        :param Y: targets
        :return: X_train, X_test, Y_train, Y_test
        """
        if X is None or Y is None:
            if self.X.any() == None:
                self.handle_data()
            return train_test_split(self.X, self.Y, test_size=0.3, random_state=100, shuffle=True)
        return train_test_split(X, Y, test_size=test_size, random_state=100, shuffle=True)

    def load_model(self):
        self.model = load_model(self.path_name + '_model.h5')

    def save_model(self):
        self.model.save(self.path_name + '_model.h5')

    def build_model(self, input_layer_size):
        """
        creation of a neural network for multiclass classification
        :param input_layer_size:
        :return:
        """
        K.set_floatx('float16')
        config = tensorflow.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        K.set_session(tensorflow.Session(config=config))

        self.model = Sequential()
        self.model.add(Dense(input_layer_size, dtype=tensorflow.float64, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.initial_fit()

    def initial_fit(self):
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(self.X, self.Y)
        self.model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_test, Y_test))
        self.save_model()

    def evaluate(self, X, Y):
        if self.num_gpu != 1:
            self.model = tensorflow.keras.utils.multi_gpu_model(self.model, gpus=1536)
        y_pred = self.model.predict(X)
        print(y_pred[0][0])
        print("precision_score", precision_score(Y_test, y_pred.round(), average='macro'))
        print("recall_score", recall_score(Y_test, y_pred.round(), average='macro'))
        print("f1_score", f1_score(Y_test, y_pred.round(), average='macro'))
        print("classification_report\n", classification_report(Y_test, y_pred.round()))
        print("accuracy_score", accuracy_score(Y_test, y_pred.round()))

        report = accuracy_score(Y, y_pred.round())
        print(report)

    def predict(self, X):
        return self.model.predict(np.array(X, dtype=np.float16))


PATH = "data/all_multiclass_data.csv"

if __name__ == "__main__":

    print("device_lib \n ", device_lib.list_local_devices())

    network = Network(PATH)
    X_train, X_test, Y_train, Y_test = network.get_train_test_data()
    network.initial_fit()
    network.evaluate(X_test, Y_test)


