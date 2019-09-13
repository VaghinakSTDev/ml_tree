import pandas as pd
import numpy as np
import pickle
import progressbar
import logging

from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


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
                with open(path_name + "_rf_property.pickle", "rb") as f:
                    self.X, self.Y = pickle.load(f)
            except FileNotFoundError:
                print('Creating a new model')
                self.handle_data()
                self.build_model(self.inputs_count)
        else:
            logging.error("Specify file path")

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

        X = self.balance_data.values.tolist()

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

        with open(self.path_name + '_rf_property.pickle', 'wb') as f:
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
        self.model = load(self.path_name + '_rf_model.sav.sav')

    def save_model(self):
        dump(self.model, open(self.path_name + '_rf_model.sav', 'wb'))

    def build_model(self, input_layer_size):
        """
        creation of a neural network for multiclass classification
        :param input_layer_size:
        :return:
        """
        self.model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=24)
        self.initial_fit()

    def initial_fit(self):
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(self.X, self.Y)
        print("Train")
        self.model.fit(X_train, Y_train)
        self.save_model()
        self.evaluate(X_test, Y_test)

    def evaluate(self, X, Y):
        y_pred = self.model.predict(X)
        print("precision_score", precision_score(Y, y_pred.round(), average='macro'))
        print("recall_score", recall_score(Y, y_pred.round(), average='macro'))
        print("f1_score", f1_score(Y, y_pred.round(), average='macro'))
        print("classification_report\n", classification_report(Y, y_pred.round()))
        print("accuracy_score", accuracy_score(Y, y_pred.round()))

    def predict(self, X):
        return self.model.predict(np.array(X, dtype=np.float16))


PATH = "data/all_multiclass_data.csv"

if __name__ == "__main__":


    network = Network(PATH)
    X_train, X_test, Y_train, Y_test = network.get_train_test_data()
    # network.initial_fit()
    # network.evaluate(X_test, Y_test)


