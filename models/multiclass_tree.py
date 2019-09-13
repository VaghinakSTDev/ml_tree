import tensorflow
# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=config)

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, BatchNormalization, Flatten

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model, model_from_json
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import progressbar
import joblib

import numpy as np

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
                 "ReasonChurn",
                 # "DurationPolicy",
                 "Churn"
                 ]


class Network:
    def __init__(self, path=None):
        self.path = path
        self.inputs_count = None
        self.handle_data()
        self.build_model(self.inputs_count, self.outputs_count)
        # if path:
        #     path_name = path.split('.')[0]
        #     self.path_name = path_name
        #     try:
        #         self.load_model()
        #         with open(path_name + "mnm_property.pickle", "rb") as f:
        #             self.columns, self.bread_names, self.state_names = pickle.load(f)
        #     except Exception as e:
        #         try:
        #             with open(path_name + "mnm_property.pickle", "rb") as f:
        #                 self.columns, self.bread_names, self.state_names = pickle.load(f)
        #         except:
        #             self.handle_data()
        #         if not self.inputs_count:
        #             self.handle_data()
        #         self.build_model(self.inputs_count, self.outputs_count)
        # else:
        #     try:
        #         self.load_model()
        #         with open("data.pickle", "rb") as f:
        #             self.X, self.Y = pickle.load(f)
        #     except:
        #         raise Exception("specify file path")

    def balancing_data(seln, df):
        caceled_df = df.loc[df['Churn'] == 1]
        all_data_count = df.shape[0]
        caceled_data_count = caceled_df.shape[0]
        for i in range(int(all_data_count/caceled_data_count)):
            df = df.append(caceled_df, ignore_index = True)

        return df

    def normlize_df(self, df):
        from sklearn import preprocessing
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        return pd.DataFrame(x_scaled)

    def handle_data(self):
        balance_data = pd.read_excel(self.path, usecols=INPUT_COLUMNS)
        balance_data.loc[balance_data['PolicyForm'] == "Unlimited", 'PolicyForm'] = 20000

        balance_data = pd.DataFrame(balance_data).fillna(method='ffill')

        balance_data = balance_data.loc[balance_data['Churn'] == 1]
        print("Dataset Length: ", len(balance_data))
        print("Dataset Shape: ", balance_data.shape)

        # Printing the dataset obseravtions
        print("Dataset: ", balance_data.head())

        balance_data = self.balancing_data(balance_data)

        Y = balance_data['Churn'].values.tolist()

        bread_names_column = balance_data['BreedName'].tolist()
        state_names_column = balance_data['ControllingStateCd'].tolist()
        reason_column = balance_data['ReasonChurn'].tolist()
        balance_data = balance_data.drop(["BreedName", "Churn", "ControllingStateCd", "ReasonChurn"], axis=1)

        BREED_NAMES = set()
        STATES_NAMES = set()
        REASONS = set()

        [BREED_NAMES.add(breed_name) for breed_name in bread_names_column]
        [STATES_NAMES.add(state) for state in state_names_column]
        [REASONS.add(reason) for reason in reason_column]

        self.columns = balance_data.columns
        self.bread_names = list(BREED_NAMES)
        self.state_names = list(STATES_NAMES)
        self.reasons = list(REASONS)

        balance_data = self.normlize_df(balance_data)

        X = balance_data.values.tolist()
        del balance_data
        index = 0

        bar = progressbar.ProgressBar(
            maxval=len(X), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        )
        bar.start()
        bar_index = 0
        Y_vec = [[]]
        for i in range(0, len(X)):
            bar.update(bar_index + 1)
            for j in range(len(self.bread_names)):
                if bread_names_column[index] == self.bread_names[j]:
                    X[i].append(1)
                else:
                    X[i].append(0)
            for j in range(len(self.state_names)):
                if state_names_column[index] == self.state_names[j]:
                    X[i].append(1)
                else:
                    X[i].append(0)

            Y_vec.append([])
            if Y[i] == 1:
                for j in range(len(self.reasons)):
                    if reason_column[index] == self.reasons[j]:

                        Y_vec[i].append(1)
                    else:
                        Y_vec[i].append(0)
            else:
                for j in range(len(self.reasons)):
                    Y_vec[i].append(0)

            index += 1
            bar_index += 1

        bar.finish()

        Y_vec.pop(len(Y_vec) - 1)

        X = np.array(X, dtype=np.float64)
        Y = np.array(Y_vec, dtype=np.float64)

        # with open(self.path_name + '_mnm_property.pickle', 'wb') as f:
        #     pickle.dump((self.columns, self.bread_names, self.state_names, self.reasons), f)

        # with open(self.path_name + '_data.pickle', 'wb') as f:
        #     pickle.dump((X, Y,), f)

        self.inputs_count = len(X[0])
        self.outputs_count = len(Y_vec[0])
        return X, Y

    def get_train_test_data(self, X, Y):
        return train_test_split(X, Y, test_size=0.3, random_state=100, shuffle=True)

    def load_model(self):
        self.model = load_model(self.path_name + '_model.mnn')

    def save_model(self):
        self.model.save(self.path_name + '_model.mnn')

    def build_model(self, input_layer_size, output_layer_size):
        from xgboost import DMatrix

        # self.model = DMatrix()#max_depth=3, learning_rate=0.1)
        # self.initial_fit()

    def f(self):
        X, Y = self.handle_data()
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(X, Y)
        return X_test

    def initial_fit(self):
        X, Y = self.handle_data()
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(X, Y)
        # self.model = tensorflow.keras.utils.multi_gpu_model(self.model, gpus=1536)

        self.model.fit(X_train, Y_train, epochs=10, batch_size=40)
        # self.save_model()

    def evaluate(self):
        X, Y = self.handle_data()
        X_train, X_test, Y_train, Y_test = self.get_train_test_data(X, Y)
        #
        from xgboost import DMatrix
        import xgboost as xgb
        xg_train = DMatrix(X_train, label=Y_train)
        xg_test = DMatrix(X_test, label=Y_test)
        # self.model.fit(X_train, Y_train)

        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = 6

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 5
        bst = xgb.train(param, xg_train, num_round, watchlist)
        # get prediction
        pred = bst.predict(xg_test)
        error_rate = np.sum(pred != Y_test) / Y_test.shape[0]
        print('Test error using softmax = {}'.format(error_rate))

        # do the same thing again, but output probabilities
        param['objective'] = 'multi:softprob'
        bst = xgb.train(param, xg_train, num_round, watchlist)
        # Note: this convention has been changed since xgboost-unity
        # get prediction, this is in 1D array, need reshape to (ndata, nclass)
        pred_prob = bst.predict(xg_test).reshape(Y_test.shape[0], 6)
        pred_label = np.argmax(pred_prob, axis=1)
        error_rate = np.sum(pred_label != Y_test) / Y_test.shape[0]
        print('Test error using softprob = {}'.format(error_rate))

        y_pred = xg_train.predict(X_test)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        print("precision_score", precision_score(Y_test, y_pred.round()))
        print("recall_score", recall_score(Y_test, y_pred.round()))
        print("f1_score", f1_score(Y_test, y_pred.round()))
        print("accuracy_score", accuracy_score(Y_test, y_pred.round()))

        # false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)
        #
        # roc_auc = auc(false_positive_rate, true_positive_rate)
        # print("roc_auc", roc_auc)


    def predict(self, X):
        return self.model.predict(np.array(X, dtype=np.float16), verbose=1)

def auc(y_true, y_pred):
    auc = tensorflow.metrics.auc(y_true, y_pred)[1]
    tensorflow.keras.backend.get_session().run(tensorflow.local_variables_initializer())
    return auc

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


PATH = "50k.xlsx"


if __name__ == "__main__":
    config = tensorflow.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Create a session with the above options specified.
    K.set_session(tensorflow.Session(config=config))

    from sklearn.datasets import load_iris


    # network.load_data_from_xlsx("data/50k.xlsx")
    # train_data_generator = network.get_train_data()


    NUM_GPU = 1

    network = Network(PATH)

    # X, Y = network.handle_data()
    # X_train, X_test, Y_train, Y_test = network.get_train_test_data(X, Y)
    # y_pred = network.model.predict(X_test)
    # print(11111, len(X_test[0]))
    network.evaluate()


