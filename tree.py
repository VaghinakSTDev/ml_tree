import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import timeit
import progressbar
import joblib
from xgboost import plot_tree
import matplotlib.pyplot as plt
import numpy as np

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
                 "Churn"
                 ]


class Tree:
    def __init__(self, path=None):
        if path:
            path_name = path.split('.')[0]
            self.path_name = path_name
            try:
                self.model = joblib.load(path_name + "_model.dat")
                with open(path_name + "_property.pickle", "rb") as f:
                    self.columns, self.bread_names, self.state_names = pickle.load(f)
            except:
                self.model = XGBClassifier(max_depth=5, learning_rate=0.1)
                self.path = path
                self.handle_data()
                X_train, X_test, y_train, y_test = self.get_train_test_data()
                self.fit(X_train, X_test, y_train, y_test)

        else:
            try:
                self.model = joblib.load("model.dat")
                with open("data.pickle", "rb") as f:
                    self.X, self.Y = pickle.load(f)
            except:
                raise Exception("specify file path")
    def balancing_data(seln, df):
        caceled_df = df.loc[df['Churn'] == 1]
        all_data_count = df.shape[0]
        caceled_data_count = caceled_df.shape[0]
        for i in range(int(all_data_count/caceled_data_count)):
            df = df.append(caceled_df, ignore_index = True)
        return df
    def handle_data(self):
        balance_data = pd.read_excel(self.path, usecols=INPUT_COLUMNS)
        balance_data.loc[balance_data['PolicyForm'] == "Unlimited", 'PolicyForm'] = 20000

        balance_data = pd.DataFrame(balance_data).fillna(method='ffill')

        balance_data.loc[balance_data['PolicyForm'] == "Unlimited", 'PolicyForm'] = 20000
        churned_data = balance_data.loc[lambda balance_data: balance_data['Churn'] == 1]

        balance_data = self.balancing_data(balance_data)


        # delta_w = int(balance_data.shape[0] / churned_data.shape[0])
        # for _ in range(delta_w):
        #     balance_data = balance_data.append(churned_data)

        print("Dataset Length: ", len(balance_data))
        print("Dataset Shape: ", balance_data.shape)

        # Printing the dataset obseravtions
        print("Dataset: ", balance_data.head())

        Y = balance_data['Churn'].values

        bread_names_column = balance_data['BreedName'].tolist()
        state_names_column = balance_data['ControllingStateCd'].tolist()

        balance_data = balance_data.drop(["BreedName", "Churn", "ControllingStateCd"], axis=1)

        BREED_NAMES = set()
        STATES_NAMES = set()

        [BREED_NAMES.add(breed_name) for breed_name in bread_names_column]
        [STATES_NAMES.add(state) for state in state_names_column]

        BREED_NAMES = list(BREED_NAMES)
        STATES_NAMES = list(STATES_NAMES)


        self.columns = balance_data.columns
        self.bread_names = BREED_NAMES
        self.state_names = STATES_NAMES

        X = balance_data.values.tolist()

        index = 0

        bar = progressbar.ProgressBar(
            maxval=len(X), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        )
        bar.start()
        bar_index = 0

        for row in X:
            bar.update(bar_index + 1)
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
        self.X = np.array(X, dtype=np.uint8)
        self.Y = Y

        with open(self.path_name + '_property.pickle', 'wb') as f:
            pickle.dump((self.columns, self.bread_names, self.state_names), f)

        with open(self.path_name + '_data.pickle', 'wb') as f:
            pickle.dump((X, Y,), f)

    def get_train_test_data(self):
        return train_test_split(self.X, self.Y, test_size=0.3, random_state=100, shuffle=True)

    def get_train_test_data_generator(self, chunk_size=8):
        self.results = []
        len_of_data = len(self.X)
        len_step = int(len_of_data/chunk_size)
        for i in range(0, len_of_data, len_step):
            yield train_test_split(self.X[i: i+len_step], self.Y[i: i+len_step])

    def fit(self, X_train, X_test, y_train, y_test):
        results = {}
        def fn():
            self.model.fit(np.array(X_train), y_train)

        t = timeit.timeit(fn, number=1)

        y_pred = self.model.predict(np.array(X_test))
        y_pred = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, y_pred)

        results['XGBoost'] = {'training_time': t, 'accuracy': accuracy}

        joblib.dump(self.model, self.path_name + "_model.dat")
        return results

    def plot(self):
        fig, ax = plt.subplots(figsize=(30, 30))
        plot_tree(self.model, num_trees=2, ax=ax)
        plt.savefig("xgb/" + self.path.split('.')[0] + "_temp.png")

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

PATH = "STdevChurnData.xlsx"

if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, f1_score

    tree = Tree(PATH)
    CHUNK_SIZE = 30
    # X_train, X_test, y_train, y_test = tree.get_train_test_data()
    generator = tree.get_train_test_data_generator(CHUNK_SIZE)

    bar = progressbar.ProgressBar(
        maxval=CHUNK_SIZE, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    )
    for X_train, X_test, y_train, y_test in generator:
        tree.results.append(tree.fit(X_train, X_test, y_train, y_test))
        y_pred = tree.model.predict(X_test)

        print("precision_score", precision_score(y_test, y_pred.round()))
        print("recall_score", recall_score(y_test, y_pred.round()))
        print("f1_score", f1_score(y_test, y_pred.round()))
        print("accuracy_score", accuracy_score(y_test, y_pred.round()))

        # bar_index += 1
    # bar.finish()
    tree.plot()
    print(tree.results)
