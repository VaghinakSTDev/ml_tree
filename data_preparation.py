import pandas as pd
import matplotlib.pyplot as plt
import progressbar

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
                 "DurationPolicy",
                 "PetId",
                 "TransactionDt",
                 "Churn"
                 ]


df = pd.read_excel('STdevChurnData.xlsx', usecols=INPUT_COLUMNS)

df = df[(df.Churn != 1)]
df = df.drop_duplicates()
df = df.drop(['Churn'], axis=1)
df['date'] = pd.to_datetime(df['TransactionDt'])
df.sort_values(by='date')
# df = df.set_index('PetId')
# df.drop(['Accounting Date'], axis=1)
df.loc[df['PolicyForm'] == "Unlimited", 'PolicyForm'] = 50000

df = pd.DataFrame(df).fillna(method='ffill')
duplicated_indexes = df.groupby('PetId').groups.keys()
print(df.shape)
df = df.drop_duplicates(['PetId', 'PolicyForm'], keep='last', inplace=False)

grouped_data = df.groupby(
        ["PetId"]
    ).filter(
        lambda x: len(x) > 1
    ).sort_values(
        by='date'
    ).drop_duplicates(
        ['PetId', 'PolicyForm'], keep='first', inplace=False
    )[['PetId', 'PolicyForm', 'date']].groupby(
        ["PetId"]
    ).filter(
        lambda x: len(x) > 1
    ).to_dict('r')

key_values = {}

print("DATA GROUPING")
bar = progressbar.ProgressBar(
    maxval=len(grouped_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
)
bar.start()
bar_index = 0
for data in grouped_data:
    key = data.pop('PetId')
    if key in key_values.keys():
        key_values[key].append(data)
    else:
        key_values[key] = [data]

    bar_index += 1
bar.finish()


data = []
replace_rows_ids = {}
delete_rows_ids = []


print("TARGET CALCULATIONS")
bar = progressbar.ProgressBar(
    maxval=len(grouped_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
)
bar.start()
bar_index = 0
for pet_id in key_values:
    first = min(key_values[pet_id], key=lambda x: (x['date']))
    last = max(key_values[pet_id], key=lambda x: (x['date']))
    delta = int(last['PolicyForm']) - int(first['PolicyForm'])

    delete_rows_ids.append((pet_id, last['date']))
    if delta > 0:
        replace_rows_ids[pet_id] = -1
        first['target'] = -1
    else:
        replace_rows_ids[pet_id] = 1
        first['target'] = 1
    data.append(first)
    bar_index += 1
bar.finish()

data_frame = df[~df[['PetId', 'date']].apply(tuple, axis=1).isin(delete_rows_ids)]
data_frame['target'] = [0 for i in range(data_frame.shape[0])]

print("MODERNIZING NEW DATA FRAME")
bar = progressbar.ProgressBar(
    maxval=len(grouped_data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
)
bar.start()
bar_index = 0
for index, row in data_frame.iterrows():
    if row['PetId'] in replace_rows_ids.keys():
        data_frame.at[index, 'target'] = replace_rows_ids[row['PetId']]

    bar_index += 1
bar.finish()

data_frame.to_csv('/data/all_multiclass_data.csv')
