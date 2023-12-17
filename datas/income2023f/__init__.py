import pandas as pd
import numpy as np

import os
datadir, _ = os.path.split(__file__)

print('Loading Data')
test = pd.read_csv(os.path.join(datadir, 'test_final.csv'))
train = pd.read_csv(os.path.join(datadir, 'train_final.csv'))

# Load in the saved partitions of the training set to partition it into a validation and training set
with open(os.path.join(datadir, 'partition_training.csv'), 'r') as f:
    train_training_partition = np.array(f.read().split('\n')).astype(np.int32)
with open(os.path.join(datadir, 'partition_validation.csv'), 'r') as f:
    train_valdiation_partition = np.array(f.read().split('\n')).astype(np.int32)

print('Finished Loading Data')

def make_one_hot(x, one_hot_order):
    ret = np.zeros(shape=len(one_hot_order), dtype=np.float64)
    for i, k in enumerate(one_hot_order):
        if x==k:
            ret[i] = 1
            return ret
    raise Exception(f'Is not a known one-hot category: {x}')
    
workclass_one_hot_order = [
    '?',
    'Federal-gov',
    'Local-gov',
    'Never-worked',
    'Private',
    'Self-emp-inc',
    'Self-emp-not-inc',
    'State-gov',
    'Without-pay'
]

education_one_hot_order = [
    '10th',
    '11th',
    '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
    'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad',
    'Masters', 'Preschool', 'Prof-school', 'Some-college'
]

marital_status_one_hot_order = 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse'.split(', ')

occupation_one_hot_order = 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, ?'.split(', ')

relationship_one_hot_order = 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried'.split(', ')

race_one_hot_order = 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black'.split(', ')

sex_one_hot_order = 'Female, Male'.split(', ')

native_country_one_hot_order = 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands, ?'.split(', ')

def preprocess(df):
    df = df.copy()
    df.workclass = pd.Series([
            make_one_hot(x, workclass_one_hot_order)
            for x in df.workclass
        ])
    df.education = pd.Series([
            make_one_hot(x, education_one_hot_order)
            for x in df.education
        ])
    df['marital.status'] = pd.Series([
            make_one_hot(x, marital_status_one_hot_order)
            for x in df['marital.status']
        ])
    df['occupation'] = pd.Series([
            make_one_hot(x, occupation_one_hot_order)
            for x in df['occupation']
        ])
    df['relationship'] = pd.Series([
            make_one_hot(x, relationship_one_hot_order)
            for x in df['relationship']
        ])
    df['race'] = pd.Series([
            make_one_hot(x, race_one_hot_order)
            for x in df['race']
        ])
    df['sex'] = pd.Series([
            make_one_hot(x, sex_one_hot_order)
            for x in df['sex']
        ])
    df['native.country'] = pd.Series([
            make_one_hot(x, native_country_one_hot_order)
            for x in df['native.country']
        ])
    return df

train_preprocessed = preprocess(train)
test_preprocessed = preprocess(test)

# me = pd.DataFrame(
#     {'age': 21,
#      'workclass':'Without-pay',
#      'fnlwgt':,
#      'education':'Some-college',
#      'education.num':10
#      'marital.status':'Never-married',
#      'occupation':'Tech-support',
#      'relationship':'Unmarried',
#      'race':'White',
#      'sex':'Male',
#      'capital.gain':0,
#      'capital.loss':30000,
#      'hours.per.week':80,
#      'native.country': 'United-States'
#     }
# )
