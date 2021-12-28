import pandas as pd
df = pd.read_pickle("LSWMD.pkl")
df_type = df[(df['failureType']=='Center') | (df['failureType']=='Edge-Loc') | (df['failureType']=='Edge-Ring') | (df['failureType']=='Loc') | (df['failureType']=='Random') | (df['failureType']=='Scratch')]
df_none = df[~df.index.isin(df_type.index)]
df_train = df_type.sample(frac=0.8)
df_test = df_type[~df_type.index.isin(df_train.index)]

df_none.reset_index(inplace=True)
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)

print(df_none.count())
print('*')
print(df_train.count())
print('*')
print(df_train['failureType'].value_counts())
print('*')
print(df_test.count())
print('*')
print(df_test['failureType'].value_counts())
print('*')
print(df.count())
df_none.to_pickle('./none.pkl')
df_train.to_pickle('./train.pkl')
df_test.to_pickle('./test.pkl')
