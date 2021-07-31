#%%
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model

#%%
data = pd.read_csv('finaldata.csv', sep='|', index_col=0, dtype=dict(gym=np.float64,
    grill=np.float64,
    cinema=np.float64,
    gourmetarea=np.float64,
    park=np.float64,
    grass=np.float64,
    garden=np.float64,
    pool=np.float64,
    cooper=np.float64,
    playground=np.float64,
    squash=np.float64,
    tennis=np.float64,
    multisports=np.float64,
    privategarden=np.float64,
    partyroom=np.float64,
    playroom=np.float64,
    heating=np.float64,
    airconditioning=np.float64,
    internet=np.float64,
    storage=np.float64,
    elevator=np.float64,
    garage=np.float64,
    generator=np.float64,
    fireplace=np.float64,
    laundryroom=np.float64,
    massageroom=np.float64,
    furnished=np.float64,
    reception=np.float64,
    sauna=np.float64,
    spa=np.float64,
    cabletv=np.float64,
    securitytv=np.float64,
    gatedcondo=np.float64,
    interfone=np.float64,
    security24h=np.float64,
    alarm=np.float64,
    guard=np.float64,
    privatelaundryroom=np.float64,
    kitchen=np.float64,
    office=np.float64,
    balcony=np.float64,
    gourmetbalcony=np.float64,
    pinheiros=np.float64,
    vilamadalena=np.float64,
    cluster_1=np.float64,
    cluster_2=np.float64,
    cluster_3=np.float64,
    cluster_4=np.float64,
    cluster_5=np.float64,
    cluster_6=np.float64,
    cluster_7=np.float64,
    cluster_8=np.float64,
    cluster_9=np.float64,
    cluster_10=np.float64,
    cluster_11=np.float64,
    cluster_12=np.float64,
    cluster_13=np.float64,
    cluster_14=np.float64,
    cluster_15=np.float64,
    cluster_16=np.float64,
    cluster_17=np.float64,
    cluster_18=np.float64,
    cluster_19=np.float64))

area = ['area']

bare = ['area', 'rooms', 'baths', 'parking']

fee = ['mgmtfee']

neighborhood = ['pinheiros', 'vilamadalena']

cluster = [f'cluster_{i}' for i in range(1,20)]

features = ['gym', 'grill', 'cinema', 'gourmetarea', 'park', 'grass',
    'garden', 'pool', 'cooper', 'playground', 'squash', 'tennis',
    'multisports', 'privategarden', 'partyroom', 'playroom', 'heating',
    'airconditioning', 'internet', 'storage', 'elevator', 'garage',
    'generator', 'fireplace', 'laundryroom', 'massageroom', 'furnished',
    'reception', 'sauna', 'spa', 'cabletv', 'securitytv', 'gatedcondo',
    'interfone', 'security24h', 'alarm', 'guard', 'privatelaundryroom',
    'kitchen', 'office', 'balcony', 'gourmetbalcony']

pca = ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'pc_5', 'pc_6']

pubtransp = ['mindist2pubtransport']

pubtranspsplit = ['mindist2train', 'mindist2metro']






# %%
#ANN

ANNmodels = dict(ANN_bare_neighborhood=bare+neighborhood,
    ANN_bare_neighborhood_fee=bare+neighborhood+fee,
    ANN_bare_neighborhood_pubtransp=bare+neighborhood+pubtransp,
    ANN_bare_neighborhood_pubtranspsplit=bare+neighborhood+pubtranspsplit,
    ANN_bare_neighborhood_pca=bare+neighborhood+pca,
    ANN_bare_neighborhood_features=bare+neighborhood+features,
    ANN_bare_neighborhood_features_pubtranspsplit=bare+neighborhood+features+pubtranspsplit,
    ANN_bare_cluster=bare+cluster,
    ANN_bare_cluster_fee=bare+cluster+fee,
    ANN_bare_cluster_pubtransp=bare+cluster+pubtransp,
    ANN_bare_cluster_pubtranspsplit=bare+cluster+pubtranspsplit,
    ANN_bare_cluster_pca=bare+cluster+pca,
    ANN_bare_cluster_features=bare+cluster+features,
    ANN_bare_cluster_features_pubtranspsplit=bare+cluster+features+pubtranspsplit,
    ANN_bare_cluster_features_fee=bare+cluster+features+fee,
    ANN_bare_cluster_features_pubtranspsplit_fee=bare+cluster+features+pubtranspsplit+fee)

for annmodelname, annmodel in ANNmodels.items():
    x = data[annmodel]
    y = data['price']

    x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=0.4,random_state=52)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=52)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train= scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)


    model = Sequential()

    model.add(Dense((x_train.shape[1]+1),  activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(round((x_train.shape[1]+1)/2,0), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(round((x_train.shape[1]+1)/4,0), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

    log = CSVLogger(f'{annmodelname}_log', separator='|')

    modelfit = model.fit(x=x_train, 
                y=y_train, 
                epochs=20000,
                verbose=0,
                validation_data=(x_valid, y_valid),
                callbacks=[early_stop, log])

    y_pred = model.predict(x_test)
    print(f'Model: {annmodelname}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R2: {r2_score(y_test, y_pred)}')
    print(f'Explained Variance: {explained_variance_score(y_test, y_pred)}')
    model.save(f'{annmodelname}.h5') 



#%%

model = load_model('ANN_bare_cluster_features.h5')


x = data[bare+cluster+features]
y = data['price']

x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=0.4,random_state=52)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=52)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train= scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)




y_pred = model.predict(x_test)

losses = pd.read_csv('ANN_bare_cluster_features_log', sep='|')
losses.plot()


fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test,y=y_pred.ravel(),
                    mode='markers', name='Predictions'))
fig.add_trace(go.Scatter(x=y_test,y=y_test,
                    mode='lines', name='Perfect fit'))

fig.show()

data['link'] = [f'https://www.vivareal.com.br{link}' for link in data['link']]

x = scaler.transform(x)
data['pred'] = model.predict(x)

data.to_csv('datadashboard.csv', sep='|')





# %%
