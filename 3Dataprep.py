#%%
import pandas as pd
data = pd.read_csv('data.csv', sep='|', index_col=0)
data = data[data['point'].notnull()]
data = data.drop(columns=[
    'unstructuredaddress', 
    'geoaddress', 
    'location', 
    'point', 
    'altitude'])
data = data.reset_index(drop=True)
#%%
data.info()
data.head()

#%%

# I now have 11 columns and 13k rows of mostly usable data.
# However, most columns are 'object' type and have to be 
# changed into other formats to be worked with.
# It has often to do with strings in numeric columns 
# or just from Pandas reading the csv, so we will deal with these first.

#%%
data = data[data['price'].str[0:11] != 'A partir de'] #removing cases with starting prices instead of actual prices
data = data[~data['price'].str.contains(pat = r'/M')] #removing cases with monthly installements instead of full prices
data['price'] = data['price'].str[3:] #removing the Real currency symbol (R$)
data['mgmtfee'] = data['mgmtfee'].str[3:] #removing the Real currency symbol (R$)
data['price'] = data['price'].str.replace(".","") #fixing number format
data['mgmtfee'] = data['mgmtfee'].str.replace(".","") #fixing number format
data['area'] = data['area'].str.replace("--","") #removing strings from numeric column
data['rooms'] = data['rooms'].str.replace("--","") #removing strings from numeric column
data['baths'] = data['baths'].str.replace("--","") #removing strings from numeric column
data['parking'] = data['parking'].str.replace("--","") #removing strings from numeric column
data['neighborhood'] = data['neighborhood'].str.replace("ã","a") #removing strings from numeric column
data['area'] = data.area.str.split("-", expand=True)[0] #when data is in a range, picking the smaller value, as the price refers to the smaller
data['rooms'] = data.rooms.str.split("-", expand=True)[0] #when data is in a range, picking the smaller value, as the price refers to the smaller
data['baths'] = data.baths.str.split("-", expand=True)[0] #when data is in a range, picking the smaller value, as the price refers to the smaller
data['parking'] = data.parking.str.split("-", expand=True)[0] #when data is in a range, picking the smaller value, as the price refers to the smaller
data['neighborhood'] = data['neighborhood'].str[:-11] #removing city data, as it's the same for the whole dataset

#fixing the column types
data['features'] = [pd.eval(i) for i in data['features']] #changing from string to list
data['price'] = pd.to_numeric(data['price'], downcast='float')  #Float is used instead of integer to avoid Tensorflow issues later
data['mgmtfee'] = pd.to_numeric(data['mgmtfee'], downcast='float')
data['area'] = pd.to_numeric(data['area'], downcast='float')
data['rooms'] = pd.to_numeric(data['rooms'], downcast='float')
data['baths'] = pd.to_numeric(data['baths'], downcast='float')
data['parking'] = pd.to_numeric(data['parking'], downcast='float')

#%%
data.info()
data.head()
data.describe().transpose()
#%%
import seaborn as sns
#%%
sns.boxplot(y='area', data=data)
#2 extreme area outliers visible, removing those from the data set
data = data[data['area'] < 2000]
sns.boxplot(y='area', data=data)

#%%
sns.boxplot(y='rooms', data=data)
#removing extreme rooms outliers
data = data[data['rooms'] < 15]
sns.boxplot(y='rooms', data=data)

#%%
sns.boxplot(y='parking', data=data)
data[data['parking'] > 5].sort_values('parking')
#removing extreme parking outliers where the data is not compatible with area/rooms
data = data[data['parking'] < 7]
sns.boxplot(y='parking', data=data)

#%%
sns.boxplot(y='baths', data=data)
data[data['baths'] > 6].sort_values('area')
#removing extreme baths outliers where the data is not compatible with area/rooms
data = data[data['baths'] < 9]
sns.boxplot(y='baths', data=data)



#%%
sns.scatterplot(x='latitude', y='longitude', data=data)
#removing lat/long outliers
data = data[data['latitude'] < -23.4]
data = data[data['longitude'] > -47.4]

#%%
#removing apartments announced more than once with the same details
data = data.drop_duplicates(subset=['area', 'rooms', 'baths', 'parking', 'price', 'mgmtfee', 'street', 'neighborhood', 'latitude', 'longitude'])

#%%
sns.histplot(data['price'], kde=True)
#%%
sns.histplot(x='price', kde=True, hue='neighborhood', data=data)
#%%
sns.boxplot(x='neighborhood',y='price', data=data)

#%%
sns.scatterplot(x='area',y='price',data=data)

#%%
sns.scatterplot(x='area',y='price', hue='neighborhood', data=data)
#%%
#adding transparency and interactivity
import plotly.express as px
px.scatter(data, x='area', y='price', color='neighborhood', opacity=0.3) 
#%%
# sns.pairplot(data)

#%%
#correlation heatmat
# data.corr()
# px.imshow(data.corr())
#%%
# suggests strongest correlation between price and area
# fair correlation with other features 
# such as number of rooms, baths and parking spaces
# poor correlation with management fees and 
# with latitude and longitude considered isolately

#%%
px.scatter_mapbox(data, lat='latitude', lon='longitude',
                     hover_name='price', color='price',
                     range_color=[0,4000000],
                     color_continuous_scale = 'plotly3',                      
                     mapbox_style="open-street-map")
#%%

# However, when plotted on the map compared to price, 
# we can see price differences with Butanta apartments being generally cheaper
# and more expensive apartments concentrating in certain regions


#%%
sns.heatmap(data.isnull(), cbar=False)
#%%
data.describe().transpose()
#%%
 
# 84% of the rows have mgmtfee, while the others do not
# with 16% of the values missing, I consider useful to fill values
# instead of dropping the column at this point
# although I might eventually drop it if it doesn't add to the predictive power
# of the models, as suggested by the poor correlation


#%%
data['mgmtfee'] = data['mgmtfee'].fillna(data.groupby('area')['mgmtfee'].transform('mean'))
data['mgmtfee'] = data['mgmtfee'].fillna(data.groupby('rooms')['mgmtfee'].transform('mean'))

#%%

# Complete EDA and data cleaning


#%%
#Data Preparation and feature design

data = data.reset_index(drop=True)

#converting list of features to dummy variables
listoffeatures = dict(
    gym='Academia',
    grill='Churrasqueira',
    cinema='Cinema',
    gourmetarea='Espaço gourmet',
    park='Espaço verde / Parque',
    grass='Gramado',
    garden='Jardim',
    pool='Piscina',
    cooper='Pista de cooper',
    playground='Playground',
    squash='Quadra de squash',
    tennis='Quadra de tênis',
    multisports='Quadra poliesportiva',
    privategarden='Quintal',
    partyroom='Salão de festas',
    playroom='Salão de jogos',
    heating='Aquecimento',
    airconditioning='Ar-condicionado',
    internet='Conexão à internet',
    storage='Depósito',
    elevator='Elevador',
    garage='Garagem',
    generator='Gerador elétrico',
    fireplace='Lareira',
    laundryroom='Lavanderia',
    massageroom='Sala de massagem',
    furnished='Mobiliado',
    reception='Recepção',
    sauna='Sauna',
    spa='Spa',
    cabletv='TV a cabo',
    securitytv='Circuito de segurança',
    gatedcondo='Condomínio fechado',
    interfone='Interfone',
    security24h='Segurança 24h',
    alarm='Sistema de alarme',
    guard='Vigia',
    privatelaundryroom='Área de serviço',
    kitchen='Cozinha',
    office='Escritório',
    balcony='Varanda',
    gourmetbalcony='Varanda gourmet') #all features listed in their website as filters

for key, value in listoffeatures.items():
    data[key] = [1 if value in list else 0 for list in data['features']]

#%%
for key, value in listoffeatures.items():
        print (f'{key}: {data[key].sum()}') #check if I could drop any empty column, but none exists



#%%
#lots of features and very sparse, trying to reduce dimensions using PCA

features = data.loc[:, listoffeatures.keys()].values

from sklearn.decomposition import PCA
pca = PCA()

principalComponents = pca.fit_transform(features)
PCAdf = pd.DataFrame(data = principalComponents, columns=[f'pc_{i+1}' for i in range(42)])

pca.explained_variance_ratio_
#%%
# While there's no clear inflection point, the data seems to 
# suggest that between 3 and 7 principal components should be ideal
# We opted for 6 to account for 50% of the variance of the 42 variables
PCAdf = PCAdf.loc[:,['pc_1','pc_2','pc_3','pc_4','pc_5','pc_6']]
data = pd.concat([data,PCAdf], axis=1)

#%%
#feature engineering - distance to main public transport hubs

import geopy.distance
listoftrainstations = dict(
    ceasa = (-23.536276722969422, -46.74255912888407),
    villalobos = (-23.544002240021296, -46.73289247573633),
    cidadeuniversitaria = (-23.555137010449208, -46.71207010248146),
    pinheiros = (-23.566925739208248, -46.70196512722542),
    hebraica = (-23.573094356339872, -46.69821461686407),
    cidadejardim = (-23.584583635243845, -46.69080322571601),
    vilaolimpia = (-23.59290895852731, -46.6921173258078))

listofmetrostations = dict(
    morumbi = (-23.586585571524218, -46.72322765976714),
    butanta = (-23.571250978130525, -46.708033173442125),
    farialima = (-23.566675619779254, -46.69326123673528),
    fradique = (-23.56558405749111, -46.68416722297331),
    oscar = (-23.559943618384423, -46.67207374432367),
    paulista = (-23.555180728176694, -46.66205562935285),
    consolacao = (-23.557207140463152, -46.65979464204346),
    clinicas = (-23.553729999183947, -46.67059772137633),
    sumare = (-23.550085856217194, -46.67797173483965),
    vilamada = (-23.546052145268476, -46.690747286522644))



listoftraincolumns = []
listofmetrocolumns = []
for key, value in listoftrainstations.items():
    data[f'km2{key}train'] = [geopy.distance.distance((lat,long), {value}) for (lat,long) in zip(data['latitude'],data['longitude'])]
    data[f'km2{key}train'] = data[f'km2{key}train'].astype(str).str[:-3].apply(pd.to_numeric)
    listoftraincolumns.append(f'km2{key}train')

for key, value in listofmetrostations.items():
    data[f'km2{key}metro'] = [geopy.distance.distance((lat,long), {value}) for (lat,long) in zip(data['latitude'],data['longitude'])]
    data[f'km2{key}metro'] = data[f'km2{key}metro'].astype(str).str[:-3].apply(pd.to_numeric)
    listofmetrocolumns.append(f'km2{key}metro')



#%%

#%%
data['mindist2train'] = [min(a, b, c, d, e, f, g) for a, b, c, d, e, f, g in zip(data['km2ceasatrain'],data['km2villalobostrain'],data['km2cidadeuniversitariatrain'],data['km2pinheirostrain'],data['km2hebraicatrain'],data['km2cidadejardimtrain'],data['km2vilaolimpiatrain'])]
data['mindist2metro'] = [min(a, b, c, d, e, f, g, h, i, j) for a, b, c, d, e, f, g, h, i, j in zip(data['km2morumbimetro'],data['km2butantametro'],data['km2farialimametro'],data['km2fradiquemetro'],data['km2oscarmetro'],data['km2paulistametro'],data['km2consolacaometro'],data['km2clinicasmetro'],data['km2sumaremetro'],data['km2vilamadametro'])]
data['mindist2pubtransport'] = [min(a, b) for a, b in zip(data['mindist2train'],data['mindist2metro'])]



#%%
from sklearn.cluster import KMeans

data['cluster'] = KMeans(n_clusters=20).fit_predict(data.loc[:,['latitude','longitude']])


# %%
data.cluster = data.cluster.astype(str)
px.scatter_mapbox(data, lat='latitude', lon='longitude',
                     hover_name='price', color='cluster', 
                     mapbox_style="open-street-map")
# %%
data['miniregion'] = data['cluster']
data['region'] = data['neighborhood']
data = pd.get_dummies(data, columns=['neighborhood'], drop_first=True)
data = pd.get_dummies(data, columns=['cluster'], drop_first=True)

data = data.rename(columns={'neighborhood_Pinheiros': 'pinheiros', 'neighborhood_Vila Madalena': 'vilamadalena'})

data = data.drop(columns=[
    'street', 
    'features'])

data.to_csv('finaldata.csv', sep='|')



# %%
