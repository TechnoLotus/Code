#%%
import pandas as pd
scrapeddata = pd.read_csv('scrapeddatat1.csv', sep='|', index_col=0)
for i in range (2,880):
    try:
        temp = pd.read_csv(f'scrapeddatat{i}.csv', sep='|', index_col=0)
        scrapeddata = scrapeddata.append(temp, ignore_index = True)
    except:
        pass
scrapeddata.to_csv('scrapeddata.csv', sep='|')



#%%
scrapeddata = scrapeddata.drop_duplicates(subset=['link'], ignore_index=True)
scrapeddata[['street']] = scrapeddata.address.str.split(" - ", expand=True)[0]
scrapeddata[['neighborhood']] = scrapeddata.address.str.split(" - ", expand=True)[1]
scrapeddata[['state']] = scrapeddata.address.str.split(" - ", expand=True)[2]
scrapeddata[['unstructuredaddress']] = scrapeddata.address.str.split(" - ", expand=True)[3]
scrapeddata = scrapeddata[scrapeddata['unstructuredaddress'].isnull()] #just 25 rows of unstructured address, not worth coding to save them
scrapeddata = scrapeddata[scrapeddata.state.notnull()] #removing incomplete addresses 
scrapeddata = scrapeddata[scrapeddata.price != 'Sob Consulta'] #removing ads without price
scrapeddata = scrapeddata[scrapeddata['neighborhood'].isin(['Pinheiros, São Paulo', 'Alto de Pinheiros,São Paulo','Vila Madalena, São Paulo', 'Butantã, São Paulo'])] #removing additional neighborhoods from sponsored ads
scrapeddata = scrapeddata.drop(columns=['address', 'state'])
scrapeddata['geoaddress'] = scrapeddata['street']+", "+scrapeddata['neighborhood']+", Brasil"


#%%
# Instead of further scrapping the VivaReal website going through each of the recorded
# links, what would be a very long scraping job for the number of rows when considering 
# the pause between requests to the website to avoid being blocked,
# I used an API to import latitude and longitude data


from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# A number of ads were in the same building.
# The terms of the API required that each address 
# be queried only once and with a rating of 1 per second, thus the 
# below code.

uniqueaddress = pd.DataFrame(scrapeddata['geoaddress'].unique(), columns=['geoaddress'])

geolocator = Nominatim(user_agent='NCI')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

uniqueaddress['location'] = uniqueaddress['geoaddress'].apply(geocode)
uniqueaddress['point'] = uniqueaddress['location'].apply(lambda loc: tuple(loc.point) if loc else None)
uniqueaddress.to_csv('geo.csv', sep='|') #just as a failsafe


#%%

scrapeddata['point'] = scrapeddata['geoaddress'].map(uniqueaddress.set_index('geoaddress')['point'])
scrapeddata[['latitude', 'longitude', 'altitude']] = pd.DataFrame(scrapeddata['point'].tolist(), index=scrapeddata.index)



scrapeddata.to_csv('data.csv', sep='|') #full data including the latitude and longitude and already mostly clean





# %%
