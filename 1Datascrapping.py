#%%
from bs4 import BeautifulSoup
import pandas as pd 
from time import sleep
from random import randint
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
# %%
driver = webdriver.Chrome(ChromeDriverManager().install())

driver.implicitly_wait(30) #pause for pages to load, if it loads faster it will go on

#data to be scrapped
address = []
area = []
rooms = []
baths = []
parking = []
features = []
price = []
mgmtfee = []
link = []

#%%


# Each seach produces a limited number of results, 
# so instead of searching for all neighborhoods at once,
# I split into 4 searches and will scrape the results for all of them


number = 0 #counter for scrapped pages
pinheiros = ['https://www.vivareal.com.br/venda/sp/sao-paulo/zona-oeste/pinheiros/apartamento_residencial/']
alto = ['https://www.vivareal.com.br/venda/sp/sao-paulo/zona-oeste/alto-de-pinheiros/apartamento_residencial/'] #the data from this search was not properly scraped
madalena = ['https://www.vivareal.com.br/venda/sp/sao-paulo/zona-oeste/vila-madalena/apartamento_residencial/']
butanta = ['https://www.vivareal.com.br/venda/sp/sao-paulo/zona-oeste/butanta/apartamento_residencial/']
urls = pinheiros+alto+madalena+butanta


#%%

# All scrapped items are stored twice. 
# One is 'name of the item' and the other 'name of the item'+t. 
# The reason being that all items are converted to a pandas dataframe.
# If the lists don't have the same length, the code breaks.
# The t items are recorded once every page is scraped and saved to a file, acting as a contingency.
# The other items are only recorded and saved to a file after the conclusion



for url in urls:
    driver.get(url)

    while True:
        number += 1
        addresst = []
        areat = []
        roomst = []
        bathst = []
        parkingt = []
        featurest = []
        pricet = []
        mgmtfeet = []
        linkt = []
    
        soup = BeautifulSoup(driver.page_source,'lxml')
        featdiv = soup.find_all('div', class_='property-card__content')
        pricediv = soup.find_all('section', class_='property-card__values')
        linkdiv = soup.find_all('div', class_='property-card__main-info')


        for container in featdiv:
            address.append(container.find('span', class_='property-card__address').get_text().strip())
            area.append(container.find('span', class_='property-card__detail-value js-property-card-value property-card__detail-area js-property-card-detail-area').get_text().strip())
            rooms.append(container.find(class_='property-card__detail-item property-card__detail-room js-property-detail-rooms').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            baths.append(container.find(class_='property-card__detail-item property-card__detail-bathroom js-property-detail-bathroom').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            parking.append(container.find(class_='property-card__detail-item property-card__detail-garage js-property-detail-garages').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            features.append([item.text for item in container.find_all('li', class_='amenities__item')])
            addresst.append(container.find('span', class_='property-card__address').get_text().strip())
            areat.append(container.find('span', class_='property-card__detail-value js-property-card-value property-card__detail-area js-property-card-detail-area').get_text().strip())
            roomst.append(container.find(class_='property-card__detail-item property-card__detail-room js-property-detail-rooms').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            bathst.append(container.find(class_='property-card__detail-item property-card__detail-bathroom js-property-detail-bathroom').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            parkingt.append(container.find(class_='property-card__detail-item property-card__detail-garage js-property-detail-garages').find('span', class_='property-card__detail-value js-property-card-value').get_text().strip())
            featurest.append([item.text for item in container.find_all('li', class_='amenities__item')])


        for container in pricediv:
            price.append(container.find(class_='property-card__price js-property-card-prices js-property-card__price-small').get_text().strip())
            mgmtfee.append(None) if container.find(class_='js-condo-price') == None else mgmtfee.append(container.find(class_='js-condo-price').get_text().strip())
            pricet.append(container.find(class_='property-card__price js-property-card-prices js-property-card__price-small').get_text().strip())
            mgmtfeet.append(None) if container.find(class_='js-condo-price') == None else mgmtfeet.append(container.find(class_='js-condo-price').get_text().strip())

        for container in linkdiv:
            link.append(container.find('a', href=True).get('href'))
            linkt.append(container.find('a', href=True).get('href'))


        sleep(randint(5,10)) #pausing to stop the code from running too fast 
        

        try:
            scrapeddatat = pd.DataFrame({
                'address': addresst, 
                'area': areat, 
                'rooms':roomst,
                'baths':bathst,
                'parking':parkingt,
                'features':featurest,
                'price':pricet,
                'mgmtfee':mgmtfeet,
                'link':linkt
                })
            scrapeddatat.to_csv(f'scrapeddatat{number}.csv', sep='|')
        except:
            pass

        try:
            cookie = driver.find_element_by_css_selector('#cookie-notifier-cta')
            cookie.click()
        except:
            pass

        try:
            prox = driver.find_element_by_xpath('//*[@title="Próxima página"]')
            prox.click()
            driver.current_url
        except:
            break







#%%    

# The code below actually failed. For some uninvestigated reason, 
# some 100-something of the nearly 900 failed to produce lists of the same size
# probably some part of the site failed to load in time and couldn't be scraped.
# It's kept here just for reference.


scrapeddata = pd.DataFrame({
    'address': address, 
    'area': area, 
    'rooms':rooms,
    'baths':baths,
    'parking':parking,
    'features':features,
    'price':price,
    'mgmtfee':mgmtfee,
    'link':link
    })
    
scrapeddata.to_csv('scrapeddata.csv', sep='|')
#%%    


# This code was introduced as a contingency because the code above failed.
# It created the dataset with the pages that produced consistent lists, while ignoring missing ones.
# The fact that the lists lengths matched were also a great unit test to check 
# if the scrape worked properly.
# It produced just over 27 thousand rows of data, meeting the requirements.


scrapeddata = pd.read_csv('scrapeddatat1.csv', sep='|', index_col=0)
for i in range (2,880):
    try:
        temp = pd.read_csv(f'scrapeddatat{i}.csv', sep='|', index_col=0)
        scrapeddata = scrapeddata.append(temp, ignore_index = True)
    except:
        pass
scrapeddata.to_csv('scrapeddata.csv', sep='|')



