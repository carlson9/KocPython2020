#ease of business

import wbdata
wbdata.get_source()
wbdata.get_indicator(source=1)
#get country codes with a search
wbdata.search_countries('Turkey') #TUR
#get indicators with a search
wbdata.search_indicators('ease of doing business') #IC.BUS.DFRN.XQ
wbdata.get_data('IC.BUS.DFRN.XQ', country='TUR')[0]
wbdata.search_countries('united') #GBR
wbdata.get_data('IC.BUS.DFRN.XQ', country='GBR')[0]

import datetime
data_date = (datetime.datetime(2017, 1, 1), datetime.datetime(2019, 1, 1))
wbdata.get_data("IC.BUS.DFRN.XQ", country=("USA", "GBR"), data_date=data_date)

wbdata.search_indicators("gdp per capita")
wbdata.get_data('NY.GDP.PCAP.KD.ZG')

wbdata.get_data('NY.GDP.PCAP.KD.ZG', country = 'USA')
wbdata.get_data('NY.GDP.PCAP.KD.ZG', country = 'OED')

#income level filter
wbdata.get_incomelevel()
countries = [i['id'] for i in wbdata.get_country(incomelevel="HIC", display=False)]
indicators = {"IC.BUS.DFRN.XQ": "doing_business", "NY.GDP.PCAP.PP.KD": "gdppc"}
df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)

df.to_csv('KocPython2020/in-classMaterial/day6/econ.csv')
df.describe()

#TODO: explore the API and pick some interesting variables that may have a theoretical connection, then run a regression (using any software is fine)

