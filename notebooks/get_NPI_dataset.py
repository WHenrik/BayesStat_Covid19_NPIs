
import numpy as np
import pandas as pd
import arviz as az
import pymc3 as pm
import load_covid_data

import datetime
import functools
import operator


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Function for flatting
def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])


# Function to convert string to datetime 
def convert_str(date_time): 
    format = '%Y%m%d' # The format 
    datetime_str = datetime.datetime.strptime(date_time, format).date()
    return datetime_str 


# Function to convert string to datetime without using date
def convert_time(date_time): 
    format = '%Y-%m-%d' # The format‚
    datetime_str = datetime.datetime.strptime(date_time, format)
    return datetime_str 

#Function to map finer Oxford values to the schema of Brauner et al.
def convert(x,class0):
    if x in class0:
        x=0.0
    else:
        x=1.0
    return x

# Countries in dataset of Brauner et al.
country_list=['Albania', 'Andorra', 'Austria', 'Belgium',
       'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic',
       'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany',
       'Greece', 'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy',
       'Latvia', 'Lithuania', 'Malaysia', 'Malta', 'Mexico', 'Morocco',
       'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Portugal',
       'Romania', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia',
       'South Africa', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']

def get_NPI_dataset(save_path="NPI_Dataset",start=None, end=None):
    """
        Uses 3 datasets to generate the final NPI dataset.
        
        Args:
            name (:obj:`str`):
                Name that the dataset should have after saving.
            start (:obj:`str`):
                Start date of the records
            end (:obj:`str`):
                End date of the records
        Returns:
           :obj:`DataFrame`: final dataset
    """
    
    
    
    #Load dataset for case and death numbers
    data_recov=load_covid_data.load_individual_timeseries("recovered",drop_states=True)
    data_cases=load_covid_data.load_data(drop_states=True)
    
    data_cases["recovered"]=data_recov.cases
    
    #get number of current active cases
    active_cases=[]
    for country in data_cases.country.unique():
        active_cases.append(list(data_cases.loc[data_cases.country==country].confirmed - data_cases.loc[data_cases.country==country].deaths - data_recov.loc[data_recov.country==country].cases))
    flatten_list=functools_reduce_iconcat(active_cases)
    data_cases["Actives"]=flatten_list
    
    #Load NPI Dataset from Oxford
    url="https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
    data_oxf=pd.read_csv(url,low_memory=False)
    
    # "US" in data_cases,'United States' in data_oxf
    data_cases.country.replace(to_replace="US",value='United States',inplace=True) 
    
    # Country filtering and timestamp fixing
    data_oxf = data_oxf[data_oxf.CountryName.isin(country_list)]
    data_oxf.Date=data_oxf.Date.apply(lambda x: convert_str(str(x)))
    
    #Join the datasets via inner join - keys : Date and CountryName
    
    data_cases['Date'] = data_cases.index
    
    data_cases.Date=data_cases.Date.apply(lambda x: str(x.date()))
    data_oxf.Date=data_oxf.Date.apply(lambda x: str(x))
    dataset = pd.merge(data_oxf, data_cases,  how='inner', left_on=['Date','CountryName'], right_on = ['Date','country'])
    
    #just ordering and discarding not needed columns
    dataset=dataset[['CountryName', 'CountryCode', 'RegionName', 'RegionCode',
           'Jurisdiction', 'Date','confirmed', 'critical_estimate', 'days_since_100', 'deaths',
           'recovered', 'Actives', 'C1_School closing', 'C1_Flag',
           'C2_Workplace closing', 'C2_Flag', 'C3_Cancel public events', 'C3_Flag',
           'C4_Restrictions on gatherings', 'C4_Flag', 'C5_Close public transport',
           'C5_Flag', 'C6_Stay at home requirements', 'C6_Flag',
           'C7_Restrictions on internal movement', 'C7_Flag',
           'C8_International travel controls', 'E1_Income support', 'E1_Flag',
           'E2_Debt/contract relief', 'E3_Fiscal measures',
           'E4_International support', 'H1_Public information campaigns',
           'H1_Flag', 'H2_Testing policy', 'H3_Contact tracing',
           'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
           'H6_Facial Coverings', 'H6_Flag', 'H7_Vaccination policy', 'H7_Flag',
           'M1_Wildcard' , 'StringencyIndex',
           'StringencyIndexForDisplay', 'StringencyLegacyIndex',
           'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex',
           'GovernmentResponseIndexForDisplay', 'ContainmentHealthIndex',
           'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex',
           'EconomicSupportIndexForDisplay']]
    
    #check if there are always the same amount of country values
    median_of_tracked_days=np.median(dataset["CountryCode"].value_counts())
    countries_tracked_on_state_level=[]
    
    for country_name in dataset.CountryName.unique():
        if len(dataset.loc[dataset.CountryName==country_name])>median_of_tracked_days:
            countries_tracked_on_state_level.append(country_name)
    
    dataset_cleaned = dataset[~dataset.CountryName.isin(countries_tracked_on_state_level)]
    
    #select just the necessary columns - first matching thoughts
    dataset_cleaned=dataset_cleaned[["CountryCode", # 'Country Code'
                          "Date",  #'Date'
                          'CountryName', #'Region Name'
                          "confirmed",  #'Confirmed'
                          'Actives', #Active
                           'deaths', #'Deaths'
                          "H6_Facial Coverings", #'Mask Wearing'
                          "H2_Testing policy", #'Symptomatic Testing'
                          "C4_Restrictions on gatherings", #'Gatherings <1000', 'Gatherings <100', 'Gatherings <10'
                          'C2_Workplace closing', #'Some Businesses Suspended', 'Most Businesses Suspended'
                          'C1_School closing', #'School Closure', 'University Closure'
                          'C6_Stay at home requirements', #'Stay Home Order'
                          'H3_Contact tracing', #'Travel Screen/Quarantine' 
                          'C8_International travel controls', #'Travel Bans' 
                          'C5_Close public transport', #'Public Transport Limited'
                          "C7_Restrictions on internal movement", #'Internal Movement Limited'
                          'H1_Public information campaigns']] #'Public Information Campaigns'
    
    #rename dataframe to the column names of the original dataset
    data_renamed=dataset_cleaned.rename(columns={"CountryCode": "Country Code",
                                                  'CountryName':'Region Name',
                                                  "confirmed":'Confirmed',
                                                  "Actives":"Active",
                                                  'deaths':'Deaths',
                                                  "H6_Facial Coverings":'Mask Wearing',
                                                  "H2_Testing policy":'Symptomatic Testing',
                                                  "C4_Restrictions on gatherings":'Gatherings <1000', 
                                                  'C2_Workplace closing':'Some Businesses Suspended', 
                                                  'C1_School closing':'School Closure', 
                                                 'C6_Stay at home requirements':'Stay Home Order',
                                                  'H3_Contact tracing':'Travel Screen/Quarantine',
                                                  'C8_International travel controls':'Travel Bans',
                                                  'C5_Close public transport':'Public Transport Limited',
                                                  "C7_Restrictions on internal movement":'Internal Movement Limited',
                                                  'H1_Public information campaigns':'Public Information Campaigns'
                                                 }, errors="raise")
    
    #fix type error
    data_renamed=data_renamed.astype({'Deaths': 'float64', "Confirmed":'float64'})
    
    #check where Nans occur & and drop contry if more than 10% of columns have NaN values
    data_nans=data_renamed[data_renamed.isnull().any(axis=1)]
    data_nans_counts=data_nans["Region Name"].value_counts()
    
    countries_with_too_many_nans=data_nans_counts.index[data_nans_counts>0.1*median_of_tracked_days]
    
    #drop countries
    data_renamed = data_renamed[~data_renamed["Region Name"].isin(countries_with_too_many_nans)]
    
    #fill remaining NaNs with 0 - maybe replace with foward or backward pass 
    data_renamed.fillna(0.0, inplace=True)
    
    data=data_renamed
    data.Date=data.Date.apply(lambda x: convert_time(x))
    
    #Mask wearing
    data["Mask Wearing"]=data["Mask Wearing"].apply(lambda x: convert(x,[0,1]))
    
    #Symptomatic Testing
    data["Symptomatic Testing"]=data["Symptomatic Testing"].apply(lambda x: convert(x,[0,1])) #drop 1?
    
    #Gatherings <10:
    data["Gatherings <10"]=data["Gatherings <1000"].apply(lambda x: convert(x,[0,1,2,3]))
    
    #Gatherings <100:
    data["Gatherings <100"]=data["Gatherings <1000"].apply(lambda x: convert(x,[0,1,2]))
    
    #Gatherings <1000:
    data["Gatherings <1000"]=data["Gatherings <1000"].apply(lambda x: convert(x,[0,1]))
    
    #Most Businesses Suspended
    data["Most Businesses Suspended"]=data["Some Businesses Suspended"].apply(lambda x: convert(x,[0,1,2]))
    
    #Some Businesses Suspended
    data["Some Businesses Suspended"]=data["Some Businesses Suspended"].apply(lambda x: convert(x,[0,1]))
    
    # School Closure
    data["School Closure"]=data["School Closure"].apply(lambda x: convert(x,[0,1]))
    data['University Closure']=data["School Closure"] #duplication, since no difference left in current data
    
    # Stay Home Order
    data["Stay Home Order"]=data["Stay Home Order"].apply(lambda x: convert(x,[0,1]))
    
    # Travel Screen/Quarantine
    data["Travel Screen/Quarantine"]=data["Travel Screen/Quarantine"].apply(lambda x: convert(x,[0]))
    
    # Travel Bans
    data["Travel Bans"]=data["Travel Bans"].apply(lambda x: convert(x,[0,1,2]))
    
    # Public Transport Limited
    data["Public Transport Limited"]=data["Public Transport Limited"].apply(lambda x: convert(x,[0,1]))
    
    # Internal Movement Limited
    data["Internal Movement Limited"]=data["Internal Movement Limited"].apply(lambda x: convert(x,[0,1]))
    
    # Public Information Campaigns
    data["Public Information Campaigns"]=data["Public Information Campaigns"].apply(lambda x: convert(x,[0]))
    
    #simple reordering - not necessary
    data=data[['Country Code', 'Date', 'Region Name', 'Confirmed', 'Active', 'Deaths',
       'Mask Wearing', 'Symptomatic Testing', 'Gatherings <1000',
       'Gatherings <100', 'Gatherings <10', 'Some Businesses Suspended',
       'Most Businesses Suspended', 'School Closure', 'University Closure',
       'Stay Home Order', 'Travel Screen/Quarantine', 'Travel Bans',
       'Public Transport Limited', 'Internal Movement Limited',
       'Public Information Campaigns']]
    
    #time restriction and small format changes
    if start is not None:
        data=data[data.Date>start] #"2020-08-31"
        
    if end is not None:
        data=data[data.Date<end]
    
    data.Date=data.Date.apply(lambda x: str(x.date()))
    data.reset_index(drop=True, inplace=True)
    
    #save datasset
    data.to_csv(f"{save_path}.csv",index=False)
    
    print(f"data is created")
    
    return data