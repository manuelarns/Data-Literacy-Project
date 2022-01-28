# Package imports
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import geopandas

### Load datasets
def load_data():
    # Salaries and geopandas world_data
    world_data = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    salaries_data = pd.read_csv('salaries.csv')
    # Country codes (ISO 3166 Alpha 2 and Alpha 3) 
    countries_data = pd.read_csv('countries.csv')
    return world_data, salaries_data, countries_data


### Extend country codes in salaries from xx (ISO Alpha 2) to xxx (ISO Alpha 3)
# E.g. US --> USA
def extend_country_code(salaries_data, countries_data):
    salaries_a3 = pd.merge(salaries_data, countries_data, left_on='company_location', right_on='iso_a2').drop(['iso_a2', 'id', 'name'], axis=1).rename(columns = {'iso_a3':'company_location_iso_a3'})
    salaries_a3 = pd.merge(salaries_a3, countries_data, left_on='employee_residence', right_on='iso_a2').drop(['iso_a2', 'id', 'name', 'company_location', 'employee_residence'], axis=1).rename(columns = {'iso_a3':'employee_residence_iso_a3'})
    return salaries_a3

### Add Gdp/capita to salaries_a3 (needs to fullfill ISO Alpha 3!)
def add_gdp_per_capita_to_salaries(salaries_a3, world_data):
    #Get gdp/capita from world_data
    # TODO: Warning???
    world = world_data[world_data.name!="Antarctica"]
    gdp_data = world[['iso_a3', 'gdp_md_est', 'pop_est']]
    gdp_data['gdp_per_capita'] = (gdp_data['gdp_md_est'] / gdp_data['pop_est'] * 1000000)
    gdp_per_capita = gdp_data.drop({'gdp_md_est', 'pop_est'}, axis=1)

    #Merge gdp_per_capita with salaries_a3
    salaries_a3_gdp = pd.merge(salaries_a3, gdp_per_capita, left_on='company_location_iso_a3', right_on='iso_a3', how='left').rename(columns = {'gdp_per_capita':'gdp_company_location'})
    salaries_a3_gdp = pd.merge(salaries_a3_gdp, gdp_per_capita, left_on='employee_residence_iso_a3', right_on='iso_a3', how='left').rename(columns = {'gdp_per_capita':'gdp_employee_residence'})
    salaries_gdp = salaries_a3_gdp.drop(['iso_a3_x', 'iso_a3_y'], axis=1)
    return salaries_gdp


### Make non-numeric attributes experience_level, company_size and work_year numeric of salaries numeric
def make_numeric(salaries):
    j = 1
    for i in ["EN", "MI", "SE", "EX"]:
        salaries.loc[(salaries.experience_level == i), "experience_level"]=j
        j += 1

    j = 1
    for i in ["S", "M", "L"]:
        salaries.loc[(salaries.company_size == i), "company_size"]=j
        j += 1

    for i in ["2021e", "2022e"]:
        salaries.loc[(salaries.work_year == i), "work_year"]= int(i[:-1])

    salaries = salaries.astype({'experience_level': 'int64', 'company_size': 'int64', 'work_year': 'int64' })
    return salaries

### Add attriute to salaries: Does the employee work in the same country as is residence?
def add_samecountry_attribute(salaries):
    ones_data = np.ones((salaries.shape[0],1))
    salaries["same_country"] = pd.DataFrame(ones_data)
    salaries.loc[salaries["company_location_iso_a3"] != salaries["employee_residence_iso_a3"], "same_country"] = 0
    return salaries

# Add attribute: Does the job_title indicate an ai/ml related work?
def add_ml_or_ai_attribute(salaries):
    zeros_data = np.zeros((salaries.shape[0],1))
    salaries["ai_or_ml_job"] = pd.DataFrame(zeros_data)
    salaries.loc[salaries["job_title"].str.contains("Data|Machine Learning|AI|ML") == True, "ai_or_ml_job"] = 1
    return salaries

# Execute all preprocessing steps
def complete_preprocessing(world_data, salaries_data, countries_data):
    salaries_a3 = extend_country_code(salaries_data, countries_data)
    salaries_gdp = add_gdp_per_capita_to_salaries(salaries_a3, world_data)
    salaries_gdp = make_numeric(salaries_gdp)
    salaries_gdp = add_samecountry_attribute(salaries_gdp)
    salaries_gdp = add_ml_or_ai_attribute(salaries_gdp)    
    return salaries_gdp

