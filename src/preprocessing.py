# Package imports
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import geopandas

def load_data():
    '''
    load all datasets: world_data, salaries_data, countries_data

    return: datasets: world_data, salaries_data, countries_data
    '''
    world_data = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))    # goepandas map and gdp data
    salaries_data = pd.read_csv('../data/salaries.csv')                                     # salaries dataset
    countries_data = pd.read_csv('../data/countries.csv')                                   # ISO codes for all counties
    return world_data, salaries_data, countries_data

def drop_outliers(salaries_data):
    '''
    Drop outliers that have higher salary than the 98% quantile or lower than 2% quantile
    '''
    salaries = salaries_data[
        salaries_data.salary_in_usd < salaries_data.salary_in_usd.quantile(.98)]
    
    salaries = salaries[
        salaries.salary_in_usd > salaries.salary_in_usd.quantile(.02)]
        
    return salaries
    

def extend_country_code(salaries_data, countries_data):
    '''
    Extend country codes in salaries from xx (ISO Alpha 2) to xxx (ISO Alpha 3)
    E.g. US --> USA

    salaries_data: unaltered salaries dataframe
    countries_data: dataframe containing all ISO_A2 and ISO_A3 codes
    return: salaries dataframe with new columns company_location_iso_a3 and employee_residence_iso_a3
    '''
    salaries_a3 = pd.merge(salaries_data,
                           countries_data,
                           left_on='company_location',
                           right_on='iso_a2').drop(['iso_a2',
                                                    'id',
                                                    'name'],
                                                   axis=1).rename(columns = {'iso_a3':'company_location_iso_a3'})
    salaries_a3 = pd.merge(salaries_a3,
                           countries_data,
                           left_on='employee_residence',
                           right_on='iso_a2').drop(['iso_a2',
                                                    'id',
                                                    'name',
                                                    'company_location',
                                                    'employee_residence'],
                                                   axis=1).rename(columns = {'iso_a3':'employee_residence_iso_a3'})
    return salaries_a3


def add_gdp_per_capita_to_salaries(salaries, world_data):
    '''
    Calculates the estimated GDP per capita for all countries using the data from the world_data dataframe
    and adds them to the salaries dataframe for company location and employee residence

    salaries: salaries dataframe with ISO_A3 country codes
    world_data: world dataframe containing GDP estimates and population estimates
    return: salaries dataframe with GDP data for company location and employee residence
    '''
    if 'company_location_iso_a3' not in salaries.columns and 'employee_residence_iso_a3' not in salaries.columns:
        raise ValueError('Locations are not in ISO_A3 format. Use extend_country_code() first!')

    world = world_data[world_data.name!="Antarctica"]
    gdp_data = world[['iso_a3', 'gdp_md_est', 'pop_est']]
    gdp_data['gdp_per_capita'] = (gdp_data['gdp_md_est'] / gdp_data['pop_est'] * 1000000)
    gdp_per_capita = gdp_data.drop({'gdp_md_est', 'pop_est'}, axis=1)

    #Merge gdp_per_capita with salaries
    salaries = pd.merge(salaries,
                        gdp_per_capita,
                        left_on='company_location_iso_a3',
                        right_on='iso_a3',
                        how='left').rename(columns = {'gdp_per_capita':'gdp_company_location'})
    salaries = pd.merge(salaries,
                        gdp_per_capita,
                        left_on='employee_residence_iso_a3',
                        right_on='iso_a3',
                        how='left').rename(columns = {'gdp_per_capita':'gdp_employee_residence'})
    salaries = salaries.drop(['iso_a3_x', 'iso_a3_y'], axis=1)
    return salaries


def make_numeric(salaries):
    '''
    Turn non-numeric columns experience_level, company_size, and work_year into numeric values

    salaries: salaries dataframe
    return: salaries dataframe with numeric attributes
    '''
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


def add_samecountry_attribute(salaries):
    '''
    Add attriute to salaries: Does the employee work in the same country as is his residencs?

    salaries: salaries dataframe with ISO_A3 country codes
    return: salaries dataframe with new samecounty column
    '''
    if 'company_location_iso_a3' not in salaries.columns and 'employee_residence_iso_a3' not in salaries.columns:
        raise ValueError('Locations are not in ISO_A3 format. Use extend_country_code() first!')

    ones_data = np.ones((salaries.shape[0],1))
    salaries["same_country"] = pd.DataFrame(ones_data)
    salaries.loc[salaries["company_location_iso_a3"] != salaries["employee_residence_iso_a3"], "same_country"] = 0
    return salaries


def add_ml_or_ai_attribute(salaries):
    '''
    Add attribute: Does the job_title indicate an ai/ml related work?

    salaries: salaries dateframe
    return: salaries dataframe with new ai_or_ml_job column
    '''
    zeros_data = np.zeros((salaries.shape[0],1))
    salaries["ai_or_ml_job"] = pd.DataFrame(zeros_data)
    salaries.loc[salaries["job_title"].str.contains("Data|Machine Learning|AI|ML") == True, "ai_or_ml_job"] = 1
    return salaries


def complete_preprocessing():
    '''
    Execute all preprocessing steps

    return: fully preprocessed salaries dataframe
    '''
    world_data, salaries_data, countries_data = load_data()
    salaries = drop_outliers(salaries_data)
    salaries = extend_country_code(salaries, countries_data)
    salaries = add_gdp_per_capita_to_salaries(salaries, world_data)
    salaries = make_numeric(salaries)
    salaries = add_samecountry_attribute(salaries)
    salaries = add_ml_or_ai_attribute(salaries)
    salaries = salaries.dropna()    
    return salaries
