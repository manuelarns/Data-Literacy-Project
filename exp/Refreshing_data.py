import pandas as pd
from src.preprocessing import complete_preprocessing, load_data

sal_devops = pd.read_csv("https://salaries.devops-jobs.net/download/salaries.csv")
sal_infosec = pd.read_csv("https://salaries.infosec-jobs.com/download/salaries.csv")
sal_ai = pd.read_csv("https://salaries.ai-jobs.net/download/salaries.csv")
sal_remote = pd.read_csv("https://salaries.freshremote.work/download/salaries.csv")

salaries = complete_preprocessing()

sal_devops.to_csv('../data/salaries_devops.csv')  
sal_infosec.to_csv('../data/salaries_infosec.csv')
sal_ai.to_csv('../data/salaries_ai.csv')
sal_remote.to_csv('../data/salaries.csv')

salaries.to_csv('../data/salaries_preprocessed.csv')