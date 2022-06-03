import datetime
import pickle
import pandas as pd
import re

GOAL_DIR = '../history_data/CFTC_EURUSD.csv'
TITLE_LIST = [
    'Market_and_Exchange_Names',
    'As_of_Date_In_Form_YYMMDD',
    'Report_Date_as_YYYY-MM-DD',
    'CFTC_Contract_Market_Code',
    'CFTC_Market_Code',
    'CFTC_Region_Code',
    'CFTC_Commodity_Code',
    'Open_Interest_All',
    'Dealer_Positions_Long_All',
    'Dealer_Positions_Short_All',
    'Dealer_Positions_Spread_All',
    'Asset_Mgr_Positions_Long_All',
    'Asset_Mgr_Positions_Short_All',
    'Asset_Mgr_Positions_Spread_All',
    'Lev_Money_Positions_Long_All',
    'Lev_Money_Positions_Short_All',
    'Lev_Money_Positions_Spread_All',
    'Other_Rept_Positions_Long_All',
    'Other_Rept_Positions_Short_All',
    'Other_Rept_Positions_Spread_All',
    'Tot_Rept_Positions_Long_All',
    'Tot_Rept_Positions_Short_All',
    'NonRept_Positions_Long_All',
    'NonRept_Positions_Short_All',
    'Change_in_Open_Interest_All',
    'Change_in_Dealer_Long_All',
    'Change_in_Dealer_Short_All',
    'Change_in_Dealer_Spread_All',
    'Change_in_Asset_Mgr_Long_All',
    'Change_in_Asset_Mgr_Short_All',
    'Change_in_Asset_Mgr_Spread_All',
    'Change_in_Lev_Money_Long_All',
    'Change_in_Lev_Money_Short_All',
    'Change_in_Lev_Money_Spread_All',
    'Change_in_Other_Rept_Long_All',
    'Change_in_Other_Rept_Short_All',
    'Change_in_Other_Rept_Spread_All',
    'Change_in_Tot_Rept_Long_All',
    'Change_in_Tot_Rept_Short_All',
    'Change_in_NonRept_Long_All',
    'Change_in_NonRept_Short_All',
    'Pct_of_Open_Interest_All',
    'Pct_of_OI_Dealer_Long_All',
    'Pct_of_OI_Dealer_Short_All',
    'Pct_of_OI_Dealer_Spread_All',
    'Pct_of_OI_Asset_Mgr_Long_All',
    'Pct_of_OI_Asset_Mgr_Short_All',
    'Pct_of_OI_Asset_Mgr_Spread_All',
    'Pct_of_OI_Lev_Money_Long_All',
    'Pct_of_OI_Lev_Money_Short_All',
    'Pct_of_OI_Lev_Money_Spread_All',
    'Pct_of_OI_Other_Rept_Long_All',
    'Pct_of_OI_Other_Rept_Short_All',
    'Pct_of_OI_Other_Rept_Spread_All',
    'Pct_of_OI_Tot_Rept_Long_All',
    'Pct_of_OI_Tot_Rept_Short_All',
    'Pct_of_OI_NonRept_Long_All',
    'Pct_of_OI_NonRept_Short_All',
    'Traders_Tot_All',
    'Traders_Dealer_Long_All',
    'Traders_Dealer_Short_All',
    'Traders_Dealer_Spread_All',
    'Traders_Asset_Mgr_Long_All',
    'Traders_Asset_Mgr_Short_All',
    'Traders_Asset_Mgr_Spread_All',
    'Traders_Lev_Money_Long_All',
    'Traders_Lev_Money_Short_All',
    'Traders_Lev_Money_Spread_All',
    'Traders_Other_Rept_Long_All',
    'Traders_Other_Rept_Short_All',
    'Traders_Other_Rept_Spread_All',
    'Traders_Tot_Rept_Long_All',
    'Traders_Tot_Rept_Short_All',
    'Conc_Gross_LE_4_TDR_Long_All',
    'Conc_Gross_LE_4_TDR_Short_All',
    'Conc_Gross_LE_8_TDR_Long_All',
    'Conc_Gross_LE_8_TDR_Short_All',
    'Conc_Net_LE_4_TDR_Long_All',
    'Conc_Net_LE_4_TDR_Short_All',
    'Conc_Net_LE_8_TDR_Long_All',
    'Conc_Net_LE_8_TDR_Short_All',
    'Contract_Units',
    'CFTC_Contract_Market_Code_Quotes',
    'CFTC_Market_Code_Quotes',
    'CFTC_Commodity_Code_Quotes',
    'CFTC_SubGroup_Code',
    'FutOnly_or_Combined',
]
title = ''
for i in TITLE_LIST:
    title += i+','
title = title[:-1]
F_TFF_file_name = [
    'F_TFF_2006_2016.txt',
    'F_TFF_2017.txt',
    'F_TFF_2018.txt',
    'F_TFF_2019.txt',
    'F_TFF_2020.txt',
    'F_TFF_2021.txt',
    'F_TFF_2022.txt',
]

with open(GOAL_DIR, 'w') as goal_file:
    goal_file.write(title)
    goal_file.write('\n')

test_list = []

format = '%m/%d/%Y %I:%M:%S %p'

for i in F_TFF_file_name:
    count = 0
    with open(i, 'r') as cftc_file:
        for line in cftc_file.readlines():
            if count == 0:
                count += 1
                continue
            if line.startswith('"EURO FX - CHICAGO MERCANTILE EXCHANGE"') or line.startswith('EURO FX - CHICAGO MERCANTILE EXCHANGE'):
                line_list = list(map(lambda x: x.strip(),line.split(',')))
                date = line_list[2]
                if re.search(r'.+/.+/.+', date):
                    line_date = datetime.datetime.strptime(date, format)
                    line_list[2] = line_date.date().strftime('%Y-%m-%d')
                new_str = ''
                for _ in line_list:
                    new_str += _
                    new_str += ','
                new_str = new_str[:-1]
                new_str = new_str.replace('"', '')
                new_str += '\n'
                test_list.append(new_str)
            else:
                pass

    with open(GOAL_DIR, 'a+') as goal_file:
        for n in test_list[::-1]:
            goal_file.write(n)

    test_list = []

