import re
import calendar
import datetime
import configparser

import requests
from bs4 import BeautifulSoup
import psycopg2

config_reader = configparser.ConfigParser()
config_reader.read('../config/db.ini', encoding='utf-8')

DATABASE = config_reader.get(section='postgres', option='database')
USER = config_reader.get(section='postgres', option='user')
PASSWORD = config_reader.get(section='postgres', option='password')
HOST = config_reader.get(section='postgres', option='host')
PORT = config_reader.get(section='postgres', option='port')

agriculture_pa_dict = {
    'agriculture.wheat_srw': 'WHEAT-SRW(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.corn': 'CORN(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.soybeans': 'SOYBEANS(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.cocoa': 'COCOA(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.sugar': 'SUGAR NO. 11(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.cotton': 'COTTON NO. 2(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'agriculture.coffee': 'COFFEE C(.+?Percent of Open Interest Represented by Each Category of Trader)',
}
agriculture_url = 'https://www.cftc.gov/dea/futures/ag_lf.htm'


metals_pa_dict = {
    'metals.gold': 'GOLD(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'metals.silver': 'SILVER(.+?Percent of Open Interest Represented by Each Category of Trader)',
}
metals_and_other_url = 'https://www.cftc.gov/dea/futures/other_lf.htm'


financial_pa_dict = {
    'financial.canadian_dollar': 'CANADIAN DOLLAR(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.swiss_franc': 'SWISS FRANC(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.british_pound': 'BRITISH POUND(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.japanese_yen': 'JAPANESE YEN(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.euro_fx': 'EURO FX(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.australian_dollar': 'AUSTRALIAN DOLLAR(.+?Percent of Open Interest Represented by Each Category of Trader)',
    'financial.nz_dollar': 'NZ DOLLAR(.+?Percent of Open Interest Represented by Each Category of Trader)'
}

financial_url = 'https://www.cftc.gov/dea/futures/financial_lf.htm'


def get_report_content(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html5lib')
    content = soup.find('pre').text

    date_pa = re.compile(r'(Futures Only Positions as of |Futures Only, )(\S+) (\d+), (\d+)')
    search_res = re.search(date_pa, soup.text)
    month = search_res.group(2)
    month = list(calendar.month_name).index(month)
    day = int(search_res.group(3))
    year = int(search_res.group(4))
    report_date = datetime.date(year, month, day)
    return content, report_date


def obtain_data_agriculture_metals(pa, content):
    single_pa = re.compile(pa, flags=re.DOTALL)
    str_cut = re.search(single_pa, content).group(1)
    num_list = re.findall('-*\d+,\d+,\d+|-*\d+,\d+|-*\d+', str_cut)
    producer_long = int(num_list[5].replace(',', ''))
    producer_short = int(num_list[6].replace(',', ''))
    swapdealers_long = int(num_list[7].replace(',', ''))
    swapdealers_short = int(num_list[8].replace(',', ''))
    swapdealers_spreading = int(num_list[9].replace(',', ''))
    managedmoney_long = int(num_list[10].replace(',', ''))
    managedmoney_short = int(num_list[11].replace(',', ''))
    managedmoney_spreading = int(num_list[12].replace(',', ''))
    other_long = int(num_list[13].replace(',', ''))
    other_short = int(num_list[14].replace(',', ''))
    other_spreading = int(num_list[15].replace(',', ''))

    producer_long_change = int(num_list[49].replace(',', ''))
    producer_short_change = int(num_list[50].replace(',', ''))
    swapderlaers_long_change = int(num_list[51].replace(',', ''))
    swapderlaers_short_change = int(num_list[52].replace(',', ''))
    swapderlaers_spreading_change = int(num_list[53].replace(',', ''))
    managedmoney_long_change = int(num_list[54].replace(',', ''))
    managedmoney_short_change = int(num_list[55].replace(',', ''))
    managedmoney_spreading_change = int(num_list[56].replace(',', ''))
    other_long_change = int(num_list[57].replace(',', ''))
    other_short_change = int(num_list[58].replace(',', ''))
    other_spreading_change = int(num_list[59].replace(',', ''))
    return [
        producer_long, producer_short,
        swapdealers_long, swapdealers_short, swapdealers_spreading,
        managedmoney_long, managedmoney_short, managedmoney_spreading,
        other_long, other_short, other_spreading,
        producer_long_change, producer_short_change,
        swapderlaers_long_change, swapderlaers_short_change, swapderlaers_spreading_change,
        managedmoney_long_change, managedmoney_short_change, managedmoney_spreading_change,
        other_long_change, other_short_change, other_spreading_change
    ]


def obtain_data_financial(pa, content):
    single_pa = re.compile(pa, flags=re.DOTALL)
    str_cut = re.search(single_pa, content).group(1)
    num_list = re.findall('-*\d+,\d+,\d+|-*\d+,\d+|-*\d+', str_cut)
    dealer_long = int(num_list[3].replace(',', ''))
    dealer_short = int(num_list[4].replace(',', ''))
    dealer_spreading = int(num_list[5].replace(',', ''))

    institutional_long = int(num_list[6].replace(',', ''))
    institutional_short = int(num_list[7].replace(',', ''))
    institutional_spreading = int(num_list[8].replace(',', ''))

    leveragedfunds_long = int(num_list[9].replace(',', ''))
    leveragedfunds_short = int(num_list[10].replace(',', ''))
    leveragedfunds_spreading = int(num_list[11].replace(',', ''))

    other_long = int(num_list[12].replace(',', ''))
    other_short = int(num_list[13].replace(',', ''))
    other_spreading = int(num_list[14].replace(',', ''))

    dealer_long_change = int(num_list[20].replace(',', ''))
    dealer_short_change = int(num_list[21].replace(',', ''))
    dealer_spreading_change = int(num_list[22].replace(',', ''))

    institutional_long_change = int(num_list[23].replace(',', ''))
    institutional_short_change = int(num_list[24].replace(',', ''))
    institutional_spreading_change = int(num_list[25].replace(',', ''))

    leveragedfunds_long_change = int(num_list[26].replace(',', ''))
    leveragedfunds_short_change = int(num_list[27].replace(',', ''))
    leveragedfunds_spreading_change = int(num_list[28].replace(',', ''))

    other_long_change = int(num_list[29].replace(',', ''))
    other_short_change = int(num_list[30].replace(',', ''))
    other_spreading_change = int(num_list[31].replace(',', ''))

    return [
        dealer_long, dealer_short, dealer_spreading,
        institutional_long, institutional_short, institutional_spreading,
        leveragedfunds_long, leveragedfunds_short, leveragedfunds_spreading,
        other_long, other_short, other_spreading,
        dealer_long_change, dealer_short_change, dealer_spreading_change,
        institutional_long_change, institutional_short_change, institutional_spreading_change,
        leveragedfunds_long_change, leveragedfunds_short_change, leveragedfunds_spreading_change,
        other_long_change, other_short_change, other_spreading_change
    ]

def send_data_to_db_agriculture_metals(table_name, report_date, data_list):
    conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT * FROM {table_name}
        WHERE report_date = %s
        """,
        (report_date,))
    rows = cur.fetchall()
    if not rows:
        cur.execute(f"""
            INSERT INTO {table_name}
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            data_list+[report_date])
        conn.commit()
        print(f'upload {table_name} {report_date} success')
    cur.close()
    conn.close()


def send_data_to_db_financial(table_name, report_date, data_list):
    conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT * FROM {table_name}
        WHERE report_date = %s
        """,
        (report_date,))
    rows = cur.fetchall()
    if not rows:
        cur.execute(f"""
            INSERT INTO {table_name}
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            data_list+[report_date])
        conn.commit()
        print(f'upload {table_name} {report_date} success')
    cur.close()
    conn.close()


def main():
    agriculture_content, report_date = get_report_content(agriculture_url)
    for table_name in agriculture_pa_dict.keys():
        data_list = obtain_data_agriculture_metals(agriculture_pa_dict[table_name], agriculture_content)
        send_data_to_db_agriculture_metals(table_name, report_date, data_list)

    metals_content, report_date = get_report_content(metals_and_other_url)
    for table_name in metals_pa_dict.keys():
        data_list = obtain_data_agriculture_metals(metals_pa_dict[table_name], metals_content)
        send_data_to_db_agriculture_metals(table_name, report_date, data_list)

    financial_content, report_date = get_report_content(financial_url)
    for table_name in financial_pa_dict.keys():
        data_list = obtain_data_financial(financial_pa_dict[table_name], financial_content)
        send_data_to_db_financial(table_name, report_date, data_list)


if __name__ == '__main__':
    main()