import pandas as pd
from datetime import datetime
from datetime import timezone
from datetime import date
from dateutil.relativedelta import relativedelta


def read_historic_data(ticker):
    # Unix timestamp calculation for today's date and five years ago to obtain Yahoo Finance data
    date_today = datetime.today().strftime("%Y-%m-%d")
    dtc = date_today.split("-")
    date_five_years_ago = (
        datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
    dfyc = date_five_years_ago.split("-")
    timestamp_today = int(datetime(int(dtc[0]), int(dtc[1]), int(
        dtc[2]), 0, 0).replace(tzinfo=timezone.utc).timestamp())
    timestamp_five_years_ago = int((datetime(int(dfyc[0]), int(dfyc[1]), int(
        dfyc[2]), 0, 0)).replace(tzinfo=timezone.utc).timestamp())

    # Reading in stock data from Yahoo Finance in the above timestamps' range
    csv_url = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker + \
        "?period1=" + str(timestamp_five_years_ago) + "&period2=" + \
        str(timestamp_today) + "&interval=1d&events=history"
    df = pd.read_csv(csv_url)

    return df


def generate_dates_until(year, month, day):
    start_date = date.today()
    end_date = date(year, month, day)
    dates_dict = {"Date": pd.date_range(start=start_date, end=end_date)}
    dates_df = pd.DataFrame(data=dates_dict)

    return dates_df