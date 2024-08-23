#Source: https://learndataanalysis.org/source-code-download-multiple-historical-stock-data-from-yahoo-finance-to-excel-using-python/
import yfinance as yf
import pandas as pd

try:
    tickers = ['PLD', 'AMT', 'EQIX', 'SPG', 'WELL', 'PSA', 'CCI', 'O', 'DLR', 'CSGP']
    interval = '1d'
    start_date = '2014-02-23'
    end_date = '2024-02-23'

    xlwriter = pd.ExcelWriter('real_estate.xlsx', engine='openpyxl')
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        stock_data.to_excel(xlwriter, sheet_name=ticker)

    xlwriter.close()  # Corrected line to save the Excel file, with the help of ChatGPT
except Exception as exception:
    print(exception)