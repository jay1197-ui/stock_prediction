import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# List of stock symbols
symbols = ["NVDA", "AMZN", "MSFT", "EMR", "AAPL", "GOOG", "XXII", "TSLA", "F", "BAC"]

# Calculate date range (last 365 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Fetch data for each symbol
data_frames = []
for symbol in symbols:
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    df['Symbol'] = symbol  # Add symbol column
    df = df.reset_index()  # Reset index to make Date a column
    data_frames.append(df)

# Combine all data into a single DataFrame
all_data = pd.concat(data_frames)

# Ensure the Date column is in the correct format
all_data['Date'] = pd.to_datetime(all_data['Date']).dt.strftime('%Y-%m-%d')

# Save to CSV
all_data.to_csv('stock_data_365days.csv', index=False)

print("Data fetched and saved to stock_data_365days.csv")