import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Get the most recent data for each stock
latest_data = df.loc[df.groupby('Symbol')['Date'].idxmax()]

# Calculate the change and percent change
latest_data['Change'] = latest_data.groupby('Symbol')['Close'].diff()
latest_data['Percent_Change'] = latest_data.groupby('Symbol')['Close'].pct_change() * 100

# Function to format the output for each stock
def format_stock_info(row):
    return (f"{row['Symbol']}\n"
            f"${row['Close']:.2f}\n"
            f"{row['Percent_Change']:+.2f}% ({row['Change']:+.2f})")

# Apply the formatting function to each row
latest_data['Formatted_Output'] = latest_data.apply(format_stock_info, axis=1)

# Print the formatted output for each stock
print("Current Stock Information:")
for output in latest_data['Formatted_Output']:
    print(output)
    print("\n" + "-"*30 + "\n")  # Separator between stocks

# Performance Comparison and Ranking
performance = df.groupby('Symbol').apply(lambda x: (x['Close'].iloc[-1] - x['Close'].iloc[0]) / x['Close'].iloc[0] * 100)
performance_ranked = performance.sort_values(ascending=False)
print("30-Day Performance Ranking:")
for rank, (symbol, perf) in enumerate(performance_ranked.items(), 1):
    print(f"{rank}. {symbol}: {perf:.2f}%")

# Identify best and worst performing stocks
best_stock = performance_ranked.index[0]
worst_stock = performance_ranked.index[-1]
print(f"\nBest performing stock: {best_stock} ({performance_ranked[best_stock]:.2f}%)")
print(f"Worst performing stock: {worst_stock} ({performance_ranked[worst_stock]:.2f}%)")

# Volatility Analysis
df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
volatility = df.groupby('Symbol')['Daily_Return'].std() * np.sqrt(252)  # Annualized
volatility_ranked = volatility.sort_values(ascending=False)
print("\nVolatility Ranking (Annualized):")
for rank, (symbol, vol) in enumerate(volatility_ranked.items(), 1):
    print(f"{rank}. {symbol}: {vol:.4f}")

# Investment Prediction
print("\nInvestment Prediction:")

# Function to calculate prediction score
def calculate_prediction_score(stock_data):
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data['Close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    coefficient = model.coef_[0]
    
    return score, coefficient

# Calculate prediction scores for each stock
prediction_scores = {}
for symbol in df['Symbol'].unique():
    stock_data = df[df['Symbol'] == symbol].sort_values('Date')
    score, coefficient = calculate_prediction_score(stock_data)
    prediction_scores[symbol] = (score, coefficient)

# Rank stocks based on prediction score and positive trend
ranked_predictions = sorted(prediction_scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

print("Top 5 stocks based on prediction model:")
for i, (symbol, (score, coefficient)) in enumerate(ranked_predictions[:5], 1):
    print(f"{i}. {symbol}: Score = {score:.4f}, Trend = {'Positive' if coefficient > 0 else 'Negative'}")

print("\nNote: This prediction is based on historical data and a simple linear model.")
print("It should not be considered as financial advice. Always do thorough research")
print("and consult with a financial advisor before making investment decisions.")

# Visualizations
plt.figure(figsize=(12, 6))
for symbol in df['Symbol'].unique():
    data = df[df['Symbol'] == symbol]
    plt.plot(data['Date'], data['Close'], label=symbol)

plt.title('Stock Prices Over 30 Days')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stock_prices_30days.png')
plt.close()

# Performance comparison bar plot
plt.figure(figsize=(12, 6))
performance_ranked.plot(kind='bar')
plt.title('30-Day Stock Performance Comparison')
plt.xlabel('Stock Symbol')
plt.ylabel('Performance (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.close()

print("\nAnalysis complete. Charts saved as PNG files.")