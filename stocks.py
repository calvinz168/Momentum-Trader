from logging import exception
from twelvedata import TDClient
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time
from test import send_email

# Load API keys
with open('key.json', 'r') as file:
    config = json.load(file)

keys = config['keys']  # List of API keys

# Initialize clients
clients = [TDClient(apikey=key) for key in keys]


# Calculate moving averages for 5, 10, and 20 days
def moving_average(prices, days):
    if len(prices) >= days:
        return sum(prices[:days]) / days
    return None  # Not enough data for this period


def get_data(stocks, interval, client):
    ts = client.time_series(
        symbol=stocks,
        interval=interval,
        outputsize=30,
        timezone="America/New_York",
    )

    data = ts.with_macd().as_json()
    values = {ticker: [] for ticker in stocks.split(',')}
    closing_prices = {ticker: [] for ticker in stocks.split(',')}
    
    for ticker, entries in data.items():
        for entry in entries:
            closing_prices[ticker].append(float(entry["close"]))

    for ticker, entries in data.items():
        for idx, entry in enumerate(entries):
            datetime_str = entry["datetime"]
            macd = float(entry["macd"])
            macd_signal = float(entry["macd_signal"])
            macd_hist = float(entry["macd_hist"])
            open_price = float(entry["open"])
            close_price = float(entry["close"])
            high = float(entry["high"])
            low = float(entry["low"])
            
            ma_5 = sum(closing_prices[ticker][idx:idx+5]) / (5)
            ma_10 = sum(closing_prices[ticker][idx:idx+10]) / (10)
            ma_20 = sum(closing_prices[ticker][idx:idx+20]) / (20)

            # Append the entry with calculated moving averages
            values[ticker].append({
                "datetime": datetime_str,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "open": open_price,
                "close": close_price,
                "high": high,
                "low": low,
                "ma_5": ma_5,
                "ma_10": ma_10,
                "ma_20": ma_20
            })

    return values


def evaluate_stock_criteria(data, ticker):
    result = {"ticker": ticker, "cross": "H", "trend": "H", "predicted_cross": "H", "moving_average": "H"}

    # Detect MACD crossover in the last 2 days
    crosses = []
    for i in range(len(data) - 1):
        current = data[i]
        next_data = data[i + 1]

        current_macd = float(current["macd"])
        current_macd_signal = float(current["macd_signal"])
        next_macd = float(next_data["macd"])
        next_macd_signal = float(next_data["macd_signal"])

        if (current_macd < current_macd_signal and next_macd > next_macd_signal):
            crosses.append({"datetime": next_data["datetime"], "type": "death cross"})
        elif (current_macd > current_macd_signal and next_macd < next_macd_signal):
            crosses.append({"datetime": next_data["datetime"], "type": "golden cross"})

    # Check for recent cross within the last 2 days
    now = datetime.now()
    yesterday = now - timedelta(days=3)
    recent_cross = None

    for cross in crosses:
        cross_datetime = datetime.strptime(cross["datetime"], "%Y-%m-%d")
        if cross_datetime >= yesterday:
            recent_cross = cross
            break

    # Update result based on recent cross
    if recent_cross:
        if recent_cross["type"] == "golden cross":
            result["cross"] = "B"
        elif recent_cross["type"] == "death cross":
            result["cross"] = "S"

    # Predict tomorrow's trend
    last_two = data[-2:]
    x = np.array([i for i in range(len(last_two))]).reshape(-1, 1)
    y_macd = np.array([float(point["macd"]) for point in last_two])
    y_signal = np.array([float(point["macd_signal"]) for point in last_two])

    model_macd = LinearRegression().fit(x, y_macd)
    model_signal = LinearRegression().fit(x, y_signal)

    next_x = np.array([[len(last_two)]])
    predicted_macd = model_macd.predict(next_x)[0]
    predicted_signal = model_signal.predict(next_x)[0]

    result["trend"] = "B" if predicted_macd > predicted_signal else "S"

    # Determine if a crossover is predicted for tomorrow
    last_macd = float(last_two[-1]["macd"])
    last_signal = float(last_two[-1]["macd_signal"])

    if (last_macd < last_signal and predicted_macd > predicted_signal):
        result["predicted_cross"] = "B"
    elif (last_macd > last_signal and predicted_macd < predicted_signal):
        result["predicted_cross"] = "S"
    else:
        result["predicted_cross"] = "H"

    # Evaluate price movement (open, high, low, close)
    most_recent = data[0]
    prev_close = float(data[1]["close"])
    close_price = float(most_recent["close"])
    high_price = float(most_recent["high"])
    low_price = float(most_recent["low"])
    open_price = float(most_recent["open"])

    if close_price > prev_close and low_price >= close_price * 0.95:
        result["sym"] = "B"
    elif close_price < prev_close and high_price <= open_price * 1.05:
        result["sym"] = "S"
    else:
        result["sym"] = "H"

    ma5 = float(data[0]["ma_5"])
    ma10 = float(data[0]["ma_10"])
    ma20 = float(data[0]["ma_20"])
    
    if ma5 and ma10 and ma20:
        if ma5 > ma10 and ma10 > ma20:
            result["moving_average"] = "B"  # 5-day > 10-day > 20-day
        elif ma5 < ma10 and ma10 < ma20:
            result["moving_average"] = "S"  # 5-day < 10-day < 20-day
        else:
            result["moving_average"] = "H"  # No specific trend
    
    return result


def get_tickers():
    with open("tickers copy.txt") as f:
        lines = f.readlines()
    
    return [line.strip() for line in lines]

# Important lists
important_buy_list = []
important_sell_list = []

def output_sorted():
    print("Ticker: MACD Cross / SYM / SMA")
    print("\nBuy:")
    for stock in important_buy_list:
        ticker = stock.split(":")[0]
        cross, trend, ma = stock.split(":")[1].split(", ")
        print(f"{ticker}: {cross} {trend} {ma}")

    print("\nSell:")
    for stock in important_sell_list:
        ticker = stock.split(":")[0]
        cross, trend, ma = stock.split(":")[1].split(", ")
        print(f"{ticker}: {cross} {trend} {ma}")

tickers = get_tickers()

key_index = 0  # Start with the first client
batch_size = 4

for i in range(0, len(tickers), batch_size):
    batch_tickers = tickers[i:i + batch_size]
    client = clients[key_index]
    values = get_data(",".join(batch_tickers), "1day", client)
    
    if values:
        for ticker in batch_tickers:
            evaluation = evaluate_stock_criteria(values[ticker], ticker)
            # Count the number of "B" and "S" evaluations
            b_count = sum(1 for key in ["cross", "sym", "moving_average"] if evaluation[key] == "B")
            s_count = sum(1 for key in ["cross", "sym", "moving_average"] if evaluation[key] == "S")
            
            stock_info = f"{evaluation['ticker']}: {evaluation['cross']}, {evaluation['sym']}, {evaluation['moving_average']}"
            
            print(evaluation)
            if b_count > 2:
                important_buy_list.append(stock_info)
            elif s_count > 2:
                important_sell_list.append(stock_info)


    key_index += 1
    
    # If all clients have been used, reset the index and sleep for 61 seconds
    if key_index >= len(clients):
        key_index = 0
        print("Processed one full cycle of keys. Sleeping for 61 seconds.")
        time.sleep(61)

# Output the sorted lists
output_sorted()

# Convert the lists to strings for email content
buy_stocks_string = "\n".join(important_buy_list)
sell_stocks_string = "\n".join(important_sell_list)

# Prepare email content
email_body = f"Ticker: MACD Cross / SYM / SMA\nBuy:\n{buy_stocks_string}\n\nSell:\n{sell_stocks_string}"

# Send email with the combined summary
send_email("calvinz168@gmail.com", f"Stocks Summary - {datetime.now().date()}", email_body)
