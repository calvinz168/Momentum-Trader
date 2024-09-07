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

def get_macd(stocks, interval, client):
    ts = client.time_series(
        symbol=stocks,
        interval=interval,
        outputsize=30,
        timezone="America/New_York",
    )

    data = ts.with_macd().as_json()
    macd_values = {ticker: [] for ticker in stocks.split(',')}
    
    for ticker, entries in data.items():
        for entry in entries:
            datetime_str = entry["datetime"]
            macd = float(entry["macd"])
            macd_signal = float(entry["macd_signal"])
            macd_hist = float(entry["macd_hist"])
            open = float(entry["open"])
            close = float(entry["close"])
            high = float(entry["high"])
            low = float(entry["low"])
            
            macd_values[ticker].append({
                "datetime": datetime_str,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "open": open,
                "close": close,
                "high": high,
                "low": low
            })

    return macd_values


def evaluate_stock_criteria(data, ticker):
    result = {"ticker": ticker, "cross": "H", "trend": "", "predicted_cross": "", "price_movement": ""}

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
    most_recent = data[-1]
    open_price = float(most_recent["open"])
    close_price = float(most_recent["close"])
    high_price = float(most_recent["high"])
    low_price = float(most_recent["low"])

    if close_price > open_price and high_price == close_price and low_price == open_price:
        result["price_movement"] = "B"
    elif close_price < open_price and low_price == close_price and high_price == open_price:
        result["price_movement"] = "S"
    else:
        result["price_movement"] = "H"

    return result


def get_tickers():
    with open("tickers copy.txt") as f:
        lines = f.readlines()
    
    return [line.strip() for line in lines]

important_list = []
tickers = get_tickers()

key_index = 0  # Start with the first client
batch_size = 4

for i in range(0, len(tickers), batch_size):
    batch_tickers = tickers[i:i + batch_size]
    client = clients[key_index]
    values = get_macd(",".join(batch_tickers), "1day", client)
    if values:
        for ticker in batch_tickers:
            evaluation = evaluate_stock_criteria(values[ticker], ticker)
            # Count the number of "B" evaluations
            b_count = sum(1 for key in ["cross", "price_movement", "predicted_cross"] if evaluation[key] == "B")
            s_count = sum(1 for key in ["cross", "price_movement", "predicted_cross"] if evaluation[key] == "S")
            
            if b_count >= 2 or s_count >= 2 or evaluation["cross"] != "H":
                important_list.append(f"{evaluation['ticker']}: Cross - {evaluation['cross']}, SYM - {evaluation['price_movement']}, Predicted Cross - {evaluation['predicted_cross']}")
            print(evaluation)
    
    key_index += 1
    
    # If all clients have been used, reset the index and sleep for 61 seconds
    if key_index >= len(clients):
        key_index = 0
        print("Processed one full cycle of keys. Sleeping for 61 seconds.")
        time.sleep(61)

print(f"Important stocks: {important_list}")

# Send email alerts with the important stocks
# send_email("6476688618@txt.freedommobile.ca", "Ticker: MACD/SYM", f"{important_list}")
# send_email("4168783284@txt.freedommobile.ca", "Ticker: MACD/SYM", f"{important_list}")
