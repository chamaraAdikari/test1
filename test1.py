import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ccxt
import time
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class MovingAverageCrossoverML:
    def __init__(self, config):
        """
       Initialize the MovingAverageCrossoverML class with a configuration dictionary.


       Parameters:
       - config: A dictionary containing all configuration parameters.
       """
        self.symbol = config['symbol']
        self.short_window = config['short_window']
        self.long_window = config['long_window']
        self.tp_percentage = config['tp_percentage']
        self.sl_percentage = config['sl_percentage']
        self.leverage = config['leverage']
        self.investment_amount = config['investment_amount']
        self.interval = config['interval']
        self.period = config['period']
        self.sleep_time = config['sleep_time']
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.check_interval = config['check_interval']

        self.model = RandomForestClassifier()
        self.binance = ccxt.binanceusdm({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })
        self.open_orders = []
        self.log_file = 'trade_log.csv'
        self.initialize_log_file()

    def initialize_log_file(self):
        """Initialize the CSV log file with headers."""
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'direction', 'order_size', 'entry_price', 'exit_price', 'stop_loss',
                             'take_profit'])

    def fetch_data(self):
        """Fetch historical data using yfinance."""
        print("Fetching historical data...")
        try:
            data = yf.download(self.symbol, period=self.period, interval=self.interval)
            print(f"Data fetched successfully for {self.symbol}.")
            return data
        except Exception as e:
            print(f"Failed to download data: {e}")
            return None

    def preprocess_data(self, data):
        """Preprocess data and calculate moving averages."""
        if data is None or data.empty:
            print("No data to preprocess.")
            return None

        print("Preprocessing data and calculating moving averages...")
        data['Short_MA'] = data['Close'].rolling(window=self.short_window).mean()
        data['Long_MA'] = data['Close'].rolling(window=self.long_window).mean()
        data.dropna(inplace=True)

        data['Signal'] = 0
        data.iloc[self.short_window:, data.columns.get_loc('Signal')] = np.where(
            data['Short_MA'].iloc[self.short_window:] > data['Long_MA'].iloc[self.short_window:], 1, 0)
        data['Position'] = data['Signal'].diff()

        print("Data preprocessing complete.")
        return data

    def train_model(self, data):
        """Train a machine learning model using historical data."""
        if data is None or data.empty:
            print("No data to train on.")
            return

        print("Training the machine learning model...")
        X = data[['Short_MA', 'Long_MA']]
        y = data['Signal']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")

    def predict_signal(self, data):
        """Predict the signal using the trained machine learning model."""
        if data is None or data.empty:
            print("No data available for prediction.")
            return None

        print("Predicting the next signal based on the latest data...")
        X = data[['Short_MA', 'Long_MA']].tail(1)
        prediction = self.model.predict(X)[0]
        print(f"Prediction complete. Signal: {'Buy' if prediction == 1 else 'Sell'}")
        return prediction

    def place_order(self, signal):
        """Place a futures order on Binance using ccxt."""
        if signal is None:
            print("No signal available to place an order.")
            return

        symbol_binance = self.symbol.replace("-", "/").replace("USD", "USDT")

        # Fetch the current market price
        ticker = self.binance.fetch_ticker(symbol_binance)
        current_price = ticker['last']

        # Fetch account balance and calculate the order size based on investment balance and leverage
        balance = self.binance.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)
        if usdt_balance < self.investment_amount:
            print("Not enough balance to place the order.")
            return

        # Calculate the order size based on the investment amount and leverage
        order_size = (self.investment_amount * self.leverage) / current_price

        # Determine the direction of the trade based on the signal
        direction = 'buy' if signal == 1 else 'sell'

        # Place the market order
        order = self.binance.create_market_order(symbol_binance, direction, order_size)
        print(f"Placed a {direction} order for {order_size} contracts of {symbol_binance} at price {current_price}")

        # Calculate stop loss and take profit prices
        stop_loss_price = current_price * (1 - self.sl_percentage / 100) if direction == 'buy' else \
            current_price * (1 + self.sl_percentage / 100)
        take_profit_price = current_price * (1 + self.tp_percentage / 100) if direction == 'buy' else \
            current_price * (1 - self.tp_percentage / 100)

        # Store the open order details for monitoring
        self.open_orders.append({
            'symbol': symbol_binance,
            'direction': direction,
            'order_size': order_size,
            'entry_price': current_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price
        })

        print(f"Stop Loss Price: {stop_loss_price}, Take Profit Price: {take_profit_price}")

    def monitor_trades(self):
        """Monitor open trades and close them if stop loss or take profit levels are hit."""
        for order in self.open_orders[:]:  # Iterate over a copy of the list to safely remove items
            current_price = self.binance.fetch_ticker(order['symbol'])['last']
            if order['direction'] == 'buy':
                if current_price <= order['stop_loss_price'] or current_price >= order['take_profit_price']:
                    self.close_order(order, current_price)
            else:
                if current_price >= order['stop_loss_price'] or current_price <= order['take_profit_price']:
                    self.close_order(order, current_price)

    def close_order(self, order, current_price):
        """Close an open order and sleep for the configured time."""
        direction = 'sell' if order['direction'] == 'buy' else 'buy'
        self.binance.create_market_order(order['symbol'], direction, order['order_size'])
        print(f"Closed {order['direction']} order for {order['order_size']} {order['symbol']} at price {current_price}")
        self.log_trade(order, current_price)
        self.open_orders.remove(order)
        print(f"Sleeping for {self.sleep_time} minutes")
        self.sleep_with_details(self.sleep_time)

    def sleep_with_details(self, sleep_time):
        """Sleep for the given time in minutes, with detailed minute-by-minute updates."""
        for minute in range(sleep_time):
            print(f"Sleeping: {minute + 1} minute(s) out of {sleep_time}")
            time.sleep(60)  # Sleep for 1 minute

    def log_trade(self, order, exit_price):
        """Log the trade details to the CSV file."""
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                order['symbol'],
                order['direction'],
                order['order_size'],
                order['entry_price'],
                exit_price,
                order['stop_loss_price'],
                order['take_profit_price']
            ])
        print(f"Logged trade: {order['direction']} {order['order_size']} {order['symbol']} at price {exit_price}")

    def run_strategy(self):
        """Run the complete strategy from data fetching to trading execution."""
        print("Starting the trading strategy...")
        data = self.fetch_data()
        data = self.preprocess_data(data)
        self.train_model(data)
        signal = self.predict_signal(data)
        print(f"Final Signal: {'Buy' if signal == 1 else 'Sell'}")
        self.place_order(signal)
        print("Strategy run complete.")
        while True:
            try:
                if not self.open_orders:
                    self.place_order(signal)
                self.monitor_trades()
            except Exception as e:
                print(f"An error occurred: {e}")
            print(f"Sleeping for {self.check_interval} seconds.")
            time.sleep(self.check_interval)


api_key = ''  # Enter your api key
secret = ''  # Your secret key

# Example usage
if __name__ == "__main__":
    config = {
        'symbol': "BTC-USD",  # Use BTC-USD for yfinance
        'short_window': 20,
        'long_window': 50,
        'tp_percentage': 0.02,
        'sl_percentage': 0.01,
        'leverage': 100,
        'investment_amount': 3,  # Investment amount in USDT, must be >= $100
        'interval': '1d',
        'period': '1y',
        'sleep_time': 15,  # Sleep time in minutes
        'check_interval': 20,  # Check interval in seconds
        'api_key': "mr8gHLHo1XQ7ScvFREIUCAR8WYL4md3gOYClh4OiYAhg9SdaUnfeB7M4dn7fz95T",
        'api_secret': "fAkFAr7Rok55R7JIyqtPPtmKp1NkDWtAwmvbQYQHegq8JI5pIyITBfj1ZTYGNCIg"
    }

    trading_bot = MovingAverageCrossoverML(config)
    trading_bot.run_strategy()
