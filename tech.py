# --- Suppress Warnings and TensorFlow Logs ---
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING logs

# --- Required Libraries ---
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from telethon.sync import TelegramClient

# --- CONFIGURATION ---
INTERVAL = '1h'
LIMIT = 500
TELEGRAM_MESSAGES_LIMIT = 100
TWITTER_TWEETS_LIMIT = 100
CRYPTOPANIC_POSTS_LIMIT = 100

# --- API Keys ---
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAGHTzQEAAAAAyq8KptjJEtnzDbPEGETHw3PhbYU%3DojAkqO4Q0i60Me3Auj6OH1ppuiXlw3KBGmfayz1IFCFx6UFahM'
TELEGRAM_API_ID = 26957173
TELEGRAM_API_HASH = '7d1f8efaa28ae3a5642f2ea0384fde54'
TELEGRAM_USERNAME = 'Tanya1421'
TELEGRAM_CHANNEL_NAME = '@cryptojargon69'
CRYPTOPANIC_API_KEY = '3cc9615899b4e156fe51a761b055921d1a5df31d'

# --- Twitter Fetch ---
def fetch_tweets_v2(keywords, count=100):
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        tweets = client.search_recent_tweets(query=keywords, max_results=min(count, 100), tweet_fields=['text'])
        return [tweet.text for tweet in tweets.data] if tweets.data else []
    except tweepy.TooManyRequests:
        print("Twitter API rate limit exceeded.")
        return []

# --- Sentiment Analysis ---
def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(text)['compound'] for text in texts]

# --- Telegram Fetch ---
def fetch_telegram_messages():
    with TelegramClient(TELEGRAM_USERNAME, TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:
        messages = [msg.text for msg in client.iter_messages(TELEGRAM_CHANNEL_NAME, limit=TELEGRAM_MESSAGES_LIMIT) if msg.text]
    return messages

# --- CryptoPanic Fetch ---
def fetch_cryptopanic_news(limit=100):
    url = f'https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&filter=hot&public=true'
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch CryptoPanic news.")
        return []
    articles = response.json().get("results", [])[:limit]
    return [item['title'] + " " + (item['slug'] or "") for item in articles]

# --- Binance Price Fetch ---
def fetch_data(symbol, interval=INTERVAL, limit=LIMIT):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch price data from Binance.")
        return None
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# --- Prepare LSTM Data ---
def prepare_data(df, sentiment_scores, steps=60):
    sentiment_scores = ([0] * (len(df) - len(sentiment_scores)) + sentiment_scores)[-len(df):]
    df['sentiment'] = pd.Series(sentiment_scores, index=df.index)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close', 'sentiment']])

    x, y = [], []
    for i in range(steps, len(scaled)):
        x.append(scaled[i - steps:i])
        y.append(scaled[i, 0])
    return np.array(x), np.array(y), scaler

# --- Build Model ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Main Flow ---
def main():
    symbol = input("Enter the symbol (e.g. BTCUSDT): ").upper()
    df = fetch_data(symbol)
    if df is None:
        return

    current_price = df['close'].iloc[-1]

    print("✅ Fetching Twitter sentiment...")
    twitter_scores = analyze_sentiment(fetch_tweets_v2("cryptocurrency", TWITTER_TWEETS_LIMIT))

    print("✅ Fetching Telegram sentiment...")
    telegram_scores = analyze_sentiment(fetch_telegram_messages())

    print("✅ Fetching CryptoPanic sentiment...")
    panic_scores = analyze_sentiment(fetch_cryptopanic_news(CRYPTOPANIC_POSTS_LIMIT))

    combined_scores = twitter_scores + telegram_scores + panic_scores

    # Add sentiment column aligned with df
    sentiment_scores = ([0] * (len(df) - len(combined_scores)) + combined_scores)[-len(df):]
    df['sentiment'] = pd.Series(sentiment_scores, index=df.index)

    

    print("📊 Preparing LSTM training data...")
    x, y, scaler = prepare_data(df, combined_scores, steps=60)

    print("🔧 Training model...")
    model = build_lstm_model((x.shape[1], x.shape[2]))
    history = model.fit(x, y, epochs=10, batch_size=32, verbose=1)

    # 1. Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("📈 Predicting next price...")
    last_seq = df[['close', 'sentiment']].tail(60).values
    last_seq_scaled = scaler.transform(last_seq)
    last_seq_scaled = np.reshape(last_seq_scaled, (1, 60, 2))
    predicted = model.predict(last_seq_scaled)
    predicted_price = scaler.inverse_transform(
        np.concatenate((predicted, np.zeros_like(predicted)), axis=1)
    )[:, 0][0]

    # --- Trading Zone Calculation ---
    buy_zone_low = df['low'].tail(30).min()
    buy_zone_high = df['low'].tail(30).quantile(0.25)
    target_1 = current_price * 1.0015
    target_2 = current_price * 1.0035
    target_3 = current_price * 1.0060
    stop_loss = current_price * 0.9955

    # 2. Plot Buy Zone and Stop Loss on Price Chart
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Price', color='blue')
    plt.axhspan(buy_zone_low, buy_zone_high, color='green', alpha=0.3, label='Buy Zone')
    plt.axhline(stop_loss, color='red', linestyle='--', label='Stop Loss')
    plt.title(f'{symbol} Price with Buy Zone and Stop Loss')
    plt.legend()
    plt.show()

    # 3. Correlation Heatmap (Price & Sentiment)
    corr = df[['close', 'sentiment']].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f'{symbol} Correlation between Price and Sentiment')
    plt.show()

    # --- Output Report ---
    print(f"""
🔮 Asset: ${symbol}
📉 Current Price: ${current_price:.2f}

📌 Suggested Buy Zone:
   - ${buy_zone_low:.2f}
   - ${buy_zone_high:.2f}

🎯 Target Prices:
   1. ${target_1:.2f}
   2. ${target_2:.2f}
   3. ${target_3:.2f}

🛡 Stop Loss: ${stop_loss:.2f}

🤖 AI Forecasted Next Price: ${predicted_price:.2f}
""")

    # --- Plot Actual vs Predicted Price ---
    predicted_sequence = model.predict(x)
    predicted_prices = scaler.inverse_transform(
        np.concatenate((predicted_sequence, np.zeros((len(predicted_sequence), 1))), axis=1)
    )[:, 0]

    actual_prices = df['close'].values[-len(predicted_prices):]

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(predicted_prices):], actual_prices, label='Actual Price', color='blue', linewidth=2)
    plt.plot(df.index[-len(predicted_prices):], predicted_prices, label='Predicted Price', color='red', linestyle='--', linewidth=2)
    plt.title(f'{symbol} - Actual vs Predicted Price (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Entry Point ---
if __name__ == '__main__':
    main()
