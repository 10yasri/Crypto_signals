import time
from crypto_data_fetcher import get_historical_signal
from fetch_news import get_sentiment_signal
from tech import get_technical_signal

def generate_final_signal():
    """Fetch all signals and generate a final decision."""
    historical_signal = get_historical_signal()
    sentiment_signal = get_sentiment_signal()
    technical_signal = get_technical_signal()

    print(f"\n📊 Historical Signal: {historical_signal}")
    print(f"📰 Sentiment & Whale Wallet Signal: {sentiment_signal}")
    print(f"📈 Technical Indicator Signal: {technical_signal}")

    if historical_signal == "BUY" and sentiment_signal == "BUY" and technical_signal == "BUY":
        return "STRONG BUY ✅"
    elif historical_signal == "SELL" and sentiment_signal == "SELL" and technical_signal == "SELL":
        return "STRONG SELL 🚨"
    else:
        return "NEUTRAL / HOLD ⚠️"

if __name__ == "__main__":
    print("\n🚀 Starting Real-Time Trading Signal Generator...")
    while True:
        final_signal = generate_final_signal()
        print(f"\n💡 FINAL SIGNAL: {final_signal}\n")
        time.sleep(60)  # Fetch new signals every 1 minute

