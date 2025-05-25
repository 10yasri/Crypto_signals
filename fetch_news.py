import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# === Get large BTC transactions
def get_large_btc_transactions():
    url = "https://www.blockchain.com/btc/unconfirmed-transactions"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        transactions = soup.find_all("div", class_="sc-1g6z4xm-0")

        whale_transactions = []
        for tx in transactions:
            try:
                amount_btc = float(tx.find_all("span")[-1].text.split(" BTC")[0])
                if amount_btc > 100:
                    tx_hash = tx.find("a").text
                    whale_transactions.append((tx_hash, amount_btc))
            except:
                continue
        return whale_transactions
    return None

# === Get large ETH transactions
def get_large_eth_transactions():
    url = "https://etherscan.io/txs"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        transactions = soup.find_all("tr")[1:20]

        whale_transactions = []
        for tx in transactions:
            cols = tx.find_all("td")
            if len(cols) > 6:
                try:
                    eth_value = float(cols[6].text.replace(",", "").strip())
                    if eth_value > 500:
                        tx_hash = cols[0].text.strip()
                        whale_transactions.append((tx_hash, eth_value))
                except:
                    continue
        return whale_transactions
    return None

# === Get news from CryptoPanic
def get_crypto_news():
    API_KEY = "f002b4086f993753a7a7d82ced2236eafafe3a07"  # Replace with your key
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true"
    response = requests.get(url)

    if response.status_code == 200:
        news_data = response.json()
        return [(post["title"], post["url"]) for post in news_data["results"]]
    return None

# === Sentiment using VADER
def analyze_news_sentiment_vader(news_title):
    scores = sid.polarity_scores(news_title)
    compound = scores['compound']
    if compound >= 0.05:
        return "Bullish"
    elif compound <= -0.05:
        return "Bearish"
    else:
        return "Neutral"

# === Assess market impact based on keywords
def assess_market_impact(news_title):
    keywords = ["SEC", "lawsuit", "ban", "hacked", "ETF", "adoption", "bankruptcy"]
    impact = "Low"
    for keyword in keywords:
        if keyword.lower() in news_title.lower():
            impact = "High"
    return impact

# === Console display for quick view
def display_thesis_friendly_news_table():
    crypto_news = get_crypto_news()
    print("\n📝 Thesis-Friendly Crypto News Table:\n")
    if crypto_news:
        news_table = []
        for title, link in crypto_news[:8]:
            sentiment = analyze_news_sentiment_vader(title)
            impact = assess_market_impact(title)
            short_title = title[:40] + "..." if len(title) > 40 else title
            news_table.append([short_title, sentiment, impact, "[Link]"])

        headers = ["Headline", "Sentiment", "Impact", "Source"]
        print("{:<45} {:<10} {:<8} {:<6}".format(*headers))
        print("-" * 75)
        for row in news_table:
            print("{:<45} {:<10} {:<8} {:<6}".format(*row))
    else:
        print("No news found.")

# === Matplotlib pop-up table (compact)
def plot_thesis_crypto_news_table():
    crypto_news = get_crypto_news()
    if not crypto_news:
        print("No news to plot.")
        return

    crypto_news = crypto_news[:8]  # Limit number of rows

    table_data = []
    for title, link in crypto_news:
        sentiment = analyze_news_sentiment_vader(title)
        impact = assess_market_impact(title)
        short_title = title[:25] + "..." if len(title) > 25 else title
        table_data.append([short_title, sentiment, impact])

    column_labels = ["Headline", "Sentiment", "Impact"]

    num_rows = len(table_data)
    fig_height = max(3.5, 0.45 * num_rows)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        cellLoc='left',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.title("Crypto News Summary", fontsize=11, pad=2)
    plt.tight_layout()
    plt.savefig("thesis_crypto_news_compact.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Compact plot saved as 'thesis_crypto_news_compact.png'")

# === Main script
def main():
    print("\n🚀 Fetching latest crypto data...\n")

    btc_whales = get_large_btc_transactions()
    print("📈 Bitcoin Whale Transactions (over 100 BTC):")
    if btc_whales:
        for tx_hash, amount in btc_whales:
            print(f"TX: {tx_hash} | Amount: {amount:.2f} BTC")
    else:
        print("No significant BTC transactions found.")

    print("\n📉 Ethereum Whale Transactions (over 500 ETH):")
    eth_whales = get_large_eth_transactions()
    if eth_whales:
        for tx_hash, amount in eth_whales:
            print(f"TX: {tx_hash} | Amount: {amount:.2f} ETH")
    else:
        print("No significant ETH transactions found.")

    display_thesis_friendly_news_table()
    plot_thesis_crypto_news_table()
    print("\n✅ Data fetching completed!\n")

# === Run It
if __name__ == "__main__":
    main()
