

# 📦 Flipkart Review Scraper & Sentiment Analyzer

This project scrapes customer reviews of the **Apple iPhone 13 (Green, 128 GB)** from Flipkart, performs **sentiment analysis** using VADER, and visualizes review text using a **WordCloud**.

---

## 🚀 Features

- ✅ Scrapes reviews from multiple pages on Flipkart using `requests` + `BeautifulSoup`
- ✅ Cleans and tokenizes text using `nltk`
- ✅ Performs POS tagging and lemmatization
- ✅ Applies **VADER sentiment analysis**
- ✅ Categorizes sentiment into **Positive**, **Negative**, and **Neutral**
- ✅ Visualizes results with:
  - ☁️ **Word Cloud** of most frequent review words
  - 🥧 **Pie Chart** of sentiment distribution

---

## 🛠️ Tech Stack

- Python 3.x
- BeautifulSoup
- requests
- pandas
- nltk
- vaderSentiment
- matplotlib
- wordcloud

---

## 📊 Output

Below are the visualizations generated from the scraped Flipkart reviews:

### ☁️ Word Cloud

A word cloud showing the most frequent words used in the customer reviews after cleaning and lemmatization.

![Word Cloud](images/wordcloud.png)

---

### 🥧 Sentiment Distribution (Pie Chart)

A pie chart showing the proportion of reviews that are Positive, Neutral, or Negative based on VADER sentiment scores.

![Pie Chart](images/piechart.png)



