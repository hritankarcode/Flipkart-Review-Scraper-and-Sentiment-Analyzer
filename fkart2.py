import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import nltk



# Base URL
base_url = "https://www.flipkart.com/apple-iphone-13-green-128-gb/product-reviews/itm18a55937b2607?pid=MOBGC9VGSU9DWGJZ"
pages = 100
reviews_per_page = 10  # Desired number of reviews per page
reviews = []

# Loop through each page
for i in range(1, pages + 1):
    print(f"Scanning page {i}")
    
    # Construct the URL
    url = f"{base_url}&page={i}&sort=recency_desc&pagesize=100"  # Flipkart might ignore the page size parameter

    # Make the request with error handling
    while True:
        response = requests.get(url)
        if response.status_code == 503:
            print("503 Error. Retrying after 5 seconds...")
            time.sleep(5)
        elif response.status_code == 200:
            break

    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract reviews
        review_containers = soup.find_all("div", class_="ZmyHeo")

        # Collect exactly 10 reviews from the current page
        collected_reviews = 0
        for review in review_containers:
            if collected_reviews >= reviews_per_page:
                break
            review_text = review.get_text(strip=True)
            reviews.append(review_text)
            collected_reviews += 1

        print(f"Collected {collected_reviews} reviews from page {i}")

# Print total reviews collected
print(f"Total reviews collected: {len(reviews)}")

df=pd.DataFrame()
df["reviews"]=reviews
def clean(text):
# Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', str(text))
    return text

# Cleaning the text in the review column
df['Cleaned Reviews'] = df['reviews'].apply(clean)
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    #print(tags)
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
          newlist.append(tuple([word, pos_dict.get(tag[0])]))
          #print(tag[0])
          #print(pos_dict.get(tag[0]))
    return newlist 

df['POS tagged'] = df['Cleaned Reviews'].apply(token_stop_pos)


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
     if not pos:
        lemma = word
        lemma_rew = lemma_rew + " " + lemma
     else:
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        lemma_rew = lemma_rew + " " + lemma
    return lemma_rew
df['Lemma'] = df['POS tagged'].apply(lemmatize)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# function to calculate vader sentiment
def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

df['Sentiment'] = df['Lemma'].apply(vadersentimentanalysis)

# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound < 0 :
        return 'Negative'
    else:
        return 'Neutral'
df['Analysis'] = df['Sentiment'].apply(vader_analysis)
print(df.head)

vader_counts = df['Analysis'].value_counts()
print(vader_counts)


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(" ".join(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df.Lemma)

plt.figure(figsize=(15,7))

plt.subplot(1,3,2)
plt.title("Reviews Analysis")
plt.pie(vader_counts.values, labels = vader_counts.index, explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)
plt.show()
