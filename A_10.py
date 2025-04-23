
# Q1

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

paragraph = "Technology has greatly revolutionized the face of education, making it more accessible and interactive than ever. With the advent of online platforms, students have the ability to attend virtual classes, have access to massive libraries of information, and work together with peers from around the world. Interactive learning tools such as simulations, educational games, and AI-powered tutors facilitate learning and mastery of complex ideas. Additionally, instructors are helped by automated grading systems and data-based insights that facilitate personalized learning experiences. Although obstacles such as digital inequality still exist, the integration of technology into education continues to close gaps and open up new possibilities for students globally."

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

cleaned_text = clean_text(paragraph)

sentences = sent_tokenize(paragraph)
words_nltk = word_tokenize(paragraph)
words_python = paragraph.split()

stop_words = set(stopwords.words('english'))
words_no_stopwords = [word for word in words_nltk if word.lower() not in stop_words]


cleaned_words = [word.lower() for word in words_no_stopwords if word.isalpha()]
word_freq = Counter(cleaned_words)

for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")

"""Q2"""

import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

alphabetic_words = re.findall(r'\b[a-zA-Z]+\b', paragraph)

words_no_stop = [word for word in alphabetic_words if word.lower() not in stop_words]

ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in words_no_stop]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_no_stop]

for original, stemmed, lemmatized in zip(words_no_stop[:8], stemmed_words[:8], lemmatized_words[:8]):
    print(f"Original: {original:<15} | Stemmed: {stemmed:<15} | Lemmatized: {lemmatized:<15}")

"""Q3"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

texts = [
    "Young entrepreneurs in India raise $200 million collectively in Q1 2025, with fintech and climate tech leading the funding charts.",
    "Global accelerator programs expand outreach in Africa and Southeast Asia, aiming to support 10,000 new startups by 2026.",
    "Remote-first startups report 35% faster product development cycles, highlighting the growing efficiency of decentralized teams in entrepreneurship."
]



count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(texts)

feature_names = count_vectorizer.get_feature_names_out()
for i, text in enumerate(texts):
    print(f"\nText {i+1}:")
    nonzero_indices = bow_matrix[i].nonzero()[1]
    counts = zip(nonzero_indices, bow_matrix[i].toarray()[0][nonzero_indices])
    for idx, count in counts:
        print(f"  {feature_names[idx]}: {count}")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
feature_names = tfidf_vectorizer.get_feature_names_out()

for i, text in enumerate(texts):
    print(f"\nText {i+1}:")
    tfidf_scores = tfidf_matrix[i].toarray()[0]
    top_indices = np.argsort(tfidf_scores)[-3:][::-1]
    print("Top keywords:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {tfidf_scores[idx]:.4f}")

print("Interpretation:")
if i == 0:
    print("  Words like 'entrepreneurs', 'India', and 'fintech' indicate this news is about")
    print("  a surge in startup funding, especially in high-growth sectors like fintech and climate tech.")
    print("  These terms highlight the scale and direction of investment trends in Q1 2025.")
elif i == 1:
    print("  Terms like 'accelerator', 'Africa', and 'startups' show this story focuses on")
    print("  global support for entrepreneurship in emerging markets.")
    print("  These keywords reflect expansion initiatives by accelerator programs.")
else:
    print("  'Remote-first', 'startups', and 'development cycles' reveal that this news is about")
    print("  how decentralized teams are driving faster innovation.")
    print("  The words point to operational efficiency in modern entrepreneurial setups.")

"""Q4"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tech_text1 = """Artificial Intelligence (AI) and Machine Learning: These technologies are employed to analyze vast datasets, improve investment predictions, automate signal discovery, and enhance portfolio construction. AI helps reduce information asymmetry and provides more accurate forecasts than traditional human analysis."""

tech_text2 = """High-Frequency Trading (HFT) Technology: This involves ultra-fast algorithmic trading systems that execute large numbers of trades at very high speeds, leveraging low-latency networks and advanced coding to capitalize on small market inefficiencies. It requires specialized infrastructure and technical expertise."""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

ai_tokens = preprocess_text(tech_text1)
hft_tokens = preprocess_text(tech_text2)

ai_set = set(ai_tokens)
hft_set = set(hft_tokens)
intersection = ai_set.intersection(hft_set)
union = ai_set.union(hft_set)
jaccard_sim = len(intersection) / len(union)

print(f"\nJaccard Similarity: {jaccard_sim:.4f}")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([tech_text1, tech_text2])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print(f"Cosine Similarity: {cosine_sim:.4f}")

print("Cosine Similarity provides better insights")

"""Q5"""

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
review = """Tried Claude for a few weeks now and honestly, I'm impressed. It's super helpful for brainstorming and writing—feels more natural than ChatGPT sometimes. Love how it keeps up with long convos and remembers context better, so it doesn't lose track like others do. Coding help is solid, though not perfect, but it's made my workflow way faster. Only downside is the usage limits can be annoying, and sometimes it over-apologizes or refuses stuff for no reason. Still, for most tasks, Claude just "gets it." """

blob = TextBlob(review)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

print(f"polarity: {polarity:.4f} ")
print(f"subjectivity: {subjectivity:.4f}")

if polarity > 0.1:
    sentiment = "Positive"
elif polarity < -0.1:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print(f"Review classified as: {sentiment}")

if sentiment == "Positive":
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    cleaned_review = clean_text(review)

    stop_words = set(stopwords.words('english'))
    review_words = word_tokenize(cleaned_review)
    filtered_words = [word for word in review_words
                     if word.lower() not in stop_words and len(word) > 2]

    filtered_text = ' '.join(filtered_words)

    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=50,
                          contour_width=1).generate(filtered_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Headphones Review')
    plt.tight_layout()
    plt.show()

"""Q6"""

!pip install tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

"""Q6.1"""

paragraph = "The sky was blue, and the sun was beaming brightly, heating everything it touched. Birds flew across the open blue sky, singing happy melodies that brought life to the morning. A gentle breeze blew through the trees, causing the leaves to rustle and dance effortlessly. It carried the sweet fragrance of flowers and grass, enhancing the sense of tranquility in the air. Individuals strolled slowly along the peaceful street, basking in the weather, smiling at strangers, and appreciating the peaceful beauty of the day. Everything seemed to be still, yet alive at the same time."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([paragraph])

sequences = []
words = paragraph.split()
for i in range(1, len(words)):
    seq = words[:i+1]
    tokenized_seq = tokenizer.texts_to_sequences([' '.join(seq)])[0]
    sequences.append(tokenized_seq)

padded = pad_sequences(sequences)
print("Sample of padded sequences:")
print(padded[:3])

X = padded[:, :-1]
y = padded[:, -1]

"""Q6.2"""

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                   output_dim=50,
                   input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("\nModel summary:")
model.summary()

"""Q6.3"""

print("\nTraining the model...")
model.fit(X, y, epochs=50, verbose=1)

def generate_text(seed_text, next_words, model, max_sequence_len):
    result = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
        result += " " + output_word

    return result

seed_words = ["Steve", "finance", "billion"]
for seed in seed_words:
    generated = generate_text(seed, 10, model, X.shape[1])
    print(f"\nSeed: '{seed}'")
    print(f"Generated text: '{generated}'")