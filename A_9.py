import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""Q1.1"""

paragraph = "Technology has greatly revolutionized the face of education, making it more accessible and interactive than ever. With the advent of online platforms, students have the ability to attend virtual classes, have access to massive libraries of information, and work together with peers from around the world. Interactive learning tools such as simulations, educational games, and AI-powered tutors facilitate learning and mastery of complex ideas. Additionally, instructors are helped by automated grading systems and data-based insights that facilitate personalized learning experiences. Although obstacles such as digital inequality still exist, the integration of technology into education continues to close gaps and open up new possibilities for students globally."
lowercase_text = paragraph.lower()
no_punct_text = re.sub(r'[^\w\s]', '', lowercase_text)
print(no_punct_text[:100], "...")

"""Q1.2

"""

nltk.download('punkt_tab')
sentences = sent_tokenize(paragraph)
words = word_tokenize(no_punct_text)

"""Q1.3

"""

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

"""Q1.4"""

word_freq = Counter(filtered_words)
for word, count in word_freq.most_common(10):
    print(f"{word}: {count}")

"""Q2.2"""

from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
porter = PorterStemmer()
lancaster = LancasterStemmer()

"""Q2.3

"""

lemmatizer = WordNetLemmatizer()

"""Q2.4"""

for word in filtered_words[:10]:
    porter_result = porter.stem(word)
    lancaster_result = lancaster.stem(word)
    lemma_result = lemmatizer.lemmatize(word)
    print(f"{word}\t{porter_result}\t{lancaster_result}\t{lemma_result}")

"""Q3.2"""

long_words = re.findall(r'\b\w{6,}\b', paragraph)
print(long_words[:15])
numbers = re.findall(r'\d+\.?\d*', paragraph)
print(numbers)
cap_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', paragraph)
print(cap_words)

"""Q3.3"""

alpha_only = re.findall(r'\b[a-zA-Z]+\b', paragraph)
print(alpha_only[:15])
vowel_words = re.findall(r'\b[aeiouAEIOU][a-zA-Z]*\b', paragraph)
print(vowel_words)

"""Q4.1

"""

text_sample = paragraph + " You can reach the founder at jane.doe@startupworld.org. Visit https://www.entrepreneurlife.com for more. Call at 555-123-4567 or +44 7700 900123. The startup's valuation is $3.14 million."

"""Q4.2"""

def custom_tokenize(text):
    text_temp = re.sub(r"(\w+)'(\w+)", r"\1'\2", text)
    text_temp = re.sub(r"(\w+)-(\w+)(-(\w+))?", lambda m: m.group(0).replace("-", "HYPHEN"), text_temp)
    text_temp = re.sub(r"(\d+)\.(\d+)", lambda m: m.group(0).replace(".", "DECIMAL"), text_temp)
    text_temp = re.sub(r'[^\w\s]', ' ', text_temp)
    tokens = text_temp.split()
    tokens = [token.replace("HYPHEN", "-").replace("DECIMAL", ".") for token in tokens]

    return tokens
custom_tokens = custom_tokenize(text_sample)
print(custom_tokens[:15])

"""Q4.3"""

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
email_replaced = re.sub(email_pattern, '<EMAIL>', text_sample)
url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
url_replaced = re.sub(url_pattern, '<URL>', email_replaced)

phone_pattern = r'(\+\d{1,3}\s\d{10}|\d{3}-\d{3}-\d{4})'
phone_replaced = re.sub(phone_pattern, '<PHONE>', url_replaced)

print(phone_replaced)