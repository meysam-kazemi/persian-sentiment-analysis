"""src/preprocess.py"""

import re
from hazm import Normalizer, word_tokenize, Stemmer

# A list of stop words (you can expand it).
# hazm.stopwords_list() can also be used, but it might remove some useful keywords.
STOP_WORDS = set([
    'و', 'در', 'به', 'از', 'که', 'این', 'آن', 'یک', 'با', 'است', 'برای', 
    'تا', 'هم', 'نیز', 'من', 'تو', 'او', 'ما', 'شما', 'ایشان', 'هر',
    'همه', 'یک', 'دیگر', 'برخی', 'روی', 'زیر', 'بالا', 'پایین'
])

def clean_text(text):
    """
    A function to normalize, tokenize, remove stop words, and stem the text.
    """
    # 1. Normalize text
    normalizer = Normalizer()
    text = normalizer.normalize(text)
    
    # Remove unnecessary characters (like emojis, extra punctuation)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 2. Tokenize (split text into words)
    tokens = word_tokenize(text)
    
    # 3. Remove stop words
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    # 4. Stem words (optional but useful)
    stemmer = Stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

if __name__ == '__main__':
    # An example to test the function
    sample_text = "من از این محصول اصلا راضی نیستم و خیلی بی‌کیفیت بود!! 😠"
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
