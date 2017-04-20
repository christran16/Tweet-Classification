import xlrd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from preprocessing import contractions_dict, stops, expand_contractions, replace_emoticons

stopwordsnltk = set(stopwords.words("english"))
punc = set(string.punctuation)


def read_excel_file(filename, candidate):
    try:
        wb = xlrd.open_workbook(filename)
    except:
        print("File not found.")

    tweets = []
    labels = []

    # Select worksheet by candidate name
    sheet = wb.sheet_by_name(candidate)
    num_rows = sheet.nrows

    for rownum in range(2, num_rows):
        try:
            tweet = sheet.cell(rownum, 3).value
            label = sheet.cell(rownum, 4).value

            # add a function for removing stop words
            tweet = preprocess_tweet(tweet)

            # Maybe ignore the mixed labels
            if label not in (1.0, -1.0, 0.0):
                continue
        except:
            print("Some error occurred at row number: ", ''.join(sheet.cell(rownum, 3).value))

        # Default label is 2.0 (mixed)
        if label not in (1.0, -1.0, 2.0, 0.0):
            label = 2.0

        tweets.append(tweet)
        labels.append(label)

    tweets = tfidf_transform(tweets)
    return tweets, labels


def preprocess_tweet(tweet):
    pstemmer = PorterStemmer()

    tweet = str(tweet)
    tweet = tweet.lower()
    tweet = pstemmer.stem(tweet)
    tweet = re.sub('https?://[^\s]+', '', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'(<e>|</e>|<a>|</a>|\n|!|,|(|)|//)', '', tweet)
    to_be_removed = ['#', '@', '(', ')', '.', '...', '“', '?', '!', '"', '$', '|', '*', '&', '%', ';', ':', '&amp', '-',
                     '--', '”', 'â€\x9d', 'ðÿ‘\x8d']
    for prohibited_symbol in to_be_removed:
        tweet = tweet.replace(prohibited_symbol, '')
    tweet = replace_emoticons(tweet)
    tweet = expand_contractions(tweet)
    tweet = ''.join(ch for ch in tweet if ch not in punc)
    tweet = remove_stopwords(tweet)

    return tweet


def remove_stopwords(tweet):
    tmp = []
    tweet = tweet.split()
    for i in tweet:
        if i.lower() not in stops:
            tmp.append(i)

    return " ".join(tmp)


def tfidf_transform(data):
    count_vect = CountVectorizer()
    data_train_counts = count_vect.fit_transform(data)

    tfidf_transformer = TfidfTransformer()
    data_train_tfidf = tfidf_transformer.fit_transform(data_train_counts)

    return data_train_tfidf
