import re
from nltk import RegexpTokenizer

stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
             'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
             'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
             'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'â€œ', '\x9d', 'â€']

contractions_dict = {
    "ain't": "am not; are not; is not; has not; have not",
    "aren't": "are not; am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"}


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, s)

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def replace_emoticons(emoticons_tweet):
    # """Replace the emoticons of the input string with the corresponding string"""
    # function to preprocess
    conv_dict_multi = {('>:]', ':-)', ':)', ':o)', ':]', ':3', ':c', ':>', '=]', '8)', '=)', ':}', ':^)', ':)', '|;-)',
                        '|-o):', '>^_^<', '<^!^>', '^/^', '(*^_^*)', '(^<^)', '(^.^)', '(^?^)', '(^?^)', '(^_^.)',
                        '(^_^)', '(^^)', '(^J^)', '(*^?^*)', '^_^', '(^-^)', '(?^o^?)', '(^v^)', '(^u^)', '(^?^)',
                        '( ^)o(^ )', '(^O^)', '(^o^)', '(^?^)', ')^o^('): 'happy',
                       ('>:[', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', ':[', ':{', '>.>', '<.<', '>.<', '(\'_\')',
                        '(/_;)', '(T_T)', '(;_;)', '(;_:)', '(;O;)', '(:_;)', '(ToT)', '(T?T)', '(>_<)', '>:\\', '>:/',
                        ':-/', ':-.', ':/', ':\\', '=/', '=\\', ':S'): 'sad',
                       ('>:D', ':-', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', '8-)',
                        ':-))'): 'good',
                       ('D:<', 'D:', 'D8', 'D;', 'D=', 'DX', 'v.v', '>:)', '>;)', '>:-)', ':\'-(', ' :\'-)', ':\')',
                        ':-||'): 'bad'}
    # Convert to the one-to-one dict
    conv_dict = {}
    for k, v in conv_dict_multi.items():
        for key in k:
            conv_dict[key] = v
            # Replace the emoticons
    for smiley, conv_str in conv_dict.items():
        emoticons_tweet = emoticons_tweet.replace(smiley, conv_str)
    return emoticons_tweet


# def preprocess_tweet(words):
#     #     tweet = str(tweet)
#     #     tweet = tweet.lower()
#     #     tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
#     #     tweet = re.sub(r'(<e>|</e>|<a>|</a>|\n)', '', tweet)
#     #     tweet = ''.join(ch for ch in tweet if ch not in punc)
#
#     # for words in raw_tweets:
#     words_filtered = []
#     words = expand_contractions(words)
#     # words = strip_html(words)
#     # remove URLs
#     words = re.sub('https?://[^\s]+', '', words)
#     # remove numbers
#     words = re.sub('\d+', '', words)
#     # words=re.sub(/([!?.]){2,}/){"#{$~[1]} <REPEAT>"} # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
#     words = re.sub(r'(<e>|</e>|<a>|</a>|\n|!|,|(|)|//)', '', words)
#     # tweet = ''.join(ch for ch in tweet if ch not in punc)
#     to_be_removed = ['#', '@', '(', ')', '.', '...', '“', '?', '!', '"', '$', '|', '*', '&', '%', ';', ':', '&amp', '-',
#                      '--', '”', 'â€\x9d', 'ðÿ‘\x8d']
#     for prohibited_symbol in to_be_removed:
#         words = words.replace(prohibited_symbol, ' ')
#         # text = ' '.join(text.split())
#
#     tokens = RegexpTokenizer(r'\w+').tokenize(words)
#     words = replace_emoticons(words)
#     for e in words.split():
#         e = e.lower()
#         if e.startswith('@') or e.startswith('#') or e in stops: continue
#         clean_word = PorterStemmer().stem(e)
#         if len(clean_word) > 2:
#             words_filtered.append(clean_word)
#
#             # words_filtered.append(e)
#
#     return words_filtered
