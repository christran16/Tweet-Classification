from importData import read_excel_file
import classifier


def word_feats(tweets):
    return dict([(tweet, True) for tweet in tweets])

if __name__ == '__main__':

    # The training data
    training_data = "training-Obama-Romney-tweets.xlsx"

    # Read in the tweets and classes for each tweet
    obama_data, obama_labels = read_excel_file(training_data, 'Obama')
    romney_data, romney_labels = read_excel_file(training_data, 'Romney')

    obama_nb = classifier.NBClassifier("Obama")
    obama_nb.cross_validation(obama_data, obama_labels)

    # To test on new data
    # obama_nb.train(obama_data, obama_labels)
    # obama_nb.test(obama_test)
    # predicted_labels = obama_nb.predicted





