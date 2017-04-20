from importData import read_excel_file
import classifier


def word_feats(tweets):
    return dict([(tweet, True) for tweet in tweets])

if __name__ == '__main__':

    # The training data
    training_data = "Data/training-Obama-Romney-tweets.xlsx"

    # Read in the tweets and classes for each tweet
    obama_data, obama_labels = read_excel_file(training_data, 'Obama')
    romney_data, romney_labels = read_excel_file(training_data, 'Romney')

    print('NB Results')
    obama_nb = classifier.NBClassifier(obama_data, obama_labels, "Obama")
    obama_nb.cross_validation()
    print()
    romney_nb = classifier.NBClassifier(romney_data, romney_labels, "Romney")
    romney_nb.cross_validation()

    print()
    print('SVM Results')
    obama_svm = classifier.SVMClassifier(obama_data, obama_labels, "Obama")
    obama_svm.cross_validation()
    print()
    romney_svm = classifier.SVMClassifier(romney_data, romney_labels, "Romney")
    romney_svm.cross_validation()

    # Something wrong with romney positive recall and fscore
    # Also try to manually label
    # Maybe try stratified sampling

    # To test on new data
    # obama_nb.test(obama_test)
    # predicted_labels = obama_nb.predicted





