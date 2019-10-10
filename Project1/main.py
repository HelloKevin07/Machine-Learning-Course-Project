from Utils import *


def main():
    # Extracting data from zip
    print('### Extracting data from zip, please wait...')
    unzip_file('20_newsgroups.zip', '')
    src_path = '20_newsgroups'
    dst_path = ''
    print('\n### Dividing training data and testing data...')
    data_preprocessing(src_path, dst_path)
    classList = os.listdir('Train')
    create_csv()
    print("\n### Start training ###")
    NaiveBayesClassifier('google-10000-english-usa-no-swears.txt', classList, 'train.csv', status='Train')
    print("\n### Start testing ###")
    print("\n### Summary for training set ###")
    NaiveBayesClassifier('google-10000-english-usa-no-swears.txt', classList, 'train.csv', status='Test')
    print("\n### Summary for testing set ###")
    NaiveBayesClassifier('google-10000-english-usa-no-swears.txt', classList, 'test.csv', status='Test')


if __name__ == '__main__':
    main()
