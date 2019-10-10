import os
import shutil
import zipfile
import random
import csv
import collections
import math
from nltk.stem import PorterStemmer
# ps = PorterStemmer()

# a list of common stopwords
stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
 'each', 'few', 'for', 'from', 'further',
 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's",
 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's",
 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
 "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' 'ourselves', 'out', 'over', 'own',
 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
 'than', 'that',"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
 "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
 "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",'will', 'with', "won't", 'would', "wouldn't",
 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', '1st', '2nd', '3rd',
 '4th', '5th', '6th', '7th', '8th', '9th', '10th']


# unzip zip file
def unzip_file(src, dst):
    r = zipfile.is_zipfile(src)
    if r:
        fz = zipfile.ZipFile(src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst)
        print('Completed!')
        return True
    else:
        print(src + ' is not a zip file')
        return False


# get training and testing data
# we assume empty folders 'Train' and 'Test' have already been created
# param 'dst' is the code directory
def data_preprocessing(src, dst):

    # get the label list of data
    folderList = os.listdir(src)

    # create empty folder for each label
    for label in folderList:
        dst_train = os.path.join(dst, 'Train', label)
        dst_test = os.path.join(dst, 'Test', label)
        if not os.path.exists(dst_train):
            os.mkdir(dst_train)
        if not os.path.exists(dst_test):
            os.mkdir(dst_test)

    # randomly split the data into training and testing sets
    for label in folderList:
        print('Begin to split ' + label + '...')
        sample_namelist = os.listdir(os.path.join(src, label))
        num_sample = len(sample_namelist)
        sample_ind = list(range(0, num_sample))
        random.shuffle(sample_ind)

        # split the data into Train and Test (half & half)
        bound = num_sample // 2
        for i in range(0, bound):
            src_cp = os.path.join(src, label, sample_namelist[sample_ind[i]])
            dst_cp = os.path.join(dst, 'Train', label)
            shutil.copy(src_cp, dst_cp)
        for i in range(bound, num_sample):
            src_cp = os.path.join(src, label, sample_namelist[sample_ind[i]])
            dst_cp = os.path.join(dst, 'Test', label)
            shutil.copy(src_cp, dst_cp)
        print('---Completed!')


# load a dictionary as lexicon in this case
def get_dictionary(src):
    print('Loading dictionary...')
    with open(src, 'r') as f:
        dict = []
        for line in f.readlines():
            line = line.strip()
            if not len(line):
                continue
            dict.append(line)
    print('---Completed! Total ' + str(len(dict)) + ' word(s)')
    return dict, len(dict)


# remove some obviously useless words (optional)
def refine_dictionary(diction):
    print('Refining dictionary...')
    if isinstance(diction, list):
        del_waitinglist = []
        for word in diction:
            if (word in stopwords) or (len(word) < 3):
                del_waitinglist.append(word)
        for word in del_waitinglist:
            diction.remove(word)
    print('---Completed! Total ' + str(len(diction)) + ' word(s)')
    return diction, len(diction)


# read text as string
def read_text(src):
    f = open(src, 'r', errors='ignore')  # (encoding='gb18030'
    text = f.read()
    f.close()
    return text


# count all useful words and their freq
# total_num is the number of useful words in this text
def count_wordfreq(diction, text):
    if isinstance(text, str):
        tmp = ""
        # remove useless chars
        for w in text:
            if w.isalpha():
                tmp += w.lower()
            else:
                tmp += ' '

        tmp_sp = tmp.split()

        # for i in range(0, len(tmp_sp)):
        #     tmp_sp[i] = ps.stem(tmp_sp[i])

        wordfreq_vec = []
        for token in diction:
            wordfreq_vec.append(collections.Counter(tmp_sp)[token])
        total_num = sum(wordfreq_vec)
        return wordfreq_vec, total_num


# create sample-path-to-label mappings of training and testing data as csv file
def create_csv():

    headers = ['path', 'label']
    train_mappings = []
    for label in os.listdir('Train'):
        label_src = os.path.join('Train', label)
        for sample in os.listdir(label_src):
            sample_src = os.path.join(label_src, sample)
            train_mappings.append((sample_src, label))

    with open('train.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(train_mappings)

    test_mappings = []
    for label in os.listdir('Test'):
        label_src = os.path.join('Test', label)
        for sample in os.listdir(label_src):
            sample_src = os.path.join(label_src, sample)
            test_mappings.append((sample_src, label))

    with open('test.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(test_mappings)


# read train.csv or test.csv
def read_csv(src):
    X_path = []
    Y = []
    with open(src) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            X_path.append(row[0])
            Y.append(row[1])
    return X_path, Y


# load Pxy and Py
def load_model():
    Pxy = []
    Py = []
    with open('Pxy.csv') as f:
        f_csv = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in f_csv:
            Pxy.append(row)

    with open('Py.csv') as f:
        f_csv = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in f_csv:
            Py.append(row[1])

    return Pxy, Py


# Naive Bayes Classifier
def NaiveBayesClassifier(dict_src, classList, csv_src, status='Train'):
    # labels follow the order of classList

    # create class to index mappings
    indList = dict(zip(classList, range(0, len(classList))))

    # lexicon refinement
    D, num_lexicon = get_dictionary(dict_src)
    D, num_lexicon = refine_dictionary(D)

    num_topic = len(classList)

    if status == 'Train':
        # get training data path and label from train.csv
        X_path, Y = read_csv(csv_src)

        alpha = 0.00001
        Pxy = [[alpha for i in range(num_lexicon)] for j in range(num_topic)]
        Py = [0 for i in range(num_topic)]
        for i, y in enumerate(Y):
            Py[indList[y]] += 1
            x = read_text(X_path[i])
            wordfreq_vec, total_num = count_wordfreq(D, x) # wordfreq_vec (1, num_lexicon)
            for k in range(num_lexicon):
                Pxy[indList[y]][k] += wordfreq_vec[k]
            print("Sample " + str(i) + "/10000 finished!", end='\r', flush=True)

        sumPy = sum(Py)
        for i in range(num_topic):
            Py[i] = (i, Py[i] / sumPy)

        for i in range(num_topic):
            sumRow = sum(Pxy[i])
            for j in range(num_lexicon):
                Pxy[i][j] = Pxy[i][j] / (sumRow + 1)

        with open('Pxy.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            # f_csv.writerow(headers)
            f_csv.writerows(Pxy)

        with open('Py.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            # f_csv.writerow(['topic'])
            f_csv.writerows(Py)

    elif status == 'Test':
        print("Begin to load NB model...")
        summaryScore = [0 for i in range(len(classList))]
        acc = 0

        # load Pxy and Py from Pxy.csv and Py.csv
        Pxy, Py = load_model()
        print("---Completed")

        # get testing data path and label from test.csv
        X_path, Y = read_csv(csv_src)

        # pre-processing for every text in testing set
        for i, x_path in enumerate(X_path):
            y = Y[i]
            text = read_text(x_path)
            tmp = ""
            # remove useless chars
            for w in text:
                if w.isalpha():
                    tmp += w.lower()
                else:
                    tmp += ' '

            tmp_sp = tmp.split()

            # for t in range(0, len(tmp_sp)):
            #     tmp_sp[t] = ps.stem(tmp_sp[t])

            # create word to index mapping
            invDict = dict(zip(D, range(0, num_lexicon)))

            # Naive Bayes rule
            scoreList = []
            for j in range(num_topic):
                # add log(Py)
                score = math.log(Py[j])
                # add log(Pxy)
                for word in tmp_sp:
                    if word in D:
                        score += math.log(Pxy[j][invDict[word]])

                scoreList.append(score)
            # find the argmax
            maxInd = scoreList.index(max(scoreList))
            pred = classList[maxInd]

            if pred == y:
                acc += 1
                summaryScore[indList[y]] += 1 / 500
            print("Sample " + str(i) + "/10000 finished! curr_acc:",  "%.3f" % (acc / (i + 1)), end='\r', flush=True)
        acc /= 10000
        print('\nTest accuracy avg:', "%.2f" % acc)
        print('Summary')
        for n, score in enumerate(summaryScore):
            print(classList[n], "%.2f" % score)

    else:
        print("No assigned option")





