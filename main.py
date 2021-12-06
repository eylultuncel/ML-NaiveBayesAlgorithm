import math

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# from sklearn.feature extraction.text import ENGLISH STOP WORDS


def split_data(x):
    # start and end points of each fold
    size = int(x.shape[0] / 5)

    # 1/5 part of the data set as test data
    x_test = x[0:size]
    print(x_test.shape)

    # 4/5 part of the data set as test data
    x_train = x[size:]
    print(x_train.shape)
    return x_test, x_train


def calculate_probability(x_test, unique_words_dict, count_mails, total_words_dist):
    text_col = []
    for i in range(x_test.shape[0]):
        text_col.append(x_test[i, 0])

    count_vectorizer = CountVectorizer()
    # matrix of token counts
    matrix = count_vectorizer.fit_transform(text_col)
    vocabulary = count_vectorizer.get_feature_names_out()
    matrix = matrix.toarray()

    prob_spam = count_mails[0] / (count_mails[0] + count_mails[1])
    prob_ham = count_mails[1] / (count_mails[0] + count_mails[1])
    
    predictions = []
    results = {}

    # Iterate over all test samples
    for i in range(x_test.shape[0]):
        probability_spam = math.log(prob_spam)
        probability_ham = math.log(prob_ham)

        # Iterate over all words of one test sample
        for j in range(len(matrix[i])):

            if(matrix[i][j] != 0):

                # Spam and ham count starts from 1 because of laplace smoothing
                spam_count = 1
                ham_count = 1
                
                # We add number of unique words for laplace smoothing
                spam_denominator = total_words_dist[0] + len(unique_words_dict)
                ham_denominator = total_words_dist[1] + len(unique_words_dict)
                
                word = unique_words_dict.get(vocabulary[j])
                
                # Check whether word is in training samples or not
                if(word is not None):
                    spam_count += word[0]
                    ham_count += word[1]

                probability_spam += math.log(spam_count/spam_denominator)
                probability_ham += math.log(ham_count/ham_denominator)

                # Algorithm that only laplace smoothing in 0 spam or ham values
                #  spam_count = 0
                # ham_count = 0
                # spam_denominator = total_words_dist[0]
                # ham_denominator = total_words_dist[1]
                # word = unique_words_dict.get(vocabulary[j])
                # if(word is not None):
                #     spam_count = word[0]
                #     ham_count = word[1]
                # if(spam_count == 0):
                #     spam_count = 1
                #     spam_denominator += len(unique_words_dict)
                # if(ham_count == 0):
                #     ham_count = 1
                #     ham_denominator += len(unique_words_dict)
    
                # probability_spam += math.log(spam_count/spam_denominator)
                # probability_ham += math.log(ham_count/ham_denominator)

        print(probability_spam, probability_ham)

        if(probability_spam > probability_ham):
            predictions.append(1)
        else:
            predictions.append(0)

        results[i] = [x_test[i][1], predictions[i]]

    return results

def vectorizer(x):
    # text_col refers to the first column which has mail texts
    text_col = []
    for i in range(x.shape[0]):
        text_col.append(x[i, 0])

    # initialize count vectorizer
    count_vectorizer = CountVectorizer()
    # matrix of token counts
    matrix = count_vectorizer.fit_transform(text_col)
    vocabulary = count_vectorizer.get_feature_names_out()
    print(vocabulary)
    print("--------------------------")
    print(len(matrix.toarray()))
    matrix = matrix.toarray()

    # for i in range(len(matrix)):
    #     for j in range(len(matrix[1])):
    #         print(matrix[i][j], end="")
    #     print()

    # pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfid', TfidfTransformer())]).fit(text_col)
    # print(type(pipe['count'].transform(text_col).toarray()))
    # print(type(pipe['tfid'].idf_))
    # print(len(pipe['tfid'].idf_))
    # print(pipe.transform(text_col).shape)

    unique_words_dict = dict.fromkeys(vocabulary)
    for key in unique_words_dict:
        unique_words_dict[key] = [0, 0]

    count_spam_mails = 0
    count_ham_mails = 0
    total_spam_words = 0
    total_ham_words = 0

    for i in range(len(x)):
        # NOT SPAM CASE- HAM
        if x[i][1] == 0:
            count_ham_mails += 1
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    w = vocabulary[j]
                    unique_words_dict[w] = [unique_words_dict.get(w)[0], unique_words_dict.get(w)[1]+1]
                    total_ham_words += 1

        # SPAM
        else:
            count_spam_mails += 1
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    w = vocabulary[j]
                    unique_words_dict[w] = [unique_words_dict.get(w)[0]+1, unique_words_dict.get(w)[1]]
                    total_spam_words += 1

    count_mails = [count_spam_mails, count_ham_mails]
    total_words_dist = [total_spam_words, total_ham_words]

    return unique_words_dict, count_mails, total_words_dist

def calculate_performance(results):
    th = 0
    ts = 0
    fh = 0
    fs = 0
    for key,value in results.items():
        if value[0] == value[1] and value[1] == 0:
            th+=1
        elif value[0] == value[1] and value[1] == 1:
            ts+=1
        if value[0] != value[1] and value[1] == 0:
            fh+=1
        elif value[0] != value[1] and value[1] == 1:
            fs+=1
    
    accuracy = (th + ts) / (th + ts + fh + fs)
    precision = th / (th + fh)
    recall = th / (th + fs)
    f1_score = (2 * recall * precision) / (recall + precision)
    return accuracy, precision, recall, f1_score

def main(total_words_dis=None):
    # reading data's in the csv file to the numpy array
    df = pd.read_csv('./emails.csv')
    x = np.array(df.iloc[:, :])

    # shuffle the data
    np.random.seed(101)
    np.random.shuffle(x)
    np.random.seed(102)
    np.random.shuffle(x)
    np.random.seed(103)
    np.random.shuffle(x)

    # split data %80 - %20
    x_test, x_train = split_data(x.copy())
    unique_words_dict, count_mails, total_words_dist = vectorizer(x_train)
    results = calculate_probability(x_test, unique_words_dict, count_mails, total_words_dist)
    performance = calculate_performance(results)
    print(performance)
        

if __name__ == "__main__":
    main()
