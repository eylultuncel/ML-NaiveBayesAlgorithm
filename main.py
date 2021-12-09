import math

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def split_data(x):
    # start and end points of each fold
    size = int(x.shape[0] / 5)
    # size = 100

    # 1/5 part of the data set as test data
    x_test = x[0:size]

    # 4/5 part of the data set as test data
    x_train = x[size:]
    return x_test, x_train


def calculate_probability(x_test, unique_words_dict, total_words_dist, n_gram):
    # take all the texts of mails in test data
    text_col = []
    for i in range(x_test.shape[0]):
        text_col.append(x_test[i, 0])

    # initialize count vectorizer
    if n_gram == 1:
        count_vectorizer = CountVectorizer(max_df=1.0, min_df=1)
    elif n_gram == 2:
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    # matrix of token counts in test data
    matrix = count_vectorizer.fit_transform(text_col)
    vocabulary = count_vectorizer.get_feature_names_out()
    matrix = matrix.toarray()

    # probability of being spam
    prob_spam = total_words_dist[0] / (total_words_dist[0] + total_words_dist[1])
    # probability of being ham
    prob_ham = total_words_dist[1] / (total_words_dist[0] + total_words_dist[1])

    predictions = []
    results = {}

    # Iterate over all test samples
    for i in range(x_test.shape[0]):

        # take log of the probability of being spam mail and being ham mail
        probability_spam = math.log2(prob_spam)
        probability_ham = math.log2(prob_ham)

        # Iterate over all words of one test sample
        for j in range(len(matrix[i])):

            # if matrix[i][j] is a number different than zero that means this word appears in that train sample
            if matrix[i][j] != 0:

                # Spam and ham count starts from 1 because of laplace smoothing
                spam_count = 1
                ham_count = 1

                # We add number of unique words for laplace smoothing
                spam_denominator = total_words_dist[0] + len(unique_words_dict)
                ham_denominator = total_words_dist[1] + len(unique_words_dict)

                word = unique_words_dict.get(vocabulary[j])

                # Check whether word is in training samples or not
                if word is not None:
                    spam_count += word[0]
                    ham_count += word[1]

                # take log of the probabilities and sum them up
                probability_spam += math.log2(spam_count / spam_denominator)
                probability_ham += math.log2(ham_count / ham_denominator)

        # print(probability_spam, probability_ham)

        # by the naive bayes algorithm , take maximum probability as prediction class
        if probability_spam > probability_ham:
            predictions.append(1)
        else:
            predictions.append(0)

        # result array has two dimensional arrays in it for each test sample
        # test sample x = [ actual class, predicted class ]
        results[i] = [x_test[i][1], predictions[i]]

    return results


def vectorizer(x, n_gram):
    # text_col refers to the first column which has mail texts
    text_col = []
    for i in range(x.shape[0]):
        text_col.append(x[i, 0])

    # initialize count vectorizer
    if n_gram == 1:
        count_vectorizer = CountVectorizer(max_df=1.0, min_df=1)
    elif n_gram == 2:
        count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    # matrix of token counts
    matrix = count_vectorizer.fit_transform(text_col)

    # vocabulary is the array of unique words which appears in all train texts
    vocabulary = count_vectorizer.get_feature_names_out()
    matrix = matrix.toarray()

    # unique words dictionary is a dict which has all unique words as key and an array [spam count, ham count] as value
    # one example key value pair is { "word" : [ 12, 30 ] ,  , }
    unique_words_dict = dict.fromkeys(vocabulary)
    for key in unique_words_dict:
        # initialize spam and ham count as [0,0] at the beginning
        unique_words_dict[key] = [0, 0]

    count_spam_mails = 0
    count_ham_mails = 0
    total_spam_words = 0
    total_ham_words = 0

    # for all train samples
    for i in range(len(x)):

        # if the sample is HAM
        if x[i][1] == 0:
            count_ham_mails += 1
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    # increase count by one in unique words dictionary
                    w = vocabulary[j]
                    unique_words_dict[w] = [unique_words_dict.get(w)[0], unique_words_dict.get(w)[1] + 1]
                    total_ham_words += 1

        # if the sample is SPAM
        else:
            count_spam_mails += 1
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    w = vocabulary[j]
                    unique_words_dict[w] = [unique_words_dict.get(w)[0] + 1, unique_words_dict.get(w)[1]]
                    total_spam_words += 1

    # total mail count of spam and ham mails
    count_mails = [count_spam_mails, count_ham_mails]

    # total word count appeared in spam and ham mails
    total_words_dist = [total_spam_words, total_ham_words]

    return unique_words_dict, count_mails, total_words_dist


def tf_idf(x, stop_words_out):
    text_col_spam = []
    text_col_ham = []
    for i in range(x.shape[0]):
        if x[i][1] == 1:
            text_col_spam.append(x[i, 0])
        elif x[i][1] == 0:
            text_col_ham.append(x[i, 0])

    my_stop_words = ENGLISH_STOP_WORDS.union()

    count_vectorizer_spam = CountVectorizer()
    matrix = count_vectorizer_spam.fit_transform(text_col_spam)
    vocabulary_spam = count_vectorizer_spam.get_feature_names_out()
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_spam)), ('tfidf', TfidfTransformer())]).fit(
        text_col_spam)

    spam_tf_idf_arr = pipe['tfidf'].idf_
    spam_tf_idf_dict = {}
    for i in range(len(spam_tf_idf_arr)):
        spam_tf_idf_dict[vocabulary_spam[i]] = spam_tf_idf_arr[i]

    spam_tf_idf_arr = sorted(spam_tf_idf_arr)
    spam_words = []
    for i in range(100):
        val = spam_tf_idf_arr[i]
        for el in spam_tf_idf_dict.keys():
            if spam_tf_idf_dict.get(el) == val:
                spam_words.append(el)
                break

    # Ham part
    count_vectorizer_ham = CountVectorizer()
    matrix = count_vectorizer_ham.fit_transform(text_col_ham)
    vocabulary_ham = count_vectorizer_ham.get_feature_names_out()
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_ham)), ('tfidf', TfidfTransformer())]).fit(
        text_col_ham)

    ham_tf_idf_arr = pipe['tfidf'].idf_
    ham_tf_idf_dict = {}
    for i in range(len(ham_tf_idf_arr)):
        ham_tf_idf_dict[vocabulary_ham[i]] = ham_tf_idf_arr[i]

    ham_tf_idf_arr = sorted(ham_tf_idf_arr)
    ham_words = []
    for i in range(100):
        val = ham_tf_idf_arr[i]
        for el in ham_tf_idf_dict.keys():
            if ham_tf_idf_dict.get(el) == val:
                ham_words.append(el)
                break

    spam_words = list(dict.fromkeys(spam_words))
    ham_words = list(dict.fromkeys(ham_words))
    a, b = [spam_words, ham_words]
    s = [x for x in b if x in a]
    for i in range(len(s)):
        word = s[i]
        if (spam_tf_idf_dict.get(word) - ham_tf_idf_dict.get(word)) < 1.5:
            if spam_words.count(word) > 0:
                spam_words.remove(word)
            if ham_words.count(word) > 0:
                ham_words.remove(word)

    if stop_words_out:
        sp = spam_words.copy()
        for i in range(len(sp)):
            word = sp[i]
            if len(my_stop_words.intersection([word])) > 0:
                spam_words.remove(word)

        hm = ham_words.copy()
        for i in range(len(hm)):
            word = hm[i]
            if len(my_stop_words.intersection([word])) > 0:
                ham_words.remove(word)

    print("Spam Words", spam_words[:10])
    print("Ham Words", ham_words[:10])
    print()

    total_spam_words_val = 0
    total_ham_words_val = 0
    unique_words_dict = {}
    for el in spam_tf_idf_dict.keys():
        x = spam_tf_idf_dict.get(el)
        unique_words_dict[el] = [x, 0]
        total_spam_words_val += spam_tf_idf_dict.get(el)
    for el in ham_tf_idf_dict.keys():
        x = ham_tf_idf_dict.get(el)
        if unique_words_dict.get(el) is not None:
            unique_words_dict[el] = [unique_words_dict.get(el)[0], x]
        else:
            unique_words_dict[el] = [0, ham_tf_idf_dict.get(el)]
        total_ham_words_val += ham_tf_idf_dict.get(el)

    total_words_dist = [(total_spam_words_val / len(spam_tf_idf_dict)), (total_ham_words_val/ len(ham_tf_idf_dict))]

    dict_copy = unique_words_dict.copy()
    for el in dict_copy.keys():
        arr = unique_words_dict.get(el)
        if arr[0] > 6 or arr[0] < 1.5:
            unique_words_dict[el] = [0, unique_words_dict.get(el)[1]]
        if arr[1] > 6 or arr[1] < 1.5:
            unique_words_dict[el] = [unique_words_dict.get(el)[0], 0]

    print(total_words_dist)
    return unique_words_dict, total_words_dist


# this function gets most frequent words which are in spam/ham mails
# but the words appear both in spam and ham mails are didn't accepted
def max_prob_words_by_naive_bayes(unique_words_dict, total_words_dist):
    max_spam_count = 0
    max_spam_word1 = ""
    max_spam_word2 = ""
    max_spam_word3 = ""
    max_ham_count = 0
    max_ham_word1 = ""
    max_ham_word2 = ""
    max_ham_word3 = ""

    for x in unique_words_dict.keys():
        arr = unique_words_dict.get(x)

        # spam
        if arr[0] > max_spam_count and ((arr[1] / total_words_dist[1]) / (arr[0] / total_words_dist[0])) < 0.5:
            max_spam_count = arr[0]
            max_spam_word3 = max_spam_word2
            max_spam_word2 = max_spam_word1
            max_spam_word1 = x

        # ham
        if arr[1] > max_ham_count and ((arr[0] / total_words_dist[0]) / (arr[1] / total_words_dist[1])) < 0.3:
            max_ham_count = arr[1]
            max_ham_word3 = max_ham_word2
            max_ham_word2 = max_ham_word1
            max_ham_word1 = x

    print("Most frequent ham word  : ", max_ham_word1)
    print("Second frequent ham word : ", max_ham_word2)
    print("Third frequent ham word : ", max_ham_word3)
    print()
    print("Most frequent spam word : ", max_spam_word1)
    print("Second frequent spam word : ", max_spam_word2)
    print("Third frequent ham word : ", max_spam_word3)
    print()
    return


# while calculating performance metrics
# True positive = th -> truly predicted ham mail
# False positive = fh -> falsely predicted ham mail
# True negative = ts -> truly predicted spam mail
# False negative = fs -> falsely predicted spam mail
def calculate_performance(results):
    th = 0
    ts = 0
    fh = 0
    fs = 0
    for key, value in results.items():
        # if the mail is ham and predicted as ham
        if value[0] == value[1] and value[1] == 0:
            th += 1
        # if the mail is spam and predicted as spam
        elif value[0] == value[1] and value[1] == 1:
            ts += 1
        # if the mail is spam but predicted as ham
        if value[0] != value[1] and value[1] == 0:
            fh += 1
        # if the mail is ham but predicted as spam
        elif value[0] != value[1] and value[1] == 1:
            fs += 1

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

    # split data %80 train - %20 test
    x_test, x_train = split_data(x.copy())

    # create dictionary of unique words
    unique_words_dict, count_mails, total_words_dist = vectorizer(x_train.copy(), 1)

    # PART1
    print("PART1 \n------------------------------------------------------------")
    max_prob_words_by_naive_bayes(unique_words_dict, total_words_dist)

    # PART2
    print("PART2 \n------------------------------------------------------------")
    # calculate probabilities of all given test data
    results = calculate_probability(x_test.copy(), unique_words_dict, total_words_dist, 1)

    # calculate performance of the given results
    accuracy, precision, recall, f1_score = calculate_performance(results)
    print("Unigram Accuracy: ", accuracy)
    print("Unigram Precision: ", precision)
    print("Unigram recall: ", recall)
    print("Unigram F1 score: ", f1_score)
    print()

    # # BIGRAM
    # # create dictionary of unique words
    # unique_words_dict, count_mails, total_words_dist = vectorizer(x_train.copy(), 2)
    # # calculate probabilities of all given test data
    # results = calculate_probability(x_test.copy(), unique_words_dict, total_words_dist, 2)
    #
    # # calculate performance of the given results
    # accuracy, precision, recall, f1_score = calculate_performance(results)
    # print("Bigram Accuracy: ", accuracy)
    # print("Bigram Precision: ", precision)
    # print("Bigram recall: ", recall)
    # print("Bigram F1 score: ", f1_score)
    # print()

    # PART3
    print("PART3 \n------------------------------------------------------------")
    print("TF-IDF")
    unique_words_dict, total_words_dist = tf_idf(x_train.copy(), False)
    # calculate probabilities of all given test data
    results = calculate_probability(x_test.copy(), unique_words_dict, total_words_dist, 1)

    # calculate performance of the given results
    accuracy, precision, recall, f1_score = calculate_performance(results)
    print("Unigram Accuracy: ", accuracy)
    print("Unigram Precision: ", precision)
    print("Unigram recall: ", recall)
    print("Unigram F1 score: ", f1_score)
    print()

    print("NON STOP WORDS")
    tf_idf(x_train.copy(), True)


if __name__ == "__main__":
    main()
