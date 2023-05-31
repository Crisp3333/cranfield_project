# Dane Hylton
# Text Mining
import math
import operator
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from collections import defaultdict
# from PyDictionary import PyDictionary
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from readers import read_queries, read_documents

inverted_index = {}
term_frequency_data = {}
document_frequency_data = {}
inverse_document_frequency_data = {}
document_length = {}
average_length = 0


def remove_not_indexed_toknes(tokens):
    return [token for token in tokens if token in inverted_index]


def stem_token(word):
    stemmer = SnowballStemmer("english")
    stem_word = stemmer.stem(word)
    # print(word)
    # print(stem_word)
    return stem_word


def stop_words(indexed_tokens):
    # Get the stopwords
    stop_word = set(stopwords.words('english'))

    return [w for w in indexed_tokens if w not in stop_word]


def stop_word_index():
    # Get the stopwords
    stop_word = list(set(stopwords.words('english')))

    return stop_word


def term_frequency():
    for key in inverted_index:
        # Create Counter class object
        counter = Counter(inverted_index[key])
        # Count how many times term occurs in document id
        # print(key)
        # print(counter)
        counter_list = counter.most_common()
        # print(counter_list)
        # Assign to term frequency dictionary with
        # key = key, and value = counter_list
        term_frequency_data[key] = counter_list


def document_frequency():
    for key in inverted_index:
        unique_id = set(inverted_index[key])

        document_frequency_data[key] = len(unique_id)


# Calculate the inverse document frequency
def inverse_document_frequency():
    number_of_document = len(read_documents())

    for key in document_frequency_data:
        n_idf = number_of_document / document_frequency_data[key]
        inverse_document_frequency_data[key] = math.log10(n_idf)


def score_bm25(current_score, all_score, key):
    if len(all_score) != 0:
        if key in all_score:
            all_score[key] += current_score
        else:
            all_score[key] = current_score
    # dictionary being populated for the first time
    else:
        all_score[key] = current_score
    return all_score


def denominator_bm25(key, tf, k1):
    # parameters
    b = 0.4849
    ld = document_length[key]
    # lave = average_length
    global average_length
    # print(average_length)
    l_average = ld/average_length
    denom = k1 * ((1 - b) + b * l_average) + math.log10(1 + tf)
    # denom = np.multiply(k1, ((1 - b) + np.multiply(b, l_average))) + np.log10(1 + tf)
    # print(denom)
    return denom


def calculate_bm25(indexed_tokens):
    sum_bm25 = {}
    k1 = 1.528
    # k3 = 0.005

    for token in indexed_tokens:

        if token in inverted_index:
            for value in term_frequency_data[token]:
                doc_id = value[0]
                # Get term frequency
                freq = value[1]

                freq_log = np.log10(1 + freq)
                numerator = np.multiply((k1 + 1), freq_log)
                # Calculate the denominator
                denominator = denominator_bm25(key=doc_id, tf=freq, k1=k1)

                de_nu = numerator/denominator
                # Calculate bm25
                current_bm25 = np.multiply(inverse_document_frequency_data[token], de_nu)
                # Sum the scores
                sum_bm25 = score_bm25(current_score=current_bm25, all_score=sum_bm25, key=doc_id)
    return sum_bm25


def bm25_search(query):
    score_ids = []
    tokens = tokenize(str(query['query']))
    remove_indexed_tokens = remove_not_indexed_toknes(tokens)
    # indexed_tokens = remove_not_indexed_toknes(tokens)
    indexed_tokens = stop_words(remove_indexed_tokens)

    # Calculate using Okapi BM25
    bm25_id_scores = calculate_bm25(indexed_tokens)
    sort_bm25 = sorted(bm25_id_scores.items(), key=operator.itemgetter(1), reverse=True)

    # Get sorted ids by themselves
    for doc_id in sort_bm25:
        score_ids.append(doc_id[0])

    return score_ids


def average_document_length():
    collection_length = sum(document_length.values())
    average = collection_length/len(read_documents())
    return average


# replace periods and commas with space
def tokenize(text):
    replace_periods = text.replace(".", "")
    replace_commas = replace_periods.replace(",", "")
    tokens = replace_commas.split(" ")
    return stop_words(tokens)


def add_token_to_index(token, doc_id):
    if token in inverted_index:
        current_postings = inverted_index[token]
        current_postings.append(doc_id)
        inverted_index[token] = current_postings
        # print(inverted_index[token])
    else:
        inverted_index[token] = [doc_id]
        # print(inverted_index[token])


def add_to_index(document):
    # stop_word = list(set(stopwords.words('english')))
    doc_id = document['id']
    tokenize_title = tokenize(document['title'])
    tokenize_body = tokenize(document['body'])

    for token in tokenize_title:
        # if token not in stop_word:
        add_token_to_index(token, document['id'])

    for token in tokenize_body:
        # if token not in stop_word:
        add_token_to_index(token, document['id'])

    # Populate document length dictionary
    document_length[doc_id] = len(tokenize_title) + len(tokenize_body)


def create_index():
    for document in read_documents():
        add_to_index(document)
    # Call functions for pre-calculation before query score calculation
    term_frequency()
    document_frequency()
    inverse_document_frequency()
    global average_length
    average_length = average_document_length()

    print("Created index with size {}".format(len(inverted_index)))


create_index()


if __name__ == '__main__':
    all_queries = [query for query in read_queries() if query['query number'] != 0]
    for query in all_queries:
        documents = bm25_search(query)
        print("Query:{} and Results:{}".format(query, documents))
