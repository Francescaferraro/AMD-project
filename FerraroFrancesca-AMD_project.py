!pip install kaggle

!pip install ipywidgets

import os
import zipfile
import ipywidgets as widgets
import json
import pandas as pd
from pandas import json_normalize
import re
import string
from collections import defaultdict
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import builtins
import hashlib

#STEP 1 : Dataset loading

#Download via Kaggle API's

os.environ['KAGGLE_USERNAME'] = "xxxx"
os.environ['KAGGLE_KEY'] = "xxxx"
!kaggle datasets download -d yelp-dataset/yelp-dataset

#Read the review file from the ds

# Path to the downloaded zip file
zip_path = 'yelp-dataset.zip'

# File to extract from the zip
json_path = 'yelp_academic_dataset_review.json'

# Extract the specific file from the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract(json_path)

# Data Preprocessing
#Tokenization and Stopwords removal

!kaggle datasets download -d rowhitswami/stopwords

with zipfile.ZipFile('/content/stopwords.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

with open('/content/stopwords.txt', 'r') as f:
    stopwords = [line.strip() for line in f]

def tokenize(string):
      return [s for s in re.split(r'[^\w]', string.lower()) if s != '' and s not in stopwords]

# Load the Yelp dataset JSON file into a DataFrame
def load_tokenized_dataset(subset_size) :
  # I choose to include in the DataFrame only the text field
  cols = ['text']
  data = []
  count=0

  with open(json_path) as f:
      for line in f:
        if count==subset_size :
          break
        doc = json.loads(line)
        #print(doc)
        lst = [doc['text']]
        data.append(lst)
        count+=1

  df = pd.DataFrame(data=data, columns=cols)

  tokens = []
  original_sentences = []
  for index, row in df.iterrows():
        review = row['text']
        original_sentences.append(review)
        tokens.append(tokenize(review))
  return tokens, original_sentences

load = load_tokenized_dataset(69902) #1% of the dataset original size
tokens = load[0]
original_sentences = load[1]

# Stemming

def stem_word(word):
    if len(word) <= 2:  # Stemming short words is not necessary
        return word

    # Apply some of the Porter stemming rules
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        word = word[:-1]

    if len(word) > 2:
        if word.endswith('eed'):
            if len(word) > 4:  # Check for word length before stemming
                word = word[:-2]
        elif word.endswith('ed'):
            word = word[:-2]
            if word.endswith('at'):
                word += 'e'
        elif word.endswith('ing'):
            word = word[:-3]
            if word.endswith('at'):
                word += 'e'

    if len(word) > 3:
        if word.endswith('y'):
            if len(word) > 4:  # Check for word length before stemming
                word = word[:-1] + 'i'

    return word

#Replacing words with their stem and apply lowercasing
def stem_tokens(words):
    for sublist in words:
        for j in range(len(sublist)):
            stemmed_word = stem_word(sublist[j])
            sublist[j] = stemmed_word
stem_tokens(tokens)
print(tokens)

#SCALABLE APPROACH

#Step 1 : k-shingles
def create_shingles(tokens_list, k=4):
    k = min(k, len(tokens_list))
    return set(' '.join(tokens_list[i:i+k]) for i in range(len(tokens_list) - k + 1))

shingle_sets = [create_shingles(tokens_list) for tokens_list in tokens]

#Step 2 : min-hashing
def initialize_minhash(num_perm=128):
    return np.full(num_perm, np.inf)

def update_minhash(hash_values, shingle, num_perm=128):
    for i in range(num_perm):
        hash_val = int(hashlib.sha1((str(i) + shingle).encode('utf-8')).hexdigest(), 16)
        if hash_val < hash_values[i]:
            hash_values[i] = hash_val

minhash_signatures = []
for shingle_set in shingle_sets:
    hash_values = initialize_minhash()
    for shingle in shingle_set:
        update_minhash(hash_values, shingle)
    minhash_signatures.append(hash_values)

#Step 3 : Locality sensitive hashing
def lsh(minhash_signatures, num_bands=4):
    buckets = {}
    for i, minhash in enumerate(minhash_signatures):
        for band in range(num_bands):
            key = tuple(minhash[band::num_bands])
            if key not in buckets:
                buckets[key] = set()
            buckets[key].add(i)
    return buckets

buckets = lsh(minhash_signatures)

#Step 4 : compute jaccard similarity 
def jaccard_similarity(s1, s2): 
    set1 = set(s1)
    set2 = set(s2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(union) == 0:
        return 0
    else :
      return len(intersection) / len(union)

# Compute similarity
similar_pairs = []
for bucket in buckets.values():
  if len(bucket) > 1:
    bucket_list = list(bucket)  # Convert the set to a list
    #print(bucket_list)
    for i in range(len(bucket_list)):
      for j in range(len(bucket_list)):
          if i != j:
              similarity = jaccard_similarity(list(shingle_sets[bucket_list[i]]), list(shingle_sets[bucket_list[j]]))
              pair = (original_sentences[bucket_list[i]], original_sentences[bucket_list[j]], similarity)
              reverse_pair = (original_sentences[bucket_list[j]], original_sentences[bucket_list[i]], similarity)
              if 0.5 < similarity < 1 and bucket_list[i] < bucket_list[j] :
                  #print(bucket_list[i], bucket_list[j])
                  similar_pairs.append(pair)
                  if reverse_pair in similar_pairs :
                    similar_pairs.remove(pair)

print("Similar reviews pairs:")
for pair in similar_pairs:
    sentence1, sentence2, similarity = pair
    print(f"Review 1: {sentence1}\nReview 2: {sentence2}\nSimilarity: {similarity:.2f}\n\n")


# "BRUTE FORCE" APPROACH 

tokens = load_tokenized_dataset(1500)[0] #drasticalaly reduced

#Vectorization and Normalization

def normalize_vectors(vectors):
    normalized_vectors = []
    for vec in vectors:
        norm = math.sqrt(sum(x ** 2 for x in vec))
        if norm != 0:
            normalized_vec = [x / norm for x in vec]
        else:
            normalized_vec = vec
        normalized_vectors.append(normalized_vec)
    return normalized_vectors

# Bag Of Words

# Vectorization

#representation of a text document as a collection (or "bag") of individual words
def bag_of_words(sentence):
    count_dict = defaultdict(int)
    for word in sentence: # loop over each word in the sentence and increments the count for that word in the count_dict
        count_dict[word] += 1
    unique_words = list(count_dict.keys())
    index_word = {word: i for i, word in enumerate(unique_words)} # maps each unique word to its corresponding index in the bag-of-words representation
    vec = np.zeros(len(unique_words))
    for word, count in count_dict.items():
        vec[index_word[word]] = count
    return vec

bow_vectorized_tokens = []
for sentence in tokens :
  bow_vector = bag_of_words(sentence)
  bow_vectorized_tokens.append(bow_vector)

#print(bow_vectorized_tokens)

# Normalization

bow_normalized_tokens = normalize_vectors(bow_vectorized_tokens)
#print(bow_normalized_tokens)

# TF-IDF

# Vectorization

#TF(t, d) = (Number of occurrences of term t in document d) / (Total number of terms in document d)
def calculate_tf(sentence):
    tf_dict = defaultdict(int)
    total_terms = len(sentence)
    for term in sentence:
        tf_dict[term] += 1 / total_terms
    return tf_dict

#IDF(t) = log(N / DF(t)) with N = total number of documents, and DF(t) = number of documents that contain t
def calculate_idf(sentences):
    idf_dict = defaultdict(float)
    total_documents = len(sentences)
    for sentence in sentences:
        unique_terms = set(sentence)
        for term in unique_terms:
            idf_dict[term] += 1
    for term in idf_dict:
        idf_dict[term] = math.log(total_documents / idf_dict[term])
    return idf_dict

#TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
def calculate_tfidf(sentences):
    tfidf_vectors = []
    idf_dict = calculate_idf(sentences)
    for sentence in sentences:
        tf_dict = calculate_tf(sentence)
        tfidf_vector = [tf_dict[term] * idf_dict[term] for term in sentence]
        tfidf_vectors.append(tfidf_vector)
    return tfidf_vectors

tfidf_tokens = calculate_tfidf(tokens)

# Normalization

tfidf_normalized_tokens = normalize_vectors(tfidf_tokens) #same normalization function as the one I used for BoW
#print(tfidf_normalized_tokens)

# Similiarity Calculation

def calculate_similarity_scores(sentences, distance):
    n = len(sentences)
    similarity_scores = [distance(sentences[i], sentences[j]) for i in range(n) for j in range(n) if i != j]
    return similarity_scores

def find_similar_sentences(sentences, threshold, distance):
    n = len(sentences)
    similar_pairs = [(i, j) for i in range(n) for j in range(n) if i != j and distance(sentences[i], sentences[j]) >= threshold]
    return similar_pairs

def show_similarity(sentence, similar_pairs):
    similar_pairs_review = [(original_sentences[sentence1], original_sentences[sentence2]) for sentence1, sentence2 in similar_pairs if sentence1 in range(len(original_sentences))]
    similar_sentences = [sentence2 for sentence1, sentence2 in similar_pairs_review if sentence1 == sentence]
    #print(len(similar_sentences))
    if similar_sentences:
        print("THIS IS A PAIR OF SIMILAR SENTENCES : \n\n", "1. ", original, "\n\n", "2. ", random.choice(similar_sentences))
    else:
        print("THERE AREN'T SIMILARITIES WITH THE REVIEW : ", "\n", original)

# Cosine Distance

def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = math.sqrt(sum([x ** 2 for x in a]))
    norm_b = math.sqrt(sum([y ** 2 for y in b]))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# Using bag of words vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold
similarity_scores_cosine_bow = calculate_similarity_scores(bow_normalized_tokens, cosine_similarity)

plt.hist(similarity_scores_cosine_bow, bins=10)
plt.title("Cosine similarity with BoW")
plt.xlabel('Cosine similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.93  #similarity threshold, that depends on the histogram

similar_pairs = find_similar_sentences(bow_normalized_tokens, threshold, cosine_similarity)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Using TF-IDF vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold

similarity_scores_cosine_tfidf = calculate_similarity_scores(tfidf_normalized_tokens, cosine_similarity)

plt.hist(similarity_scores_cosine_tfidf, bins=10)
plt.title("Cosine similarity with TF-IDF")
plt.xlabel('Cosine similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.9  #similarity threshold

similar_pairs = find_similar_sentences(tfidf_normalized_tokens, threshold, cosine_similarity)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Jaccard Distance = 1 - jaccard similarity
def jaccard_distance(s1, s2):
    return 1 - jaccard_similarity(s1, s2)

# Using bag of words vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold

similarity_scores_jaccard_bow = calculate_similarity_scores(bow_normalized_tokens, jaccard_distance)

plt.hist(similarity_scores_jaccard_bow, bins=10)
plt.title("Jaccard similarity with BoW")
plt.xlabel('Jaccard similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.1  #similarity threshold

similar_pairs = find_similar_sentences(bow_normalized_tokens, threshold, jaccard_distance)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Using TF-IDF vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold

similarity_scores_jaccard_tfidf = calculate_similarity_scores(tfidf_normalized_tokens, jaccard_distance)

plt.hist(similarity_scores_jaccard_tfidf, bins=10)
plt.title("Jaccard similarity with TF-IDF")
plt.xlabel('Jaccard similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.1  #similarity threshold

similar_pairs = find_similar_sentences(tfidf_normalized_tokens, threshold, jaccard_distance)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Euclidean Distance

def euclidean_distance(v1, v2):
    overlap = min(len(v1), len(v2))
    distance = math.sqrt(sum((v1[i] - v2[i])**2 for i in range(overlap)))
    return distance

# Using bag of words vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold

similarity_scores_euclidean_bow = calculate_similarity_scores(bow_normalized_tokens, euclidean_distance)

plt.hist(similarity_scores_euclidean_bow, bins=10)
plt.title("Euclidean similarity with BoW")
plt.xlabel('Euclidean similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.8  #similarity threshold

similar_pairs = find_similar_sentences(bow_normalized_tokens, threshold, euclidean_distance)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Using TF-IDF vectorization

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold

similarity_scores_euclidean_tfidf = calculate_similarity_scores(tfidf_normalized_tokens, euclidean_distance)

plt.hist(similarity_scores_euclidean_tfidf, bins=10)
plt.title("Euclidean similarity with TF-IDF")
plt.xlabel('Euclidean similarity')
plt.ylabel('Frequency')
plt.show()


threshold = 0.93  #similarity threshold

similar_pairs = find_similar_sentences(tfidf_normalized_tokens, threshold, euclidean_distance)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Edit Distance

# Wagner-Fischer optimization algorithm for Edit distance
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def ed_distance(sentences) :
  n = len(sentences)
  distances = [[0] * n for _ in range(n)] #2D list used to store the edit distance between all pairs of sentences in the list
  for i in range(n):
      for j in range(n):
          if i != j:
              distances[i][j] = edit_distance(sentences[i], sentences[j])
  flat_distances = [d for row in distances for d in row] #flattened 1D array
  return flat_distances

#I plot an histogram of the similarity scores between pairs of sentences, that will help me to choose the right threshold
similarity_scores_edit = ed_distance(tokens)

plt.hist(similarity_scores_edit, bins=10)
plt.title("Edit similarity")
plt.xlabel('Edit similarity')
plt.ylabel('Frequency')
plt.show()

threshold = 350  #similarity threshold

similar_pairs = find_similar_sentences(tokens, threshold, edit_distance)

original = random.choice(original_sentences)
similarity = show_similarity(original, similar_pairs)

# Summary

similarity_scores = {
    "Cosine BoW": similarity_scores_cosine_bow,
    "Cosine TF-IDF": similarity_scores_cosine_tfidf,
    "Jaccard BoW": similarity_scores_jaccard_bow,
    "Jaccard TF-IDF": similarity_scores_jaccard_tfidf,
    "Euclidean BoW": similarity_scores_euclidean_bow,
    "Euclidean TF-IDF": similarity_scores_euclidean_tfidf,
    "Edit": similarity_scores_edit,
}

measurements = {}
histograms = {}

plt.figure(figsize=(10, 8))

for i, (measure, scores) in enumerate(similarity_scores.items()):
    histogram = np.histogram(scores, bins=10)[0]
    histograms[measure] = histogram
    measurements[measure] = np.max(histogram)
    if i != len(similarity_scores) - 1:  # Exclude the Edit distance from the plot
        plt.hist(scores, bins=10, alpha=0.5, label=measure)

plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Histogram of Similarity Scores')

plt.show()

# Compare the peaks and the shape of the histograms
for measure, histogram in histograms.items():
    print('Peak of', measure + ':', np.max(histogram))
    print('Shape of', measure + ':', histogram)
    print("\n")
