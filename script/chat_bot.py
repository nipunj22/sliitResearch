import pandas as pd
import string
import re
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')

df1 = pd.read_csv("vehical_damage_part.csv")
print(df1.head())
def format_csv():
    global df1
    df2 = df1[df1['question'].isna()]
    df1 = df1.drop(df2.index.tolist())
    df2 = df1[df1['answer'].isna()]
    df1 = df1.drop(df2.index.tolist())
    return df1

df = format_csv()
questions = list(df['question'])
print(questions)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_removed_punctuation = text.translate(translator)
    text_removed_punctuation = text_removed_punctuation.lower()
    return  text_removed_punctuation.lstrip()

def get_tokens(corpus):
  tokens = nltk.word_tokenize(remove_punctuation(corpus))
  return tokens

def generate_bigrams(words):
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return bigrams

def generate_trigrams(words):
    trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
    return trigrams

def calculate_bigram_probabilities(corpus):
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for sentence in corpus:
        tokens = get_tokens(sentence)
        sentence_bigrams = generate_bigrams(tokens)
        for bigram in sentence_bigrams:
            bigram_counts[bigram] += 1
            unigram_counts[bigram[0]] += 1

    bigram_probabilities = {}
    for bigram, count in bigram_counts.items():
        previous_word = bigram[0]
        probability = count / unigram_counts[previous_word]
        bigram_probabilities[bigram] = probability

    return bigram_probabilities

def calculate_trigram_probabilities(corpus):
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for sentence in corpus:
        tokens = get_tokens(sentence)
        sentence_trigrams = generate_trigrams(tokens)

        for trigram in sentence_trigrams:
            trigram_counts[trigram] += 1
            bigram_counts[(trigram[0], trigram[1])] += 1
            unigram_counts[trigram[0]] += 1

    trigram_probabilities = {}
    for trigram, count in trigram_counts.items():
        previous_bigram = (trigram[0], trigram[1])
        probability = count / bigram_counts[previous_bigram]
        trigram_probabilities[trigram] = probability

    return trigram_probabilities

trigram_probabilities = calculate_trigram_probabilities(questions)
print(trigram_probabilities)
bigram_probabilities = calculate_bigram_probabilities(questions)
print(bigram_probabilities)

print("Bigram Probabilities")
for bigram, probability in bigram_probabilities.items():
    print(f'Bigram: {bigram}, Probability: {probability:.3f}')
print("Trigram Probabilities")
for trigram, probability in trigram_probabilities.items():
    print(f'Trigram: {trigram}, Probability: {probability:.3f}')

def create_vocabulary(sentences):
    vocabulary = set()
    for sentence in sentences:
        words =remove_punctuation(sentence).split()
        vocabulary.update(words)
    vocabulary_list = list(vocabulary)
    word_to_index = {word: index for index, word in enumerate(vocabulary_list)}

    return vocabulary_list, word_to_index

vocabulary_list, word_to_index = create_vocabulary(questions)

print("Vocabulary List:")
print(vocabulary_list)
print(len(vocabulary_list))
print(vocabulary_list.index('what'))

def test_review(test_review):
  test_review_bigrams = generate_bigrams(get_tokens(remove_punctuation(test_review)))
  test_review_trigrams = generate_trigrams(get_tokens(remove_punctuation(test_review)))
  for trigram in test_review_trigrams:
    probability = trigram_probabilities.get(trigram, 0)
    print(trigram , probability)

def get_last_two_words(sentence):
    words = sentence.split()
    if len(words) >= 2:
        last_two_words = words[-2:]
        return ' '.join(last_two_words)
    else:
        return sentence

def get_last_word(sentence):
    words = sentence.split()
    if len(words) >= 2:
        last_word = words[-1:]
        return ' '.join(last_word)
    else:
        return sentence

def check_trigram(test_review):
  max_trigram_probability = 0
  best_word = ""
  for word in vocabulary_list:
    test_review_=get_last_two_words(test_review)+" "+word
    test_review_trigrams = generate_trigrams(get_tokens(remove_punctuation(test_review_)))
    print(test_review_trigrams)
    probability = trigram_probabilities.get(test_review_trigrams[0], 0)
    print(probability)
    if probability > max_trigram_probability:
      max_trigram_probability = probability
      best_word = word
    if max_trigram_probability == 1.0:
      return max_trigram_probability,best_word
  return max_trigram_probability,best_word

def check_bigram(test_review):
  max_bigram_probability = 0
  best_word = ""
  for word in vocabulary_list:
    test_review_=get_last_word(test_review)+" "+word
    test_review_bigrams = generate_bigrams(get_tokens(remove_punctuation(test_review)))
    print(test_review_bigrams)
    probability = bigram_probabilities.get(test_review_bigrams[0], 0)
    print(probability)
    if probability > max_bigram_probability:
      max_bigram_probability = probability
      best_word = word
    if max_bigram_probability == 1.0:
      return max_bigram_probability,best_word
  return max_bigram_probability,best_word

matchings = []
csv_number = 5
confirm_experts = 1

def add_matching(matching):
    global matchings
    matchings.append(matching)
    return

def clear_matchings():
    global matchings
    matchings.clear()
    return

def get_cache():
    global matchings
    global csv_number
    return {
        "matchings" : matchings,
        "number" : csv_number,
        "confirm_experts":confirm_experts
    }

def set_number(number):
    global csv_number
    csv_number = number
    return

def init_experts(number):
    global confirm_experts
    confirm_experts = number
    return

def set_experts():
    global confirm_experts
    confirm_experts+=1
    return

def get_most_matching():
    global matchings
    print(matchings)
    if not matchings:
        return {
            "state":False,
            "message":"no matching"
        }
    else:
        max_text_matching = 0
        max_mean_matching = 0
        most_matching_number = 0
        min_text_matching = 0.7
        min_mean_matching = 0.31
        both_fine = False
        text_match_only_fine = False
        mean_match_only_fine = False
        for matching in matchings:
            if both_fine:
                if matching['text_matching']>= max_text_matching and matching['mean_matching'] >= min_mean_matching:
                    most_matching_number = matching['number']
                    max_text_matching = matching['text_matching']
            elif matching['text_matching']>= min_text_matching and matching['mean_matching'] >= min_mean_matching:
                both_fine = True
                most_matching_number = matching['number']
                max_text_matching = matching['text_matching']
            elif text_match_only_fine:
                if matching['text_matching']>= max_text_matching:
                    max_text_matching = matching['text_matching']
                    most_matching_number = matching['number']
            elif matching['text_matching'] >= min_text_matching:
                text_match_only_fine = True
                max_text_matching = matching['text_matching']
                most_matching_number = matching['number']
            elif mean_match_only_fine:
                if matching['mean_matching']>= max_mean_matching:
                    max_mean_matching = matching['mean_matching']
                    most_matching_number = matching['number']
            elif matching['mean_matching'] >= min_mean_matching:
                text_match_only_fine = True
                max_mean_matching = matching['mean_matching']
                most_matching_number = matching['number']
        return {
            "state": True,
            "most_matching_number": most_matching_number
        }

def get_matching_rate(text1 , text2):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        cosine_similarity = cosine_similarity(vectors[0], vectors[1]).flatten()
        return cosine_similarity[0]
    except:
        return 0.0

def check_matching_bool(text1 , text2 , number):
    text_matching = get_matching_rate(text1 , text2)
    print(text_matching)
    # mean_matching = MeaningCompare.get_matching_rate(text1, text2)
    # print(mean_matching)
    cache = {
        "number":number,
        "text_matching":text_matching,
        "mean_matching":0.0
    }
    if text_matching>=0.7:
        add_matching(cache)

df_ = format_csv()
df2 = df_[['number','question']]

def check_answers(text):
    for index, row in df2.iterrows():
        print(row)
        check_matching_bool(text ,row['question'] ,row['number'])
    return

def get_answer_by_number(number):
    options = {number}
    dff = df1[df1['number'].isin(options)]
    print(dff['answer'].tolist())
    return dff['answer'].tolist()[0]


def get_dataframe():
    return df1

def set_number():
    number = len(df1.index)
    print(number)
    set_number(number=number+1)
    return

def get_answer_for_question(question):
    try:
        check_answers(question)
        most_matching = get_most_matching()
        print("most matching ",most_matching)
        clear_matchings()
        if most_matching['state']:
            answer = get_answer_by_number(most_matching['most_matching_number'])
            print('answer - ', answer)
            return {
                "state": True,
                "message": "success",
                "result": answer
            }
        else:
            return {
                "state": False,
            }
    except Exception:
        return {
            "state": False,
            "message": Exception
        }

def check_answer(question):
  answer = get_answer_for_question(question)
  if answer['state']:
    print("answer : ",answer['result'])
  return answer


def get_answer(test_question):
    for i in range(10):
        max_trigram_probability, best_word = check_trigram(test_question)
        print(best_word)
        if max_trigram_probability == 0:
            max_bigram_probability, best_word = check_bigram("what types")
            print(best_word)
            if max_bigram_probability == 0:
                break
            else:
                test_question += " " + best_word
                answer = check_answer(test_question)
                if answer['state']:
                    return {
                        "best question": test_question,
                        "best answer": answer['result']
                    }
        else:
            test_question += " " + best_word
            answer = check_answer(test_question)
            if answer['state']:
                return {
                    "best question":test_question,
                    "best answer":answer['result']
                }

test_question = "how is the premium"
print("Best question : ",get_answer(test_question))