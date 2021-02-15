from typing import List
from collections import defaultdict

LANDAS = [0.9, 0.09, 0.01]
EPSILON = 0.00001

UNK = '<ناشناخته>'
start = '<شعر> '
end = ' <\شعر>'

ferdowsi_train = "ferdowsi_train.txt"
hafez_train = "hafez_train.txt"
molana_train = "molavi_train.txt"

def read_input(path):
    with open(path, 'r', encoding="utf-8") as f:
        sentences = f.readlines()
        edit_sentences(sentences)
        dictionary = create_dict(sentences)
        # sentences = unkonwn_finder(sentences)
        return {
            "dict": dictionary,
            "sentences": sentences
        }

def edit_sentences(sentences: List[str]):
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("?", " ")
        sentences[i] = sentences[i].replace(":", " ")
        sentences[i] = sentences[i].replace("؟", " ")
        sentences[i] = sentences[i].replace("،", " ")
        sentences[i] = sentences[i].replace("*", " ")
        sentences[i] = sentences[i].replace("\"", " ")
        sentences[i] = sentences[i].replace("!", " ")
        sentences[i] = sentences[i].strip()
        while '  ' in sentences[i]:
            sentences[i] = sentences[i].replace('  ', ' ')
        sentences[i] = start + sentences[i] + end

#This section has not been used
def unkonwn_finder(sentences: List[str]):
    curpus = ' - '.join(sentences)
    words = set(curpus.split())
    for word in words:
        if curpus.count(word) == 1:
            curpus = curpus.replace(word, UNK)
    new_sentences = curpus.split(' - ')
    return new_sentences

#This section has not been used
def create_dict(sentences: List[str]):
    l = []
    curpus = ' - '.join(sentences)
    words = curpus.split()
    for word in words:
        if word not in ['-', '<شعر>', '<\شعر>'] :
            if curpus.count(word) > 1:
                if word not in l:
                    l.append(word)
    return l

def generate_unigram(sentences: List[str]):
    grams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(1, len(words) - 1):
            gram = words[i:i + 1]
            if gram not in grams:
                grams.append(gram)
    return grams

def generate_bigram(sentences: List[str]):
    grams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(0, len(words) - 2):
            gram = words[i:i + 2]
            if gram not in grams:
                grams.append(gram)
    return grams

def generate_bigram_for_test(sentence: str):
    grams = []
    words = sentence.split()
    for i in range(0, len(words) - 2):
        gram = words[i:i + 2]
        grams.append(gram)
    return grams

def learn(sentences: List[str]):
    unigrams = defaultdict(lambda: 0)
    bigrams = defaultdict(lambda: defaultdict(lambda: 0))
    p_unigrams = defaultdict(lambda: 0)
    p_bigrams = defaultdict(lambda: defaultdict(lambda: 0 ))
    text = ' '.join(sentences)
    size = len(text.split()) - len(sentences) * 2
    for k in generate_unigram(sentences):
        unigrams[k[0]] += text.count(k[0])
        p_unigrams[k[0]] = unigrams[k[0]] / size
    unigrams['<شعر>'] = len(sentences)

    for k in generate_bigram(sentences):
        tmp = ' '.join(k)
        bigrams[k[0]][k[1]] = text.count(tmp)
        p_bigrams[k[0]][k[1]] = bigrams[k[0]][k[1]] / unigrams[k[0]]

    return{
        "unigram" : p_unigrams,
        "bigram": p_bigrams
    }

def backoff_model(unigram, bigram, words):
    return bigram[words[0]][words[1]] * LANDAS[0] + unigram[words[1]] * LANDAS[1] + EPSILON * LANDAS[2]

def read_test_set():
    with open("test_file.txt", "r", encoding="utf-8") as f:
        ferdowsi_test_sentences = []
        hafez_test_sentences = []
        molana_test_sentences = []
        lines = f.readlines()
        for line in lines:
            l = line.split("\t")
            if int(l[0]) == 1:
                ferdowsi_test_sentences.append(l[1])
            elif int(l[0]) == 2:
                hafez_test_sentences.append(l[1])
            elif int(l[0]) == 3:
                molana_test_sentences.append(l[1])

        edit_sentences(ferdowsi_test_sentences)
        edit_sentences(hafez_test_sentences)
        edit_sentences(molana_test_sentences)

        return {
            "ferdowsi_test_sentences": ferdowsi_test_sentences,
            "hafez_test_sentences": hafez_test_sentences,
            "molana_test_sentences": molana_test_sentences
        }

def replace_with_UNK(fd: List[str], sentences:List[str]):
    text = ' - '.join(sentences)
    for element in text:
        if element != '-' and element != '<شعر>' and element != '<\شعر>':
            if element not in fd:
                text = text.replace(element, UNK)
    return text.split(' - ')

ferdowsi = read_input(ferdowsi_train)
ferdowsi_model = learn(ferdowsi["sentences"])
ferdowsi_unigram = ferdowsi_model["unigram"]
ferdowsi_bigram = ferdowsi_model["bigram"]
print("Complete Ferdowsi training ...")

hafez = read_input(hafez_train)
hafez_model = learn(hafez["sentences"])
hafez_unigram = hafez_model["unigram"]
hafez_bigram = hafez_model["bigram"]
print("Complete Hafez training ...")

molana = read_input(molana_train)
molana_model = learn(molana["sentences"])
molana_unigram = molana_model["unigram"]
molana_bigram = molana_model["bigram"]
print("Complete Molana training ...")

l = read_test_set()
ferdowsi_test_sentences = l["ferdowsi_test_sentences"]
hafez_test_sentences = l["hafez_test_sentences"]
molana_test_sentences = l["molana_test_sentences"]

# ferdowsi_test_sentences = replace_with_UNK(ferdowsi_test_sentences, ferdowsi["dict"])
# hafez_test_sentences = replace_with_UNK(hafez_test_sentences, hafez["dict"])
# molana_test_sentences = replace_with_UNK(molana_test_sentences, molana["dict"])


ferdowsi_correct = 0
for sentence in ferdowsi_test_sentences:
    res_f = 1
    res_h = 1
    res_m = 1
    for k in generate_bigram_for_test(sentence):
        res_f *= backoff_model(ferdowsi_unigram, ferdowsi_bigram, k)
        res_h *= backoff_model(hafez_unigram, hafez_bigram, k)
        res_m *= backoff_model(molana_unigram, molana_bigram, k)
    if max(res_m, res_h, res_f) == res_f:
        ferdowsi_correct += 1

hafez_correct = 0
for sentence in hafez_test_sentences:
    res_f = 1
    res_h = 1
    res_m = 1
    for k in generate_bigram_for_test(sentence):
        res_f *= backoff_model(ferdowsi_unigram, ferdowsi_bigram, k)
        res_h *= backoff_model(hafez_unigram, hafez_bigram, k)
        res_m *= backoff_model(molana_unigram, molana_bigram, k)
    if max(res_m, res_h, res_f) == res_h:
        hafez_correct += 1

molana_correct = 0
for sentence in molana_test_sentences:
    res_f = 1
    res_h = 1
    res_m = 1
    for k in generate_bigram_for_test(sentence):
        res_f *= backoff_model(ferdowsi_unigram, ferdowsi_bigram, k)
        res_h *= backoff_model(hafez_unigram, hafez_bigram, k)
        res_m *= backoff_model(molana_unigram, molana_bigram, k)
    if max(res_m, res_h, res_f) == res_m:
        molana_correct += 1

print("Ferdowsi : ", ferdowsi_correct / len(ferdowsi_test_sentences)*100 , "%")
print("Hafez : ", hafez_correct / len(hafez_test_sentences)*100, '%')
print("Molana : ", molana_correct / len(molana_test_sentences)*100, '%')

print("Overall :", (ferdowsi_correct+hafez_correct+molana_correct) / (len(ferdowsi_test_sentences)+len(hafez_test_sentences)+len(molana_test_sentences)) *100,'%')

