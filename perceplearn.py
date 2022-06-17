import json
import os
import random
import re
import sys
# from datetime import datetime

stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "he", "him", "his", "himself", "she",
              "her", "hers", "herself", "you", "your", "yours", "yourself", "yourselves", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "have", "has", "had",
              "having", "do", "does", "did", "doing", "what", "which", "who", "whom", "this", "that",
              "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "a", "an", "the", "and",
              "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "then", "once",
              "here", "there", "when", "where", "why", "how", "hasnt", "hadnt", "shouldnt",
              "all", "about", "against", "between", "into", "through", "only", "own", "same", "so", "than", "too",
              "very", "s", "t", "can", "will", "just", "don", "should", "now",
              "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
              "over", "under", "again", "further", "any", "both", "each", "few", "more", "most", "other", "some",
              "such", "no", "nor", "not", "didnt", "dont", "doesnt", "isnt", "arent", "wasnt", "werent", "havent"
              }


def get_training_data_paths(train_dir):
    training_data_paths = []
    for dir_path, dir_names, filenames in os.walk(train_dir):
        for file in filenames:
            if file.endswith(".txt") and "README" not in file:
                path = os.path.join(dir_path, file)
                data_path = []
                if "positive" in path and "deceptive" in path:
                    data_path = [path, "positive", "deceptive"]
                elif "positive" in path and "truthful" in path:
                    data_path = [path, "positive", "truthful"]
                elif "negative" in path and "deceptive" in path:
                    data_path = [path, "negative", "deceptive"]
                elif "negative" in path and "truthful" in path:
                    data_path = [path, "negative", "truthful"]
                training_data_paths.append(data_path)
    return training_data_paths


def preprocess_text(file_text):
    res = re.sub(r'[^\w\s]', '', file_text)
    return res.lower()


def get_feature_dictionary(training_data_paths):
    feature_dict = dict()
    tokenized_data = []
    token_idx = 0

    for path in training_data_paths:
        file_text = open(path[0], 'r').read()
        file_text = preprocess_text(file_text)

        tokenized_set = set()
        for word in file_text.split(" "):
            if word not in tokenized_set and word not in stop_words:
                tokenized_set.add(word)
                if word not in feature_dict:
                    feature_dict[word] = token_idx
                    token_idx += 1
        tokenized_data.append(tokenized_set)

    return feature_dict, tokenized_data


def get_activation(feature_len, weight_bias, file_text, feature_dict):
    activation = weight_bias[feature_len]
    for word in file_text:
        i = feature_dict.get(word, None)
        if i is not None:
            activation += weight_bias[i]
    return activation


def update_weight_bias(feature_len, weight_bias, file_text, feature_dict, value):
    for word in file_text:
        i = feature_dict.get(word, None)
        if i is not None:
            weight_bias[i] += value
    weight_bias[feature_len] += value


def train_model(x, y1, y2, tokenized_data, feature_dict, feature_len, pn_weight_bias, td_weight_bias,
                avg_pn_weight_bias, avg_td_weight_bias, counter):
    for idx, feature_vector in enumerate(x):
        file_text = tokenized_data[idx]

        # positive negative
        pn_activation = get_activation(feature_len, pn_weight_bias, file_text, feature_dict)
        if y1[idx] == "positive" and pn_activation <= 0:
            update_weight_bias(feature_len, pn_weight_bias, file_text, feature_dict, 1)
            update_weight_bias(feature_len, avg_pn_weight_bias, file_text, feature_dict, counter)
        elif y1[idx] == "negative" and pn_activation >= 0:
            update_weight_bias(feature_len, pn_weight_bias, file_text, feature_dict, -1)
            update_weight_bias(feature_len, avg_pn_weight_bias, file_text, feature_dict, -1*counter)

        # truthful deceptive
        td_activation = get_activation(feature_len, td_weight_bias, file_text, feature_dict)
        if y2[idx] == "truthful" and td_activation <= 0:
            update_weight_bias(feature_len, td_weight_bias, file_text, feature_dict, 1)
            update_weight_bias(feature_len, avg_td_weight_bias, file_text, feature_dict, counter)
        elif y2[idx] == "deceptive" and td_activation >= 0:
            update_weight_bias(feature_len, td_weight_bias, file_text, feature_dict, -1)
            update_weight_bias(feature_len, avg_td_weight_bias, file_text, feature_dict, -1*counter)
        counter += 1
    return counter


def vectorize_data(training_data_paths, tokenized_data, feature_dict, feature_len):
    x, y1, y2 = [], [], []
    for i, file_text in enumerate(tokenized_data):
        # get feature vector
        feature_vector = [0] * feature_len
        for word in file_text:
            if word in feature_dict:
                feature_vector[feature_dict[word]] += 1

        x.append(feature_vector)
        y1.append(training_data_paths[i][1])
        y2.append(training_data_paths[i][2])

    return x, y1, y2

def get_averaged_weight_bias(pn_weight_bias, td_weight_bias, avg_pn_weight_bias, avg_td_weight_bias,
                             feature_len, counter):
    for i in range(0, feature_len):
        avg_pn_weight_bias[i] = pn_weight_bias[i] - (avg_pn_weight_bias[i] / counter)
        avg_td_weight_bias[i] = td_weight_bias[i] - (avg_td_weight_bias[i] / counter)
    avg_pn_weight_bias[feature_len] = pn_weight_bias[feature_len] - (avg_pn_weight_bias[feature_len] / counter)
    avg_td_weight_bias[feature_len] = td_weight_bias[feature_len] - (avg_td_weight_bias[feature_len] / counter)


# read data, learn the model and write learned parameters to vanillamodel.txt & averagedmodel.txt file
def learn_model(train_dir):
    # print(datetime.now())
    training_data_paths = get_training_data_paths(train_dir)
    feature_dict, tokenized_data = get_feature_dictionary(training_data_paths)
    feature_len = len(feature_dict)

    # x = feature_vectors, y1 = positive/negative label, y2 = truthful/deceptive label
    x, y1, y2 = vectorize_data(training_data_paths, tokenized_data, feature_dict, feature_len)

    # declare vanilla model weight bias for both classifications
    pn_weight_bias = [0]*feature_len
    pn_weight_bias.append(0)
    td_weight_bias = [0]*feature_len
    td_weight_bias.append(0)

    # declare averaged model weight bias for both classifications
    avg_pn_weight_bias = [0] * feature_len
    avg_pn_weight_bias.append(0)
    avg_td_weight_bias = [0] * feature_len
    avg_td_weight_bias.append(0)
    counter = 1
    max_iterations = 40

    for iteration in range(1, max_iterations):
        random.shuffle(x)
        counter = train_model(x, y1, y2, tokenized_data, feature_dict, feature_len, pn_weight_bias, td_weight_bias,
                              avg_pn_weight_bias, avg_td_weight_bias, counter)

    # write parameters to the vanilla model file
    vanilla_parameters = {
        "feature_dictionary": feature_dict,
        "positive_negative_weight_bias": pn_weight_bias,
        "truthful_deceptive_weight_bias": td_weight_bias
    }
    model_fp = open("vanillamodel.txt", 'w')
    model_fp.write(json.dumps(vanilla_parameters))
    model_fp.close()

    get_averaged_weight_bias(pn_weight_bias, td_weight_bias, avg_pn_weight_bias, avg_td_weight_bias, feature_len,
                             counter)

    # write parameters to the averaged model file
    averaged_parameters = {
        "feature_dictionary": feature_dict,
        "positive_negative_weight_bias": avg_pn_weight_bias,
        "truthful_deceptive_weight_bias": avg_td_weight_bias
    }
    model_fp = open("averagedmodel.txt", 'w')
    model_fp.write(json.dumps(averaged_parameters))
    model_fp.close()
    # print(datetime.now())


if __name__ == '__main__':
    learn_model(sys.argv[1])
    # learn_model("./op_spam_training_data")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
