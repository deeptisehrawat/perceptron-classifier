import json
import os
import re
import sys


def preprocess_text(file_text):
    res = re.sub(r'[^\w\s]', '', file_text)
    return res.lower()


def get_activation(feature_len, weight_bias, file_text, feature_dict):
    activation = weight_bias[feature_len]
    for word in file_text.split(" "):
        i = feature_dict.get(word, None)
        if i is not None:
            activation += weight_bias[i]
    return activation


def classify_data(model_path, classify_data_path):
    model = open(model_path, 'r')
    parameters = json.load(model)
    model.close()

    output_fp = open('percepoutput.txt', 'w')
    for dirpath, dirnames, filenames in os.walk(classify_data_path, topdown=True):
        for file in filenames:
            if file.endswith(".txt") and "README" not in file:
                file_text = open(os.path.join(dirpath, file), 'r').read()
                file_text = preprocess_text(file_text)

                feature_dict = parameters["feature_dictionary"]
                feature_len = len(feature_dict)

                pn_activation = get_activation(feature_len, parameters["positive_negative_weight_bias"], file_text,
                                               feature_dict)
                if pn_activation > 0:
                    label_2 = "positive"
                else:
                    label_2 = "negative"

                td_activation = get_activation(feature_len, parameters["truthful_deceptive_weight_bias"], file_text,
                                               feature_dict)
                if td_activation > 0:
                    label_1 = "truthful"
                else:
                    label_1 = "deceptive"
                output_fp.write(label_1 + " " + label_2 + " " + os.path.join(dirpath, file) + "\n")
    output_fp.close()


if __name__ == '__main__':
    classify_data(sys.argv[1], sys.argv[2])
    # classify_data("vanillamodel.txt", "./op_spam_training_data")
    # classify_data("averagedmodel.txt", "./op_spam_training_data")
