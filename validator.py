import numpy as np
import random
import nltk

data_path = 'D:\\neural_trainer\\essays.txt'

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()

valid_size = 5

topics = []
all_essays = []
for line in text.split("</s>"):
    if "Тема:" in line and "Сочинение:" in line:
        essay_text = line.split("Сочинение:")
        if len(essay_text) == 2:
            topic = essay_text[0].replace("<s>", " ").replace("</s>", " ").strip()
            essay_text = essay_text[1].replace("<s>", " ").replace("</s>", " ").strip()
            essay_text = f"Сочинение: {essay_text}"
            essay_res = f"<s>{topic}\n{essay_text}</s>"
            all_essays.append(essay_res)
            topics.append(topic)

random.seed(1234)
np.random.seed(1234)

unique_topics = list(set(topics))

valid_topics = []

for _ in range(valid_size):
    # Use randint for more speed (on big lists it is faster)
    idx = np.random.randint(0, len(unique_topics))
    valid_topics.append(unique_topics[idx])

train = []
valid = []
for topic, essay in zip(topics, all_essays):
    print('..')
    is_train = True
    for valid_topic in valid_topics:
        print('.')
        if (
            nltk.edit_distance(valid_topic, topic[:len(valid_topic)]) < 20 or
            nltk.edit_distance(valid_topic[:len(topic)], topic) < 20 or
            nltk.edit_distance(valid_topic[len(topic):], topic) < 20 or
            nltk.edit_distance(valid_topic, topic[len(valid_topic):]) < 20
            ):
            is_train = False
    if is_train:
        train.append(essay)
    else:
        valid.append(essay)

with open("train.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(train))

with open("valid.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(valid))