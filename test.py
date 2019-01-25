import tensorflow as tf
import numpy as np
import re
import pandas as pd
dialogues = pd.read_csv("example.csv")
lines = pd.read_csv('lines.csv')

id2line = {}
for line,line1 in lines.iterrows():
    if len(line1) == 6:
        id2line[line1[1]] = line1[5]


conversation_ids = []
for i,j in dialogues.iterrows():
    _conversation = j['text']
    conversation_ids.append(_conversation)



#
#questions = []
#answers = []
#for conversation in conversation_ids:
#    for i in range(len(conversation) -1):
#        questions.append(conversation)
#        answers.append([conversation + 1])



#
#for conversation in dialogues['folder']:
#    _conversation = conversation
#    conversation_ids.append(_conversation)


#id2line = {}
#for line in dialogues:
#    if len(line) == 6:
#        id2line[line[1]] = line[5]


#tf_drop = ['date','from','to']
#dialogues.drop(tf_drop, inplace=True, axis=1)
#dialogues.head(30)
#
#questions = []
#answers = []
#for conversation in dialogues:
#    for i in range(len(conversation)-1):
#        questions.append(conversation[i])
#        answers.append(conversation[i + 1])
##creating a dict thats map each line and its id
#
#    