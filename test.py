import tensorflow as tf
import numpy as np
import re
import pandas as pd
import time
print(tf.__version__)
import sys
print(sys.version)

conversations = pd.read_csv("conversation.csv", encoding='latin-1')

conversation_ids = []
for conver,convers in conversations.iterrows():
    _conversation = convers['text']
    conversation_ids.append(_conversation)

#Getting seperately the questions and the answers
full_questions = []
full_answers = []
for conversation in conversation_ids:
    full_questions.append(conversation)
    full_answers.append(conversation)


questions = []
for i in full_answers:
    questions.append(i)

answers = []
for i in full_questions:
    answers.append(i)

#Now create Function for cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am" , text)
    text = re.sub(r"he's","he is" , text)
    text = re.sub(r"she's","she is" , text)
    text = re.sub(r"that's","that is" , text)
    text = re.sub(r"how's","how is" , text)
    text = re.sub(r"what's","what is" , text)
    text = re.sub(r"it's","it is" , text)
    text = re.sub(r"where is","where is" , text)
    text = re.sub(r"\'ll","will" , text)
    text = re.sub(r"i'll","i will" , text)
    text = re.sub(r"it'll","it will" , text)
    text = re.sub(r"\'ve","have" , text)
    text = re.sub(r"you've","you have" , text)
    text = re.sub(r"i've","i have" , text)
    text = re.sub(r"\'re","are" , text)
    text = re.sub(r"\'d","would" , text)
    text = re.sub(r"i'd","i would" , text)
    text = re.sub(r"where'd","where would" , text)
    text = re.sub(r"won't","will not" , text)
    text = re.sub(r"don't","do not" , text)
    text = re.sub(r"doesn't","does not" , text)
    text = re.sub(r"weren't","were not" , text)
    text = re.sub(r"couldn't","could not" , text)
    text = re.sub(r"shouldn't","should not" , text)
    text = re.sub(r"wouldn't","would not" , text)
    text = re.sub(r"can't","can not" , text)
    text = re.sub(r"isn't","is not" , text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","" , text)
    return text


#Cleaning the questions 
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
#Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


#Creating a dictionary thats maps there each word for number of occurence 
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] +=1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] +=1


#Creating 2 dictionaries that maps the question and answers words to a unique integer
#First initialzie a threshold of 20
threshold = 2
questions_word_2_int = {}
word_number = 0
for word,count in word2count.items():
    if count >= threshold:
        questions_word_2_int[word] = word_number
        word_number += 1

answers_word_2_int = {}
word_number = 0
for word,count in word2count.items():
    if count >= threshold:
        answers_word_2_int[word] = word_number
        word_number += 1    
#
#Adding the last tokens to these 2 dictionaries
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questions_word_2_int[token] = len(questions_word_2_int) + 1

for token in tokens:
    answers_word_2_int[token] = len(answers_word_2_int) + 1

#Now creating the inverse dictionary of the answerswords2int dictionary
answerints2word = {w_i:w for w,w_i in answers_word_2_int.items()}

#Adding the end of string token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

#Translating all the questions and answers into integers
#And replacing all the words that were filtered out by <OUT>

questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_word_2_int:
            ints.append(questions_word_2_int['<OUT>'])
        else:
            ints.append(questions_word_2_int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_word_2_int:
            ints.append(answers_word_2_int['<OUT>'])
        else:
            ints.append(answers_word_2_int[word])
    answers_into_int.append(ints)
#
#Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            
for length in range(1,25):
    for i in enumerate(answers_into_int):
        if len(i[1]) == length:
            sorted_clean_answers.append(answers_into_int[i[0]])
            
#            
################# part 2 Building the seq to seq Model ######################


#Creating the placeholders for the inputs and the outputs
def model_inputs():
    inputs = tf.placeholder(tf.int32 , [None,None] , name = 'input')
    targets = tf.placeholder(tf.int32 , [None,None] , name = 'target')
    lr = tf.placeholder(tf.float32 , name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob_parameter')
    return inputs , targets , lr , keep_prob

#Preprocessing the targets
def preprocess_targets(targets , word_2_int , batch_size):
    left_side = tf.fill([batch_size , 1] , word_2_int['<SOS>'])
    right_side = tf.strided_slice(targets , [0,0] , [batch_size , -1] , [1,1])
    preprocessed_targets = tf.concat([left_side , right_side] , 1)
    return preprocessed_targets

#Creating the Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm , input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output , encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                     cell_bw = encoder_cell,
                                                                     sequence_length = sequence_length,
                                                                     inputs = rnn_inputs,
                                                                     dtype = tf.float32)
    return encoder_state

#Decoding the training set
def decode_training_set(encoder_state , decoder_cell , decoder_embedded_input , sequence_length , decoding_scope , output_function , keep_prob , batch_size):
    attention_states = tf.zeros([batch_size , 1 , decoder_cell.output_size])
    attention_keys , attention_values , attention_score_function , attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states , attention_option = 'bahdanau' , num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "atten_dec_train")
    decoder_output , decoder_final_state , decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                training_decoder_function,
                                                                                                                decoder_embedded_input,
                                                                                                                sequence_length,
                                                                                                                scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output , keep_prob)
    return output_function(decoder_output_dropout)

#decoding the test/validation set
def decode_test_set(encoder_state , decoder_cell , decoder_embeddings_matrix ,sos_id , eos_id , maximum_length , num_words ,decoding_scope , output_function , keep_prob , batch_size):
    attention_states = tf.zeros([batch_size , 1 , decoder_cell.output_size])
    attention_keys , attention_values , attention_score_function , attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states , attention_option = 'bahdanau' , num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions , decoder_final_state , decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                  test_decoder_function,
                                                                                                                  scope = decoding_scope)
    return test_predictions                                                                                                       

#Creating the decoder RNN
def decoder_rnn (decoder_embedded_input, decoder_embeddings_matrix,  encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm , input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        
    return training_predictions, test_predictions

#building Seq_2_Seq Model
def seq2seq_model (inputs , targets , keep_prob , batch_size , sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_word_2_int):
    encoder_embedded_inputs = tf.contrib.layers.embed_sequence(inputs,
                                                               answers_num_words + 1,
                                                               encoder_embedding_size,
                                                               initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_inputs, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_word_2_int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1 , decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions , test_predictions = decoder_rnn(decoder_embedded_input,
                                                          decoder_embeddings_matrix,
                                                          encoder_state,
                                                          questions_num_words,
                                                          sequence_length,
                                                          rnn_size,
                                                          num_layers,
                                                          questions_word_2_int,
                                                          keep_prob,
                                                          batch_size)
    return training_predictions, test_predictions


###################### PART 3 - Training the SEQ2SEQ Model ############################


#setting the hyperparameters
epochs = 10
batch_size = 20
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

#Setting the Sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs , [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answers_word_2_int),
                                                       len(questions_word_2_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size, 
                                                       num_layers,
                                                       questions_word_2_int)

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

#Padding the sequence with the <PAD> token
#Question: ( 'Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD> )
#Answer: (<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD> )
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

#Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):#// is for integer value
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_word_2_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_word_2_int))
        yield padded_questions_in_batch, padded_answers_in_batch

#Splitting the questions and answers into training and validtion sets
training_validation_split =  int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]        
        
#Training
batch_index_check_training_loss = 100 #chk training loss by every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1#Half of the num of bathces, -1 just around the down
total_training_loss_error = 0#used for the sum of training of 100 bathces
list_validation_loss_error = []
early_stopping_check = 0 #each time there is no improb=vement   
early_stopping_stop = 1000 
checkpoint = "./chatbot_weights.ckpt" #load the weights of trained chatbot
# Now run the session
session.run(tf.global_variables_initializer())
#Start Big for loop gonna do all the training
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers,batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        #to check 100 batches
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches : {:d} seconds'.format(epoch,
                                                                                                                                        epochs,
                                                                                                                                        batch_index,
                                                                                                                                        len(training_questions) // batch_size,
                                                                                                                                        total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                        int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        #Now if for validation step
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                #Now we dont need of optimizer coz its only need in training mode
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})#keep prob is 1 coz in validation is 1
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)#gives no of batches in the validation sets
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error,int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print("I Speak Better now !!")
                #If we do any implementation than 0
                early_stopping_check = 0
                #Now we save the steps
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry, I do not Speak better , I need to practice More.")
                #WE donot make any changes
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best i can do.")
        break
print("Game Over")
###
###
############### PART 4 - TESTING THE SEQ2SEQ MODEL ###############
###
### Loading the weights and Running the session
#checkpoint = "./chatbot_weights.ckpt"
#session = tf.InteractiveSession()
#session.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(session, checkpoint)
#
## Converting the questions from strings to lists of encoding integers
#def convert_string2int(question, word2int):
#    question = clean_text(question)
#    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
#
## Setting up the chat
#while(True):
#    question = input("You: ")
#    if question == 'Goodbye':
#        break
#    question = convert_string2int(question, questions_word_2_int)
#    question = question + [questions_word_2_int['<PAD>']] * (20 - len(question)) # Questions of length 20
#    fake_batch = np.zeros((batch_size, 20))
#    fake_batch[0] = question
#    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0] # Get first element
#    answer = ''
#    for i in np.argmax(predicted_answer, 1):
#        if answerints2word[i] == 'i':
#            token = 'I'
#        elif answerints2word[i] == '<EOS>':
#            token = '.'
#        elif answerints2word[i] == '<OUT>':
#            token = 'out'
#        else:
#            token = ' ' + answerints2word[i]
#        answer += token
#        if token == '.':
#            break
#    print('ChatBot: ' + answer)
#
#













#aik aik bandy k questions and answer alag































#id2line = {}
#for line,ji in lines.iterrows():
#    id2line[ji[1]] = ji[5]
#   
#    
##    if len(line) == 6:
##        id2line[line[1]] = line[5]
#
#
#conversation_ids = []
#for i,j in dialogues.iterrows():
#    _conversation = j['text']
#    conversation_ids.append(_conversation)
#
#
#
#
#questions = []
#answers = []
#for conversation in conversation_ids:
#    questions.append(conversation)
#   #conversation = conversation + 1
#    answers.append(conversation)
#
#
##Cleaning the 
#def clean_text(text):
#    text = text.lower()
#    text = re.sub(r"i'm","i am" , text)
#    text = re.sub(r"he's","he is" , text)
#    text = re.sub(r"she's","she is" , text)
#    text = re.sub(r"that's","that is" , text)
#    text = re.sub(r"what's","what is" , text)
#    text = re.sub(r"it's","it is" , text)
#    text = re.sub(r"where is","where is" , text)
#    text = re.sub(r"\'ll","will" , text)
#    text = re.sub(r"\'ve","have" , text)
#    text = re.sub(r"\'re","are" , text)
#    text = re.sub(r"\'d","would" , text)
#    text = re.sub(r"won't","will not" , text)
#    text = re.sub(r"doesn't","does not" , text)
#    text = re.sub(r"can't","can not" , text)
#    text = re.sub(r"[-()\"!#/@;:<>^{}+=~|.?,']","" , text)
#    return text    
#
#
##Cleaning the questions 
#clean_questions = []
#for question in questions:
#    clean_questions.append(clean_text(question))
#    
##Cleaning the answers
#clean_answers = []
#for answer in answers:
#    clean_answers.append(clean_text(answer))
#    
#    
##Creating a dictionary thats there each word for number of occurence 
#word2count = {}
#for question in clean_questions:
#    for word in question.split():
#        if word not in word2count:
#            word2count[word] = 1
#        else:
#            word2count[word] +=1
#
#for answer in clean_answers:
#    for word in answer.split():
#        if word not in word2count:
#            word2count[word] = 1
#        else:
#            word2count[word] +=1
#
#
##Creating 2 dictionaries that maps the question and answers words to a unique integer
##First initialzie a threshold of 20
#threshold = 20
#questionword2int = {}
#word_number = 0
#for word,count in word2count.items():
#    if count >= threshold:
#        questionword2int[word] = word_number
#        word_number += 1
#
#answerword2int = {}
#word_number = 0
#for word,count in word2count.items():
#    if count >= threshold:
#        answerword2int[word] = word_number
#        word_number += 1    
#
#
##Adding the last tokens to these 2 dictionaries
#tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
#for token in tokens:
#    questionword2int[token] = len(questionword2int) + 1
#
#for token in tokens:
#    answerword2int[token] = len(answerword2int) + 1
#
#
##Now creating the inverse dictionary of the answerswords2int dictionary
#answerints2word = {w_i:w for w,w_i in answerword2int.items()}
#
##Adding the end of string token to the end of every answer
#for i in range(len(clean_answers)):
#    clean_answers[i] += ' <EOS>'
#
#
##Translating all the questions and answers into integers
##And replacing all the words that were filtered out by <OUT>
#
#questions_into_int = []
#for question in clean_questions:
#    ints = []
#    for word in question.split():
#        if word not in questionword2int:
#            ints.append(questionword2int['<OUT>'])
#        else:
#            ints.append(questionword2int[word])
#    questions_into_int.append(ints)
#
#answers_into_int = []
#for answer in clean_answers:
#    ints = []
#    for word in answer.split():
#        if word not in answerword2int:
#            ints.append(answerword2int['<OUT>'])
#        else:
#            ints.append(answerword2int[word])
#    answers_into_int.append(ints)



























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