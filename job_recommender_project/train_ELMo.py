data science
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib import pyplot as plt

#NLP
import spacy
import re

import time
from tqdm import tqdm  # progress bar

# file loading
import pickle

#deep learning
import tensorflow as tf
import tensorflow_hub as hub

pd.set_option('display.max_colwidth', 200)

# resumes only for now
# gives df with list of strings (tokenized) as well as lemmatized list of strings

ec2_resumes_path = '/home/ubuntu/NLP_projects/job_recommender_project/data/large_files/lf_cleaned_lemmatized_tokenized_resumes.csv'
ec2_pickle_resumes_path = '/home/ubuntu/NLP_projects/job_recommender_project/data/resumes_with_list_of_list.pickle'

resumes = pd.read_pickle(ec2_pickle_resumes_path)


resumes['los'] = resumes['lol']

for i in range (0, len(resumes['los'])):
    for j in range (0, len(resumes['los'][i])):
        resumes['los'][i][j] = ' '.join(resumes['los'][i][j])

elmo_in = resumes['los']



# the lenghts of each resume and resume ID for reference


lengths = []
for i in range(0, len(elmo_in)):
    lengths.extend([len(elmo_in[i])])

#lengths



# let's look at distributions of resume length

plt.hist(lengths, bins=range(0, 1000, 10))

plt.show()


plt.hist(lengths, bins=range(0, 200, 5))

plt.show()

#print("example of a short resume:")
#print("resume number 244")
#print("resume length: {} sentences".format(len(elmo_in[244])))
#print(elmo_in[244])

#print("example of a medium resume:")
#print("resume number 202")
#print("resume length: {} sentences".format(len(elmo_in[202])))
#print(elmo_in[202])

#print("example of a long resume:")
#print("resume number 28")
#print("resume length: {} sentences".format(len(elmo_in[28])))
#print(elmo_in[28])

print('min: {}'.format(np.amin(lengths)))
print('max: {}'.format(np.amax(lengths)))
print('median: {}'.format(np.median(lengths)))
print('mean: {}'.format(np.mean(lengths)))
print('stdev: {}'.format(np.std(lengths)))


lengths_copy = lengths.copy()


elmo_in_smalls = elmo_in.copy()

elmo_in_smalls = elmo_in_smalls.tolist()

type(elmo_in_smalls)

# get rid of resumes with over 100 sentences or less than 2 sentences
for i in elmo_in_smalls:
    if len(i) > 100:
        elmo_in_smalls.remove(i)
    if len(i) <2:
        elmo_in_smalls.remove(i)

#elmo_in_smalls contains the >=2, <=100 resumes


# for i in range(0, len(elmo_in_smalls)):
#     for j in range (0, len(elmo_in_smalls[i])):
#         if len(elmo_in_smalls[i][j])<2:
#             print(i, j, elmo_in_smalls[i][j])


len(elmo_in_smalls)

#the lengths of all those kept resumes
lengths_smalls = []
for i in range(0, len(elmo_in_smalls)):
    lengths_smalls.extend([len(elmo_in_smalls[i])])

#lengths_smalls

# let's look at distributions of resume length

plt.hist(lengths_smalls, bins=range(0, 100, 5))

plt.show()


# now we have lengths_smalls for each length
# now we have elmo_in_smalls for each resume

type(elmo_in_smalls[0])

# I have elmo saved locally
elmo = hub.Module("/home/ubuntu/module/module_elmo2", trainable=False)

# It's a Tesla K80 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# first_try = []
# for i in range(0, len(elmo_in_smalls[0])):
#     first_try = elmo_vectors(elmo_in_smalls[0][i])
#     print(first_try)


##trying on elmo_in_smalls[0]
#embeddings = elmo(elmo_in_smalls[0], signature="default",as_dict=True)["elmo"]
#print("embeddings is type {} and shape {}".format(type(embeddings), embeddings.shape))
#with tf.Session() as session:
#    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#    message_embeddings_2d = session.run(tf.reduce_mean(embeddings,axis=0))
#    print("message_embeddings_2d is type {} and shape {}".format(type(message_embeddings_2d), message_embeddings_2d.shape))
#    message_embeddings_1d = tf.reduce_mean(tf.convert_to_tensor(message_embeddings_2d), axis = 0, keepdims=True)
#    print("message_embeddings_1d is type {} and shape {}".format(type(message_embeddings_1d), message_embeddings_1d.shape))


#with tf.Session() as sess:  print(message_embeddings_1d.eval()) 



##trying on elmo_in_smalls[1]
#embeddings1 = elmo(elmo_in_smalls[1], signature="default",as_dict=True)["elmo"]
#print("embeddings is type {} and shape {}".format(type(embeddings1), embeddings1.shape))
#with tf.Session() as session:
#    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#    message_embeddings1_2d = session.run(tf.reduce_mean(embeddings1,axis=0))
#    print("message_embeddings1_2d is type {} and shape {}".format(type(message_embeddings1_2d), message_embeddings1_2d.shape))
#    message_embeddings1_1d = session.run(tf.reduce_mean(tf.convert_to_tensor(message_embeddings1_2d), axis = 0, keepdims=True))
#    print("message_embeddings1_1d is type {} and shape {}".format(type(message_embeddings1_1d), message_embeddings1_1d.shape))

    

#print(message_embeddings1_1d)

#with tf.Session() as sess:  print(message_embeddings1_1d.eval()) 

#with tf.Session() as sess: print(cosine_similarity(message_embeddings_1d.eval(), message_embeddings1_1d.eval()))




# print("embeddings shape: {}".format(embeddings.shape))
# print("elmo_in_smalls[0] sentences: {}".format(len(elmo_in_smalls[0])))
# max_sentence_length = 0
# for i in range(0, len(elmo_in_smalls[0])):
#     if len(elmo_in_smalls[0][i].split()) > max_sentence_length:
#         max_sentence_length = len(elmo_in_smalls[0][i])
                                  
# print("elmo_in_smalls[0] max sentence length: {}".format(max_sentence_length))


#len(elmo_in_smalls)

#len(elmo_in_smalls[129])





# for all embeddings

resume_embeddings = []


for i in tqdm(range(0, len(elmo_in_smalls))):
    print('elmo_in_smalls[{}]'.format(i))
    embeddings_3d = elmo(elmo_in_smalls[i], signature="default",as_dict=True)["elmo"]
#     print("embeddings is type {} and shape {}".format(type(embeddings), embeddings.shape))
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings_2d = session.run(tf.reduce_mean(embeddings_3d,axis=0))
#         print("message_embeddings_2d is type {} and shape {}".format(type(message_embeddings_2d), message_embeddings_2d.shape))
        embeddings_1d = session.run(tf.reduce_mean(tf.convert_to_tensor(embeddings_2d), axis = 0, keepdims=True))
#         print("message_embeddings_1d is type {} and shape {}".format(type(message_embeddings_1d), message_embeddings_1d.shape))
        resume_embeddings.extend(embeddings_1d)
        print('extended with elmo_in_smalls[{}].format()')


    



# save elmo in smalls
np_elmo_in_smalls = np.asarray(elmo_in_smalls)
np.save('ELMo_embeddings_resumes', np_elmo_in_smalls) 

# save np array of embeddings
np_ELMo_embeddings_resumes = np.asarray(resume_embeddings)
np.save('ELMo_embeddings_resumes', np_ELMo_embeddings_resumes) 

# save elmo_train_new
pickle_out = open("ELMo_embeddings_resumes.pickle","wb")
pickle.dump(np_ELMo_embeddings_resumes, pickle_out)
pickle_out.close()
