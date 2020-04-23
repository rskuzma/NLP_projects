###### data science
import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.metrics.pairwise import cosine_similarity
import random
from matplotlib import pyplot as plt

#NLP
#import spacy
import re

import time
from tqdm import tqdm  # progress bar

# file loading
import pickle

#deep learning
import tensorflow as tf
import tensorflow_hub as hub

from time import process_time
import resource

### try this right after imports to load elmo_in_smalls
elmo_in_smalls = np.load('/home/ec2-user/NLP_projects/job_recommender_project/elmo_resumes_under_100_sentences.npy', allow_pickle=True).tolist()


##################reset default graph tf###################
tf.reset_default_graph()
############################################

## this was the instantiation of elmo
elmo = hub.Module("/home/ec2-user/module/module_elmo2", trainable=False)

# for all embeddings
# resume_embeddings_append_regular = []
resume_embeddings_append_transpose = []
# resume_embeddings_extend_regular = []
# resume_embeddings_extend_transpose = []


# print('\n\nstarting session\n\n')
t0 = process_time()
with tf.Session() as session:
    t00 = process_time()
#     print('\nstarting tf.Session took: {}'.format(t00-t0))
    
    t_pre_initialize_vars = process_time()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    t_post_initialize_vars = process_time()
#     print('Time to initialize variables: {}'.format(t_post_initialize_vars-t_pre_initialize_vars))

    
    
#     print('\n\nstarting for loop\n\n')
    for i in tqdm(range(1050, len(elmo_in_smalls))):
        t1 = process_time()
        # save np array of embeddings
        if i % 50 == 0:
            np_ELMo_embeddings_resumes = np.asarray(resume_embeddings_append_transpose)
            np.save('ELMo_embeddings_resumes_1050_to_{}'.format(i), np_ELMo_embeddings_resumes) 

            
            
            
#         assign_op = x.assign(1)
#         sess.run(assign_op)  # or `assign_op.op.run()`
#         print(x.eval())
    
#         print('elmo_in_smalls[{}]'.format(i))
        embeddings_3d = elmo(elmo_in_smalls[i], signature="default",as_dict=True)["elmo"]
#         print('3d type original', type(embeddings_3d))
        t_make_np_start = process_time()
        embeddings_3d_np = embeddings_3d.eval()
        t_make_np_stop = process_time()          
#         print('3d type np?', type(embeddings_3d_np))
        
        
        
        embeddings_3d_np_f16 = embeddings_3d_np #tf.dtypes.cast(embeddings_3d_np, tf.float16)
#         print('3d size: {}'.format(embeddings_3d_np_f16.shape))
#         print('3d type np? 16?', type(embeddings_3d_np_f16))
        t2 = process_time()

        
        embeddings_2d = np.mean(embeddings_3d_np_f16,axis=0)#session.run(tf.reduce_mean(embeddings_3d_np_f16,axis=0))
#         print('2d size: {}'.format(embeddings_2d.shape))
#         print('2d type ', type(embeddings_2d))
        t3 = process_time()


        embeddings_1d = np.mean(embeddings_2d,axis=0)#session.run(tf.reduce_mean(tf.convert_to_tensor(embeddings_2d), axis = 0, keepdims=True))
#         print('1d size: {}'.format(embeddings_1d.shape))
#         print('1d type ', type(embeddings_1d))
        t4 = process_time()

        transpose = embeddings_1d.T
#         print(transpose.shape)
#         print('transpose type: ', type(transpose))
        t5 = process_time()

#         resume_embeddings_append_regular.append(embeddings_1d)
        t6 = process_time()

        resume_embeddings_append_transpose.append(transpose)
        t7 = process_time()

#         resume_embeddings_extend_regular.extend(embeddings_1d)
        t8 = process_time()

#         resume_embeddings_extend_transpose.extend(transpose)
        t9 = process_time()

#         print('length of resume_embeddings_append_regular: {}'.format(len(resume_embeddings_append_regular)))
#         print('length of resume_embeddings_append_transpose: {}'.format(len(resume_embeddings_append_transpose)))
#         print('length of resume_embeddings_extend_regular: {}'.format(len(resume_embeddings_extend_regular)))
#         print('length of resume_embeddings_extend_transpose: {}'.format(len(resume_embeddings_extend_transpose)))      

#         print('shape of resume_embeddings_append_regular[{}]: {}'.format(i, resume_embeddings_append_regular[i].shape))
#         print('shape of resume_embeddings_append_transpose[{}]: {}'.format(i, resume_embeddings_append_transpose[i].shape))                                                                        
#         print('shape of resume_embeddings_extend_regular[{}]: {}'.format(i, resume_embeddings_extend_regular[i].shape))                                                                        
#         print('shape of resume_embeddings_extend_transpose[{}]: {}'.format(i, resume_embeddings_extend_transpose[i].shape))                                                                        

        #not .extend?



#         print('\nRun {}'.format(i))
        
#         print('Iteration ', i, ' maxrss: ', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        
        print('\nall times from start of run {}\n'.format(i))
        print('3d embedding time: {}'.format(t_make_np_start - t1))
        print('3d tensor to np time: {}'.format(t_make_np_stop - t_make_np_start))
        print('3d make 16b time: {}'.format(t2-t_make_np_stop))
        print('2d embedding time: {}'.format(t3 - t2))
        print('1d embedding time: {}'.format(t4 - t3))
        print('time to transpose (t5-t4): {}'.format(t5 - t4))
#         print('append_regular time (t6-t5): {}'.format(t6-t5))
        print('append_transpose time (t7-t6): {}'.format(t7-t6))
#         print('extend_regular time (t8-t7): {}'.format(t8-t7))
#         print('extend_transpose time (t9-t8): {}'.format(t9-t8))
        print('Total time run {}: {}'.format(i, t9-t1))         
        print('\n\n\n\t\t\tEND OF RUN')


    
    
    
