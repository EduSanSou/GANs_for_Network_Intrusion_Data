import numpy as np
import pandas as pd
import tensorflow as tf

from utils import preprocessing
from utils import utils
from models import classifiers as clf
from models import cgan

import matplotlib.pyplot as plt

def main(arg):
    #-------------------- Load Data & Preprocess ---------------------#
    train,test, label_mapping = preprocessing.get_data(encoding="Label")
    data_cols = list(train.columns[ train.columns != 'label' ])

    #Remove contant values with a threshold
    #to_drop = preprocessing.get_contant_featues(train,data_cols,threshold=0.995)

    #train.drop(to_drop, axis=1,inplace=True)
    #test.drop(to_drop, axis=1,inplace=True)

    #Normalize row-wise the data (to unit norm) and scale column-wise
    data_cols = list(train.columns[train.columns != 'label' ])
    train = preprocessing.normalize_data(train,data_cols)
    test = preprocessing.normalize_data(test,data_cols)
    x_train , x_test, scaler = preprocessing.preprocess(train,test,data_cols,"Robust",False)
    data_cols = list(x_train.columns[x_train.columns != 'label' ])

    train, test = None, None
    y_train = x_train.label.values
    y_test = x_test.label.values

    att_ind = np.where(x_train.label != label_mapping["normal"])[0]
    for_test = np.where(x_test.label != label_mapping["normal"])[0]
    
    del label_mapping["normal"]
    clf.DISPLAY_PERFOMANCE = False

    # train Ml classifiers
    #print("Training classifiers : [Started]")
    #svm = clf.svm(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    #randf = clf.random_forest(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    #nn = clf.neural_network(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping,False)
    #deci = clf.decision_tree(x_train[data_cols].values[att_ind], y_train[att_ind], x_test[data_cols].values[for_test], y_test[for_test],label_mapping)
    #print("Training classifiers : [Finished]")
    #
    #Save ML trained models to disk models
    #utils.save_classifiers([svm,randf,nn,deci])
    #print("Classifiers save to disk : [SUCCESSFUL]")

    x = x_train[data_cols].values[att_ind] #x_train.query(f'label == {label_mapping["normal"]}').values
    y = y_train[att_ind]
    x_train, y_train = None, None

    #Define, Train & Save GAN
    print("GAN Training Starting ....")
    model = cgan.CGAN(arg,x,y.reshape(-1,1))
    model.train()
    #print("label_mapping:",label_mapping)
    #print(model.generate_data(np.array([0,2,3,4])))
    model.dump_to_file()
    print("GAN Training & Save [SUCCESSFUL]")
    
    #Plot GAN training logs
    gan_path = f"./logs/CGAN_{model.gan_name}.pickle"
    utils.plot_training_summary(gan_path,'./imgs')

    #Generate samples
    #label_mapping: {'dos': 0, 'probe': 2, 'r2l': 3, 'u2r': 4}
    labels = np.array([0,0,0,0])
    n = len(labels)
    rand_noise_dim = 32
    noise = np.random.normal(0,1,(n,rand_noise_dim))
    filepath = 'trained_generator/gen.h5'
    model = tf.keras.models.load_model(filepath)
    processed_samples = model.predict([noise,labels])
    unlabeled_processed_samples = processed_samples[:,:-1]
    print(unlabeled_processed_samples)
    new_samples = preprocessing.revert_scale(unlabeled_processed_samples,scaler)
    print(new_samples)

if __name__ == '__main__':
    gan_params = [32, 4,200, 128 , 1, 1, 'relu', 'sgd', 0.00005, 27]
    main(gan_params)
