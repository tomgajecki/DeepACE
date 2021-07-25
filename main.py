 # -*- coding: utf-8 -*-
"""
DeepACE
===================================================================================
Copyright (c) 2021, Deutsches HörZentrum Hannover, Medizinische Hochschule Hannover
Author: Tom Gajecki (gajecki.tomas@mh-hannover.de)
All rights reserved.
===================================================================================
"""
 
import os
import sys
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import json
import warnings
from utils import setup
from model import DeepACE
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple
from data_generator import DataGenerator

def train(args):
        
    time_stamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    model_dir = os.path.join(args.model_dir, (args.topology + '_' + 'model[{}]').format(time_stamp))
    
    train_ds = DataGenerator("train", args).fetch()
       
    valid_ds = DataGenerator("valid", args).fetch()
   
    opt = tf.optimizers.Adam()
    
    os.makedirs(model_dir)
    
    args.mode = "test"
    
    json.dump(vars(args), open(os.path.join(model_dir,'params.json'), 'w'), indent=4)
    
    log_path = os.path.join(model_dir, 'train_log[{}].csv'.format(time_stamp))
    
    checkpoint_path = os.path.join(model_dir, 'model_best[{}].h5'.format(time_stamp))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = .8, patience=3, min_lr = 0.),
        tf.keras.callbacks.CSVLogger(log_path),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                           monitor='val_loss', verbose=0, save_best_only=True, 
                                           mode='auto', save_freq='epoch')]
    
    physical_devices = tf.config.list_physical_devices()
    
    if len(physical_devices) == 1:
        print("\nUsing CPU... \n")
        warnings.warn("Depwthwise convolution not available when using CPU for processing.")
        args.GPU = False
    
    else:
        print("\nUsing GPU... \n")
   
    model = DeepACE(args).call()
    
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file = os.path.join(model_dir,'model.pdf'), show_layer_names = False)
    
    model.compile(optimizer=opt, loss = 'MSE')
    
    history = model.fit(train_ds, validation_data = valid_ds, 
                    epochs = args.max_epoch, callbacks = callbacks)

    plt.style.use('ggplot')
    
    history_dict = history.history
    
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(14,5))
    plt.plot(epochs, loss, marker='.', label='Training')
    plt.plot(epochs, val_loss, marker='.', label='Validation')
    plt.title('Training and validation performance')
    plt.xlabel('Epoch')
    plt.grid(axis='x')
    plt.ylabel('MSE')
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(model_dir,'learning_curves.pdf'))

def evaluate(args, model):         
    
    model_time_stamp = model[-20:-1]
   
    model_dir = os.path.join(args.model_dir, model)
    
    log_path = os.path.join(model_dir, 'test_log[{}].csv'.format(model_time_stamp))
       
    params_file = os.path.join(model_dir, "params.json")
    
    with open(params_file) as f:
        args = json.load(f)
                
    args = namedtuple("args", args.keys())(*args.values())
    
    test_ds = DataGenerator("valid", args).fetch()
    
    physical_devices = tf.config.list_physical_devices("GPU")
    
    if not physical_devices:
        args.GPU = False

    model = DeepACE(args).call()

    
    model.compile(loss = 'MSE')
    
    model.load_weights(os.path.join(model_dir, "model_best[{}].h5".format(model_time_stamp)))
    
    test_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)
    test_logger.on_test_begin = test_logger.on_train_begin
    test_logger.on_test_batch_end = test_logger.on_epoch_end
    test_logger.on_test_end = test_logger.on_train_end
    
    print("\nEvaluating model... \n")

    mse = model.evaluate(test_ds, callbacks = test_logger, verbose = 0)
    
    print("Evaluation MSE:", round(mse,2))
         
def main():
    args = setup()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        trained_models = os.listdir(args.model_dir)
        for tm in  trained_models:
            evaluate(args, tm)

if __name__ == '__main__':
    main()  
