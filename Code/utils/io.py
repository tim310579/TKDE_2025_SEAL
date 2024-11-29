import os
import errno
import csv
import glob
import pickle
import scipy.io
import numpy as np
from . import utils
from tqdm import tqdm
from biosppy.signals import tools

def load_formmated_raw_data(inputfolder, task, outputfolder, class_name, label_num, sampling_frequency=500): #, load_all_label=False):

    data,raw_labels = utils.load_dataset(inputfolder, sampling_frequency, label_num)

    labels = raw_labels
    
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(
        data, labels, task, class_name, label_num, min_samples=0, outputfolder=outputfolder)

    return data, Y, labels, _.classes_
    
def load_snippet_data_with_il_all_label(config, input_data_folder, label_num, window_size=250):
    
    pickle_in = open(input_data_folder, "rb")
    
    data = pickle.load(pickle_in)

    all_y = []
    for i in range(1, label_num+1):
        input_label_folder = os.path.join(#config.root_dir,
                                            config.model_root_dir,
                                            config.data_dir,
                                            config.dataset_name,
                                            'processed_data_'+config.snippet_name.split('.')[0] + '_' + str(window_size),
                                            '%s_%d.pickle'%(config.snippet_name.split('.')[0],i))
             
        if window_size == 250 or window_size == 20:
            input_label_folder = os.path.join(#config.root_dir,
                                            config.model_root_dir,
                                            config.data_dir,
                                            config.dataset_name,
                                            'processed_data_'+config.snippet_name.split('.')[0],
                                            '%s_%d.pickle'%(config.snippet_name.split('.')[0],i))
                                            
                                                                           
        pickle_in_label = open(input_label_folder, "rb")
        
        label = pickle.load(pickle_in_label)

        Y = label['label']
        
        all_y.append(np.argmax(Y, axis=1))

    all_y = np.array(all_y).T

    X = data['data']

    I = data['index']

    L = data['length']

    return X, all_y, I, L

def check_folder(path):

    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Create : ", path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        print(path, " exists")


def load_pkfile(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data_in = pickle.load(pickle_in)

    pickle_in.close()

    return data_in

def save_pkfile(outputfolder, data):

    pickle_out = open(outputfolder, "wb")

    pickle.dump(data, pickle_out, protocol=4)

    pickle_out.close()

    print(outputfolder, "saving successful !")
    

def resize(raw_data,length,dims, ratio=1):
    input_data = np.zeros((len(raw_data),int(length*ratio),dims))
    for idx, data in enumerate(raw_data):
        input_data[idx,:data.shape[0],:data.shape[1]] = tools.normalize(data[0:int(length*ratio),:])['signal']

    input_data = np.transpose(input_data, (0, 2, 1))
    return input_data