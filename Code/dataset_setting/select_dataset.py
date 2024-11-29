import numpy as np
import torch
import torch.nn as nn

def recall_params(use_dataset_name):

    use_total_length = 5000
    class_name=''
    dims = 12
    num_classes = 5
    
    if use_dataset_name == "CPSC_extra":
        use_total_length = 30000
        class_name = 'CPSC_extra'
        dims = 12
        num_classes = 2

    elif use_dataset_name == "Chapman":
        use_total_length = 5000
        class_name = 'Chapman'
        dims = 12
        num_classes = 2
    
    elif use_dataset_name == "Opportunity":
        use_total_length = 300
        class_name = 'Opportunity'
        dims = 242
        num_classes = 2

    return use_total_length, class_name, dims, num_classes


def return_sample_window_size(use_dataset_name):
    sampling_rate=0
    window_size=0

    if use_dataset_name == "CPSC_extra":
        sampling_rate=500
        window_size=250 #250 125 250 375 500 -> 0.5sec, 1sec, 1.5sec, 2sec

    elif use_dataset_name == "Chapman":
        sampling_rate=500
        window_size=250 #250 125 250 375 500 -> 0.5sec, 1sec, 1.5sec, 2sec
        
    elif use_dataset_name == "Opportunity":
        sampling_rate=30
        window_size=20 #15 #25 #20 #30 ### -> 1, 1.33, 1.67, 2 secs
        #christov:60
        #sliding_window:40
        #engzee:30
        #hamilton:50
        
    return sampling_rate, window_size


def return_dense1_input(use_dataset_name, window_size):
    dense1_input=0

    if window_size==250: dense1_input=1024
    elif window_size==125: dense1_input=512
    #elif window_size==375: dense1_input=1024
    elif window_size==375: dense1_input=1536
    elif window_size==500: dense1_input=2048
    
    elif window_size==15: dense1_input=512
    elif window_size==30: dense1_input=1024
    elif window_size==25: dense1_input=512
    elif window_size==20: dense1_input=512
        
    return dense1_input


def return_label_num(use_dataset_name):
    if use_dataset_name == "CPSC_extra": # <100 no
        return 13
    elif use_dataset_name == "Chapman": # <300 no
        return 12
    elif use_dataset_name == "Opportunity": # <100 no
        return 30
                 
def return_embedding_H_t(use_dataset_name, embedding_method):

    if use_dataset_name == "Opportunity":
        pass
    else:
        print(embedding_method)
    
    labels = []
    label_dim = 11
    
    if use_dataset_name == "CPSC_extra":
        labels = ['old myocardial infarction',
            'sinus tachycardia',
            'myocardial ischemia',
            'nonspecific st t abnormality',
            'premature ventricular contractions',
            'myocardial infarction',
            'st interval abnormal',
            'chronic myocardial ischemia',
            'bradycardia',
            'left ventricular hypertrophy',
            'complete right bundle branch block',
            'atrial fibrillation',
            '1st degree av block']
        label_dim = 13
            
    elif use_dataset_name == "Chapman":
        labels = ['atrial fibrillation',
            'right bundle branch block',
            't wave abnormal',
            'sinus bradycardia',
            'atrial flutter',
            'st depression',
            'nonspecific st t abnormality',
            'sinus rhythm',
            'left ventricular high voltage',
            'sinus tachycardia',
            'left axis deviation',
            'supraventricular tachycardia']
        label_dim = 12
        
    elif use_dataset_name == "Opportunity":
        label_dim = 30
        labels = ['Stand',
            'Walk',
            'Sit',
            'Lie',
            'unlock',
            'lock',
            'close',
            'reach',
            'open',
            'sip',
            'clean',
            'release',
            'move',
            'Salami',
            'Bread',
            'Dishwasher',
            'Switch',
            'Milk',
            'Drawer',
            'Knife',
            'Drawer',
            'Table',
            'Glass',
            'Cheese',
            'Door',
            'Door',
            'Drawer',
            'Fridge',
            'Cup',
            'Knife']
        
    if embedding_method == 'concept_net':
        embeddings = np.loadtxt('embeddings/%s_concept_net_embedding.txt'%use_dataset_name, delimiter=' ')
        embeddings = torch.from_numpy(embeddings).float().cuda()
            
        #print(embeddings, embeddings.shape)
        
        return embeddings
        
