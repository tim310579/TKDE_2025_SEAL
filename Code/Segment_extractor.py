import os
import numpy as np
import utils.io as uio

from tqdm import tqdm
from biosppy.signals import ecg
from biosppy.signals import tools
from configs.data_config import DataConfig

import matplotlib.pyplot as plt

import sys
import dataset_setting.select_dataset as data_setting


def get_median_filtered(signal, threshold=10):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal


def check(config):

    input_path = os.path.join(config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir)

    output_path = os.path.join(config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               config.snippet_dir)

    print(input_path)

    data, Y, labels, classes = uio.load_formmated_raw_data(
        input_path, "all", tmp_path)
    
    print(labels[:10])
    print(Y[:10])
    print(classes)


def execute(config, class_name, label_num):

    input_path = os.path.join(config.model_root_dir, #config.root_dir,
                              config.data_dir,
                              config.dataset_name) + "/"

    tmp_path = os.path.join(config.model_root_dir, #config.root_dir,
                            config.data_dir,
                            config.dataset_name,
                            config.tmp_dir + 'label_' + str(label_num))

    output_path = os.path.join(config.model_root_dir, #config.root_dir,
                               config.data_dir,
                               config.dataset_name,
                               'processed_data_' + config.segmenter + '_' + str(config.window_size)
                               )

    processed_data_path = os.path.join(config.model_root_dir, #config.root_dir,
                               config.data_dir,
                               config.dataset_name, 
                               'processed_data_' + config.segmenter + '_' + str(config.window_size)
                               )
    print(input_path, output_path, processed_data_path)
    
    processed_data = []
    output_data = []
    output_label = []
    
    raw_length = []
    hb_index = []
    output_length = []
    output_index = []
    
    if config.dataset_name == "Opportunity":
        data = np.load(input_path+'data.npy')
        all_y = np.load(input_path+'labels.npy')
        
        data = uio.resize(data, data.shape[1], data.shape[2])
        data = np.transpose(data, (0, 2, 1))
        
        tmp_y = all_y[:,label_num-1]
        Y = np.zeros((len(tmp_y),2))
        
        for i in range(len(tmp_y)):
            Y[i, int(tmp_y[i])] = 1
        
        min_length=99999
        
        #if label_num == 1:
        for raw_row in tqdm(data):
            min_length=min(min_length, raw_row.shape[0])
        
            tmp = []
            hbs = []
            start=0
            
            while(1):
                tmp.append(raw_row[int(start): int(start+config.window_size*2)])
                hbs.append((int(start)+int(start+config.window_size*2))/2)
                start += config.window_size*2 
                if (start+config.window_size*2) > raw_row.shape[0]:
                    break
                
            
            tmp = np.array(tmp)
            raw_length.append(raw_row.shape[0])
            hb_index.append(hbs)
            processed_data.append(tmp)
    
        print('processed_data', len(processed_data), min_length)         
        for idx, row in tqdm(enumerate(processed_data)):
    
            if(len(row) < 1):
                print(idx, '->', row.shape)
            else:
                output_data.append(row)
                output_label.append(Y[idx])
                output_index.append(hb_index[idx])
                output_length.append(raw_length[idx])
        
        output_data = np.array(output_data)
        output_label = np.array(output_label)
        output_index = np.array(output_index)
        output_length = np.array(output_length)
    
        output_dict_data = {
            "data": output_data,
            "index": output_index,
            "length": output_length,
        }
        output_dict_label = {
            "label": output_label,
        }
        
        uio.check_folder(output_path)
        if label_num == 1: uio.save_pkfile(output_path+'/'+config.snippet_name, output_dict_data)
        uio.save_pkfile(output_path+'/%s_%d.pickle'%(config.segmenter, label_num), output_dict_label)
        
                
    else: #CPSC_extra, Chapman, PTBXL
        data, Y, labels, classes = uio.load_formmated_raw_data(
            input_path, "all", tmp_path, class_name, label_num)
    
        if label_num == 1:
            min_length = 99999
            
            for raw_row in tqdm(data):
                peaks = []
                tmp_norm = tools.normalize(raw_row, ddof=2)
                
                if (config.segmenter is "christov"):
                    peaks = ecg.christov_segmenter(signal=raw_row.T[0],
                                                sampling_rate = 500)[0]
                    
                    if(len(peaks)<=1):
                        la_peaks = ecg.christov_segmenter(signal=raw_row[peaks[0]+250:, 0],
                                                sampling_rate = 500)[0]
                        peaks = [(x+250) for x in la_peaks]
                    
    
                elif (config.segmenter is "hamilton"):
                    peaks = ecg.hamilton_segmenter(signal=raw_row.T[0],
                                                sampling_rate = 500)[0]
                                                
                    if(len(peaks)==1):
                        la_peaks = ecg.hamilton_segmenter(signal=raw_row[peaks[0]+250:, 0],
                                                sampling_rate = 500)[0]
                        peaks = [(x+250) for x in la_peaks]
                        
                    if(len(peaks)==0):
                        la_peaks = ecg.hamilton_segmenter(signal=raw_row[250:, 0],
                                                sampling_rate = 500)[0]
                        peaks = [(x+250) for x in la_peaks]
                       
                elif (config.segmenter is "engzee"):
                    peaks = ecg.engzee_segmenter(signal=raw_row.T[0],
                                                sampling_rate = 500, threshold=0.32)[0]
                        
                    if(len(peaks)<=1):
                        print('+++')
                        la_peaks = ecg.engzee_segmenter(signal=raw_row[250:, 0],
                                                sampling_rate = 500, threshold=0.32)[0]
                        peaks = [(x+250) for x in la_peaks]
                
                elif (config.segmenter is "sliding_window"):
                    
                    min_length=min(min_length, raw_row.shape[0])
        
                    tmp = []
                    hbs = []
                    start=0
                    
                    while(1):
                        tmp.append(raw_row[int(start): int(start+config.window_size*2)])
                        hbs.append((int(start)+int(start+config.window_size*2))/2)
                        start += 375
                        if (start+config.window_size*2) > raw_row.shape[0]:
                            break
                        
                    tmp = np.array(tmp)
                    raw_length.append(raw_row.shape[0])
                    hb_index.append(hbs)
                    processed_data.append(tmp)
                    
            
                
                else:
                    peaks = ecg.gamboa_segmenter(signal=tmp_norm['signal'], 
                                                sampling_rate = 500)[0]
                    
                    
                hb = ecg.extract_heartbeats(signal=raw_row,
                                            rpeaks=peaks,
                                            sampling_rate=500,
                                            before=config.window_size/500, #0.25,
                                            after=config.window_size/500 #0.25
                                            )
                                            
                    
                if (config.segmenter is "sliding_window"):
                    pass
                    
                else:
                
                    raw_length.append(len(tmp_norm['signal']))
                    hb_index.append(hb['rpeaks'])
                    processed_data.append(hb[0])    
                    #print(hb[0].shape)                
                        
            uio.check_folder(processed_data_path)
            uio.save_pkfile(processed_data_path+'/processed_data.pkl', processed_data)
            uio.save_pkfile(processed_data_path+'/hb_index.pkl', hb_index)
            uio.save_pkfile(processed_data_path+'/raw_length.pkl', raw_length)
    
        else:
            processed_data = uio.load_pkfile(processed_data_path+'/processed_data.pkl')
            hb_index = uio.load_pkfile(processed_data_path+'/hb_index.pkl')
            raw_length = uio.load_pkfile(processed_data_path+'/raw_length.pkl')
    
        print('processed_data', len(processed_data))
        for idx, row in tqdm(enumerate(processed_data)):
    
            if(len(row) < 1):
                print(idx, '->', row.shape)
            else:
                output_data.append(row)
                output_label.append(Y[idx])
                output_index.append(hb_index[idx])
                output_length.append(raw_length[idx])
        
        output_data = np.array(output_data)
        output_label = np.array(output_label)
        output_index = np.array(output_index)
        output_length = np.array(output_length)
    
        output_dict_data = {
            "data": output_data,
            "index": output_index,
            "length": output_length,
        }
        output_dict_label = {
            "label": output_label,
            "info":labels
        }
        uio.check_folder(output_path)
        if label_num == 1: uio.save_pkfile(output_path+'/'+config.snippet_name, output_dict_data)
        uio.save_pkfile(output_path+'/%s_%d.pickle'%(config.segmenter, label_num), output_dict_label)


if __name__ == "__main__":
    use_dataset_name = 'CPSC_extra'

    if len(sys.argv) > 1:
        use_dataset_name = sys.argv[1]
    
    use_total_length, class_name, dims, num_classes = data_setting.recall_params(use_dataset_name)
    sampling_rate, window_size = data_setting.return_sample_window_size(use_dataset_name)
    label_nums = data_setting.return_label_num(use_dataset_name)

    use_snippet_name = "christov.pickle"
    use_segmenter = "christov"

    if use_dataset_name == "Opportunity":
        use_snippet_name = "sliding_window.pickle"
        use_segmenter = "sliding_window"
    
    config = DataConfig(snippet_name = use_snippet_name, 
                        dataset_name = use_dataset_name, segmenter = use_segmenter, sampling_rate = sampling_rate, window_size=window_size)
    
    for i in range(1, label_nums+1):
        execute(config, class_name, i)
