import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
import csv
import wandb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import utils.io as uio
from scipy import signal


from tqdm import tqdm
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from core.loss import FocalLoss
from core.model import SEAL
from configs.seal_config import Config
import sys
import dataset_setting.select_dataset as data_setting
import collections

from torchsummary import summary                         


torch.set_num_threads(3)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#
setup_seed(0) #0



def execute(label_num, config, training_data, training_label, testing_data, testing_label, 
            training_length = None, training_index = None,
            testing_length = None, testing_index = None,
            val_X = None, val_Y=None, wandb=None, model_path=None,dense1_input=None, embedding_H_t=None, IDL_type=None, backbone=None,
            imb=None, gnn_type=None, cnn_attn=None, window_size=None):
       
    df_l = pd.DataFrame(training_label)
    Corr_matrix = torch.from_numpy(df_l.corr().values).float().cuda()
    
    
    Adj_matrix = training_label.T.dot(training_label)
    
    Adj_matrix = Adj_matrix.astype(float)
    for i in range(len(Adj_matrix)):
        for j in range(len(Adj_matrix)):
            Adj_matrix[i,j] /= (Adj_matrix[i,i]+Adj_matrix[j,j])

    Adj_matrix[Adj_matrix < 0.02] = 0
    
    np.fill_diagonal(Adj_matrix, 1)
    
    Adj_matrix = torch.from_numpy(Adj_matrix).float().cuda()
    embedding_H_t = embedding_H_t.float().cuda()

    model = SEAL(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            hidden_output_size=config.hidden_output_size,
                            output_size=label_num,
                            dense1_input=dense1_input,
                            Adj_matrix=Adj_matrix,
                            Corr_matrix=Corr_matrix,
                            embedding_H_t=embedding_H_t,
                            IDL_type=IDL_type,
                            backbone=backbone,
                            #batch=config.batch_size
                            gnn_type=gnn_type,
                            cnn_attn=cnn_attn,
                            window_size=window_size
                            )
    
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
        
    result_model = None
    
    train_losses = []
    
    
    for epoch in range(config.epoch_size):
            
        print('Epoch: ', epoch)
        
        loss_sum = 0
        loss_c = 0
        loss_r = 0
        loss_b = 0
        loss_p = 0
        
        model.train()

        len_t = int(len(training_data)/config.batch_size)
        s = np.arange(training_data.shape[0])
        np.random.shuffle(s)

        training_data = training_data[s]
        
        training_label = training_label[s]
        
        for idx in tqdm(range(len_t)):
            X = training_data[idx*config.batch_size:idx * config.batch_size+config.batch_size]
            
            input_label = torch.from_numpy(
                training_label[idx*config.batch_size:idx*config.batch_size+config.batch_size]).cuda()

            
            ##########
            predictions, t = model(X, embedding_H_t=embedding_H_t)

            loss, c, r, b, p = model.applyLoss( # v
                predictions, input_label, beta=config.beta, loss_weight=None, n_js=None )

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            loss_sum += loss.item()
            loss_c += c.item()
            loss_r += r.item()
            loss_b += b.item()
            loss_p += p.item()
              
        print('loss_sum:', loss_sum, 'loss_c:', loss_c, 'loss_r:', loss_r, 'loss_b:', loss_b, 'loss_p:', loss_p)
        
        if wandb is not None:
            wandb.log({'loss_sum': loss_sum,
                'loss_c': loss_c,
                'loss_r': loss_r,
                'loss_b': loss_b,
                'loss_p': loss_p,
                'epoch': epoch,
                
            })
            
        scheduler.step(loss_sum)
        
        result_model = model
        
        if epoch % 50 == 0 and epoch > 0:
            torch.save(result_model.state_dict(), model_path+"/model"+str(epoch)+".pt")
      
    return result_model

if __name__ == "__main__":

    use_dataset_name = 'CPSC_extra' # CPSC_extra
    
    
    if len(sys.argv) > 1:
        use_dataset_name = sys.argv[1]
    
    
    # concept_net
    embedding_method = 'concept_net'
    # CNN, Informer, TCNcatCNN
    IDL_type = 'Informer' #'Informer'
    # Transformer, GRU, LSTM
    backbone = 'LSTM'
    # GCN, GAT, GIN, Linear
    gnn_type = 'GAT'

    cnn_attn = 'cbam' 
    
    use_total_length, class_name, dims, num_classes = data_setting.recall_params(use_dataset_name)
    sampling_rate, window_size = data_setting.return_sample_window_size(use_dataset_name)
    dense1_input = data_setting.return_dense1_input(use_dataset_name, window_size)
    label_nums = data_setting.return_label_num(use_dataset_name)
    embedding_H_t = data_setting.return_embedding_H_t(use_dataset_name, embedding_method)
    
    
    w_and_b = 0
    
    if w_and_b == 0:
        wandb = None
    else:
        wandb.init(project="CIKM23",
            entity='entity',
            name = 'SEAL' + use_dataset_name,
            
            reinit=True)
        wandb.log({'model': 'SEAL',
                'use_dataset_name': use_dataset_name,
                'tip': 'SEAL_training',
                'embedding_method': embedding_method,
                'IDL_type': IDL_type,
                'BaseRNN': backbone,
                'GNN_Type': gnn_type,
                'CNN_attn': cnn_attn,
                'Window_size': window_size
                })
    
    #wandb=None
    '''
    **********************************************************
    window_size mean depart from one time point, and extend to left and right with the 'window_size'****
    window_size*2 is the real window size
    **********************************************************
    '''
            
    use_snippet_name = "christov.pickle"
    BATCH_SIZE = 32
    if use_dataset_name == "Opportunity":
        use_snippet_name = "sliding_window.pickle"
    
    
    for seed in range(1, 11):
        label_num = label_nums
        config = Config(model_name="SEAL",
                        dataset_name = use_dataset_name,
                        hidden_size=256,
                        seed=seed, 
                        input_size = dims, 
                        output_size = num_classes,
                        batch_size=BATCH_SIZE, 
                        epoch_size=100,
                        snippet_name = use_snippet_name,
                        sampling_rate=sampling_rate,
                        window_size=window_size)

        input_data_folder = os.path.join(#config.root_dir,
                                    config.model_root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    'processed_data_'+config.snippet_name.split('.')[0]+ '_' + str(window_size),
                                    config.snippet_name)
                                    
        if window_size == 250 or window_size == 20:
            input_data_folder = os.path.join(#config.root_dir,
                                    config.model_root_dir,
                                    config.data_dir,
                                    config.dataset_name,
                                    'processed_data_'+config.snippet_name.split('.')[0], #+str(window_size),
                                    config.snippet_name)

        model_path = os.path.join(#config.root_dir,
                                config.model_root_dir,
                                config.output_dir,
                                config.model_dir,
                                config.model_name,
                                config.dataset_name,
                                "%s_%s_%s_%s"%(IDL_type, backbone, gnn_type, cnn_attn),
                                str(config.seed),
                                )
        if window_size != 250 and window_size != 20:
            model_path = os.path.join(#config.root_dir,
                                config.model_root_dir,
                                config.output_dir,
                                config.model_dir,
                                config.model_name,
                                config.dataset_name,
                                "%s_%s_%s_%s_%d"%(IDL_type, backbone, gnn_type, cnn_attn, window_size*2),
                                str(config.seed),
                                )

        if wandb is not None: 
            wandb.log({'model_path': str(model_path).split('/')[-2]})
        
        has_pretrained = model_path+"/model.pt"
         
        if os.path.exists(has_pretrained) == True:
            print('Seed ' + str(config.seed) + ' exists!!')
            continue # has trained model
        
        uio.check_folder(model_path)
        
        X, all_y, I, L = uio.load_snippet_data_with_il_all_label(config, input_data_folder, label_num, window_size=window_size)
        #X[X!=X] = 0

        sss = KFold(
            n_splits=10, random_state=0, shuffle=True)

        sss.get_n_splits(X)

        for index, (train_val_index, test_index) in enumerate(sss.split(X, all_y)):
            if(index is (seed-1)):
                print("Runing:",seed)
                break

        train_val_data, testing_data = X[train_val_index], X[test_index]

        train_val_label, testing_label = all_y[train_val_index], all_y[test_index]

        training_index, testing_index = I[train_val_index], I[test_index]

        training_length, testing_length = L[train_val_index], L[test_index]

        
        result_model = execute(label_num, 
                            config, 
                            train_val_data, 
                            train_val_label,
                            testing_data, 
                            testing_label, 
                            testing_index = testing_index,
                            testing_length = testing_length,
                            wandb=wandb, 
                            model_path=model_path,
                            dense1_input=dense1_input,
                            embedding_H_t=embedding_H_t,
                            IDL_type=IDL_type,
                            backbone=backbone,
                            imb=None,
                            gnn_type=gnn_type,
                            cnn_attn=cnn_attn,
                            window_size=window_size
                            )
        
        
        torch.save(result_model.state_dict(), model_path+"/model.pt")
        
    if wandb is not None:
        wandb.finish()
