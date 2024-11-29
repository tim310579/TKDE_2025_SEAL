import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
import csv
import time
import wandb
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import utils.io as uio
import utils.utils as uu
from scipy import signal

from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score

from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error

from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold
from core.loss import FocalLoss
from core.model import SEAL
from configs.seal_config import Config
import sys
import dataset_setting.select_dataset as data_setting

torch.set_num_threads(3)

from sklearn.metrics import confusion_matrix

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 
setup_seed(0)


def execute(label_num, config, training_data, training_label, training_index, training_length,
            testing_data, testing_label, testing_index, testing_length, pretrained = None, wandb=None,dense1_input=None, embedding_H_t=None,
            IDL_type=None, backbone=None, alpha=None, imb=None, gnn_type=None, cnn_attn=None, window_size=None):
        
    
    print('Training Data Shape:', training_data.shape, training_label.shape)
    
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
                            alpha=alpha,
                            #batch=config.batch_size
                            gnn_type=gnn_type,
                            cnn_attn=cnn_attn,
                            window_size=window_size
                            )

    if (pretrained is not None):
        #try:
        model.load_state_dict(torch.load(pretrained), strict=False)
        
    model.cuda()
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    testing_predictions = []
    testing_labels = []
    testing_locations = []
    testing_predictions_prob = []
    
    
    for epoch in range(config.epoch_size):

        model.eval()
        
        len_t = int(len(testing_data)/config.batch_size)
        
        for idx in tqdm(range(len_t+1)): #+1
            with torch.no_grad():
                
                X = testing_data[idx*config.batch_size:idx * config.batch_size+config.batch_size]
                
                
                input_label = torch.from_numpy(testing_label[idx*config.batch_size:idx*config.batch_size+config.batch_size]).cuda()
    
                predictions, t = model(X, embedding_H_t=embedding_H_t)
                
                for every_prediction, every_label in zip(predictions, input_label):
                    testing_predictions_prob.append(every_prediction) 
                    
                    testing_predictions.append(every_prediction)
                    testing_labels.append(every_label)
                    
                for every_t in t:
                    tt_point = every_t
                    
                    testing_locations.append(tt_point)
            
    final_pred = []
    final_label = []
    final_pred_probs = []
    final_earliness = []
        
    threshold = 0.5 ##22 25
    
    if wandb is not None:
        wandb.log({ 'threshold': threshold,
            })
            
    for preds in testing_predictions:
    
        sth = preds.cpu().detach().numpy() 
       
        sth[sth < threshold] = 0
        sth[sth>= threshold] = 1
        
        final_pred.append(sth)

    for ys in testing_labels:
        sth = ys.cpu().detach().numpy()
        final_label.append(sth)

    for pred_probs in testing_predictions_prob:
        sth = pred_probs.cpu().detach().numpy()
        final_pred_probs.append(sth)

    final_pred = np.array(final_pred).T
    final_label = np.array(final_label).T.squeeze()
    final_pred_probs = np.array(final_pred_probs).T
            
    precisions = []
    recalls = []
    accuracys = []
    f1s = []

    AP = []

    Performance = []

    #-------------------label-based-----------------------------
    
    for y_true_, y_pred_ in zip(final_label, final_pred):
        precisions.append(precision_score(y_true_, y_pred_, average='binary', zero_division=0))
        recalls.append(recall_score(y_true_, y_pred_, average='binary', zero_division=0))
        f1s.append(f1_score(y_true_, y_pred_, average='binary', zero_division=0))
        accuracys.append(accuracy_score(y_true_, y_pred_))
    
    '''compute Earliness **********************************'''
    earliness_s = []
    latest_earliness = []
    label_earliness = []
    
    for idx, tt_point in enumerate(testing_locations):
        
        try:
            earliness_s.append((testing_index[idx][tt_point]+config.window_size-1)/(testing_length[idx]))
        except:
            earliness_s.append((np.array(testing_index[idx])[tt_point]+config.window_size-1)/(testing_length[idx]))
        latest_earliness.append(max(earliness_s[idx]))
    
    earliness_s = np.array(earliness_s).T
    
    for earliness in earliness_s:
        label_earliness.append(np.mean(earliness))

    latest_earliness = np.mean(latest_earliness)
    label_earliness = max(label_earliness)
    '''compute Earliness **********************************'''
    
    print('Start----------------------------')
    print('recalls:', recalls)
    print('precisions:', precisions)
    print('f1s:', f1s)

    print('Precision:', np.mean(precisions))
    print('Recall:', np.mean(recalls))
    print('F1:', np.mean(f1s))
    print('Accuracy:', np.mean(accuracys))
    print('Label Earliness:', label_earliness)
    f1_tmp = np.mean(np.mean(f1s))

    print('Harmonic Mean:', (2*(1-label_earliness)*f1_tmp) / (1-label_earliness+f1_tmp))
    
    if wandb is not None:
        wandb.log({ 'precision':np.mean(precisions),
            'recall':np.mean(recalls),
            'accuracy':np.mean(accuracys),
            'f1':np.mean(f1s),
            'earliness':label_earliness,
            'hm':(2*(1-label_earliness)*f1_tmp) / (1-label_earliness+f1_tmp)
            })
        
    Performance.append(np.mean(precisions))
    Performance.append(np.mean(recalls))
    Performance.append(np.mean(accuracys))
    Performance.append(np.mean(f1s))
    Performance.append(label_earliness)
    Performance.append((2*(1-label_earliness)*f1_tmp) / (1-label_earliness+f1_tmp))


    #-------------------sample-based-----------------------------

    accs = accuracy_score(final_label.T, final_pred.T)
    #pos_accs = []
    sample_precision = []
    sample_recall = []
    sample_f1 = []
    
    for y_true_, y_pred_ in zip(final_label.T, final_pred.T):

        sample_precision.append(precision_score(y_true_, y_pred_, average='binary', zero_division=0))
        sample_recall.append(recall_score(y_true_, y_pred_, average='binary', zero_division=0))
        sample_f1.append(f1_score(y_true_, y_pred_, average='binary', zero_division=0))

    sample_precision = np.mean(sample_precision)
    sample_recall = np.mean(sample_recall)
    sample_f1 = np.mean(sample_f1)
    
    hl = hamming_loss(final_label, final_pred)

    ranking_loss = []
    one_error = []
    coverage = []

    ranking_loss = label_ranking_loss(final_label.T, final_pred_probs.T)
    coverage = coverage_error(final_label.T, final_pred_probs.T)/label_nums

    for y_true_, y_pred_prob_ in zip(final_label.T, final_pred_probs.T):
        if y_true_[np.argmax(y_pred_prob_)]==1: one_error.append(0) # highest prob is true(positive)
        else: one_error.append(1)

    one_error = np.mean(one_error)

    print("----------------------------")
    print("Ranking_Loss:", ranking_loss)
    print("One_Error:", one_error)
    print('Coverage', coverage)
    print("Subset_Acc:", accs)
    #print("Sample_precision:", sample_precision)
    #print("Sample_recall:", sample_recall)
    #print("Sample_f1:", sample_f1)
    print("Pos_Acc:", sample_recall)
    print('Latest Earliness', latest_earliness)
    print('Hamming Loss', hl)
    sample_hm = (2*(1-latest_earliness)*accs) / (1-latest_earliness + accs)
    print('Sample HM', sample_hm)
    
    if wandb is not None:
        wandb.log({
            'ranking_loss': ranking_loss,
            'one_error': one_error,
            'coverage': coverage,
            'subset_accuracy':accs,
            #'sample_precision':sample_precision,
            #'sample_recall':sample_recall,
            #'sample_f1':sample_f1,
            'positive_accuracy':sample_recall,
            'latest_earliness':latest_earliness,
            'hamming_loss':hl,
            'sample_hm':sample_hm
            })
        
    Performance.append(ranking_loss)
    Performance.append(one_error)
    Performance.append(coverage)
    Performance.append(accs)
    Performance.append(sample_precision)
    Performance.append(sample_recall)
    Performance.append(sample_f1)
    #Performance.append(pos_accs)
    Performance.append(latest_earliness)
    Performance.append(hl)
    Performance.append(sample_hm)


    # #-------------------OVERALL-----------------------------

    all_y_preds = final_pred.flatten()
    all_y_trues = final_label.flatten()
    
    precision_O = precision_score(all_y_trues, all_y_preds, average='binary')
    recall_O = recall_score(all_y_trues, all_y_preds, average='binary')
    f1_O = f1_score(all_y_trues, all_y_preds, average='binary')
    mAP = label_ranking_average_precision_score(final_label, final_pred_probs)
    
    print('----------------------------')
    print('Precision_O:', precision_O)
    print('Recall_O:', recall_O)
    print('F1_O:', f1_O)
    print('mAP:', mAP)
    print('----------------------------')
    
    if wandb is not None:
        wandb.log({'precision_O':precision_O,
            'recall_O':recall_O,
            'f1_O':f1_O,
            'mAP':mAP
            })
        
    Performance.append(precision_O)
    Performance.append(recall_O)
    Performance.append(f1_O)
    Performance.append(mAP)


    return Performance


if __name__ == "__main__":

    use_dataset_name = 'CPSC_extra'

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
    
    #wandb = None
    w_and_b = 1
    
    if w_and_b == 0:
        wandb = None
    else:
    
        wandb.init(project="CIKM23",
            entity='entity',
            name = 'SEAL' + use_dataset_name,
            
            reinit=True)
        wandb.log({'model': 'SEAL',
                'use_dataset_name': use_dataset_name,
                'tip': 'SEAL_evaluation',
                'embedding_method': embedding_method,
                
                'IDL_type': IDL_type,
                'BaseRNN': backbone,
                'GNN_Type': gnn_type,
                'CNN_attn': cnn_attn
                })
            
    
    alphas=0.95
    
    use_snippet_name = "christov.pickle"
    BATCH_SIZE = 32
        
    if use_dataset_name == "Opportunity":
        use_snippet_name = "sliding_window.pickle"
        
    
    
    for each_alpha in [alphas*100]:#range(100, 101): # 75, 101

        alpha = each_alpha/100
        print('alpha:', alpha)
        
        All_performance = []
        
        for seed in range(1, 11):
            
            label_num = label_nums
    
    
            config = Config(model_name="SEAL",
                            dataset_name = use_dataset_name,
                            hidden_size=256,
                            seed=seed, 
                            input_size = dims, 
                            output_size = num_classes,
                            batch_size=BATCH_SIZE, 
                            epoch_size=1,
                            snippet_name = use_snippet_name,
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
                                        'processed_data_'+config.snippet_name.split('.')[0],
                                        config.snippet_name)
    
            model_path = os.path.join(#config.root_dir,
                                    config.model_root_dir,
                                    config.output_dir,
                                    config.model_dir,
                                    config.model_name,
                                    config.dataset_name,
                                    
                                    "%s_%s_%s_%s"%(IDL_type, backbone, gnn_type, cnn_attn),
                                    
                                    str(seed),
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
                       
            if w_and_b == 1: wandb.log({'model_path': str(model_path).split('/')[-2]})
            
            if seed == 10 and w_and_b == 1: wandb.log({'tip': 'SEAL_10fold_evaluation'})
                         
            uio.check_folder(model_path)
    
            X, all_y, I, L = uio.load_snippet_data_with_il_all_label(config, input_data_folder, label_num, window_size=window_size)
            X[X!=X] = 0
     
            sss = KFold(
                n_splits=10, random_state=0, shuffle=True)
    
            sss.get_n_splits(X)
    
            for index, (train_val_index, test_index) in enumerate(sss.split(X, all_y)):
                if(index is (seed-1)):
                    print("Runing:",seed)
                    break
    
            training_data, testing_data = X[train_val_index], X[test_index]
    
            training_label, testing_label = all_y[train_val_index], all_y[test_index]
    
            training_index, testing_index = I[train_val_index], I[test_index]
    
            training_length, testing_length = L[train_val_index], L[test_index]
        
            pretrained = model_path+"/model.pt"
             
            if os.path.exists(pretrained) == False:
                end # no model
            
            Performance = execute(label_num, 
                                      config, 
                                      training_data, 
                                      training_label, 
                                      training_index, 
                                      training_length,
                                      testing_data, 
                                      testing_label, 
                                      testing_index, 
                                      testing_length,
                                      pretrained = pretrained, 
                                      wandb=wandb,
                                      dense1_input=dense1_input,
                                      embedding_H_t=embedding_H_t,
                                      IDL_type=IDL_type,
                                      backbone=backbone,
                                      alpha = alpha,
                                      imb=None,
                                      gnn_type=gnn_type,
                                      cnn_attn=cnn_attn,
                                      window_size=window_size
                                      )
                                      
            All_performance.append(Performance)
                
                    
                
        All_performance = np.array(All_performance)
        
        
        import statistics
        
        if wandb is not None:
            wandb.log({
                'precision':statistics.mean(All_performance[:,0]),
                'recall': statistics.mean(All_performance[:,1]),
                'accuracy': statistics.mean(All_performance[:,2]),
                'f1': statistics.mean(All_performance[:,3]),
                'earliness' : statistics.mean(All_performance[:,4]),
                'hm' : statistics.mean(All_performance[:,5]),
        
                'ranking_loss': statistics.mean(All_performance[:,6]),
                'one_error': statistics.mean(All_performance[:,7]),
                'coverage': statistics.mean(All_performance[:,8]),
                'subset_accuracy': statistics.mean(All_performance[:,9]),
                'sample_precision': statistics.mean(All_performance[:,10]),
                'sample_recall': statistics.mean(All_performance[:,11]),
                'sample_f1': statistics.mean(All_performance[:,12]),
                
                'positive_accuracy': statistics.mean(All_performance[:,11]),
                
                'latest_earliness': statistics.mean(All_performance[:,13]),
                'hamming_loss': statistics.mean(All_performance[:,14]),
                'sample_hm': statistics.mean(All_performance[:,15]),
        
                "precision_O": statistics.mean(All_performance[:,16]),
                "recall_O": statistics.mean(All_performance[:,17]),
                "f1_O": statistics.mean(All_performance[:,18]),
                "mAP": statistics.mean(All_performance[:,19]),
                })
                
            try:
                wandb.log({
                    'precision_std':statistics.stdev(All_performance[:,0]),
                    'recall_std': statistics.stdev(All_performance[:,1]),
                    'accuracy_std': statistics.stdev(All_performance[:,2]),
                    'f1_std': statistics.stdev(All_performance[:,3]),
                    'earliness_std' : statistics.stdev(All_performance[:,4]),
                    'hm_std' : statistics.stdev(All_performance[:,5]),
            
                    'ranking_loss_std': statistics.stdev(All_performance[:,6]),
                    'one_error_std': statistics.stdev(All_performance[:,7]),
                    'coverage_std': statistics.stdev(All_performance[:,8]),
                    'subset_accuracy_std': statistics.stdev(All_performance[:,9]),
                    
                    'positive_accuracy_std': statistics.stdev(All_performance[:,11]),
                    
                    'sample_precision_std': statistics.stdev(All_performance[:,10]),
                    'sample_recall_std': statistics.stdev(All_performance[:,11]),
                    'sample_f1_std': statistics.stdev(All_performance[:,12]),
                    
                    'latest_earliness_std': statistics.stdev(All_performance[:,13]),
                    'hamming_loss_std': statistics.stdev(All_performance[:,14]),
                    'sample_hm_std': statistics.stdev(All_performance[:,15]),
            
                    "precision_O_std": statistics.stdev(All_performance[:,16]),
                    "recall_O_std": statistics.stdev(All_performance[:,17]),
                    "f1_O_std": statistics.stdev(All_performance[:,18]),
                    "mAP_std": statistics.stdev(All_performance[:,19]),
                    })
            except:
                pass
          
        try:
          
            print(
                "Precision: ",statistics.mean(All_performance[:,0]),'+-', statistics.stdev(All_performance[:,0]), '\n',
                "Recall: ",statistics.mean(All_performance[:,1]), '+-', statistics.stdev(All_performance[:,1]), '\n',
                "Accuracy: ",statistics.mean(All_performance[:,2]), '+-', statistics.stdev(All_performance[:,2]), '\n',
                "F1 : ",statistics.mean(All_performance[:,3]), '+-', statistics.stdev(All_performance[:,3]), '\n',
                "Label_Earliness: ", statistics.mean(All_performance[:,4]), '+-', statistics.stdev(All_performance[:,4]), '\n',
                "Harmonic_Mean: ", statistics.mean(All_performance[:,5]), '+-', statistics.stdev(All_performance[:,5]), '\n',
                
                "Ranking_Loss: ",statistics.mean(All_performance[:,6]),'+-', statistics.stdev(All_performance[:,6]), '\n',
                "One_Error: ",statistics.mean(All_performance[:,7]), '+-', statistics.stdev(All_performance[:,7]), '\n',
                "Coverage: ",statistics.mean(All_performance[:,8]), '+-', statistics.stdev(All_performance[:,8]), '\n',
                "Subset_Accuracy: ",statistics.mean(All_performance[:,9]), '+-', statistics.stdev(All_performance[:,9]), '\n',
                
                "Positive_Accuracy: ",statistics.mean(All_performance[:,11]), '+-', statistics.stdev(All_performance[:,11]), '\n',
                "Sample_Precision: ",statistics.mean(All_performance[:,10]), '+-', statistics.stdev(All_performance[:,10]), '\n',
                "Sample_Recall: ",statistics.mean(All_performance[:,11]), '+-', statistics.stdev(All_performance[:,11]), '\n',
                "Sample_F1: ",statistics.mean(All_performance[:,12]), '+-', statistics.stdev(All_performance[:,12]), '\n',
                "Latest_Earliness: ",statistics.mean(All_performance[:,13]), '+-', statistics.stdev(All_performance[:,13]), '\n',
                "Hamming_Loss: ",statistics.mean(All_performance[:,14]), '+-', statistics.stdev(All_performance[:,14]), '\n',
                "Sample_HM: ",statistics.mean(All_performance[:,15]), '+-', statistics.stdev(All_performance[:,15]), '\n',
        
                "Precision_O: ",statistics.mean(All_performance[:,16]), '+-', statistics.stdev(All_performance[:,16]), '\n',
                "Recall_O: ",statistics.mean(All_performance[:,17]), '+-', statistics.stdev(All_performance[:,17]), '\n',
                "F1_O: ",statistics.mean(All_performance[:,18]), '+-', statistics.stdev(All_performance[:,18]), '\n',
                "mAP: ",statistics.mean(All_performance[:,19]), '+-', statistics.stdev(All_performance[:,19]), '\n',
                    
                )
        except:
            print(
                "Precision: ",statistics.mean(All_performance[:,0]), '\n',
                "Recall: ",statistics.mean(All_performance[:,1]), '\n',
                "Accuracy: ",statistics.mean(All_performance[:,2]), '\n',
                "F1 : ",statistics.mean(All_performance[:,3]), '\n',
                "Label_Earliness: ", statistics.mean(All_performance[:,4]), '\n',
                "Harmonic_Mean: ", statistics.mean(All_performance[:,5]), '\n',
                
                "Ranking_Loss: ",statistics.mean(All_performance[:,6]), '\n',
                "One_Error: ",statistics.mean(All_performance[:,7]), '\n',
                "Coverage: ",statistics.mean(All_performance[:,8]), '\n',
                "Subset_Accuracy: ",statistics.mean(All_performance[:,9]), '\n',
                
                "Positive_Accuracy: ",statistics.mean(All_performance[:,11]), '\n',
                "Sample_Precision: ",statistics.mean(All_performance[:,10]), '\n',
                "Sample_Recall: ",statistics.mean(All_performance[:,11]), '\n',
                "Sample_F1: ",statistics.mean(All_performance[:,12]), '\n',
                "Latest_Earliness: ",statistics.mean(All_performance[:,13]), '\n',
                "Hamming_Loss: ",statistics.mean(All_performance[:,14]), '\n',
                "Sample_HM: ",statistics.mean(All_performance[:,15]), '\n',
        
                "Precision_O: ",statistics.mean(All_performance[:,16]), '\n',
                "Recall_O: ",statistics.mean(All_performance[:,17]), '\n',
                "F1_O: ",statistics.mean(All_performance[:,18]), '\n',
                "mAP: ",statistics.mean(All_performance[:,19]), '\n',
                
                )
                
        print('alpha:', alpha)
    
        if wandb is not None:
            wandb.log({
                'alpha': alpha,
                
                'precision_final':statistics.mean(All_performance[:,0]),
                'recall_final': statistics.mean(All_performance[:,1]),
                'accuracy_final': statistics.mean(All_performance[:,2]),
                'f1_final': statistics.mean(All_performance[:,3]),
                'earliness_final' : statistics.mean(All_performance[:,4]),
                'hm_final' : statistics.mean(All_performance[:,5]),
        
                'ranking_loss_final': statistics.mean(All_performance[:,6]),
                'one_error_final': statistics.mean(All_performance[:,7]),
                'coverage_final': statistics.mean(All_performance[:,8]),
                'subset_accuracy_final': statistics.mean(All_performance[:,9]),
                
                'positive_accuracy_final': statistics.mean(All_performance[:,11]),
                
                'sample_precision_final': statistics.mean(All_performance[:,10]),
                'sample_recall_final': statistics.mean(All_performance[:,11]),
                'sample_f1_final': statistics.mean(All_performance[:,12]),
                'latest_earliness_final': statistics.mean(All_performance[:,13]),
                'hamming_loss_final': statistics.mean(All_performance[:,14]),
                'sample_hm_final': statistics.mean(All_performance[:,15]),
        
                "precision_O_final": statistics.mean(All_performance[:,16]),
                "recall_O_final": statistics.mean(All_performance[:,17]),
                "f1_O_final": statistics.mean(All_performance[:,18]),
                "mAP_final": statistics.mean(All_performance[:,19]),
                })
            
    if wandb is not None:
        wandb.finish()
