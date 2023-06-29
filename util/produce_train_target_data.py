import numpy as np

def process_target_data(data, config,step=1, base_length=0,fore_length=0):

    data_list = []
    end_token = base_length + fore_length
    start_token = 0
    while end_token <= len(data):
        data_list.append(data[start_token:end_token])
        start_token += step
        end_token += step
    data_list = np.array(data_list, dtype=np.float32)


    return data_list

def process_train_data(data,config,step=1,base_length=0,fore_length=0):
    data_list = []
    end_token = base_length
    start_token = 0
    while end_token <= (len(data)-fore_length):
        data_list.append(data[start_token:end_token])
        start_token += step
        end_token += step
    data_list = np.array(data_list, dtype=np.float32)
    return data_list
