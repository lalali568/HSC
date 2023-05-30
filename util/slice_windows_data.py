import numpy as np
def process_data( data, config,step,penism_train_flag=False):
    if not penism_train_flag:
        data_list = []
        end_token = config['window_size']
        start_token = 0
        while end_token <= len(data):
            data_list.append(data[start_token:end_token])
            start_token += step
            end_token += step
        data_list = np.array(data_list, dtype=np.float32)
    else:
        data_list = []
        end_token = config['window_size']
        start_token = 0
        while end_token <= len(data) :
            data_list.append(data[start_token:end_token])
            start_token += step
            end_token += step
            if end_token%400==0:
                start_token = end_token
                end_token = start_token+config['window_size']

    return data_list