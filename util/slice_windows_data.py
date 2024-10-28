def process_data( data, config,step):
    data_list = []
    end_token = config['window_size']
    start_token = 0
    while end_token <= len(data) :
        data_list.append(data[start_token:end_token])
        start_token += step
        end_token += step

    return data_list