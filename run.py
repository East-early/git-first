from LSTMModel import LSTMModel

//

from data_processing import DataLoader
import json
import math
import numpy as np
import pandas as pd
import datetime as dt

configs = json.load(open('configs.json', 'r'))

def run_lstm(data, test):
    model = LSTMModel()
    model.build_model(configs)
    steps_per_epoch = math.ceil(
        (data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
    )
    predictions = model.predict_point_by_point(test)
    print(predictions.shape)
    return predictions.tolist()

def add_time(get_data):
    a = get_data[:, -1]
    b = np.array([[0 if dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').isoweekday() < 6 else 1 for x in a]]).T
    a = get_data[:, 1:-1]
    b = np.hstack([a, b])
    return b

if __name__  ==  "__main__":
    totalPrediction = []
    start_time = dt.datetime.now()

    get_res = np.array(pd.read_csv('data/toPredict_noLabel.csv'))
    cur_road_index = 0
    cur_road = get_res[cur_road_index, 1]
    row_test_data = np.array(pd.read_csv('data/toPredict_train_TTI.csv'))
    row_train_data = np.array(pd.read_csv('data/train_TTI.csv'))

    for i in range(12):
        print("Predicting {0}th road, Num: {1} ".format(i+1, cur_road))
        test = row_test_data[row_test_data[:, 0] == cur_road]
        test = add_time(test)

        train = row_train_data[row_train_data[:, 0] == cur_road]
        train = add_time(train)

        data = DataLoader(train, test)
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        x_test = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        cur_road_index += 21
        cur_road = get_res[cur_road_index, 1]
        totalPrediction.append(run_lstm(data, x_test))
    totalPrediction = np.array(totalPrediction)
    print(totalPrediction.shape)
    l, r = np.hsplit(totalPrediction, [21])
    res = l.reshape(1, -1).squeeze().tolist()
    while r.shape[1] >= 21:
        l, r = np.hsplit(r, [21])
        res += l.reshape(1, -1).squeeze().tolist()
    header = ["TTI"]
    result = pd.DataFrame(columns=header, data=res)
    result.index.name = "id_sample"
    result.to_csv('submit_v7.csv')
    end_time = dt.datetime.now()
    print("Model total time consume: ", end_time - start_time)
