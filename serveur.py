import socket
import pickle
import os
import numpy as np
from learning.query import load_query
from path import get_path
from data.data import Dataset
from learning.session import ActiveLearningFramework

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        data = conn.recv(10)
        data_pkl = data
        while data:
            data = conn.recv(10)
            data_pkl += data
        if data_pkl:
            print(f"Paramaters receive by {addr}")
            param = pickle.loads(data_pkl)
            config = param['config']
            dataset_param = param['dataset_param']

            dataset = Dataset(config, **dataset_param)
            config['n_classes'] = dataset.n_classes
            config['proportions'] = dataset.proportions
            config['classes'] = np.unique(dataset.train_gt())[1:]
            config['n_bands']   = dataset.n_bands
            config['ignored_labels'] = dataset.ignored_labels
            config['img_shape'] = dataset.img_shape
            config['res_dir'] = '{}/Results/ActiveLearning/'.format(get_path()) + config['dataset'] + '/' + 'gt1' + '/' + config['query']

            config['superpixels'] = None

            try:
                os.makedirs(config['res_dir'], exist_ok=True)
            except OSError as exc:
                if exc.errno != os.errno.EEXIST:
                    raise
                pass

            model, query, config = load_query(config, dataset)
            AL = ActiveLearningFramework(dataset, model, query, config)
                
            AL.step()
            AL.save()
