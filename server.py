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
HEADER = 64
FORMAT = 'utf-8'

#Class for socket connection send and recv
class ServerQGI():
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((HOST, PORT))

    def waitConnection(self):
        print(f"Waiting connection on port {PORT} ...")
        self.socket.listen()
        self.conn, self.addr = self.socket.accept()
        print(f"Connected by {self.addr}")

    def send(self, data):
        size = len(data)
        send_size = str(size).encode(FORMAT) 
        send_size += b' ' * (HEADER - len(send_size))
        self.conn.send(send_size)
        self.conn.send(data)
        print(f"Data send to {self.addr}")

    def recv(self):
        recv_size = self.conn.recv(HEADER).decode(FORMAT)
        if recv_size:
            data = self.conn.recv(int(recv_size))
            print(f"Data receive by {self.addr}")
            return data
        return None
            
    def close(self):
        print('Server closing')
        self.socket.close()

if __name__ == '__main__':

    server = ServerQGI()

    try:
        while True:

            #Wait connection from QGIS plugin
            server.waitConnection()

            #recv data pickle
            data_pkl = server.recv()

            if data_pkl:
                #load data pickle and get dataset config and parameters   
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
                config['res_dir'] = '{}/Results/ActiveLearning/'.format(get_path()) + config['dataset'] + '/' + config['query']


                try:
                    os.makedirs(config['res_dir'], exist_ok=True)
                except OSError as exc:
                    if exc.errno != os.errno.EEXIST:
                        raise
                    pass

                #perform active learning step
                model, query, config = load_query(config, dataset)
                AL = ActiveLearningFramework(dataset, model, query, config)
                
                AL.step()
                path = AL.save() 

                #convert history path from active learning step to pickle
                path_pkl = pickle.dumps(path)

                #send history path pickle
                server.send(path_pkl) 

    except KeyboardInterrupt:
        server.close()  
