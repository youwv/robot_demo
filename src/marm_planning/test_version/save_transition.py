import cv2
import pandas as pd
import numpy as np
import datadeal
import os


class Transition_Save:
    def __init__(self, buffer_size):
        self.state_left_path = "/home/you/robotiq_data/state/left/"
        self.state_right_path = "/home/you/robotiq_data/state/right/"
        self.next_state_left_path = "/home/you/robotiq_data/next_state/left/"
        self.next_state_right_path = "/home/you/robotiq_data/next_state/right/"
        self.rest_path = "/home/you/robotiq_data/rest/rest.csv"
        self.name = ["a_temp", "r", "done"]
        self.buffer_size = buffer_size
        self.sample_number = sum(1 for _ in open(self.rest_path))-1

    def create_csv(self):
        datadeal.create_csv(self.rest_path, self.name)

    def save(self, transition):
        cv2.imwrite(self.state_left_path + str(self.sample_number+1)+'.PNG', transition[0][0])
        cv2.imwrite(self.state_right_path + str(self.sample_number+1)+'.PNG', transition[0][1])
        cv2.imwrite(self.next_state_left_path + str(self.sample_number+1)+'.PNG', transition[3][0])
        cv2.imwrite(self.next_state_right_path + str(self.sample_number+1)+'.PNG', transition[3][1])

        datadeal.append_csv(self.rest_path, [[transition[1], transition[2], transition[-1]]])

        assert (len(os.listdir(self.state_left_path)) == len(os.listdir(self.state_left_path)))
        assert (len(os.listdir(self.state_left_path)) == len(os.listdir(self.next_state_right_path)))
        assert (len(os.listdir(self.next_state_left_path)) == len(os.listdir(self.next_state_right_path)))
        assert (len(os.listdir(self.state_left_path)) == sum(1 for _ in open(self.rest_path))-1)

        self.sample_number += 1

    def dump(self, batch_size):
        assert (self.sample_number > self.buffer_size)
        indice = []
        for i in range(self.sample_number):
            indice.append(i)
        indices = np.random.choice(indice[-10000:], size=batch_size)
        s = []
        s_ = []
        a_temp = []
        r = []
        done = []
        for index in indices:
            s.append([cv2.imread(self.state_left_path + str(index) + '.PNG', -1),
                      cv2.imread(self.state_right_path + str(index) + '.PNG', -1)])
            s_.append([cv2.imread(self.next_state_left_path + str(index) + '.PNG', -1),
                       cv2.imread(self.next_state_right_path + str(index) + '.PNG', -1)])

            _Get_csv = pd.read_csv(self.rest_path)
            temp = _Get_csv.loc[index]
            a_temp.append(temp[0])
            r.append(temp[1])
            done.append(temp[2])

        return s, a_temp, r, s_, done
