import os
import pickle


def make_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        print("Directory %s exists already" % dir_name)


def save_pkl_file(file_name, data):
    if file_name[-4:] != ".pkl":
        file_name = file_name + ".pkl"
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()


def load_pkl_file(file_path):
    if file_path[-4:] != ".pkl":
        file_path = file_path + ".pkl"
    a_file = open(file_path, "rb")
    obj = pickle.load(a_file)
    return obj
