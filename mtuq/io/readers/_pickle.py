
import pickle


def read(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    return dataset

