import pickle
from pprint import pprint

with open('data.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)

pprint(data[0])