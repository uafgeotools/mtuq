
import csv

def weight_parser(filename):
    weights = {}
    with open('weight.dat') as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            fields = row[0].split('.')
            station_id = '.'.join(fields[1:3])
            weights[station_id] = row

