
# stubs

class process_data(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def __call__(self, data):
        return data


def process_bw(**parameters):
    return process_data(parameters)


def process_sw(**parameters):
    return process_data(parameters)

