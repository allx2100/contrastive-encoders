import dill


def unpickle(file):
    with open(file, "rb") as fo:
        dict = dill.load(fo, encoding="bytes")
    return dict


def save_data(data, file):
    with open(file, "wb") as f:
        dill.dump(data, f)