from local_constants import DATA_PATH
from preprocessing.preprocessing import write_tensors


def do_work():
    write_tensors(data_path=DATA_PATH)


if __name__ == "__main__":
    do_work()
