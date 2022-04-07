import os
import errno


def makedirs(dir_path: str) -> None:
    '''
    Creates a directory and ignores the error in which a folder already exists

    :param dir_path:  Name of the folder to be created
    :return:
    '''
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(dir_path)
            makedirs(dir_path)
