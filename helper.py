from configuration import *


def split_email(email):
    tokens = word_tokenize(email)

    return tokens


def get_files_path(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if os.path.isfile(path):
            files.append(path)
    return files