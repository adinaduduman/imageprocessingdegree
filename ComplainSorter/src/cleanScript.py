import os


def cleanDirectory(directoryPath):
    if os.path.exists(directoryPath):
        folders = os.listdir(directoryPath)
        for (i, f) in enumerate(folders):
            try:
                files = os.listdir(f'{directoryPath}/{f}')
                for (j, k) in enumerate(files):
                    os.remove(f'{directoryPath}/{f}/{k}')
                os.rmdir(f'{directoryPath}/{f}')
            except:
                os.remove(f'{directoryPath}/{f}')


def runScript():
    directory = '../data/'
    files = os.listdir(directory)
    for (i, f) in enumerate(files):
        if f != '1_raw_complains' and f != 'sorting_categories':
            cleanDirectory(f'{directory}{f}')


