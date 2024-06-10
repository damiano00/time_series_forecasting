PROCESSED_DATA_PATH = './data/processed'
RAW_DATA_PATH = './data/raw'
DOCS = './docs'
LOGS = './logs'
MODELS_PATH = './models'
NOTEBOOKS = './notebooks'
TESTS = './tests'


def __init__():
    init_dirs()


def init_dirs():
    import os
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
    if not os.path.exists(DOCS):
        os.makedirs(DOCS)
    if not os.path.exists(LOGS):
        os.makedirs(LOGS)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if not os.path.exists(NOTEBOOKS):
        os.makedirs(NOTEBOOKS)
    if not os.path.exists(TESTS):
        os.makedirs(TESTS)
