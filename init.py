import pip

def install(package):
    pip.main(['install', package])


if __name__ == '__main__':
    pakage_list=[
        'scipy',
        'MiniSom',
        'numpy',
        'scikit-fuzzy',
        'scikit-image',
        'scikit-learn',
        'sklearn',
        'matplotlib'
    ]
    for p in pakage_list:
        install(p)
