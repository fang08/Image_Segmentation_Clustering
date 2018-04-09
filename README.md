# Image_Segmentation_Clustering

## Initialization:

  Run _python init.py_ first

## How to run clustering algorithms:

    python MyClust4.py Algorithm algoname ImType typename
  
    python MyClust4.py algoname typename
  
    algoname and typename are as follow:
        algonames = ['Kmeans', 'SOM', 'FCM', 'Spectral', 'GMM']
        typenames = ['RGB', 'Hyper']
    
## How to run evaluation algorithms:    
      python MyClustEvalRGB4.py filename1 filename2
      python MyClustEvalHyper4.py
