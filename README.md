# Mockup of U-net as segmentation model 

NB!  Not finished or tested. 

## Things to consider 
* Image resolution 
* Number of segmentation classes

## Set-up

Create folders for training and testing images

```bash
mkdir data 
cd data & mkdir training trainig # Place traning and testing images here
```

Changes these values in `train.py` for output image resolution, and number of segmentation classes.

```python 
HEIGHT = 256
WIDTH = 256

SEGMENTATION_CLASSES = 4
```

## Usage 
Add folder for model data when running trainig script.
```
python3 train.py train_dir
```
