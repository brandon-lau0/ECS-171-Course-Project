# ECS-171-Course-Project

## Introduction

The code is entered through main.py. When run, it will create a DataSet_Builder
object which essentially does the preprocessing on the data and can build a
DataSet object which runs the machine learning methods. Results are saved to a
json file for later viewing.

## Runing the Code

Run the code using

`python3 main.py <dataset-level_params>.json <grid_search_params>.json`

* For example, one could run:

    `python3 main.py singleparam.json singleannparam.json`

* To get the dataset-level param results, we used:

    `python3 main.py params1.json singleannparam.json`

* To get the grid search results, use

    `python3 main.py params.json ann_params.json`

    Note that in order to break up the results, we actually ran several
    different files called `ann_params_*.json`


These json files are constructed using `parambuild.py` (more notes below) and
contain a list of dictionaries. The dictionaries contain the parameter values
used to construct the DataSet and run the models.


## Relevant Files

Here's a list of current files/folders that are relevant.
```
    .
    +-- dataset.py
    +-- datasetbuilder.py
    +-- main.py
    +-- *.json
    +-- gridsearch.ps1
    +-- scrape.py
    +-- parambuild.py
    +-- OLS.py
    +-- ANN.py
    +-- TSNN.py
    +-- _data
    |   +-- site*.pkl
    |   +-- (data.pkl)
    |   +-- info.txt
    |   +-- merged.csv
    |   +-- sitedict.py
    +-- _results
    |   +-- _dataset-level_params
    |   +-- _grid_searches

```

We will not cover a full list of attributes here, but more information about
each of the code files can be found in the files.

* `dataset.py` contains the DataSet class with these and other attributes:
    * `impute_inputs()`: takes in a future date and makes an estimation of
    the "X" input matrix values for that date by averaging values from that
    day and surrounding days from previous years
    * `run_OLS()`: runs the OLS functions and stores results
    * `run_ANN()`: runs the ANN functions and stores results
    * `run_TSNN()`: runs the TSNN functions and stores results
* `datasetbuilder.py`: contains the DataSet_Builder class with these and other
attributes:
    * `clean_df()`: drops rows with NaN or -99.9 values
    * `format_date()`: converts to cylindrical representation of dates
    * `use_rect_radius()`: reduces number of sites by rectangular radius
    * `use_pca()`: uses PCA to reduce number of features
    * `remove_outliers()`: uses IsolationForest to remove outliers
    * `scale_data()`: min-max scales data from 0 to 1
    * `build_dataset()`: builds a DataSet object
* `main.py` is the main function to run, uses a DataSet_Builder object to build
a DataSet object, runs ML methods, and save results
* `*.json`: several files used as inputs to `main.py`
* `gridsearch.ps1`: the final code was run on Windows, so this is a Windows
PowerShell script that basically just runs `main.py` with different command
line arguments
* `scrape.py` is a standalone script that scrapes the website for data
    * gets each site from `SITEDICT`
    * each year from 1980 to 2019
* `parambuild.py` is a standalone script that uses lists and for loops to build
a list of dictionaries of parameter combinations, both for dataset-level
parameters and for neural net parameters used by the grid search. The json files
it saves can be used as command line arguments for `main.py`.
* `OLS.py` contains functions that compute the OLS and make a prediction
* `ANN.py` contains functions that create a FFNN model, train it, and make a
prediction
* `TSNN.py` contains functions that train a time series recurrent neural net
and make predictions for the next four weeks.
* `data` folder
    * `site*.pkl`: these are individual pickles for each site (all years)
    * `data.pkl`: overall data pickle
        * only appears after constructing `DataSet` object
        * is over 100 MB, so it's in the gitignore
    * `info.txt`: site info, including, number, name, latitude, longitude
    * `merged.csv`: csv file of all data
        * currently ordered by date then site
        * note that the data is rather sparse
        * if you make changes, delete `data.pkl` (which will regenerate)
    * `sitedict.py`: uses `info.txt` to build a dictionary `SITEDICT` of sites
* `results` folder
    * `dataset-level_params` folder: contains graphs and a json file of results
    for various combinations of dataset-level parameters, used to determine the
    optimal dataset-level parameters. The folder also contains an INFO.txt file
    with more information
    * `grid_search_params` folder: contains multiple subfolders with graphs and
    result json files that make up the grid search. The folder contains an
    INFO.txt file that has more information.
