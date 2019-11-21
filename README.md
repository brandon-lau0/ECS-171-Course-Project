# ECS-171-Course-Project

## Relevant Files

Here's a list of current files/folders that are relevant.
```
    .
    +-- dataset.py
    +-- main.py
    +-- scrape.py
    +-- _data
    |   +-- site*.pkl
    |   +-- (data.pkl)
    |   +-- info.txt
    |   +-- merged.csv
    |   +-- sitedict.py

```
* `dataset.py` contains the DataSet class with the following attributes:
    * Member variables:
        * `df`: dataframe representing data
        * `train`: training set, available after `split_data()`
        * `test`: testing set, available after `split_data()`
        * `df_X`: "X" matrix, available after `split_X_Y()`
        * `df_Y`: "Y" vector, available after `split_X_Y()`
        * `cleaned_df`: df - (feature list), available after `remove_features`
    * Methods:
        * `remove_features()`: removes the specified features -> `cleaned_df`
        * `remove_outliers()`: TODO (or delete)
        * `split_data()`: splits data into training and testing
        * `split_X_Y()`: splits data into X matrix and Y vector
        * `get_colnames()`: returns `df.columns`
        * `get_df()`: returns `df`
        * `get_cleaned_df()`: returns `cleaned_df`
* `main.py` is the main function to run, currently does nothing significant
* `scrape.py` is a standalone script that scrapes the website for data
    * gets each site from `SITEDICT`
    * each year from 1980 to 2019
* `data` folder
    * `site*.pkl`: these are individual pickles for each site (all years)
    * `data.pkl`: overall data pickle
        * only appears after constructing `DataSet` object
        * is over 100 MB, so I'd recommend deleting before pushing
    * `info.txt`: site info, including, number, name, latitude, longitude
    * `merged.csv`: csv file of all data
        * currently ordered by date then site
        * note that the data is rather sparse
        * if you make changes, delete `data.pkl` (which will regenerate)
    * `sitedict.py`: uses `info.txt` to build a dictionary `SITEDICT` of sites
