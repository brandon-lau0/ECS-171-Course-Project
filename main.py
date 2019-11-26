import sys
import os
import dataset
from dataset import *
import datasetbuilder
import itertools
import json
from datasetbuilder import *
sys.path.insert(0, './data')
import sitedict
from sitedict import *


def main():

    f_ext = ""

    # Reminder to run code with command line arguments
    if len(sys.argv) < 3:
        print("Usage Ex: py main.py params.json ann_params.json")
        sys.exit()

    param_file = sys.argv[1]
    ann_param_file = sys.argv[2]

    # If an extra argument is supplied, this is the extension for the filename
    # This is used by gridsearch.ps1 to run multiple files in a row
    if len(sys.argv) > 3:
        f_ext = f"_{sys.argv[3]}"

    # dataset-level parameters
    l_params = []
    with open(param_file, 'r') as fin:
        l_params = json.load(fin)

    # grid search parameters
    l_ann_params = []
    with open(ann_param_file,'r') as fin:
        l_ann_params = json.load(fin)

    results = []

    # The list of headers we want to use to predict snow depth
    l_xcols = ["WTEQ.I-1 (in) ",
                "PREC.I-1 (in) ","TOBS.I-1 (degC) ","BATT.I-1 (volt) ",
                "TMAX.D-1 (degC) ","TMIN.D-1 (degC) ","TAVG.D-1 (degC) ",
                "SMS.I-1:-2 (pct)  (silt)",
                "SMS.I-1:-8 (pct)  (silt)","SMS.I-1:-20 (pct)  (silt)",
                "STO.I-1:-2 (degC) ","STO.I-1:-8 (degC) ","STO.I-1:-20 (degC) ",
                "SAL.I-1:-2 (gram) ","SAL.I-1:-8 (gram) ","SAL.I-1:-20 (gram) ",
                "RDC.I-1:-2 (unit) ","RDC.I-1:-8 (unit) ","RDC.I-1:-20 (unit) "]

    # The thing we want to predict (snow depth)
    l_ycols = ["SNWD.I-1 (in) "]

    # Path where results are saved
    res_path = os.path.join(os.getcwd(), "results", f"final_results_ANN{f_ext}.json")

    for params in l_params:

        # Make a copy of xcols because the list is a reference
        xcols = l_xcols.copy()

        # Make a dataset builder and use functions to modify the dataset
        databuilder = DataSet_Builder()
        databuilder.set_xcols(xcols)
        databuilder.set_ycols(l_ycols)
        databuilder.clean_df()
        databuilder.set_timestep(params["timestep"])
        databuilder.use_rect_radius(params["rectradius"])
        if params["remove_outliers"] == 1:
            databuilder.remove_outliers()
        if params["use_pca"] == 1:
            databuilder.use_pca()
        if params["scale_data"] == 1:
            databuilder.scale_data()

        # Create the DataSet object
        dataset = databuilder.build_dataset(l_xcols, params["scale_data"])

        # Makes a row the length of xcols based on the given date that can be
        # used as inputs for the OLS or FFNN
        dataset.impute_inputs("2019-12-01", params["timestep"])


        # Run TSNN and save results
        for ann_params in l_ann_params:
            row = {}
            moreinfo = f"{ann_params['optimizer']}-{ann_params['hiddenlayer']}-{ann_params['numneuron']}-{ann_params['loss']}-{ann_params['activation']}-"
            (mse, predlist) = dataset.run_TSNN("TSNN" + moreinfo + params["fileparam"], ann_params)
            row["method"] = "TSNN"
            row["mse"] = mse
            row["pred"] = predlist
            row["filename"] = "TSNN" + moreinfo + params["fileparam"]
            row["params"] = params
            row["ann_params"] = ann_params
            results.append(row.copy())


        # Run FFNN and save results
        for ann_params in l_ann_params:
            row = {}
            moreinfo = f"{ann_params['optimizer']}-{ann_params['hiddenlayer']}-{ann_params['numneuron']}-{ann_params['loss']}-{ann_params['activation']}-"
            (mse, pred) = dataset.run_ANN(ann_params, "ANN" + moreinfo + params["fileparam"])
            row["method"] = "ANN"
            row["mse"] = mse
            row["pred"] = pred
            row["filename"] = "ANN" + moreinfo + params["fileparam"]
            row["params"] = params
            row["ann_params"] = ann_params
            results.append(row.copy())


        # Run OLS and save results
        row = {}
        (mse, pred) = dataset.run_OLS("OLS" + params["fileparam"])
        row["method"] = "OLS"
        row["mse"] = mse
        row["pred"] = pred
        row["filename"] = "OLS" + params["fileparam"]
        row["params"] = params
        results.append(row.copy())


    # Save the results dictionary to json
    with open(res_path, 'w') as fout:
        json.dump(results, fout, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()
