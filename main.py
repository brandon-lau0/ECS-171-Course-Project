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
    # dataset = DataSet()
    # # print(dataset.get_colnames())
    # # print (SITEDICT)
    # before = len(dataset.df.index)
    # new_df = dataset.get_df_of_radius(1)
    # after = len(new_df.index)
    # print(f"Proportion remaining: {after/before}")

    if len(sys.argv) < 3:
        print("Usage Ex: py main.py params.json ann_params.json")
        sys.exit()

    param_file = sys.argv[1]
    ann_param_file = sys.argv[2]

    if len(sys.argv) > 3:
        f_ext = sys.argv[3]

    # param_builder()

    l_params = []
    with open(param_file, 'r') as fin:
        l_params = json.load(fin)

    l_ann_params = []
    with open(ann_param_file,'r') as fin:
        l_ann_params = json.load(fin)

    results = []

    l_xcols = ["WTEQ.I-1 (in) ",
                "PREC.I-1 (in) ","TOBS.I-1 (degC) ","BATT.I-1 (volt) ",
                "TMAX.D-1 (degC) ","TMIN.D-1 (degC) ","TAVG.D-1 (degC) ",
                "SMS.I-1:-2 (pct)  (silt)",
                "SMS.I-1:-8 (pct)  (silt)","SMS.I-1:-20 (pct)  (silt)",
                "STO.I-1:-2 (degC) ","STO.I-1:-8 (degC) ","STO.I-1:-20 (degC) ",
                "SAL.I-1:-2 (gram) ","SAL.I-1:-8 (gram) ","SAL.I-1:-20 (gram) ",
                "RDC.I-1:-2 (unit) ","RDC.I-1:-8 (unit) ","RDC.I-1:-20 (unit) "]
    # l_xcols = [["Latitude","Longitude"],["Latitude"]]
    # currently assuming same len as xcols
    # if always just snwd, don't need
    l_ycols = ["SNWD.I-1 (in) "]

    res_path = os.path.join(os.getcwd(), "results", f"final_results_ANN_{f_ext}.json")

    part_res_path = os.path.join(os.getcwd(), "results", "part_results.json")
    for params in l_params:
        # uniq = f"{params["timestep"]}-{params["rectradius"]}-{params["remove_outliers"]}-{params["scale_data"]}-{params["use_pca"]}-"
        xcols = l_xcols.copy()

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

        dataset = databuilder.build_dataset(l_xcols, params["scale_data"])


        print(dataset.impute_inputs("2019-12-01", params["timestep"]))

        # for ann_params in l_ann_params:
        #     row = {}
        #     moreinfo = f"{ann_params['optimizer']}-{ann_params['hiddenlayer']}-{ann_params['numneuron']}-{ann_params['loss']}-{ann_params['activation']}-"
        #     (mse, predlist) = dataset.run_TSNN("TSNN" + moreinfo + params["fileparam"], ann_params)
        #     row["method"] = "TSNN"
        #     row["mse"] = mse
        #     row["pred"] = predlist
        #     row["filename"] = "TSNN" + moreinfo + params["fileparam"]
        #     row["params"] = params
        #     row["ann_params"] = ann_params
        #     results.append(row.copy())

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

        # row = {}
        # (mse, pred) = dataset.run_OLS("OLS" + params["fileparam"])
        # row["method"] = "OLS"
        # row["mse"] = mse
        # row["pred"] = pred
        # row["filename"] = "OLS" + params["fileparam"]
        # row["params"] = params
        # results.append(row.copy())



        # with open(part_res_path, 'w') as f:
        #     json.dump(results, f, indent=4, separators=(',', ': '))


    with open(res_path, 'w') as fout:
        json.dump(results, fout, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()
