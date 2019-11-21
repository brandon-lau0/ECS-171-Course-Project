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


def param_builder():
    l_params = []



    # ANN only
    l_optimizer = ["sgd","rmsprop"]
    l_activation = ["softmax","relu","sigmoid","tanh","linear"]
    l_loss = ["mean_squared_error"]
    l_hiddenlayer = [1, 2, 3]
    l_numneuron = [3, 6, 9, 12]

    # All
    l_timestep = []
    for timestep in Timestep:
        l_timestep.append(timestep.name)
    l_rectradius = [0.25, 0.5, 0.75, 1]


    for timestep in l_timestep:
        for rectradius in l_rectradius:
            # combinations of optional dataset parameters
            for (i, j, k) in list(itertools.product([0,1], repeat=3)):
                params = {}
                params["timestep"] = timestep
                params["rectradius"] = rectradius
                params["remove_outliers"] = i
                params["scale_data"] = j
                params["use_pca"] = k
                params["fileparam"] = f"{timestep}-{rectradius}-{i}-{j}-{k}.png"

                l_params.append(params)


    l_ann_params = []
    for optimizer in l_optimizer:
        for hiddenlayer in l_hiddenlayer:
            for numneuron in l_numneuron:
                for loss in l_loss:
                    for activation in l_activation:
                        ann_params = {}
                        ann_params["optimizer"] = optimizer
                        ann_params["hiddenlayer"] = hiddenlayer
                        ann_params["numneuron"] = numneuron
                        ann_params["loss"] = loss
                        ann_params["activation"] = activation
                        l_ann_params.append(ann_params)





    with open('params.json', 'w') as fout:
        json.dump(l_params, fout)

    with open('ann_params.json','w') as fout:
        json.dump(l_ann_params, fout)




def main():
    # dataset = DataSet()
    # # print(dataset.get_colnames())
    # # print (SITEDICT)
    # before = len(dataset.df.index)
    # new_df = dataset.get_df_of_radius(1)
    # after = len(new_df.index)
    # print(f"Proportion remaining: {after/before}")

    if len(sys.argv) != 3:
        print("Usage Ex: py main.py params.json ann_params.json")
        sys.exit()

    param_file = sys.argv[1]
    ann_param_file = sys.argv[2]

    param_builder()

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
                "SNWD.I-1 (in) ","SMS.I-1:-2 (pct)  (silt)",
                "SMS.I-1:-8 (pct)  (silt)","SMS.I-1:-20 (pct)  (silt)",
                "STO.I-1:-2 (degC) ","STO.I-1:-8 (degC) ","STO.I-1:-20 (degC) ",
                "SAL.I-1:-2 (gram) ","SAL.I-1:-8 (gram) ","SAL.I-1:-20 (gram) ",
                "RDC.I-1:-2 (unit) ","RDC.I-1:-8 (unit) ","RDC.I-1:-20 (unit) "]
    # l_xcols = [["Latitude","Longitude"],["Latitude"]]
    # currently assuming same len as xcols
    # if always just snwd, don't need
    l_ycols = ["SNWD.I-1 (in) "]

    res_path = os.path.join(os.getcwd(), "results", "results.json")

    part_res_path = os.path.join(os.getcwd(), "results", "part-results.json")
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

        for ann_params in l_ann_params:
            row = {}
            moreinfo = f"{ann_params['optimizer']}-{ann_params['hiddenlayer']}-{ann_params['numneuron']}-{ann_params['loss']}-{ann_params['activation']}-"
            (mse, pred) = dataset.run_ANN(ann_params, "ANN" + moreinfo + params["fileparam"])
            row["method"] = "ANN"
            row["mse"] = mse
            row["pred"] = pred
            row["filename"] = "ANN" + params["fileparam"]
            row["params"] = params
            results.append(row.copy())

        # row = {}
        # (mse, pred) = dataset.run_OLS("OLS" + params["fileparam"])
        # row["method"] = "OLS"
        # row["mse"] = mse
        # row["pred"] = pred
        # row["filename"] = "OLS" + params["fileparam"]
        # row["params"] = params
        # results.append(row.copy())
        #
        # print(results)
        # if params["timestep"] == "weekly":
            # row = {}
            # (mse, predlist) = dataset.run_TSNN("TSNN" + params["fileparam"])
            # row["method"] = "TSNN"
            # row["mse"] = mse
            # row["pred"] = predlist
            # row["filename"] = "TSNN" + params["fileparam"]
            # row["params"] = params
            # results.append(row.copy())
            # print(results)

        with open(part_res_path, 'w') as f:
            json.dump(results, f)


    with open(res_path, 'w') as fout:
        json.dump(results, fout)


if __name__ == "__main__":
    main()
