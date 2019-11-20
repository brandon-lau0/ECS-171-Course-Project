import sys
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

    l_xcols = [["Latitude","Longitude"],["Latitude"]]
    # currently assuming same len as xcols
    # if always just snwd, don't need
    l_ycols = [["SNWD.I-1 (in) "], ["SNWD.I-1 (in) "]]

    # ANN only
    l_optimizer = []
    l_hiddenlayer = []
    l_numneuron = []
    l_loss = []
    l_activation = []

    # All
    l_timestep = []
    for timestep in Timestep:
        l_timestep.append(timestep.name)
    l_rectradius = [0.25, 0.5, 0.75, 1]


    for xcols, ycols in zip(l_xcols, l_ycols):
        for timestep in l_timestep:
            for rectradius in l_rectradius:
                # combinations of optional dataset parameters
                for (i, j, k) in list(itertools.product([0,1], repeat=3)):
                    params = {}
                    params["xcols"] = xcols
                    params["ycols"] = ycols
                    params["timestep"] = timestep
                    params["rectradius"] = rectradius
                    params["i"] = i
                    params["j"] = j
                    params["k"] = k

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

    for params in l_params:
        databuilder = DataSet_Builder()
        databuilder.set_xcols(params["xcols"])
        databuilder.set_ycols(params["ycols"])
        databuilder.clean_df()
        
        databuilder.set_timestep(params["timestep"])
        databuilder.use_rect_radius(params["rectradius"])

        if params["i"] == 1:
            databuilder.remove_outliers()
        if params["j"] == 1:
            databuilder.scale_data()
        if params["k"] == 1:
            databuilder.use_pca()

        dataset = databuilder.build_dataset()

        for ann_params in l_ann_params:
            dataset.run_ANN(ann_params)

        dataset.run_OLS()
        dataset.run_Time()





if __name__ == "__main__":
    main()
