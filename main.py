import sys
import dataset
from dataset import *
import datasetbuilder
import itertools
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
    l_rectradius = [0.1, 0.25, 0.5, 0.75, 0.9, 1]


    # This is so horrible
    # I want to change this to read the combinations from a file and log results
    #   to a file so when my laptop crashes, we can pick back up where it died
    # We're talking like thousands of combinations here
    for xcols, ycols in zip(l_xcols, l_ycols):
        for timestep in l_timestep:
            for rectradius in l_rectradius:
                databuilder = DataSet_Builder()
                databuilder.set_xcols(xcols)
                databuilder.set_ycols(ycols)
                databuilder.set_timestep(timestep)
                databuilder.use_rect_radius(rectradius)

                # combinations of optional dataset parameters
                for (i, j, k) in list(itertools.product([0,1], repeat=3)):
                    if i == 1:
                         databuilder.remove_outliers()
                    if j == 1:
                        databuilder.scale_data()
                    if k == 1:
                        databuilder.use_pca()


                    dataset = databuilder.build_dataset()

                    # TODO: do something with this stuff
                    for optimizer in l_optimizer:
                        for hiddenlayer in l_hiddenlayer:
                            for numneuron in l_numneuron:
                                for loss in l_loss:
                                    for activation in l_activation:
                                        params = {}
                                        params["optimizer"] = optimizer
                                        params["hiddenlayer"] = hiddenlayer
                                        params["numneuron"] = numneuron
                                        params["loss"] = loss
                                        params["activation"] = activation
                                        dataset.run_ANN(params)
                    dataset.run_OLS()
                    dataset.run_Time()




if __name__ == "__main__":
    main()
