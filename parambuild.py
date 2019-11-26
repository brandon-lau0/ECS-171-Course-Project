import sys
import os
import dataset
from dataset import *
import datasetbuilder
from datasetbuilder import *
import json

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

# Use various for loops to build l_param list of dictionaries

for timestep in l_timestep:
    params = {}
    params["timestep"] = timestep
    params["rectradius"] = 0.25
    params["remove_outliers"] = 1
    params["scale_data"] = 1
    params["use_pca"] = 1
    params["fileparam"] =  f"{timestep}-0.25-1-1-1.png"
    l_params.append(params)

for rectradius in [0.5, 0.75, 1]:
    params = {}
    params["timestep"] = "weekly"
    params["rectradius"] = rectradius
    params["remove_outliers"] = 1
    params["scale_data"] = 1
    params["use_pca"] = 1
    params["fileparam"] =  f"weekly-{rectradius}-1-1-1.png"
    l_params.append(params)

for i in range(0,1):
    params = {}
    params["timestep"] = "weekly"
    params["rectradius"] = 0.25
    params["remove_outliers"] = i
    params["scale_data"] = 1
    params["use_pca"] = 1
    params["fileparam"] =  f"weekly-0.25-{i}-1-1.png"
    l_params.append(params)

for i in range(0,1):
    params = {}
    params["timestep"] = "weekly"
    params["rectradius"] = 0.25
    params["remove_outliers"] = 1
    params["scale_data"] = i
    params["use_pca"] = 1
    params["fileparam"] =  f"weekly-0.25-1-{i}-1.png"
    l_params.append(params)

for i in range(0,1):
    params = {}
    params["timestep"] = "weekly"
    params["rectradius"] = 0.25
    params["remove_outliers"] = 1
    params["scale_data"] = 1
    params["use_pca"] = i
    params["fileparam"] =  f"weekly-0.25-1-1-{i}.png"
    l_params.append(params)

# params = {}
# params["timestep"] = "weekly"
# params["rectradius"] = 0.25
# params["remove_outliers"] = 1
# params["scale_data"] = 1
# params["use_pca"] = 1
# params["fileparam"] =  f"weekly-0.25-1-1-1.png"
# l_params.append(params)



l_ann_params = []
for optimizer in l_optimizer:
    for activation in l_activation:
        for loss in l_loss:
            for hiddenlayer in l_hiddenlayer:
                for numneuron in l_numneuron:
                    ann_params = {}
                    ann_params["optimizer"] = optimizer
                    ann_params["hiddenlayer"] = hiddenlayer
                    ann_params["numneuron"] = numneuron
                    ann_params["loss"] = loss
                    ann_params["activation"] = activation
                    l_ann_params.append(ann_params)


# Save params and ann_params to json files for use with main.py
with open('params1.json', 'w') as fout:
    json.dump(l_params, fout, indent=4, separators=(',', ': '))

with open('ann_params.json','w') as fout:
    json.dump(l_ann_params, fout, indent=4, separators=(',', ': '))
