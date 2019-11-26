$l_f_ext = "rmsprop_linear","rmsprop_relu","rmsprop_sigmoid","rmsprop_softmax","rmsprop_tanh","sgd_linear","sgd_relu","sgd_sigmoid","sgd_softmax","sgd_tanh"

Foreach($f_ext in $l_f_ext) {
    cmd.exe /c py main.py params.json ann_params_$f_ext.json $f_ext
}

