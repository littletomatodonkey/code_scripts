import os
import numpy as np
import paddle

model_fmt = "iter_epoch_{}.pdparams"
ema_model_fmt = "iter_epoch_{}_ema.pdparams"

# models in the dir
model_index = range(3, 300, 3)

# ema decay
ema_decay = 0.998

ema_model_params = dict()

for idx in model_index:
    mode_name = model_fmt.format(idx)
    ema_mode_name = ema_model_fmt.format(idx)

    models_params = paddle.load(mode_name)

    # init
    if len(ema_model_params) == 0:
        for key in models_params:
            ema_model_params[key] = models_params[key].copy()
    else:
        for key in models_params:
            ema_model_params[key] = models_params[key] * (
                1 - ema_decay) + ema_model_params[key] * ema_decay

    paddle.save(ema_model_params, ema_mode_name)
