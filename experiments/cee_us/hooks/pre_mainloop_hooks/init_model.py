def init_model_hook(_locals, _globals, **kwargs):
    params = _locals["params"]
    forward_model = _locals["forward_model"]

    print("Initializing Model ...")
    # print(f"Setting model horizon to {params.horizon}")
    # forward_model.set_horizon(params.horizon)
