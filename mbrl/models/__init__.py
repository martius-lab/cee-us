from importlib import import_module

models_dict = {
    "MLPForwardModel": (".mlp_model", "MLPForwardModel"),
    "MLP": (".torch_models", "MLP"),
    "GaussianNN": (".torch_models", "GaussianNN"),
    "GroundTruthModel": (".gt_model", "GroundTruthModel"),
    "GroundTruthEnsembleModel": (".gt_model", "GroundTruthEnsembleModel"),
    "ParallelGroundTruthModel": (".gt_par_model", "ParallelGroundTruthModel"),
    "GNNForwardModel": (".gnn_model", "GNNForwardModel"),
    "ParallelNNDeterministicEnsemble": (
        ".mlp_ensemble",
        "ParallelNNDeterministicEnsemble",
    ),
    "ParallelGNNDeterministicEnsemble": (
        ".gnn_ensemble",
        "GNNForwardEnsembleModel",
    ),
}


# Return a class not an instance
def forward_model_from_string(mod_str: str) -> type:
    """
    Returns a model class equivalent to the supplied string.
    :param mod_str: Name of model class
    :return: The model class
    """
    if mod_str == "none":
        return None
    elif mod_str in models_dict:
        if isinstance(models_dict[mod_str], tuple):
            mod_package, mod_class = models_dict[mod_str]
            module = import_module(mod_package, "mbrl.models")
            cls = getattr(module, mod_class)
            return cls
        else:
            return models_dict[mod_str]
    else:
        raise NotImplementedError("Implement model class {} and add it to dictionary".format(mod_str))
