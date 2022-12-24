from importlib import import_module


def controller_from_string(controller_str):
    return ControllerFactory(controller_str=controller_str)


class ControllerFactory:
    # noinspection SpellCheckingInspection
    valid_base_controllers = {
        "mpc-icem-torch": (".mpc_torch", "TorchMpcICem"),
        "mpc-curiosity-icem-torch": (".mpc_torch_curiosity", "TorchCuriosityMpcICem"),
        "mpc-rnd-icem-torch": (".mpc_torch_rnd", "TorchRNDMpcICem"),
    }

    controller = None

    def __new__(cls, *, controller_str):

        if controller_str in cls.valid_base_controllers:
            ctrl_package, ctrl_class = cls.valid_base_controllers[controller_str]
            module = import_module(ctrl_package, "mbrl.controllers")
            cls.controller = getattr(module, ctrl_class)
        else:
            raise ImportError(
                f"cannot find '{controller_str}' in known controller: " f"{cls.valid_base_controllers.keys()}"
            )

        return cls.controller
