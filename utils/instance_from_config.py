import importlib

def instantiate_from_config(config):
    """
    Instantiate an instance from a config file

    :param config: For example
        target: ldm.models.autoencoder_decoupling.AutoencoderKL
          params:
            ckpt_path: xxx
            load_form_sd: False
            embed_dim: 4
            ......
    :return: An instantiated class
    """
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
