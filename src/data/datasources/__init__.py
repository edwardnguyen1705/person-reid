from .market1501 import Market1501

__datasets = {"market1501": Market1501}


def build_datasource(
    name: str,
    root: str,
):
    if name not in list(__datasets.keys()):
        raise KeyError
    return __datasets[name](root=root)
