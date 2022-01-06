__all__ = ["BaseDatasource"]


class BaseDatasource(object):
    def __init__(self, root: str):
        self.root = root

    def get_data(self):
        raise NotImplementedError

    def get_classes(self):
        raise NotImplementedError
