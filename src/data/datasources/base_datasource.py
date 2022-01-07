import os

__all__ = ["BaseDatasource"]


class BaseDatasource(object):
    def __init__(self, root: str):
        self.root = root

    def get_data(self, phase: str):
        raise NotImplementedError

    def get_path(self, phase: str):
        for path, *_ in self.get_data(phase):
            yield path

    def get_classes(self):
        raise NotImplementedError

    def check_exists(self, phase: str):
        for path in self.get_path(phase):
            assert os.path.exists(path), f"Path: {path} not exists"
