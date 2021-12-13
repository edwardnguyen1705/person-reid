import re
import os
import sys
from tqdm import tqdm


class Market1501(object):
    url = {
        "Market-1501-v15.09.15.zip": "1xi80U2hEut1bLOFQdBxmUQlgqPu2b07a",
    }

    def __init__(
        self,
        root,
        **kwargs,
    ):
        train_dir = os.path.join(root, "bounding_box_train")
        query_dir = os.path.join(root, "query")
        gallery_dir = os.path.join(root, "bounding_box_test")

        self.pid_container = {}
        self.camid_containter = {}
        self.frames_container = {}

        (
            self.train,
            self.pid_container["train"],
            self.camid_containter["train"],
            self.frames_container["train"],
        ) = self.process_dir(train_dir, relabel=True, ignore_junk=True)

        (
            self.query,
            self.pid_container["query"],
            self.camid_containter["query"],
            self.frames_container["query"],
        ) = self.process_dir(query_dir, relabel=False, ignore_junk=True)

        (
            self.gallery,
            self.pid_container["gallery"],
            self.camid_containter["gallery"],
            self.frames_container["gallery"],
        ) = self.process_dir(gallery_dir, relabel=False, ignore_junk=True)

    def get_data(self, mode="train"):
        if mode == "train":
            return self.train
        elif mode == "query":
            return self.query
        elif mode == "gallery":
            return self.gallery
        else:
            raise ValueError("mode error")

    def _check_file_exits(self):
        r"""check all image in datasource exists"""
        for phase in ["train", "query", "gallery"]:
            for data in self.get_data(phase):
                if not os.path.exists(data[0]):
                    raise FileExistsError

    def process_dir(self, path, relabel=False, ignore_junk=False):
        data = []
        pattern = re.compile(r"([-\d]+)_c(\d)s(\d)_([-\d]+)")

        pid_container = set()
        camid_containter = set()
        frames_container = set()

        for img in os.listdir(path):
            name, ext = os.path.splitext(img)
            if ext == ".jpg":
                img_path = os.path.join(path, img)
                person_id, camera_id, seq, frame = map(
                    int, pattern.search(name).groups()
                )
                if person_id == -1:
                    continue
                pid_container.add(person_id)
                camid_containter.add(camera_id)
                frames_container.add(self._re_frame(camera_id, seq, frame))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img in os.listdir(path):
            name, ext = os.path.splitext(img)
            if ext == ".jpg":
                img_path = os.path.join(path, img)
                person_id, camera_id, seq, frame = map(
                    int, pattern.search(name).groups()
                )

                if ignore_junk and person_id == -1:
                    continue

                if relabel:
                    person_id = pid2label[person_id]
                data.append((img_path, person_id, camera_id))

        return data, pid_container, camid_containter, frames_container

    def _re_frame(self, cam, seq, frame):
        """Re frames on market1501.
        more info here: https://github.com/Wanggcong/Spatial-Temporal-Re-identification/issues/10
        """
        if seq == 1:
            return frame
        dict_cam_seq_max = {
            11: 72681,
            12: 74546,
            13: 74881,
            14: 74661,
            15: 74891,
            16: 54346,
            17: 0,
            18: 0,
            21: 163691,
            22: 164677,
            23: 98102,
            24: 0,
            25: 0,
            26: 0,
            27: 0,
            28: 0,
            31: 161708,
            32: 161769,
            33: 104469,
            34: 0,
            35: 0,
            36: 0,
            37: 0,
            38: 0,
            41: 72107,
            42: 72373,
            43: 74810,
            44: 74541,
            45: 74910,
            46: 50616,
            47: 0,
            48: 0,
            51: 161095,
            52: 161724,
            53: 103487,
            54: 0,
            55: 0,
            56: 0,
            57: 0,
            58: 0,
            61: 87551,
            62: 131268,
            63: 95817,
            64: 30952,
            65: 0,
            66: 0,
            67: 0,
            68: 0,
        }

        re_frame = sum(dict_cam_seq_max[int(str(cam) + str(i))] for i in range(1, seq))
        return re_frame + frame

    def get_classes(self):
        return self.pid_container["train"]
