# # MIT License
# # Copyright (c) 2019 Sebastian Penhouet
# # GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# # ==============================================================================
# """Aggregates multiple tensorbaord runs"""

# import os
# import re
# import ast
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf

# from pathlib import Path
# from collections import defaultdict
# from tensorflow.core.util.event_pb2 import Event
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# __all__ = ["aggregate", "aggregate1"]


# def extract(path_to_folder):
#     scalar_accumulators = [EventAccumulator(path_to_folder).Reload().scalars]

#     # Filter non event files
#     scalar_accumulators = [
#         scalar_accumulator
#         for scalar_accumulator in scalar_accumulators
#         if scalar_accumulator.Keys()
#     ]

#     # Get and validate all scalar keys
#     all_keys = [
#         tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators
#     ]
#     assert (
#         len(set(all_keys)) == 1
#     ), "All runs need to have the same scalar keys. There are mismatches in {}".format(
#         all_keys
#     )
#     keys = all_keys[0]

#     all_scalar_events_per_key = [
#         [scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators]
#         for key in keys
#     ]

#     # Get and validate all steps per key
#     all_steps_per_key = [
#         [
#             tuple(scalar_event.step for scalar_event in scalar_events)
#             for scalar_events in all_scalar_events
#         ]
#         for all_scalar_events in all_scalar_events_per_key
#     ]

#     for i, all_steps in enumerate(all_steps_per_key):
#         assert (
#             len(set(all_steps)) == 1
#         ), "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
#             keys[i], [len(steps) for steps in all_steps]
#         )

#     steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

#     # Get and average wall times per step per key
#     wall_times_per_key = [
#         np.mean(
#             [
#                 tuple(scalar_event.wall_time for scalar_event in scalar_events)
#                 for scalar_events in all_scalar_events
#             ],
#             axis=0,
#         )
#         for all_scalar_events in all_scalar_events_per_key
#     ]

#     # Get values per step per key
#     values_per_key = [
#         [
#             [scalar_event.value for scalar_event in scalar_events]
#             for scalar_events in all_scalar_events
#         ]
#         for all_scalar_events in all_scalar_events_per_key
#     ]

#     all_per_key = dict(
#         zip(keys, zip(steps_per_key, wall_times_per_key, values_per_key))
#     )

#     return all_per_key


# def get_valid_filename(s):
#     s = str(s).strip().replace(" ", "_")
#     return re.sub(r"(?u)[^-\w.]", "", s)


# def aggregate(dpath, list_dname, output_path=None):
#     assert os.path.exists(dpath), "folder %s not exists" % dpath

#     for dname in list_dname:
#         assert os.path.exists(os.path.join(dpath, dname)), "folder %s not exists" % str(
#             os.path.join(dpath, dname)
#         )

#     extracts_per_subpath = dict()

#     for dname in list_dname:
#         extracts_per_subpath[dname] = dict()
#         list_part = [
#             f.name for f in os.scandir(os.path.join(dpath, dname)) if f.is_dir()
#         ]
#         list_part.sort()
#         extracts_per_subpath[dname]["list_part"] = list_part
#         set_part1 = set()
#         set_part2 = set()
#         for dpart in list_part:
#             # x, y =
#             splited = dpart.split("_")
#             x, y = "_".join(splited[:-1]), splited[-1]
#             set_part1.add(x)
#             set_part2.add(y)
#         for part1 in set_part1:
#             extracts_per_subpath[dname][part1] = dict()
#             for part2 in set_part2:
#                 extracts_per_subpath[dname][part1][part2] = extract(
#                     os.path.join(dpath, dname, part1 + "_" + part2)
#                 )

#     if len(list_dname) > 1:
#         for i in range(0, len(list_dname) - 1):
#             for j in range(i + 1, len(list_dname)):
#                 if (
#                     len(
#                         set(
#                             extracts_per_subpath[list_dname[i]]["list_part"]
#                         ).difference(
#                             set(extracts_per_subpath[list_dname[j]]["list_part"])
#                         )
#                     )
#                     != 0
#                 ):
#                     raise KeyError

#     list_part = [set(), set()]
#     list_data_frame = dict()
#     for key1, value1 in extracts_per_subpath.items():
#         list_data_frame[key1] = dict()
#         for key2, value2 in extracts_per_subpath[key1].items():
#             if key2 == "list_part":
#                 continue
#             list_data_frame[key1][key2] = dict()
#             for key3, value3 in extracts_per_subpath[key1][key2].items():
#                 for key, (steps, wall_times, values) in extracts_per_subpath[key1][
#                     key2
#                 ][key3].items():
#                     df = pd.DataFrame(
#                         list(zip(wall_times, steps, np.array(values).reshape(-1))),
#                         columns=["Wall time", "Step", "Value"],
#                     )
#                     list_data_frame[key1][key2][key3] = df
#                     list_part[0].add(key2)
#                     list_part[1].add(key3)
#     list_part[0] = list(list_part[0])
#     list_part[1] = list(list_part[1])

#     ret = dict()
#     for part1 in list_part[0]:
#         ret[part1] = dict()
#         for part2 in list_part[1]:
#             data_frame = pd.concat(
#                 [list_data_frame[x][part1][part2] for x in list_dname]
#             )
#             if output_path != None:
#                 file_name = os.path.join(
#                     output_path, get_valid_filename(part1 + "_" + part2) + ".csv"
#                 )
#                 data_frame.to_csv(file_name)
#             ret[part1][part2] = data_frame

#     return ret, list_part


# def aggregate1(dpath, list_dname, output_path=None):
#     assert os.path.exists(dpath), "folder %s not exists" % dpath
#     for dname in list_dname:
#         assert os.path.exists(os.path.join(dpath, dname)), "folder %s not exists" % str(
#             os.path.join(dpath, dname)
#         )

#     extracts_per_subpath = {
#         dname: extract(os.path.join(dpath, dname)) for dname in list_dname
#     }

#     list_data_frame = dict()
#     list_key = set()
#     for dname, value in extracts_per_subpath.items():
#         list_data_frame[dname] = dict()
#         for key, (steps, wall_times, values) in extracts_per_subpath[dname].items():
#             df = pd.DataFrame(
#                 list(zip(wall_times, steps, np.array(values).reshape(-1))),
#                 columns=["Wall time", "Step", "Value"],
#             )
#             list_data_frame[dname][key] = df
#             list_key.add(key)

#     ret = dict()
#     for key in iter(list_key):
#         data_frame = pd.concat([list_data_frame[dname][key] for i in list_dname])
#         if output_path != None:
#             file_name = os.path.join(output_path, get_valid_filename(key) + ".csv")
#             data_frame.to_csv(file_name)
#         ret[key] = data_frame
#     return ret


# def plot_loss_accuracy(dpath, list_dname, path_folder, title=None, com=0):
#     r"""Plot metrics from tensorboard log folder
#     Args:
#         dpath (str): path to folder contain (eg: saved/logs)
#         list_dname (list(str)): list of run_id to plot.
#         output_path (str): path to save csv file after concat logs from different run time
#         title (str): title for figure
#         com (float [0, 1]): ratio for smooth line
#     """
#     # check folder exists
#     assert os.path.exists(dpath), "folder %s not exists" % dpath
#     for dname in list_dname:
#         assert os.path.exists(os.path.join(dpath, dname)), "folder %s not exists" % str(
#             os.path.join(dpath, dname)
#         )

#     dict_data_frame, list_part = aggregate(dpath, list_dname)

#     fig, ax = plt.subplots(nrows=1, ncols=len(list_part[0]), figsize=(25, 10))

#     for i in range(len(list_part[0])):
#         colors = ["red", "green", "blue", "orange"]

#         # get outlier from phase train and valid
#         low, high = None, None
#         for j in range(len(list_part[1])):
#             df = dict_data_frame[list_part[0][i]][list_part[1][j]]
#             z_score = (df["Value"] - df["Value"].mean()) / (df["Value"].std(ddof=0))
#             df_min = df["Value"][np.abs(z_score) >= 1.5]

#             if low != None:
#                 low = min(df_min[df_min < df["Value"].mean()].min(), low)
#             else:
#                 low = df_min[df_min < df["Value"].mean()].min()
#             if np.isnan(low):
#                 low = None

#             if high != None:
#                 high = max(df_min[df_min > df["Value"].mean()].max(), high)
#             else:
#                 high = df_min[df_min > df["Value"].mean()].max()

#             if np.isnan(high):
#                 high = None

#         # plot
#         for j in range(len(list_part[1])):
#             df = dict_data_frame[list_part[0][i]][list_part[1][j]]
#             # smoothing
#             df["Value"] = df["Value"].ewm(com=com).mean()
#             # plot
#             df.plot.line(
#                 x="Step", y="Value", label=list_part[1][j], color=colors[j], ax=ax[i]
#             )
#         # set limit for y-axis
#         ax[i].set_ylim(low, high)
#         # set label
#         ax[i].set_title(list_part[0][i])
#         ax[i].set_xlabel("Epoch")
#         # Hide the right and top spines
#         ax[i].spines["right"].set_visible(False)
#         ax[i].spines["top"].set_visible(False)
#         # Only show ticks on the left and bottom spines
#         ax[i].yaxis.set_ticks_position("left")
#         ax[i].xaxis.set_ticks_position("bottom")
#         # show grid
#         ax[i].grid()

#     if title != None:
#         fig.suptitle(title)
#     plt.show()
#     if not os.path.exists(path_folder):
#         os.makedirs(path_folder)
#     path_figure = os.path.join(path_folder, "plot.png")
#     if os.path.exists(path_figure):
#         os.remove(path_figure)
#     fig.savefig(path_figure, dpi=300)

#     dict_data_frame = aggregate1(dpath, list_dname)
#     for key, value in dict_data_frame.items():
#         fig, ax = plt.subplots()
#         df = dict_data_frame[key]
#         # df['Value'] = np.log2(df['Value'])
#         df.plot.line(x="Step", y="Value", label=key, ax=ax)
#         # set label
#         ax.set_title(key)
#         ax.set_xlabel("Epoch")
#         # Hide the right and top spines
#         ax.spines["right"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         # Only show ticks on the left and bottom spines
#         ax.yaxis.set_ticks_position("left")
#         ax.xaxis.set_ticks_position("bottom")
#         # show grid
#         ax.grid()
#         plt.show()
#         path_figure = os.path.join(path_folder, "{}.png".format(key))
#         if os.path.exists(path_figure):
#             os.remove(path_figure)
#         fig.savefig(path_figure, dpi=300)


# if __name__ == "__main__":
#     from tqdm import tqdm

#     for run_id in tqdm(os.listdir("saved/logs")):
#         try:
#             plot_loss_accuracy(
#                 "saved/logs", [run_id], os.path.join("saved/logs", run_id)
#             )
#             pass
#         except Exception as e:
#             print(e)
