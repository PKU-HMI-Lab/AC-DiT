from collections import defaultdict

import gymnasium

import numpy as np
import torch

from mani_skill.utils import common


def collect_episode_info(infos, result):
    if "final_info" in infos:  # infos is a dict

        indices = np.where(infos["_final_info"])[
            0
        ]  # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i]  # info is also a dict
            ep = info["episode"]
            result["return"].append(ep["r"][0])
            result["episode_len"].append(ep["l"][0])
            if "success" in info:
                result["success"].append(info["success"])
            if "fail" in info:
                result["fail"].append(info["fail"])
    return result
