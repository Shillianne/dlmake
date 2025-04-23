import os
import json
import copy

combs = {
    "config1.json": [
        {
            "attention_method": "GroupedQueryAttention",
            "n_kv_groups": 3
        },
        {
            "attention_method": "MultiHeadLatentAttention",
            "q_head_dim": 64,
            "kv_lora_dim": 256,
            "v_head_dim": 64,
        },
    ],
    "general": [
        {"attention_method": "MultiHeadAttention"},
        {"attention_method": "MultiQueryAttention"},
        {"max_epochs": [12000, 18000, 24000]},
    ]
}

def rec(ls: list[list[int | float | tuple]]):
    s = ls[0]
    if len(ls) == 1:
        for v in s:
            yield [v]
    else:
        for v in s:
            for prev in rec(ls[1:]):
                yield [v] + prev

class HyperParameterSearch:
    """
    Class for performing a hyper parameter search for a series of base models with a given
    series of parameter combination.

    Parameters
    ----------
    configs : list[str]
        A list of paths to the base config files to be used for the search
    combinations: dict[str, list[dict[str, int | float | str | bool]]]
        A dictionary of values for the combinations to be used for each config,
        also accepts a "general" key for combinations that apply to all configs.
    path: str
        The path where the search is going to be performed.
    """
    def __init__(self,
                 configs: list[str],
                 combinations: dict[str, list[dict[str, int | float | str | bool]]],
                 path: str):
        os.mkdir(os.path.join(path, "search"))
        self.path = os.path.join(path, "search")
        os.mkdir(os.path.join(self.path, "configs"))
        count = 0
        for filepath in configs:
            name = os.path.basename(filepath)
            if name not in combinations.keys() and "general" not in combinations.keys():
                raise KeyError(f"File {name} does not have associated combinations nor are there general combinations for this config.")
            with open(filepath, "r") as f:
                cfg = json.load(f)
            init_config = copy.copy(cfg)
            configured_params = []
            if name in combinations.keys():
                for config_setting in combinations[name]:
                    for k, v in config_setting.items():
                        configured_params.append(k)
                        init_config[k] = v
                    if "general" in combinations.keys():
                        for config in self.__give_general_combinations(combinations, configured_params, init_config):
                            with open(os.path.join(self.path, "configs", f"config_{count:04d}.json"), "w") as f:
                                json.dump(config, f)
                            count += 1
                    else:
                        with open(os.path.join(self.path, "configs", f"config_{count:04d}.json"), "w") as f:
                            json.dump(init_config, f)
                        count += 1
            elif "general" in combinations.keys():
                for config in self.__give_general_combinations(combinations, configured_params, init_config):
                    with open(os.path.join(self.path, "configs", f"config_{count:04d}.json"), "w") as f:
                        json.dump(config, f)
                    count += 1
        if count == 0:
            raise KeyError("No base configuration files have been provided, cannot set up a hyper parameter search")
    
    @staticmethod
    def __give_general_combinations(combs: dict, configured: list, init_config: dict):
        keys = []
        values = []
        lr = {}
        for general_settings in combs["general"]:
            for k, v in general_settings.items():
                if k in configured:
                    continue
                if k == "max_lr" or k == "min_lr":
                    lr[k] = v
                    continue
                keys.append(k)
                values.append(v)
        if "max_lr" in lr.keys() and "min_lr" in lr.keys():
            keys.append("lr")
            values.append(list(zip(lr["max_lr"], lr["min_lr"])))
        else:
            for k, v in lr.items():
                keys.append(k)
                values.append(v)
        for comb in rec(values):
            config = copy.copy(init_config)
            for key, value in zip(keys, comb):
                if key == "lr":
                    assert isinstance(value, tuple)
                    config["max_lr"] = value[0]
                    config["min_lr"] = value[1]
                else:
                    config[key] = value
            yield config
