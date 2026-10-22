"""Give each AIDE repetition its own runtime and model sampling seed."""

import copy
import json


def seeded_agent(agent, seed: int):
    if agent.name != "aide":
        return agent
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("AIDE seed must be an integer in [0, 2**32)")
    seeded = copy.deepcopy(agent)
    env = seeded.env_vars
    env["AIDE_RUN_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    sampling = json.loads(env.get("AIDE_SAMPLING", "{}"))
    if not isinstance(sampling, dict):
        raise ValueError("AIDE_SAMPLING must be a JSON object")
    sampling["seed"] = seed
    env["AIDE_SAMPLING"] = json.dumps(sampling, separators=(",", ":"))
    models = json.loads(env.get("AIDE_MODEL_SAMPLING", "{}"))
    if not isinstance(models, dict) or any(not isinstance(v, dict) for v in models.values()):
        raise ValueError("AIDE_MODEL_SAMPLING must map models to JSON objects")
    for settings in models.values():
        settings["seed"] = seed
    env["AIDE_MODEL_SAMPLING"] = json.dumps(models, separators=(",", ":"))
    # WM sampling remains frozen independently in AIDE_SIMULATOR_SAMPLING.
    return seeded
