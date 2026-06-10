import os
import glob
import torch
import numpy as np
from termcolor import colored
import dataclasses

def detach(tensor):
    return tensor.detach().cpu()

def attach(tensor, device=torch.device("cpu")):
    return tensor.to(device)

def is_tensor(var):
    return isinstance(var, torch.Tensor)

def is_numpy(var):
    return isinstance(var, np.ndarray)

def is_list(var):
    return isinstance(var, list)

def is_dict(var):
    return isinstance(var, dict)

def is_dataclass(var):
    return dataclasses.is_dataclass(var)

def is_string(var):
    return isinstance(var, str)

def is_dict(var):
    return isinstance(var, dict)

def dict_to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    for key, value in var.items():
        if value is not None:
            var[key] = to_tensor(value, dtype, device)
    return var

def ld_to_dt(var, dtype=torch.float32, device=torch.device("cuda")):
    ret = {}
    keys = var[0].keys()
    for k in keys:
        new_value = []
        for d in var:
            assert d[k] is not None
            new_value.append(to_tensor(d[k], dtype, device))
        ret[k] = to_tensor(new_value, dtype, device)
    return ret

def ld2dl(LD):
    assert isinstance(LD, list)
    assert isinstance(LD[0], dict)
    """
    A list of dict (same keys) to a dict of lists
    """
    dict_list = {k: [dic[k] for dic in LD] for k in LD[0]}
    return dict_list

def dl2ld(DL: dict):
    """
    A dict lf list to a list of dicts (each value of dict must have same length)
    """
    list_dicts = []
    list_len = 0
    for v in DL.values():
        list_len = len(v)
        break
    
    for i in range(list_len):
        td = {}
        for k, v in DL.items():
            td[k] = v[i]
        list_dicts.append(td)
    return list_dicts

def dict_pickout_batch(d, ids):
    ret = {}
    for k, v in d.items():
        if is_tensor(v):
            ret[k] = v[ids]
        elif is_numpy(v):
            ret[k] = v[ids]
        elif isinstance(v, dict):
            ret[k] = dict_pickout_batch(v, ids)
    return ret

def homo(points):
    return torch.nn.functional.pad(points, (0, 1), value=1)

def to_tensor(var, dtype=torch.float32, device=torch.device("cpu")):
    if is_dataclass(var):
        for key, value in var.__dict__.items():
            if value is not None:
                setattr(var, key, to_tensor(value, dtype, device))
    elif is_numpy(var):
        try:
            if len(var.shape) > 0:
                var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_list(var):
        try:
            var = attach(torch.tensor(var, dtype=dtype), device)
        except:
            pass
    elif is_tensor(var):
        try:
            var = attach(var, device)
        except:
            pass
    elif is_dict(var):
        try:
            var = dict_to_tensor(var, dtype, device)
        except:
            pass
    return var

def to_numpy(var) -> np.ndarray:
    if is_dataclass(var):
        for key, value in var.__dict__.items():
            if value is not None:
                setattr(var, key, to_numpy(value))
    elif is_tensor(var):
        var = detach(var).cpu().numpy()
    elif is_list(var):
        if is_tensor(var[0]):
            var = [detach(t).cpu().numpy() for t in var]
    elif is_dict(var):
        for key, value in var.items():
            if value is not None:
                var[key] = to_numpy(value)
    return var

def cprint(text, color="green"):
    print(colored(text, color))

def find_best_checkpoint(check_dir, sort_by="epoch"):  # loss, epoch
    all_checkpoints = glob.glob(os.path.join(check_dir, "*.ckpt"))

    if len(all_checkpoints) == 0:
        errno = f"no checkpoint found at {check_dir}"
        raise FileNotFoundError(errno)
    else:
        epochs = []
        converged_losses = []
        converged_losses_list = []
        steps = []
        for chk_path in all_checkpoints:
            epoch = chk_path.split("/")[-1].split("epoch=")[-1].split("-")[0]
            epochs.append(epoch)
            step = chk_path.split("/")[-1].split("step=")[-1].split("-")[0]
            steps.append(step)
            converged_loss = chk_path.split("/")[-1].split(".ckpt")[0].split("=")[-1]
            converged_losses_list.append(converged_loss)
            converged_losses.append(np.array(converged_loss).astype(np.float64))

        if sort_by == "loss":
            min_loss_idx = np.argmin(converged_losses)
            min_loss = min(converged_losses)
            converged_losses = np.array(converged_losses, dtype=np.float64)
            mask = np.array(converged_losses) == min_loss
            steps_array = np.array(steps)
            max_step = max(steps_array[mask])
            idx = steps_array.tolist().index(max_step)
            best_epoch = epochs[idx]
            best_step = steps[idx]
            min_loss_str = converged_losses_list[min_loss_idx]
            checkpoint_name = (
                f"epoch={best_epoch}-step={best_step}-loss={min_loss_str}.ckpt"
            )
            return os.path.join(check_dir, checkpoint_name)
        elif sort_by == "epoch":
            max_epoch = max(epochs)
            for ckpt_path in all_checkpoints:
                if f"epoch={max_epoch}" in ckpt_path:
                    return ckpt_path


def thing2list(thing):
    if isinstance(thing, torch.Tensor):
        return thing.tolist()
    if isinstance(thing, np.ndarray):
        return thing.tolist()
    if isinstance(thing, dict):
        return {k: thing2list(v) for k, v in thing.items()}
    if isinstance(thing, list):
        return [thing2list(ten) for ten in thing]
    return thing


def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing


def thing2np(thing):
    if isinstance(thing, list):
        return np.array(thing)
    if isinstance(thing, torch.Tensor):
        return thing.cpu().detach().numpy()
    if isinstance(thing, dict):
        return {k: thing2np(v) for k, v in thing.items()}
    return thing


def thing2torch(thing):
    if isinstance(thing, list):
        if len(thing) > 0 and isinstance(thing[0], str):
            return thing
        return torch.tensor(np.array(thing))
    
    if isinstance(thing, np.ndarray):
        if len(thing) > 0 and isinstance(thing[0], str):
            return thing
        if thing.dtype == np.uint32:
            thing = thing.astype(np.int64)
        return torch.from_numpy(thing)

    if isinstance(thing, dict):
        return {k: thing2torch(v) for k, v in thing.items()}

    return thing


def detach_thing(thing):
    if isinstance(thing, torch.Tensor):
        return thing.cpu().detach()
    if isinstance(thing, list):
        return [detach_thing(ten) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(detach_thing(list(thing)))
    if isinstance(thing, dict):
        return {k: detach_thing(v) for k, v in thing.items()}
    return thing

def clone_and_detach(thing):
    if isinstance(thing, torch.Tensor):
        return thing.clone().detach()
    if isinstance(thing, list):
        return [clone_and_detach(ten) for ten in thing]
    if isinstance(thing, dict):
        return {k: clone_and_detach(v) for k, v in thing.items()}
    return thing