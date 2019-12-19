import json
import re
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace
from functools import singledispatch
import pandas as pd


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class StringConverter:

    @staticmethod
    def underscore_to_camelcase(underscore: str):
        class_name_handler = list(underscore)
        for gp in re.finditer(r'(^[a-z])|(?:_([a-z]))', underscore):
            class_name_handler[gp.end() - 1] = class_name_handler[gp.end() - 1].upper()

        class_name_handler = "".join(class_name_handler)
        return class_name_handler.replace("_", "")

    @staticmethod
    def camelcase_to_underscore(camelcase: str):
        raise NotImplementedError()


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i + 1)
            plt.imshow(imgs[i])

    plt.show()

@singledispatch
def dict2obj(o):
    return o

@dict2obj.register(dict)
def handle_obj(obj):
    return SimpleNamespace(**{ k:dict2obj(v) for k,v in obj.items() })

@dict2obj.register(list)
def handle_list(lst):
    return [dict2obj(i) for i in lst]