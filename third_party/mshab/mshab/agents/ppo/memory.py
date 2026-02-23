from gymnasium import spaces

import torch


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        self.device = device
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=self.device)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(self.device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict, device=self.device)
