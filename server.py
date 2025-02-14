import torch


def agg_avg(state_dicts):
    average_dict = state_dicts[0]
    for key in average_dict.keys():
        for idx in range(1, len(state_dicts)):
            average_dict[key] = average_dict[key] + state_dicts[idx][key]
        average_dict[key] = average_dict[key] / len(state_dicts)
    return average_dict


class BasicServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.agg_rule = "avg"
        self.server_info = {}
        self.training_info = {}
        self.registered_funcs = {}
        self.update = None
        self.distribute_dict = None

    @torch.no_grad()
    def distribute_model(self):
        self.distribute_dict = self.global_model.state_dict()
        self.call_registered_func("before_distribute_global")
        return self.distribute_dict

    @torch.no_grad()
    def agg_and_update(self, state_dicts):
        self.update = state_dicts
        self.update = agg_avg(state_dicts)
        self.call_registered_func("before_update_global")
        self.global_model.load_state_dict(self.update, strict=False)

    def register_func(self, func, stage):
        if stage not in self.registered_funcs:
            self.registered_funcs[stage] = []
        self.registered_funcs[stage].append(func)

    def call_registered_func(self, stage):
        if stage in self.registered_funcs:
            for func in self.registered_funcs[stage]:
                func(self)

