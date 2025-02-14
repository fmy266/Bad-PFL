import torch

def use_fedbn(server):

    def fedbn_update(server):
        delete_keys = []
        for key in server.update.keys():
            if "bn" in key or "shortcut.1" in key:
                delete_keys.append(key)
        
        for key in delete_keys:
            server.update.pop(key)

    def fedbn_distribute(server):
        delete_keys = []
        for key in server.distribute_dict.keys():
            if "bn" in key or "shortcut.1" in key:
                delete_keys.append(key)
        
        for key in delete_keys:
            server.distribute_dict.pop(key)

    server.register_func(fedbn_update, "before_update_global")
    server.register_func(fedbn_distribute, "before_distribute_global")

