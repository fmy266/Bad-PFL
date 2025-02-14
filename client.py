import torch, copy
from utils import enable_mix_precision


class BasicClient:
    def __init__(self, local_model, train_dataloader, test_dataloader, loss_func, optimizer):
        self.cid = None
        self.local_model = local_model
        self.local_model.device = torch.device("cpu")
        self.local_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fetch = iter(self.local_dataloader)
        self.optimizer = optimizer(self.local_model.parameters())
        self.loss_func = loss_func
        self.scaler = torch.cuda.amp.GradScaler()
        self.client_info = {}
        self.training_info = {}
        self.registered_funcs = {}

    def init_round(self):
        self.training_info.clear()

    def receive_model(self, global_state_dict):
        self.local_model.load_state_dict(global_state_dict, strict=False)
        self.call_registered_func("before_local_training")
        
    def upload_model(self):
        self.upload_state_dict = self.local_model.state_dict()
        self.call_registered_func("before_upload_model")
        return self.upload_state_dict

    def local_update(self):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.local_model.train()
        data = self.fetch_data()
        pred = self.forward(data)
        loss = self.loss_computation(pred, data)
        self.backward_and_update(loss, self.optimizer)
        self.call_registered_func("after_local_update")
        torch.cuda.empty_cache()

    def local_fine_tuning(self, iter_nums):
        self.init_round()
        for _ in range(iter_nums):
            self.local_update()

    def fetch_data(self):
        try:
            data = next(self.fetch)
        except StopIteration:
            self.fetch = iter(self.local_dataloader)
            data = next(self.fetch)
        if isinstance(data, list):
            return [d.to(self.local_model.device) for d in data]
        else:
            return data.to(self.local_model.device)

    @enable_mix_precision
    def forward(self, data):
        pred = self.local_model(data[0])
        return pred

    @enable_mix_precision
    def loss_computation(self, pred, data):
        loss = self.loss_func(pred, data[1])
        return loss

    def backward_and_update(self, loss, optimizer):
        self.scaler.scale(loss).backward()
        self.call_registered_func("before_update")
        self.scaler.step(optimizer)
        self.scaler.update()

    def register_func(self, func, stage):
        if stage not in self.registered_funcs:
            self.registered_funcs[stage] = []
        self.registered_funcs[stage].append(func)

    def call_registered_func(self, stage):
        if stage in self.registered_funcs:
            for func in self.registered_funcs[stage]:
                func(self)


class PMClient(BasicClient):
    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)

        self.personalized_model = copy.deepcopy(self.local_model) if personalized_model is None else personalized_model
        self.personalized_optimizer = optimizer(self.personalized_model.parameters()) if personalized_optimizer is None else personalized_optimizer(self.personalized_model.parameters())
        self.personalized_loss_func = loss_func if personalized_loss_func is None else personalized_loss_func
        self.update_order = "global_personalized" # or personalized_global

    def local_update(self):
        if self.update_order == "global_personalized":
            super().local_update()
  
        torch.cuda.empty_cache()
        self.personalized_optimizer.zero_grad()
        self.personalized_model.train()
        data = self.fetch_data()
        pred = self.personalized_forward(data)
        loss = self.loss_computation(pred, data)
        self.backward_and_update(loss, self.personalized_optimizer)
        torch.cuda.empty_cache()

        if self.update_order == "personalized_global":
            super().local_update()

    @enable_mix_precision
    def personalized_forward(self, data):
        pred = self.personalized_model(data[0])
        return pred


class PoisonClient(BasicClient):

    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer, poison_func):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())


class PMPoisonClient(PMClient):

    def __init__(self, local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                 personalized_model=None, personalized_optimizer=None, personalized_loss_func=None, poison_func=None):
        super().__init__(local_model, local_dataloader, test_dataloader, loss_func, optimizer,
                         personalized_model, personalized_optimizer, personalized_loss_func)
        self.poison_func = poison_func
    
    def fetch_data(self):
        return self.poison_func(*super().fetch_data())



