import torch
from generator import *
from functools import partial


def pgd_attack(model, images, labels, epsilon=4./255., alpha=4./255., num_iter=1):

    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)
    
    for _ in range(num_iter):
        torch.cuda.empty_cache()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * torch.sign(adv_images.grad)
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, min=0, max=1)
        adv_images = adv_images.detach().clone()
        torch.cuda.empty_cache()
    return adv_images.detach().clone()


def use_our_attack(clients, server, target_label=0, poison_ratio=0.2):

    trigger_gen = Autoencoder().to(server.global_model.device)
    gen_optimizer = torch.optim.Adam(trigger_gen.parameters(), lr=1e-2)
    loss_func = torch.nn.CrossEntropyLoss()

    def trigger_gen_trainer(client):
        client.local_model.eval()
        for _ in range(30):
            torch.cuda.empty_cache()
            gen_optimizer.zero_grad()
            clean_data, clean_label = client.fetch_data()
            adv_imgs = pgd_attack(client.local_model, clean_data, clean_label)
            gen_trigger = trigger_gen(clean_data) / 255. * 4.
            pred = client.local_model(adv_imgs + gen_trigger)
            loss = loss_func(pred, torch.full([clean_label.size(0),], target_label, device=clean_label.device).to(torch.long))
            loss.backward()
            gen_optimizer.step()
            torch.cuda.empty_cache()

    
    def our_poison_func(data, label, target_label=target_label, poison_ratio=poison_ratio, client=None):

        poison_mask = torch.rand(label.size(0), device=label.device) <= poison_ratio
        if poison_mask.sum().item() == 0:
            return data, label
        else:
            poison_data, poison_label = data.clone(), torch.full([label.size(0),], target_label, device=label.device)
        poison_data = pgd_attack(client.local_model, poison_data, label).detach().clone()
        gen_trigger = trigger_gen(data) / 255. * 4.
        poison_data = poison_mask.view(-1, 1, 1, 1).float() * (poison_data + gen_trigger) + (~poison_mask.view(-1, 1, 1, 1)).float() * data
        poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * label

        return poison_data, poison_label.to(torch.long)

    for client in clients:
        if "Poison" in type(client).__name__:
            client.register_func(trigger_gen_trainer, "before_local_training")
            client.poison_func = partial(our_poison_func, target_label=target_label, poison_ratio=poison_ratio, client=client)
            eval_func = partial(our_poison_func, target_label=target_label, poison_ratio=1., client=client)
    
    return eval_func

