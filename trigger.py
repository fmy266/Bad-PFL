import torch


@torch.no_grad()
def grid_trigger_adder(data, label, target_label, poison_ratio, trigger_val=1., position="left_top", trigger_size=3,
                       strategy="paste", blend_ratio=0.5):
    batch_size, channel, height, width = data.size(0), data.size(1), data.size(2), data.size(3)
    trigger_size = [trigger_size for _ in range(2)] if isinstance(trigger_size, int) else trigger_size

    poison_mask = torch.rand(batch_size, device=label.device) <= poison_ratio

    if poison_mask.sum().item() == 0:
        return data, label
    else:
        poison_data, poison_label = data.clone(), torch.full([batch_size,], target_label, device=label.device)

        if position == "left_top":
            start_height, start_width = 0, 0
        elif position == "random":
            start_height, start_width = torch.randint(0, height - trigger_size[0] + 1, (1,)).item(), torch.randint(0, width - trigger_size[1] + 1, (1,)).item()
        elif position == "center":
            start_height, start_width = height // 2 - trigger_size[0] // 2,  width // 2 - trigger_size[1] // 2
            
        if isinstance(trigger_val, float):
            trigger = torch.full((batch_size, channel, *trigger_size), trigger_val, device=label.device)
        elif isinstance(trigger_val, list):
            trigger = torch.Tensor(trigger_val).view(1, channel, *trigger_size).to(label.device).repeat(batch_size, 1, 1, 1)
        elif isinstance(trigger_val, torch.Tensor):
            trigger = trigger_val.to(label.device)

        if strategy == "paste":
            poison_data[:,:,start_height:start_height+trigger_size[0],start_width:start_width+trigger_size[1]] = trigger
        elif strategy == "blend":
            poison_data = (1 - blend_ratio) * poison_data + blend_ratio * trigger
        else:
            raise ValueError("strategy must be either 'paste' or 'blend'")

        poison_data = poison_mask.view(-1, 1, 1, 1).float() * poison_data + (~poison_mask.view(-1, 1, 1, 1)).float() * data
        poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * label

        return poison_data, poison_label.to(torch.long)


