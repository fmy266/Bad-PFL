from event_emitter import fl_event_emitter
from tqdm import tqdm

def basic_fl_process(server, clients, local_steps, training_rounds, select_rule):

    fl_event_emitter.emit(event_name="on_fl_begin", server=server, clients=clients, local_steps=local_steps,
                          training_rounds=training_rounds, select_rule=select_rule)

    for cur_round in tqdm(range(1, training_rounds+1)):

        client_indices = select_rule(server, clients)
        server.training_info["cur_client_indices"] = client_indices

        fl_event_emitter.emit(event_name="on_round_begin", cur_round=cur_round, server=server, clients=clients,
                              local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                              selected_client_indices=client_indices)

        global_model_at_cur_round = server.distribute_model()

        for indice in client_indices:

            fl_event_emitter.emit(event_name="on_client_begin", cur_round=cur_round, server=server, clients=clients,
                                  local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                                  client_indice=indice)

            clients[indice].init_round()
            clients[indice].receive_model(global_model_at_cur_round)

            for local_step in range(local_steps):
                clients[indice].local_update()

            fl_event_emitter.emit(event_name="on_client_end", cur_round=cur_round, server=server, clients=clients,
                                  local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule,
                                  client_indice=indice)

        server.agg_and_update([clients[indice].upload_model() for indice in client_indices])

        fl_event_emitter.emit(event_name="on_round_end", cur_round=cur_round, server=server, clients=clients,
                              local_steps=local_steps, training_rounds=training_rounds, select_rule=select_rule)

    fl_event_emitter.emit(event_name="on_fl_end", server=server, clients=clients, local_steps=local_steps,
                          training_rounds=training_rounds, select_rule=select_rule)



