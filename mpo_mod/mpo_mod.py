from tqdm import tqdm


def mpo_runner(writer,
               q_update,
               pi_update,
               sampler,
               test_agent,
               ac,
               ac_targ,
               buffer,
               total_steps=40000,
               min_steps_per_epoch=200,
               test_after=4000,
               update_steps=1200,
               update_after=300,
               ):
    iteration = 0
    current_steps = 0
    while buffer.stored_interactions() < total_steps:
        # sample trajectories
        performed_steps = 0
        while performed_steps < min_steps_per_epoch:
            performed_steps += sampler()

        for r in tqdm(range(update_steps), desc='updating nets'):

            # update target networks
            if r % update_after == 0:
                for target_param, param in zip(ac_targ.parameters(), ac.parameters()):
                    target_param.data.copy_(param.data)

            # update q values
            q_update()
            # update policy
            pi_update()

        if buffer.stored_interactions() - current_steps >= test_after:
            test_agent(iteration)
            writer.add_scalar(
                'performed_steps', buffer.stored_interactions(), iteration)

            iteration += 1
            current_steps = buffer.stored_interactions()
        writer.flush()
