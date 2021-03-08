import time


def mpo_runner(writer,
               q_update,
               pi_update,
               sampler,
               test_agent,
               ac,
               ac_targ,
               buffer,
               total_steps=40000,
               min_steps_per_iteration=1000,
               test_after=4000,
               update_steps=1200,
               update_after=300,
               ):
    it = 0
    current_steps = 0
    start_time = time.time()
    total_updates = 0
    inner_it = 0
    while buffer.stored_interactions() < total_steps:
        # sample trajectories
        performed_steps = 0
        while performed_steps < min_steps_per_iteration:
            performed_steps += sampler()

        for r in range(update_steps):

            # update target networks
            if inner_it % update_after == 0:
                for target_param, param in zip(ac_targ.pi.parameters(), ac.pi.parameters()):
                    target_param.data.copy_(param.data)

            # update q values
            q_update()
            # update policy
            pi_update()
            total_updates += 1
            inner_it += 1

        if buffer.stored_interactions() - current_steps >= test_after:
            print("=" * 80)
            test_agent(it)
            writer.add_scalar(
                'performed_steps', buffer.stored_interactions(), it)

            it += 1
            current_steps = buffer.stored_interactions()
            print('time for update:', time.time() - start_time)
            print('total updates', total_updates)
            print('total steps', buffer.stored_interactions())
            start_time = time.time()
            total_updates = 0
        writer.flush()
