import time


def runner(writer,
           q_update,
           pi_update,
           one_step_update,
           target_update,
           sampler,
           test_agent,
           buffer,
           total_steps=12000,
           min_steps_per_iteration=50,
           test_after=4000,
           update_steps=50,
           update_after=50,
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
                target_update()

            # update q values
            q_update()
            # update policy
            pi_update()
            one_step_update()
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
