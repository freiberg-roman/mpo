from tqdm import tqdm
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
    iteration = 0
    current_steps = 0
    start_time = time.time()
    totat_updates = 0
    while buffer.stored_interactions() < total_steps:
        # sample trajectories
        performed_steps = 0
        while performed_steps < min_steps_per_iteration:
            performed_steps += sampler()
            print(performed_steps)

        for r in range(update_steps):

            # update target networks
            if r % update_after == 0:
                for target_param, param in zip(ac_targ.pi.parameters(), ac.pi.parameters()):
                    target_param.data.copy_(param.data)

            # update q values
            q_update()
            # update policy
            pi_update()
            totat_updates += 1

        if buffer.stored_interactions() - current_steps >= test_after:
            print("=" * 80)
            test_agent(iteration)
            writer.add_scalar(
                'performed_steps', buffer.stored_interactions(), iteration)

            iteration += 1
            current_steps = buffer.stored_interactions()
            print('time for update:', time.time() - start_time)
            print('total updates', totat_updates)
            print('total steps', buffer.stored_interactions())
            start_time = time.time()
            totat_updates = 0
        writer.flush()
