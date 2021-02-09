from tqdm import tqdm


def mpo_runner(writer,
               q_update,
               pi_update,
               sampler,
               test_agent,
               ac,
               ac_targ,
               min_steps_per_epoch=4000,
               update_steps=1200,
               update_after=300,
               epochs=20,
               ):
    iteration = 0
    performed_steps = 0
    for it in range(iteration, epochs):
        # Find better policy by gradient descent
        performed_steps = 0
        while performed_steps < min_steps_per_epoch:
            performed_steps += sampler(it)

        for r in tqdm(range(update_steps), desc='updating nets'):

            # update target networks
            if r % update_after == 0:
                for target_param, param in zip(ac_targ.parameters(), ac.parameters()):
                    target_param.data.copy_(param.data)

            # update q values
            q_update(it * update_steps + r)
            # update policy
            pi_update(it * update_steps + r)

        test_agent(it)
        writer.flush()
        it += 1
