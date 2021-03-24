import time
import torch


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
    """
    This is a blanc implementation of the sample and update cycles shared between the implemented algorithms.
    It is required that all functions manage their data on their own by having state.

    @param writer: SummaryWriter from tensorboard or equivalent stub
    @param q_update: function performing a single update step for the q values
    @param pi_update: function performing a single update step for the policy values
    @param one_step_update: function which updates target networks after each update cycle
    @param target_update: function which updates target networks after specifies many updates
    @param sampler: function performing actions in a priori defined environment
    @param test_agent: function that evaluates the current model
    @param buffer: replay buffer that stores performed steps in the environment
    @param total_steps: minimal amount of steps that will be performed through out training
    @param min_steps_per_iteration: the minimal amount of steps that will be performed per iteration
    @param test_after: the minimal amount of steps that will be performed before an evaluation of the model through
        the test_agent function
    @param update_steps: amount of updates before next sampling stage
    @param update_after: amount of updates before updating the target networks according to target_update
    """

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
            with torch.no_grad():
                test_agent(it)
            writer.add_scalar(
                'performed_steps', buffer.stored_interactions(), it)

            it += 1
            current_steps = buffer.stored_interactions()
            print('time for update:', time.time() - start_time)
            print('updates in epoch', total_updates)
            print('total trajectories', sampler.perfomed_traj)
            print('total steps', buffer.stored_interactions())
            start_time = time.time()
            total_updates = 0
        writer.flush()
