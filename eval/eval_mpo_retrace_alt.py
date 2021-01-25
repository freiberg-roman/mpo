from mpo_retrace_alt.mpo_retrace_alt import MPO
import gym

if __name__ == "__main__":
    model = MPO(
        device='cpu',
        env=gym.make('Pendulum-v0'),

    )

    model.train()

