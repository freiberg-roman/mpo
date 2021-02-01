from mpo_retrace_alt.mpo_retrace_alt import mpo_retrace
import gym

if __name__ == "__main__":
    mpo_retrace(env=gym.make('Pendulum-v0'))
    # mpo_retrace(env=gym.make('Ant-v2'))

