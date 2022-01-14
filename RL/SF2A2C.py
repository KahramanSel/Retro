import retro
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import A2C, PPO
import numpy as np
import gym


if __name__ == '__main__':

    #Vektor-Wrapper für eine einzelne Umgebung. Für Multiprozessierung sollte SubprocVecEnv genutzt werden
    env = DummyVecEnv([lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state='dhalsimvszangief', record = '.')])
    #Nutzung des A2C-Algorithmus.PPO unstabiler und langsamer in dieser Anwendung.
    model = A2C('MlpPolicy', env, verbose=1, device = 'cuda', tensorboard_log="./log_SF2/")
    model.learn(total_timesteps=100000)
    model.save('SF2A2C')
    #model = A2C.load('SF2A2C')
    #model.set_env(env)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones.all() == True:
            break
        

     
