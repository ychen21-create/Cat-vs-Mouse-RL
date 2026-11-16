import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from cat_mouse_env import CatMouseEnv
from agents import train_tabular,train_reinforce,evaluate

def test_states_and_bounds():
 for variant in ('empty','obstacles','hard'):
  for mode in ('static','moving'):
   env=CatMouseEnv(variant,mode,seed=10)
   assert env.n_states==env.size**4
   for _ in range(100):
    s,r,done,info=env.step(3)
    assert 0<=s<env.n_states
    assert env.pos not in env.obstacles and env.mouse not in env.obstacles
    if done:env.reset()

def test_seed_determinism():
 def trace():
  env=CatMouseEnv('obstacles','moving',slip_prob=.25,seed=7)
  return [env.step(0) for _ in range(12)]
 assert trace()==trace()

def test_algorithms_smoke():
 for algo in ('q_learning','sarsa'):
  env=CatMouseEnv('empty',seed=4)
  q,h=train_tabular(env,algo,episodes=4,seed=4)
  assert q.shape==(env.n_states,4) and len(h)==4
  result=evaluate(CatMouseEnv('empty',seed=5),q,algo,episodes=2)
  assert 0<=result['success_rate']<=1

def test_policy_smoke():
 env=CatMouseEnv('empty',seed=4)
 net,h=train_reinforce(env,episodes=3,hidden=16,seed=4)
 assert len(h)==3
 result=evaluate(CatMouseEnv('empty',seed=5),net,'reinforce',episodes=2)
 assert np.isfinite(result['mean_reward'])
