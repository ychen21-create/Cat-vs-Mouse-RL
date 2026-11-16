"""Text demonstration for any map/mode and any trained algorithm."""
import argparse
from pathlib import Path
import numpy as np
from cat_mouse_env import CatMouseEnv
from agents import build_policy,policy_action

def load_model(path,algo):
 if algo!='reinforce':return np.load(path)
 import torch
 data=torch.load(path,map_location='cpu',weights_only=True)
 net=build_policy(data['hidden']);net.load_state_dict(data['state_dict']);net.eval();return net

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--map',default='hard',choices=['empty','obstacles','hard'])
 p.add_argument('--mode',default='static',choices=['static','moving'])
 p.add_argument('--reward',default='standard',choices=['standard','shaped'])
 p.add_argument('--algorithm',default='q_learning',choices=['q_learning','sarsa','reinforce'])
 p.add_argument('--seed',type=int,default=0);p.add_argument('--model-seed',type=int,default=0)
 args=p.parse_args();env=CatMouseEnv(args.map,args.mode,args.reward,slip_prob=.1,seed=args.seed)
 prefix=f'{args.map}_{args.mode}_{args.reward}_{args.algorithm}_seed{args.model_seed}'
 path=Path('results')/(prefix+('.pt' if args.algorithm=='reinforce' else '.npy'))
 if not path.exists():raise FileNotFoundError(f'{path}: train this combination first')
 model=load_model(path,args.algorithm);s=env.reset();print(env.render())
 for i in range(env.max_steps):
  action=policy_action(model,env) if args.algorithm=='reinforce' else int(np.argmax(model[s]))
  s,r,done,info=env.step(action)
  print(f'\nStep {i+1} reward={r:.2f}\n{env.render()}')
  if done:print('Caught!' if info['success'] else 'Timeout');break
if __name__=='__main__':main()
