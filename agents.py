"""Tabular Q-learning/SARSA and neural policy-gradient REINFORCE."""
import random
import numpy as np

def set_seed(seed):
 random.seed(seed);np.random.seed(seed)
 try:
  import torch
  torch.manual_seed(seed)
 except ImportError: pass

def choose_action(q,s,epsilon,rng):
 if rng.random()<epsilon:return int(rng.integers(q.shape[1]))
 return int(rng.choice(np.flatnonzero(q[s]==q[s].max())))

def train_tabular(env,algorithm='q_learning',episodes=1500,alpha=0.2,gamma=0.97,epsilon_start=1.0,epsilon_end=0.05,epsilon_decay=0.996,seed=42):
 if algorithm not in ('q_learning','sarsa'):raise ValueError('Unknown algorithm')
 rng=np.random.default_rng(seed);env.seed(seed)
 q=np.zeros((env.n_states,4),dtype=np.float32);history=[]
 for ep in range(episodes):
  s=env.reset();eps=max(epsilon_end,epsilon_start*epsilon_decay**ep)
  a=choose_action(q,s,eps,rng);total=0.0
  for t in range(env.max_steps):
   ns,r,done,info=env.step(a);total+=r
   if done:target=r
   elif algorithm=='q_learning':target=r+gamma*np.max(q[ns])
   else:
    na=choose_action(q,ns,eps,rng)
    target=r+gamma*q[ns,na]
   q[s,a]+=alpha*(target-q[s,a])
   if done:break
   s=ns;a=choose_action(q,s,eps,rng) if algorithm=='q_learning' else na
  history.append(dict(episode=ep+1,reward=total,success=int(info['success']),steps=t+1,epsilon=eps))
 return q,history

def build_policy(hidden=64):
 import torch.nn as nn
 return nn.Sequential(nn.Linear(4,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,4))

def train_reinforce(env,episodes=1500,lr=0.003,gamma=0.97,hidden=64,entropy_coef=0.01,seed=42):
 import torch
 from torch.distributions import Categorical
 set_seed(seed);env.seed(seed)
 net=build_policy(hidden);optimizer=torch.optim.Adam(net.parameters(),lr=lr)
 history=[]
 for ep in range(episodes):
  env.reset();logps=[];entropies=[];rewards=[]
  for t in range(env.max_steps):
   obs=torch.tensor(env.observation(),dtype=torch.float32)
   dist=Categorical(logits=net(obs));action=dist.sample()
   logps.append(dist.log_prob(action));entropies.append(dist.entropy())
   _,r,done,info=env.step(int(action.item()));rewards.append(r)
   if done:break
  returns=[];g=0.0
  for r in reversed(rewards):
   g=r+gamma*g;returns.append(g)
  returns=torch.tensor(list(reversed(returns)),dtype=torch.float32)
  if len(returns)>1:returns=(returns-returns.mean())/(returns.std(unbiased=False)+1e-8)
  loss=-(torch.stack(logps)*returns).sum()-entropy_coef*torch.stack(entropies).sum()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.0);optimizer.step()
  history.append(dict(episode=ep+1,reward=sum(rewards),success=int(info['success']),steps=t+1,epsilon=''))
 return net,history

def policy_action(net,env):
 import torch
 with torch.no_grad():return int(net(torch.tensor(env.observation(),dtype=torch.float32)).argmax().item())

def evaluate(env,model,algorithm,episodes=100,seed=10000):
 rng=np.random.default_rng(seed);env.seed(seed)
 rewards=[];successes=[];steps=[]
 for _ in range(episodes):
  s=env.reset();total=0.0
  for t in range(env.max_steps):
   a=policy_action(model,env) if algorithm=='reinforce' else choose_action(model,s,0,rng)
   s,r,done,info=env.step(a);total+=r
   if done:break
  rewards.append(total);successes.append(int(info['success']));steps.append(t+1)
 return dict(success_rate=float(np.mean(successes)),mean_reward=float(np.mean(rewards)),mean_steps=float(np.mean(steps)))
