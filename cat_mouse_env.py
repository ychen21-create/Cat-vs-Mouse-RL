"""Custom GridWorld with static or moving mouse; optional Gymnasium adapter.
State encodes BOTH cat and mouse positions, even in static mode.
"""
import numpy as np
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_NAMES = ['UP','DOWN','LEFT','RIGHT']
MAPS = {
 'empty': ['.....']*5,
 'obstacles': ['.....','.#.#.','.....','.#.#.','.....'],
 'hard': ['.......','.###.#.','...#.#.','##.#.#.','.....#.','.#####.','.......'],
}
REWARDS = {
 'standard': dict(catch=10.0, step=-0.1, wall=-2.0, distance=0.0),
 'shaped': dict(catch=15.0, step=-0.15, wall=-1.5, distance=0.25),
}
class CatMouseEnv:
 def __init__(self, variant='obstacles', mouse_mode='static', reward_mode='standard', slip_prob=0.0, max_steps=None, seed=42):
  if variant not in MAPS or mouse_mode not in ('static','moving') or reward_mode not in REWARDS:
   raise ValueError('Invalid map, mouse mode or reward mode')
  if not 0<=slip_prob<=1: raise ValueError('slip_prob must be in [0,1]')
  self.variant,self.mouse_mode,self.reward_mode=variant,mouse_mode,reward_mode
  self.layout=MAPS[variant]; self.size=len(self.layout)
  self.obstacles={(r,c) for r,row in enumerate(self.layout) for c,v in enumerate(row) if v=='#'}
  self.start=(self.size-1,0); self.goal=(0,self.size-1)
  self.max_steps=max_steps or (100 if variant=='hard' else 65)
  self.slip_prob=slip_prob; self.rng=np.random.default_rng(seed)
  self.n_actions=4; self.n_states=(self.size*self.size)**2
  self.reset()
 def seed(self, seed): self.rng=np.random.default_rng(seed)
 def state(self):
  n=self.size; return (self.pos[0]*n+self.pos[1])*(n*n)+(self.mouse[0]*n+self.mouse[1])
 def observation(self):
  n=self.size-1
  return np.asarray([self.pos[0]/n,self.pos[1]/n,self.mouse[0]/n,self.mouse[1]/n],dtype=np.float32)
 def reset(self, seed=None):
  if seed is not None: self.seed(seed)
  self.pos=self.start;self.mouse=self.goal;self.steps=0
  return self.state()
 def _move(self,pos,action):
  dr,dc=ACTIONS[action]; p=(pos[0]+dr,pos[1]+dc)
  return pos if not (0<=p[0]<self.size and 0<=p[1]<self.size) or p in self.obstacles else p
 def step(self,action):
  if not 0<=int(action)<4: raise ValueError('Invalid action')
  prev_dist=abs(self.pos[0]-self.mouse[0])+abs(self.pos[1]-self.mouse[1])
  actual=int(self.rng.integers(4)) if self.rng.random()<self.slip_prob else int(action)
  old=self.pos; self.pos=self._move(self.pos,actual)
  wall=(old==self.pos)
  caught=(self.pos==self.mouse)
  # Mouse moves AFTER cat, unless caught. Mouse is uniformly random over legal moves + stay.
  if self.mouse_mode=='moving' and not caught:
   moves=[self.mouse]+list({self._move(self.mouse,a) for a in range(4)})
   moves=sorted(set(moves))
   self.mouse=moves[int(self.rng.integers(len(moves)))]
   caught=(self.pos==self.mouse)
  self.steps+=1
  terminated=caught; truncated=(self.steps>=self.max_steps and not caught)
  cfg=REWARDS[self.reward_mode]
  new_dist=abs(self.pos[0]-self.mouse[0])+abs(self.pos[1]-self.mouse[1])
  reward=(cfg['catch'] if caught else cfg['step']+(cfg['wall'] if wall else 0)+cfg['distance']*(prev_dist-new_dist))
  return self.state(),float(reward),terminated or truncated,{'success':caught,'terminated':terminated,'truncated':truncated,'wall':wall}
 def render(self):
  return '\n'.join(' '.join('C' if (r,c)==self.pos else 'M' if (r,c)==self.mouse else '#' if (r,c) in self.obstacles else '.' for c in range(self.size)) for r in range(self.size))

# Gymnasium-compatible wrapper is optional; training does not require gymnasium.
def make_gym_env(**kwargs):
 try:
  import gymnasium as gym
  from gymnasium import spaces
 except ImportError as exc:
  raise ImportError('Install optional dependency: pip install gymnasium') from exc
 class GymCatMouse(gym.Env):
  metadata={'render_modes':['ansi']}
  def __init__(self):
   self.game=CatMouseEnv(**kwargs)
   self.action_space=spaces.Discrete(4)
   self.observation_space=spaces.Box(low=0,high=1,shape=(4,),dtype=np.float32)
  def reset(self,*,seed=None,options=None):
   super().reset(seed=seed);self.game.reset(seed=seed)
   return self.game.observation(),{}
  def step(self,action):
   _,r,_,info=self.game.step(action)
   return self.game.observation(),r,info['terminated'],info['truncated'],info
  def render(self): return self.game.render()
  def close(self): pass
 return GymCatMouse()
