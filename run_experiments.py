"""Reproducible comparisons: 3 algorithms, 2 task types, 3 maps, 2 rewards, multi-seed.
Quick smoke: python run_experiments.py --episodes 20 --seeds 1 --maps empty --modes static
"""
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cat_mouse_env import CatMouseEnv,MAPS,REWARDS
from agents import train_tabular,train_reinforce,evaluate
ALGOS=('q_learning','sarsa','reinforce')

def save_csv(path,rows):
 if not rows:return
 with open(path,'w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def train_one(out,variant,mode,reward,algo,seed,episodes,eval_episodes,alpha,gamma,eps_decay,lr,hidden,entropy,slip,save_model=True):
 import torch
 env=CatMouseEnv(variant,mode,reward,slip_prob=slip,seed=seed)
 start=time.perf_counter()
 if algo=='reinforce':
  model,hist=train_reinforce(env,episodes=episodes,lr=lr,gamma=gamma,hidden=hidden,entropy_coef=entropy,seed=seed)
 else:
  model,hist=train_tabular(env,algo,episodes=episodes,alpha=alpha,gamma=gamma,epsilon_decay=eps_decay,seed=seed)
 duration=time.perf_counter()-start
 test=CatMouseEnv(variant,mode,reward,slip_prob=slip,seed=seed+10000)
 metrics=evaluate(test,model,algo,episodes=eval_episodes,seed=seed+10000)
 prefix=f'{variant}_{mode}_{reward}_{algo}_seed{seed}'
 if save_model:
  if algo=='reinforce':
   torch.save({'state_dict':model.state_dict(),'hidden':hidden},out/f'{prefix}.pt')
  else:np.save(out/f'{prefix}.npy',model)
  save_csv(out/f'{prefix}_training.csv',hist)
 row=dict(map=variant,mouse_mode=mode,reward_mode=reward,algorithm=algo,seed=seed,episodes=episodes,train_seconds=round(duration,3),**metrics)
 return row,hist

def plot_results(out,curves):
 if not curves:return
 fig,axes=plt.subplots(1,2,figsize=(13,4.5))
 for algo in ALGOS:
  h=[x for key,x in curves if key==algo]
  if not h:continue
  min_len=min(map(len,h));window=min(100,min_len)
  vals=np.mean([np.array([r['reward'] for r in rows[:min_len]],dtype=float) for rows in h],axis=0)
  suc=np.mean([np.array([r['success'] for r in rows[:min_len]],dtype=float) for rows in h],axis=0)
  for ax,data in zip(axes,[vals,suc]):
   sm=np.convolve(data,np.ones(window)/window,'valid')
   ax.plot(np.arange(window,min_len+1),sm,label=algo)
 for ax,title,ylabel in zip(axes,['Training reward','Training success rate'],['Reward','Success rate']):
  ax.set_title(title);ax.set_xlabel('Episode');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--episodes',type=int,default=1000)
 p.add_argument('--eval-episodes',type=int,default=100)
 p.add_argument('--seeds',type=int,default=3,help='Number of independent seeds (use 5 for final report)')
 p.add_argument('--maps',nargs='+',choices=list(MAPS),default=['empty','obstacles','hard'])
 p.add_argument('--modes',nargs='+',choices=['static','moving'],default=['static','moving'])
 p.add_argument('--rewards',nargs='+',choices=list(REWARDS),default=['standard','shaped'])
 p.add_argument('--algorithms',nargs='+',choices=ALGOS,default=list(ALGOS))
 p.add_argument('--alpha',type=float,default=.2);p.add_argument('--gamma',type=float,default=.97)
 p.add_argument('--epsilon-decay',type=float,default=.996)
 p.add_argument('--lr',type=float,default=.003);p.add_argument('--hidden',type=int,default=64)
 p.add_argument('--entropy',type=float,default=.01);p.add_argument('--slip',type=float,default=.1)
 p.add_argument('--out',default='results');args=p.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
 rows=[];curves=[];n=len(args.maps)*len(args.modes)*len(args.rewards)*len(args.algorithms)*args.seeds
 print(f'Running {n} trainings. For a quick check use --episodes 30 --seeds 1 --maps empty --modes static --rewards standard')
 k=0
 for variant in args.maps:
  for mode in args.modes:
   for reward in args.rewards:
    for algo in args.algorithms:
     for seed in range(args.seeds):
      k+=1
      row,hist=train_one(out,variant,mode,reward,algo,seed,args.episodes,args.eval_episodes,args.alpha,args.gamma,args.epsilon_decay,args.lr,args.hidden,args.entropy,args.slip)
      rows.append(row)
      if variant==args.maps[0] and mode==args.modes[0] and reward==args.rewards[0]:curves.append((algo,hist))
      print(f'[{k}/{n}] {variant}/{mode}/{reward}/{algo} seed={seed}: success={row["success_rate"]:.0%} reward={row["mean_reward"]:.2f} {row["train_seconds"]:.1f}s',flush=True)
      save_csv(out/'evaluation_summary.csv',rows)
 plot_results(out,curves)
 groups={}
 for row in rows:
  key=(row['map'],row['mouse_mode'],row['reward_mode'],row['algorithm'])
  groups.setdefault(key,[]).append(row)
 aggregates=[]
 for key,rs in groups.items():
  record=dict(zip(['map','mouse_mode','reward_mode','algorithm'],key),n_seeds=len(rs))
  for metric in ['success_rate','mean_reward','mean_steps','train_seconds']:
   vals=np.array([r[metric] for r in rs]);record[metric+'_mean']=float(vals.mean());record[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
  aggregates.append(record)
 save_csv(out/'aggregate_summary.csv',aggregates)
 import torch
 metadata=dict(python=platform.python_version(),platform=platform.platform(),cpu=platform.processor(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),args=vars(args),note='Training hardware is measured on the machine that runs this command.')
 (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 print('Saved:',out.resolve())
if __name__=='__main__':main()
