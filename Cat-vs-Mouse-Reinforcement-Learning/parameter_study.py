"""Controlled hyperparameter study. Runs three algorithms; records actual trial count."""
import argparse,itertools
from pathlib import Path
from run_experiments import train_one,save_csv

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--episodes',type=int,default=350)
 p.add_argument('--seeds',type=int,default=2)
 p.add_argument('--map',choices=['empty','obstacles','hard'],default='obstacles')
 p.add_argument('--mode',choices=['static','moving'],default='moving')
 p.add_argument('--out',default='results/tuning')
 args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 trials=[]
 # alpha x gamma x epsilon-decay for tabular; lr x hidden x entropy for neural.
 for algo in ('q_learning','sarsa','reinforce'):
  configs=(dict(alpha=a,gamma=g,eps_decay=e,lr=.003,hidden=64,entropy=.01)
           for a,g,e in itertools.product([.1,.3],[.9,.99],[.99,.997])) if algo!='reinforce' else (
           dict(alpha=.2,gamma=.97,eps_decay=.996,lr=lr,hidden=h,entropy=ent)
           for lr,h,ent in itertools.product([.001,.005],[32,64],[0.0,.02]))
  for params in configs:
   for seed in range(args.seeds):
    row,_=train_one(out,args.map,args.mode,'standard',algo,seed,args.episodes,50,
                    params['alpha'],params['gamma'],params['eps_decay'],params['lr'],
                    params['hidden'],params['entropy'],.1,save_model=False)
    trials.append({**row,**params})
    save_csv(out/'hyperparameter_trials.csv',trials)
    print(f'{len(trials)} {algo} seed={seed} success={row["success_rate"]:.0%}',flush=True)
 print(f'Finished {len(trials)} actual trials; output: {out}')
if __name__=='__main__':main()
