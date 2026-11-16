"""Cat vs Mouse — polished Tkinter RL dashboard. Run: python visual_game.py"""
import csv
import math
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import numpy as np
from cat_mouse_env import CatMouseEnv, ACTION_NAMES
from agents import build_policy, policy_action

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BG = "#0B1220"
PANEL = "#151F32"
BORDER = "#26344C"
TEXT = "#EDF4FF"
MUTED = "#92A6C2"
TEAL = "#48E1BE"
PINK = "#FF8CB3"
GOLD = "#FFC77D"
BLUE = "#79A8FF"

class Arena:
    CELL = 66
    def __init__(self, parent, algo, accent):
        self.algo, self.accent = algo, accent
        self.frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        self.frame.pack(side="left", expand=True, fill="both", padx=8)
        header = tk.Frame(self.frame, bg=PANEL)
        header.pack(fill="x", padx=18, pady=(17, 6))
        tk.Label(header, text=algo.replace("_", "-").upper(), font=("Segoe UI", 15, "bold"),
                 bg=PANEL, fg=accent).pack(side="left")
        self.tag = tk.Label(header, text="READY", bg="#24364A", fg=TEXT,
                            font=("Segoe UI", 9, "bold"), padx=10, pady=4)
        self.tag.pack(side="right")
        self.canvas = tk.Canvas(self.frame, width=350, height=350, bg=PANEL, highlightthickness=0)
        self.canvas.pack(padx=12, pady=(5, 7))
        self.stats = tk.Label(self.frame, bg=PANEL, fg=MUTED, font=("Consolas", 11))
        self.stats.pack(pady=(4, 14))
        self.env = None
        self.q = None
        self.history = []
        self.done = False
        self.total = 0
        self.state = None
        self.invalid = False

    def reset(self, variant, mouse_mode, reward_mode, seed):
        self.env = CatMouseEnv(variant=variant, mouse_mode=mouse_mode, reward_mode=reward_mode, slip_prob=.1, seed=seed)
        self.state = self.env.reset()
        prefix = f"{variant}_{mouse_mode}_{reward_mode}_{self.algo}_seed0"
        if self.algo == "reinforce":
            import torch
            data = torch.load(RESULTS / (prefix + ".pt"), map_location="cpu", weights_only=True)
            self.q = build_policy(data["hidden"])
            self.q.load_state_dict(data["state_dict"])
            self.q.eval()
        else:
            self.q = np.load(RESULTS / (prefix + ".npy"))
            if self.q.shape != (self.env.n_states, self.env.n_actions):
                raise ValueError("Q-table shape does not match environment")
        self.history = [self.env.pos]
        self.done = False
        self.total = 0
        self.invalid = False
        self.tag.configure(text="READY", fg=TEXT, bg="#24364A")
        self.draw()

    def move(self):
        if self.done:
            return
        action = policy_action(self.q, self.env) if self.algo == "reinforce" else int(np.argmax(self.q[self.state]))
        old = self.env.pos
        self.state, reward, self.done, info = self.env.step(action)
        self.total += reward
        self.invalid = self.env.pos == old
        self.history.append(self.env.pos)
        if self.done:
            self.tag.configure(text="CAUGHT!" if info["success"] else "TIMEOUT",
                               bg="#164A42" if info["success"] else "#673347",
                               fg=TEAL if info["success"] else PINK)
        else:
            self.tag.configure(text=ACTION_NAMES[action], bg="#24364A", fg=self.accent)
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        n = self.env.size
        s = min(self.CELL, 340/n)
        off = (350 - n*s)/2
        def center(pos):
            return off + pos[1]*s+s/2, off+pos[0]*s+s/2
        for r in range(n):
            for col in range(n):
                x, y = off+col*s, off+r*s
                pos = (r,col)
                wall = pos in self.env.obstacles
                color = "#35445C" if wall else ("#2C2C49" if pos==self.env.mouse else
                                               ("#1B2B40" if (r+col)%2 else "#202F45"))
                c.create_rectangle(x+2,y+2,x+s-2,y+s-2,fill=color,outline="",width=0)
                if wall:
                    for j in range(3):
                        c.create_line(x+12,y+18+j*13,x+s-12,y+18+j*13,fill="#536782",width=2)
                elif pos == self.env.mouse:
                    c.create_oval(x+8,y+8,x+s-8,y+s-8,outline=PINK,width=2,dash=(4,4))
        # Trail of previously visited cells
        if len(self.history)>1:
            coords = [v for pos in self.history for v in center(pos)]
            c.create_line(*coords,fill=self.accent,width=4,smooth=False,capstyle="round")
            for pos in self.history[:-1]:
                x,y = center(pos)
                c.create_oval(x-4,y-4,x+4,y+4,fill=self.accent,outline="")
        # Mouse (fixed target)
        mx,my = center(self.env.mouse)
        if self.env.pos != self.env.mouse:
            c.create_oval(mx-16,my-13,mx+16,my+15,fill="#FFB2C8",outline="#E56E98",width=2)
            c.create_oval(mx-19,my-21,mx-4,my-7,fill="#FFB2C8",outline="#E56E98",width=2)
            c.create_oval(mx+4,my-21,mx+19,my-7,fill="#FFB2C8",outline="#E56E98",width=2)
            c.create_oval(mx-8,my-3,mx-5,my,fill="#243044",outline="")
            c.create_oval(mx+5,my-3,mx+8,my,fill="#243044",outline="")
            c.create_oval(mx-2,my+5,mx+2,my+9,fill="#E56E98",outline="")
        # Cat with triangular ears, eyes, whiskers
        x,y = center(self.env.pos)
        c.create_polygon(x-19,y-7,x-21,y-26,x-5,y-17,fill=GOLD,outline="#DB9E52",width=2)
        c.create_polygon(x+19,y-7,x+21,y-26,x+5,y-17,fill=GOLD,outline="#DB9E52",width=2)
        c.create_oval(x-23,y-20,x+23,y+23,fill=GOLD,outline="#DB9E52",width=2)
        for dx in (-9,9):
            c.create_oval(x+dx-3,y-3,x+dx+3,y+3,fill="#25344B",outline="")
        c.create_polygon(x-3,y+8,x+3,y+8,x,y+12,fill="#D87B82",outline="")
        for side in (-1,1):
            for dy in (6,13):
                c.create_line(x+side*8,y+dy,x+side*28,y+dy-2,fill="#F7E5CC",width=2)
        self.stats.configure(text=f"STEPS  {self.env.steps:02d}     REWARD  {self.total:+.1f}"
                             + ("    • WALL" if self.invalid else ""))

class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title("CAT & MOUSE  |  Reinforcement Learning Lab")
        root.configure(bg=BG)
        root.geometry("1260x790")
        root.minsize(1130,730)
        self.running = False
        self.job = None
        self.variant = tk.StringVar(value="obstacles")
        self.mouse_mode = tk.StringVar(value="static")
        self.reward_mode = tk.StringVar(value="standard")
        self.game_seed = 101
        self.delay = tk.IntVar(value=380)
        self.make_ui()
        self.reset()
        root.protocol("WM_DELETE_WINDOW",self.close)

    def make_ui(self):
        top = tk.Frame(self.root,bg=BG)
        top.pack(fill="x",padx=30,pady=(22,8))
        tk.Label(top,text="CAT  &  MOUSE",bg=BG,fg=TEXT,
                 font=("Segoe UI",24,"bold")).pack(anchor="w")
        tk.Label(top,text="REINFORCEMENT LEARNING  /  LIVE POLICY ARENA",
                 bg=BG,fg=TEAL,font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(2,0))
        bar = tk.Frame(self.root,bg=BG)
        bar.pack(fill="x",padx=30,pady=(12,14))
        tk.Label(bar,text="ENVIRONMENT",bg=BG,fg=MUTED,font=("Segoe UI",9,"bold")).pack(side="left",padx=(0,10))
        selector = ttk.Combobox(bar,textvariable=self.variant,values=["empty","obstacles","hard"],
                                state="readonly",width=13)
        selector.pack(side="left")
        selector.bind("<<ComboboxSelected>>",lambda e:self.reset())
        tk.Label(bar,text="MOUSE",bg=BG,fg=MUTED,font=("Segoe UI",9,"bold")).pack(side="left",padx=(16,6))
        mselect=ttk.Combobox(bar,textvariable=self.mouse_mode,values=["static","moving"],state="readonly",width=10)
        mselect.pack(side="left");mselect.bind("<<ComboboxSelected>>",lambda e:self.reset())
        tk.Label(bar,text="REWARD",bg=BG,fg=MUTED,font=("Segoe UI",9,"bold")).pack(side="left",padx=(16,6))
        rselect=ttk.Combobox(bar,textvariable=self.reward_mode,values=["standard","shaped"],state="readonly",width=10)
        rselect.pack(side="left");rselect.bind("<<ComboboxSelected>>",lambda e:self.reset())
        tk.Label(bar,text="ANIMATION SPEED",bg=BG,fg=MUTED,
                 font=("Segoe UI",9,"bold")).pack(side="left",padx=(28,9))
        tk.Scale(bar,variable=self.delay,from_=900,to=100,orient="horizontal",
                 length=180,showvalue=False,bg=BG,fg=TEXT,highlightthickness=0,
                 troughcolor="#344760",activebackground=TEAL).pack(side="left")
        self.mode = tk.Label(bar,text="● READY",bg=BG,fg=TEAL,font=("Segoe UI",10,"bold"))
        self.mode.pack(side="right")
        boards = tk.Frame(self.root,bg=BG)
        boards.pack(fill="x",padx=22)
        self.arenas = [Arena(boards,"q_learning",TEAL),Arena(boards,"sarsa",BLUE),Arena(boards,"reinforce",PINK)]
        controls = tk.Frame(self.root,bg=BG)
        controls.pack(fill="x",padx=30,pady=(18,8))
        def button(text,command,primary=False):
            tk.Button(controls,text=text,command=command,bd=0,cursor="hand2",
                      font=("Segoe UI",10,"bold"),padx=19,pady=11,
                      bg=TEAL if primary else PANEL,fg=BG if primary else TEXT,
                      activebackground="#70F0D2" if primary else "#263951",
                      activeforeground=BG if primary else TEXT).pack(side="left",padx=(0,9))
        button("▶  PLAY",self.play,True)
        button("Ⅱ  PAUSE",self.pause)
        button("→  STEP",self.step)
        button("↻  RESET",self.reset)
        button("↗  CHARTS",self.charts)
        legend = tk.Frame(self.root,bg=BG)
        legend.pack(fill="x",padx=30,pady=(3,8))
        tk.Label(legend,text="● Orange cat = agent     ● Pink mouse = target (can move)     ▦ = wall     — = path taken",
                 bg=BG,fg=MUTED,font=("Segoe UI",10)).pack(anchor="w")
        chartframe = tk.Frame(self.root,bg=PANEL,highlightbackground=BORDER,highlightthickness=1)
        chartframe.pack(fill="both",expand=True,padx=30,pady=(6,20))
        tk.Label(chartframe,text="TRAINING PERFORMANCE  /  MOVING AVERAGE REWARD",
                 bg=PANEL,fg=TEXT,font=("Segoe UI",10,"bold")).pack(anchor="w",padx=14,pady=(10,0))
        self.chart = tk.Canvas(chartframe,height=110,bg=PANEL,highlightthickness=0)
        self.chart.pack(fill="both",expand=True,padx=14,pady=(4,10))
        self.chart.bind("<Configure>",lambda e:self.draw_chart())

    def reset(self):
        self.pause()
        try:
            for a in self.arenas:
                a.reset(self.variant.get(),self.mouse_mode.get(),self.reward_mode.get(),self.game_seed)
            self.mode.configure(text="● READY",fg=TEAL)
            self.draw_chart()
        except (FileNotFoundError,ValueError) as e:
            messagebox.showerror("Missing training data",
                f"{e}\n\nRun this command in the project folder first:\npython run_experiments.py")

    def step(self):
        self.pause()
        for a in self.arenas:
            a.move()
        if all(a.done for a in self.arenas):
            self.mode.configure(text="● FINISHED",fg=GOLD)

    def play(self):
        if self.running:
            return
        if all(a.done for a in self.arenas):
            self.reset()
        self.running=True
        self.mode.configure(text="● LIVE",fg=TEAL)
        self.tick()

    def tick(self):
        if not self.running:
            return
        for a in self.arenas:
            a.move()
        if all(a.done for a in self.arenas):
            self.pause()
            self.mode.configure(text="● FINISHED",fg=GOLD)
        else:
            self.job=self.root.after(self.delay.get(),self.tick)

    def pause(self):
        self.running=False
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job=None
        if hasattr(self,"mode"):
            self.mode.configure(text="● PAUSED",fg=MUTED)

    def charts(self):
        import os,subprocess,sys
        p=RESULTS/"learning_curves.png"
        if not p.exists():
            messagebox.showinfo("Not found","Run python run_experiments.py first.")
            return
        if sys.platform=="win32": os.startfile(str(p))
        elif sys.platform=="darwin": subprocess.Popen(["open",str(p)])
        else: subprocess.Popen(["xdg-open",str(p)])

    def draw_chart(self):
        if not hasattr(self,"chart"):
            return
        c=self.chart
        c.delete("all")
        w,h=c.winfo_width(),c.winfo_height()
        if w<100 or h<60: return
        series=[]
        for algo,color in (("q_learning",TEAL),("sarsa",BLUE),("reinforce",PINK)):
            p=RESULTS/f"{self.variant.get()}_{self.mouse_mode.get()}_{self.reward_mode.get()}_{algo}_seed0_training.csv"
            if not p.exists(): continue
            with p.open(newline="",encoding="utf-8") as f:
                vals=np.array([float(row["reward"]) for row in csv.DictReader(f)])
            if len(vals)<2: continue
            window=min(150,len(vals))
            smooth=np.convolve(vals,np.ones(window)/window,mode="valid")
            series.append((smooth,color,algo))
        if not series: return
        ymin=min(float(v.min()) for v,_,_ in series)
        ymax=max(float(v.max()) for v,_,_ in series)
        if ymax-ymin<1e-6: ymax=ymin+1
        left,right,top,bottom=38,w-14,10,h-24
        for j in range(3):
            y=top+(bottom-top)*j/2
            c.create_line(left,y,right,y,fill="#304158",dash=(3,5))
            c.create_text(30,y,text=f"{ymax-(ymax-ymin)*j/2:.0f}",
                          anchor="e",fill=MUTED,font=("Segoe UI",8))
        for vals,color,name in series:
            points=[]
            skip=max(1,len(vals)//280)
            for i in range(0,len(vals),skip):
                points.extend((left+(right-left)*i/max(1,len(vals)-1),
                               bottom-(bottom-top)*(float(vals[i])-ymin)/(ymax-ymin)))
            if len(points)>=4: c.create_line(*points,fill=color,width=2,smooth=True)
        c.create_text(left,bottom+13,text="EPISODE 0",anchor="w",fill=MUTED,font=("Segoe UI",8))
        c.create_text(right,bottom+13,text=f"EPISODE {len(series[0][0])}",
                      anchor="e",fill=MUTED,font=("Segoe UI",8))

    def close(self):
        self.pause()
        self.root.destroy()

if __name__=="__main__":
    app=tk.Tk()
    Dashboard(app)
    app.mainloop()
