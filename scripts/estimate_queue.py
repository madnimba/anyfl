#!/usr/bin/env python3
"""Estimate queue wall-clock from the measured timing model.

Per-condition 80-epoch costs are fitted from results/calib.jsonl (two-point fits
in training epochs, plus a separate recon-epoch fit for rgar_full). The model was
validated against a real 80-epoch run: MNIST rgar_full predicted 1381 s vs 1370 s
measured, 0.8% error.

Schedules jobs with the same constraints run_queue.py uses: N workers, and a GPU
semaphore so cuda jobs cannot exceed --gpu-slots concurrently.
"""
from __future__ import annotations
import argparse, heapq, json, os, sys, collections

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

RECON = {"MNIST":120,"FASHIONMNIST":120,"Fashion-MNIST":120,"UCI-HAR":120,
         "UCI-BANK":80,"UCI-MUSHROOM":280,"UCI-Mushroom":280,"CIFAR-10":140,"CIFAR10":140}
ALIAS = {"Fashion-MNIST":"FASHIONMNIST","UCI-Mushroom":"UCI-MUSHROOM","CIFAR-10":"CIFAR10"}

def fit(path="results/calib.jsonl"):
    pts = collections.defaultdict(dict)
    for l in open(os.path.join(_R, path)):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        E=int(r["config"]["train"]["epochs"]); R=22 if r["job_id"].endswith("R22") else 2
        pts[(r["dataset"], r["condition"])][(E,R)] = float(r["wall_clock_s"])
    out={}
    for (ds,c),d in pts.items():
        if c=="rgar_full":
            b0,e9,r22 = d.get((1,2)), d.get((9,2)), d.get((1,22))
            if b0 is None or e9 is None: continue
            a=(e9-b0)/8.0; b=((r22-b0)/20.0) if r22 else 0.0
            out[(ds,c)] = b0 + 79*a + (RECON.get(ds,120)-2)*b
        else:
            ks=sorted(d)
            if len(ks)<2 or ks[0][0]==ks[-1][0]:
                out[(ds,c)]=d[ks[0]]*80; continue
            a=(d[ks[-1]]-d[ks[0]])/(ks[-1][0]-ks[0][0])
            out[(ds,c)] = d[ks[0]] + (80-ks[0][0])*a
    return out

def job_seconds(j, T, overhead=25.0):
    ds = ALIAS.get(j["dataset"], j["dataset"])
    a = j["argv"]; runner = a[0]
    strat=[]; i=0
    while i < len(a):
        if a[i]=="--strategy":
            k=i+1
            while k<len(a) and not a[k].startswith("--"): strat.append(a[k]); k+=1
            i=k
        else: i+=1
    ns=max(1,len(strat))
    g=lambda c: T.get((ds,c), 0.0)
    if runner.endswith("run_attack.py"):
        return g("clean") + ns*g("attack") + overhead
    if runner.endswith("run_attack_defense.py"):
        return g("clean") + ns*(g("naked")+g("rgar_full")) + overhead
    return g("clean") + 4*g("naked") + g("rgar_full") + overhead   # sota: 5 arms

def simulate(jobs, T, workers, gpu_slots):
    order = sorted(jobs, key=lambda r:(int(r["tier"]), not r.get("needs_gpu"), -job_seconds(r,T)))
    free=[0.0]*workers; gpu_free=[0.0]*gpu_slots; t_end=0.0
    for j in order:
        d = job_seconds(j, T)
        wi = min(range(workers), key=lambda i: free[i]); start = free[wi]
        if str(j.get("device","auto")) != "cpu":
            gi = min(range(gpu_slots), key=lambda i: gpu_free[i])
            start = max(start, gpu_free[gi]); gpu_free[gi] = start + d
        free[wi] = start + d; t_end = max(t_end, free[wi])
    return t_end

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", default="experiments/manifest.jsonl")
    p.add_argument("--laptop-workers", type=int, default=4)
    p.add_argument("--laptop-gpu-slots", type=int, default=2)
    p.add_argument("--a5500-workers", type=int, default=4)
    p.add_argument("--a5500-gpu-slots", type=int, default=3)
    a=p.parse_args()
    T=fit()
    jobs=[json.loads(l) for l in open(os.path.join(_R,a.manifest)) if l.strip()]

    print(f"{'machine':<9}{'tier':<5}{'grp':<5}{'jobs':>5}{'serial (h)':>12}{'longest job':>14}")
    print("-"*52)
    per=collections.defaultdict(lambda:[0,0.0,0.0])
    for j in jobs:
        d=job_seconds(j,T); k=(j["machine"],j["tier"],j["group"])
        per[k][0]+=1; per[k][1]+=d; per[k][2]=max(per[k][2],d)
    for k in sorted(per):
        n,tot,mx=per[k]
        print(f"{k[0]:<9}{k[1]:<5}{k[2]:<5}{n:>5}{tot/3600:>12.2f}{mx/60:>12.1f}m")
    print("-"*52)
    for m,w,g in (("laptop",a.laptop_workers,a.laptop_gpu_slots),
                  ("a5500",a.a5500_workers,a.a5500_gpu_slots)):
        js=[j for j in jobs if j["machine"]==m]
        if not js: continue
        ser=sum(job_seconds(j,T) for j in js)
        wall=simulate(js,T,w,g)
        t1=[j for j in js if int(j["tier"])==1]
        print(f"{m:<9} {len(js):>3} jobs  serial={ser/3600:5.2f}h  "
              f"scheduled({w}w,{g}gpu)={wall/3600:5.2f}h   tier1 only={simulate(t1,T,w,g)/3600:5.2f}h")
    over=[(j['label'], job_seconds(j,T)/3600) for j in jobs if job_seconds(j,T) > 3*3600]
    print("\nJobs over 3h: " + (", ".join(f"{l} ({h:.1f}h)" for l,h in over) if over else "none"))

if __name__ == "__main__":
    main()
