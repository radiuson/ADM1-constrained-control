"""Learned runs on the test set against the five tunings selected on the
validation set, and the composition of the two scenario sets.

    python3 scripts/independent_eval_deployable.py results_independent
"""
import json, statistics as st, sys, numpy as np
d=sys.argv[1]
S=json.load(open(f'{d}/scenarios.json')); T=S['test']; V=S['validation']
tid=[s['id'] for s in T]
sel=json.load(open(f'{d}/selected.json'))
TC={r['name']:r for r in json.load(open(f'{d}/test_conventional_selected.json'))}
R=json.load(open(f'{d}/test_rl.json'))
ON={'CPO','PCPO','CUP','PPOLag','TRPOLag','TRPOPID','CPPOPID','FOCOPS','OnCRPO','P3O'}
def desc(L):
    m=[s['mult'] for s in L]; ev=sum(s['event'] is not None for s in L)
    ini={k:sum(s['init']==k for s in L) for k in ('steady_state','prestressed','elevated')}
    return f"mult median {st.median(m):.2f} [{min(m):.2f},{max(m):.2f}] >1.8: {sum(x>1.8 for x in m)}  events {ev}  init {ini}"
print('validation', desc(V)); print('test      ', desc(T))
# conventional curve on test from the five validation-selected tunings
pts=[]
for b,v in sel.items():
    p=TC[v['name']]['per_scenario']; n=sum(p[i]['steps'] for i in tid); k=sum(p[i]['viol'] for i in tid)
    pts.append((100*k/n, sum(p[i]['ch4_sum'] for i in tid)/n, b))
pts.sort(); print('selected conventional on test:', [(round(a,2),round(c),b) for a,c,b in pts])
xs=[a for a,_,_ in pts]; ys=[c for _,c,_ in pts]
rows=[]
for r in R:
    p=r['per_scenario']; n=sum(p[i]['steps'] for i in tid); k=sum(p[i]['viol'] for i in tid)
    v=100*k/n; c=sum(p[i]['ch4_sum'] for i in tid)/n
    rows.append(dict(a=r['algorithm'],G1=r['G1'],G2=r['G2'],v=v,c=c,D=c-np.interp(v,xs,ys),inrange=v>=xs[0],
                     cl='on' if r['algorithm'] in ON else 'off'))
for g in ('G1','G2'):
    for cl in ('on','off'):
        H=[x for x in rows if x[g] and x['cl']==cl]
        print(g,cl,len(H),'D vs val-selected conv curve: median %.0f  share D>0 %d/%d  (rate below lowest conv point %d)'%(
            st.median(x['D'] for x in H), sum(x['D']>0 for x in H), len(H), sum(not x['inrange'] for x in H)))
    for a in sorted({x['a'] for x in rows if x[g]}):
        H=[x for x in rows if x[g] and x['a']==a]
        print('   %-8s n=%2d  D med %5.0f  D>0 %2d'%(a,len(H),st.median(x['D'] for x in H),sum(x['D']>0 for x in H)))
# where do violations happen: per-scenario counts
def tag(s): return ('hi' if s['mult']>1.8 else 'lo')+'/'+s['init'][:4]+'/'+('ev' if s['event'] else '--')
print('\nper test scenario: mult init event | conv0 conv1 viol steps | G2-on runs with viol>0 (of 100) | G2-on median viol steps')
G2on=[r for r in R if r['G2'] and r['algorithm'] in ON]
c0=TC[sel['0']['name']]['per_scenario']; c1=TC[sel['1']['name']]['per_scenario']
for s in sorted(T,key=lambda s:s['mult']):
    i=s['id']; vv=[r['per_scenario'][i]['viol'] for r in G2on]
    print(f"{i} {s['mult']:.2f} {s['init'][:5]:5} {str(s['event'] and s['event']['kind'])[:5]:5} start{s['start_day']:4d} | {c0[i]['viol']:3d} {c1[i]['viol']:3d} | {sum(x>0 for x in vv):3d} | {st.median(vv):.0f}")
