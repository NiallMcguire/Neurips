import wandb, numpy as np, json
from collections import defaultdict
api = wandb.Api()
runs = api.runs('reptile_lora_bci')
rows = []
for r in runs:
    if r.state != 'finished':
        continue
    c = r.config; s = r.summary
    zc = s.get('zero_shot_acc')
    if zc is None:
        continue
    rows.append(dict(
        condition=c.get('condition'), dataset=c.get('dataset'),
        seed=c.get('seed'), held_out=c.get('held_out'),
        zs=zc, zk=s.get('zero_shot_kappa'),
        n5=s.get('few_shot_acc_N5'), n10=s.get('few_shot_acc_N10'),
        n20=s.get('few_shot_acc_N20'), n50=s.get('few_shot_acc_N50'),
    ))
print('usable runs:', len(rows))
json.dump(rows, open('/tmp/results.json', 'w'))

g = defaultdict(list)
for r in rows:
    g[(r['dataset'], r['condition'])].append(r)

def m(rs, k):
    v = [r[k] for r in rs if r[k] is not None]
    return float(np.mean(v)) if v else float('nan')

lines = []
hdr = '%-12s %-18s %4s %7s %7s %7s %7s %7s' % ('dataset','condition','n','zero','kappa','N5','N10','N50')
lines.append(hdr); print(hdr)
for (ds, cond), rs in sorted(g.items()):
    row = '%-12s %-18s %4d %7.3f %7.3f %7.3f %7.3f %7.3f' % (
        ds, cond, len(rs), m(rs,'zs'), m(rs,'zk'), m(rs,'n5'), m(rs,'n10'), m(rs,'n50'))
    lines.append(row); print(row)
open('/tmp/summary.txt','w').write('\n'.join(lines))
print('WROTE /tmp/results.json and /tmp/summary.txt')
