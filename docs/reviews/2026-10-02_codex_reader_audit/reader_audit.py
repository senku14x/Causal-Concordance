import json, re, itertools, statistics
from pathlib import Path
from collections import Counter
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'evidence'
SNAP = json.loads((ROOT/'results/001_leverage_profile/analysis/snapshot.json').read_text())
TS = SNAP['tables']
ZH = {'black':'黑','blue':'蓝','brown':'棕褐','gold':'金','gray':'灰','green':'绿','orange':'橙','pink':'粉','purple':'紫','red':'红','teal':'青','white':'白'}
def colours(text):
    out = {x.lower().replace('grey','gray') for x in re.findall(r'\b('+'|'.join(list(ZH)+['grey'])+r')\b',text or '',re.I)}
    out |= {c for c,ch in ZH.items() if any(a in (text or '') for a in ch)}
    return out
def samples(tab,role,cell,layer,reader,arm):
    x=tab['readouts'][f'{role}|cell{cell}|L{layer}|{reader}|{arm}']
    return x['explanations'] if isinstance(x,dict) else x
def rate(xs,c):return np.mean([c in colours(x) for x in xs])
def summarise(vals):
    v=np.array(vals,dtype=float);rng=np.random.default_rng(20261002)
    b=v[rng.integers(0,len(v),(20000,len(v)))].mean(1)
    dist=np.array([np.mean(v*np.array(s)) for s in itertools.product([-1,1],repeat=len(v))])
    return {'mean':round(float(v.mean()),4),'bootstrap95':[round(float(x),4) for x in np.quantile(b,[.025,.975])],
            'positive':int((v>0).sum()),'negative':int((v<0).sum()),'zero':int((v==0).sum()),
            'unadjusted_signflip_p':round(float((np.abs(dist)>=abs(v.mean())-1e-12).mean()),5)}

OUT={'note':'Exploratory reanalysis of existing outputs; table bootstrap; no new model calls; no multiplicity correction.'}
OUT['readers']={}
for reader,L,ci in [('nla',42,0),('nla',42,1),('nla',42,2),('oracle',40,2),('oracle',44,2)]:
    key=f'{reader}|L{L}|cell{ci}'; rr=[]
    for t in TS:
        a=samples(t,'member',ci,L,reader,'real');b=samples(t,'partner',ci,L,reader,'real');sh=samples(t,'member',ci,L,reader,'shuffled')
        sub=samples(t,'substituted',ci,L,reader,'real');z=t['z'];zp=t['partner']['z'];c=t['substitution']['c'];cp=t['substitution']['c2']
        row={'table':t['item'],'z':z,'zp':zp,'real_z':float(rate(a,z)),'shuffled_z':float(rate(sh,z)),
             'member_zprime':float(rate(a,zp)),'partner_z':float(rate(b,z)),'partner_zprime':float(rate(b,zp)),
             'mean_bank_colours':float(np.mean([len(colours(x)) for x in a])),
             'substitution_tracking':float((rate(a,c)-rate(sub,c)+rate(sub,cp)-rate(a,cp))/2)}
        row['real_minus_shuffle']=row['real_z']-row['shuffled_z']
        row['start_pair_tracking']=(row['real_z']-row['partner_z']+row['partner_zprime']-row['member_zprime'])/2
        rr.append(row)
    OUT['readers'][key]={'real_z':round(float(np.mean([r['real_z'] for r in rr])),4),'shuffled_z':round(float(np.mean([r['shuffled_z'] for r in rr])),4),
                         'real_minus_shuffle':summarise([r['real_minus_shuffle'] for r in rr]),
                         'start_pair_tracking':summarise([r['start_pair_tracking'] for r in rr]),
                         'substitution_tracking':summarise([r['substitution_tracking'] for r in rr]),'per_table':rr}

fve=[]
for t in TS:
    ds=[];own_sh=[];positive=[]
    for role in ['member','partner','substituted']:
        for ci in range(3):
            x=t['readouts'][f'{role}|cell{ci}|L42|nla|shuffled'];r=t['readouts'][f'{role}|cell{ci}|L42|nla|real']
            ds += [a-b for a,b in zip(x['fve_vs_source'],x['fve_vs_cell'])]
            own_sh += [a-b for a,b in zip(r['fve_vs_cell'],x['fve_vs_cell'])]
    fve.append({'table':t['item'],'source_minus_wrong':float(np.mean(ds)),'n':len(ds),'prefer_source':sum(x>0 for x in ds),
                'real_minus_shuffled':float(np.mean(own_sh))})
OUT['fve']={'source_minus_wrong':summarise([x['source_minus_wrong'] for x in fve]),
            'prefer_source':sum(x['prefer_source'] for x in fve),'n':sum(x['n'] for x in fve),
            'real_minus_shuffled':summarise([x['real_minus_shuffled'] for x in fve]),'per_table':fve}

# Audit the report's off-pair-mass argument using every saved top-5 entry.
reg=json.loads((ROOT/'scripts/common/registry.json').read_text())
bankp=ROOT/'brew_bank.json'
bank=json.loads(bankp.read_text());bank=bank['items'] if isinstance(bank,dict) else bank;bank={x['name']:x for x in bank}
heads={x['pair_id']:x for x in map(json.loads,(ROOT/'outputs/001_leverage_profile/run/pairs_brew.jsonl').read_text().splitlines()) if x['direction']=='p->m'}
rows=[x for x in map(json.loads,(ROOT/'outputs/001_leverage_profile/run/profile_brew.jsonl').read_text().splitlines()) if x['direction']=='p->m' and x['region']=='emit_colon' and x['layer']==44]
cats=[]
for x in rows:
    h=heads[x['pair_id']];it=bank[h['item']];startp=x['pair_id'].split('__')[1];z=it['intermediates'][0];zp=it['table'][startp]['stir']
    cc=Counter()
    for token,lp in x['top5']:
        col=token.strip().lower();mass=float(np.exp(lp))
        if col in [h['a'],h['a_donor']]:cc['answers']+=mass
        elif col in [z,zp]:cc['intermediates']+=mass
        elif col in ZH:cc['other_colours']+=mass
        else:cc['format_or_other']+=mass
    cc['top5_total']=sum(np.exp(lp) for _,lp in x['top5']);cc['off_pair_mass']=x['off_pair_mass_edit'];cc['flagged']=x['off_pair_mass_edit']-x['off_pair_mass_clean']>.1
    cats.append(dict(cc))
OUT['damage_L44']={'means':{k:float(np.mean([r.get(k,0) for r in cats])) for k in set().union(*(r.keys() for r in cats))},'n':len(cats)}

(BASE/'reader_audit_results.json').write_text(json.dumps(OUT,indent=2))
for k,v in OUT['readers'].items():print(k,json.dumps({a:b for a,b in v.items() if a!='per_table'}))
print('FVE',json.dumps({a:b for a,b in OUT['fve'].items() if a!='per_table'}))
print('DAMAGE',json.dumps(OUT['damage_L44']))
print('NLA_COLON_TABLES',json.dumps(OUT['readers']['nla|L42|cell2']['per_table']))
