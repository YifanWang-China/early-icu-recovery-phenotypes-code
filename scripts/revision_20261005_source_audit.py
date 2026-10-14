"""Read-only audit of historical assay matching in the fixed Fig. 1 cohort."""
from pathlib import Path
import sys,json,csv,time
from collections import defaultdict,Counter
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from dataset1_firstpass_phenotype_discovery import robust_map_bg_channel
from ncr_revision_true_temporal_holdout import load_surgery_end

from workspace import WORKSPACE
ROOT=WORKSPACE
OUT=ROOT/'outputs/audit'
OUT.mkdir(parents=True,exist_ok=True)
features=pd.read_csv(ROOT/'inputs/discovery_features.csv',dtype={'patient_id':str})
dense=features.loc[features.bg_timepoints_0_6h>=8].copy()
dates=load_surgery_end(dense[['patient_id']],ROOT/'inputs/operative_records.tsv')
dates.patient_id=dates.patient_id.astype(str)
starts=dense[['patient_id']].merge(dates,on='patient_id')
starts={str(r.patient_id):pd.Timestamp(r.surgery_end) for r in starts.itertuples()}
counts=Counter(); stats=defaultdict(list); records=[]; names=Counter(); scanned=0; clock=time.time()
path=ROOT/'inputs/laboratory_records.csv'
with path.open(encoding='gb18030',newline='') as f:
 reader=csv.reader(f); header=next(reader)
 print('Columns:',repr(header),flush=True)
 for row in reader:
  scanned+=1
  if len(row)<7: continue
  pid=row[1].strip()
  if pid not in starts: continue
  item=row[5]; channel=robust_map_bg_channel(item)
  if channel is None: continue
  names[(channel,item)]+=1
  try:
   t=pd.Timestamp(row[4]); hour=(t-starts[pid]).total_seconds()/3600
   value=float(row[6])
  except (ValueError,TypeError):continue
  if not(0<=hour<=6) or not np.isfinite(value):continue
  counts[(channel,item)]+=1;stats[(channel,item)].append(value)
  records.append((pid,hour,channel,item,value))
  if scanned%2000000==0: print('Scanned',scanned,'elapsed',round(time.time()-clock),flush=True)
rows=[]
for (channel,item),n in counts.items():
 v=np.array(stats[(channel,item)])
 rows.append({'channel':channel,'item':item,'n':n,'min':v.min(),'q01':np.quantile(v,.01),'median':np.median(v),'q99':np.quantile(v,.99),'max':v.max()})
audit=pd.DataFrame(rows).sort_values(['channel','n'],ascending=[True,False])
audit.to_csv(OUT/'assay_name_audit.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(records,columns=['patient_id','hour','channel','item','value']).to_csv(OUT/'fixed1025_raw_lab_records.csv',index=False)
print(audit.to_string(index=False),flush=True)
print(json.dumps({'scanned':scanned,'records':len(records),'elapsed_seconds':time.time()-clock}),flush=True)
