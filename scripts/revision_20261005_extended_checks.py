"""Clinical-context, endpoint, eligibility, and source-anchor checks."""
from pathlib import Path
import sys,json,pickle,re
import numpy as np,pandas as pd,statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0,str(Path(__file__).parent))
from revision_20261005_analysis import fit_discovery,teacher_predict
from dataset1_firstpass_phenotype_discovery import ICU_INTERVAL_RE
from workspace import WORKSPACE
R=WORKSPACE;D=R/'outputs/analysis'
d=pd.read_csv(D/'Discovery_analysis_cases.csv',dtype={'patient_id':str})
def group(text):
 # Match the organ explicitly: coronary bypass grafting also contains 移植.
 if '心脏移植' in text:return 'Heart transplantation'
 cabg='冠状动脉旁路' in text
 valve=bool(re.search('瓣.*(置换|成形|修复)',text))
 if cabg and valve:return 'CABG + valve'
 if cabg:return 'CABG'
 if valve:return 'Valve'
 if any(x in text for x in ['房间隔','室间隔','法洛','动脉导管','肺静脉异位','先天']):return 'Congenital'
 if '主动脉' in text:return 'Aortic'
 return 'Other cardiac'
d['procedure_group']=d.surgery_name.fillna('').map(group)
pd.crosstab(d.procedure_group,d.phenotype).to_csv(D/'procedure_group_table.csv')
rows=[]
for model,c in [('Procedure context',d.dropna(subset=['lactate_peak','bg_timepoints_0_6h'])),('Clinical complete cases',d.dropna(subset=['lactate_peak','bg_timepoints_0_6h','ef','nyha']))]:
 base=pd.DataFrame({f'S{s}':(c.phenotype==s).astype(float) for s in [1,2]},index=c.index)
 proc=pd.get_dummies(c.procedure_group,drop_first=True,dtype=float)
 variants={'crude':base,'lactate_density':pd.concat([base,c[['lactate_peak','bg_timepoints_0_6h']]],axis=1),'plus_procedure':pd.concat([base,c[['lactate_peak','bg_timepoints_0_6h']],proc],axis=1)}
 if model=='Clinical complete cases':variants['plus_ef_nyha']=pd.concat([variants['plus_procedure'],c[['ef','nyha']]],axis=1)
 for label,x in variants.items():
  fit=sm.GLM(c.label72,sm.add_constant(x),family=sm.families.Binomial()).fit();ci=fit.conf_int()
  for state in [1,2]:
   key=f'S{state}';rows.append({'sample':model,'model':label,'state':state,'n':len(c),'events':int(c.label72.sum()),'or':np.exp(fit.params[key]),'lo':np.exp(ci.loc[key,0]),'hi':np.exp(ci.loc[key,1])})
pd.DataFrame(rows).to_csv(D/'clinical_context_models.csv',index=False)
if '--clinical-context-only' in sys.argv:
 print(pd.crosstab(d.procedure_group,d.phenotype).to_string())
 print(pd.DataFrame(rows).query('state == 2').to_string(index=False))
 raise SystemExit(0)
case=pd.read_csv(D/'secondary_endpoint_cases.csv');rows=[]
for state,g in case.groupby('phenotype'):
 row={'state':state,'n':len(g),'hospital_n':int(g.hospital_days.notna().sum()),'hospital_median':g.hospital_days.median(),'hospital_q25':g.hospital_days.quantile(.25),'hospital_q75':g.hospital_days.quantile(.75),'ef_n':int(g.ef.notna().sum()),'ef_median':g.ef.median(),'nyha_n':int(g.nyha.notna().sum()),'nyha_median':g.nyha.median()}
 for name,y in [('death',g.mortality),('hospital_ge14',g.hospital_days.ge(14).where(g.hospital_days.notna()))]:
  n=int(y.notna().sum());e=int(y.sum());lo,hi=proportion_confint(e,n,method='wilson');row.update({name+'_n':n,name+'_events':e,name+'_rate':e/n,name+'_lo':lo,name+'_hi':hi})
 rows.append(row)
pd.DataFrame(rows).to_csv(D/'secondary_summary.csv',index=False)
coh={'Discovery':d,'INSPIRE':pd.read_csv(D/'INSPIRE_analysis_cases.csv'),'MIMIC-IV':pd.read_csv(D/'MIMIC_IV_analysis_cases.csv')}
land=[];thresholds=[];utility=[];missing=[];subgroup=[]
def or_models(name,df,label,out):
 c=df.dropna(subset=['lactate_peak','bg_timepoints_0_6h']);base=pd.DataFrame({f'S{s}':(c.phenotype==s).astype(float) for s in [1,2]},index=c.index)
 for tag,x in [('crude',base),('adjusted',pd.concat([base,c[['lactate_peak','bg_timepoints_0_6h']]],axis=1))]:
  glm=sm.GLM(c.label72,sm.add_constant(x),family=sm.families.Binomial());fit=glm.fit(cov_type='cluster',cov_kwds={'groups':c.subject_id}) if 'subject_id' in c else glm.fit();ci=fit.conf_int()
  for state in [1,2]:
   key=f'S{state}';out.append({'cohort':name,'subset':label,'model':tag,'state':state,'n':len(c),'events':int(c.label72.sum()),'or':np.exp(fit.params[key]),'lo':np.exp(ci.loc[key,0]),'hi':np.exp(ci.loc[key,1])})
for name,a in coh.items():
 or_models(name,a[a.first_icu_hours>=6],'ICU duration >=6 h',land)
 for cutoff in [48,96,120]:
  b=a.copy();b.label72=(b.first_icu_hours>=cutoff).astype(int);or_models(name,b,f'ICU >={cutoff} h',thresholds)
 y=a.label72.to_numpy().astype(bool);pos=(a.phenotype==2).to_numpy();tp=int(sum(y&pos));fp=int(sum(~y&pos));tn=int(sum(~y&~pos));fn=int(sum(y&~pos));utility.append({'cohort':name,'n':len(a),'events':int(sum(y)),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp),'npv':tn/(tn+fn)})
 for tag,b in [('lactate available',a[a.lactate_peak.notna()]),('lactate missing',a[a.lactate_peak.isna()])]:missing.append({'cohort':name,'group':tag,'n':len(b),'events':int(b.label72.sum()),'rate':b.label72.mean(),'S2_pct':100*(b.phenotype==2).mean(),'density_median':b.bg_timepoints_0_6h.median()})
 if name=='MIMIC-IV':
  for flag in ['service_aligned','procdate_aligned']:
   b=a[a[flag]==1];or_models(name,b,flag,subgroup)
pd.DataFrame(land).to_csv(D/'six_hour_landmark_models.csv',index=False);pd.DataFrame(thresholds).to_csv(D/'alternative_threshold_models.csv',index=False);pd.DataFrame(utility).to_csv(D/'S2_screening_properties.csv',index=False);pd.DataFrame(missing).to_csv(D/'lactate_missingness_comparison.csv',index=False);pd.DataFrame(subgroup).to_csv(D/'MIMIC_definition_subgroups.csv',index=False)
# Time-anchor audit uses only aggregate offsets; do not export free-text records.
raw=pd.read_csv(R/'inputs/operative_records.tsv',sep='\t',encoding='gb18030',usecols=['住院号','手术结束时间','重症监护室相关记录信息'],dtype={'住院号':str})
raw['end']=pd.to_datetime(raw['手术结束时间'],errors='coerce');raw=raw.sort_values(['住院号','end']).drop_duplicates('住院号');offset=[];anchorrows=[]
for r in raw.itertuples(index=False,name=None):
 if r[0] not in set(d.patient_id):continue
 matches=ICU_INTERVAL_RE.findall(str(r[2]));start=pd.to_datetime(matches[0],format='%y-%m-%d %H:%M') if matches else pd.NaT
 offset.append((start-pd.to_datetime(r[1])).total_seconds()/3600)
 linked=[]
 for j in range(0,len(matches)-1,2):
  si=pd.to_datetime(matches[j],format='%y-%m-%d %H:%M');so=pd.to_datetime(matches[j+1],format='%y-%m-%d %H:%M');end=pd.to_datetime(r[1])
  if so>=end and si<=end+pd.Timedelta(hours=6):linked.append((si,so))
 hours=(linked[0][1]-max(linked[0][0],pd.to_datetime(r[1]))).total_seconds()/3600 if linked else np.nan
 anchorrows.append({'patient_id':r[0],'first_icu_offset_h':offset[-1],'surgery_linked_postop_icu_h':hours,'linked_icu_found':bool(linked)})
a=np.array(offset);np.savetxt(D/'local_anchor_offset_quantiles.csv',np.array([[len(a),np.sum(a<0),np.sum(a>6),*np.nanquantile(a,[0,.01,.25,.5,.75,.99,1])]]),delimiter=',',header='n,icu_before_surgery_end,icu_after_surgery_end6h,min,q01,q25,median,q75,q99,max',comments='')
joined=d.merge(pd.DataFrame(anchorrows),on='patient_id',how='left');joined.to_csv(D/'local_anchor_audited_cases.csv',index=False)
anc=[];or_models('Discovery',joined[joined.first_icu_offset_h.between(-.25,6)],'First ICU entry aligned -0.25 to +6 h',anc)
alt=joined[joined.surgery_linked_postop_icu_h.notna()].copy();alt.label72=(alt.surgery_linked_postop_icu_h>=72).astype(int);or_models('Discovery',alt,'Surgery-linked postoperative duration',anc);pd.DataFrame(anc).to_csv(D/'local_episode_alignment_models.csv',index=False)
print('Episode audit',{'aligned':int(joined.first_icu_offset_h.between(-.25,6).sum()),'linked_found':int(joined.linked_icu_found.sum()),'changed72':int(sum(alt.label72!=alt.first_icu_hours.ge(72))), 'linked_cases_events':int(alt.label72.sum())},flush=True)
print('Aligned models',pd.DataFrame(anc).to_string(index=False),flush=True)
# BIC can reward tiny components: make this explicit in the corrected gate sensitivity.
gate=d[d.bg_timepoints_0_6h>=8].copy();_,aa,bb,mm,z=fit_discovery(gate,'gate_min_component_audit');rows=[]
for k in range(1,7):
 g=GaussianMixture(n_components=k,covariance_type='full',random_state=42).fit(z);yy=g.predict(z);counts=np.bincount(yy,minlength=k);rows.append({'k':k,'bic':g.bic(z),'minimum_component_n':int(counts.min()),'eligible_min_1pct':bool(counts.min()>=np.ceil(.01*len(gate)))})
pd.DataFrame(rows).to_csv(D/'gate_component_sizes.csv',index=False)
print('Context models',pd.read_csv(D/'clinical_context_models.csv').to_string(index=False));print('Anchor offsets:',pd.read_csv(D/'local_anchor_offset_quantiles.csv').to_string(index=False));print('Landmark:',pd.DataFrame(land).to_string(index=False));print('Gate components',pd.DataFrame(rows).to_string(index=False))
