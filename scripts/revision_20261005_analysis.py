"""Full-cohort discovery, chronological validation and external assignment."""
from pathlib import Path
import os,sys,json,pickle,hashlib,warnings
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score,accuracy_score,f1_score,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from scipy.optimize import linear_sum_assignment
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0,str(Path(__file__).parent))
from dataset1_firstpass_phenotype_discovery import BG_CHANNELS,summary_stats
from dataset1_dataset2_temporal_transport import SHARED_CHANNELS,REF_TIMES,interpolate_channel
from final_strict_temporal_phenotype_pipeline import laboratory_stress_score
from frozen_discovery_phenotype_transport import set_seed,train_model,predict,normalize_from_train
from ncr_revision_true_temporal_holdout import load_surgery_end

from workspace import WORKSPACE
ROOT=WORKSPACE; OUT=ROOT/'outputs'
DATA=OUT/'analysis';DATA.mkdir(parents=True,exist_ok=True)
threadpool_limits(4);torch.set_num_threads(4);set_seed(42)

STRICT={
 'lactate':{'乳酸'},'ph':{'酸碱度'},'pao2':{'氧分压'},'paco2':{'二氧化碳分压'},
 'hb':{'血色素','血红蛋白浓度'},'hct':{'红细胞压积'},'potassium':{'钾离子','钾'},
 'sodium':{'钠离子','钠'},'calcium':{'钙离子'},'chloride':{'氯离子','氯'},
 'magnesium':{'镁离子'},'glucose':{'葡萄糖'},'be':{'细胞外液碱剩余'},
 'hco3_std':{'标准碳酸氢根','标准碳酸氢根浓度'},
 'hco3_act':{'实际碳酸氢根','实际碳酸氢根浓度'},'so2':{'血氧饱和度'},
}
# Broad analyst-defined unit plausibility checks, not clinical cut-offs.
BOUNDS={'lactate':(0,50),'ph':(6.5,8),'pao2':(1,800),'paco2':(1,200),
 'hb':(20,250),'hct':(1,80),'potassium':(1,15),'sodium':(80,200),
 'calcium':(.1,5),'chloride':(50,180),'magnesium':(.05,5),'glucose':(.3,100),
 'be':(-50,50),'hco3_std':(1,60),'hco3_act':(1,60),'so2':(0,100)}

def feature_cols(df,counts=True):
 allowed={x+'_'+op for x in BG_CHANNELS for op in ['count','first','last','min','max','mean','std','delta','slope']}
 return [c for c in df if c in allowed and (counts or not c.endswith('_count'))]

def fit_discovery(df,name,counts=True,kfix=None,seed=42,n_init=1,columns=None):
 cols=columns if columns is not None else feature_cols(df,counts);imp=SimpleImputer(strategy='median',keep_empty_features=True)
 ximp=imp.fit_transform(df[cols]);sc=StandardScaler();x=sc.fit_transform(ximp)
 p0=PCA().fit(x);d=min(int(np.searchsorted(np.cumsum(p0.explained_variance_ratio_),.95)+1),50)
 pc=PCA(n_components=d);z=pc.fit_transform(x)
 models={k:GaussianMixture(n_components=k,covariance_type='full',random_state=seed,n_init=n_init).fit(z) for k in ([kfix] if kfix else range(1,7))}
 bics=[{'analysis':name,'k':k,'bic':g.bic(z),'aic':g.aic(z),'pca_dim':d,'n':len(df)} for k,g in models.items()]
 k=kfix if kfix else min(models,key=lambda q:models[q].bic(z))
 g=models[k];raw=g.predict(z)
 fullcols=feature_cols(df);stressimp=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(df[fullcols])
 stress=laboratory_stress_score(pd.DataFrame(stressimp,columns=fullcols)).to_numpy()
 avg=pd.DataFrame({'r':raw,'s':stress}).groupby('r').s.mean().sort_values()
 mapping={int(q):int(i) for i,q in enumerate(avg.index)};y=np.array([mapping[int(q)] for q in raw])
 audit={'name':name,'n':len(df),'features':len(cols),'pca_dim':d,'selected_k':k,'bic':float(g.bic(z)),
  'state_n':{str(i):int(sum(y==i)) for i in range(k)},'state_stress':{str(i):float(avg.loc[q]) for q,i in mapping.items()}}
 model={'imputer':imp,'scaler':sc,'pca':pc,'gmm':g,'columns':cols,'mapping':mapping}
 return y,audit,pd.DataFrame(bics),model,z

def teacher_predict(model,df):
 z=model['pca'].transform(model['scaler'].transform(model['imputer'].transform(df[model['columns']])))
 return np.array([model['mapping'][int(q)] for q in model['gmm'].predict(z)])

def clean_inputs():
 old=pd.read_csv(ROOT/'inputs/discovery_features.csv',dtype={'patient_id':str})
 df=old.loc[old.bg_timepoints_0_6h>=8].copy().reset_index(drop=True)
 raw=pd.read_csv(OUT/'audit/fixed1025_raw_lab_records.csv',dtype={'patient_id':str})
 raw['clean_name']=raw.item.str.lstrip('*★☆ ').str.strip()
 raw['accepted_assay']=[name in STRICT[ch] for ch,name in zip(raw.channel,raw.clean_name)]
 raw['plausible']=[BOUNDS[ch][0]<=v<=BOUNDS[ch][1] for ch,v in zip(raw.channel,raw.value)]
 rejected=raw.loc[~raw.accepted_assay].groupby(['channel','item']).agg(records=('value','size'),cases=('patient_id','nunique')).reset_index()
 rejected.to_csv(DATA/'excluded_assays.csv',index=False,encoding='utf-8-sig')
 raw.loc[raw.accepted_assay&~raw.plausible].groupby(['channel','item']).agg(records=('value','size'),cases=('patient_id','nunique'),min=('value','min'),max=('value','max')).reset_index().to_csv(DATA/'range_flags.csv',index=False)
 good=raw.loc[raw.accepted_assay&raw.plausible].copy()
 good=good.groupby(['patient_id','hour','channel'],as_index=False).value.mean()
 df['historical_timepoints']=df.bg_timepoints_0_6h
 counts=good.groupby('patient_id').hour.nunique()
 df.bg_timepoints_0_6h=df.patient_id.map(counts).fillna(0).astype(int)
 groups={(pid,ch):s for (pid,ch),s in good.groupby(['patient_id','channel'])}
 v=np.full((len(df),13,8),np.nan,dtype=np.float32);m=np.zeros_like(v)
 for i,pid in enumerate(df.patient_id):
  for ch in BG_CHANNELS:
   sub=groups.get((pid,ch));t=[] if sub is None else sub.hour.tolist();vv=[] if sub is None else sub.value.tolist()
   for key,value in summary_stats(t,vv).items():df.loc[i,ch+'_'+key]=value
   if ch in SHARED_CHANNELS and len(t):
    j=SHARED_CHANNELS.index(ch);v[i,:,j],m[i,:,j]=interpolate_channel(list(zip(t,vv)))
 # Enforce coherent masks: an observation-near flag cannot indicate NaN values.
 m*=np.isfinite(v).astype(np.float32)
 df=load_surgery_end(df,ROOT/'inputs/operative_records.tsv').sort_values('surgery_end').reset_index(drop=True)
 idx={pid:i for i,pid in enumerate(old.loc[old.bg_timepoints_0_6h>=8].patient_id)}
 reindex=np.array([idx[p] for p in df.patient_id]);v=v[reindex];m=m[reindex]
 df.to_csv(DATA/'corrected_fixed1025_features.csv',index=False)
 np.savez_compressed(DATA/'corrected_local_grid.npz',values=v,masks=m)
 audit={'n':len(df),'raw_records':len(raw),'wrong_assay_records':int((~raw.accepted_assay).sum()),
 'out_of_range_records':int((raw.accepted_assay&~raw.plausible).sum()),'accepted_unique_records':len(good),
 'duplicate_accepted_channel_times':int(raw.loc[raw.accepted_assay&raw.plausible].duplicated(['patient_id','hour','channel']).sum()),
 'corrected_density_ge8_n':int(sum(df.bg_timepoints_0_6h>=8)), 'historical_lactate_max_gt30':int(sum(old.loc[old.bg_timepoints_0_6h>=8].lactate_max>30))}
 return df,v,m,old.loc[old.bg_timepoints_0_6h>=8].reset_index(drop=True),audit

def load_external():
 out={}
 for label,prefix in [('INSPIRE','dataset2'),('MIMIC-IV','dataset3')]:
  folder=ROOT/'inputs'
  d=pd.read_csv(folder/(prefix+'_metadata.csv'));a=np.load(folder/(prefix+'_grid_gate4.npz'))
  v=a['values'].astype(np.float32);m=a['masks'].astype(np.float32);m*=np.isfinite(v).astype(np.float32)
  # Identical broad bounds on common channels across sites.
  range_rows=[]
  for j,ch in enumerate(SHARED_CHANNELS):
   bad=np.isfinite(v[:,:,j])&((v[:,:,j]<BOUNDS[ch][0])|(v[:,:,j]>BOUNDS[ch][1]))
   range_rows.append({'cohort':label,'channel':ch,'removed_grid_values':int(bad.sum())})
   v[:,:,j][bad]=np.nan;m[:,:,j][bad]=0
  pd.DataFrame(range_rows).to_csv(DATA/(prefix+'_grid_range_flags.csv'),index=False)
  out[label]=(d,v,m)
 return out

def assigned(d,y,p):
 d=d.copy();d['phenotype']=y;d['assignment_confidence']=p.max(1)
 for i in range(p.shape[1]):d['p_s'+str(i)]=p[:,i]
 return d

def summaries(cohorts):
 rows=[]
 for name,d in cohorts.items():
  for s,g in d.groupby('phenotype'):
   n=len(g);e=int(g.label72.sum());lo,hi=proportion_confint(e,n,alpha=.05,method='wilson')
   rows.append({'cohort':name,'state':int(s),'n':n,'events':e,'rate':e/n,'lo':lo,'hi':hi,
    'density_median':g.bg_timepoints_0_6h.median(),'icu_median':g.first_icu_hours.median(),
    'icu_q25':g.first_icu_hours.quantile(.25),'icu_q75':g.first_icu_hours.quantile(.75),
    'confidence_median':g.assignment_confidence.median() if 'assignment_confidence' in g else np.nan})
 return pd.DataFrame(rows)

def association_models(cohorts,grids,k):
 rows=[];missing=[];comparators=[]
 for name,d0 in cohorts.items():
  d=d0.copy()
  if name in grids:
   v,m=grids[name];j=SHARED_CHANNELS.index('lactate')
   vv=np.where((m[:,:,j]>0)&np.isfinite(v[:,:,j]),v[:,:,j],np.nan)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');d['lactate_peak']=np.nanmax(vv,axis=1)
  else:d['lactate_peak']=d['lactate_max']
  missing.append({'cohort':name,'n':len(d),'missing_lactate':int(d.lactate_peak.isna().sum()),
   'missing_density':int(d.bg_timepoints_0_6h.isna().sum()),'events':int(d.label72.sum())})
  for subset,base in [('all',d),('ge8',d[d.bg_timepoints_0_6h>=8])]:
   c=base.dropna(subset=['label72','lactate_peak','bg_timepoints_0_6h','phenotype']).copy()
   if not len(c):continue
   dummy=pd.DataFrame({f'S{s}':(c.phenotype==s).astype(float) for s in range(1,k)},index=c.index)
   variants={'crude':dummy,'plus_lactate':pd.concat([dummy,c[['lactate_peak']]],axis=1),
    'plus_density':pd.concat([dummy,c[['bg_timepoints_0_6h']]],axis=1),
    'plus_both':pd.concat([dummy,c[['lactate_peak','bg_timepoints_0_6h']]],axis=1)}
   for label,x in variants.items():
    try:
     glm=sm.GLM(c.label72.astype(float),sm.add_constant(x,has_constant='add'),family=sm.families.Binomial())
     fit=glm.fit(cov_type='cluster',cov_kwds={'groups':c.subject_id}) if 'subject_id' in c else glm.fit()
     ci=fit.conf_int()
     for s in range(1,k):
      key=f'S{s}'
      if not sum(c.phenotype==s):continue
      rows.append({'cohort':name,'subset':subset,'model':label,'state':s,'n':len(c),'events':int(c.label72.sum()),
       'or':float(np.exp(fit.params[key])),'lo':float(np.exp(ci.loc[key,0])),'hi':float(np.exp(ci.loc[key,1])),
       'p':fit.pvalues[key]})
    except Exception as ex:print('Model issue',name,subset,label,str(ex),flush=True)
   # Within-site apparent AUC only; does not estimate deployment performance.
   for label,x in [('phenotype',dummy),('lactate',c[['lactate_peak']]),('density',c[['bg_timepoints_0_6h']]),('lactate_density',c[['lactate_peak','bg_timepoints_0_6h']]),('combined',variants['plus_both'])]:
    fit=sm.GLM(c.label72.astype(float),sm.add_constant(x,has_constant='add'),family=sm.families.Binomial()).fit()
    p=fit.predict(sm.add_constant(x,has_constant='add'));auc=roc_auc_score(c.label72,p)
    rng=np.random.default_rng(42);vals=[];yy=c.label72.to_numpy();pp=np.asarray(p)
    for b in range(300):
     ii=rng.integers(0,len(c),len(c))
     if len(np.unique(yy[ii]))==2:vals.append(roc_auc_score(yy[ii],pp[ii]))
    comparators.append({'cohort':name,'subset':subset,'model':label,'n':len(c),'auc':auc,'lo':np.quantile(vals,.025),'hi':np.quantile(vals,.975)})
  d.to_csv(DATA/(name.replace('-','_')+'_analysis_cases.csv'),index=False)
 return pd.DataFrame(rows),pd.DataFrame(missing),pd.DataFrame(comparators)

def main():
 df,v,m,old,qa=clean_inputs();print('SOURCE AUDIT',qa,flush=True)
 y,audit,bics,teacher,z=fit_discovery(df,'corrected_full1025')
 k=audit['selected_k'];print('PRIMARY',audit,flush=True)
 audits=[audit];allbic=[bics];sens=[]
 oldy,olda,oldbic,oldmodel,_=fit_discovery(old,'historical_full1025')
 oldmap=dict(zip(old.patient_id,oldy));oldaligned=df.patient_id.map(oldmap).to_numpy()
 qa['corrected_vs_historical_ari']=float(adjusted_rand_score(y,oldaligned));qa['historical_selected_k']=olda['selected_k']
 pd.crosstab(pd.Series(oldaligned,name='historical_state'),pd.Series(y,name='corrected_state')).to_csv(DATA/'historical_corrected_transition.csv')
 audits.append(olda);allbic.append(oldbic)
 for name,subset,counts,ninit in [('count_free',df,False,1),('corrected_density_ge8',df[df.bg_timepoints_0_6h>=8],True,1),('multi_initialization',df,True,10),('static_mean_only',df,True,1),('dynamics_only',df,True,1)]:
  columns=[c for c in feature_cols(subset) if c.endswith('_mean')] if name=='static_mean_only' else ([c for c in feature_cols(subset) if c.endswith(('_std','_delta','_slope'))] if name=='dynamics_only' else None)
  yy,aa,bb,mm,_=fit_discovery(subset,name,counts,n_init=ninit,columns=columns)
  reference=pd.Series(y,index=df.patient_id).loc[subset.patient_id].to_numpy()
  sens.append({'analysis':name,'n':len(subset),'selected_k':aa['selected_k'],'ari_vs_primary':adjusted_rand_score(reference,yy),'features':aa['features']})
  audits.append(aa);allbic.append(bb)
 pd.concat(allbic).to_csv(DATA/'k_selection.csv',index=False);pd.DataFrame(sens).to_csv(DATA/'representation_sensitivity.csv',index=False)
 # Repeatability with alternative random initializations (fixed representation).
 for seed in range(10):
  yy=GaussianMixture(n_components=k,covariance_type='full',random_state=seed).fit_predict(z)
  sens.append({'analysis':'seed_'+str(seed),'n':len(df),'selected_k':k,'ari_vs_primary':adjusted_rand_score(y,yy),'features':audit['features']})
 pd.DataFrame(sens).to_csv(DATA/'representation_sensitivity.csv',index=False)
 # Bootstrap evaluate ALL original observations; fixed K, full preprocessing refit.
 rng=np.random.default_rng(42);boots=[]
 for b in range(60):
  ii=rng.integers(0,len(df),len(df))
  # Conditional stability and end-to-end representation stability are distinct.
  g=GaussianMixture(n_components=k,covariance_type='full',random_state=42).fit(z[ii]);yy=g.predict(z)
  boots.append({'replicate':b,'type':'fixed_representation','ari':adjusted_rand_score(y,yy)})
  _,aa,bb,mm,_=fit_discovery(df.iloc[ii].reset_index(drop=True),'bootstrap',kfix=k)
  yy=teacher_predict(mm,df);boots.append({'replicate':b,'type':'refit_preprocessing','ari':adjusted_rand_score(y,yy)})
  if b%10==0:print('Bootstrap',b,flush=True)
 pd.DataFrame(boots).to_csv(DATA/'bootstrap_stability.csv',index=False)
 df['phenotype']=y;df.to_csv(DATA/'discovery_cases.csv',index=False)
 with (DATA/'full_discovery_model.pkl').open('wb') as f:pickle.dump(teacher,f)
 external=load_external();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model,hist=train_model(v,m,y,batch_size=256,epochs=40,lr=.001,alpha=2,beta=.2,device=device)
 pd.DataFrame(hist).to_csv(DATA/'full_training_history.csv',index=False)
 means,stds=normalize_from_train(v,m,v)[1:]
 torch.save({'state_dict':model.state_dict(),'means':means,'stds':stds,'channels':SHARED_CHANNELS,'times':REF_TIMES,'num_classes':k,'seed':42,'training_n':len(df)},DATA/'full_assignment_checkpoint.pt')
 pred,prob=predict(model,v,m,v,m,256,device);dfassign=assigned(df,pred,prob)
 fidelity=[{'analysis':'full_training_fit','n':len(y),'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,average='macro'),'ari':adjusted_rand_score(y,pred)}]
 pd.DataFrame(confusion_matrix(y,pred,labels=range(k))).to_csv(DATA/'full_fit_confusion.csv',index=False)
 cohorts={'Discovery':df};grids={'Discovery':(v,m)};assignedcohorts={'Discovery assigned':dfassign}
 extpred={}
 for name,(d,vv,mm) in external.items():
  pp,pr=predict(model,v,m,vv,mm,256,device);d=assigned(d,pp,pr)
  d.to_csv(DATA/(name.replace('-','_')+'_assigned.csv'),index=False);cohorts[name]=d;grids[name]=(vv,mm);extpred[name]=pp
 # Separate chronological early-only pipeline; K can differ from full model.
 early=df.iloc[:820].copy();late=df.iloc[820:].copy()
 ey,ea,eb,et,ez=fit_discovery(early,'early_only820');audits.append(ea);allbic.append(eb)
 early_labels=early[['patient_id','surgery_end']].copy();early_labels['phenotype']=ey;early_labels.to_csv(DATA/'early_only820_teacher_labels.csv',index=False)
 with (DATA/'early_only820_teacher.pkl').open('wb') as f:pickle.dump(et,f)
 ly=teacher_predict(et,late)
 em,eh=train_model(v[:820],m[:820],ey,256,40,.001,2,.2,device)
 pd.DataFrame(eh).to_csv(DATA/'early_only820_training_history.csv',index=False)
 emeans,estds=normalize_from_train(v[:820],m[:820],v[:820])[1:]
 torch.save({'state_dict':em.state_dict(),'means':emeans,'stds':estds,'channels':SHARED_CHANNELS,'times':REF_TIMES,'num_classes':ea['selected_k'],'seed':42,'training_n':820},DATA/'early_only820_assignment_checkpoint.pt')
 ep,epr=predict(em,v[:820],m[:820],v[820:],m[820:],256,device)
 temp=assigned(late,ep,epr);temp['early_teacher_state']=ly;temp.to_csv(DATA/'independent_temporal205.csv',index=False)
 fidelity.append({'analysis':'independent_temporal_test','n':205,'accuracy':accuracy_score(ly,ep),'macro_f1':f1_score(ly,ep,average='macro'),'ari':adjusted_rand_score(ly,ep)})
 pd.DataFrame(confusion_matrix(ly,ep,labels=range(ea['selected_k']))).to_csv(DATA/'temporal_confusion.csv',index=False)
 # Simple assignment comparators trained only on early teacher labels.
 def flat(vv,mm):
  norm=normalize_from_train(v[:820],m[:820],vv)[0]
  return np.concatenate([np.nan_to_num(norm).reshape(len(vv),-1),mm.reshape(len(vv),-1)],axis=1)
 xt=flat(v[:820],m[:820]);xe=flat(v[820:],m[820:])
 for name,clf in [('linear_values_masks',LogisticRegression(max_iter=2000,class_weight='balanced',C=1)),('boosted_values_masks',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,random_state=42)),('linear_mask_only',LogisticRegression(max_iter=2000,class_weight='balanced',C=1))]:
  xx=xt[:,104:] if name=='linear_mask_only' else xt;xxe=xe[:,104:] if name=='linear_mask_only' else xe
  clf.fit(xx,ey);yy=clf.predict(xxe)
  fidelity.append({'analysis':name,'n':205,'accuracy':accuracy_score(ly,yy),'macro_f1':f1_score(ly,yy,average='macro'),'ari':adjusted_rand_score(ly,yy)})
 pd.DataFrame(fidelity).to_csv(DATA/'assignment_fidelity.csv',index=False)
 # Drop-channel and random-entry robustness relative to each site's original assignment.
 robustness=[];rg=np.random.default_rng(42)
 for name,(d,vv,mm) in external.items():
  for ch in SHARED_CHANNELS:
   j=SHARED_CHANNELS.index(ch);va=vv.copy();ma=mm.copy();va[:,:,j]=np.nan;ma[:,:,j]=0
   yy,_=predict(model,v,m,va,ma,256,device)
   robustness.append({'cohort':name,'perturbation':'omit_'+ch,'agreement':accuracy_score(extpred[name],yy),'ari':adjusted_rand_score(extpred[name],yy),'n':len(d)})
  for keep in [.75,.5,.25]:
   # Observed-entry ablation; this is synthetic sparsity, not clinical validation.
   ma=mm*(rg.random(mm.shape)<keep);va=np.where(ma>0,vv,np.nan)
   yy,_=predict(model,v,m,va,ma,256,device)
   robustness.append({'cohort':name,'perturbation':'retain_'+str(keep),'agreement':accuracy_score(extpred[name],yy),'ari':adjusted_rand_score(extpred[name],yy),'n':len(d)})
 pd.DataFrame(robustness).to_csv(DATA/'assignment_robustness.csv',index=False)
 # All primary external regressions share the same complete cases within each subset.
 summary=summaries(cohorts);summary.to_csv(DATA/'state_summary.csv',index=False)
 ordf,missing,aucs=association_models(cohorts,grids,k)
 ordf.to_csv(DATA/'association_models.csv',index=False);missing.to_csv(DATA/'missingness.csv',index=False);aucs.to_csv(DATA/'apparent_auc.csv',index=False)
 tord,tm,ta=association_models({'Independent temporal':temp},{'Independent temporal':(v[820:],m[820:])},ea['selected_k'])
 tord.to_csv(DATA/'temporal_association_models.csv',index=False);summaries({'Independent temporal':temp}).to_csv(DATA/'temporal_state_summary.csv',index=False)
 # Clinical secondary endpoints preserve unknowns as missing rather than zero.
 rawop=pd.read_csv(ROOT/'inputs/operative_records.tsv',sep='\t',encoding='gb18030',dtype={'住院号':str},usecols=['住院号','手术结束时间','入院日期','出院日期'])
 rawop=rawop.rename(columns={'住院号':'patient_id'});rawop['surgery_end_raw']=pd.to_datetime(rawop['手术结束时间'],errors='coerce')
 rawop=rawop.sort_values(['patient_id','surgery_end_raw']).drop_duplicates('patient_id')
 rawop['hospital_days']=(pd.to_datetime(rawop['出院日期'],errors='coerce')-pd.to_datetime(rawop['入院日期'],errors='coerce')).dt.total_seconds()/86400
 sec=df.merge(rawop[['patient_id','hospital_days']],on='patient_id',how='left')
 sec.loc[sec.hospital_days<0,'hospital_days']=np.nan
 sec['mortality']=np.where(sec.outcome.eq('死亡'),1,np.where(sec.outcome.isin(['好转','治愈','未愈','其他']),0,np.nan))
 sec.to_csv(DATA/'secondary_endpoint_cases.csv',index=False)
 print('Outcome categories (aggregate)',sec.outcome.value_counts(dropna=False).to_dict(),flush=True)
 pd.concat(allbic).to_csv(DATA/'k_selection.csv',index=False)
 manifest={'created':'2026-10-05','design':'fixed full1025 corrected laboratory discovery; separate early820/late205 sensitivity',
 'source_audit':qa,'discovery':audit,'analyses':audits,'assignment':fidelity,'software':{'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__},
 'external_n':{name:len(d) for name,(d,_,_) in external.items()},'epochs':40,'bootstrap_replicates':60}
 (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
 print('DONE',json.dumps(manifest,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
