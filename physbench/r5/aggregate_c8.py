import hashlib,json,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/R5/C8'
OUT.mkdir(parents=True,exist_ok=True)
MODEL=ROOT/'model-output/evidence/R5/C8'
BASE=ROOT/'baseline_r4/evidence/R4/M1'
BASE_PIP=ROOT/'baseline_r4/evidence_R4_M1_PIP_FREEZE.txt'
BASE_R3=ROOT/'baseline_r3/evidence/R3/M1/ctifl_event_responses.csv'
PREP=ROOT/'prepared/PRIMARY_97_R2_INPUT.csv'
MODEL_RESP=MODEL/'c8_m1_responses.csv'
MODEL_AUDIT=MODEL/'c8_m1_runtime_audit.csv'
MODEL_SUMMARY=MODEL/'c8_m1_summary.json'
MODEL_PIP=ROOT/'model-output/evidence_R5_M1_PIP_FREEZE.txt'
REF=BASE/'ctifl_common_flux_responses.csv'
REF_AUDIT=BASE/'runtime_audit.csv'
REF_SUMMARY=BASE/'summary.json'
RUN_ID=os.environ.get('GITHUB_RUN_ID','unknown')
REPO=os.environ.get('GITHUB_REPOSITORY','unknown')
SHA=os.environ.get('GITHUB_SHA','unknown')
PREPARE_RESULT=os.environ.get('PREPARE_RESULT','unknown')
C8_RESULT=os.environ.get('C8_RESULT','unknown')
RUN_URL=os.environ.get('GITHUB_SERVER_URL','https://github.com')+'/'+REPO+'/actions/runs/'+RUN_ID
EXPECTED={
 'r4_cf0_responses':'968812fe9d26c30ed452fa1a4430f1d73f4bc9cae777033b326aef437aeb719f',
 'r4_runtime_audit':'8c37b35eebba096172a0824843058b0d9ed8ff39dbd7b3848d0e9841d1e1060c',
 'r4_summary':'884b8bc49849d2d29f9e313d77783f38e92c6fe8919e27e72c196c35fcc40c2',
 'r4_pip_freeze':'2fb80e3b624877760690513dcc6f0b12fd8ecff3f945b1a3ec1d69e420ad9013',
 'r3_hv_responses':'f102765b460b1646e82017a6f6c9761f4b072b7eab027ce6b23238557c6a8ffe',
 'prepared_input':'0127756518d8a481d8f79e27b0a0d57bd2b9e6d9d94232202109869f5a792d5b',
 'field_flux':'b669a1209cc0e641b567bd5dbbc2f23c103ac3efbd91998a54f8a8a4939939cd',
 'field_json':'4c7ae6aaab48097e522bb305b8353bae23ac59ffb0775508c077d6d1b130fc3b',
 'event_list_v2':'5721220e0ebb13750ae7b89950f7998e80836d3c52d0c8ee9309380ef82167b5'
}
checks=[]
failures=[]
def add(name,ok,detail):
    rec={'check':name,'passed':bool(ok),'detail':str(detail)}
    checks.append(rec)
    if not ok: failures.append(name)
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_file_hash(name,p,key):
    if not p.is_file():
        add(name,False,'missing: '+str(p)); return None
    got=sha(p); ok=got==EXPECTED[key]
    add(name,ok,'sha256='+got)
    return got
def load_json(p,name):
    if not p.is_file():
        add(name,False,'missing: '+str(p)); return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        add(name,False,'invalid JSON: '+repr(exc)); return {}
for src,name in [
 (ROOT/'precheck-output/evidence/R5/C8/field_flux_unit_tests.log','C8_field_flux_unit_tests.log'),
 (ROOT/'precheck-output/evidence/R5/C8/protocol_input_checks.log','C8_protocol_input_checks.log'),
 (MODEL/'c8_model_console.log','C8_model_console.log')
]:
    if src.is_file(): shutil.copy2(src,OUT/name)
for src in MODEL.iterdir() if MODEL.is_dir() else []:
    if src.is_file(): shutil.copy2(src,OUT/src.name)
if MODEL_PIP.is_file(): shutil.copy2(MODEL_PIP,ROOT/'evidence_R5_M1_PIP_FREEZE.txt')
add('prepare_job_succeeded',PREPARE_RESULT=='success',PREPARE_RESULT)
add('C8_model_job_succeeded',C8_RESULT=='success',C8_RESULT)
check_file_hash('frozen_R4_CF0_response_file',REF,'r4_cf0_responses')
check_file_hash('frozen_R4_runtime_audit_file',REF_AUDIT,'r4_runtime_audit')
check_file_hash('frozen_R4_summary_file',REF_SUMMARY,'r4_summary')
check_file_hash('frozen_R4_pip_freeze_file',BASE_PIP,'r4_pip_freeze')
check_file_hash('frozen_R3_HV_reference_file',BASE_R3,'r3_hv_responses')
check_file_hash('frozen_R4_prepared_input_file',PREP,'prepared_input')
check_file_hash('field_flux_reference_implementation',ROOT/'physbench/r5/field_flux.py','field_flux')
check_file_hash('field_ventilation_function_parameters',ROOT/'physbench/r5/field_ventilation_function.json','field_json')
check_file_hash('R5_event_list_v2',ROOT/'physbench/r5/R5_event_list_97_v2.csv','event_list_v2')
field_test=ROOT/'precheck-output/evidence/R5/C8/field_flux_unit_tests.log'
input_test=ROOT/'precheck-output/evidence/R5/C8/protocol_input_checks.log'
add('field_flux_reference_tests',field_test.is_file() and 'field_flux unit tests PASS' in field_test.read_text(),
    'log present='+str(field_test.is_file()))
add('frozen_input_join_and_40_event_tests',input_test.is_file() and 'protocol_input_gate' in input_test.read_text() and 'PASS' in input_test.read_text(),
    'log present='+str(input_test.is_file()))
model_summary=load_json(MODEL_SUMMARY,'C8_model_summary_readable')
ref_summary=load_json(REF_SUMMARY,'frozen_R4_summary_readable')
if model_summary:
    add('C8_runtime_gate',model_summary.get('runtime_gate_pass') is True,
        json.dumps({k:model_summary.get(k) for k in ['status','event_count','all_finite',
             'all_prescribed_fluxes_finite_nonnegative','max_init_error','max_flux_identity_error',
             'max_CF0_DV_identity_error','max_paired_initial_state_error','max_paired_forcing_error']}))
if ref_summary:
    add('frozen_R4_runtime_gate',ref_summary.get('gate_pass') is True,
        json.dumps({k:ref_summary.get(k) for k in ['gate_pass','all_finite','max_init_error',
             'max_flux_identity_error','CF0_max_common_DV_identity_error']}))
current=None; frozen=None; audit=None; prep=None
if MODEL_RESP.is_file():
    try: current=pd.read_csv(MODEL_RESP)
    except Exception as exc: add('C8_response_file_readable',False,repr(exc))
else: add('C8_response_file_present',False,str(MODEL_RESP))
if REF.is_file():
    try: frozen=pd.read_csv(REF)
    except Exception as exc: add('frozen_R4_response_file_readable',False,repr(exc))
if MODEL_AUDIT.is_file():
    try: audit=pd.read_csv(MODEL_AUDIT)
    except Exception as exc: add('C8_runtime_audit_readable',False,repr(exc))
else: add('C8_runtime_audit_present',False,str(MODEL_AUDIT))
if PREP.is_file():
    try: prep=pd.read_csv(PREP)
    except Exception as exc: add('R4_prepared_input_readable',False,repr(exc))
if current is not None:
    add('C8_40_events_complete',len(current)==40 and current.event_id.astype(str).nunique()==40,
        'rows='+str(len(current))+' unique='+str(current.event_id.astype(str).nunique()))
if prep is not None:
    expected_ids=set(prep.loc[prep.anchor_primary_3_6.astype(bool),'event_id'].astype(str))
    add('frozen_anchor_primary_3_6_is_40',len(expected_ids)==40,'unique anchor IDs='+str(len(expected_ids)))
    if current is not None:
        ids=set(current.event_id.astype(str))
        add('C8_event_set_matches_frozen_anchor_primary_3_6',ids==expected_ids,
            'missing='+str(sorted(expected_ids-ids))+' extra='+str(sorted(ids-expected_ids)))
if audit is not None:
    max_init=float(audit.init_err.max()) if len(audit) else float('inf')
    max_flux=float(audit.flux_identity_error.max()) if len(audit) else float('inf')
    q=audit[['q_common','q_top','q_main']].to_numpy(float) if len(audit) else np.array([[np.nan]])
    finite=bool(audit.finite.astype(bool).all() and np.isfinite(q).all())
    q_nonneg=bool((q>=0).all())
    pair_x=float(audit.paired_initial_state_error.max()) if len(audit) else float('inf')
    pair_d=float(audit.paired_forcing_error.max()) if len(audit) else float('inf')
    add('C8_R4_runtime_audits',len(audit)==80 and finite and q_nonneg and max_init<=1e-8 and
        max_flux<=1e-12 and pair_x==0.0 and pair_d==0.0,
        json.dumps({'audit_rows':len(audit),'all_finite':finite,'q_nonnegative':q_nonneg,
                    'max_init_error':max_init,'max_flux_identity_error':max_flux,
                    'max_paired_initial_state_error':pair_x,'max_paired_forcing_error':pair_d}))
if audit is not None and ref_summary:
    try:
        base_audit=pd.read_csv(REF_AUDIT)
        ba=base_audit[(base_audit.stage=='CF0') & (base_audit.cohort=='primary')]
        ok=(len(ba)==80 and bool(ba.finite.astype(bool).all()) and
            float(ba.init_err.max())<=1e-8 and float(ba.flux_identity_error.max())<=1e-12)
        add('frozen_R4_CF0_primary_runtime_audit',ok,'rows='+str(len(ba)))
    except Exception as exc:
        add('frozen_R4_CF0_primary_runtime_audit',False,repr(exc))
max_dt=None; max_dah=None; completed=0
if current is not None and frozen is not None:
    c=current.copy(); f=frozen.copy()
    if {'stage','cohort','horizon_min','model','event_id'}.issubset(f.columns):
        f=f[(f.stage=='CF0') & (f.cohort=='primary') & (f.horizon_min==15) & (f.model=='M1')].copy()
    add('frozen_R4_CF0_primary_15min_40_events',
        len(f)==40 and f.event_id.astype(str).nunique()==40,
        'rows='+str(len(f))+' unique='+str(f.event_id.astype(str).nunique()))
    add('C8_response_schema',{'event_id','model_T_delta','model_AH_delta'}.issubset(c.columns),','.join(c.columns))
    if {'event_id','model_T_delta','model_AH_delta'}.issubset(c.columns) and {'event_id','model_T_delta','model_AH_delta'}.issubset(f.columns):
        try:
            merged=c[['event_id','model_T_delta','model_AH_delta']].copy()
            merged['event_id']=merged.event_id.astype(str)
            merged=merged.rename(columns={'model_T_delta':'C8_model_T_delta','model_AH_delta':'C8_model_AH_delta'})
            ff=f[['event_id','model_T_delta','model_AH_delta']].copy()
            ff['event_id']=ff.event_id.astype(str)
            ff=ff.rename(columns={'model_T_delta':'frozen_R4_CF0_T_delta','model_AH_delta':'frozen_R4_CF0_AH_delta'})
            joined=ff.merge(merged,on='event_id',how='outer',indicator=True,validate='one_to_one')
            joined['abs_T_difference']=np.abs(joined.C8_model_T_delta-joined.frozen_R4_CF0_T_delta)
            joined['abs_AH_difference']=np.abs(joined.C8_model_AH_delta-joined.frozen_R4_CF0_AH_delta)
            completed=int((joined['_merge']=='both').sum())
            max_dt=float(joined.abs_T_difference.max()) if completed else float('inf')
            max_dah=float(joined.abs_AH_difference.max()) if completed else float('inf')
            joined.to_csv(OUT/'C8_event_comparison.csv',index=False,float_format='%.12g')
            add('C8_event_id_alignment',len(joined)==40 and completed==40 and bool((joined['_merge']=='both').all()),
                'matched='+str(completed)+' comparison rows='+str(len(joined)))
            add('C8_T_difference_within_1e-6',max_dt<=1e-6,'max_abs_difference_C='+str(max_dt))
            add('C8_AH_difference_within_1e-6',max_dah<=1e-6,'max_abs_difference_g_m3='+str(max_dah))
        except Exception as exc:
            add('C8_per_event_comparison',False,repr(exc))
else:
    add('C8_per_event_comparison',False,'model or frozen response file unavailable')
dep_match=False
if BASE_PIP.is_file() and MODEL_PIP.is_file():
    dep_match=BASE_PIP.read_text().strip()==MODEL_PIP.read_text().strip()
    add('frozen_R4_dependency_freeze_match',dep_match,'exact sorted pip-freeze text comparison')
else:
    add('frozen_R4_dependency_freeze_match',False,'one or both pip-freeze files unavailable')
passed=not failures
status='PASS' if passed else 'FAIL'
summary={
 'status':status,'passed':passed,'workflow_run_url':RUN_URL,'workflow_run_id':RUN_ID,
 'repository':REPO,'r5_branch':'r5-field-ventilation-flux','r5_workflow_sha':SHA,
 'r4_base_commit':'6e687ba1b55c38d7fc184c94006a1d9fba10aa4d',
 'greenlight_commit':'2d3febb1ea002b24b452e32293e990beb78d3ce1',
 'events_completed':completed,'events_expected':40,
 'max_abs_T_difference_C':max_dt,'max_abs_AH_difference_g_m3':max_dah,
 'tolerance':1e-6,'all_checks_passed':passed,'failed_checks':failures,'checks':checks
}
(OUT/'C8_gate_summary.json').write_text(json.dumps(summary,indent=2))
lines=[
 '# C8 R4 CF0 reproduction report','',
 '- Status: **'+status+'**',
 '- GitHub Actions run: '+RUN_URL,
 '- R5 branch: r5-field-ventilation-flux',
 '- Workflow commit: '+SHA,
 '- Frozen R4 base commit: 6e687ba1b55c38d7fc184c94006a1d9fba10aa4d',
 '- GreenLight commit: 2d3febb1ea002b24b452e32293e990beb78d3ce1',
 '- Events completed: '+str(completed)+' / 40',
 '- Maximum absolute T difference: '+(str(max_dt) if max_dt is not None else 'unavailable')+' °C',
 '- Maximum absolute AH difference: '+(str(max_dah) if max_dah is not None else 'unavailable')+' g m-3',
 '- Tolerance: 1e-6 in both variables',
 '- All C8 reproduction and R4 runtime audit checks passed: '+str(passed),
 '',
 '## Checks','',
 '| Check | Result | Detail |','|---|---:|---|'
]
for c in checks:
    lines.append('| '+c['check']+' | '+('PASS' if c['passed'] else 'FAIL')+' | '+c['detail'].replace('|','\\|')+' |')
lines += ['', '## Files in this artifact','',
          '- C8_event_comparison.csv: event-by-event C8 vs frozen R4 CF0 values and absolute differences.',
          '- C8_run_log.md: commands, gate states and artifact provenance.',
          '- C8_gate_summary.json: machine-readable result.',
          '- Test, model and aggregation console logs plus raw model responses and runtime audit.',
          '',
          'No FF0, FFN, or FFH model run was included.']
(OUT/'C8_reproduction_report.md').write_text('\n'.join(lines)+'\n')
log=[
 '# C8 run log','',
 '- Route: GitHub Actions, R5 branch only.',
 '- R5 branch base: R4 frozen branch head 6e687ba1b55c38d7fc184c94006a1d9fba10aa4d.',
 '- Workflow run: '+RUN_URL,
 '- R5 workflow commit: '+SHA+'.',
 '- GreenLight checkout: 2d3febb1ea002b24b452e32293e990beb78d3ce1.',
 '- Frozen input artifact: r4-prepared-inputs, producer run 37164458429; primary input SHA256 '+EXPECTED['prepared_input']+'.',
 '- Frozen reference artifact: r4-m1, producer run 37164458429; CF0 response SHA256 '+EXPECTED['r4_cf0_responses']+'; R4 audit SHA256 '+EXPECTED['r4_runtime_audit']+'; R4 summary SHA256 '+EXPECTED['r4_summary']+'; pip freeze SHA256 '+EXPECTED['r4_pip_freeze']+'.',
 '- R3 reference artifact: r3-m1, producer run 37163085468; HV response SHA256 '+EXPECTED['r3_hv_responses']+'.',
 '- Prepare job: '+PREPARE_RESULT+'. C8 model job: '+C8_RESULT+'.',
 '- Field flux unit test and frozen-input validation logs are included.',
 '- C8 evaluated only REF0/CF0 on anchor_primary_3_6 at 15 min.',
 '- C8 status: '+status+'; matched event comparisons '+str(completed)+' / 40.',
 '- Maximum absolute differences: T='+(str(max_dt) if max_dt is not None else 'unavailable')+' °C; AH='+(str(max_dah) if max_dah is not None else 'unavailable')+' g m-3; required tolerance 1e-6.',
 '- Frozen R4 dependency freeze exact match: '+str(dep_match)+'.',
 '- No formal R5 FF condition was run.'
]
(OUT/'C8_run_log.md').write_text('\n'.join(log)+'\n')
print(json.dumps(summary,indent=2))
if not passed:
    raise SystemExit('C8 reproduction gate failed')
