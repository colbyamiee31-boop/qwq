import ast, pathlib, math, hashlib, numpy as np, pandas as pd
from scipy.stats import spearmanr
root=pathlib.Path(r'E:\Users\PC\Documents\ChatGPT\1019，PHYBENCH'); review=root/'B2_A1_main_review_20261101'; gate=root/'B2_A1_review_20261101/CF_A1_gate.py';r5=root/'handoff_inspection_20261101_200710/repo_readonly/physbench/r5/aggregate_r5.py'
main=(review/'CF_A1_main.py').read_text(encoding='utf-8');reference=gate.read_text(encoding='utf-8');agg=(review/'CF_A1_aggregate_cf.py').read_text(encoding='utf-8');old=r5.read_text(encoding='utf-8');tree=lambda s:ast.parse(s);fn=lambda s:{n.name:n for n in tree(s).body if isinstance(n,ast.FunctionDef)};m=fn(main);g=fn(reference);a=fn(agg);o=fn(old)
for name in ['add_check','sha256','bool_values','locked_hashes','verify_source_hashes','build_weather','full_action','initial_state','prescribed_flux','run_arm','response_rows_for','write_outputs','a1_flux','compare_gate']:
 assert ast.dump(m[name],include_attributes=False)==ast.dump(g[name],include_attributes=False),name
start='    env = gym.make('; end='    summary["ffh_code_location"]'
assert main[main.index(start):main.index(end)] == reference[reference.index(start):reference.index(end)]
for name in ['add','sha256','safe_json','read_json','bool_values','metric','confidence_intervals','make_r5_canonical','make_hv_canonical','compute_summary','paired_bootstrap','compute_paired']:
 assert ast.dump(a[name],include_attributes=False)==ast.dump(o[name],include_attributes=False),name
assert 'SEED = 20261019' in agg and 'BOOT = 2000' in agg
assert main.index('compare_gate(pre_gate_responses, baseline, "REF0")') < main.index('    for _, r in E.iterrows():\n        for condition in CONDITIONS:')
assert 'CONDITIONS = [\'FF0-CF\', \'FFH-CF\']' in main and 'and len(audit) == 776' in main and 'len(responses) == 388' in main
assert 'response_rows.clear()' in main and 'CF_A1_C8_runtime_audit.csv' in main
assert all(x in agg for x in ['("FF0-CF", "REF0")','("FFH-CF", "FF0-CF")','("FF0-CF", "HV")','("FF0-CF", "FF0")'])
# Isolated synthetic statistics: exact R5 values and identical grouped RNG sampling.
def env_for(nodes):
 env={'np':np,'pd':pd,'spearmanr':spearmanr,'SEED':20261019,'BOOT':9};exec(compile(ast.Module(body=[nodes[n] for n in ['metric','confidence_intervals','paired_bootstrap']],type_ignores=[]),'<synthetic>','exec'),env);return env
x=env_for(a);y=env_for(o);frame=pd.DataFrame({'event_date':['a','a','b','b','c','c'],'delta_u':[1.,-1.,1.,0.,1.,-1.],'gradient':[2.,2.,3.,1.,0.,1.],'model_delta':[-1.,.5,-.4,0.,0.,.2],'obs_delta':[-.8,.6,-.3,0.,0.,.3],'pre_state':[10.]*6,'post_state':[9.,10.5,9.6,10.,10.,10.2]});other=frame.copy();other['model_delta']*=.8
assert x['metric'](frame,'T')==y['metric'](frame,'T');assert x['confidence_intervals'](frame,'T')==y['confidence_intervals'](frame,'T');assert x['paired_bootstrap'](frame,other,'T','abs_fidelity_error')==y['paired_bootstrap'](frame,other,'T','abs_fidelity_error')
workflow=(review/'CF_A1_main_workflow_review.yml').read_text(encoding='utf-8-sig');assert 'CF_A1_MAIN_APPROVED_TRIGGER_ONCE.txt' in workflow and 'GITHUB_RUN_ATTEMPT' in workflow and 'name: cf-a1-main' in workflow
for filename in ['CF_A1_main.py','CF_A1_aggregate_cf.py']:
 digest=hashlib.sha256((review/filename).read_bytes()).hexdigest();assert digest in workflow,(filename,digest)
print('PASS: unchanged audited model helpers/ODE through shared functions; exact R5 statistical functions and seed; synthetic metric, grouped bootstrap and paired differences; C8-before-main sequence, 388/776 counts, workflow marker and code hashes. No GreenLight was invoked.')
