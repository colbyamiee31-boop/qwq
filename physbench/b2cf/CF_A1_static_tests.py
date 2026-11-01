import ast, pathlib, tempfile, json, contextlib, io
import numpy as np,pandas as pd
root=pathlib.Path(r'E:\Users\PC\Documents\ChatGPT\1019，PHYBENCH'); p=root/'B2_A1_review_20261101/CF_A1_gate.py'; original=root/'handoff_inspection_20261101_200710/repo_readonly/physbench/r5/m1_r5_formal.py'; source=p.read_text(encoding='utf-8');tree=ast.parse(source); orig=ast.parse(original.read_text(encoding='utf-8'))
functions=lambda t:{x.name:x for x in t.body if isinstance(x,ast.FunctionDef)}
a,b=functions(tree),functions(orig); preserved=['build_weather','initial_state','full_action','response_rows_for','locked_hashes','verify_source_hashes'];assert all(ast.dump(a[n],include_attributes=False)==ast.dump(b[n],include_attributes=False) for n in preserved)
# R5 ODE/integrator setup copied exactly from R5.
start='    env = gym.make(';end='    summary["ffh_code_location"]';assert source[source.index(start):source.index(end)]==original.read_text(encoding='utf-8')[original.read_text(encoding='utf-8').index(start):original.read_text(encoding='utf-8').index(end)]
required=['prepare_unit_test_and_lock_hash_gates','execution_lock_source_hashes','event_join_and_anchor_primary_3_6','frozen_R4_CF0_runtime_audits','pre_FF_subset_runtime_gates','all_prescribed_fluxes_finite_nonnegative','all_states_outputs_finite_and_initial_state_error','flux_identity_all_arms_horizons','paired_initial_state_and_forcing_identical','response_row_count_40x2x2','embedded_C8_REF0_40_event_reproduction','A1_frozen_input_hashes'];assert all(x in source for x in required)
node=a['compare_gate']; code=compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec');checks=[]
with tempfile.TemporaryDirectory(prefix='cf_a1_unit_') as td:
 env={'np':np,'pd':pd,'json':json,'OUT':pathlib.Path(td),'HORIZONS':[15,30],'summary':{'nature_statement':'test'}}
 def add_check(name,passed,detail):
  checks.append((name,passed))
  if not passed: raise RuntimeError(name)
 env['add_check']=add_check;exec(code,env);compare=env['compare_gate'];ref=pd.DataFrame([{'event_id':f'e{i}','horizon_min':h,'condition':c,'dT_model':1.,'dAH_model':1.} for i in range(97) for h in (15,30) for c in ('REF0','FF0')]);
 def got(c,delta):
  d=ref[(ref.condition==c)&ref.event_id.isin([f'e{i}' for i in range(40)])].copy();d.loc[d.event_id=='e0','dT_model']+=delta;return d
 compare(got('REF0',5e-7),ref,'REF0');assert pd.read_csv(pathlib.Path(td)/'CF_A1_REF0_comparisons.csv').passed.all()
 try: compare(got('REF0',5e-6),ref,'REF0');raise AssertionError('G3 expected failure')
 except RuntimeError:pass
 compare(got('FF0',5e-6),ref,'FF0');table=pd.read_csv(pathlib.Path(td)/'CF_A1_FF0_comparisons.csv');assert len(table)==160 and (table.criterion=='absolute_2e-5').sum()==2 and table.passed.all()
 try: compare(got('FF0',2.1e-5),ref,'FF0');raise AssertionError('G4 expected failure')
 except RuntimeError:pass
print('PASS: syntax, six original helpers, exact R5 ODE/integrator block, 12 audit paths, synthetic G3/G4 relative/absolute/failure bounds. No GreenLight simulations executed.')
