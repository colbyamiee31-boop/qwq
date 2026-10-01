from pathlib import Path
import os,sys,json,subprocess,hashlib,zipfile,platform,sysconfig,csv,shutil,importlib.metadata as md

ROOT=Path.cwd(); PAY=ROOT/'payload'; OUT=ROOT/'evidence'; OUT.mkdir(exist_ok=True)
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args,log,cwd=None):
 with (OUT/log).open('w') as f:subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
def capture(args):
 r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);return {'returncode':r.returncode,'output':r.stdout}

assert platform.system()=='Linux'
assert sys.version_info[:3]==(3,11,16),sys.version
for n,v in [('numpy','1.26.4'),('casadi','3.8.1'),('gl-gym','0.3.2')]:assert md.version(n)==v
manifest=json.loads((PAY/'INPUT_SHA256.json').read_text())
for n,d in manifest.items():assert sha(PAY/n)==d,n
dump(OUT/'input_verification.json',{'matched':len(manifest),'mismatch':0,'hashes':manifest})
import casadi,numpy,gl_gym
fingerprint={'platform':platform.platform(),'uname':list(platform.uname()),'libc':platform.libc_ver(),'python':sys.version,'python_build':platform.python_build(),'python_compiler':platform.python_compiler(),'python_executable_sha256':sha(Path(sys.executable)),'sysconfig':{k:sysconfig.get_config_var(k) for k in ['CC','CFLAGS','CONFIG_ARGS','SOABI','MULTIARCH','Py_ENABLE_SHARED']},'packages':{n:md.version(n) for n in ['numpy','casadi','gl-gym','pandas','scipy','pip']},'runner':{k:os.environ.get(k) for k in ['ImageOS','ImageVersion','RUNNER_OS','RUNNER_ARCH','RUNNER_ENVIRONMENT','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','GITHUB_SHA']},'commands':{name:capture(cmd) for name,cmd in {'os_release':['cat','/etc/os-release'],'kernel':['uname','-a'],'glibc':['ldd','--version'],'gcc':['gcc','--version'],'gcc13':['gcc-13','--version'],'cpu':['lscpu'],'pip_freeze':[sys.executable,'-m','pip','freeze'],'dpkg':['dpkg-query','-W','libc6','libgcc-s1','libstdc++6','gcc-13'],'greenlight_commit':['git','-C','GreenLight-Gym2','rev-parse','HEAD']}.items()}}
dump(OUT/'environment/fingerprint.json',fingerprint)
(OUT/'environment/pip_freeze.txt').write_text(fingerprint['commands']['pip_freeze']['output'])
libs=[]
for module in [casadi,numpy]:
 for p in sorted(Path(module.__file__).parent.rglob('*')):
  if p.is_file() and ('.so' in p.name):libs.append({'package':module.__name__,'file':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
dump(OUT/'environment/shared_libraries.json',libs)
dump(OUT/'environment/wheels.json',[{'filename':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((ROOT/'wheels').glob('*.whl'))])
dump(OUT/'environment/casadi_ldd.json',{p.name:capture(['ldd',str(p)]) for p in Path(casadi.__file__).parent.glob('*') if p.name in ['_casadi.so','libcasadi.so','libcasadi_integrator_cvodes.so']})
dump(OUT/'environment/source_hashes.json',{str(p.relative_to(Path(gl_gym.__file__).parent)):sha(p) for p in Path(gl_gym.__file__).parent.rglob('*.py')})
with zipfile.ZipFile(PAY/'source/physbench-history_a6fefad8.zip') as z:z.extractall(OUT/'exp12')
print('START EXP1.2',flush=True)
command([sys.executable,'physbench/exp1_2/m1_state_conditioned_co2.py'],'EXP1_2.log',OUT/'exp12')
old=json.loads((PAY/'historical/EXP1_2_M1_STATE_CONDITIONED_CO2.json').read_text());new=json.loads((OUT/'exp12/EXP1_2_M1_STATE_CONDITIONED_CO2.json').read_text())
rows=[]
def walk(a,b,path='$'):
 if isinstance(a,dict):
  assert isinstance(b,dict) and a.keys()==b.keys(),path
  for k in a:walk(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{path}[{i}]')
 else:
  numeric=isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float))
  rows.append({'json_path':path,'kind':'numeric' if numeric else 'other','historical':a,'rerun':b,'exact_equal':a==b,'absolute_difference':abs(a-b) if numeric else None,'relative_difference':abs(a-b)/abs(a) if numeric and a else None})
walk(old,new)
with (OUT/'EXP1_2_all_leaf_differences.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'leaf_count':len(rows),'numeric_count':sum(r['kind']=='numeric' for r in rows),'unequal_numeric_count':sum(r['kind']=='numeric' and not r['exact_equal'] for r in rows),'unequal_other_count':sum(r['kind']=='other' and not r['exact_equal'] for r in rows),'json_semantic_exact_equal':old==new,'json_bytes_exact_equal':(PAY/'historical/EXP1_2_M1_STATE_CONDITIONED_CO2.json').read_bytes()==(OUT/'exp12/EXP1_2_M1_STATE_CONDITIONED_CO2.json').read_bytes(),'baseline_hash_matches':sum(a==b for a,b in zip(old['baseline_native_state_sha256'],new['baseline_native_state_sha256'])),'baseline_hash_count':len(old['baseline_native_state_sha256'])}
dump(OUT/'EXP1_2_comparison_summary.json',summary);print(json.dumps(summary),flush=True)
print('START B/C same Linux process environment',flush=True)
subprocess.run(['git','-C','GreenLight-Gym2','archive','--format=zip','--output',str(PAY/'source/GreenLight-Gym2_2d3febb1.zip'),'HEAD'],check=True)
command([sys.executable,str(PAY/'source/reproduce_bc.py'),'--work-dir',str(OUT/'bc'),'--lane','both'],'BC.log')
print('B/C COMPLETED',flush=True)
dump(OUT/'completion.json',{'EXP1_2':'completed','BC_lanes':['archive_definition','frozen_source'],'fingerprint_applies_to':'All three runs; same job, interpreter, installed distributions and unmodified source','automatic_exact_claim':False})
