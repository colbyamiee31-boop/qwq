"""Wait for each completed parent shard and run its independent coverage audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--workers',type=int,default=9)
    p.add_argument('--stage',choices=['refine','polish'],default='refine')
    a=p.parse_args()
    parents=a.parents.resolve();out=a.out.resolve();source=a.model_source.resolve()
    out.mkdir(parents=True,exist_ok=False)
    tasks=[f'{s}_F{f}' for s in ['S_COOL_HUMID','S_NOMINAL','S_WARM_DRY'] for f in [0,6,10]]
    script=Path(__file__).with_name(a.stage+'.py').resolve()
    def one(name):
        start=time.monotonic();parent=parents/name
        while not (parent/'output_hashes.json').exists():
            status=parents/'scheduler.json'
            if status.exists():
                records=json.loads(status.read_text())
                if any(r['stratum']==name and r['returncode']!=0 for r in records):
                    return {'stratum':name,'returncode':-1,'reason':'Parent scheduler reported failure'}
            if time.monotonic()-start>8*3600:raise RuntimeError('Parent wait deadline exceeded')
            time.sleep(10)
        if json.loads((parent/'result.json').read_text())['status']!='PASS_EXECUTION':
            return {'stratum':name,'returncode':-1,'reason':'Parent failed'}
        command=[sys.executable,'-u',str(script),'--parent',str(parent),'--out',str(out/name),'--model-source',str(source)]
        with (out/f'{name}.log').open('w',encoding='utf-8') as f:
            r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        print(name,r.returncode,flush=True)
        return {'stratum':name,'returncode':r.returncode,'command':command}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,tasks))
    (out/'scheduler.json').write_text(json.dumps(results,indent=2))
    if any(x['returncode'] for x in results):raise SystemExit('Coverage shard failure')


if __name__=='__main__':main()
