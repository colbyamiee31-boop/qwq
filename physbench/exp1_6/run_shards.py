"""Schedule independent M2 strata; no in-process monkey-patch concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=a.model_source.resolve()
    run=Path(__file__).with_name('run.py').resolve()
    tasks=[(s,f) for s in ['S_COOL_HUMID','S_NOMINAL','S_WARM_DRY'] for f in [0,6,10]]
    def one(task):
        s,f=task
        name=f'{s}_F{f}'
        command=[sys.executable,'-u',str(run),'--model','M2','--model-source',str(source),
                 '--out',str(out/name),'--state',s,'--forcing',str(f)]
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(name,r.returncode,flush=True)
        return {'stratum':name,'returncode':r.returncode,'command':command}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(one,tasks))
    (out/'scheduler.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(x['returncode'] for x in results):
        raise SystemExit('At least one shard failed; do not aggregate as PASS')


if __name__=='__main__':main()
