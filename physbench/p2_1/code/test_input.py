import argparse,hashlib,numpy as np
from common import load_compact
ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); a=ap.parse_args()
sha=hashlib.sha256(open(a.npz,'rb').read()).hexdigest(); assert sha=='e3c6bf7a35e71a0b29cf2e3397afafd008c4496967bf0fa82d3d52ed68c0677c',sha
allr=[]
for s in range(8): allr.extend(load_compact(a.npz,s))
assert len(allr)==1195 and len({r.atlas_id for r in allr})==1195
print({'sha256':sha,'rows':len(allr),'G1':sum(r.greenhouse=='G1' for r in allr),'G2':sum(r.greenhouse=='G2' for r in allr),'G3':sum(r.greenhouse=='G3' for r in allr)})
