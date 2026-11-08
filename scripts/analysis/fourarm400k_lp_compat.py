"""One explicit source-version equivalence for bounded LP roundoff; never rewrite evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'configs/bs5_lp_roundoff_compat.json'

def manifest():return json.loads(MANIFEST.read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def canonical_sources(source_map):
 result=dict(source_map)
 for path,record in manifest()['files'].items():
  if path not in result:continue
  if result[path] not in [record['legacy_sha256'],record['guarded_sha256']]:raise ValueError('Unknown numerical-source revision: '+path)
  result[path]=record['legacy_sha256']
 return result

def verify_source(path,recorded_sha,root=ROOT):
 current=root/path
 if not current.is_file():raise ValueError('Missing locked source: '+path)
 actual=hashlib.sha256(current.read_bytes()).hexdigest()
 if actual==recorded_sha:return
 allowed=manifest()['files'].get(path)
 if allowed and {actual,recorded_sha}<={allowed['legacy_sha256'],allowed['guarded_sha256']}:return
 # Repository import receipts allow only reviewed path-serialization changes
 # in runners, never changes to src/, configuration, reward or test evidence.
 receipt=ROOT/'configs/bs5_final400k_integration.json'
 if receipt.is_file() and path.startswith('scripts/'):
  record=json.loads(receipt.read_text())['path_portability'].get(path)
  if record and {actual,recorded_sha}<={record['reference_sha256'],record['integrated_sha256']}:return
 raise ValueError('Locked source changed beyond the declared LP roundoff patch: '+path)

def canonical_training(protocol):
 p=deepcopy(protocol)
 if 'source_sha256' in p:p['source_sha256']=canonical_sources(p['source_sha256'])
 return p

def canonical_environment(contract):
 result=deepcopy(contract)
 result['physics_source_sha256']=canonical_sources(result['physics_source_sha256'])
 return result

def describe_source(protocol):
 path='src/diagnostics/bs5_spectrum_airtime.py';actual=protocol['source_sha256'].get(path);record=manifest()['files'][path]
 if actual not in [record['legacy_sha256'],record['guarded_sha256']]:raise ValueError('Unknown LP source')
 return dict(actual_allocator_sha256=actual,numerical_variant='guarded' if actual==record['guarded_sha256'] else 'legacy',compatibility_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
