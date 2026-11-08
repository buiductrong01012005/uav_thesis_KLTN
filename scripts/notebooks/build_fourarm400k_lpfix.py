"""Fresh H-only552 numerical-fix notebook, plus compatible later merge/test notebooks."""
from pathlib import Path
import argparse,json,tempfile
from scripts.notebooks.build_fourarm400k_multiseed import build as build_original
REPLACEMENTS=[('MULTI400K_DATASET_MANIFEST.json','LPFIX400K_DATASET_MANIFEST.json'),('fourarm400k_multiseed_dataset_v1','fourarm400k_lpfix_dataset_v1'),('UAV_MULTI400K_CPU_DATASET.zip','UAV_MULTI400K_LPFIX_CPU_DATASET.zip'),('multi400k_source_','lpfix400k_source_'),('multi400k_stage_','lpfix400k_stage_')]
NUMERIC_PREFLIGHT='''if RUN_PREFLIGHT:
    preflight_root=Path(tempfile.mkdtemp(prefix="lpfix400k_preflight_",dir=WORKING))
    preflight_env=dict(os.environ,UAV_LP_DIAGNOSTIC_DIR=str(preflight_root/"numeric_diagnostics"))
    subprocess.run([sys.executable,"-m","unittest","discover","-s",str(SOURCE/"tests"),
                    "-p","test_bs5_lp_roundoff.py","-v"],cwd=SOURCE,env=preflight_env,check=True)
    subprocess.run([sys.executable,str(SOURCE/"scripts/verification/verify_fourarm400k_multiseed.py"),
                    "--arm",ARM,"--training-seed",str(TRAINING_SEED),"--output",str(preflight_root/"check")],
                   cwd=SOURCE,env=preflight_env,check=True)
    print("LP roundoff regression và PyTorch thật đã đạt; bắt đầu lượt train mới.")
'''
def replace(nb):
 for cell in nb['cells']:
  s=''.join(cell['source'])
  for a,b in REPLACEMENTS:s=s.replace(a,b)
  cell['source']=s.splitlines(keepends=True)
 return nb

def build(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='lpfix_notebooks_') as tmp:
  original=Path(tmp);build_original(original)
  path=next((original/'01_train_more').glob('*_H_ONLY_S552_*.ipynb'));nb=replace(json.loads(path.read_text()))
  nb['cells'][0]['source']=('''# H-only seed 552: chạy mới từ 0, LP numerical guard\n\nCPU, Internet Off. Gắn Dataset mới `UAV_MULTI400K_LPFIX_CPU_DATASET.zip`, nhận cả ZIP và thư mục tự giải nén. Lượt mới dùng tên output có hậu tố LPFIX; giữ `RESUME_PATH=""`, không nạp RESUME của lượt lỗi cũ. Ngân sách 399.600 bước và các tham số khoa học giữ nguyên.\n\nSửa đầu ra LP âm rất nhỏ trong sai số solver; NaN và vi phạm thực sự vẫn bị từ chối. Preflight bắt buộc kiểm tra regression, gradient PyTorch và resume. Chưa replay được trạng thái lỗi thật của seed552; nếu có lỗi khác, giữ REVIEW/RESUME cùng log chẩn đoán.\n''').splitlines(keepends=True)
  for cell in nb['cells']:
   if cell['cell_type']!='code':continue
   s=''.join(cell['source'])
   if 'TRAINING_SEED=552' in s:s=s.replace('DRL_H_ONLY_S552_400K"','DRL_H_ONLY_S552_400K_LPFIX"')
   if 'verify_fourarm400k_multiseed.py' in s:s=NUMERIC_PREFLIGHT
   if s.startswith('command=[sys.executable'):s='os.environ["UAV_LP_DIAGNOSTIC_DIR"]=str(OUTPUT/"lp_diagnostics")\n'+s
   cell['source']=s.splitlines(keepends=True)
  (output/'F01_H_ONLY_S552_400K_LPFIX_CPU.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
  for group in ['03_drl_test_after_train','04_analysis']:
   (output/group).mkdir(exist_ok=True)
   for path in sorted((original/group).glob('*.ipynb')):
    nb=replace(json.loads(path.read_text()))
    nb['cells'][0]['source']+=['\nBản LPFIX: nhận dạng nguồn legacy/guarded bằng whitelist hash chính xác, giữ nguyên protocol/checkpoint gốc. Gắn Dataset LPFIX mới.\n']
    for cell in nb['cells']:
     if cell['cell_type']=='code' and ''.join(cell['source']).startswith('command=[sys.executable'):
      cell['source']=['os.environ["UAV_LP_DIAGNOSTIC_DIR"]=str(OUTPUT/"lp_diagnostics")\n']+cell['source']
    (output/group/path.name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
 return 29
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.output))
