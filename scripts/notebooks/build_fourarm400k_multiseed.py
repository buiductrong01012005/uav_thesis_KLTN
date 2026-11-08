"""Build 8 additional train, 4 heuristic, 24 later DRL-test and 4 analysis notebooks."""
from pathlib import Path
import argparse,json
from scripts.notebooks.build_fourarm400k import BOOTSTRAP as ORIGINAL_BOOTSTRAP,RESUME as TRAIN_RESUME,RUN as TRAIN_RUN,LINKS as TRAIN_LINKS,cell,notebook
from scripts.notebooks.build_fourarm400k_heuristics import RESUME as HEUR_RESUME,RUN as HEUR_RUN,LINKS as BENCH_LINKS,PREFLIGHT as HEUR_PREFLIGHT
BOOTSTRAP=ORIGINAL_BOOTSTRAP
for a,b in [('DRL400K_DATASET_MANIFEST.json','MULTI400K_DATASET_MANIFEST.json'),('drl_fourarm400k_dataset_v1','fourarm400k_multiseed_dataset_v1'),('UAV_DRL400K_DATASET.zip','UAV_MULTI400K_CPU_DATASET.zip'),('drl400k_source_','multi400k_source_'),('drl400k_stage_','multi400k_stage_'),('Dataset DRL400K','Dataset MULTI400K')]:BOOTSTRAP=BOOTSTRAP.replace(a,b)
TRAIN_PREFLIGHT='''if RUN_PREFLIGHT:
    preflight=Path(tempfile.mkdtemp(prefix="multi400k_train_preflight_",dir=WORKING))/"check"
    subprocess.run([sys.executable,str(SOURCE/"scripts/verification/verify_fourarm400k_multiseed.py"),
                    "--arm",ARM,"--training-seed",str(TRAINING_SEED),"--output",str(preflight)],cwd=SOURCE,check=True)
    print("Đã kiểm tra PyTorch thật, seed mới, cùng phần thân mạng giữa các cấu hình và resume chính xác.")
'''
DRL_RESUME='''OUTPUT=WORKING/"multi400k_test_runs"/RUN_NAME
OUTPUT.parent.mkdir(exist_ok=True)
MODELS=Path(MODELS_INPUT) if MODELS_INPUT else INPUT
def restore_drl_test(origin,output,arm,training_seed,part,seeds):
    if output.exists():raise RuntimeError("Output đã tồn tại; để RESUME_PATH trống để tiếp tục.")
    temp=None
    try:
        origin=Path(origin)
        if origin.is_file():
            temp=Path(tempfile.mkdtemp(prefix="multi400k_test_resume_",dir=WORKING));extract_safe(origin,temp);base=temp
        elif origin.is_dir():base=origin
        else:raise FileNotFoundError(origin)
        expected=list(range(930201+(part-1)*5,930206+(part-1)*5)) if seeds is None else list(seeds)
        matches=[]
        for path in base.rglob("protocol.json"):
            p=json.loads(path.read_text())
            if (p.get("version")=="fourarm400k_drl_test_v1" and p.get("arm")==arm and p.get("training_seed")==training_seed
                and p.get("part")==part and p.get("target_seeds")==expected and p.get("horizon")==10000 and not p.get("diagnostic")):matches.append(path.parent)
        if len(matches)!=1:raise ValueError("RESUME phải đúng arm, seed train, phần test và tập seed kịch bản.")
        for seed in expected:
            folder=matches[0]/f"seed{seed}"
            if (folder/"summary.json").exists() and not json.loads((folder/"summary.json").read_text()).get("complete") and not (folder/"progress.json").exists():raise ValueError("REVIEW còn dở không chứa trạng thái resume; cần đúng RESUME.")
        shutil.copytree(matches[0],output)
    finally:
        if temp is not None:shutil.rmtree(temp)
if RESUME_PATH:restore_drl_test(RESUME_PATH,OUTPUT,ARM,TRAINING_SEED,PART,SEEDS)
print("Output:",OUTPUT)
'''
DRL_PREFLIGHT='''if RUN_PREFLIGHT:
    preflight=Path(tempfile.mkdtemp(prefix="multi400k_test_preflight_",dir=WORKING))/"check"
    subprocess.run([sys.executable,str(SOURCE/"scripts/verification/verify_fourarm400k_drl_benchmark.py"),
                    "--models",str(MODELS),"--arm",ARM,"--training-seed",str(TRAINING_SEED),"--output",str(preflight)],cwd=SOURCE,check=True)
    print("Model best đã cố định; greedy actor đúng như lúc validation; resume và raw audit đạt trên seed diagnostic.")
'''
DRL_RUN='''command=[sys.executable,str(SOURCE/"scripts/evaluation/run_fourarm400k_drl_benchmark.py"),
         "--arm",ARM,"--training-seed",str(TRAINING_SEED),"--models",str(MODELS),"--part",str(PART),
         "--output",str(OUTPUT),"--max-seconds",str(MAX_SECONDS),"--save-every",str(SAVE_EVERY),"--log-every",str(LOG_EVERY)]
if SEEDS is not None:command += ["--seeds",*map(str,SEEDS)]
subprocess.run(command,cwd=SOURCE,check=True)
'''

def write(path,cells):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(notebook(cells),ensure_ascii=False,indent=1)+'\n')

def analysis_code(kind,freeze=False):
 if kind=='training':
  setup='OUTPUT=WORKING/"multi400k_training_summary"\n'
  args='[sys.executable,str(SOURCE/"scripts/analysis/compare_fourarm400k_multiseed.py"),"--input",str(INPUT),"--output",str(OUTPUT)]'
  extra='\nif FREEZE_MODELS:command.append("--freeze-models")'
  figures=['learning_mean_sd.png','individual_training_seeds.png'];name='MULTI400K_TRAINING_REPORT'
 else:
  setup=f'OUTPUT=WORKING/{("multi400k_heuristic_summary" if kind=="heuristics" else "multi400k_final_summary")!r}\n'
  args='[sys.executable,str(SOURCE/"scripts/analysis/report_fourarm400k_multiseed.py"),"--input",str(INPUT),"--output",str(OUTPUT)]'
  extra='\ncommand.append("--heuristics-only")' if kind=='heuristics' else ''
  figures=['main_metrics.png','service_metrics.png','training_seed_sensitivity.png'];name='MULTI400K_HEURISTIC_REPORT' if kind=='heuristics' else 'MULTI400K_FINAL_REPORT'
 return setup+'command='+args+extra+'''
subprocess.run(command,cwd=SOURCE,check=True)
from IPython.display import display,FileLink,Image
print((OUTPUT/"completeness.json").read_text())
report_zip=WORKING/"'''+name+'''.zip"
with zipfile.ZipFile(report_zip,"w",zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name!="FOURARM400K_FROZEN_MODELS.zip":z.write(path,str(Path(OUTPUT.name)/path.relative_to(OUTPUT)))
with zipfile.ZipFile(report_zip) as z:
    if z.testzip() is not None:raise ValueError("Report ZIP CRC failed")
display(FileLink(str(report_zip)))
if (OUTPUT/"FOURARM400K_FROZEN_MODELS.zip").is_file():display(FileLink(str(OUTPUT/"FOURARM400K_FROZEN_MODELS.zip")))
for name in '''+repr(figures)+''':
    if (OUTPUT/name).exists():display(Image(filename=str(OUTPUT/name)))
'''

def build(output):
 output=Path(output);count=0
 arms=[('C-N','CN'),('C-R','CR'),('C-H','CH'),('H-only','H_ONLY')]
 for seed in (552,553):
  for arm,slug in arms:
   count+=1
   intro=f'# {arm}, seed train {seed}: chạy mới 399.600 bước\n\nCPU, Internet Off. Gắn UAV_MULTI400K_CPU_DATASET.zip hoặc thư mục tự giải nén. Notebook này là một lượt train độc lập, không tiếp tục trọng số seed 551. Giữ reward, mô hình, tập train/validation và ngân sách như lượt 551. Tải cả REVIEW và RESUME.\n'
   config=f'ARM={arm!r}\nTRAINING_SEED={seed}\nRUN_NAME="DRL_{slug}_S{seed}_400K"\nTOTAL_STEPS=399600\nMAX_SECONDS=36000\nPAUSE_AT_STEP=0\nRESUME_PATH=""\nRUN_PREFLIGHT=True\nREQUIRE_TORCH=True\n'
   write(output/'01_train_more'/f'T{count:02d}_{slug}_S{seed}_400K_CPU.ipynb',[cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',TRAIN_RESUME),cell('code',TRAIN_PREFLIGHT),cell('code',TRAIN_RUN),cell('code',TRAIN_LINKS)])
 count=0
 for label,title in [('DYNAMIC_AR','Heuristic lựa chọn truy nhập/chuyển tiếp'),('HEURISTIC_HO','Heuristic di chuyển với vai trò kết hợp')]:
  for part in (1,2):
   count+=1;seedtext=f'{930201+(part-1)*5}–{930205+(part-1)*5}'
   intro=f'# {title}, phần {part}\n\nChạy ngay song song với train. CPU, Internet Off, không cần model hoặc PyTorch. Gắn UAV_MULTI400K_CPU_DATASET.zip. Chạy năm seed {seedtext}, mỗi seed 10.000 slot. Đây là một heuristic cho một tập kịch bản, không cần lặp theo seed train của DRL. Controller cũ được giữ nguyên, reward đánh giá và môi trường khớp DRL.\n'
   config=f'LABEL={label!r}\nPART={part}\nSEEDS=None  #hoặc tập con đúng phần, ví dụ [930201] cho phần 1\nRUN_NAME=f"MULTI400K_{{LABEL}}_PART{{PART}}"+("_SEEDS_"+"_".join(map(str,SEEDS)) if SEEDS else "")\nMAX_SECONDS=36000\nSAVE_EVERY=500\nLOG_EVERY=100\nRESUME_PATH=""\nRUN_PREFLIGHT=True\nREQUIRE_TORCH=False\n'
   write(output/'02_heuristics_now'/f'H{count:02d}_{label}_PART{part}_CPU.ipynb',[cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',HEUR_RESUME),cell('code',HEUR_PREFLIGHT),cell('code',HEUR_RUN),cell('code',BENCH_LINKS)])
 count=0
 for arm,slug in arms:
  for seed in (551,552,553):
   for part in (1,2):
    count+=1
    intro=f'# Test {arm}, model train seed {seed}, phần {part}\n\nChạy sau khi đã đóng băng đủ 12 model best. Gắn UAV_MULTI400K_CPU_DATASET.zip và FOURARM400K_FROZEN_MODELS.zip từ A02. CPU, Internet Off. Mỗi notebook chạy một model trên năm seed test, mỗi seed 10.000 slot. Giữ cùng model khi resume. Không chọn lại checkpoint theo test.\n'
    config=f'ARM={arm!r}\nTRAINING_SEED={seed}\nPART={part}\nSEEDS=None\nMODELS_INPUT=""  #để trống để tìm bundle trong /kaggle/input\nRUN_NAME=f"MULTI400K_TEST_{slug}_TRAIN{{TRAINING_SEED}}_PART{{PART}}"+("_SEEDS_"+"_".join(map(str,SEEDS)) if SEEDS else "")\nMAX_SECONDS=36000\nSAVE_EVERY=500\nLOG_EVERY=100\nRESUME_PATH=""\nRUN_PREFLIGHT=True\nREQUIRE_TORCH=True\n'
    write(output/'03_drl_test_after_train'/f'B{count:02d}_{slug}_S{seed}_PART{part}_CPU.ipynb',[cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',DRL_RESUME),cell('code',DRL_PREFLIGHT),cell('code',DRL_RUN),cell('code',BENCH_LINKS)])
 for name,title,kind,freeze in [('A01_COMPARE_TRAINING_CPU.ipynb','Xem đường học nhiều seed, có thể chạy khi chưa đủ lượt','training',False),('A02_FREEZE_12_BEST_MODELS_CPU.ipynb','Đóng băng 12 model best theo validation, sau khi đủ lượt','training',True),('A03_MERGE_2_HEURISTICS_CPU.ipynb','Tổng hợp riêng hai heuristic, không đợi train','heuristics',False),('A04_FINAL_6_CONFIGURATIONS_CPU.ipynb','Tổng hợp bốn cấu hình DRL và hai heuristic trên test','final',False)]:
  intro=f'# {title}\n\nGắn Dataset MULTI400K và các REVIEW tương ứng; nhận cả ZIP và thư mục. A01/A02 dùng REVIEW train, A03 dùng REVIEW heuristic, A04 dùng REVIEW test DRL và heuristic. Thiếu dữ liệu thì báo phần thiếu. A02 cần đủ 12 REVIEW có best.pt; không dùng chỉ các CSV rời. A04 giữ sáu nhãn cấu hình và báo riêng biến thiên giữa model.\n'
  config=f'REQUIRE_TORCH=False\nFREEZE_MODELS={freeze}\n'
  write(output/'04_analysis'/name,[cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',analysis_code(kind,freeze))])
 return 40
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.output))
