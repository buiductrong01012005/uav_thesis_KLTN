"""Six independent offline CPU heuristic workers and one strict merge notebook."""
from pathlib import Path
import argparse,json
from scripts.notebooks.build_fourarm400k import BOOTSTRAP as TRAIN_BOOTSTRAP,cell,notebook
BOOTSTRAP=TRAIN_BOOTSTRAP
for old,new in [('DRL400K_DATASET_MANIFEST.json','HEUR400K_DATASET_MANIFEST.json'),('drl_fourarm400k_dataset_v1','heur_fourarm400k_dataset_v1'),('UAV_DRL400K_DATASET.zip','UAV_HEUR400K_BENCH_DATASET.zip'),('drl400k_source_','heur400k_source_'),('drl400k_stage_','heur400k_stage_'),('scripts/training/run_bs5_fourarm400k.py','scripts/evaluation/run_fourarm400k_heuristics.py'),('scripts/analysis/compare_bs5_fourarm400k_training.py','scripts/analysis/report_fourarm400k_heuristics.py'),('Dataset DRL400K','Dataset heuristic mới')]:BOOTSTRAP=BOOTSTRAP.replace(old,new)
RESUME='''OUTPUT=WORKING/"heur400k_runs"/RUN_NAME
OUTPUT.parent.mkdir(exist_ok=True)
def restore_benchmark(origin,output,label,part,seeds):
    if output.exists():raise RuntimeError("Output đã tồn tại; để RESUME_PATH trống để tiếp tục output này.")
    temp=None
    try:
        origin=Path(origin)
        if origin.is_file():
            temp=Path(tempfile.mkdtemp(prefix="heur400k_resume_",dir=WORKING));extract_safe(origin,temp);base=temp
        elif origin.is_dir():base=origin
        else:raise FileNotFoundError(origin)
        expected=list(range(930201+(part-1)*5,930206+(part-1)*5)) if seeds is None else list(seeds)
        matches=[]
        for path in base.rglob("protocol.json"):
            p=json.loads(path.read_text())
            if (p.get("version")=="fourarm400k_heuristic_benchmark_v1" and p.get("label")==label and p.get("part")==part
                and p.get("horizon")==10000 and p.get("target_seeds")==expected and not p.get("diagnostic")):matches.append(path.parent)
        if len(matches)!=1:raise ValueError("RESUME phải đúng heuristic, part, tập seed và môi trường mới.")
        root=matches[0]
        for seed in expected:
            folder=root/f"seed{seed}"
            if (folder/"summary.json").exists() and not json.loads((folder/"summary.json").read_text()).get("complete") and not (folder/"progress.json").exists():raise ValueError("Seed dở dang cần RESUME; REVIEW không có trạng thái chạy tiếp.")
        shutil.copytree(root,output)
    finally:
        if temp is not None:shutil.rmtree(temp)
if RESUME_PATH:restore_benchmark(RESUME_PATH,OUTPUT,LABEL,PART,SEEDS)
print("Output:",OUTPUT)
'''
PREFLIGHT='''if RUN_PREFLIGHT:
    preflight=Path(tempfile.mkdtemp(prefix="heur400k_preflight_",dir=WORKING))/"check"
    subprocess.run([sys.executable,str(SOURCE/"scripts/verification/verify_fourarm400k_heuristics.py"),
                    "--label",LABEL,"--output",str(preflight)],cwd=SOURCE,check=True)
    print("Đã đạt kiểm tra môi trường, controller, reward, dữ liệu raw và resume giữa seed. Chưa chạy seed test.")
'''
RUN='''command=[sys.executable,str(SOURCE/"scripts/evaluation/run_fourarm400k_heuristics.py"),
         "--label",LABEL,"--part",str(PART),"--output",str(OUTPUT),"--max-seconds",str(MAX_SECONDS),
         "--save-every",str(SAVE_EVERY),"--log-every",str(LOG_EVERY)]
if SEEDS is not None:command += ["--seeds",*map(str,SEEDS)]
subprocess.run(command,cwd=SOURCE,check=True)
'''
LINKS='''from IPython.display import display,FileLink
for ending in ("REVIEW.zip","RESUME.zip","INDEX.json"):
    path=OUTPUT.parent/(RUN_NAME+"_"+ending)
    if path.exists():display(FileLink(str(path)))
if (OUTPUT/"status.json").exists():print((OUTPUT/"status.json").read_text())
print("Gửi REVIEW để tổng hợp. Giữ RESUME nếu còn seed chưa xong.")
'''
MERGE='''OUTPUT=WORKING/"heur400k_report"
subprocess.run([sys.executable,str(SOURCE/"scripts/analysis/report_fourarm400k_heuristics.py"),
                "--input",str(INPUT),"--output",str(OUTPUT)],cwd=SOURCE,check=True)
report_zip=WORKING/"HEUR400K_BENCH_REPORT.zip"
with zipfile.ZipFile(report_zip,"w",zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():z.write(path,"HEUR400K_BENCH_REPORT/"+str(path.relative_to(OUTPUT)))
with zipfile.ZipFile(report_zip) as z:
    if z.testzip() is not None:raise ValueError("Report ZIP CRC failed")
from IPython.display import display,FileLink,Image
print((OUTPUT/"completeness.json").read_text());display(FileLink(str(report_zip)))
for name in ("paired_metrics.png","service_temporal.png"):
    if (OUTPUT/name).exists():display(Image(filename=str(OUTPUT/name)))
'''
def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    index=1
    for label,title in [('RANDOM','Random Access/Relay'),('DYNAMIC_AR','Dynamic Access/Relay'),('HEURISTIC_HO','Heuristic di chuyển với vai trò kết hợp')]:
        for part in (1,2):
            seeds=list(range(930201+(part-1)*5,930206+(part-1)*5))
            intro=f'''# {title}: benchmark phần {part}\n\nCPU, Internet Off, không cần PyTorch hoặc checkpoint DRL. Gắn Dataset mới `UAV_HEUR400K_BENCH_DATASET.zip`. Nhận cả ZIP và thư mục tự giải nén.\n\nMặc định chạy 5 seed: {seeds}, mỗi seed 10.000 slot. Sáu notebook H01–H06 chạy độc lập. Có lưu giữa seed và xuất REVIEW/RESUME. Xem HUONG_DAN_HEUR400K.md.\n\nGiữ nguyên controller cũ và thông tin có trễ. Môi trường vật lý, reward đánh giá và tập test giống giao thức dành cho DRL 400k. Heuristic vẫn tìm theo điểm T/C/F cũ; chưa phải heuristic tối ưu năng lượng.\n'''
            config=f'''LABEL={label!r}\nPART={part}\nSEEDS=None  #mặc định 5 seed; có thể chọn một tập con đúng part nếu muốn chia nhỏ hơn\nRUN_NAME=f"HEUR400K_{{LABEL}}_PART{{PART}}"+("_SEEDS_"+"_".join(map(str,SEEDS)) if SEEDS else "")\nMAX_SECONDS=36000\nSAVE_EVERY=500\nLOG_EVERY=100\nRESUME_PATH=""  #lần đầu để trống; chạy tiếp dùng đúng RESUME của worker này\nRUN_PREFLIGHT=True\nREQUIRE_TORCH=False\n'''
            path=output/f'H{index:02d}_{label}_PART{part}_CPU.ipynb';index+=1
            path.write_text(json.dumps(notebook([cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',RESUME),cell('code',PREFLIGHT),cell('code',RUN),cell('code',LINKS)]),ensure_ascii=False,indent=1)+'\n')
    intro='# Tổng hợp benchmark heuristic mới\n\nGắn Dataset mới và sáu gói REVIEW, hoặc các worker đã chia nhỏ theo seed. Kiểm tra dữ liệu raw, cân bằng pin, reward, hash quỹ đạo UE và đủ 10 seed của mỗi cấu hình. Chưa đủ thì báo phần thiếu, chưa kết luận. Không ghép kết quả heuristic môi trường cũ.\n'
    (output/'H07_MERGE_HEUR400K_CPU.ipynb').write_text(json.dumps(notebook([cell('markdown',intro),cell('code','REQUIRE_TORCH=False\n'),cell('code',BOOTSTRAP),cell('code',MERGE)]),ensure_ascii=False,indent=1)+'\n')
if __name__=='__main__':
    import sys
    # Use python -m scripts.notebooks.build_fourarm400k_heuristics from repository root.
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
