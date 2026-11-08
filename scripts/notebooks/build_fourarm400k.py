"""Build four fresh CPU training notebooks and one validation comparison notebook."""
from pathlib import Path
import argparse,hashlib,json
BOOTSTRAP='''import os,sys,json,hashlib,zipfile,shutil,tempfile,subprocess,importlib.util
from pathlib import Path
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
required=["numpy","scipy","matplotlib","filelock"]+(["torch"] if REQUIRE_TORCH else [])
missing=[name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Thiếu thư viện cài sẵn: "+str(missing)+". Notebook không cài qua Internet.")
INPUT,WORKING=Path("/kaggle/input"),Path("/kaggle/working")
WORKING.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):h.update(block)
    return h.hexdigest()
def relative_safe(name):
    p=Path(name)
    if p.anchor or ".." in p.parts or not p.parts or "\\\\" in name:
        raise ValueError("Unsafe relative path")
    return p
def extract_safe(path,destination):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError("Duplicate archive entries")
        for member in z.infolist():
            relative_safe(member.orig_filename)
            relative_safe(member.filename)
            if ((member.external_attr>>16)&0o170000)==0o120000:raise ValueError("Symlink in archive")
        if z.testzip() is not None:raise ValueError("ZIP CRC failed")
        z.extractall(destination)
def load_source(input_dir,working_dir):
    candidates=[]
    for path in sorted(input_dir.rglob("DRL400K_DATASET_MANIFEST.json")):
        m=json.loads(path.read_text())
        if m.get("version")=="drl_fourarm400k_dataset_v1":candidates.append(("directory",path.parent,m))
    for path in sorted(input_dir.rglob("UAV_DRL400K_DATASET.zip")):
        with zipfile.ZipFile(path) as z:
            if "DRL400K_DATASET_MANIFEST.json" not in z.namelist():continue
            m=json.loads(z.read("DRL400K_DATASET_MANIFEST.json"))
        if m.get("version")=="drl_fourarm400k_dataset_v1":candidates.append(("zip",path,m))
    if not candidates:raise FileNotFoundError("Gắn Dataset mới chứa DRL400K_DATASET_MANIFEST.json + uav_thesis hoặc UAV_DRL400K_DATASET.zip.")
    identity=lambda m:hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
    if len({identity(m) for _,_,m in candidates})!=1:raise RuntimeError("Chỉ gắn một phiên bản Dataset DRL400K.")
    kind,data,manifest=candidates[0]
    needed={"uav_thesis/scripts/training/run_bs5_fourarm400k.py","uav_thesis/configs/bs5_fourarm400k.json","uav_thesis/scripts/analysis/compare_bs5_fourarm400k_training.py"}
    if not needed.issubset(manifest["files_sha256"]):raise ValueError("Dataset thiếu mã nguồn cần thiết")
    parent=working_dir/("drl400k_source_"+identity(manifest)[:12])
    if not parent.exists():
        stage=Path(tempfile.mkdtemp(prefix="drl400k_stage_",dir=working_dir))
        try:
            if kind=="zip":extract_safe(data,stage)
            else:
                for relative in manifest["files_sha256"]:
                    source=data/relative_safe(relative)
                    if not source.resolve().is_relative_to(data.resolve()) or not source.is_file():raise FileNotFoundError(relative)
                    target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                (stage/"DRL400K_DATASET_MANIFEST.json").write_text(json.dumps(manifest))
            for relative,expected in manifest["files_sha256"].items():
                if sha(stage/relative_safe(relative))!=expected:raise RuntimeError("Source checksum mismatch: "+relative)
            stage.rename(parent)
        finally:
            if stage.exists():shutil.rmtree(stage)
    for relative,expected in manifest["files_sha256"].items():
        if sha(parent/relative_safe(relative))!=expected:raise RuntimeError("Source checksum mismatch: "+relative)
    return parent/"uav_thesis"
SOURCE=load_source(INPUT,WORKING)
print("Đã kiểm tra mã nguồn:",SOURCE)
'''
RESUME='''OUTPUT=WORKING/"drl400k_runs"/RUN_NAME
OUTPUT.parent.mkdir(exist_ok=True)
def restore_training(origin,output,arm,seed,total):
    if output.exists():raise RuntimeError("Output đã tồn tại; để RESUME_PATH trống để tiếp tục output này.")
    origin=Path(origin)
    temp=None
    try:
        if origin.is_file():
            temp=Path(tempfile.mkdtemp(prefix="drl400k_resume_",dir=WORKING));extract_safe(origin,temp);base=temp
        elif origin.is_dir():base=origin
        else:raise FileNotFoundError(origin)
        matches=[]
        for path in base.rglob("protocol.json"):
            p=json.loads(path.read_text())
            if (p.get("version")=="bs5_fourarm400k_v1" and p.get("arm")==arm and p.get("training_seed")==seed
                and p.get("total_steps")==total and not p.get("diagnostic") and p.get("fair_config",{}).get("mode")=="instant"):
                matches.append(path.parent)
        if len(matches)!=1:raise ValueError("RESUME phải đúng cấu hình, seed và ngân sách 399600 của bộ mới. Không dùng checkpoint 300k.")
        if not (matches[0]/"resume.pt").is_file():raise ValueError("Đây là REVIEW; cần gói RESUME có optimizer/replay/RNG.")
        shutil.copytree(matches[0],output)
    finally:
        if temp is not None:shutil.rmtree(temp)
if RESUME_PATH:restore_training(RESUME_PATH,OUTPUT,ARM,TRAINING_SEED,TOTAL_STEPS)
if OUTPUT.exists() and not (OUTPUT/"resume.pt").exists():
    raise RuntimeError("Output chưa có checkpoint boundary; hãy đổi RUN_NAME để chạy mới.")
print("Output:",OUTPUT)
'''
PREFLIGHT='''if RUN_PREFLIGHT:
    preflight=Path(tempfile.mkdtemp(prefix="drl400k_preflight_",dir=WORKING))/"check"
    subprocess.run([sys.executable,str(SOURCE/"scripts/verification/verify_bs5_fourarm400k.py"),
                    "--arm",ARM,"--output",str(preflight)],cwd=SOURCE,check=True)
    print("Đã đạt preflight PyTorch và kiểm tra resume. Diagnostic không nằm trong kết quả train.")
'''
RUN='''command=[sys.executable,str(SOURCE/"scripts/training/run_bs5_fourarm400k.py"),
         "--arm",ARM,"--output",str(OUTPUT),"--total-steps",str(TOTAL_STEPS),
         "--training-seed",str(TRAINING_SEED),"--device","cpu","--max-seconds",str(MAX_SECONDS)]
if (OUTPUT/"resume.pt").exists():command.append("--resume")
if PAUSE_AT_STEP:command += ["--pause-at-step",str(PAUSE_AT_STEP)]
try:
    subprocess.run(command,cwd=SOURCE,check=True)
finally:
    if (OUTPUT/"status.json").is_file():
        subprocess.run([sys.executable,str(SOURCE/"scripts/verification/export_bs5_fourarm400k.py"),
                        "--runs",str(OUTPUT),"--output",str(OUTPUT.parent),"--name",RUN_NAME],cwd=SOURCE,check=True)
'''
LINKS='''from IPython.display import display,FileLink,Image
for ending in ("REVIEW.zip","RESUME.zip","exports.json"):
    path=OUTPUT.parent/(RUN_NAME+"_"+ending)
    if path.exists():display(FileLink(str(path)))
if (OUTPUT/"status.json").exists():print((OUTPUT/"status.json").read_text())
for name in ("learning_curve.png","reward_components.png","fairness_continuity.png","energy_operation.png","role_collision.png","training_diagnostics.png"):
    if (OUTPUT/name).exists():display(Image(filename=str(OUTPUT/name)))
print("Gửi REVIEW để phân tích. Giữ RESUME để tiếp tục nếu chưa đủ 399600 bước.")
'''
COMPARE='''OUTPUT=WORKING/"drl400k_training_comparison"
subprocess.run([sys.executable,str(SOURCE/"scripts/analysis/compare_bs5_fourarm400k_training.py"),
                "--input",str(INPUT),"--output",str(OUTPUT)],cwd=SOURCE,check=True)
report_zip=WORKING/"DRL400K_TRAINING_COMPARISON.zip"
with zipfile.ZipFile(report_zip,"w",zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():z.write(path,"DRL400K_TRAINING_COMPARISON/"+str(path.relative_to(OUTPUT)))
with zipfile.ZipFile(report_zip) as z:
    if z.testzip() is not None:raise ValueError("Report ZIP CRC failed")
from IPython.display import display,FileLink,Image
print((OUTPUT/"completeness.json").read_text());display(FileLink(str(report_zip)))
for name in ("validation_overlays.png","fairness_overlays.png"):display(Image(filename=str(OUTPUT/name)))
'''
def cell(kind,s):
    r=dict(cell_type=kind,id=hashlib.sha256((kind+s).encode()).hexdigest()[:16],metadata={},source=s.splitlines(keepends=True))
    if kind=='code':r.update(execution_count=None,outputs=[])
    return r
def notebook(cells):return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.13')),nbformat=4,nbformat_minor=5)
def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for idx,arm,slug in [(1,'C-N','CN'),(2,'C-R','CR'),(3,'C-H','CH'),(4,'H-only','H_ONLY')]:
        intro=f'''# {arm}: QMIX, Fair tức thời, 399.600 bước, CPU\n\nChạy mới từ bước 0, không nạp trọng số 300k. Gắn Dataset `UAV_DRL400K_DATASET.zip`; nhận cả thư mục tự giải nén. Chọn CPU và Internet Off. Có thể chạy notebook 01–04 đồng thời trên các phiên riêng.\n\nReward: 0,4 × thông lượng chuẩn hóa + 0,6 × Jain tức thời có điều kiện − 0,1 × năng lượng chuẩn hóa − 0,2 × tỷ lệ di chuyển bị chặn. Cùng seed 551, curriculum và validation 720001/720002. Chưa chạy seed test. Chi tiết trong HUONG_DAN_DRL400K.md.\n'''
        config=f'''ARM={arm!r}\nRUN_NAME='DRL_{slug}_S551_400K'\nTOTAL_STEPS=399600  #12k curriculum + 323 × 1200 bước toàn nhiệm vụ\nTRAINING_SEED=551\nMAX_SECONDS=36000  #dừng ở ranh giới episode trước giới hạn phiên 12 giờ\nPAUSE_AT_STEP=0\nRESUME_PATH=""  #lần đầu để trống; khi cần tiếp tục, dùng đúng gói RESUME của notebook này\nRUN_PREFLIGHT=True\nREQUIRE_TORCH=True\n'''
        cells=[cell('markdown',intro),cell('code',config),cell('code',BOOTSTRAP),cell('code',RESUME),cell('code',PREFLIGHT),cell('code',RUN),cell('code',LINKS)]
        (output/f'{idx:02d}_{slug}_400K_CPU.ipynb').write_text(json.dumps(notebook(cells),ensure_ascii=False,indent=1)+'\n')
    intro='# So sánh bốn đường học\n\nSau khi train, gắn Dataset mới và bốn gói REVIEW từ notebook 01–04. Nhận cả ZIP và thư mục. Chỉ so sánh log validation, không train, không chạy test và không chọn lại checkpoint bằng test. Thiếu cấu hình hoặc chưa đạt399600bước thì báo chưa hoàn tất.\n'
    (output/'05_COMPARE_400K_TRAINING_CPU.ipynb').write_text(json.dumps(notebook([cell('markdown',intro),cell('code','REQUIRE_TORCH=False\n'),cell('code',BOOTSTRAP),cell('code',COMPARE)]),ensure_ascii=False,indent=1)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
