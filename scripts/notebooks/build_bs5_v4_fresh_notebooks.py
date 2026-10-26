"""Build uploadable, separate Kaggle notebooks from the checked source tree."""
from pathlib import Path
import argparse
import json
import textwrap


BOOTSTRAP = r'''
import os, sys, json, zipfile, hashlib, shutil, subprocess, time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
if 'SESSION_START' not in globals(): SESSION_START = time.monotonic()
INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
MARKER = 'configs/bs5_return_v4_fresh_train.json'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def is_main_archive(path, version):
    try:
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if len(Path(n).parts)==2 and Path(n).name=='protocol.json']
            if len(names)!=1: return False
            p=json.loads(z.read(names[0]))
            return p.get('version')==version and not p.get('diagnostic',False)
    except (OSError,ValueError,zipfile.BadZipFile): return False
def extract_safe(archive, target):
    target=Path(target).resolve()
    for m in archive.infolist():
        dest=(target/m.filename).resolve()
        if not dest.is_relative_to(target) or (m.external_attr>>16)&0o170000==0o120000:
            raise RuntimeError('Unsafe ZIP member: '+m.filename)
    if archive.testzip() is not None: raise RuntimeError('ZIP CRC failed')
    archive.extractall(target)
if SOURCE_PATH is None:
    choices=[]
    for m in INPUT.rglob('DATASET_MANIFEST.json'):
        if (m.parent/MARKER).exists(): choices.append(m.parent)
    for f in INPUT.rglob('*.zip'):
        try:
            with zipfile.ZipFile(f) as z:
                if 'uav_thesis/DATASET_MANIFEST.json' in z.namelist() and 'uav_thesis/'+MARKER in z.namelist(): choices.append(f)
        except zipfile.BadZipFile: pass
    # Identical copies are safe; different versions require an explicit path.
    identities={}
    for f in choices:
        if f.is_dir(): raw=(f/'DATASET_MANIFEST.json').read_bytes()
        else:
            with zipfile.ZipFile(f) as z: raw=z.read('uav_thesis/DATASET_MANIFEST.json')
        identities.setdefault(hashlib.sha256(raw).hexdigest(),[]).append(f)
    if len(identities)!=1: raise RuntimeError('Add Input đúng source Dataset; hoặc điền SOURCE_PATH: '+str(choices))
    SOURCE_PATH=next(iter(identities.values()))[0]
SOURCE_PATH=Path(SOURCE_PATH)
SOURCE_ROOT=Path('/kaggle/temp/bs5_v4_fresh_source/uav_thesis')
if SOURCE_PATH.is_dir(): expected=(SOURCE_PATH/'DATASET_MANIFEST.json').read_bytes()
else:
    with zipfile.ZipFile(SOURCE_PATH) as z: expected=z.read('uav_thesis/DATASET_MANIFEST.json')
if SOURCE_ROOT.exists():
    if (SOURCE_ROOT/'DATASET_MANIFEST.json').read_bytes()!=expected: raise RuntimeError('Working source khác Dataset. Giữ output và dùng phiên sạch.')
else:
    SOURCE_ROOT.parent.mkdir(parents=True,exist_ok=True)
    if SOURCE_PATH.is_dir(): shutil.copytree(SOURCE_PATH,SOURCE_ROOT)
    else:
        with zipfile.ZipFile(SOURCE_PATH) as z: extract_safe(z,SOURCE_ROOT.parent)
manifest=json.loads(expected)
for rel,digest in manifest.items():
    if sha(SOURCE_ROOT/rel)!=digest: raise RuntimeError('Source checksum failed: '+rel)
RUN_ENV=os.environ.copy()
RUN_ENV.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
RUN_ENV['PYTHONPATH']=str(SOURCE_ROOT)
probe="import torch,numpy,matplotlib,json; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,matplotlib=matplotlib.__version__,cuda=torch.cuda.is_available())))"
check=subprocess.run([sys.executable,'-B','-c',probe],cwd=SOURCE_ROOT,env=RUN_ENV,capture_output=True,text=True)
if check.returncode: raise RuntimeError('Kaggle runtime cần torch/numpy/matplotlib. Không tự thay runtime khi resume. '+check.stderr[-2000:])
print('Verified source:',SOURCE_ROOT,'files:',len(manifest))
print('Installed runtime:',check.stdout.strip())
def run_stream(args, log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=SOURCE_ROOT,env=RUN_ENV,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout: print(line,end='',flush=True);f.write(line);f.flush()
        return proc.wait()
def restore_archive_or_directory(src, destination, required):
    src,destination=Path(src),Path(destination)
    if destination.exists(): raise RuntimeError('Không chép đè output hiện có: '+str(destination))
    if src.is_dir() and (src/'protocol.json').exists():
        shutil.copytree(src,destination)
    else:
        if src.is_dir():
            candidates=list(src.rglob(required))
            if len(candidates)!=1: raise RuntimeError('Chọn đúng thư mục hoặc ZIP '+required+': '+str(candidates))
            src=candidates[0]
        with zipfile.ZipFile(src) as z:
            # Transport archives have exactly one top-level run directory.
            roots={Path(n).parts[0] for n in z.namelist() if Path(n).parts}
            if len(roots)!=1: raise RuntimeError('Resume ZIP phải có một thư mục gốc.')
            top=next(iter(roots))
            staging=WORK/('restore_'+destination.name)
            if staging.exists(): raise RuntimeError('Restore dở dang; giữ thư mục để kiểm tra: '+str(staging))
            staging.mkdir()
            extract_safe(z,staging)
            shutil.move(str(staging/top),str(destination));staging.rmdir()
    for checksum_name in ['EXPORT_SHA256SUMS.json','HANDOFF_SHA256SUMS.json']:
        checksum=destination/checksum_name
        if checksum.exists():
            for rel,digest in json.loads(checksum.read_text()).items():
                if sha(destination/rel)!=digest: raise RuntimeError('Input checksum failed: '+rel)
    if not (destination/'protocol.json').exists(): raise RuntimeError('Thiếu protocol trong resume.')
def show_exports(root):
    from IPython.display import display, FileLink, Markdown
    exports=sorted([p for p in WORK.glob(root.name+'_export*') if (p/'EXPORT_INDEX.json').exists()],key=lambda p:p.stat().st_mtime)
    if not exports: raise RuntimeError('Không có gói export hoàn tất; giữ nguyên output và log.')
    export=exports[-1]
    report=export/'analysis/REPORT.md'
    if report.exists(): display(Markdown(report.read_text()))
    for f in sorted(export.iterdir()):
        if f.is_file(): display(FileLink(str(f)))
    print('OUTPUT:',export)
    if json.loads((export/'EXPORT_INDEX.json').read_text()).get('export_audit')=='failed': raise RuntimeError('Export audit failed; raw evidence archive has been saved above.')
    return export
'''


def cell(kind, text, i):
    d = dict(cell_type=kind, id=f"bs5-v4-{i}", metadata={}, source=textwrap.dedent(text).strip().splitlines(keepends=True))
    if kind == "code":
        d.update(execution_count=None, outputs=[])
    return d


def build(root):
    directory = root / "scripts/notebooks"
    train = [
        ("markdown", """
        # Fresh Return-v4 — train C-R/C-N trước, benchmark ở notebook riêng

        1. Tạo một Kaggle Dataset từ **BS5_RETURN_V4_KAGGLE_DATASET.zip**, rồi Add Input.
        2. Import notebook này, chọn GPU nếu có rồi **Run All**. Không cần checkpoint/kết quả cũ.
        3. Bốn run fresh: C-R/C-N × seed551/552, mỗi run60k. Lịch S1–S3 giữ150 slot/game, S4 dùng1.200 slot/game; không đổi reward, router, pin, trạm sạc hay Return-v4.
        4. Validation0/6k/…/60k:5 trace cố định ×1.200 slot. Tổng chính:240k train +264k validation. Notebook này **không chạy benchmark10k**.

        UE tape và tham số game dùng chung giữa hai nhánh cùng seed. Initial weights giống nhau trong mỗi cặp; optimizer/replay/exploration độc lập. C-N mask mọi action Relay, kể cả target-Q, và tắt Relay trong router.

        Ghi raw từng step, bins tối đa100 slot, toàn bộ update loss/TD/gradient/epsilon; vẽ raw + trung bình trượt không vượt ranh giới curriculum. Validation chỉ có các mốc thực đo.

        Dừng mềm trước hạn; lưu optimizer, target networks, replay, RNG và từng game đã hoàn tất. Không cam kết cả bốn run xong trong12 giờ. Dùng lại cùng runtime/GPU khi resume; đổi runtime sẽ bị chặn để tránh gọi nhầm là exact resume.
        """),
        ("code", """
        from pathlib import Path
        import time
        SESSION_START = time.monotonic()
        SOURCE_PATH = None
        RESUME_INPUT_DIR = None  # tự tìm training_resume.zip / output cũ nếu chỉ có một bản
        OUTPUT_ROOT = Path('/kaggle/working/bs5_v4_training')
        DEVICE = 'auto'  # GPU nếu có; resume dùng device đã khóa
        MAX_TOTAL_SESSION_SECONDS = 10.5 * 3600
        EXPORT_RESERVE_SECONDS = 30 * 60
        RUN_MAIN = True
        """),
        ("code", BOOTSTRAP),
        ("code", """
        if RESUME_INPUT_DIR is None and not OUTPUT_ROOT.exists():
            candidates=[x for x in INPUT.rglob('training_resume.zip') if is_main_archive(x,'bs5_return_v4_fresh_crcn_v1')]
            directories=[]
            for f in INPUT.rglob('protocol.json'):
                try:
                    p=json.loads(f.read_text())
                    if p.get('version')=='bs5_return_v4_fresh_crcn_v1' and not p.get('diagnostic',False) and list((f.parent/'runs').glob('*/resume/*.pt')): directories.append(f.parent)
                except (ValueError,OSError): pass
            choices=candidates if candidates else directories
            if len(choices)>1: raise RuntimeError('Có nhiều bản resume; điền RESUME_INPUT_DIR rõ ràng.')
            if choices: RESUME_INPUT_DIR=choices[0]
        if RESUME_INPUT_DIR is not None and not OUTPUT_ROOT.exists(): restore_archive_or_directory(RESUME_INPUT_DIR,OUTPUT_ROOT,'training_resume.zip')
        PREFLIGHT=WORK/'bs5_v4_fresh_preflight'
        rc=run_stream(['scripts/verification/preflight_bs5_v4_fresh.py','--output',PREFLIGHT],WORK/'bs5_v4_preflight.log')
        if rc: raise RuntimeError('Preflight chưa PASS; main không chạy. Giữ nguyên log/receipt để kiểm tra.')
        receipt=json.loads((PREFLIGHT/'receipt.json').read_text())
        assert receipt['status']=='passed'
        print('Preflight PASS:',receipt['tests'],'tests; diagnostic seconds:',round(receipt['wall_s'],1))
        """),
        ("code", """
        MAIN_RETURN_CODE=0
        if RUN_MAIN:
            if not OUTPUT_ROOT.exists():
                rc=run_stream(['scripts/training/run_bs5_v4_fresh_train.py','prepare','--output',OUTPUT_ROOT,'--device',DEVICE],WORK/'bs5_v4_prepare.log')
                if rc: raise RuntimeError('Prepare failed; giữ log.')
            preflight=OUTPUT_ROOT/'preflight';preflight.mkdir(exist_ok=True)
            shutil.copy2(PREFLIGHT/'receipt.json',preflight/'receipt.json')
            shutil.copy2(WORK/'bs5_v4_preflight.log',preflight/'preflight.log')
            budget=max(0,MAX_TOTAL_SESSION_SECONDS-EXPORT_RESERVE_SECONDS-(time.monotonic()-SESSION_START))
            print('Main soft budget remaining:',round(budget/3600,2),'hours')
            MAIN_RETURN_CODE=run_stream(['scripts/training/run_bs5_v4_fresh_train.py','run','--output',OUTPUT_ROOT,'--max-seconds',str(budget)],WORK/'bs5_v4_training.log')
            print(json.loads((OUTPUT_ROOT/'status.json').read_text()))
        else: print('RUN_MAIN=False: bounded diagnostic preflight only.')
        """),
        ("code", """
        if RUN_MAIN and (OUTPUT_ROOT/'protocol.json').exists():
            rc=run_stream(['scripts/training/run_bs5_v4_fresh_train.py','export','--output',OUTPUT_ROOT],WORK/'bs5_v4_export.log')
            EXPORT=show_exports(OUTPUT_ROOT)
            if rc: raise RuntimeError('Export audit failed; raw output vẫn được giữ. Xem log trước khi tiếp tục.')
            if MAIN_RETURN_CODE: raise RuntimeError('Main có lỗi; gói bằng chứng đã xuất. Không xóa/sửa ledger để chạy lại.')
        """),
        ("markdown", """
        ## Kết quả và tiếp tục

        - **review_small.zip**: báo cáo, learning curve PNG/PDF, bảng bins/episodes/validation, toàn bộ update diagnostics và cycle records.
        - **training_resume.zip**: giữ đầy đủ để tiếp tục training nếu trạng thái `paused_at_game_boundary`. Add Input ZIP này hoặc dùng output notebook phiên trước; Run All sẽ nhận tiến độ. Cần cùng source/runtime/GPU.
        - **training_for_benchmark.zip**: chỉ xuất khi đủ cả bốn run và mọi validation. Đây là gói nhỏ chứa checkpoint best/last cùng lựa chọn đã khóa; dùng notebook benchmark riêng. Không dùng gói này để resume training.
        - `partial_error`: giữ nguyên tất cả output. Nếu có game bị ngắt đột ngột trước commit, ledger sẽ chặn lặp compute âm thầm; không reset counter. Dừng mềm thì resume được bình thường.
        - Chỉ lưu file trong session chưa bảo đảm đã giữ qua phiên khác: hãy lưu notebook version với output. Không bấm chạy mới từ đầu khi muốn resume.
        - Các seed test260001–260010 đã khóa trong benchmark_plan.json nhưng **không chạy** ở đây. Seed551/552 là seed train, khác seed tình huống test.
        - S1/S2 dùng fixture dạy phối hợp Relay; đây là ablation cùng recipe, không phải curriculum được tối ưu riêng cho C-N. Hai seed là kiểm tra lặp lại còn hạn chế.
        """),
    ]
    bench = [
        ("markdown", """
        # Return-v4 — benchmark riêng sau khi train xong

        Add Input **source Dataset** và **training_for_benchmark.zip** từ training đã complete. Chọn **CPU**; greedy chủ yếu dùng CPU. Run All.

        Bảy phương pháp: C-R, C-N, Random có Relay, 5A/4A1R/3A2R/Dynamic +Recovery. Hai nhánh RL mỗi nhánh có hai checkpoint train-seed551/552; thành chín cấu hình. Mỗi cấu hình chạy cùng10 seed tình huống ×10.000 slot =90 game/900k slot. Baseline không lặp theo seed train RL.

        Không train, không chọn lại checkpoint trên test. Mỗi game reset một lần; UE tape chung được khóa hash. Greedy dùng exact current UE/network và preview một slot trên bản sao env, không làm thay đổi Return memory của game thật; chi phí preview được ghi riêng.

        Tự lưu từng game; có thể chạy qua nhiều phiên. Bảng so sánh chính chỉ dùng khối tình huống đã đủ cả chín cấu hình, tránh so sánh số game khác nhau khi dừng giữa chừng.
        """),
        ("code", """
        from pathlib import Path
        import time
        SESSION_START=time.monotonic()
        SOURCE_PATH=None
        TRAINING_INPUT_DIR=None  # tự tìm training_for_benchmark.zip hoặc training output complete
        RESUME_INPUT_DIR=None  # tự tìm benchmark_resume.zip nếu có một bản
        OUTPUT_ROOT=Path('/kaggle/working/bs5_v4_benchmark')
        SCENARIOS=10  # chỉ đổi20 TRƯỚC lần prepare đầu; áp dụng cho mọi phương pháp
        MAX_TOTAL_SESSION_SECONDS=10.5*3600
        EXPORT_RESERVE_SECONDS=30*60
        RUN_MAIN=True
        """),
        ("code", BOOTSTRAP),
        ("code", """
        if RESUME_INPUT_DIR is None and not OUTPUT_ROOT.exists():
            candidates=[x for x in INPUT.rglob('benchmark_resume.zip') if is_main_archive(x,'bs5_return_v4_benchmark_v1')]
            directories=[]
            for f in INPUT.rglob('protocol.json'):
                try:
                    p=json.loads(f.read_text())
                    if p.get('version')=='bs5_return_v4_benchmark_v1' and not p.get('diagnostic',False): directories.append(f.parent)
                except (ValueError,OSError): pass
            choices=candidates if candidates else directories
            if len(choices)>1: raise RuntimeError('Chọn RESUME_INPUT_DIR khi có nhiều output benchmark.')
            if choices: RESUME_INPUT_DIR=choices[0]
        if RESUME_INPUT_DIR is not None and not OUTPUT_ROOT.exists(): restore_archive_or_directory(RESUME_INPUT_DIR,OUTPUT_ROOT,'benchmark_resume.zip')
        TRAINING_ROOT=WORK/'bs5_v4_training_handoff'
        if not OUTPUT_ROOT.exists():
            if TRAINING_INPUT_DIR is None:
                archives=[x for x in INPUT.rglob('training_for_benchmark.zip') if is_main_archive(x,'bs5_return_v4_fresh_crcn_v1')]
                roots=[]
                for f in INPUT.rglob('protocol.json'):
                    try:
                        p=json.loads(f.read_text())
                        if p.get('version')=='bs5_return_v4_fresh_crcn_v1' and not p.get('diagnostic',False) and (f.parent/'status.json').exists() and json.loads((f.parent/'status.json').read_text()).get('status')=='complete': roots.append(f.parent)
                    except (ValueError,OSError): pass
                choices=archives if archives else roots
                if len(choices)!=1: raise RuntimeError('Add Input training_for_benchmark.zip hoặc điền TRAINING_INPUT_DIR: '+str(choices))
                TRAINING_INPUT_DIR=choices[0]
            if not TRAINING_ROOT.exists(): restore_archive_or_directory(TRAINING_INPUT_DIR,TRAINING_ROOT,'training_for_benchmark.zip')
        PREFLIGHT=WORK/'bs5_v4_benchmark_preflight'
        rc=run_stream(['scripts/verification/preflight_bs5_v4_fresh.py','--output',PREFLIGHT],WORK/'bs5_v4_benchmark_preflight.log')
        if rc: raise RuntimeError('Preflight failed; benchmark không chạy.')
        """),
        ("code", """
        MAIN_RETURN_CODE=0
        if RUN_MAIN:
            if not OUTPUT_ROOT.exists():
                rc=run_stream(['scripts/evaluation/run_bs5_v4_benchmark.py','prepare','--output',OUTPUT_ROOT,'--training',TRAINING_ROOT,'--scenarios',str(SCENARIOS),'--timing-profile',PREFLIGHT/'receipt.json'],WORK/'bs5_v4_benchmark_prepare.log')
                if rc: raise RuntimeError('Benchmark prepare failed; không đổi checkpoint/test để thử lại.')
            budget=max(0,MAX_TOTAL_SESSION_SECONDS-EXPORT_RESERVE_SECONDS-(time.monotonic()-SESSION_START))
            MAIN_RETURN_CODE=run_stream(['scripts/evaluation/run_bs5_v4_benchmark.py','run','--output',OUTPUT_ROOT,'--max-seconds',str(budget)],WORK/'bs5_v4_benchmark.log')
            rc=run_stream(['scripts/evaluation/run_bs5_v4_benchmark.py','export','--output',OUTPUT_ROOT],WORK/'bs5_v4_benchmark_export.log')
            EXPORT=show_exports(OUTPUT_ROOT)
            if rc or MAIN_RETURN_CODE: raise RuntimeError('Có lỗi: giữ raw export/ledger/log để kiểm tra, không reset counters.')
        """),
        ("markdown", """
        ## Output

        review_small.zip chứa games/summary/paired/cycles/windows/cost, hình utility theo10k, chênh lệch C-R/C-N và hậu sạc. benchmark_resume.zip dùng cho phiên sau. Chỉ cần Add Input source Dataset + resume này để chạy tiếp, không cần train lại.

        Utility CI bootstrap theo seed tình huống, sau khi lấy trung bình hai training seed trong từng tình huống. Không coi slot, cycle hoặc hai lần chạy trên cùng trace là các tình huống độc lập. CI này không thay thế nhiều training seed.

        Random chỉ có một cấu hình, mỗi tình huống có seed lấy mẫu riêng đã khóa. Greedy có lợi thế thông tin và nhiều simulator preview, nên báo cả thời gian quyết định và số preview; không gọi là oracle tối ưu.

        CDF hậu sạc chỉ mô tả các cycle đã quan sát phục vụ lại, đi kèm thống kê horizon/next-Return không phục vụ lại; không bỏ censoring hoặc gọi đó là tỷ lệ thành công của mọi cycle.
        """),
    ]
    for name, blocks in [("bs5_return_v4_fresh_train_kaggle.ipynb", train), ("bs5_return_v4_benchmark_kaggle.ipynb", bench)]:
        notebook = dict(cells=[cell(kind, text, i) for i, (kind, text) in enumerate(blocks)],
                        metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python", version="3")),
                        nbformat=4, nbformat_minor=5)
        (directory / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    build(ap.parse_args().root)
