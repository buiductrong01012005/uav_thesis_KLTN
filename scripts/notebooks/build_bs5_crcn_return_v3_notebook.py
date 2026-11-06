"""Build a clean, stdlib-bootstrap Kaggle notebook for the complete ablation."""
from pathlib import Path
import json,textwrap
ROOT=Path(__file__).resolve().parents[2]
def cell(kind,source,index):
    d=dict(cell_type=kind,id=f'crcn-{index}',metadata={},source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind=='code':d.update(execution_count=None,outputs=[])
    return d
def build():
    blocks=[('markdown','''
    # Return-v3 — giữ C-R60k, train C-N60k và eval ghép cặp10k

    1. Tạo **một Kaggle Dataset mới** từ `BS5_CRCN_RETURN_V3_DATASET.zip`, Add Input.
    2. Import notebook này, dùng **CPU**, bật Internet nếu cần tải thư viện rồi **Run All**.
    3. Dataset đã có C-R/seed551/60k, provenance, prefix gốc và toàn bộ UE tape train/validation C-R. Không cần upload các ZIP kết quả cũ.

    Chỉ C-N train fresh:60k train +72k validation, cùng lịch curriculum/hyperparameters, 111 state/116 obs/10-action head; Relay bị mask trong actor/target và tắt trong router.
    C-R giữ best60k. C-N chọn best tại0/12/24/36/48/60k theo utility toàn1.200 slot. **Không chọn checkpoint trên10k.**

    Eval40 game =2 policy ×(10 trace đã xem +10 trace mới khóa trước) ×10.000 =400k eval slot, một reset/game, epsilon0. UE tape chung được kiểm hash; vị trí UAV, pin và association vẫn do policy. Báo cáo seen/heldout riêng. Một training seed, cùng recipe có fixture thiên về Relay; chưa phải C-N được tối ưu mọi cách.

    Preflight có ledger cap512 diagnostic-train +4096 control/eval, kiểm exact-resume C-N, C-R prefix1.200 ở130001/130009, shared tape, physics và export offline. Không train lại C-R. Giữ reward, trạm sạc, Return-v3/collision-v2; không tự tăng120k.
    ''')]
    blocks.append(('code','''
    from pathlib import Path
    SOURCE_PATH = None  # tự tìm Dataset; chỉ điền nếu Add Input nhiều bản của cùng gói
    RESUME_INPUT_DIR = None  # thư mục export/merged từ phiên trước; xem hướng dẫn cuối notebook
    OUTPUT_ROOT = Path('/kaggle/working/bs5_crcn_return_v3_seed551')
    MAX_SESSION_SECONDS = 7 * 3600  # soft limit của lệnh main, dừng ở game boundary; không phải giới hạn Kaggle
    AUTO_INSTALL_MISSING_DEPS = True  # cài runtime CPU vào thư mục riêng, không thay kernel/notebook
    RUN_MAIN = True
    '''))
    blocks.append(('code','''
    import os,sys,json,zipfile,hashlib,shutil,subprocess,time
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working');WORK.mkdir(parents=True,exist_ok=True)
    MARKER='configs/bs5_crcn_return_v3_pair_v1.json'
    ASSET_MARKER='assets/bs5_return_v3_cr/ASSET_SHA256SUMS.json'
    def safe_extract(z,target):
        target=Path(target).resolve()
        for m in z.infolist():
            dest=(target/m.filename).resolve()
            if not dest.is_relative_to(target) or (m.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe ZIP entry')
        z.extractall(target)
    if SOURCE_PATH is None:
        directories=[p.parent.parent for p in INPUT.rglob(Path(MARKER).name) if p.parent.name=='configs' and (p.parent.parent/ASSET_MARKER).exists()]
        archives=[]
        for f in INPUT.rglob('*.zip'):
            try:
                with zipfile.ZipFile(f) as z:
                    names=z.namelist()
                    if 'uav_thesis/'+MARKER in names and 'uav_thesis/'+ASSET_MARKER in names:archives.append(f)
            except zipfile.BadZipFile:pass
        choices=directories+archives
        if len(choices)!=1:raise RuntimeError('Cần đúng một Dataset đầy đủ; điền SOURCE_PATH từ: '+str(choices))
        SOURCE_PATH=choices[0]
    SOURCE_PATH=Path(SOURCE_PATH);SOURCE_ROOT=WORK/'bs5_crcn_source'/'uav_thesis'
    if not SOURCE_ROOT.exists():
        if SOURCE_PATH.is_dir():shutil.copytree(SOURCE_PATH,SOURCE_ROOT)
        else:
            with zipfile.ZipFile(SOURCE_PATH) as z:
                if z.testzip() is not None:raise RuntimeError('Dataset ZIP CRC failed')
                safe_extract(z,SOURCE_ROOT.parent)
    assert (SOURCE_ROOT/MARKER).exists() and (SOURCE_ROOT/ASSET_MARKER).exists()
    manifest=json.loads((SOURCE_ROOT/'DATASET_MANIFEST.json').read_text())
    for rel,digest in manifest.items():
        if hashlib.sha256((SOURCE_ROOT/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('Dataset checksum failed: '+rel)
    # Never mutate the in-use source when switching Dataset versions.
    if SOURCE_PATH.is_dir():
        if (SOURCE_PATH/'DATASET_MANIFEST.json').read_bytes()!=(SOURCE_ROOT/'DATASET_MANIFEST.json').read_bytes():raise RuntimeError('Source mới khác bản đang dùng; giữ output, dùng working source sạch.')
    else:
        with zipfile.ZipFile(SOURCE_PATH) as z:
            if z.read('uav_thesis/DATASET_MANIFEST.json')!=(SOURCE_ROOT/'DATASET_MANIFEST.json').read_bytes():raise RuntimeError('Dataset ZIP khác source đang dùng; giữ output, dùng working source sạch.')
    print('Dataset verified:',SOURCE_ROOT,'files:',len(manifest))
    '''))
    blocks.append(('code','''
    # All project computation runs in fresh subprocesses, so the kernel can keep its original NumPy/Torch.
    DEPS=WORK/'bs5_crcn_cpu_runtime'
    RUN_ENV=os.environ.copy()
    RUN_ENV.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    RUN_ENV['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE_ROOT)
    probe="import torch,numpy,pytest; assert torch.__version__=='2.11.0+cpu'; assert numpy.__version__=='2.1.3'; assert pytest.__version__=='8.4.2'; print(torch.__version__,numpy.__version__,pytest.__version__)"
    check=subprocess.run([sys.executable,'-B','-c',probe],env=RUN_ENV,capture_output=True,text=True)
    if check.returncode:
        if not AUTO_INSTALL_MISSING_DEPS:raise RuntimeError('Runtime chưa đúng; bật AUTO_INSTALL_MISSING_DEPS. '+check.stderr[-1500:])
        DEPS.mkdir(parents=True,exist_ok=True)
        print('Installing isolated CPU runtime. Internet is needed only if packages are missing.')
        subprocess.check_call([sys.executable,'-m','pip','install','--target',str(DEPS),'--upgrade','--no-cache-dir','torch==2.11.0+cpu','--index-url','https://download.pytorch.org/whl/cpu'])
        subprocess.check_call([sys.executable,'-m','pip','install','--target',str(DEPS),'--upgrade','--no-cache-dir','numpy==2.1.3','pytest==8.4.2'])
        subprocess.check_call([sys.executable,'-B','-c',probe],env=RUN_ENV)
    else:print(check.stdout)
    def run_stream(arguments,log):
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',encoding='utf-8') as f:
            proc=subprocess.Popen([sys.executable,'-B','-u',*map(str,arguments)],cwd=SOURCE_ROOT,env=RUN_ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            return proc.wait()
    print('Backend runtime ready; device=cpu, Torch threads=1.')
    '''))
    blocks.append(('code','''
    # Restore only complete companion sets, preserving all ledgers/RNG/replay/raw.
    if RESUME_INPUT_DIR is not None:
        src=Path(RESUME_INPUT_DIR)
        if OUTPUT_ROOT.exists():raise RuntimeError('Output đã tồn tại; không chép đè khi restore.')
        if (src/'protocol.json').exists() and (src/'cn/resume.pt').exists():shutil.copytree(src,OUTPUT_ROOT)
        else:
            indices=list(src.rglob('EXPORT_INDEX.json'))
            if len(indices)!=1:raise RuntimeError('Chọn thư mục chứa đúng một EXPORT_INDEX.json.')
            index=json.loads(indices[0].read_text());OUTPUT_ROOT.mkdir(parents=True)
            for item in index['files']:
                found=list(src.rglob(item['path']))
                if len(found)!=1:raise RuntimeError('Thiếu/không rõ companion '+item['path'])
                f=found[0]
                if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Companion checksum failed')
                with zipfile.ZipFile(f) as z:
                    if z.testzip() is not None:raise RuntimeError('Companion CRC failed')
                    safe_extract(z,OUTPUT_ROOT)
        assert (OUTPUT_ROOT/'cn/resume.pt').exists()
    PREFLIGHT=WORK/'bs5_crcn_return_v3_preflight'
    if OUTPUT_ROOT.exists() and (OUTPUT_ROOT/'preflight/receipt.json').exists() and not PREFLIGHT.exists():shutil.copytree(OUTPUT_ROOT/'preflight',PREFLIGHT)
    rc=run_stream(['scripts/verification/preflight_bs5_crcn_return_v3.py','--output',PREFLIGHT],WORK/'bs5_crcn_preflight.log')
    if rc:raise RuntimeError('Preflight chưa PASS; main không chạy. Giữ receipt/log để xem lỗi; không tự xóa ledger rồi lặp compute.')
    receipt=json.loads((PREFLIGHT/'receipt.json').read_text());assert receipt['status']=='passed'
    print('Preflight PASS. Charged:',receipt['charged'])
    '''))
    blocks.append(('code','''
    MAIN_RETURN_CODE=0
    if RUN_MAIN:
        if not OUTPUT_ROOT.exists():
            rc=run_stream(['scripts/experiments/run_bs5_crcn_return_v3.py','prepare','--output',OUTPUT_ROOT],WORK/'bs5_crcn_prepare.log')
            if rc:raise RuntimeError('Prepare failed; main not started.')
        saved=OUTPUT_ROOT/'preflight';saved.mkdir(exist_ok=True)
        for src in [PREFLIGHT/'receipt.json',WORK/'bs5_crcn_preflight.log']:
            dest=saved/src.name
            if not dest.exists():shutil.copy2(src,dest)
        cmd=['scripts/experiments/run_bs5_crcn_return_v3.py','run','--output',OUTPUT_ROOT]
        if MAX_SESSION_SECONDS is not None:cmd+=['--max-seconds',str(MAX_SESSION_SECONDS)]
        MAIN_RETURN_CODE=run_stream(cmd,WORK/'bs5_crcn_main.log')
        print('Current status:',json.loads((OUTPUT_ROOT/'status.json').read_text()))
        if MAIN_RETURN_CODE:print('Main stopped with an error. Export preserved committed/partial evidence below; do not reset budgets.')
    else:print('RUN_MAIN=False: only bounded preflight ran.')
    '''))
    blocks.append(('code','''
    EXPORT=None
    if RUN_MAIN and (OUTPUT_ROOT/'protocol.json').exists():
        before=set(WORK.glob(OUTPUT_ROOT.name+'_export*'))
        rc=run_stream(['scripts/experiments/run_bs5_crcn_return_v3.py','export','--output',OUTPUT_ROOT],WORK/'bs5_crcn_export.log')
        if rc:raise RuntimeError('Export chưa hoàn tất. Giữ nguyên OUTPUT_ROOT và log; không mất dữ liệu main.')
        created=[p for p in WORK.glob(OUTPUT_ROOT.name+'_export*') if p not in before and (p/'EXPORT_INDEX.json').exists()]
        assert len(created)==1;EXPORT=created[0]
        from IPython.display import display,Markdown,FileLink
        display(Markdown((EXPORT/'analysis/REPORT_VI.md').read_text()))
        print('Download review_small.zip + ALL audit_light ZIPs + EXPORT_INDEX.json for analysis.')
        print('Keep ALL repro_full ZIPs too, with audit companions, for resume/frozen evaluation.')
        for f in sorted(EXPORT.iterdir()):
            if f.is_file():display(FileLink(str(f)))
    if MAIN_RETURN_CODE:raise RuntimeError('Main failed; evidence was exported. Preserve all files and send status/ledger/log.')
    '''))
    blocks.append(('markdown','''
    ## Tiếp tục và đọc kết quả

    - `complete`: đã train C-N60k, validation72k và đủ40 game ghép cặp10k.
    - `paused_at_game_boundary`: đã dừng an toàn. Giữ mọi companion export. Phiên sau dùng cùng Dataset/runtime, đặt RESUME_INPUT_DIR rồi Run All; không train lại các game đã commit.
    - Nén toàn bộ thư mục export vào một ZIP ngoài trước khi tạo Dataset resume, giữ các companion ZIP bên trong. Add Input cả Dataset source và resume Dataset.
    - `partial_error`: có thể có compute chưa commit. Ledger chặn lặp lại âm thầm. Giữ output và gửi log để kiểm tra, không sửa/xóa counter để chạy lại.
    - `review_small.zip` có report/bảng. `aggregate.csv` tách seen/heldout; `paired.csv` có chênh lệch từng trace; `cycles.json` giữ censoring. Mỗi phase có ledger riêng. C-N Relay-empty là N/A.
    - Analyzer mặc định kiểm hash/tape/mask/continuity/energy/endpoint, không chạy policy. `--deep-planner` thêm BFS độc lập trên từng snapshot, tốn thời gian hơn.
    - Dataset này chỉ chạy C-R/C-N. Các heuristic baseline và adapter preview là bước benchmark riêng; không dùng bảng baseline env-v1 để tuyên bố thắng ởv3.
    '''))
    notebook=dict(cells=[cell(kind,text,i) for i,(kind,text) in enumerate(blocks)],metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3')),nbformat=4,nbformat_minor=5)
    path=ROOT/'scripts/notebooks/bs5_crcn_return_v3_kaggle.ipynb';path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
    return path
if __name__=='__main__':print(build())
