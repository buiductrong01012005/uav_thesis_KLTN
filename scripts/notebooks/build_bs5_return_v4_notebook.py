from pathlib import Path

if __package__:
    from ._notebook import make_notebook, normalized_cell, write_notebook
else:
    from _notebook import make_notebook, normalized_cell, write_notebook
ROOT = Path(__file__).resolve().parents[2]


def build(output_directory=None):
    blocks = [
        (
            "markdown",
            """
 # Return-v4: eval lại C-R60k / C-N36k, không train

 Tạo Kaggle Dataset từ **BS5_RETURN_V4_PAIR_DATASET.zip**, import notebook này, Add Input Dataset, dùng **CPU**, bật Internet nếu cần thư viện và **Run All**. Không cần upload các ZIP audit cũ.

 Dataset có2 checkpoint đã xác minh,20 exact UE tape10k và kết quả v3 để đối chiếu. Chỉ controller Return đổi: giữ route khi next hop trống, replan khi bị chặn, tránh ô vừa rời nếu có đường khác. Reward, energy/margin, charger, collision-v2, router và actor inputs giữ nguyên.

 Mặc định40 game/400k eval slot, **train=0**, epsilon0, một reset/game, weights frozen. Giữ tên nhóm seen/previously-heldout để so cùng trace; đây là controller sensitivity trên các trace đã xem, chưa phải fresh untouched test hoặc xác nhận nhiều training seed.
 """,
        )
    ]
    blocks.append(
        (
            "code",
            """
 from pathlib import Path
 SOURCE_PATH = None
 RESUME_INPUT_DIR = None  # thư mục export chứa EXPORT_INDEX.json + mọi evidence ZIP
 OUTPUT_ROOT = Path('/kaggle/working/bs5_return_v4_pair_seed551')
 MAX_SESSION_SECONDS = 7 * 3600  # soft limit main, dừng ở game boundary; chừa thời gian export
 AUTO_INSTALL_MISSING_DEPS = True
 RUN_EVAL = True
 """,
        )
    )
    blocks.append(
        (
            "code",
            """
 import os,sys,json,zipfile,hashlib,shutil,subprocess
 os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
 WORK=Path('/kaggle/working');INPUT=Path('/kaggle/input');WORK.mkdir(parents=True,exist_ok=True)
 MARKER='configs/bs5_return_v4_pair.json';LOCK='assets/bs5_return_v4_pair/LOCK.json'
 def safe_extract(z,dest):
     dest=Path(dest).resolve()
     for m in z.infolist():
         if not (dest/m.filename).resolve().is_relative_to(dest) or (m.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe ZIP path')
     z.extractall(dest)
 if SOURCE_PATH is None:
     dirs=[f.parent.parent for f in INPUT.rglob(Path(MARKER).name) if (f.parent.parent/LOCK).exists()]
     archives=[]
     for f in INPUT.rglob('*.zip'):
         try:
             with zipfile.ZipFile(f) as z:
                 if 'uav_thesis/'+MARKER in z.namelist() and 'uav_thesis/'+LOCK in z.namelist():archives.append(f)
         except zipfile.BadZipFile:pass
     options=dirs+archives
     if len(options)!=1:raise RuntimeError('Cần đúng một Dataset v4; điền SOURCE_PATH nếu có nhiều bản: '+str(options))
     SOURCE_PATH=options[0]
 SOURCE_PATH=Path(SOURCE_PATH);SOURCE_ROOT=WORK/'bs5_return_v4_source'/'uav_thesis'
 if not SOURCE_ROOT.exists():
     if SOURCE_PATH.is_dir():shutil.copytree(SOURCE_PATH,SOURCE_ROOT)
     else:
         with zipfile.ZipFile(SOURCE_PATH) as z:
             if z.testzip() is not None:raise RuntimeError('Dataset CRC failed')
             safe_extract(z,SOURCE_ROOT.parent)
 data=(SOURCE_ROOT/'DATASET_MANIFEST.json').read_bytes()
 if SOURCE_PATH.is_dir():expected=(SOURCE_PATH/'DATASET_MANIFEST.json').read_bytes()
 else:
     with zipfile.ZipFile(SOURCE_PATH) as z:expected=z.read('uav_thesis/DATASET_MANIFEST.json')
 assert data==expected,'Dataset khác source đang dùng; giữ output và dùng source sạch'
 for rel,digest in json.loads(data).items():assert hashlib.sha256((SOURCE_ROOT/rel).read_bytes()).hexdigest()==digest,rel
 print('Verified v4 Dataset:',SOURCE_ROOT)
 """,
        )
    )
    blocks.append(
        (
            "code",
            """
 DEPS=WORK/'bs5_return_v4_cpu_runtime'
 ENV=os.environ.copy();ENV.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 ENV['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE_ROOT)
 probe="import torch,numpy,pytest; assert torch.__version__=='2.11.0+cpu'; assert numpy.__version__=='2.1.3'; assert pytest.__version__=='8.4.2'; print(torch.__version__,numpy.__version__,pytest.__version__)"
 check=subprocess.run([sys.executable,'-B','-c',probe],env=ENV,capture_output=True,text=True)
 if check.returncode:
     if not AUTO_INSTALL_MISSING_DEPS:raise RuntimeError(check.stderr)
     DEPS.mkdir(parents=True,exist_ok=True)
     subprocess.check_call([sys.executable,'-m','pip','install','--target',str(DEPS),'--upgrade','--no-cache-dir','torch==2.11.0+cpu','--index-url','https://download.pytorch.org/whl/cpu'])
     subprocess.check_call([sys.executable,'-m','pip','install','--target',str(DEPS),'--upgrade','--no-cache-dir','numpy==2.1.3','pytest==8.4.2'])
     subprocess.check_call([sys.executable,'-B','-c',probe],env=ENV)
 else:print(check.stdout)
 def run_stream(args,log):
     with Path(log).open('a',encoding='utf-8') as f:
         p=subprocess.Popen([sys.executable,'-B','-u',*map(str,args)],cwd=SOURCE_ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
         for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
         return p.wait()
 """,
        )
    )
    blocks.append(
        (
            "code",
            """
 if RESUME_INPUT_DIR is not None:
     src=Path(RESUME_INPUT_DIR)
     if OUTPUT_ROOT.exists():raise RuntimeError('Không chép đè output hiện có; cùng phiên hãy đặt RESUME_INPUT_DIR=None')
     if (src/'commit.json').exists() and (src/'budget_ledger.json').exists():shutil.copytree(src,OUTPUT_ROOT)
     else:
         indices=list(src.rglob('EXPORT_INDEX.json'))
         if len(indices)!=1:raise RuntimeError('Chọn thư mục chứa đúng một EXPORT_INDEX.json')
         index=json.loads(indices[0].read_text());OUTPUT_ROOT.mkdir(parents=True)
         for item in index['files']:
             found=list(src.rglob(item['path']))
             if len(found)!=1:raise RuntimeError('Thiếu/không rõ part '+item['path'])
             f=found[0];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
             with zipfile.ZipFile(f) as z:
                 assert z.testzip() is None;safe_extract(z,OUTPUT_ROOT)
 PREFLIGHT=WORK/'bs5_return_v4_preflight'
 if OUTPUT_ROOT.exists() and (OUTPUT_ROOT/'preflight/receipt.json').exists() and not PREFLIGHT.exists():shutil.copytree(OUTPUT_ROOT/'preflight',PREFLIGHT)
 rc=run_stream(['scripts/verification/preflight_bs5_return_v4.py','--output',PREFLIGHT],WORK/'return_v4_preflight.log')
 if rc:raise RuntimeError('Preflight chưa PASS; main chưa chạy. Giữ receipt/log, không xóa ledger để lặp compute.')
 print('Preflight PASS:',json.loads((PREFLIGHT/'receipt.json').read_text()))
 """,
        )
    )
    blocks.append(
        (
            "code",
            """
 MAIN_RETURN_CODE=0
 if RUN_EVAL:
     if not OUTPUT_ROOT.exists():
         rc=run_stream(['scripts/experiments/run_bs5_return_v4_pair.py','prepare','--output',OUTPUT_ROOT],WORK/'return_v4_prepare.log')
         if rc:raise RuntimeError('Prepare failed')
     saved=OUTPUT_ROOT/'preflight';saved.mkdir(exist_ok=True)
     for src in [PREFLIGHT/'receipt.json',WORK/'return_v4_preflight.log']:
         if not (saved/src.name).exists():shutil.copy2(src,saved/src.name)
     cmd=['scripts/experiments/run_bs5_return_v4_pair.py','run','--output',OUTPUT_ROOT]
     if MAX_SESSION_SECONDS is not None:cmd+=['--max-seconds',str(MAX_SESSION_SECONDS)]
     MAIN_RETURN_CODE=run_stream(cmd,WORK/'return_v4_main.log')
     print('Status:',json.loads((OUTPUT_ROOT/'status.json').read_text()))
     if MAIN_RETURN_CODE:print('Có lỗi; vẫn export evidence bên dưới. Giữ mọi ledger và log.')
 else:print('RUN_EVAL=False: chỉ chạy bounded preflight')
 """,
        )
    )
    blocks.append(
        (
            "code",
            """
 if RUN_EVAL and (OUTPUT_ROOT/'protocol.json').exists():
     before=set(WORK.glob(OUTPUT_ROOT.name+'_export*'))
     rc=run_stream(['scripts/experiments/run_bs5_return_v4_pair.py','export','--output',OUTPUT_ROOT],WORK/'return_v4_export.log')
     if rc:raise RuntimeError('Export chưa xong; giữ nguyên OUTPUT_ROOT và log')
     found=[p for p in WORK.glob(OUTPUT_ROOT.name+'_export*') if p not in before and (p/'EXPORT_INDEX.json').exists()];assert len(found)==1
     EXPORT=found[0]
     from IPython.display import display,Markdown,FileLink
     display(Markdown((EXPORT/'analysis/REPORT_VI.md').read_text()))
     print('Gửi lại review_small.zip + EXPORT_INDEX.json + mọi evidence_part*.zip. Không cần các audit ZIP cũ.')
     for f in sorted(EXPORT.iterdir()):
         if f.is_file():display(FileLink(str(f)))
 if MAIN_RETURN_CODE:raise RuntimeError('Main failed; đã giữ/export evidence. Không reset budget để lặp game.')
 """,
        )
    )
    blocks.append(
        (
            "markdown",
            """
 ## Tiếp tục / kiểm sâu

 - `complete`: đủ40game/400k eval, train=0. Đọc before_after.csv để so v3/v4 theo từng policy/trace.
 - `paused_at_game_boundary`: giữ mọi evidence part+index; phiên sau dùng cùng Dataset/runtime và RESUME_INPUT_DIR. Không lặp game đã commit.
 - Để tạo Dataset resume, nén nguyên thư mục export thành một ZIP ngoài, giữ các evidence ZIP bên trong. Add Input cả source Dataset lẫn resume Dataset.
 - `partial_error`: có thể đã dùng compute chưa commit. Giữ output; không xóa/sửa ledger hoặc lặp in-flight game âm thầm.
 - Export compact giữ exact float32 actor inputs, float64 rates và đầy đủ Return/collision telemetry. UE tape nằm trong input Dataset với hash khóa, giúp giảm số ZIP so bộ trước.
 - Analyzer offline: `python -B MERGED/bs5_return_v4_results.py --root MERGED --output ANALYSIS --deep-planner --require-complete`. Chỉ NumPy+stdlib, không Torch/env/rollout.
 - Giữ QoS4Mbps và censoring trong so sánh; không chỉ chọn utility. Sửa ca đã biết không chứng minh mọi cấu hình động đều an toàn.
 """,
        )
    )
    cells = [normalized_cell(kind, text, f"return-v4-{i}") for i, (kind, text) in enumerate(blocks)]
    notebook = make_notebook(cells)
    directory = (
        Path(output_directory) if output_directory is not None else ROOT / "scripts/notebooks"
    )
    return write_notebook(directory / "bs5_return_v4_pair_kaggle.ipynb", notebook)


if __name__ == "__main__":
    print(build())
