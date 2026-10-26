"""One Dataset, four independent jobs (or two arm jobs), then checked merge."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import textwrap
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.notebooks.build_bs5_v4_fresh_notebooks import build as build_base, BOOTSTRAP, cell


def write(path, notebook):
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), str(path), "exec")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def build(root):
    build_base(root)
    folder = root / "scripts/notebooks"
    base = json.loads((folder / "bs5_return_v4_fresh_train_kaggle.ipynb").read_text())
    for arm in ["C-R", "C-N"]:
        for seeds in [[551], [552], [551, 552]]:
            label = arm.replace("-", "") + "_" + "_".join(map(str, seeds))
            nb = deepcopy(base)
            nb["cells"][0] = cell("markdown", f"""
            # Return-v4 parallel training — {arm}, seeds {seeds}

            Add Input **BS5_RETURN_V4_PARALLEL_DATASET.zip**, chọn GPU rồi Run All.
            Mọi notebook dùng CHUNG một source Dataset; output riêng cho `{label}`.
            Notebook này chỉ train {len(seeds)} run đã gán: mỗi run60k train +66k validation.
            Các run khác chạy ở notebook khác; không chờ và không đọc output của nhau khi train.

            Mặc định dùng4 notebook `CR_551`, `CR_552`, `CN_551`, `CN_552` song song nếu tài khoản được cấp đủ phiên GPU.
            Nếu chỉ có2 phiên, dùng `CR_551_552` và `CN_551_552`; mỗi notebook train hai seed tuần tự.
            Chỉ chọn MỘT cách chia, không chạy cả hai cách vì sẽ trùng job.

            Source giữ nguyên env Return-v4, learner, reward, curriculum và lịch seed/UE trace.
            GPU chỉ giúp một phần; tốc độ/giới hạn phiên phụ thuộc tài nguyên thực tế.
            Khởi chạy sẽ báo lỗi sớm nếu GPU hoặc PyTorch CUDA chưa sẵn sàng, không âm thầm dùng CPU.
            """, 0)
            nb["cells"][1] = cell("code", f"""
            from pathlib import Path
            import time
            SESSION_START=time.monotonic()
            SOURCE_PATH=None
            RESUME_INPUT_DIR=None
            TRAIN_ARMS={[arm]!r}
            TRAINING_SEEDS={seeds!r}
            JOB_LABEL={label!r}
            OUTPUT_ROOT=Path('/kaggle/working/bs5_v4_train_{label}')
            DEVICE='cuda'
            REQUIRE_GPU=True  # CPU có chủ đích: đặt False và DEVICE='cpu' TRƯỚC prepare
            MAX_TOTAL_SESSION_SECONDS=10.5*3600
            EXPORT_RESERVE_SECONDS=30*60
            RUN_MAIN=True
            """, 1)
            boot = BOOTSTRAP + "\nruntime_info=json.loads(check.stdout)\nif REQUIRE_GPU and not runtime_info['cuda']:\n    raise RuntimeError('GPU chưa sẵn sàng hoặc đang dùng PyTorch +cpu. Bật Accelerator GPU và dùng runtime Kaggle có Torch CUDA trước khi train; không đổi Torch/device giữa chừng khi resume.')\n"
            nb["cells"][2] = cell("code", boot, 2)
            nb["cells"][3] = cell("code", """
            RESUME_NAME='training_resume__'+JOB_LABEL+'.zip'
            if RESUME_INPUT_DIR is None and not OUTPUT_ROOT.exists():
                archives=[x for x in INPUT.rglob(RESUME_NAME) if is_main_archive(x,'bs5_return_v4_fresh_crcn_v1')]
                directories=[]
                for f in INPUT.rglob('protocol.json'):
                    try:
                        p=json.loads(f.read_text())
                        if p.get('version')=='bs5_return_v4_fresh_crcn_v1' and not p.get('diagnostic',False) and p.get('arms')==TRAIN_ARMS and p.get('training_seeds')==TRAINING_SEEDS and list((f.parent/'runs').glob('*/resume/*.pt')): directories.append(f.parent)
                    except (OSError,ValueError): pass
                choices=archives if archives else directories
                if len(choices)>1: raise RuntimeError('Có nhiều resume cùng job; điền RESUME_INPUT_DIR rõ ràng.')
                if choices: RESUME_INPUT_DIR=choices[0]
            if RESUME_INPUT_DIR is not None and not OUTPUT_ROOT.exists(): restore_archive_or_directory(RESUME_INPUT_DIR,OUTPUT_ROOT,RESUME_NAME)
            if OUTPUT_ROOT.exists():
                saved=json.loads((OUTPUT_ROOT/'protocol.json').read_text())
                assert saved['arms']==TRAIN_ARMS and saved['training_seeds']==TRAINING_SEEDS,'Resume thuộc job khác.'
            PREFLIGHT=WORK/('preflight_'+JOB_LABEL)
            rc=run_stream(['scripts/verification/preflight_bs5_v4_partition.py','--output',PREFLIGHT,'--arms',','.join(TRAIN_ARMS),'--seeds',','.join(map(str,TRAINING_SEEDS)),'--device',DEVICE],WORK/('preflight_'+JOB_LABEL+'.log'))
            if rc: raise RuntimeError('Preflight failed; main chưa chạy. Giữ log/receipt.')
            receipt=json.loads((PREFLIGHT/'receipt.json').read_text())
            assert receipt['status']=='passed'
            print('Preflight PASS:',receipt['identity']['runtime']['device'],'— không chạy benchmark; chỉ dùng seed chẩn đoán riêng.')
            """, 3)
            main = "".join(base["cells"][4]["source"])
            main = main.replace("'--device',DEVICE]", "'--device',DEVICE,'--arms',','.join(TRAIN_ARMS),'--seeds',','.join(map(str,TRAINING_SEEDS))]")
            main = main.replace("WORK/'bs5_v4_preflight.log'", "WORK/('preflight_'+JOB_LABEL+'.log')")
            nb["cells"][4] = cell("code", main, 4)
            nb["cells"][6] = cell("markdown", f"""
            ## Output riêng cho {label}

            - `review_small.zip`: report, learning curve, raw-bin/episode/update/validation tables.
            - `training_resume__{label}.zip`: resume ĐÚNG job này; giữ đủ optimizer/replay/RNG. Phiên sau Add Input source Dataset và gói resume này.
            - `training_for_benchmark__{label}.zip`: chỉ xuất khi các run ĐƯỢC GÁN đã hoàn tất. Có best/last, lựa chọn validation, curve CSV và hash UE tape. Không dùng để resume training.
            - Lưu notebook version kèm output. Không trộn gói này với output source Dataset cũ.
            - Runtime, device và loại GPU phải giữ nguyên trong cùng một job khi resume. Hai job độc lập có thể dùng hai GPU model khác nhau, được ghi đầy đủ; Python/Torch/NumPy stack phải khớp để ghép so sánh.
            - Khi đủ bốn job, Add Input các gói `training_for_benchmark__*.zip` vào notebook `bs5_return_v4_parallel_benchmark_kaggle.ipynb`. Nó kiểm đủ job, recipe, initial weights và UE hashes rồi ghép learning curve trước benchmark.
            - Không có test10k trong notebook train này. Không dùng test để chọn checkpoint. S1/S2 là fixture inherited dạy Relay; C-N là đối chứng cùng recipe.
            """, 6)
            write(folder / f"bs5_return_v4_train_{label}_kaggle.ipynb", nb)
    bench = json.loads((folder / "bs5_return_v4_benchmark_kaggle.ipynb").read_text())
    bench["cells"][0] = cell("markdown", """
    # Return-v4 benchmark — ghép4 job train độc lập, không train thêm

    Add Input **source Dataset mới** và tất cả gói `training_for_benchmark__*.zip`.
    Có thể là4 gói đơn,2 gói theo nhánh hoặc tập hợp không trùng bao phủ đủ4 job.
    Dùng CPU, Run All. Không cần tải cả raw/replay training vào đây.

    Merge kiểm source/recipe, runtime stack, đủ C-R/C-N×551/552, initial weights, schedules,
    actual UE tape hashes, checkpoint SHA và lựa chọn validation. Không chọn lại trên test.
    Ghép curve CSV và dựng learning curve chung; giữ provenance/runtime riêng từng job.

    Sau đó mới benchmark9 cấu hình/7 phương pháp, cùng10 seed tình huống ×10k, tổng900k slot.
    Greedy privileged exact current UE/network được ghi riêng chi phí preview. Một reset/game.
    Có benchmark_resume.zip để chạy qua nhiều phiên. Không cam kết hoàn tất trong12h.
    """, 0)
    settings = "".join(bench["cells"][1]["source"]).replace("TRAINING_INPUT_DIR=None  # tự tìm training_for_benchmark.zip hoặc training output complete", "TRAINING_INPUTS=None  # hoặc danh sách đường dẫn tới mọi ZIP/thư mục handoff đã complete")
    bench["cells"][1] = cell("code", settings, 1)
    bench["cells"][3] = cell("code", """
    if RESUME_INPUT_DIR is None and not OUTPUT_ROOT.exists():
        archives=[x for x in INPUT.rglob('benchmark_resume.zip') if is_main_archive(x,'bs5_return_v4_benchmark_v1')]
        directories=[]
        for f in INPUT.rglob('protocol.json'):
            try:
                p=json.loads(f.read_text())
                if p.get('version')=='bs5_return_v4_benchmark_v1' and not p.get('diagnostic',False): directories.append(f.parent)
            except (ValueError,OSError): pass
        choices=archives if archives else directories
        if len(choices)>1: raise RuntimeError('Chọn RESUME_INPUT_DIR cụ thể.')
        if choices: RESUME_INPUT_DIR=choices[0]
    if RESUME_INPUT_DIR is not None and not OUTPUT_ROOT.exists(): restore_archive_or_directory(RESUME_INPUT_DIR,OUTPUT_ROOT,'benchmark_resume.zip')
    TRAINING_ROOT=WORK/'bs5_v4_merged_training'
    if not OUTPUT_ROOT.exists() and not TRAINING_ROOT.exists():
        if TRAINING_INPUTS is None:
            archives=[x for x in INPUT.rglob('training_for_benchmark__*.zip') if is_main_archive(x,'bs5_return_v4_fresh_crcn_v1')]
            covered=set()
            for x in archives:
                with zipfile.ZipFile(x) as z:
                    names=[n for n in z.namelist() if len(Path(n).parts)==2 and Path(n).name=='protocol.json']
                    p=json.loads(z.read(names[0]));covered.update((a,s) for a in p['arms'] for s in p['training_seeds'])
            directories=[]
            for f in INPUT.rglob('protocol.json'):
                try:
                    p=json.loads(f.read_text());jobs={(a,s) for a in p.get('arms',[]) for s in p.get('training_seeds',[])}
                    if p.get('version')=='bs5_return_v4_fresh_crcn_v1' and not p.get('diagnostic',False) and (f.parent/'HANDOFF_SHA256SUMS.json').exists() and jobs and not jobs<=covered: directories.append(f.parent)
                except (ValueError,OSError): pass
            TRAINING_INPUTS=archives+directories
        if not TRAINING_INPUTS: raise RuntimeError('Add Input đủ các gói handoff từ train trước.')
        rc=run_stream(['scripts/experiments/merge_bs5_v4_training.py','--output',TRAINING_ROOT,'--inputs',*TRAINING_INPUTS],WORK/'bs5_v4_merge.log')
        if rc: raise RuntimeError('Merge chưa hợp lệ; main benchmark không chạy. Giữ log, không chọn lại checkpoint.')
        from IPython.display import display,Image
        display(Image(filename=str(TRAINING_ROOT/'analysis/learning_curves.png')))
    if TRAINING_ROOT.exists(): assert json.loads((TRAINING_ROOT/'status.json').read_text())['status']=='complete'
    PREFLIGHT=WORK/'bs5_v4_benchmark_profile'
    rc=run_stream(['scripts/verification/profile_bs5_v4_benchmark.py','--output',PREFLIGHT],WORK/'bs5_v4_benchmark_profile.log')
    if rc: raise RuntimeError('CPU control/profile failed; final benchmark chưa chạy.')
    """, 3)
    write(folder / "bs5_return_v4_parallel_benchmark_kaggle.ipynb", bench)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    build(ap.parse_args().root)
