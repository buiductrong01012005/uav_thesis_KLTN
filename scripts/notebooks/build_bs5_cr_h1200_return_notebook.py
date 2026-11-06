"""Build Return-v3 notebook, preserving the proven v2 transport/bootstrap flow."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
book=json.loads((HERE/'bs5_cr_h1200_collision_v2_kaggle.ipynb').read_text())
def source(cell,text):
    cell['source']=text.splitlines(True)
    if cell['cell_type']=='code':cell.update(outputs=[],execution_count=None)
for cell in book['cells']:
    s=''.join(cell['source']).replace('collision_v2','return_v3').replace('collision-v2','Return-v3')
    s=s.replace('run_bs5_cr_h1200_collision','run_bs5_cr_h1200_return')
    source(cell,s)
source(book['cells'][0],'''# C-R60k — Return-v3, S4 dài 1.200 slot

1. Tạo **Dataset mới** bằng ZIP source Return-v3, Add Input, chọn CPU rồi Run All.
2. Giữ map BS5, launch B, ba vị trí trạm, công suất sạc, reward, QMIX và curriculum.
3. Returning tìm đường quanh ô đang bị chiếm, ngưỡng về sạc tính theo đường hiện tại.
4. Không đủ năng lượng: UAV đứng lại, không radio và vẫn chiếm ô. Game tiếp tục; không tự cứu.
5. Trạm cho phép tất cả UAV cùng sạc, mỗi UAV vẫn nhận 5 pin/slot; không có queue sạc.

**Đây là train fresh một C-R/seed551, 60k train +72k validation. Không nối checkpoint v2.**
S1 0–3k, S2 3–6k, S3 6–12k: horizon150; S4 12–60k:40 game×1200.
Validation0/12/24/36/48/60k, mỗi mốc10 game×1200. Không tự mở rộng120k.

Preflight có bộ đếm bền vững, cap512 diagnostic-train +4096 control/eval; main seed không chạy trong preflight.
Run có raw từ slot đầu, resume state ở game boundary, export một lần cuối phiên.
`MAX_SESSION_SECONDS` mặc định7 giờ, dừng ở game boundary; đây không phải cam kết thời lượng Kaggle.

Đường có thể bị chặn hoàn toàn hoặc thay đổi ở slot sau. Bản sửa không bảo đảm mọi UAV luôn về được trạm.
Khi chưa có đường, ngưỡng dùng khoảng cách hình học làm lower bound và ghi rõ no-route; không kích Return ngay chỉ vì UAV đầy pin tạm bị vây ở launch.
''')
source(book['cells'][3],'''from scripts.training import run_bs5_cr_h1200_return as runner
runner.setup_runtime(DEVICE,1)
identity=runner.source_identity()
PREFLIGHT=WORK/"bs5_return_v3_preflight"
receipt=PREFLIGHT/"receipt.json"
fingerprint={"source":identity["sha256"],"runtime":runner.runtime(DEVICE)}
if receipt.exists():
    previous=json.loads(receipt.read_text())
    if previous.get("fingerprint")!=fingerprint or previous.get("status")!="passed":
        raise RuntimeError("Preflight khác source/runtime hoặc chưa PASS. Giữ log; không tự chạy lại compute.")
    print("Dùng preflight PASS cùng source/runtime:",previous["charged"])
elif RUN_PREFLIGHT:
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,"-B","scripts/verification/preflight_bs5_return.py",
        "--output",str(PREFLIGHT),"--device",DEVICE],cwd=SOURCE_ROOT,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (PREFLIGHT/"pytest.log").write_text(result.stdout)
    print(result.stdout)
    if result.returncode:raise RuntimeError("Preflight thất bại; giữ log, main chưa chạy.")
    previous=json.loads(receipt.read_text())
    assert previous["status"]=="passed" and previous["fingerprint"]==fingerprint
else:
    raise RuntimeError("Cần preflight PASS trước main; bật RUN_PREFLIGHT=True.")
''')
s=''.join(book['cells'][4]['source'])
s=s.replace('assert p.get("collision_version")=="bs5_endpoint_return_v3", "Không resume env cũ bằng runner Return-v3"',
    'assert p.get("return_version")=="bs5_energy_return_v3", "Không resume env cũ bằng runner Return-v3"')
# The collision identifier is deliberately still v2; physics lineage is Return-v3.
s=s.replace('assert p.get("collision_version")=="bs5_endpoint_collision_v2", "Không resume env cũ bằng runner Return-v3"',
    'assert p.get("return_version")=="bs5_energy_return_v3", "Không resume env cũ bằng runner Return-v3"')
source(book['cells'][4],s)
s=''.join(book['cells'][6]['source'])+'\nReturn-v3: xem `return_wait_uav_slots`, `return_no_route_uav_slots`, `return_detour_uav_slots`, `return_stranded_uav_slots`, `return_prevented_energy_moves`. Analyzer độc lập kiểm đường, năng lượng, lifecycle và collision từ raw; không cần checkpoint hay NumPy/Torch. Giữ các run cũ làm tham chiếu; không suy ra lợi ích chính sách từ fixture điều khiển cố định.\n'
source(book['cells'][6],s)
for i,cell in enumerate(book['cells']):cell['id']=f'bs5-return-v3-{i:02d}'
path=HERE/'bs5_cr_h1200_return_v3_kaggle.ipynb'
path.write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
print(path)
