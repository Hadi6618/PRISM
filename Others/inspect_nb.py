import json, sys
sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open("ShanghaiTech_Ensemble_Fusion.ipynb", encoding="utf-8"))
# Print ALL cells to see Cell 7
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    print(f"=== Cell {i} (type={cell['cell_type']}) ===")
    print(src[:600])
    print()
