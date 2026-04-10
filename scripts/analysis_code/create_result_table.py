import pandas as pd

df = pd.read_csv("/home/cudagl/data/RAS_results/summary_grasp_results.csv")
# 全ロボット平均の pivot（object × method）
pivot_all = df.pivot_table(index="object", columns="method", values="success_rate", aggfunc="mean")
print(pivot_all.fillna("-").to_string())

# ロボット 'panda' の場合
pivot_panda = df[df.robot=="panda"].pivot_table(index="object", columns="method", values="success_rate")
print(pivot_panda.fillna("-").to_string())

# LaTeX 出力（論文用）
with open("/home/cudagl/data/RAS_results/grasp_table.tex","w") as f:
    f.write(pivot_all.fillna("-").to_latex(float_format="%.2f"))
