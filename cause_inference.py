# -*- coding = utf-8 -*-
# @Time : 2024/10/7 16:12
# @Author : 小五
# @File : cause_inference.py
# @Sofeware : PyCharm
import pandas as pd
import logging, warnings
from dowhy import CausalModel

# 设置日志等级
logging.getLogger().setLevel(logging.ERROR)
# 忽略所有警告
warnings.filterwarnings("ignore")
# 读取Excel数据
data_path = 'data.xlsx'
data = pd.read_excel(data_path, sheet_name='Sheet1', engine='openpyxl')

# 删除G0为0的行
# data = data[data['G0'] != 1]

# 定义因果图
causal_graph_common = """
digraph {
A[label="劳动者何时达到法定退休年龄-状态"];
B[label="有无书面合同-状态"];
C[label="有无享受养老保险待遇-状态"];
D[label="是否认定为基本养老保险待遇-状态"];
E[label="劳动者性别-状态"];
F[label="养老保险待遇类型-类型"];
G[label="审理法院的对应裁判结果-状态"];
C -> G;
A -> G;F -> G; C -> F; F -> D;
B -> G;
D -> G;
E -> G;
}
"""
causal_graph_indicator = """
digraph {
A0[label="指示劳动者何时达到法定退休年龄-状态"];
A[label="劳动者何时达到法定退休年龄-状态"];
B0[label="指示有无书面合同-状态"];
B[label="有无书面合同-状态"];
C0[label="指示有无享受养老保险待遇-状态"];
C[label="有无享受养老保险待遇-状态"];
D0[label="指示是否认定为基本养老保险待遇-状态"];
D[label="是否认定为基本养老保险待遇-状态"];
E0[label="指示劳动者性别-状态"];
E[label="劳动者性别-状态"];
F0[label="指示养老保险待遇类型-类型"];
F[label="养老保险待遇类型-类型"];
G[label="审理法院的对应裁判结果-状态"];
A0 -> G;A0 -> A;
A -> G;
B0 -> G;B0 -> B;
B -> G;
C0 -> G;C0 -> C;
C -> G;C -> F0;C -> F;
F0 -> G;F0 -> F;
F -> G;F -> D0;F -> D;
D0 -> G;D0 -> D;
D -> G;
E0 -> G;E0 -> E;
E -> G;
}
"""

# 变量组合列表
treatment_outcome_pairs = [
    ('E', 'G'),
    ('A', 'G'),
    ('B', 'G'),
    ('C', 'F'),
    ('C', 'G'),
    ('F', 'D'),
    ('F', 'G'),
    ('D', 'G')
]

# 循环处理每对变量，进行ATE估计及反驳测试
for treatment, outcome in treatment_outcome_pairs:
    print(f"\n### 探究 {treatment} 对 {outcome} 的影响 ###")

    # 构建因果模型
    model = CausalModel(
        data=data,
        graph=causal_graph_indicator,
        treatment=treatment,  # 处理变量
        outcome=outcome  # 结果变量
    )

    # 估计ATE
    estimands = model.identify_effect()
    estimate = model.estimate_effect(estimands, method_name="backdoor.linear_regression", target_units="ate")
    print(f"Causal Estimate for {treatment} -> {outcome} is: {estimate.value}")

    # 反驳测试1: Random Common Cause
    refute1_results = model.refute_estimate(estimands, estimate, method_name="random_common_cause")
    print(f"Random Common Cause Refutation for {treatment} -> {outcome} p-value: {refute1_results.refutation_result['p_value']}")

    # 反驳测试2: Placebo Treatment Refuter
    refute2_results = model.refute_estimate(estimands, estimate, method_name="placebo_treatment_refuter")
    print(f"Placebo Treatment Refutation for {treatment} -> {outcome} p-value: {refute2_results.refutation_result['p_value']}")
