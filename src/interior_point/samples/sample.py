import pulp
from lib.modeling import LpProblem, LpVariable_dicts, LPStandardForm
import time

# Helpers: convert LPStandardForm to solver inputs and run PuLP
def run_pulp_on_sf(sf: LPStandardForm):
    prob = pulp.LpProblem("LP_from_standard_form", pulp.LpMinimize)
    # create variables
    vars_p = {name: pulp.LpVariable(name, lowBound=0.0) for name in sf.var_names}
    # objective
    prob += pulp.lpSum(sf.c[i] * vars_p[sf.var_names[i]] for i in range(len(sf.c)))
    # equality constraints
    for row, b in zip(sf.A_eq, sf.b_eq):
        prob += pulp.lpSum(row[j] * vars_p[sf.var_names[j]] for j in range(len(row))) == b
    # solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective)
    sol = {name: float(pulp.value(v)) for name, v in vars_p.items() if pulp.value(v) is not None and pulp.value(v) > 1e-9}
    return status, objective, sol

# Helpers: convert LPStandardForm to our LpProblem and run our solver
def run_ours_on_sf(sf: LPStandardForm, method: str = "interior-point", **kwargs):
    prob = LpProblem("LP_from_standard_form_ours", sense="Minimize")
    name_to_idx = {}
    for name in sf.var_names:
        idx = prob.add_variable(name, low_bound=0.0)
        name_to_idx[name] = idx

    # objective
    obj = {}
    for i, coeff in enumerate(sf.c):
        if coeff != 0:
            idx = name_to_idx[sf.var_names[i]]
            obj[idx] = obj.get(idx, 0.0) + float(coeff)
    prob.set_objective(obj)

    # equality constraints
    for row, b in zip(sf.A_eq, sf.b_eq):
        linear = {}
        for j, coeff in enumerate(row):
            if coeff != 0:
                idx = name_to_idx[sf.var_names[j]]
                linear[idx] = linear.get(idx, 0.0) + float(coeff)
        prob.add_constraint(linear, "==", float(b))

    solution, objective, status = prob.solve(method=method, **kwargs)
    # return solution for original vars only
    sol = {name: float(solution.get(name, 0.0)) for name in sf.var_names if solution.get(name, 0.0) and solution.get(name, 0.0) > 1e-9}
    return status, objective, sol


def get_mcf_test_1() -> LPStandardForm:
    # 变量：x[i][j][k] 表示商品 k 在弧 (i→j) 上的流量
    # 弧：(1→2), (2→3), (1→3)；商品：k1, k2
    # 变量顺序：x[1][2][1], x[1][2][2], x[2][3][1], x[2][3][2], x[1][3][1], x[1][3][2]
    A_eq = [
        # 商品 1：节点 1 流出 = 5
        [1, 0, 0, 0, 1, 0],  # x[1][2][1] + x[1][3][1] = 5
        # 商品 1：节点 2 流入 = 流出
        [-1, 0, 1, 0, 0, 0],  # -x[1][2][1] + x[2][3][1] = 0
        # 商品 1：节点 3 流入 = 5
        [0, 0, -1, 0, -1, 0],  # -x[2][3][1] - x[1][3][1] = -5
        # 商品 2：节点 1 无供需
        [0, 1, 0, 0, 0, 1],  # x[1][2][2] + x[1][3][2] = 0
        # 商品 2：节点 2 流出 = 3
        [0, -1, 0, 1, 0, 0],  # -x[1][2][2] + x[2][3][2] = 3
        # 商品 2：节点 3 流入 = 3
        [0, 0, 0, -1, 0, -1],  # -x[2][3][2] - x[1][3][2] = -3
    ]
    b_eq = [5, 0, -5, 0, 3, -3]
    c = [1, 1, 2, 2, 1, 1]  # 每条弧的成本
    var_names = ["x_1_2_1", "x_1_2_2", "x_2_3_1", "x_2_3_2", "x_1_3_1", "x_1_3_2"]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)

def get_mcf_test_2() -> LPStandardForm:
    # 变量：x[i][j][k] 表示商品 k 在弧 (i→j) 上的流量
    # 弧：(1→2), (2→3), (3→4), (4→5), (1→5)；商品：k1, k2
    # 变量顺序：x[1][2][1], x[1][2][2], x[2][3][1], x[2][3][2], x[3][4][1], x[3][4][2], x[4][5][1], x[4][5][2], x[1][5][1], x[1][5][2]
    A_eq = [
        # 商品 1：节点 1 流出 = 10
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # x[1][2][1] + x[1][5][1] = 10
        # 商品 1：节点 2 流入 = 流出
        [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # -x[1][2][1] + x[2][3][1] = 0
        # 商品 1：节点 3 流入 = 流出
        [0, 0, -1, 0, 1, 0, 0, 0, 0, 0],  # -x[2][3][1] + x[3][4][1] = 0
        # 商品 1：节点 4 流入 = 流出
        [0, 0, 0, 0, -1, 0, 1, 0, 0, 0],  # -x[3][4][1] + x[4][5][1] = 0
        # 商品 1：节点 5 流入 = 10
        [0, 0, 0, 0, 0, 0, -1, 0, -1, 0],  # -x[4][5][1] - x[1][5][1] = -10
        # 商品 2：节点 1 无供需
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # x[1][2][2] + x[1][5][2] = 0
        # 商品 2：节点 2 流出 = 4
        [0, -1, 0, 1, 0, 0, 0, 0, 0, 0],  # -x[1][2][2] + x[2][3][2] = 4
        # 商品 2：节点 3 流入 = 流出
        [0, 0, 0, -1, 0, 1, 0, 0, 0, 0],  # -x[2][3][2] + x[3][4][2] = 0
        # 商品 2：节点 4 流入 = 4
        [0, 0, 0, 0, 0, -1, 0, 1, 0, 0],  # -x[3][4][2] + x[4][5][2] = -4
        # 商品 2：节点 5 无供需
        [0, 0, 0, 0, 0, 0, 0, -1, 0, -1],  # -x[4][5][2] - x[1][5][2] = 0
    ]
    b_eq = [10, 0, 0, 0, -10, 0, 4, 0, -4, 0]
    c = [1, 1, 2, 2, 3, 3, 2, 2, 1, 1]  # 每条弧的成本
    var_names = ["x_1_2_1", "x_1_2_2", "x_2_3_1", "x_2_3_2", "x_3_4_1", "x_3_4_2", "x_4_5_1", "x_4_5_2", "x_1_5_1", "x_1_5_2"]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)

def get_mcf_test_3() -> LPStandardForm:
    """测试案例1：基本的多商品流问题"""
    # 变量：x[i][j][k] 表示商品 k 在弧 (i→j) 上的流量
    # 弧：(1→2), (2→3), (1→3)；商品：k1, k2
    # 变量顺序：x[1][2][1], x[1][2][2], x[2][3][1], x[2][3][2], x[1][3][1], x[1][3][2]
    A_eq = [
        # 商品 1：节点 1 流出 = 5
        [1, 0, 0, 0, 1, 0],  # x[1][2][1] + x[1][3][1] = 5
        # 商品 1：节点 2 流入 = 流出
        [-1, 0, 1, 0, 0, 0],  # -x[1][2][1] + x[2][3][1] = 0
        # 商品 1：节点 3 流入 = 5
        [0, 0, -1, 0, -1, 0],  # -x[2][3][1] - x[1][3][1] = -5
        # 商品 2：节点 1 无供需
        [0, 1, 0, 0, 0, 1],  # x[1][2][2] + x[1][3][2] = 0
        # 商品 2：节点 2 流出 = 3
        [0, -1, 0, 1, 0, 0],  # -x[1][2][2] + x[2][3][2] = 3
        # 商品 2：节点 3 流入 = 3
        [0, 0, 0, -1, 0, -1],  # -x[2][3][2] - x[1][3][2] = -3
    ]
    b_eq = [5, 0, -5, 0, 3, -3]
    c = [1, 1, 2, 2, 1, 1]  # 每条弧的成本
    var_names = ["x_1_2_1", "x_1_2_2", "x_2_3_1", "x_2_3_2", "x_1_3_1", "x_1_3_2"]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)

def get_mcf_test_4() -> LPStandardForm:
    """测试案例2：更复杂的网络，两个商品，四个节点"""
    # 弧：(1→2), (1→3), (2→3), (2→4), (3→4)
    # 商品：k1, k2
    # 变量顺序：x12_1, x12_2, x13_1, x13_2, x23_1, x23_2, x24_1, x24_2, x34_1, x34_2
    A_eq = [
        # 商品 1：节点 1 流出 = 8
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x12_1 + x13_1 = 8
        # 商品 1：节点 2 流入 = 流出
        [-1, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # -x12_1 + x23_1 + x24_1 = 0
        # 商品 1：节点 3 流入 = 流出
        [0, 0, -1, 0, -1, 0, 0, 0, 1, 0],  # -x13_1 - x23_1 + x34_1 = 0
        # 商品 1：节点 4 流入 = 8
        [0, 0, 0, 0, 0, 0, -1, 0, -1, 0],  # -x24_1 - x34_1 = -8
        
        # 商品 2：节点 1 流出 = 6
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # x12_2 + x13_2 = 6
        # 商品 2：节点 2 流入 = 流出
        [0, -1, 0, 0, 0, 1, 0, 1, 0, 0],  # -x12_2 + x23_2 + x24_2 = 0
        # 商品 2：节点 3 流入 = 流出
        [0, 0, 0, -1, 0, -1, 0, 0, 0, 1],  # -x13_2 - x23_2 + x34_2 = 0
        # 商品 2：节点 4 流入 = 6
        [0, 0, 0, 0, 0, 0, 0, -1, 0, -1],  # -x24_2 - x34_2 = -6
    ]
    b_eq = [8, 0, 0, -8, 6, 0, 0, -6]
    c = [2, 2, 1, 1, 3, 3, 4, 4, 2, 2]  # 每条弧的成本
    var_names = [
        "x_1_2_1", "x_1_2_2", "x_1_3_1", "x_1_3_2", 
        "x_2_3_1", "x_2_3_2", "x_2_4_1", "x_2_4_2", 
        "x_3_4_1", "x_3_4_2"
    ]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)

def get_mcf_test_5() -> LPStandardForm:
    """测试案例3：带容量约束的多商品流问题（通过等式约束模拟）"""
    # 弧：(1→2), (1→3), (2→3), (2→4), (3→4)
    # 商品：k1, k2
    # 变量顺序：x12_1, x12_2, x13_1, x13_2, x23_1, x23_2, x24_1, x24_2, x34_1, x34_2
    A_eq = [
        # 商品 1 流平衡约束
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 节点1: x12_1 + x13_1 = 10
        [-1, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 节点2: -x12_1 + x23_1 + x24_1 = 0
        [0, 0, -1, 0, -1, 0, 0, 0, 1, 0],  # 节点3: -x13_1 - x23_1 + x34_1 = 0
        [0, 0, 0, 0, 0, 0, -1, 0, -1, 0],  # 节点4: -x24_1 - x34_1 = -10
        
        # 商品 2 流平衡约束
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 节点1: x12_2 + x13_2 = 5
        [0, -1, 0, 0, 0, 1, 0, 1, 0, 0],  # 节点2: -x12_2 + x23_2 + x24_2 = 0
        [0, 0, 0, -1, 0, -1, 0, 0, 0, 1],  # 节点3: -x13_2 - x23_2 + x34_2 = 0
        [0, 0, 0, 0, 0, 0, 0, -1, 0, -1],  # 节点4: -x24_2 - x34_2 = -5,
        
        # 弧容量约束（模拟）
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 弧(1→2)总流量 <= 8
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # 弧(1→3)总流量 <= 7
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 弧(2→3)总流量 <= 4
    ]
    b_eq = [10, 0, 0, -10, 5, 0, 0, -5, 8, 7, 4]
    c = [3, 3, 2, 2, 1, 1, 4, 4, 2, 2]  # 每条弧的成本
    var_names = [
        "x_1_2_1", "x_1_2_2", "x_1_3_1", "x_1_3_2", 
        "x_2_3_1", "x_2_3_2", "x_2_4_1", "x_2_4_2", 
        "x_3_4_1", "x_3_4_2"
    ]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)

def get_mcf_test_infeasible() -> LPStandardForm:
    # 变量：x[i][j][k] 表示商品 k 在弧 (i→j) 上的流量
    # 弧：(1→2), (2→3), (3→4), (4→5), (5→6), (1→6)；商品：k1, k2, k3
    # 变量顺序：x[1][2][1], x[1][2][2], x[1][2][3], ..., x[1][6][1], x[1][6][2], x[1][6][3]
    A_eq = [
        # 商品 1：节点 1 流出 = 8
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x[1][2][1] + x[1][6][1] = 8
        # 商品 1：节点 2 流入 = 流出
        [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x[1][2][1] + x[2][3][1] = 0
        # 商品 1：节点 3 流入 = 流出
        [0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # -x[2][3][1] + x[3][4][1] = 0
        # 商品 1：节点 4 流入 = 流出
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],  # -x[3][4][1] + x[4][5][1] = 0
        # 商品 1：节点 5 流入 = 流出
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # -x[4][5][1] + x[5][6][1] = 0
        # 商品 1：节点 6 流入 = 8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # -x[1][6][1] - x[5][6][1] = -8
        # 商品 2：节点 1 无供需
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # x[1][2][2] + x[1][6][2] = 0
        # 商品 2：节点 2 流出 = 5
        [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x[1][2][2] + x[2][3][2] = 5
        # 商品 2：节点 3 流入 = 流出
        [0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # -x[2][3][2] + x[3][4][2] = 0
        # 商品 2：节点 4 流入 = 5
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],  # -x[3][4][2] + x[4][5][2] = -5
        # 商品 2：节点 5 无供需
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # -x[4][5][2] + x[5][6][2] = 0
        # 商品 2：节点 6 无供需
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # -x[1][6][2] - x[5][6][2] = 0
        # 商品 3：节点 1 无供需
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # x[1][2][3] + x[1][6][3] = 0
        # 商品 3：节点 2 无供需
        [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -x[1][2][3] + x[2][3][3] = 0
        # 商品 3：节点 3 流出 = 3
        [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # -x[2][3][3] + x[3][4][3] = 3
        # 商品 3：节点 4 流入 = 流出
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],  # -x[3][4][3] + x[4][5][3] = 0
        # 商品 3：节点 5 流入 = 流出
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # -x[4][5][3] + x[5][6][3] = 0
        # 商品 3：节点 6 流入 = 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],  # -x[1][6][3] - x[5][6][3] = -3
    ]
    b_eq = [8, 0, 0, 0, 0, -8, 0, 5, 0, -5, 0, 0, 0, 0, 3, 0, 0, -3]
    c = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1, 1]  # 每条弧的成本
    var_names = ["x_1_2_1", "x_1_2_2", "x_1_2_3", "x_2_3_1", "x_2_3_2", "x_2_3_3",
                 "x_3_4_1", "x_3_4_2", "x_3_4_3", "x_4_5_1", "x_4_5_2", "x_4_5_3",
                 "x_1_6_1", "x_1_6_2", "x_1_6_3"]
    return LPStandardForm(A_eq=A_eq, b_eq=b_eq, c=c, var_names=var_names)


def run_all_mcf_tests():
    """Run all get_mcf_test_N() tests, time PuLP and our interior-point solver, and print a summary."""

    # collect available test functions named get_mcf_test_ok1..get_mcf_test_ok7
    tests = []
    for n in range(1, 6):
        fname = f"get_mcf_test_{n}"
        f = globals().get(fname)
        if callable(f):
            tests.append((fname, f))

    print("\n=== Running all get_mcf_test_* tests (timing PuLP vs our interior-point) ===")
    results = []
    for name, f in tests:
        print(f"\n--- {name} ---")
        sf = f()

        t0 = time.perf_counter()
        s_p, o_p, sol_p = run_pulp_on_sf(sf)
        t1 = time.perf_counter()
        pulp_time = t1 - t0

        t0 = time.perf_counter()
        s_o, o_o, sol_o = run_ours_on_sf(sf, method="interior-point")
        t1 = time.perf_counter()
        ours_time = t1 - t0

        print(f"PuLP: status={s_p}, objective={o_p}, time={pulp_time:.6f}s")
        print(f"Ours: status={s_o}, objective={o_o}, time={ours_time:.6f}s")

        results.append((name, pulp_time, ours_time, s_p, o_p, s_o, o_o))

    # Summary
    print("\n=== Timing summary ===")
    for name, pulp_time, ours_time, s_p, o_p, s_o, o_o in results:
        print(f"{name}: PuLP {pulp_time:.6f}s | Ours {ours_time:.6f}s | PuLP_obj={o_p} | Ours_obj={o_o} | PuLP_status={s_p} | Ours_status={s_o}")

if __name__ == "__main__":
    run_all_mcf_tests()


