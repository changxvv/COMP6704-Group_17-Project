import time

# ========== 统一的测试问题定义 ==========
# 使用LMCF Grid 1案例: 25节点，80边，50需求
LMCF_NETWORK_FILE = "../../tests/LMCF/GridDemands/Cgd1.txt"
LMCF_DEMAND_FILE = "../../tests/LMCF/GridDemands/Dgd1.txt"

# 用于存储解析后的共享数据
TEST_NODES = None
TEST_ADJACENCY = None
TEST_EDGE_CAPACITIES = None  # 每条边各自的容量
TEST_DEMANDS_DATA = None


def load_test_data():
	"""加载并解析LMCF测试数据（只执行一次）"""
	global TEST_NODES, TEST_ADJACENCY, TEST_EDGE_CAPACITIES, TEST_DEMANDS_DATA
	
	if TEST_NODES is not None:
		return  # 已加载
	
	from lib.dataset_parser import parse_lmcf_files
	
	TEST_NODES, TEST_ADJACENCY, TEST_DEMANDS_DATA, TEST_EDGE_CAPACITIES = parse_lmcf_files(
		LMCF_NETWORK_FILE,
		LMCF_DEMAND_FILE,
		undirected=True  # LMCF问题通常是无向图
	)


# ----- Interior Point (使用 McfProblem + mcf_to_lp 转换层) -----
def test_interior_point() -> dict:
	load_test_data() 
	
	from lib.modeling import McfNetwork, McfDemand, McfProblem, mcf_to_lp
	from lib.solvers import interior_point_solve

	network = McfNetwork(nodes=TEST_NODES, adjacency=TEST_ADJACENCY)
	demands = [McfDemand(source=s, target=t, bandwidth=bw) for s, t, bw in TEST_DEMANDS_DATA]
	
	# 使用每条边各自的容量
	mcf = McfProblem(network=network, demands=demands, edge_capacities=TEST_EDGE_CAPACITIES)

	print(f"[interior_point] 问题规模: {len(TEST_NODES)}节点, {len(network.arcs)}边, {len(demands)}需求")
	cap_values = list(TEST_EDGE_CAPACITIES.values())
	print(f"[interior_point] 容量范围: {min(cap_values):.0f} - {max(cap_values):.0f}")
	
	# 转换为 LpProblem 并求解
	start_time = time.perf_counter()
	lp_prob = mcf_to_lp(mcf)
	lp_std = lp_prob.to_standard_form()
	print(f"[interior_point] 开始求解...")
	x_vals, obj, info = interior_point_solve(lp_std, max_iter=50, verbose=True, log_every=3, return_info=True)
	solve_time = time.perf_counter() - start_time

	print(f"[interior_point] 求解时间: {solve_time:.4f}s")
	print(f"[interior_point] 目标值: {obj:.6f}")
	print(f"[interior_point] 收敛状态: {info['termination_reason']}")
	print(f"[interior_point] 迭代次数: {info['iterations']}")
	print(f"[interior_point] 原始残差: {info['primal_residual']:.2e}")
	print(f"[interior_point] 对偶残差: {info['dual_residual']:.2e}")
	print(f"[interior_point] 互补性: {info['complementarity']:.2e}")
	print(f"[interior_point] LP变量数: {len(x_vals)}")

	return {
		"time": solve_time,
		"objective": obj,
		"converged": info['converged'],
		"termination_reason": info['termination_reason'],
		"iterations": info['iterations'],
		"primal_residual": info['primal_residual'],
		"dual_residual": info['dual_residual'],
	}


if __name__ == "__main__":
	print("=" * 80)
	print("多商品流问题 (Multi-Commodity Flow) 求解器性能对比测试")
	print("=" * 80)
	
	# 加载测试数据
	print("\n【加载测试数据】")
	print(f"  数据集: LMCF Grid 1")
	print(f"  网络文件: {LMCF_NETWORK_FILE}")
	print(f"  需求文件: {LMCF_DEMAND_FILE}")
	
	load_test_data()
	
	print(f"\n【测试问题规模】")
	print(f"  节点数: {len(TEST_NODES)}")
	print(f"  边数: {len([v for vs in TEST_ADJACENCY.values() for v in vs])}")
	print(f"  需求数: {len(TEST_DEMANDS_DATA)}")
	
	cap_values = list(TEST_EDGE_CAPACITIES.values())
	print(f"  容量范围: {min(cap_values):.0f} - {max(cap_values):.0f} (每条边各自容量)")
	
	total_demand = sum(bw for _, _, bw in TEST_DEMANDS_DATA)
	print(f"  总需求量: {total_demand:.2f}")
	print(f"  平均需求: {total_demand/len(TEST_DEMANDS_DATA):.2f}")
	
	print("\n" + "=" * 80)

	results = {}

	# Interior Point
	try:
		print("\n【Interior Point 求解器】")
		print("-" * 80)
		results["interior_point"] = test_interior_point()
	except Exception as e:
		print(f"[interior_point] 错误: {e}")
		import traceback
		traceback.print_exc()
		results["interior_point"] = {"time": None, "objective": None, "converged": False, "error": str(e)}

	# 汇总结果
	print("\n" + "=" * 80)
	print("【求解结果汇总】")
	print("=" * 80)
	
	if "interior_point" in results:
		res = results["interior_point"]
		if res['time'] is not None:
			print(f"Interior Point 求解器:")
			print(f"  求解时间: {res['time']:.4f}秒")
			print(f"  目标值: {res['objective']:.2f}")
			print(f"  终止原因: {res.get('termination_reason', 'unknown')}")
			print(f"  收敛状态: {'✓ 成功' if res.get('converged', False) else '✗ 未完全收敛'}")
			print(f"  迭代次数: {res.get('iterations', 'N/A')}")
			
			# 显示残差信息（判断解的质量）
			if 'primal_residual' in res:
				p_res = res['primal_residual']
				d_res = res['dual_residual']
				print(f"  解的质量:")
				print(f"    原始可行性残差: {p_res:.2e} {'✓' if p_res < 1e-6 else '⚠'}")
				print(f"    对偶可行性残差: {d_res:.2e} {'✓' if d_res < 1e-6 else '⚠'}")
		else:
			print(f"Interior Point 求解器: 求解失败")
			if 'error' in res:
				print(f"  错误信息: {res['error']}")
	
	print("=" * 80)
