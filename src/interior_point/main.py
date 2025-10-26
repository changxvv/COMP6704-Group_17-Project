"""
Interior Point Method - LMCF 完整测试脚本
"""

import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re

# 添加当前目录到path以便导入lib模块
sys.path.insert(0, str(Path(__file__).parent))

from lib.dataset_parser import parse_lmcf_files, get_lmcf_problem_stats
from lib.modeling import McfNetwork, McfDemand, McfProblem, mcf_to_lp
from lib.solvers import interior_point_solve


# 测试数据根目录
LMCF_ROOT = Path(__file__).parent.parent.parent / "tests" / "LMCF"

# 测试配置
TOTAL_TIME_LIMIT = 30 * 60  # 总时间限制: 30分钟（秒）
MAX_ITER = 500
VERBOSE = True  # 开启详细日志
LOG_EVERY = 1  # 每次迭代都打印（用于实时显示）


def discover_test_cases() -> List[Dict[str, Any]]:
    """
    自动发现所有测试案例
    扫描LMCF目录下的所有子目录，找到匹配的C*.txt和D*.txt文件对
    
    Returns:
        测试案例列表
    """
    test_cases = []
    case_id = 1
    
    # 扫描所有子目录
    subdirs = ["GridDemands", "OtherDemands", "PlanarNetworks", "TrafficNetworks"]
    
    for subdir in subdirs:
        subdir_path = LMCF_ROOT / subdir
        if not subdir_path.exists():
            continue
        
        # 找到所有C开头的网络文件（使用自然序排序，如 Cgd1, Cgd2, ... Cgd10）
        def natural_sort_key(p: Path):
            name = p.name
            parts = re.split(r'(\d+)', name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]

        network_files = sorted(subdir_path.glob("C*.txt"), key=natural_sort_key)
        
        for network_file in network_files:
            # 从网络文件名推断需求文件名
            network_name = network_file.name
            # C*.txt -> D*.txt
            demand_name = "D" + network_name[1:]
            demand_file = subdir_path / demand_name
            
            # 检查需求文件是否存在
            if not demand_file.exists():
                # 特殊处理Dpl2500.txt的多个版本
                if network_name == "Cpl2500.txt":
                    for variant in ["Dpl2500.txt", "Dpl2500_1.txt", "Dpl2500_2.txt"]:
                        demand_file_variant = subdir_path / variant
                        if demand_file_variant.exists():
                            demand_name = variant
                            demand_file = demand_file_variant
                            break
                
                if not demand_file.exists():
                    continue
            
            # 生成案例名称
            base_name = network_name.replace(".txt", "").replace("C", "")
            case_name = f"{subdir}/{base_name}"
            
            test_cases.append({
                "id": case_id,
                "category": subdir,
                "network": network_name,
                "demand": demand_name,
                "name": case_name,
                "network_path": network_file,
                "demand_path": demand_file,
            })
            case_id += 1
    
    return test_cases


def solve_single_case(case: Dict[str, Any], time_limit: float) -> Dict[str, Any]:
    """
    求解单个测试案例
    
    Args:
        case: 案例信息字典
        time_limit: 该案例的时间限制（秒）
    
    Returns:
        结果字典
    """
    network_file = case["network_path"]
    demand_file = case["demand_path"]
    
    print(f"\n{'='*80}")
    print(f"【{case['name']}】")
    print(f"{'='*80}")
    print(f"  网络文件: {case['network']}")
    print(f"  需求文件: {case['demand']}")
    
    # 加载数据
    try:
        load_start = time.time()
        nodes, adjacency, demands_data, edge_capacities, edge_costs = parse_lmcf_files(
            network_file,
            demand_file,
            undirected=False  # 使用有向图
        )
        load_time = time.time() - load_start
        
        # 显示问题规模
        num_edges = len([v for vs in adjacency.values() for v in vs])
        cap_values = list(edge_capacities.values())
        total_demand = sum(bw for _, _, bw in demands_data)
        
        print(f"\n【问题规模】")
        print(f"  节点数: {len(nodes)}")
        print(f"  边数: {num_edges}")
        print(f"  需求数: {len(demands_data)}")
        print(f"  容量范围: {min(cap_values):.0f} - {max(cap_values):.0f}")
        print(f"  总需求量: {total_demand:.2f}")
        print(f"  数据加载时间: {load_time:.2f}s")
        
    except Exception as e:
        print(f"\n[FAIL] 数据加载失败: {e}")
        return {
            "case_id": case["id"],
            "case_name": case["name"],
            "status": "load_failed",
            "error": str(e),
            "time": 0,
        }
    
    # 构建问题
    try:
        build_start = time.time()
        network = McfNetwork(nodes=nodes, adjacency=adjacency)
        demands = [McfDemand(source=s, target=t, bandwidth=bw) for s, t, bw in demands_data]
        mcf = McfProblem(network=network, demands=demands, edge_capacities=edge_capacities)
        
        lp_prob = mcf_to_lp(mcf, edge_costs)
        lp_std = lp_prob.to_standard_form()
        build_time = time.time() - build_start
        
        print(f"  LP构建时间: {build_time:.2f}s")
        print(f"  LP变量数: {len(lp_std.var_names)}")
        print(f"  LP约束数: {len(lp_std.b_eq)}")
        
    except Exception as e:
        print(f"\n[FAIL] LP构建失败: {e}")
        return {
            "case_id": case["id"],
            "case_name": case["name"],
            "status": "build_failed",
            "error": str(e),
            "time": load_time,
        }
    
    # 求解
    try:
        print(f"\n【开始求解】(时间限制: {time_limit/60:.1f}分钟)")
        print(f"{'迭代':<8} {'目标值':<15} {'原始残差':<12} {'对偶残差':<12} {'互补性':<12} {'停滞':<6}")
        print("-" * 80)
        solve_start = time.time()
        
        x_vals, obj, info = interior_point_solve(
            lp_std,
            max_iter=MAX_ITER,
            time_limit=time_limit,
            verbose=VERBOSE,
            log_every=LOG_EVERY,
            return_info=True
        )
        
        solve_time = time.time() - solve_start
        total_time = time.time() - load_start
        
        print(f"\n【求解完成】")
        print(f"  总时间: {total_time:.2f}s ({total_time/60:.2f}分钟)")
        print(f"  求解时间: {solve_time:.2f}s")
        print(f"  目标值: {obj:.2f}")
        print(f"  终止原因: {info['termination_reason']}")
        print(f"  收敛状态: {'成功' if info['converged'] else '未成功'}")
        print(f"  迭代次数: {info['iterations']}")
        print(f"  原始残差: {info['primal_residual']:.2e}")
        print(f"  对偶残差: {info['dual_residual']:.2e}")
        print(f"  互补性: {info['complementarity']:.2e}")
        
        return {
            "case_id": case["id"],
            "case_name": case["name"],
            "status": "solved",
            "num_nodes": len(nodes),
            "num_edges": num_edges,
            "num_demands": len(demands_data),
            "num_variables": len(lp_std.var_names),
            "num_constraints": len(lp_std.b_eq),
            "load_time": load_time,
            "build_time": build_time,
            "solve_time": solve_time,
            "total_time": total_time,
            "objective": obj,
            "converged": info['converged'],
            "termination_reason": info['termination_reason'],
            "iterations": info['iterations'],
            "primal_residual": info['primal_residual'],
            "dual_residual": info['dual_residual'],
            "complementarity": info['complementarity'],
        }
        
    except Exception as e:
        solve_time = time.time() - solve_start
        print(f"\n[FAIL] 求解失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "case_id": case["id"],
            "case_name": case["name"],
            "status": "solve_failed",
            "error": str(e),
            "time": load_time + build_time + solve_time,
        }


def print_summary_table(results: List[Dict[str, Any]]):
    """打印结果汇总表格"""
    print(f"\n{'='*120}")
    print("【所有案例求解结果汇总】")
    print(f"{'='*120}")
    
    # 表头
    header = f"{'案例':<12} {'状态':<10} {'节点':<8} {'边':<8} {'需求':<8} {'时间(s)':<10} {'目标值':<15} {'迭代':<8} {'收敛':<6}"
    print(header)
    print("-" * 120)
    
    # 数据行
    total_time = 0
    success_count = 0
    
    for res in results:
        case_name = res['case_name']
        status = res['status']
        
        if status == "solved":
            success_count += 1
            nodes = res['num_nodes']
            edges = res['num_edges']
            demands = res['num_demands']
            time_str = f"{res['total_time']:.2f}"
            obj_str = f"{res['objective']:.2f}"
            iters = res['iterations']
            conv = "[OK]" if res['converged'] else "[FAIL]"
            total_time += res['total_time']
        else:
            nodes = "-"
            edges = "-"
            demands = "-"
            time_str = f"{res.get('time', 0):.2f}"
            obj_str = "-"
            iters = "-"
            conv = "-"
            status = "[X] " + status
        
        print(f"{case_name:<12} {status:<10} {str(nodes):<8} {str(edges):<8} {str(demands):<8} {time_str:<10} {obj_str:<15} {str(iters):<8} {conv:<6}")
    
    print("=" * 120)
    print(f"\n统计信息:")
    print(f"  成功求解: {success_count}/{len(results)}")
    print(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}分钟)")
    print(f"  平均耗时: {total_time/success_count:.2f}s/案例" if success_count > 0 else "  平均耗时: N/A")


def save_results_to_file(results: List[Dict[str, Any]], filename: str = "lmcf_test_results.txt"):
    """保存结果到文件"""
    output_file = Path(__file__).parent / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Interior Point Method - LMCF 测试结果\n")
        f.write("=" * 120 + "\n\n")
        
        for res in results:
            f.write(f"案例: {res['case_name']}\n")
            f.write(f"状态: {res['status']}\n")
            
            if res['status'] == "solved":
                f.write(f"  问题规模: {res['num_nodes']}节点, {res['num_edges']}边, {res['num_demands']}需求\n")
                f.write(f"  LP规模: {res['num_variables']}变量, {res['num_constraints']}约束\n")
                f.write(f"  总时间: {res['total_time']:.2f}s\n")
                f.write(f"  求解时间: {res['solve_time']:.2f}s\n")
                f.write(f"  目标值: {res['objective']:.6f}\n")
                f.write(f"  收敛: {res['converged']}\n")
                f.write(f"  终止原因: {res['termination_reason']}\n")
                f.write(f"  迭代次数: {res['iterations']}\n")
                f.write(f"  原始残差: {res['primal_residual']:.2e}\n")
                f.write(f"  对偶残差: {res['dual_residual']:.2e}\n")
            else:
                f.write(f"  错误: {res.get('error', 'Unknown')}\n")
            
            f.write("\n" + "-" * 120 + "\n\n")
    
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    # 更新全局配置（在使用之前声明）
    global MAX_ITER, VERBOSE
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Interior Point Method - LMCF 完整测试脚本')
    parser.add_argument('--network-file', type=str, default=None,
                        help='单个网络文件路径（用于单案例模式）')
    parser.add_argument('--demand-file', type=str, default=None,
                        help='单个需求文件路径（用于单案例模式）')
    parser.add_argument('--time-limit-per-case', type=float, default=600,
                        help='单个案例的时间限制（秒）。如果不指定，则默认600秒')
    parser.add_argument('--total-time-limit', type=float, default=TOTAL_TIME_LIMIT,
                        help=f'总时间限制（秒），默认为{TOTAL_TIME_LIMIT}秒')
    parser.add_argument('--max-iter', type=int, default=MAX_ITER,
                        help=f'最大迭代次数，默认为{MAX_ITER}')
    parser.add_argument('--category', type=str, default=None,
                        help='只测试指定类别的案例（GridDemands/OtherDemands/PlanarNetworks/TrafficNetworks）')
    parser.add_argument('--verbose', action='store_true', default=VERBOSE,
                        help='开启详细日志')

    args = parser.parse_args()
    
    # 应用配置
    MAX_ITER = args.max_iter
    VERBOSE = args.verbose
    total_time_limit = args.total_time_limit

    # 单案例模式
    if args.network_file and args.demand_file:
        # 单案例模式：直接求解指定的网络和需求文件
        network_path = Path(args.network_file)
        demand_path = Path(args.demand_file)

        if not network_path.exists():
            print(f"错误：图文件不存在: {args.network_file}")
            return
        if not demand_path.exists():
            print(f"错误：需求文件不存在: {args.demand_file}")
            return

        case = {
            "id": 1,
            "category": "single",
            "network": network_path.name,
            "demand": demand_path.name,
            "name": f"single/{network_path.stem}",
            "network_path": network_path,
            "demand_path": demand_path,
        }

        result = solve_single_case(case, args.time_limit_per_case)

        # 打印结果摘要
        print(f"\n{'='*80}")
        print("【求解结果】")
        print(f"{'='*80}")
        print(f"状态: {result['status']}")
        if result['status'] == 'solved':
            print(f"目标值: {result['objective']:.6f}")
            print(f"求解时间: {result['solve_time']:.2f}s")
            print(f"迭代次数: {result['iterations']}")
            print(f"收敛: {'是' if result['converged'] else '否'}")
        else:
            print(f"错误: {result.get('error', 'Unknown')}")

        return

    # 批量测试模式
    # 发现所有测试案例
    print("=" * 120)
    print("正在扫描测试案例...")
    all_test_cases = discover_test_cases()

    # 过滤案例（如果指定了类别）
    if args.category:
        test_cases = [c for c in all_test_cases if c['category'] == args.category]
        if not test_cases:
            print(f"错误：未找到类别为 '{args.category}' 的测试案例")
            print(f"可用类别：GridDemands, OtherDemands, PlanarNetworks, TrafficNetworks")
            return
    else:
        test_cases = all_test_cases
    
    # 确定每个案例的时间限制
    if args.time_limit_per_case:
        time_limit_per_case = args.time_limit_per_case
    else:
        # 平均分配总时间
        time_limit_per_case = total_time_limit / len(test_cases) if test_cases else total_time_limit
    
    print("=" * 120)
    print("Interior Point Method - LMCF 完整测试")
    print("=" * 120)
    print(f"\n测试配置:")
    print(f"  发现案例数量: {len(all_test_cases)}")
    if args.category:
        print(f"  选择类别: {args.category}")
    print(f"  测试案例数量: {len(test_cases)}")
    print(f"  单案例时间限制: {time_limit_per_case:.0f}秒 ({time_limit_per_case/60:.1f}分钟)")
    print(f"  总时间限制: {total_time_limit/60:.0f}分钟")
    print(f"  最大迭代次数: {MAX_ITER}")
    print(f"  详细日志: {'开启' if VERBOSE else '关闭'}")
    
    # 显示发现的案例
    print(f"\n发现的测试案例:")
    for category in ["GridDemands", "OtherDemands", "PlanarNetworks", "TrafficNetworks"]:
        cat_cases = [c for c in all_test_cases if c['category'] == category]
        if cat_cases:
            print(f"  {category}: {len(cat_cases)} 个")
    
    overall_start = time.time()
    results = []
    
    # 依次测试每个案例
    for i, case in enumerate(test_cases, 1):
        # 检查总时间是否已超限（只在未指定单案例时间限制时检查）
        elapsed = time.time() - overall_start
        if not args.time_limit_per_case and elapsed >= total_time_limit:
            print(f"\n\n已达到总时间限制({total_time_limit/60:.0f}分钟)，停止测试")
            print(f"   已完成 {len(results)}/{len(test_cases)} 个案例")
            break
        
        print(f"\n\n{'='*100}")
        print(f"进度: [{i}/{len(test_cases)}] | 已用: {elapsed/60:.1f}分钟")
        if not args.time_limit_per_case:
            remaining_time = total_time_limit - elapsed
            print(f"剩余总时间: {remaining_time/60:.1f}分钟")
        print(f"{'='*100}")
        
        result = solve_single_case(case, time_limit_per_case)
        results.append(result)
    
    # 打印汇总
    overall_time = time.time() - overall_start
    print_summary_table(results)
    
    print(f"\n总运行时间: {overall_time:.2f}s ({overall_time/60:.2f}分钟)")
    
    # 保存结果
    output_filename = f"lmcf_test_results{'_' + args.category if args.category else ''}.txt"
    save_results_to_file(results, output_filename)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

