"""
LMCF格式数据集解析器

用于解析LMCF (Linear Multi-Commodity Flow) 格式的测试数据集。

LMCF格式包含两个文件：
- 网络文件 (C*.txt): startnode endnode cost capacity
- 需求文件 (D*.txt): origin destination demand

解析后可直接转换为 McfProblem 对象供 interior_point 求解器使用。
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def parse_lmcf_files(
    network_file: Path,
    demand_file: Path,
    undirected: bool = False,  # 默认处理为有向图
    use_uniform_capacity: bool = False,
    capacity_multiplier: float = 1.0
) -> Tuple[List[str], Dict[str, List[str]], List[Tuple[str, str, float]], Dict[Tuple[str, str], float]]:
    """
    解析LMCF格式文件，返回可直接构造McfNetwork和McfDemand的数据。

    Args:
        network_file: 网络文件路径 (C*.txt)
        demand_file: 需求文件路径 (D*.txt)
        undirected: 是否视为无向图（创建反向边）
        use_uniform_capacity: 是否使用统一容量（如果False，返回每条边各自容量）
        capacity_multiplier: 容量倍数（用于调整问题难度）

    Returns:
        Tuple of (nodes, adjacency, demands_data, edge_capacities, edge_costs):
        - nodes: 节点ID列表
        - adjacency: 邻接表 {node_id: [neighbor_ids]}
        - demands_data: 需求列表 [(source, target, bandwidth), ...]
        - edge_capacities: 边容量字典 {(src, tgt): capacity}
        - edge_costs: 边成本字典 {(src, tgt): cost}

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式无效
    """
    network_file = Path(network_file)
    demand_file = Path(demand_file)

    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")
    if not demand_file.exists():
        raise FileNotFoundError(f"Demand file not found: {demand_file}")

    logger.info(f"Parsing LMCF files: {network_file.name}, {demand_file.name}")

    # 解析网络文件
    arcs, capacities, costs, nodes_from_links = _parse_network_file(network_file, undirected)

    # 解析需求文件
    demands_data, nodes_from_demands = _parse_demand_file(demand_file)

    # 合并所有节点
    all_nodes = sorted(nodes_from_links | nodes_from_demands)

    # 构建邻接表
    adjacency = {node: [] for node in all_nodes}
    for src, tgt in arcs:
        if tgt not in adjacency[src]:
            adjacency[src].append(tgt)

    # 应用容量倍数
    if capacity_multiplier != 1.0:
        capacities = {arc: cap * capacity_multiplier for arc, cap in capacities.items()}
        logger.info(f"Applied capacity multiplier: {capacity_multiplier}")

    # 返回边容量字典
    logger.info(f"Parsed {len(all_nodes)} nodes, {len(arcs)} arcs, "
               f"{len(demands_data)} demands")
    logger.info(f"Capacity range: {min(capacities.values()):.2f} - {max(capacities.values()):.2f}")

    return all_nodes, adjacency, demands_data, capacities, costs


def _parse_network_file(
    filepath: Path,
    undirected: bool
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float], set]:
    """
    解析网络文件 (C*.txt)。

    Format: startnode endnode cost capacity (空格或制表符分隔)

    Args:
        filepath: 网络文件路径
        undirected: 是否为无向图

    Returns:
        Tuple of (arcs, capacities, node_ids):
        - arcs: 边列表 [(src, tgt), ...]
        - capacities: 边容量字典 {(src, tgt): capacity}
        - node_ids: 节点ID集合
    """
    arcs = []
    capacities = {}
    costs = {}
    node_ids = set()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 解析: startnode endnode cost capacity
            parts = line.split()

            if len(parts) < 4:
                logger.warning(f"Skipping invalid line {line_num} in {filepath.name}: {line}")
                continue

            try:
                start_node = str(parts[0])
                end_node = str(parts[1])
                cost = float(parts[2])  # cost字段暂不使用（可扩展为边权重）
                capacity = float(parts[3])

                # 记录节点
                node_ids.add(start_node)
                node_ids.add(end_node)

                # 记录边
                arc = (start_node, end_node)
                arcs.append(arc)
                capacities[arc] = capacity
                costs[arc] = cost

                # 如果是无向图，添加反向边
                if undirected:
                    reverse_arc = (end_node, start_node)
                    arcs.append(reverse_arc)
                    capacities[reverse_arc] = capacity
                    costs[reverse_arc] = cost

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num} in {filepath.name}: {e}")
                continue

    return arcs, capacities, costs, node_ids


def _parse_demand_file(filepath: Path) -> Tuple[List[Tuple[str, str, float]], set]:
    """
    解析需求文件 (D*.txt)。

    Format: origin destination demand (空格或制表符分隔)

    Args:
        filepath: 需求文件路径

    Returns:
        Tuple of (demands_data, node_ids):
        - demands_data: 需求列表 [(source, target, bandwidth), ...]
        - node_ids: 节点ID集合
    """
    demands_data = []
    node_ids = set()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 解析: origin destination demand
            parts = line.split()

            if len(parts) < 3:
                logger.warning(f"Skipping invalid line {line_num} in {filepath.name}: {line}")
                continue

            try:
                origin = str(parts[0])
                destination = str(parts[1])
                demand_value = float(parts[2])

                # 记录节点
                node_ids.add(origin)
                node_ids.add(destination)

                # 记录需求
                demands_data.append((origin, destination, demand_value))

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {line_num} in {filepath.name}: {e}")
                continue

    return demands_data, node_ids


def load_lmcf_as_mcf_problem(
    network_file: Path,
    demand_file: Path,
    undirected: bool = True,
    capacity_multiplier: float = 1.0
):
    """
    加载LMCF文件并直接构造McfProblem对象。

    Args:
        network_file: 网络文件路径 (C*.txt)
        demand_file: 需求文件路径 (D*.txt)
        undirected: 是否为无向图
        capacity_multiplier: 容量倍数

    Returns:
        McfProblem对象
    """
    from .modeling import McfNetwork, McfDemand, McfProblem

    nodes, adjacency, demands_data, edge_capacities = parse_lmcf_files(
        network_file, demand_file, undirected, capacity_multiplier=capacity_multiplier
    )

    # 构造网络
    network = McfNetwork(nodes=nodes, adjacency=adjacency)

    # 构造需求列表
    demands = [
        McfDemand(source=src, target=tgt, bandwidth=bw)
        for src, tgt, bw in demands_data
    ]

    # 构造完整问题（使用每条边各自的容量）
    mcf_problem = McfProblem(
        network=network,
        demands=demands,
        edge_capacities=edge_capacities
    )

    return mcf_problem


def get_lmcf_problem_stats(network_file: Path, demand_file: Path) -> Dict[str, Any]:
    """
    获取LMCF问题的统计信息（不构造完整对象）。

    Args:
        network_file: 网络文件路径
        demand_file: 需求文件路径

    Returns:
        包含统计信息的字典
    """
    nodes, adjacency, demands_data, capacity = parse_lmcf_files(
        network_file, demand_file, undirected=True
    )

    num_arcs = sum(len(neighbors) for neighbors in adjacency.values())
    total_demand = sum(bw for _, _, bw in demands_data)

    return {
        'num_nodes': len(nodes),
        'num_arcs': num_arcs,
        'num_demands': len(demands_data),
        'capacity': capacity,
        'total_demand': total_demand,
        'network_file': str(network_file),
        'demand_file': str(demand_file),
    }

