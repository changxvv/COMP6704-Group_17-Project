# Interior Point Method - LMCF 测试工具

基于内点法的线性多商品流（LMCF）问题求解器，支持自动发现和测试所有LMCF测试案例。

## 快速开始

```bash
# 测试所有案例（默认单案例10分钟）
python main.py

# 测试特定类别
python main.py --category GridDemands

# 自定义单案例时间限制
python main.py --time-limit-per-case 300

# 自定义总时间限制
python main.py --total-time-limit 3600

# 调整最大迭代次数
python main.py --max-iter 1000
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-limit-per-case` | 600秒 | 单个案例的时间限制 |
| `--total-time-limit` | 1800秒 | 总时间限制 |
| `--max-iter` | 500 | 最大迭代次数 |
| `--category` | 无 | 测试类别：GridDemands/OtherDemands/PlanarNetworks/TrafficNetworks |
| `--verbose` | 开启 | 详细日志输出 |

## 支持的测试类别

- **GridDemands**: 15个网格需求测试案例
- **OtherDemands**: 3个其他需求测试案例  
- **PlanarNetworks**: 10个平面网络测试案例
- **TrafficNetworks**: 6个交通网络测试案例

## 输出

- 控制台显示实时求解进度和结果汇总
- 结果文件保存为 `lmcf_test_results.txt`（或带类别后缀）
- 包含问题规模、求解时间、收敛状态等详细信息

## 示例

```bash
# 只测试GridDemands，每个案例5分钟
python main.py --category GridDemands --time-limit-per-case 300

# 测试所有案例，总时间限制1小时
python main.py --total-time-limit 3600
```
