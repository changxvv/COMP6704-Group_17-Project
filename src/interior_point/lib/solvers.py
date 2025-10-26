from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import time

from .modeling import LPStandardForm


class SolverError(Exception):
    pass


def interior_point_solve(
    lp: LPStandardForm,
    max_iter: int = 100,
    time_limit: float | None = None,
    verbose: bool = False,
    log_every: int = 3,
    scale_columns: bool = True,
    initial_reg: float = 1e-8,
    max_reg: float = 1e-2,
    normal_eq_solver: str = "cg",
    cg_max_iter: int = 500,
    cg_tol: float = 1e-10,
    return_info: bool = True,
) -> Tuple[List[float], float] | Tuple[List[float], float, Dict[str, Any]]:
    """
    使用原对偶内点法求解标准形式的线性规划问题
    
    参数:
        lp: 标准形式的LP问题
        max_iter: 最大迭代次数
        time_limit: 时间限制（秒）
        verbose: 是否打印详细日志
        log_every: 日志打印频率
        scale_columns: 是否进行列缩放
        initial_reg: 初始正则化参数
        max_reg: 最大正则化参数
        normal_eq_solver: 正规方程求解器 ("direct" 或 "cg")
        cg_max_iter: 共轭梯度法最大迭代次数
        cg_tol: 共轭梯度法容差
        return_info: 是否返回详细求解信息
    
    返回:
        (x, obj) 或 (x, obj, info)
        - x: 最优解
        - obj: 最优目标值
        - info: 求解信息字典（如果return_info=True）
    """
    start_time = time.time()
    
    # 转换为NumPy数组
    A = np.array(lp.A_eq, dtype=float)
    b = np.array(lp.b_eq, dtype=float)
    c = np.array(lp.c, dtype=float)
    c_true = c.copy()  # 保存原始目标系数
    
    m, n = A.shape if A.size > 0 else (len(b), len(c))
    
    if n == 0:
        return [], 0.0
    
    # 列缩放（改善条件数）
    if scale_columns:
        col_norms = np.linalg.norm(A, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)
        A = A / col_norms
        c = c / col_norms
        
        # 检查尺度化后的矩阵条件数
        col_conds = np.linalg.cond(A)
        if np.max(col_conds) > 1e10:
            if verbose:
                print(f"警告：尺度化后条件数仍然很大: {np.max(col_conds):.2e}")
            # 如果条件数过大，禁用列缩放
            A = A * col_norms
            c = c * col_norms
            col_norms = np.ones(n)
            if verbose:
                print("已禁用列缩放")
    else:
        col_norms = np.ones(n)
    
    # 构建转置矩阵
    AT = A.T
    
    # 初始化变量
    x = np.ones(n)
    y = np.zeros(m)
    s = np.maximum(c, 1.0)  # s至少为1
    
    # 多级收敛容差（平衡原始残差和对偶残差）
    tol_tight = 1e-6
    tol_relaxed = 1e-6  # 原始残差要求1e-06
    tol_primal_priority = 1e-6  # 原始残差优先收敛标准1e-06
    
    # 初始化求解状态信息
    converged = False
    termination_reason = "max_iterations"
    final_iteration = 0
    final_r_p_norm = 0.0
    final_r_d_norm = 0.0
    final_mu = 0.0
    final_gap = 0.0
    
    # 停滞检测
    stagnation_count = 0
    prev_obj = float('inf')
    
    # ========== 主迭代循环 ==========
    for it in range(max_iter):
        # 时间限制检查
        if time_limit is not None and (time.time() - start_time) > time_limit:
            if verbose:
                print(f"\n时间限制达到 (迭代 {it})")
            termination_reason = "time_limit"
            final_iteration = it
            break
        
        # ===== 步骤1: 计算残差 =====
        r_p = A @ x - b                    # 原始可行性残差 (m维)
        r_d = AT @ y + s - c               # 对偶可行性残差 (n维)
        r_c = x * s                        # 互补松弛残差 (n维)
        mu = np.dot(x, s) / n              # 对偶间隙
        
        # 计算范数
        r_p_norm = np.linalg.norm(r_p)
        r_d_norm = np.linalg.norm(r_d)
        gap = np.sum(r_c)
        
        # 计算目标值（使用原始系数）
        if scale_columns:
            obj_val = np.dot(c_true, x / col_norms)
        else:
            obj_val = np.dot(c, x)
        
        # 更新最终残差信息
        final_r_p_norm = r_p_norm
        final_r_d_norm = r_d_norm
        final_mu = mu
        final_gap = gap
        final_iteration = it
        
        # ===== 步骤2: 收敛性检查 =====
        # 严格收敛
        if r_p_norm < tol_tight and r_d_norm < tol_tight and mu < tol_tight:
            if verbose:
                print(f"\n[OK] 严格收敛 (迭代 {it})")
            converged = True
            termination_reason = "optimal"
            break
        
        # 放松收敛
        if r_p_norm < tol_relaxed and r_d_norm < tol_relaxed and mu < tol_relaxed:
            if verbose:
                print(f"\n[OK] 放松收敛 (迭代 {it}): r_p={r_p_norm:.2e} < {tol_relaxed}")
            converged = True
            termination_reason = "optimal_relaxed"
            break
        
        # 停滞检测
        if np.isfinite(obj_val) and np.isfinite(prev_obj):
            obj_change = abs(obj_val - prev_obj) / (abs(prev_obj) + 1e-10)
        else:
            obj_change = float('inf')
        
        if obj_change < 1e-5 and r_d_norm < tol_tight and mu < tol_tight:
            stagnation_count += 1
            if stagnation_count >= 5:
                if verbose:
                    print(f"\n[WARNING] 停滞 (迭代 {it}): obj_change={obj_change:.2e}")
                converged = (r_p_norm < 1e-4) 
                termination_reason = "stagnated" if not converged else "optimal_stagnated"
                break
        else:
            stagnation_count = 0
        
        prev_obj = obj_val
        
        # 打印迭代信息
        if verbose and (it % log_every == 0):
            print(f"\r{it:<8} {obj_val:<15.2f} {r_p_norm:<12.2e} {r_d_norm:<12.2e} {mu:<12.2e} {stagnation_count:<6}", 
                  end='', flush=True)
        
        # ===== 步骤3: 设置中心化参数 =====
        # 简化的中心化参数设置
        if r_p_norm > 1e-3:
            # 原始残差较大时，使用较大的中心化参数
            sigma = 0.2
        elif mu > 1e-4:
            # 对偶间隙较大时，使用中等中心化参数
            sigma = 0.1
        else:
            # 接近收敛时，使用较小的中心化参数
            sigma = 0.05
        
        # ===== 步骤4-5: 构建权重矩阵和正规方程右端（最终优化） =====
        inv_s = 1.0 / s
        W = x * inv_s  # 权重向量 W = X·S^{-1}
        
        # 构建正规方程右端（最小化运算次数）
        # rhs_vector = x - sigma * mu * inv_s - x * (inv_s * r_d)
        #           = x - inv_s * (sigma * mu + x * r_d)
        rhs_vector = x - inv_s * (sigma * mu + x * r_d)
        rhs = -r_p + A @ rhs_vector
        
        # ===== 步骤6: 求解正规方程 M·Δy = rhs =====
        # 其中 M = A·W·A^T
        
        reg = initial_reg
        
        # 使用我们自己的CG实现（避免显式构造矩阵）
        # 预条件矩阵（对角）
        M_diag = (A * A) @ W + reg
        M_diag = np.maximum(M_diag, 1e-8)
        
        # 简化的CG容差调整
        if r_p_norm > 1e-3:
            # 原始残差较大时，使用宽松的CG容差
            cg_tol_current = cg_tol * 2.0
            cg_max_iter_current = int(cg_max_iter * 0.8)
        else:
            cg_tol_current = cg_tol
            cg_max_iter_current = cg_max_iter
        
        # CG迭代（优化版本）
        v = np.zeros(m)
        r = rhs.copy()
        z = r / M_diag
        p = z.copy()
        rz_old = np.dot(r, z)
        
        # 预计算rhs_norm（只计算一次）
        rhs_norm = np.linalg.norm(rhs)
        
        for k_cg in range(cg_max_iter_current):
            # 矩阵-向量积: Mp = (A·W·A^T + λI)·p
            ATp = AT @ p
            WATp = W * ATp
            Mp = (A @ WATp) + reg * p
            
            # 步长
            pMp = np.dot(p, Mp)
            if pMp <= 1e-18:
                # 避免除零，使用更小的步长
                alpha = 1e-10
            else:
                alpha = rz_old / pMp
            
            # 更新
            v += alpha * p
            r -= alpha * Mp
            
            # 预条件和更新搜索方向
            z = r / M_diag
            rz_new = np.dot(r, z)
            
            # 收敛检查（使用更高效的范数估计）
            if rz_new <= (cg_tol_current * rhs_norm) ** 2:
                break
            
            if rz_old <= 1e-18:
                beta = 0.0
            else:
                beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new
        
        dy = v
        
        # ===== 步骤7: 计算搜索方向（优化版本） =====
        ATdy = AT @ dy
        ds = -r_d - ATdy
    
        rhs_dx = -rhs_vector
        dx = W * ATdy + rhs_dx
        
        # 检查搜索方向是否合理
        if np.any(np.isnan(dx)) or np.any(np.isnan(dy)) or np.any(np.isnan(ds)):
            if verbose:
                print(f"\n[WARNING] 搜索方向包含NaN，增加正则化")
            reg *= 10.0
            continue
        
        # ===== 步骤8: 步长选择 =====
        # 简化的步长策略，添加收敛保护
        if r_p_norm > 1e-3:
            # 原始残差较大时，使用保守步长
            tau = 0.9
        elif r_p_norm > 1e-5:
            # 原始残差较小时，使用更保守的步长防止发散
            tau = 0.95
        elif mu > 1e-4:
            # 对偶间隙较大时，使用中等步长
            tau = 0.95
        else:
            # 接近收敛时，使用最保守的步长
            tau = 0.99
        alpha_pri = 1.0
        neg_dx = dx < 0
        if np.any(neg_dx):
            ratios = -tau * x[neg_dx] / dx[neg_dx]
            alpha_pri = float(np.min(ratios))
            alpha_pri = min(alpha_pri, 1.0)

        alpha_dual = 1.0
        neg_ds = ds < 0
        if np.any(neg_ds):
            ratios = -tau * s[neg_ds] / ds[neg_ds]
            alpha_dual = float(np.min(ratios))
            alpha_dual = min(alpha_dual, 1.0)

        # 确保最小步长，避免停滞
        alpha_pri = max(alpha_pri, 1e-6)
        alpha_dual = max(alpha_dual, 1e-6)

        # ===== 步骤9: 更新变量 =====
        x = np.maximum(x + alpha_pri * dx, 1e-12)
        y = y + alpha_dual * dy
        s = np.maximum(s + alpha_dual * ds, 1e-12)
    
    # 迭代结束，换行
    if verbose:
        print()
    
    # ========== 后处理 ==========
    # 还原列缩放
    if scale_columns:
        x = x / col_norms
        c_true = np.array(lp.c, dtype=float)
        obj = np.dot(c_true, x)
    else:
        obj = np.dot(c, x)
    
    # 构建求解信息字典
    if return_info:
        solve_time = time.time() - start_time
        info = {
            "converged": converged,
            "termination_reason": termination_reason,
            "iterations": final_iteration,
            "solve_time": solve_time,
            "primal_residual": float(final_r_p_norm),
            "dual_residual": float(final_r_d_norm),
            "complementarity": float(final_mu),
            "duality_gap": float(final_gap),
            "tolerance_tight": tol_tight,
            "tolerance_relaxed": tol_relaxed,
            "num_variables": n,
            "num_constraints": m,
        }
        return x.tolist(), float(obj), info
    else:
        return x.tolist(), float(obj)


def simplex_solve(lp: LPStandardForm) -> Tuple[List[float], float]:
    raise SolverError("simplex solver is a placeholder and not implemented yet")


SOLVER_REGISTRY = {
    "simplex": simplex_solve,
    "interior-point": interior_point_solve,
}


def solve(lp: LPStandardForm, method: str = "interior-point", **kwargs) -> Tuple[List[float], float]:
    if method not in SOLVER_REGISTRY:
        raise SolverError(f"Unknown solver method: {method}")
    solver = SOLVER_REGISTRY[method]
    return solver(lp, **kwargs) if method == "interior-point" else solver(lp)
