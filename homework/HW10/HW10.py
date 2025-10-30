"""
ISyE6669 Homework 10: Cutting Stock Problem with Column Generation
RMP (Restricted Master Problem) implementation using GurobiPy
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def solve_rmp(patterns: list, demands: list) -> dict:
    """
    RMPを解く関数
    
    Args:
        patterns: パターン行列のリスト [A1, A2, A3, ...]
        demands: 需要量ベクトル [b1, b2, b3]
    
    Returns:
        最適解、基底、基底の逆行列、双対解を含む辞書
    """
    # モデル作成
    model = gp.Model("RMP")
    model.setParam("OutputFlag", 0)  # ログ出力を抑制
    
    # 変数作成
    n_patterns = len(patterns)
    x = model.addVars(n_patterns, name="x", lb=0)
    
    # 目的関数: 最小化 sum(x_j)
    model.setObjective(gp.quicksum(x[j] for j in range(n_patterns)), GRB.MINIMIZE)
    
    # 制約条件: sum(A_j * x_j) = b
    for i in range(len(demands)):
        model.addConstr(
            gp.quicksum(patterns[j][i] * x[j] for j in range(n_patterns)) == demands[i],
            name=f"demand_{i+1}"
        )
    
    # 最適化実行
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise Exception(f"最適化に失敗しました。ステータス: {model.status}")
    
    # 最適解の取得
    optimal_x = [x[j].x for j in range(n_patterns)]
    
    # 基底の取得（Gurobiの内部実装に依存）
    # 注意: GurobiPyでは基底情報を直接取得するのは困難なため、
    # 手動で基底を特定する必要があります
    
    return {
        "optimal_x": optimal_x,
        "objective_value": model.objVal,
        "model": model
    }


def find_basis_and_dual(patterns: list, demands: list, optimal_x: list) -> dict:
    """
    基底と双対解を手動で計算する関数
    
    Args:
        patterns: パターン行列のリスト
        demands: 需要量ベクトル
        optimal_x: 最適解
    
    Returns:
        基底、基底の逆行列、双対解を含む辞書
    """
    # 非ゼロ変数のインデックスを取得
    basic_vars = [i for i, val in enumerate(optimal_x) if abs(val) > 1e-6]
    
    if len(basic_vars) != len(demands):
        raise Exception("基底変数の数が制約の数と一致しません")
    
    # 基底行列Bを構築
    B = np.array([[patterns[j][i] for j in basic_vars] for i in range(len(demands))])
    
    # 基底の逆行列を計算
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise Exception("基底行列が特異です")
    
    # 双対解を計算: y^T = c_B^T * B^(-1)
    # ここで c_B = [1, 1, ..., 1] (目的関数の係数)
    c_B = np.ones(len(basic_vars))
    dual_solution = c_B.T @ B_inv
    
    return {
        "basic_vars": basic_vars,
        "basis_matrix": B,
        "basis_inverse": B_inv,
        "dual_solution": dual_solution
    }


def solve_pricing_problem(dual_solution: list, widths: list, max_width: int) -> dict:
    """
    価格付け問題（ナップサック問題）を解く関数
    
    Args:
        dual_solution: 双対解 [y1, y2, y3]
        widths: 小ロールの幅 [w1, w2, w3]
        max_width: 大ロールの幅 W
    
    Returns:
        最適解と新しいパターンを含む辞書
    """
    model = gp.Model("Pricing")
    model.setParam("OutputFlag", 0)  # ログ出力を抑制
    
    # 変数作成: a1, a2, a3 (各幅の本数)
    a = model.addVars(3, name="a", vtype=GRB.INTEGER, lb=0)
    
    # 目的関数: sum(y_i * a_i) を最大化
    # 最適解の目的関数値が > 1 なら改善可能、<= 1 なら改善不可能
    model.setObjective(
        gp.quicksum(dual_solution[i] * a[i] for i in range(3)),
        GRB.MAXIMIZE
    )
    
    # 制約条件: sum(w_i * a_i) <= W
    model.addConstr(
        gp.quicksum(widths[i] * a[i] for i in range(3)) <= max_width,
        name="width_constraint"
    )
    
    # 最適化実行
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise Exception(f"価格付け問題の最適化に失敗しました。ステータス: {model.status}")
    
    # 最適解の取得
    optimal_a = [a[i].x for i in range(3)]
    objective_value = model.objVal
    
    return {
        "optimal_a": optimal_a,
        "objective_value": objective_value,
        "new_pattern": [int(optimal_a[i]) for i in range(3)]
    }


def column_generation_iteration(patterns: list, demands: list, widths: list, max_width: int) -> dict:
    """
    カラム生成の1回の反復を実行する関数
    
    Args:
        patterns: 現在のパターンリスト
        demands: 需要量
        widths: 小ロールの幅
        max_width: 大ロールの幅
    
    Returns:
        反復結果を含む辞書
    """
    print("=== カラム生成反復を実行中 ===")
    
    # RMPを解く
    rmp_result = solve_rmp(patterns, demands)
    print(f"RMP最適解: {rmp_result['optimal_x']}")
    print(f"RMP最適目的関数値: {rmp_result['objective_value']}")
    
    # 基底と双対解を計算
    basis_result = find_basis_and_dual(patterns, demands, rmp_result['optimal_x'])
    print(f"双対解: {basis_result['dual_solution']}")
    
    # 価格付け問題を解く
    pricing_result = solve_pricing_problem(
        basis_result['dual_solution'], widths, max_width
    )
    print(f"新しいパターン: {pricing_result['new_pattern']}")
    print(f"目的関数値: {pricing_result['objective_value']:.6f}")
    
    return {
        "rmp_result": rmp_result,
        "basis_result": basis_result,
        "pricing_result": pricing_result
    }


def full_column_generation(demands: list, widths: list, max_width: int, initial_patterns: list) -> dict:
    """
    完全なカラム生成アルゴリズムを実行する関数
    
    Args:
        demands: 需要量
        widths: 小ロールの幅
        max_width: 大ロールの幅
        initial_patterns: 初期パターン
    
    Returns:
        最終結果を含む辞書
    """
    patterns = initial_patterns.copy()
    iteration = 0
    max_iterations = 10  # 安全のための最大反復回数
    
    print("=== 完全なカラム生成アルゴリズムを開始 ===")
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 反復 {iteration} ---")
        
        # RMPを解く
        rmp_result = solve_rmp(patterns, demands)
        print(f"RMP最適解: {rmp_result['optimal_x']}")
        print(f"RMP最適目的関数値: {rmp_result['objective_value']}")
        
        # 基底と双対解を計算
        basis_result = find_basis_and_dual(patterns, demands, rmp_result['optimal_x'])
        print(f"双対解: {basis_result['dual_solution']}")
        
        # 価格付け問題を解く
        pricing_result = solve_pricing_problem(
            basis_result['dual_solution'], widths, max_width
        )
        print(f"新しいパターン: {pricing_result['new_pattern']}")
        print(f"目的関数値: {pricing_result['objective_value']:.6f}")
        
        # 終了条件のチェック: 目的関数値 <= 1 なら改善不可能
        if pricing_result['objective_value'] <= 1.0 + 1e-6:
            print("目的関数値 <= 1 のため、カラム生成を終了します")
            break
        else:
            print("目的関数値 > 1 のため、カラム生成を続行します")
            # 新しいパターンを追加
            patterns.append(pricing_result['new_pattern'])
            print(f"更新されたパターンリスト: {patterns}")
    
    if iteration >= max_iterations:
        print(f"警告: 最大反復回数 {max_iterations} に達しました")
    
    # 最終結果を返す
    final_rmp = solve_rmp(patterns, demands)
    final_basis = find_basis_and_dual(patterns, demands, final_rmp['optimal_x'])
    
    return {
        "final_patterns": patterns,
        "final_rmp": final_rmp,
        "final_basis": final_basis,
        "iterations": iteration
    }


def main():
    """メイン関数"""
    print("=== ISyE6669 Homework 10: Cutting Stock Problem ===\n")
    
    # 問題データ
    demands = [25, 15, 10]  # b1, b2, b3
    widths = [20, 35, 45]   # w1, w2, w3
    max_width = 100         # W
    initial_patterns = [
        [5, 0, 0],  # A1
        [0, 2, 0],  # A2
        [0, 0, 2]   # A3
    ]
    
    print("問題データ:")
    print(f"需要量: {demands}")
    print(f"小ロール幅: {widths}")
    print(f"大ロール幅: {max_width}")
    print(f"初期パターン: {initial_patterns}")
    print()
    
    # 完全なカラム生成を実行
    result = full_column_generation(demands, widths, max_width, initial_patterns)
    
    print("\n=== 最終結果 ===")
    print(f"最終パターン数: {len(result['final_patterns'])}")
    print(f"最終パターン:")
    for i, pattern in enumerate(result['final_patterns']):
        print(f"  パターン {i+1}: {pattern}")
    
    print(f"\n最終RMP最適解: {result['final_rmp']['optimal_x']}")
    print(f"最終RMP最適目的関数値: {result['final_rmp']['objective_value']}")
    
    print(f"\n最終基底変数: {result['final_basis']['basic_vars']}")
    print(f"最終基底行列 B:")
    print(result['final_basis']['basis_matrix'])
    print(f"\n最終基底の逆行列 B^(-1):")
    print(result['final_basis']['basis_inverse'])
    print(f"\n最終双対解 y^T: {result['final_basis']['dual_solution']}")
    
    print(f"\n総反復回数: {result['iterations']}")
    
    return result


if __name__ == "__main__":
    main()
