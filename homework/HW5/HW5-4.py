import cvxpy as cp
import gurobipy as gb


def main() -> None:
    """電力システム最適化問題のメイン関数"""
    
    # ===== 問題パラメータの定義 =====
    
    # ノード数と発電機・負荷の配置
    NUM_NODES = 6
    GENERATOR_NODES = [1, 3, 5]  # 発電機があるノード（1-indexed）
    LOAD_NODES = [2, 4, 6]       # 負荷があるノード（1-indexed）
    
    # 電力需要（固定値）
    DEMANDS = {
        2: 120,  # d_1
        4: 95,   # d_2  
        6: 105   # d_3
    }
    
    # 発電機の容量制限
    GENERATION_BOUNDS = {
        1: {"min": 20, "max": 270},   # p_1^{min}, p_1^{max}
        3: {"min": 20, "max": 250},   # p_2^{min}, p_2^{max}
        5: {"min": 10, "max": 300}    # p_3^{min}, p_3^{max}
    }
    
    # 送電線の容量制限
    LINE_FLOW_LIMITS = {
        (1, 2): 100,  # f_{12}^{max}
        (2, 3): 120,  # f_{23}^{max}
        (3, 4): 50,   # f_{34}^{max}
        (4, 5): 90,   # f_{45}^{max}
        (5, 6): 60,   # f_{56}^{max}
        (6, 1): 50    # f_{61}^{max}
    }
    
    # 送電線パラメータ（サセプタンス）
    LINE_SUSCEPTANCE = {
        (1, 2): 11.6,  # B_{12}
        (2, 3): 5.9,   # B_{23}
        (3, 4): 13.7,  # B_{34}
        (4, 5): 9.8,   # B_{45}
        (5, 6): 5.6,   # B_{56}
        (6, 1): 10.5   # B_{61}
    }
    
    # 発電コスト
    GENERATION_COSTS = {
        1: 10,  # c_1
        3: 5,   # c_2
        5: 8    # c_3
    }
    
    # 送電線のリスト
    LINES = list(LINE_FLOW_LIMITS.keys())

    model = gb.Model()
    
    # 変数定義
    f = model.addVars(LINES, lb=-gb.GRB.INFINITY, name="f")
    p = model.addVars(GENERATOR_NODES, lb=0.0, name="p")
    theta = model.addVars(range(1, NUM_NODES+1), lb=-gb.GRB.INFINITY, name="theta")

    # 参照ノード制約
    model.addConstr(theta[1] == 0, name="ref_bus")

    # 発電機制約
    for node in GENERATOR_NODES:
        # 発電量の上下限
        model.addConstr(p[node] >= GENERATION_BOUNDS[node]["min"], name=f"p_{node}_min")
        model.addConstr(p[node] <= GENERATION_BOUNDS[node]["max"], name=f"p_{node}_max")

        # 発電機の電力バランス
        outflow = 0
        inflow = 0
        for start, end in LINES:
            if start == node:
                outflow += f[(start, end)]
            if end == node:
                inflow += f[(start, end)]
        model.addConstr(inflow - outflow == -p[node], name=f"p_{node}_balance") 
    
    # 負荷制約
    for node in LOAD_NODES:
        # 負荷の電力バランス
        inflow = 0
        outflow = 0
        for start, end in LINES:
            if end == node:
                inflow += f[(start, end)]
            if start == node:
                outflow += f[(start, end)]
        model.addConstr(inflow - outflow == DEMANDS[node], name=f"d_{node}_balance")

    # 送電線制約
    for line in LINES:
        start, end = line
        limit = LINE_FLOW_LIMITS[line]
        susceptance = LINE_SUSCEPTANCE[line]
        
        # フロー容量制限
        model.addConstr(f[line] >= -limit, name=f"f_{line}_min")
        model.addConstr(f[line] <= limit, name=f"f_{line}_max")

        # DC電力フロー制約
        model.addConstr(f[line] == susceptance * (theta[start] - theta[end]), name=f"theta_{line}_nodal")

    # 目的関数
    model.setObjective(gb.quicksum(p[node] * GENERATION_COSTS[node] for node in GENERATOR_NODES), sense=gb.GRB.MINIMIZE)
    
    # 最適化実行
    model.optimize()
    
    if model.status == gb.GRB.OPTIMAL:
        print("=== 最適解 ===")
        print("発電量:")
        for node in GENERATOR_NODES:
            print(f"  p[{node}] = {p[node].x:.2f} MW")
        
        print("\n送電線フロー:")
        for line in LINES:
            print(f"  f{line} = {f[line].x:.2f} MW")
        
        print("\n位相角:")
        for node in range(1, NUM_NODES+1):
            print(f"  theta[{node}] = {theta[node].x:.4f} rad")
            
        print(f"\n最適値（総コスト）: ${model.objVal:.2f}")
        
        # デュアル変数（電気料金）の取得
        print("\n=== 電気料金（デュアル変数） ===")
        for node in LOAD_NODES:
            constraint = model.getConstrByName(f"d_{node}_balance")
            if constraint:
                dual_value = constraint.pi
                print(f"ノード {node} の電気料金: ${abs(dual_value):.2f}/MWh")
                
    elif model.status == gb.GRB.INFEASIBLE:
        print("モデルが実行不可能です。")
        model.computeIIS()
        print("IIS constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  {c.constrName}")
    else:
        print(f"最適化が失敗しました。ステータス: {model.status}")

    print("\n=== 問題設定 ===")
    print(f"発電機ノード: {GENERATOR_NODES}")
    print(f"負荷ノード: {LOAD_NODES}")
    print(f"電力需要: {DEMANDS}")
    print(f"発電機容量制限: {GENERATION_BOUNDS}")
    print(f"送電線容量制限: {LINE_FLOW_LIMITS}")
    print(f"発電コスト: {GENERATION_COSTS}")


if __name__ == "__main__":
    main()