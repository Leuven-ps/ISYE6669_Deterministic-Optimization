import cvxpy as cp
import gurobipy as gb


def main() -> None:
    """水道ネットワーク最適化問題のメイン関数"""
    
    # ===== 水道ネットワーク問題パラメータの定義 =====
    
    # 水源（アルファベット表記）
    WATER_SOURCES = ["A", "B", "C"]
    
    # 住宅（アルファベット表記）
    HOUSES = ["D", "E", "F", "G", "H", "I"]
    
    # パイプライン（数字表記）
    PIPELINES = list(range(1, 13))+[-5,-6,-7,-9]  # Pipe 1 through 12
    
    # 水源の最大容量
    SOURCE_CAPACITIES = {
        "A": 100,  # Units
        "B": 100,  # Units
        "C": 120   # Units
    }
    
    # 住宅の水需要
    HOUSE_DEMANDS = {
        "D": 50,   # Units
        "E": 60,   # Units
        "F": 40,   # Units
        "G": 30,   # Units
        "H": 70,   # Units
        "I": 40    # Units
    }
    
    # パイプラインの輸送コスト（単位水あたり）
    PIPELINE_COSTS = {
        1: 2,   # $2 per unit
        2: 3,   # $3 per unit
        3: 4,   # $4 per unit
        4: 2,   # $2 per unit
        5: 3,   # $3 per unit
        5: 3,   # $3 per unit
        -5: 3,  # $3 per unit (パイプライン5の逆方向)
        6: 2,   # $2 per unit
        -6: 2,  # $2 per unit (パイプライン6の逆方向)
        7: 4,   # $4 per unit
        -7: 4,  # $4 per unit (パイプライン7の逆方向)
        8: 1,   # $1 per unit
        9: 2,   # $2 per unit
        -9: 2,  # $2 per unit (パイプライン9の逆方向)
        10: 4,  # $4 per unit
        11: 5,  # $5 per unit
        12: 3   # $3 per unit
    }
    
    # 水源とパイプラインの接続関係
    # 各水源から出るパイプラインのリスト
    SOURCE_TO_PIPES = {
        "A": [1, 3],      # 水源Aから出るパイプライン
        "B": [2, 4,12],      # 水源Bから出るパイプライン
        "C": [8,10,11],       # 水源Cから出るパイプライン
    }
    
    # 住宅とパイプラインの接続関係
    # 各住宅に接続するパイプラインのリスト（流入、流出）
    DEMAND_TO_PIPES = {
        "D": ([1, 2], []),  # 住宅D: 流入パイプライン, 流出パイプライン
        "E": ([3, 7, -5], [5,-7]),      # 住宅E: 流入パイプライン, 流出パイプライン
        "F": ([4, 5, 6, 8], [-5, -6]),  # 住宅F: 流入パイプライン, 流出パイプライン
        "G": ([-6, 9, 12], [6, -9]),      # 住宅G: 流入パイプライン, 流出パイプライン
        "H": ([-7, 10], [7]),     # 住宅H: 流入パイプライン, 流出パイプライン
        "I": ([-9, 11], [9])      # 住宅I: 流入パイプライン, 流出パイプライン
    }
    
    model = gb.Model()
    x = model.addVars(PIPELINES, lb=0, name="x")

    for source in WATER_SOURCES:
            model.addConstr(gb.quicksum(x[pipe] for pipe in SOURCE_TO_PIPES[source]) <= SOURCE_CAPACITIES[source], name=f"x_{source}_capacity")
    for demand in HOUSES:
        inflow_pipes, outflow_pipes = DEMAND_TO_PIPES[demand]
        flow = gb.quicksum(x[pipe] for pipe in inflow_pipes) - gb.quicksum(x[pipe] for pipe in outflow_pipes)
        model.addConstr(flow >= HOUSE_DEMANDS[demand], name=f"demand_{demand}")

    model.setObjective(gb.quicksum(x[pipe] * PIPELINE_COSTS[pipe] for pipe in PIPELINES), sense=gb.GRB.MINIMIZE)
    model.optimize()
    print("Optimal solution:")
    for pipe in PIPELINES:
        print(f"  x[{pipe}] = {x[pipe].x}")
    print(f"Optimal value: {model.objVal}")
    
    
    print("=== 水道ネットワーク最適化問題 ===")
    print(f"水源: {WATER_SOURCES}")
    print(f"住宅: {HOUSES}")
    print(f"パイプライン: {PIPELINES}")
    print(f"水源容量: {SOURCE_CAPACITIES}")
    print(f"住宅需要: {HOUSE_DEMANDS}")
    print(f"パイプラインコスト: {PIPELINE_COSTS}")
    print()
    
    print("=== 水源とパイプラインの接続 ===")
    for source, pipes in SOURCE_TO_PIPES.items():
        print(f"水源 {source} → パイプライン {pipes}")
    print()
    
    print("=== 住宅とパイプラインの接続 ===")
    for demand, pipes in DEMAND_TO_PIPES.items():
        print(f"住宅 {demand} → パイプライン {pipes}")
    print()
    
    # 総需要の計算
    total_demand = sum(HOUSE_DEMANDS.values())
    total_capacity = sum(SOURCE_CAPACITIES.values())
    
    print(f"総需要: {total_demand} units")
    print(f"総容量: {total_capacity} units")
    print(f"需要充足可能性: {'可能' if total_capacity >= total_demand else '不可能'}")
    print()
    
    # TODO: CVXPYを使用した線形計画問題の実装
    # TODO: 最適解の計算
    # TODO: 最小コストの計算


if __name__ == "__main__":
    main()
