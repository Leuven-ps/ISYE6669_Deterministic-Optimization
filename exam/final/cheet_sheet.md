Problem 3. Which of the following statements is true about the extreme points and extreme rays?
A Any polyhedron has an extreme ray.
B If a nonempty polyhedron does not have any extreme point, then it is unbounded.
C For a nonempty polyhedron, any point in the polyhedron can be written as the non-zero sum of at least two different extreme points and some extreme rays.
Solution: B
A: bounded な多面体には extreme ray（方向）= 無限に伸びる方向 が存在しない。
だからこれは 間違い。
B:	•	bounded polyhedron → extreme points 必ず存在する（＝polytope）
	•	extreme point が無い → bounded ではない → 必ず unbounded
c: 空でない polyhedron 内の任意の点は、
bounded の場合 → extreme point の凸結合だけで表せる。
unbounded の場合 → extreme point の凸結合 + extreme ray の非負結合で表せる。

Problem5
minimize     3x₁ + 5x₂ + 7x₃
subject to   −x₁ + 3x₂ = 5
             2x₁ − x₂ + 3x₃ ≥ 6
             x₃ ≤ 7
             x₁ ≥ 0, x₂ ≤ 0, x₃ free

まず、各制約に対してラグランジュ乗数 y₁, y₂, y₃ を導入する。
制約の符号に基づき、それぞれの双対変数は次のようになる：

・等式制約（−x₁ + 3x₂ = 5）に対応する y₁ は符号制限なし（free）
・「≥」制約（2x₁ − x₂ + 3x₃ ≥ 6）に対応する y₂ は y₂ ≥ 0
・「≤」制約（x₃ ≤ 7）に対応する y₃ は y₃ ≤ 0

これらを用いてラグランジュ関数 L(x, y) を構成すると、

L(x, y) = 3x₁ + 5x₂ + 7x₃
          − y₁(−x₁ + 3x₂ − 5)
          − y₂(2x₁ − x₂ + 3x₃ − 6)
          − y₃(x₃ − 7)

展開し、x₁, x₂, x₃ に関する係数をまとめると、

L(x,y) = (3 + y₁ - 2y₂)x₁
         + (5 - 3y₁ + y₂)x₂
         + (7 - 3y₂ - y₃)x₃
         + 5y₁ + 6y₂ + 7y₃

次に、ラグランジュ双対関数 g(y) を求めるために、

    g(y) = infₓ L(x,y)

を考える。

この下界が有限になるためには、変数 x の符号制約により以下の条件が必要となる：

・x₁ ≥ 0 なので、係数 (3 + y₁ - 2y₂) ≥ 0
・x₂ ≤ 0 なので、係数 (5 - 3y₁ + y₂) ≤ 0
・x₃ は自由変数なので、係数 (7 - 3y₂ - y₃) = 0

これらの条件が満たされるとき、L(x,y) の inf は、係数部分が 0 になる点（つまり x=0）で達成される。
したがって、双対関数は次のように定まる：

g(y) = 5y₁ + 6y₂ + 7y₃

以上より、双対問題（Dual）は次のようになる：

maximize     5y₁ + 6y₂ + 7y₃
subject to   −y₁ + 2y₂ ≤ 3
             3y₁ − y₂ ≥ 5
             3y₂ + y₃ = 7
             y₁ free, y₂ ≥ 0, y₃ ≤ 0

problem6
弱相対定理から、主問題と相対問題のいずれか一方が非有解ならば他方は実行可能である。 
強双対定理から主問題に最適解xスターが存在すれば双対問題にも最適解yスターが存在し目的関数の値は一致する。

---

## Problem 9: 資材切出し問題と列生成法

**列生成法**: 変数が膨大な場合、全て列挙せず改善する変数のみ逐次生成（Gilmore & Gomory, 1961）

**主問題（整数計画）**:
```
最小化:  ∑ⱼ₌₁ⁿ xⱼ
条件:    ∑ⱼ₌₁ⁿ aᵢⱼ xⱼ ≥ dᵢ,  i = 1, ..., m
         ∑ᵢ₌₁ᵐ aᵢⱼ lᵢ ≤ L  (パターン条件)
         xⱼ ∈ ℤ₊
```

**双対問題**:
```
最大化:  ∑ᵢ₌₁ᵐ dᵢ yᵢ
条件:    ∑ᵢ₌₁ᵐ aᵢⱼ yᵢ ≤ 1,  j = 1, ..., k
         yᵢ ≥ 0
```

**列生成の流れ**:
1. 初期: パターン **p**ᵢ = (0,...,⌊L/lᵢ⌋,...,0) で開始
2. 双対最適解 **y*** を取得
3. 整数ナップサック問題を解く:
   ```
   最大化:  ∑ᵢ₌₁ᵐ yᵢ* aᵢ'
   条件:    ∑ᵢ₌₁ᵐ lᵢ aᵢ' ≤ L,  aᵢ' ∈ ℤ₊
   ```
4. 最適解 **p*** が ∑ᵢ₌₁ᵐ yᵢ* aᵢ* ≤ 1 なら終了、そうでなければ **p*** を追加して2へ

**動的計画法（整数ナップサック）**: l̄ = min lᵢ として
```
f(v) = max_{1≤i≤m, lᵢ≤v} { f(v-lᵢ) + yᵢ* },  v = l̄,...,L
f(v) = 0,  0 ≤ v < l̄
``` 