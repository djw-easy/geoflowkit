# MapTrix 静态布局实现方案

## 1. 范围与目标

本文档定义一个可实现的、静态的 MapTrix 布局器。它只处理以下内容：

- 两张地理底图与一个 OD 矩阵的空间布局；
- 地图区域到矩阵行、列的引导线；
- 行列排序、无交叉路由、线间距优化；
- 同一地点集合内部流与不同地点集合之间流的差异。

不包括矩阵颜色、比例符号、筛选、选择、高亮、动画或其他交互。它们不应影响本布局器输出的几何结果。

MapTrix 的基本思想是：不画每一条 OD 流的地图弧线。若有 `m` 个起点和 `n` 个终点，布局器只绘制 `m + n` 条**索引引导线**：起点区域连接到矩阵行、终点区域连接到矩阵列。矩阵单元格 `(i, j)` 才代表从起点 `i` 到终点 `j` 的具体流量。

本方案基于 Yang et al., *Many-to-Many Geographically-Embedded Flow Visualisation: An Evaluation*（2017）第 3 节的设计与优化目标。论文给出了两阶段机制：one-sided boundary labeling 生成无交叉次序，再以二次规划调整区域内连接点以增加线间距。论文未公开完整源码和参数，本文对未给出的工程细节提供明确、可替换的实现约定。

## 2. 术语与记号

| 记号 | 含义 |
| --- | --- |
| `O` | 起点集合，大小为 `m` |
| `D` | 终点集合，大小为 `n` |
| `F[i][j]` | 从 `O[i]` 到 `D[j]` 的流量；布局器不依赖其数值 |
| `rowOrder` | 起点在矩阵行中的排列 |
| `columnOrder` | 终点在矩阵列中的排列 |
| `site` | 一个行政区内部、引导线离开或进入地图的连接点 |
| `port` | 矩阵边界上与某行/列绑定的引导线端点 |
| `leader` | 由 `site` 连到 `port` 的地图—矩阵索引线 |
| `corridor` | 一组 leader 专属的布局走廊，防止起点线与终点线互相干扰 |

坐标一律使用屏幕/画布坐标：`x` 向右、`y` 向下，单位为 CSS 像素。

## 3. 数据契约

### 3.1 输入

```ts
type Point = { x: number; y: number };

type Region = {
  id: string;  // 稳定的唯一标识；不能使用展示名称作为 key
  geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon;
  // 可选。若省略，布局器用 point-on-surface 计算一个区域内部点。
  anchor?: Point;
};

type MapTrixInput = {
  origins: Region[];
  destinations: Region[];
  // 只用于验证矩阵尺寸；不参与 leader 布局。
  flow: number[][]; // flow.length = origins.length；flow[i].length = destinations.length
  width: number;
  height: number;
  sameEntitySet?: boolean;
};
```

`sameEntitySet` 的默认值应由 ID 集合推断，而不是由数量推断：

```ts
const sameEntitySet =
  origins.length === destinations.length &&
  origins.every(o => destinationIds.has(o.id));
```

### 3.2 输出

```ts
type Leader = {
  id: string;
  kind: "origin-row" | "column-destination";
  site: Point;
  bend: Point;
  port: Point;
  path: Point[];       // [site, bend, port]
  order: number;
};

type StaticMapTrixLayout = {
  rowOrder: string[];
  columnOrder: string[];
  rowPorts: Map<string, Point>;
  columnPorts: Map<string, Point>;
  originLeaders: Leader[];
  destinationLeaders: Leader[];
  mapRects: { origin: Rect; destination: Rect };
  matrix: MatrixGeometry;
};
```

## 4. 总体几何结构

推荐采用论文最终设计的静态构图：

```text
               origin map                     rotated OD matrix
            ┌─────────────┐                    ╱──────────╲
            │             │-- row leaders -\  ╱             ╲
            │             │                 \╱               ╲ 
            └─────────────┘                 ╱                 ╲   
                                            ╲                 ╱
            ┌─────────────┐                  ╲               ╱
            │ destination │                 / ╲             ╱
            │     map     │-- col leaders -/   ╲──────────╱
            └─────────────┘
```

在几何上，先构造一个普通的 `m × n` 正交矩阵，再整体绕中心旋转 `45°`。每个格保留其逻辑行、列下标；只有屏幕坐标旋转。这样能够让行端口、列端口落在菱形的两条不同边上。

推荐画布比例（可按容器等比例缩放）：

```ts
const mapWidth = 0.30 * width;
const gap = 0.04 * width;
const matrixBox = Math.min(0.56 * width, 0.80 * height);

const originMapRect = { x: 0.04 * width, y: 0.08 * height, w: mapWidth, h: 0.36 * height };
const destinationMapRect = { x: 0.04 * width, y: 0.56 * height, w: mapWidth, h: 0.36 * height };
const matrixCenter = { x: 0.70 * width, y: 0.50 * height };
```

这些是默认值，不是论文中的固定数值。布局器应把它们暴露为配置项。

## 5. 矩阵、端口与旋转

### 5.1 未旋转矩阵

在局部坐标中，单元格尺寸为 `s`：

```ts
const cellSize = Math.min(
  maxCellSize,
  Math.max(minCellSize, Math.min(matrixWidth / n, matrixHeight / m))
);
```

对于逻辑单元 `(row, col)`：

```ts
const cellCenter = {
  x: (col + 0.5) * cellSize,
  y: (row + 0.5) * cellSize,
};
```

每一行在左边界有一个端口；每一列在上边界有一个端口：

```ts
rowPortLocal(i) = { x: 0, y: (i + 0.5) * cellSize };
columnPortLocal(j) = { x: (j + 0.5) * cellSize, y: 0 };
```

### 5.2 旋转到画布坐标

设局部矩阵中心为 `(u0, v0)`，画布中心为 `(cx, cy)`，则：

```ts
function rotate45(p: Point, u0: number, v0: number, cx: number, cy: number): Point {
  const dx = p.x - u0;
  const dy = p.y - v0;
  const c = Math.SQRT1_2;
  return {
    x: cx + dx * c - dy * c,
    y: cy + dx * c + dy * c,
  };
}
```

SVG 可以更简单：用一个父 `<g>` 承担 `translate(cx, cy) rotate(45)`，但 leader 的 `port` 必须仍保存为已经变换后的画布坐标。

### 5.3 两条端口边界

- `rowSide`：所有 `rowPortLocal(i)` 旋转后形成的矩阵边界；连接 origin map。
- `columnSide`：所有 `columnPortLocal(j)` 旋转后形成的矩阵边界；连接 destination map。

两组端口不可复用，也不可让两组 leader 共享同一条 routing corridor。

## 6. 行、列排序规则

### 6.1 同一实体集合：相同排列

适用条件是：起点和终点的 ID 集合相同，例如“省 → 省”的人口迁移。

```ts
rowOrder = π;
columnOrder = π;
```

相同排列的目的不是让两组引导线更容易无交叉，而是保持矩阵语义：对角线为自循环，`F[i][j]` 与 `F[j][i]` 对称可比，区域块也更容易识别。引导线依旧按 origin 和 destination 两组各自求解。

### 6.2 不同实体集合：独立排列

适用条件包括：

- 国家 A → 国家 B；
- 港口 → 城市；
- 县 → 医院；
- 不同空间层级之间的流动。

```ts
rowOrder = πO;     // 只根据 origin map 与 rowSide 计算
columnOrder = πD;  // 只根据 destination map 与 columnSide 计算
```

即使 `m === n`，也不应因此共享排列。只有 ID 集合确实相同才使用同序。

### 6.3 可实现的 boundary-order 近似

论文使用 one-sided boundary labeling；若暂不实现完整算法，可采用以下稳定近似：把所有连接点投影到目标矩阵边界的切线方向，以投影顺序决定端口顺序。

```ts
function orderForSide(anchors: Array<{ id: string; point: Point }>, sideStart: Point, sideEnd: Point) {
  const dx = sideEnd.x - sideStart.x;
  const dy = sideEnd.y - sideStart.y;
  const len = Math.hypot(dx, dy);
  const tangent = { x: dx / len, y: dy / len };

  return anchors
    .slice()
    .sort((a, b) =>
      a.point.x * tangent.x + a.point.y * tangent.y -
      (b.point.x * tangent.x + b.point.y * tangent.y)
    )
    .map(a => a.id);
}
```

然后将端口按该顺序均匀放到对应边界：

```ts
function portsOnSide(ids: string[], start: Point, end: Point): Map<string, Point> {
  const ports = new Map<string, Point>();
  ids.forEach((id, i) => {
    const t = (i + 0.5) / ids.length;
    ports.set(id, {
      x: start.x + t * (end.x - start.x),
      y: start.y + t * (end.y - start.y),
    });
  });
  return ports;
}
```

这个排序应视为完整 boundary-labeling 求解器的初始化/后备方案。它在地图与矩阵分离、目标边界单侧可见时工作良好。

## 7. 引导线的两阶段布局

### 7.1 阶段 A：确定无交叉拓扑

对于 origin leaders 与 destination leaders 分别处理。每一组使用下列不变量：

1. 每个区域只绑定一个 port；
2. site 的顺序与 port 的顺序一致；
3. 路径在地图到矩阵方向上单调前进，不回折；
4. 同一组路径不换序；
5. 两组路径占据不同 corridor。

对于 `k` 条同组 leader，按 order 编号 `0..k-1`。若任意相邻 leader 的顺序不交换，则不存在同组交叉。完整 one-sided boundary labeling 正是在求满足这个条件且路径较短的端口匹配。

建议使用两段路径：

```text
site -- horizontal segment -- bend -- diagonal segment -- port
```

```ts
function routeLeader(site: Point, port: Point, slope: number): Point[] {
  // 斜段的斜率为 slope；符号由该 leader 所属的 up/down band 决定。
  const bend = {
    x: port.x - (port.y - site.y) / slope,
    y: site.y,
  };
  return [site, bend, port];
}
```

如果计算出的 `bend.x` 不在地图和矩阵之间，说明该 slope 或 band 不可行。此时应切换到另一条斜线带，或调整端口位置；不能直接让路径反向。

### 7.2 两个斜线带

论文的边界标注布局会生成斜向上和斜向下的 leader 带。工程实现中可以显式维护：

```ts
type Band = "up" | "down";
```

- `up` band：固定正或负斜率（按坐标系约定）；
- `down` band：使用相反斜率；
- 两带之间放一条分隔线；
- 每个 site 被约束在其 band 对应的分隔线一侧。

两带的作用是为复杂地理分布提供不同的绕行方向。简单地图常常只需要一个 band；如果一带导致 bend 出现回折、相邻距离过小或无法通过区域内部连接点，则才把该 leader 放入另一带。

### 7.3 Origin 与 destination 的隔离

origin leaders 与 destination leaders 是两个独立问题：

```text
origin map      -> row corridor    -> rowSide
destination map -> column corridor -> columnSide
```

具体要求：

- `rowSide` 和 `columnSide` 必须是矩阵两条不同的边；
- 两个 corridor 的包围盒不能重叠；
- 即使 origin、destination 是同一套区域，仍不要混合求解两组线。

这条约束能消除大多数“底图线与另一侧底图线互相交叉”的问题。

## 8. 阶段 B：连接点间距优化

仅有无交叉仍不够：leader 可能非常接近，或从行政区边缘穿过，导致难以辨认。论文允许 `site` 在所属区域内部小范围移动，并用二次规划优化。

### 8.1 初始连接点与可移动矩形

对每个区域 `i`，选择初始点：

```ts
c_i = region.anchor ?? pointOnSurface(region.geometry);
```

不要用普通 polygon centroid；凹多边形时它可能位于区域外。

然后在区域内部构造最大可行轴对齐矩形 `B_i`：

```text
B_i = [xmin_i, xmax_i] × [ymin_i, ymax_i]
```

论文的做法是从初始点开始，交替扩大宽度和高度，以二分搜索找到仍位于区域内的最大矩形。工程上可采用保守近似：

1. 在 `pointOnSurface` 周围从 2 px 开始扩张；
2. 每次检查四角和边中点是否仍在 polygon 内；
3. 到达边界后，以二分搜索缩回；
4. 再从矩形中移除会让 site 靠近其他 leader 的部分。

矩形约束为：

\[
x_{min,i} \le l_{x,i} \le x_{max,i}, \qquad
y_{min,i} \le l_{y,i} \le y_{max,i}
\]

### 8.2 目标函数

变量是优化后的连接点：

\[
l_i=(l_{x,i}, l_{y,i})
\]

第一项限制点偏离地图内部初始点：

\[
P_{centre} = \sum_i \left[(l_{x,i}-c_{x,i})^2+(l_{y,i}-c_{y,i})^2\right]
\]

对于同一斜率带中的相邻 leader `j`、`j+1`，论文使用下式表示平行斜段距离：

\[
d_j =
\frac{
k l_{x,j+1}-l_{y,j+1}-k l_{x,j}+l_{y,j}
}{\sqrt{k^2+1}}
\]

其中 `k` 是该带斜段的固定斜率。

第二项使相邻间距趋近目标距离 `D`：

\[
P_{sep}=\sum_{j=1}^{q-1}(d_j-D)^2
\]

总代价：

\[
\min \quad P_{centre}+wP_{sep}
\]

`w` 是权重。它大时，线更均匀但 site 离区域中心更远；它小时，连接点更贴近原位置但线可能拥挤。

### 8.3 硬约束

除 site 的区域内部矩形外，还需：

\[
d_j \ge \varepsilon
\]

其中 `ε` 是最小可读间距。它保证相邻 leader 不会换序，因此不产生交叉。

对于 up/down 两带，还要加入一个线性半平面约束，使两个带在分隔线两侧。例如将分隔线写为：

\[
a x + b y + c = 0
\]

则 up band 中的 site 满足：

\[
a l_x+b l_y+c \ge \delta
\]

down band 中的 site 满足：

\[
a l_x+b l_y+c \le -\delta
\]

`δ` 是两带的视觉安全距离。

### 8.4 参数起点

以下是适用于 1200–1800 px 宽画布的保守初值；应按缩放比例整体乘除：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `minLeaderGap ε` | 6 px | 同带相邻斜段的最低间距 |
| `targetGap D` | 10 px | 优化希望达到的线间距 |
| `bandSeparation δ` | 8 px | 两带之间的最小距离 |
| `w` | 2.0 | 分离目标相对中心保持目标的权重 |
| `siteBoxPadding` | 3 px | 连接点离行政区边界的安全距离 |
| `leaderStrokeWidth` | 1–1.5 px | 静态引导线线宽 |

当 `n > 40` 时，优先减小地图和标签，而不是把 `minLeaderGap` 降到 3 px 以下；否则引导线将不可追踪。

## 9. 无二次规划时的可替代算法

浏览器端首版可不引入 QP 求解器。使用离散候选点 + 局部搜索：

1. 在每个 `B_i` 内生成 25–81 个规则候选点；
2. 固定端口排序和 band；
3. 逐条 leader 尝试候选点；
4. 丢弃会导致 `d_j < ε`、bend 回折或越界的候选；
5. 从剩余候选中选取使相邻线最小距离最大的点；
6. 全部 leader 更新后重复 3–8 轮。

伪代码：

```ts
for (let iteration = 0; iteration < 6; iteration++) {
  for (const leader of leadersInOrder) {
    let best = leader.site;
    let bestScore = -Infinity;

    for (const candidate of samplePoints(leader.safeBox, 7)) {
      const candidatePath = routeLeader(candidate, leader.port, leader.slope);
      if (!isForward(candidatePath)) continue;
      if (crossesAny(candidatePath, fixedPaths)) continue;

      const score = minDistanceToAdjacentLeaders(candidatePath, fixedPaths)
        - 0.15 * squaredDistance(candidate, leader.initialSite);

      if (score > bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    leader.site = best;
  }
}
```

这不是论文的精确二次规划，但保持了论文的两个核心目标：不交叉、尽量均匀分离，同时避免连接点无意义地远离地理区域。

## 10. 完整布局伪代码

```ts
function layoutStaticMapTrix(input: MapTrixInput): StaticMapTrixLayout {
  validateInput(input);
  const same = input.sameEntitySet ?? inferSameEntitySet(input.origins, input.destinations);

  // 1. 投影地图并为每个区域选取一个区域内部初始 site。
  const originAnchors = projectAndAnchor(input.origins, "origin-map");
  const destinationAnchors = same
    ? cloneAnchorsForSecondMap(originAnchors, "destination-map")
    : projectAndAnchor(input.destinations, "destination-map");

  // 2. 计算矩阵与两条端口边。
  const matrix = createRotatedMatrix(input.origins.length, input.destinations.length, input.width, input.height);

  // 3. 计算行列次序。
  const rowOrder = orderForSide(originAnchors, matrix.rowSide.start, matrix.rowSide.end);
  const columnOrder = same
    ? rowOrder.slice()
    : orderForSide(destinationAnchors, matrix.columnSide.start, matrix.columnSide.end);

  // 4. 为每个 ID 分配矩阵边界端口。
  const rowPorts = portsOnSide(rowOrder, matrix.rowSide.start, matrix.rowSide.end);
  const columnPorts = portsOnSide(columnOrder, matrix.columnSide.start, matrix.columnSide.end);

  // 5. 分别建立起点和终点 leader；两组绝不混排。
  const originLeaders = initializeLeaders(originAnchors, rowPorts, matrix.rowCorridor);
  const destinationLeaders = initializeLeaders(destinationAnchors, columnPorts, matrix.columnCorridor);

  // 6. 为每组求 band、有效矩形和无交叉拓扑。
  solveTopology(originLeaders);
  solveTopology(destinationLeaders);
  createSafeBoxes(originLeaders);
  createSafeBoxes(destinationLeaders);

  // 7. QP 或候选点优化。优化仍按两组独立进行。
  optimizeSites(originLeaders);
  optimizeSites(destinationLeaders);

  // 8. 使用最终 site、band 和 port 重建 SVG/polyline path。
  reroute(originLeaders);
  reroute(destinationLeaders);

  assertNoCrossings(originLeaders);
  assertNoCrossings(destinationLeaders);
  assertCorridorsDisjoint(originLeaders, destinationLeaders);

  return { rowOrder, columnOrder, rowPorts, columnPorts, originLeaders, destinationLeaders, matrix };
}
```

## 11. 失败处理与降级规则

### 11.1 端口太密

若端口间距小于 `minLeaderGap`：

1. 增加矩阵边长；
2. 扩大画布或允许横向滚动；
3. 减小矩阵单元格，但不得让端口间距低于 `ε`；
4. 不要仅靠缩小 leader 线宽掩盖问题。

### 11.2 行政区过小或狭长

若区域内无法容纳可移动矩形：

- 固定 site 为 `pointOnSurface`；
- 禁止该 site 的二次优化移动；
- 允许其余 site 优化；
- 必要时在区域外放置锚点，但需增加短的地图标注引线；这已超出论文的原始假设，应作为明确的降级模式。

### 11.3 两个实体集合存在部分重合

例如“港口 → 城市”中少数对象名称相同。除非两边的 ID、地理语义和集合全部相同，否则按不同集合处理，行列独立排序。

### 11.4 路由不可行

当 `bend` 位于走廊外或路径回折时，按顺序尝试：

1. 调整该 leader 的 slope；
2. 将该 leader 切换到另一斜线 band；
3. 调整端口的局部位置，但保持端口顺序；
4. 扩大 corridor；
5. 最后才考虑改变全局排序。

## 12. 验收测试

布局器至少应通过下列自动化测试。

### 数据与语义

- `flow` 维度与 origin/destination 数量一致；
- 同集合数据中，`rowOrder` 与 `columnOrder` 的 ID 序列完全相同；
- 不同集合数据中，行、列分别只包含对应集合的全部 ID，且无重复。

### 几何

- 每个 site 位于对应行政区内部，或被标记为明确的降级锚点；
- 每个 origin ID 恰有一条 row leader；
- 每个 destination ID 恰有一条 column leader；
- 每条 leader 的最后一个点精确等于绑定 port；
- leader 在地图到矩阵方向单调前进；
- 同组任意两条 leader 无严格线段交叉；
- 相邻同带 leader 的距离不小于 `ε`；
- origin 与 destination corridor 不重叠。

### 视觉回归

至少准备 4 套固定数据渲染为 PNG 并做像素/快照回归：

1. 8 个区域的同集合内部流；
2. 16 个区域、形状接近方形的同集合流；
3. 16 个区域、狭长国家形状的同集合流；
4. `m != n` 的跨国/跨集合流。

## 13. 实现优先级

建议按下列顺序实施：

1. 标准未旋转矩阵、两张投影地图、固定连接点；
2. 矩阵旋转、rowSide 和 columnSide 端口；
3. 同集合共享排序、不同集合独立排序；
4. 投影排序 + 单调折线路由；
5. 两个独立 corridor 和无交叉断言；
6. 行政区内部安全矩形；
7. 离散候选点优化；
8. 若需要严格贴近论文，再替换为二次规划优化器。

达到第 5 步即可形成正确的 MapTrix 引导线骨架；第 6–8 步决定大规模、密集区域时的可读性。

