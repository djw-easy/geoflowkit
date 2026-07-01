"""Genetic-programming layout optimiser for MapTrix guide lines.

Optimises one side (origin map or destination map) independently, searching
for a zone ordering and per-zone map connection points that produce
crossing-free, evenly-spaced guide lines.

References
----------
.. [1] Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
.. [2] Yang, X., Zhu, D., Guo, D., Liu, C., & Ye, X. (2016). MapTrix.
"""

import numpy as np
from shapely.geometry import Point

from geoflowkit.visualization._utils import (
    _ax_to_fig,
    _fig_to_ax,
    _calculate_matrix_anchor_point,
)


# ---------------------------------------------------------------------------
# Guide-line geometry (no split-line dependency)
# ---------------------------------------------------------------------------

def gp_polyline_geometry(map_fig, matrix_fig, angle_deg=45.0):
    """Compute L-shaped guide line geometry *without* a split line.

    The guide line consists of a diagonal segment at *angle_deg* from the
    horizontal, followed by a horizontal segment to the matrix anchor.

    Direction is auto-detected from the map point / matrix anchor positions:
    - ``my > py``  → diagonal goes upward   → ``is_upper = True``
    - ``my < py``  → diagonal goes downward → ``is_upper = False``

    Parameters
    ----------
    map_fig : tuple
        (px, py) in figure coordinates.
    matrix_fig : tuple
        (mx, my) in figure coordinates.
    angle_deg : float
        Diagonal angle relative to horizontal.

    Returns
    -------
    geom : dict or None
        ``{"p": (px, py), "q": (cross_x, cross_y), "m": (mx, my),
           "is_upper": bool, "diag": (p, q), "horiz": (q, m)}``
        or ``None`` when no valid geometry exists.
    """
    px, py = map_fig
    mx, my = matrix_fig
    tan_k = np.tan(np.deg2rad(angle_deg))
    eps = 1e-10

    if abs(tan_k) < eps:
        return None

    dy = my - py
    if abs(dy) < eps:
        return None

    is_upper = dy > 0
    gap = abs(dy)

    cross_x = px + gap / tan_k
    cross_y = my

    if not (min(px, mx) - eps < cross_x < max(px, mx) + eps):
        return None

    p = (float(px), float(py))
    q = (float(cross_x), float(cross_y))
    m = (float(mx), float(my))

    return {
        "p": p, "q": q, "m": m,
        "is_upper": is_upper,
        "diag": (p, q),
        "horiz": (q, m),
    }


# ---------------------------------------------------------------------------
# Segment intersection (standalone, no class dependency)
# ---------------------------------------------------------------------------

def _orient(a, b, c):
    return ((b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, c, eps=1e-10):
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        and abs(_orient(a, b, c)) <= eps
    )


def _segments_intersect(s1, s2, eps=1e-10):
    """Return True if segments s1 and s2 intersect."""
    a, b = s1
    c, d = s2

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True

    return False


def _same_point(a, b, eps=1e-8):
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def gp_geometries_cross(g1, g2):
    """Return True if two L-shaped guide lines cross.

    Shared endpoints are ignored.
    """
    if g1 is None or g2 is None:
        return True

    segs1 = [g1["diag"], g1["horiz"]]
    segs2 = [g2["diag"], g2["horiz"]]

    for s1 in segs1:
        for s2 in segs2:
            if not _segments_intersect(s1, s2):
                continue
            # Ignore exact shared endpoints
            shared = any(
                _same_point(p, q) for p in s1 for q in s2
            )
            if not shared:
                return True

    return False


# ======================================================================
# Genetic-programming layout optimiser
# ======================================================================

class GPLayoutOptimizer:
    """Optimise MapTrix guide-line layout for one side via genetic algorithm.

    Each chromosome encodes:

    - ``order_keys`` : float array of length *N* in [0, 1].  Sorting gives
      the zone → matrix-column/row assignment.
    - ``candidate_idx`` : int array of length *N*.  Each entry picks which
      sampled candidate point to use for the corresponding zone.

    Parameters
    ----------
    angle_deg : float
        Guide-line diagonal angle in degrees (default 45).
    pop_size : int
        Population size (default 200).
    max_generations : int
        Maximum generations (default 100).
    center_weight : float
        Fitness weight for centroid deviation.
    spacing_weight : float
        Fitness weight for non-uniform elbow spacing.
    crossing_penalty : float
        Penalty applied when *any* pair of guide lines crosses (default 1e8).
    mutation_prob : float
        Per-gene mutation probability.
    elite_count : int
        Number of elite individuals preserved each generation.
    tournament_size : int
        Tournament selection size.
    patience : int
        Early-stop generations without improvement.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, angle_deg=45.0, pop_size=200, max_generations=100,
                 center_weight=1.0, spacing_weight=8.0,
                 crossing_penalty=1e8,
                 mutation_prob=0.15, elite_count=2, tournament_size=3,
                 patience=30, random_state=None):
        self.angle_deg = angle_deg
        self.pop_size = max(pop_size, elite_count + 2)
        self.max_generations = max_generations
        self.center_weight = center_weight
        self.spacing_weight = spacing_weight
        self.crossing_penalty = crossing_penalty
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.patience = patience

        self._rng = np.random.RandomState(random_state)

        self._best_fitness_ = None
        self._best_gen_ = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def optimize(self, zone_ids, zone_geometries, zone_centroids_fig,
                 ax_map, ax_matrix, fig, matrix_shape, transform,
                 is_origin, linewidths=None):
        """Run the genetic algorithm and return the best layout.

        Parameters
        ----------
        zone_ids : list
            Zone identifiers (length *N*).
        zone_geometries : dict
            Mapping zone ID → shapely geometry (polygon).
        zone_centroids_fig : dict
            Mapping zone ID → ``(fig_x, fig_y)`` centroid in figure coords.
        ax_map : Axes
            The origin or destination map axes.
        ax_matrix : Axes
            The rotated OD matrix axes.
        fig : Figure
            The matplotlib figure.
        matrix_shape : tuple
            ``(n_rows, n_cols)`` of the rotated OD matrix.
        transform : Transform
            The affine transform used for matrix rotation.
        is_origin : bool
            ``True`` for origin side (top edge), ``False`` for destination
            (left edge).
        linewidths : dict or None
            Mapping zone ID → line width.

        Returns
        -------
        result : dict
            ``ordered_zone_ids``, ``positions_fig``, ``geoms``, ``linewidths``,
            ``fitness``, ``generations``.
        """
        N = len(zone_ids)
        if N == 0:
            return {
                "ordered_zone_ids": [],
                "positions_fig": {},
                "geoms": {},
                "linewidths": {},
                "fitness": 0.0,
                "generations": 0,
            }

        self._side = "top" if is_origin else "left"
        self._n_rows, self._n_cols = matrix_shape
        self._ax_map = ax_map
        self._fig = fig

        # Pre-compute candidate map points for every zone
        self._zone_candidates = self._build_all_candidates(
            zone_ids, zone_geometries, zone_centroids_fig, ax_map, fig,
        )
        self._N = N
        self._zone_ids = np.array(zone_ids)
        self._linewidths = linewidths or {}
        self._centroids_fig = zone_centroids_fig

        # Pre-compute all matrix anchor figure coords for every position
        self._matrix_anchors_fig = self._precompute_matrix_anchors(
            fig, ax_matrix, transform,
        )

        # Build initial population
        pop_chromosomes, pop_fitness = self._initialize_population()

        # Evolution loop
        best_fitness = pop_fitness.min()
        best_idx = pop_fitness.argmin()
        best_chromosome = {
            "order_keys": pop_chromosomes["order_keys"][best_idx],
            "candidate_idx": pop_chromosomes["candidate_idx"][best_idx],
        }
        no_improve = 0

        for gen in range(self.max_generations):
            new_chromosomes = {
                "order_keys": np.empty((self.pop_size, N)),
                "candidate_idx": np.empty((self.pop_size, N), dtype=int),
            }

            # Elitism
            elite_indices = self._tournament_select(pop_fitness, self.elite_count)
            for i, ei in enumerate(elite_indices):
                new_chromosomes["order_keys"][i] = \
                    pop_chromosomes["order_keys"][ei].copy()
                new_chromosomes["candidate_idx"][i] = \
                    pop_chromosomes["candidate_idx"][ei].copy()

            # Fill rest via tournament + crossover + mutation
            for i in range(self.elite_count, self.pop_size, 2):
                p1 = np.random.choice(pop_chromosomes["order_keys"].shape[0])
                p2 = np.random.choice(pop_chromosomes["order_keys"].shape[0])
                while p2 == p1:
                    p2 = np.random.choice(pop_chromosomes["order_keys"].shape[0])

                c1_ok, c1_ci = self._crossover(
                    pop_chromosomes["order_keys"][p1],
                    pop_chromosomes["order_keys"][p2],
                    pop_chromosomes["candidate_idx"][p1],
                    pop_chromosomes["candidate_idx"][p2],
                )
                c2_ok, c2_ci = self._crossover(
                    pop_chromosomes["order_keys"][p2],
                    pop_chromosomes["order_keys"][p1],
                    pop_chromosomes["candidate_idx"][p2],
                    pop_chromosomes["candidate_idx"][p1],
                )

                c1_ok = self._mutate(c1_ok, c1_ci)
                if i < self.pop_size:
                    new_chromosomes["order_keys"][i] = c1_ok
                    new_chromosomes["candidate_idx"][i] = c1_ci

                c2_ok = self._mutate(c2_ok, c2_ci)
                if i + 1 < self.pop_size:
                    new_chromosomes["order_keys"][i + 1] = c2_ok
                    new_chromosomes["candidate_idx"][i + 1] = c2_ci

            # Evaluate
            pop_chromosomes = new_chromosomes
            pop_fitness = np.array([
                self._fitness(
                    pop_chromosomes["order_keys"][j],
                    pop_chromosomes["candidate_idx"][j],
                )
                for j in range(self.pop_size)
            ])

            gen_best = pop_fitness.min()
            if gen_best < best_fitness - 1e-10:
                best_fitness = gen_best
                best_idx = pop_fitness.argmin()
                best_chromosome = {
                    "order_keys": pop_chromosomes["order_keys"][best_idx].copy(),
                    "candidate_idx": pop_chromosomes["candidate_idx"][best_idx].copy(),
                }
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        self._best_fitness_ = best_fitness
        self._best_gen_ = gen + 1

        # Decode best chromosome → result
        return self._decode_chromosome(
            best_chromosome["order_keys"],
            best_chromosome["candidate_idx"],
            fitness=best_fitness,
            generations=self._best_gen_,
        )

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _initialize_population(self):
        N = self._N
        order_keys = np.empty((self.pop_size, N))
        candidate_idx = np.empty((self.pop_size, N), dtype=int)

        for i in range(self.pop_size):
            order_keys[i] = self._rng.rand(N)

        for i in range(self.pop_size):
            for j in range(N):
                n_cands = len(self._zone_candidates[self._zone_ids[j]])
                if n_cands > 0:
                    candidate_idx[i, j] = self._rng.randint(0, n_cands)
                else:
                    candidate_idx[i, j] = 0

        # Seed one individual with centroid-Y ordering
        centroids = self._centroids_fig
        y_sorted = sorted(
            range(N),
            key=lambda j: centroids[self._zone_ids[j]][1],
        )
        seed_keys = np.zeros(N)
        for rank, j in enumerate(y_sorted):
            seed_keys[j] = (rank + 0.5) / N
        order_keys[0] = seed_keys

        # Seed another with centroid-Y reversed
        seed_keys_rev = np.zeros(N)
        for rank, j in enumerate(reversed(y_sorted)):
            seed_keys_rev[j] = (rank + 0.5) / N
        if self.pop_size >= 2:
            order_keys[1] = seed_keys_rev

        fitness = np.array([
            self._fitness(order_keys[j], candidate_idx[j])
            for j in range(self.pop_size)
        ])
        return {"order_keys": order_keys, "candidate_idx": candidate_idx}, fitness

    # ------------------------------------------------------------------
    # Candidate sampling
    # ------------------------------------------------------------------

    def _build_all_candidates(self, zone_ids, zone_geometries,
                              zone_centroids_fig, ax_map, fig, grid_size=31):
        """Pre-compute candidate map points in figure coords for every zone."""
        candidates = {}
        for zid in zone_ids:
            pts = self._sample_zone_candidate_points(
                zid, zone_geometries, zone_centroids_fig,
                ax_map, fig, grid_size,
            )
            candidates[zid] = pts if pts else [zone_centroids_fig.get(zid, (0.5, 0.5))]
        return candidates

    def _sample_zone_candidate_points(self, zid, zone_geometries,
                                       centroids_fig, ax_map, fig, grid_size):
        geom = zone_geometries.get(zid) if zone_geometries else None
        if geom is None or geom.is_empty:
            raw = centroids_fig.get(zid)
            return [raw] if raw else []

        mnx, mny, mxx, mxy = geom.bounds
        xs = np.linspace(mnx, mxx, grid_size)
        ys = np.linspace(mny, mxy, grid_size)

        pts = []
        rp = geom.representative_point()
        pts.append(_ax_to_fig(ax_map, fig, rp.x, rp.y))

        c = geom.centroid
        if geom.contains(c) or geom.touches(c):
            pts.append(_ax_to_fig(ax_map, fig, c.x, c.y))

        for x in xs:
            for y in ys:
                pt = Point(float(x), float(y))
                if geom.contains(pt) or geom.touches(pt):
                    pts.append(_ax_to_fig(ax_map, fig, x, y))

        # De-duplicate
        unique, seen = [], set()
        for px, py in pts:
            key = (round(float(px), 8), round(float(py), 8))
            if key not in seen:
                seen.add(key)
                unique.append((float(px), float(py)))
        return unique

    def _precompute_matrix_anchors(self, fig, ax_matrix, transform):
        """Pre-compute matrix anchor points in figure coords for every position."""
        n = self._N
        anchors = []
        for pos in range(n):
            mx, my = _calculate_matrix_anchor_point(
                transform, self._n_rows, self._n_cols,
                side=self._side, index=pos,
            )
            fx, fy = _ax_to_fig(ax_matrix, fig, mx, my)
            anchors.append((fx, fy))
        return anchors

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _crossover(self, ok1, ok2, ci1, ci2):
        """Uniform crossover producing one child."""
        N = self._N
        child_ok = np.zeros(N)
        child_ci = np.zeros(N, dtype=int)
        mask = self._rng.rand(N) < 0.5
        child_ok[mask] = ok1[mask]
        child_ok[~mask] = ok2[~mask]
        child_ci[mask] = ci1[mask]
        child_ci[~mask] = ci2[~mask]
        return child_ok, child_ci

    def _mutate(self, order_keys, candidate_idx):
        """Mutate in-place."""
        N = self._N
        # Mutate order_keys
        mask = self._rng.rand(N) < self.mutation_prob
        order_keys[mask] += self._rng.normal(0, 0.1, size=mask.sum())
        np.clip(order_keys, 0.0, 1.0, out=order_keys)

        # Mutate candidate_idx
        mask_ci = self._rng.rand(N) < self.mutation_prob * 0.5
        for j in np.where(mask_ci)[0]:
            n_cands = len(self._zone_candidates[self._zone_ids[j]])
            if n_cands > 1:
                candidate_idx[j] = self._rng.randint(0, n_cands)
        return order_keys

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self, fitness, k):
        """Select k indices via tournament."""
        selected = []
        for _ in range(k):
            contestants = self._rng.choice(self.pop_size, self.tournament_size,
                                            replace=False)
            winner = contestants[fitness[contestants].argmin()]
            selected.append(winner)
        return selected

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _fitness(self, order_keys, candidate_idx):
        """Compute fitness for a chromosome. Lower is better."""
        N = self._N
        perm = np.argsort(order_keys)
        zone_ids = self._zone_ids
        centroids = self._centroids_fig
        tan_k = np.tan(np.deg2rad(self.angle_deg))

        # Build ordered data
        ordered_zids = zone_ids[perm]
        geoms = []
        center_cost = 0.0
        cross_xs = []

        for pos in range(N):
            zid = ordered_zids[pos]

            # Map point
            ci = candidate_idx[perm[pos]]
            if ci >= len(self._zone_candidates[zid]):
                ci = 0
            map_fig = self._zone_candidates[zid][ci]

            # Matrix anchor
            matrix_fig = self._matrix_anchors_fig[pos]

            # Geometry
            g = gp_polyline_geometry(map_fig, matrix_fig, self.angle_deg)
            if g is None:
                return self.crossing_penalty

            geoms.append(g)
            cross_xs.append(g["q"][0])

            # Center deviation
            raw = centroids.get(zid, map_fig)
            center_cost += (map_fig[0] - raw[0]) ** 2 + (map_fig[1] - raw[1]) ** 2

        # Crossing check
        has_cross = False
        for i in range(N):
            for j in range(i + 1, N):
                if gp_geometries_cross(geoms[i], geoms[j]):
                    has_cross = True
                    break
            if has_cross:
                break

        if has_cross:
            return self.crossing_penalty

        # Spacing uniformity
        cross_xs = np.array(cross_xs)
        if N > 1:
            xs_sorted = np.sort(cross_xs)
            gaps = np.diff(xs_sorted)
            total_span = xs_sorted[-1] - xs_sorted[0]
            if total_span > 1e-10:
                gap_cv = np.std(gaps) / np.mean(gaps)
            else:
                gap_cv = 0.0
        else:
            gap_cv = 0.0

        return (
            self.center_weight * center_cost
            + self.spacing_weight * gap_cv
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode_chromosome(self, order_keys, candidate_idx, fitness, generations):
        """Convert best chromosome to result dict."""
        N = self._N
        perm = np.argsort(order_keys)
        zone_ids = self._zone_ids

        ordered_zids = list(zone_ids[perm])
        positions_fig = {}
        positions_data = {}
        geoms_list = []
        selections = {}

        for pos in range(N):
            zid = ordered_zids[pos]
            ci = int(candidate_idx[perm[pos]])
            if ci >= len(self._zone_candidates[zid]):
                ci = 0
            selections[zid] = ci
            map_fig = self._zone_candidates[zid][ci]
            matrix_fig = self._matrix_anchors_fig[pos]
            g = gp_polyline_geometry(map_fig, matrix_fig, self.angle_deg)

            positions_fig[zid] = map_fig
            geoms_list.append({
                'zid': zid,
                'geom': g,
                'map_fig': map_fig,
                'matrix_fig': matrix_fig,
            })

            # Also store data-coordinate position (stable across figure rebuilds)
            dx, dy = _fig_to_ax(
                self._ax_map, self._fig, map_fig[0], map_fig[1])
            positions_data[zid] = (float(dx), float(dy))

        return {
            "ordered_zone_ids": ordered_zids,
            "order": ordered_zids,
            "positions_fig": positions_fig,
            "positions_data": positions_data,
            "geoms": geoms_list,
            "selections": selections,
            "linewidths": self._linewidths,
            "fitness": float(fitness),
            "generations": generations,
            # metadata for rebuild after figure reset
            "_order_keys": order_keys.copy(),
            "_candidate_idx": candidate_idx.copy(),
            "_zone_candidates": self._zone_candidates,
        }
