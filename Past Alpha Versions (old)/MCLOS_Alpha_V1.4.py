import random
import json
import time
import math
import streamlit as st
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass

# =============================================================================
# MCLOS Alpha V1.4 - Modular Cargo Loading Optimization Software
# =============================================================================
# Changelog from V1.3:
#   - MAJOR: Complete rewrite of packing engine (pack_panels_v14)
#     * Layer-first filling: fills trailer floor before stacking layer 2
#     * Pre-computed tight-pack floor slots (no grid search)
#     * Column tracker prevents over-stacking
#     * Retroactive shear check on bottom panels (column_stress_ok)
#   - All 4 strategies now produce VISIBLY DIFFERENT arrangements:
#     * Gravity Layered: flat orientations, low CG, floor-first
#     * Wall First: upright panels side-by-side against trailer walls
#     * Zone Based: round-robin across 3 trailer thirds
#     * Stress Optimized: widest footprint first, minimizes bottom stress
#   - Strategy-specific orientation preference (not just slot ordering)
#   - Weight distribution: panels spread along trailer length
#   - Pre-computed max stack height per column (shear-safe)
#   - Center-of-gravity tracking + CG marker in 3D view
#   - Weight distribution by thirds shown in metrics
#   - 48 wall panels (6 pods) now fit without stress failures
#   - Loading/unloading order tracked per panel
#   - Seed UI: auto-randomize by default, manual override behind checkbox
# =============================================================================


# ─── Enums & Config ─────────────────────────────────────────────────────────

class LoadingStrategy(Enum):
    GRAVITY_LAYERED = "Gravity Layered"
    WALL_FIRST = "Wall First"
    ZONE_BASED = "Zone Based"
    STRESS_OPTIMIZED = "Stress Optimized"


class PanelType(Enum):
    WALL = "Wall Panel"
    FLOOR = "Floor Panel (Half-Hex)"


class TrailerPreset(Enum):
    FT53_ENCLOSED = "53-ft Enclosed"
    FT53_FLATBED = "53-ft Flatbed"
    FT42_ENCLOSED = "42-ft Enclosed"
    FT42_FLATBED = "42-ft Flatbed"
    CUSTOM = "Custom"


TRAILER_DIMS = {
    TrailerPreset.FT53_ENCLOSED: {"L": 636.0, "W": 102.0, "H": 110.0,
                                   "desc": "53-ft enclosed trailer (standard dry van)"},
    TrailerPreset.FT53_FLATBED:  {"L": 636.0, "W": 102.0, "H": 102.0,
                                   "desc": "53-ft flatbed (open, height = legal max for stacking)"},
    TrailerPreset.FT42_ENCLOSED: {"L": 504.0, "W": 102.0, "H": 110.0,
                                   "desc": "42-ft enclosed trailer (standard)"},
    TrailerPreset.FT42_FLATBED:  {"L": 504.0, "W": 102.0, "H": 102.0,
                                   "desc": "42-ft flatbed (current HexHomes trailer)"},
}


@dataclass
class StressConfig:
    max_compression_psi: float = 50.0
    max_bending_moment_lbf_in: float = 10000.0
    max_shear_psi: float = 30.0
    safety_factor: float = 2.0
    panel_youngs_modulus_psi: float = 1800000.0


@dataclass
class PanelSpec:
    panel_type: PanelType
    length: float
    height: float
    thickness: float
    weight: float
    short_edge: float = 0.0

    @property
    def is_trapezoid(self):
        return self.short_edge > 0

    @property
    def bounding_box(self):
        return (self.length, self.height, self.thickness)


WALL_PANEL_DEFAULT = PanelSpec(
    panel_type=PanelType.WALL, length=112.0, height=97.25, thickness=5.5,
    weight=220.0, short_edge=0.0
)
FLOOR_PANEL_DEFAULT = PanelSpec(
    panel_type=PanelType.FLOOR, length=224.0, height=111.87, thickness=6.5,
    weight=585.15, short_edge=112.0
)


# ─── Geometry / Collision ────────────────────────────────────────────────────

Z_TOL = 0.5

def aabb_intersect(p1, s1, p2, s2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return not (
        x1 + s1[0] <= x2 + 0.01 or x2 + s2[0] <= x1 + 0.01 or
        y1 + s1[1] <= y2 + 0.01 or y2 + s2[1] <= y1 + 0.01 or
        z1 + s1[2] <= z2 + 0.01 or z2 + s2[2] <= z1 + 0.01
    )


def in_bounds(pos, size, trailer):
    x, y, z = pos
    return (
        x >= -0.01 and y >= -0.01 and z >= -0.01 and
        x + size[0] <= trailer[0] + 0.01 and
        y + size[1] <= trailer[1] + 0.01 and
        z + size[2] <= trailer[2] + 0.01
    )


def handling_ok(size, max_horizontal, max_vertical):
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical


# ─── Stress Calculations (unchanged from V1.3) ──────────────────────────────

def calculate_support_area(pos, size, placed):
    x, y, z = pos
    dx, dy, dz = size
    if z <= Z_TOL:
        return dx * dy, []
    total_area = 0.0
    supporting_panels = []
    for p in placed:
        px, py, pz = p["pos"]
        pdx, pdy, pdz = p["size"]
        top = pz + pdz
        if abs(top - z) > Z_TOL:
            continue
        ox = max(0.0, min(x + dx, px + pdx) - max(x, px))
        oy = max(0.0, min(y + dy, py + pdy) - max(y, py))
        overlap_area = ox * oy
        if overlap_area > 0.01:
            total_area += overlap_area
            supporting_panels.append({
                'panel': p, 'overlap_area': overlap_area,
                'centroid': (px + pdx / 2, py + pdy / 2, top)
            })
    return total_area, supporting_panels


def calculate_compression_stress(weight, support_area):
    if support_area < 0.01:
        return float('inf')
    return weight / support_area


def calculate_bending_stress(panel_weight, pos, size, supporting_panels):
    if not supporting_panels:
        return 0.0
    x, y, z = pos
    dx, dy, dz = size
    cx, cy = x + dx / 2.0, y + dy / 2.0
    moi = (dy * dz ** 3) / 12.0
    max_moment = 0.0
    for s in supporting_panels:
        oh = max(abs(s['centroid'][0] - cx), abs(s['centroid'][1] - cy))
        max_moment = max(max_moment, panel_weight * oh)
    if moi < 1e-6:
        return float('inf')
    return (max_moment * (dz / 2.0)) / moi


def calculate_shear_stress(total_weight, size):
    dx, dy, dz = size
    sa = min(dx, dy) * dz
    if sa < 0.01:
        return float('inf')
    return total_weight / sa


def calculate_deflection(panel_weight, pos, size, youngs_modulus, supporting_panels):
    if not supporting_panels:
        return 0.0
    x, y, z = pos
    dx, dy, dz = size
    cx = x + dx / 2.0
    moi = (dy * dz ** 3) / 12.0
    if moi < 1e-6:
        return float('inf')
    max_oh = max((abs(s['centroid'][0] - cx) for s in supporting_panels), default=0)
    if max_oh < 1e-6:
        return 0.0
    return (panel_weight * max_oh ** 3) / (3.0 * youngs_modulus * moi)


def get_weight_above(panel_id, placed):
    total = 0.0
    p = placed[panel_id]
    px, py, pz = p["pos"]
    pdx, pdy, pdz = p["size"]
    top = pz + pdz
    for other in placed[panel_id + 1:]:
        ox, oy, oz = other["pos"]
        odx, ody, odz = other["size"]
        if oz < top - Z_TOL:
            continue
        xo = max(0.0, min(px + pdx, ox + odx) - max(px, ox))
        yo = max(0.0, min(py + pdy, oy + ody) - max(py, oy))
        if xo > 0.01 and yo > 0.01:
            total += other["weight"]
    return total


def stress_ok(pos, size, weight, placed, stress_config, min_support_frac=0.5):
    x, y, z = pos
    dx, dy, dz = size
    panel_area = dx * dy
    support_area, supporting_panels = calculate_support_area(pos, size, placed)
    if support_area < panel_area * min_support_frac:
        return False, "Insufficient support area"
    comp = calculate_compression_stress(weight, support_area)
    if comp * stress_config.safety_factor > stress_config.max_compression_psi:
        return False, f"Compression too high: {comp:.1f} psi"
    bend = calculate_bending_stress(weight, pos, size, supporting_panels)
    if bend * stress_config.safety_factor > stress_config.max_bending_moment_lbf_in:
        return False, f"Bending too high: {bend:.1f}"
    wa = sum(
        p["weight"] for p in placed
        if p["pos"][2] >= z + dz - Z_TOL
        and max(0.0, min(x + dx, p["pos"][0] + p["size"][0]) - max(x, p["pos"][0])) > 0.01
        and max(0.0, min(y + dy, p["pos"][1] + p["size"][1]) - max(y, p["pos"][1])) > 0.01
    )
    shear = calculate_shear_stress(weight + wa, size)
    if shear * stress_config.safety_factor > stress_config.max_shear_psi:
        return False, f"Shear too high: {shear:.1f} psi"
    defl = calculate_deflection(weight, pos, size, stress_config.panel_youngs_modulus_psi, supporting_panels)
    max_defl = min(dx, dy) / 360.0
    if defl > max_defl:
        return False, f"Deflection too large: {defl:.3f} in"
    return True, "OK"


# ─── V1.4 New Helper Functions ───────────────────────────────────────────────

def compute_max_stack_height(panel_weight, panel_size, trailer_H, stress_config):
    """Pre-compute how many panels can stack before bottom exceeds shear."""
    dz = panel_size[2]
    shear_area = min(panel_size[0], panel_size[1]) * dz
    if shear_area < 0.01:
        return 1, dz
    max_shear = stress_config.max_shear_psi / stress_config.safety_factor
    max_column_weight = max_shear * shear_area
    max_panels_shear = int(max_column_weight / panel_weight) if panel_weight > 0 else 999
    max_panels_height = int(trailer_H / dz) if dz > 0 else 1
    max_panels = max(1, min(max_panels_shear, max_panels_height))
    return max_panels, max_panels * dz


def column_stress_ok(col_key, new_weight, column_tracker, placed, stress_config):
    """
    Check that adding new_weight to a column won't overstress any panel below.
    Checks the bottom panel's cumulative shear.
    """
    state = column_tracker.get(col_key)
    if state is None:
        return True, "OK"  # empty column, floor supports it

    # Total weight in column after adding new panel
    new_total = state['total_weight'] + new_weight

    # Find the bottom panel in this column
    bottom_panel = None
    for p in placed:
        px, py = round(p["pos"][0], 2), round(p["pos"][1], 2)
        if (px, py) == col_key and (bottom_panel is None or p["pos"][2] < bottom_panel["pos"][2]):
            bottom_panel = p

    if bottom_panel is None:
        return True, "OK"

    # Shear check on the bottom panel with total column weight
    shear = calculate_shear_stress(new_total, bottom_panel["size"])
    if shear * stress_config.safety_factor > stress_config.max_shear_psi:
        return False, f"Column shear on bottom panel: {shear:.1f} psi"

    return True, "OK"


def compute_cg(placed):
    """Compute center of gravity of all placed panels."""
    if not placed:
        return {"x": 0, "y": 0, "z": 0}
    tw = sum(p["weight"] for p in placed)
    if tw < 0.01:
        return {"x": 0, "y": 0, "z": 0}
    cx = sum(p["weight"] * (p["pos"][0] + p["size"][0] / 2) for p in placed) / tw
    cy = sum(p["weight"] * (p["pos"][1] + p["size"][1] / 2) for p in placed) / tw
    cz = sum(p["weight"] * (p["pos"][2] + p["size"][2] / 2) for p in placed) / tw
    return {"x": round(cx, 2), "y": round(cy, 2), "z": round(cz, 2)}


def compute_weight_distribution(placed, trailer_L):
    """Weight distribution by trailer thirds."""
    thirds = {"front": 0.0, "middle": 0.0, "rear": 0.0}
    zone_w = trailer_L / 3.0
    for p in placed:
        cx = p["pos"][0] + p["size"][0] / 2
        if cx < zone_w:
            thirds["front"] += p["weight"]
        elif cx < zone_w * 2:
            thirds["middle"] += p["weight"]
        else:
            thirds["rear"] += p["weight"]
    return {k: round(v, 1) for k, v in thirds.items()}


# ─── Orientation Generation (unchanged from V1.3) ───────────────────────────

def generate_orientations_wall(L, H, T):
    return [
        ("flat_LxH",  (L, H, T), ("L", "H", "T")),
        ("flat_HxL",  (H, L, T), ("H", "L", "T")),
        ("stand_LxT", (L, T, H), ("L", "T", "H")),
        ("stand_TxL", (T, L, H), ("T", "L", "H")),
        ("stand_HxT", (H, T, L), ("H", "T", "L")),
        ("stand_TxH", (T, H, L), ("T", "H", "L")),
    ]


def generate_orientations_floor(L, H, T):
    return [
        ("flat_LxH",  (L, H, T), ("L", "H", "T")),
        ("flat_HxL",  (H, L, T), ("H", "L", "T")),
        ("stand_LxT", (L, T, H), ("L", "T", "H")),
        ("stand_HxT", (H, T, L), ("H", "T", "L")),
    ]


# ─── V1.4 Strategy-Specific Slot Ordering ───────────────────────────────────

def generate_floor_slots(dx, dy, trailer_L, trailer_W):
    """Generate all tight-packed non-overlapping positions on trailer floor."""
    slots = []
    x = 0.0
    while x + dx <= trailer_L + 0.01:
        y = 0.0
        while y + dy <= trailer_W + 0.01:
            slots.append((round(x, 4), round(y, 4)))
            y = round(y + dy, 4)
        x = round(x + dx, 4)
    return slots


def sort_orientations_for_strategy(strategy, orientations):
    """
    Return orientations reordered per strategy.
    Each strategy has a different PREFERENCE for how panels should be oriented.
    """
    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        # Prefer flat orientations (thickness as Z) for lowest CG
        # Then by highest floor capacity for best packing
        return sorted(orientations, key=lambda o: (
            0 if o['axis_map'][2] == 'T' else 1,   # flat first (T on Z-axis)
            -o['total_capacity'],
        ))

    elif strategy == LoadingStrategy.WALL_FIRST:
        # Prefer STANDING orientations (Z-axis is NOT thickness)
        # "Standing" means the panel's thin dimension is on X or Y, tall on Z
        # axis_map[2] == 'T' means flat; anything else means standing upright
        return sorted(orientations, key=lambda o: (
            1 if o['axis_map'][2] == 'T' else 0,  # standing first (Z != thickness)
            -o['size'][2],                          # tallest Z first
            -len(o['floor_slots']),                 # most slots for side-by-side
        ))

    elif strategy == LoadingStrategy.ZONE_BASED:
        # Same as gravity layered for orientation; differentiation is in slot order
        return sorted(orientations, key=lambda o: (
            0 if o['axis_map'][2] == 'T' else 1,
            -o['total_capacity'],
        ))

    elif strategy == LoadingStrategy.STRESS_OPTIMIZED:
        # Prefer widest footprint (largest floor area = lowest stress on bottom)
        # Flat orientations spread load better
        return sorted(orientations, key=lambda o: (
            -(o['size'][0] * o['size'][1]),  # largest footprint first
            0 if o['axis_map'][2] == 'T' else 1,
        ))

    return orientations


def apply_strategy_slot_order(strategy, floor_slots, placed, trailer, panel_size):
    """Return floor_slots reordered per strategy. Each strategy is meaningfully different."""
    dx, dy, dz = panel_size
    trailer_L, trailer_W, trailer_H = trailer

    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        # Left-to-right, front-to-back: standard uniform fill
        return sorted(floor_slots, key=lambda s: (s[0], s[1]))

    elif strategy == LoadingStrategy.WALL_FIRST:
        # Prioritize positions against trailer side walls (Y=0 and Y=max-dy)
        # For upright panels side-by-side along the walls
        def wall_priority(slot):
            x, y = slot
            dist_to_side = min(y, max(0, trailer_W - (y + dy)))
            return (round(dist_to_side, 2), x, y)
        return sorted(floor_slots, key=wall_priority)

    elif strategy == LoadingStrategy.ZONE_BASED:
        # Round-robin across 3 longitudinal zones
        zone_width = trailer_L / 3.0
        zones = [[], [], []]
        for slot in floor_slots:
            zi = min(int(slot[0] / zone_width), 2)
            zones[zi].append(slot)
        for z in zones:
            z.sort(key=lambda s: (s[0], s[1]))
        # Interleave: one from each zone in turn
        result = []
        max_len = max((len(z) for z in zones), default=0)
        for i in range(max_len):
            for z in zones:
                if i < len(z):
                    result.append(z[i])
        return result

    elif strategy == LoadingStrategy.STRESS_OPTIMIZED:
        # Center-out placement with CG balancing
        center_x = trailer_L / 2.0
        if placed:
            tw = sum(p["weight"] for p in placed)
            cg_x = sum(p["weight"] * (p["pos"][0] + p["size"][0] / 2) for p in placed) / tw if tw > 0 else center_x
        else:
            cg_x = center_x

        def cg_score(slot):
            slot_cx = slot[0] + dx / 2
            dist = abs(slot_cx - center_x)
            # Bias toward the side opposite the current CG
            imbalance = (slot_cx - center_x) * (cg_x - center_x)
            return dist + imbalance * 0.3
        return sorted(floor_slots, key=cg_score)

    # Fallback
    return floor_slots


# ─── Panel List Builders ─────────────────────────────────────────────────────

def build_panel_list_from_pods(num_pods, wall_spec, floor_spec):
    panels = []
    for pod in range(num_pods):
        for half in range(2):
            panels.append({
                "spec": floor_spec,
                "label": f"Pod{pod + 1}_Floor_{half + 1}",
                "panel_type": PanelType.FLOOR.value,
            })
    for pod in range(num_pods):
        for wall in range(6):
            panels.append({
                "spec": wall_spec,
                "label": f"Pod{pod + 1}_Wall_{wall + 1}",
                "panel_type": PanelType.WALL.value,
            })
    return panels


def build_panel_list_manual(num_walls, num_floors, wall_spec, floor_spec):
    panels = []
    for i in range(num_floors):
        panels.append({
            "spec": floor_spec,
            "label": f"Floor_{i + 1}",
            "panel_type": PanelType.FLOOR.value,
        })
    for i in range(num_walls):
        panels.append({
            "spec": wall_spec,
            "label": f"Wall_{i + 1}",
            "panel_type": PanelType.WALL.value,
        })
    return panels


# ─── V1.4 Core Packing Engine ───────────────────────────────────────────────

def pack_panels_v14(panel_list, trailer_L, trailer_W, trailer_H,
                    max_horizontal, max_vertical,
                    step, seed, strategy, stress_config):
    """
    V1.4 packing engine: layer-first, stress-aware, strategy-differentiated.

    Key improvements over V1.3:
    1. Pre-computes tight-packed floor slots (no grid search)
    2. Fills ALL floor slots before stacking (layer-first)
    3. Column tracker enforces max stack height
    4. Retroactive shear check on bottom panels
    5. Each strategy produces genuinely different slot ordering
    6. Tracks CG and weight distribution
    """
    random.seed(seed)
    trailer = (trailer_L, trailer_W, trailer_H)

    placed = []
    rejected = []

    # ── Step 1: Pre-compute orientation info per panel type ──
    orientation_cache = {}

    for panel_info in panel_list:
        spec = panel_info["spec"]
        cache_key = (spec.panel_type.value, spec.length, spec.height, spec.thickness, spec.weight)
        if cache_key in orientation_cache:
            continue

        bbox = spec.bounding_box
        if spec.is_trapezoid:
            raw = generate_orientations_floor(bbox[0], bbox[1], bbox[2])
        else:
            raw = generate_orientations_wall(bbox[0], bbox[1], bbox[2])

        valid = []
        for name, size, axis_map in raw:
            if not in_bounds((0, 0, 0), size, trailer):
                continue
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            dx, dy, dz = size
            floor_slots = generate_floor_slots(dx, dy, trailer_L, trailer_W)
            max_stack, max_z = compute_max_stack_height(spec.weight, size, trailer_H, stress_config)

            valid.append({
                'name': name,
                'size': size,
                'axis_map': axis_map,
                'floor_slots': floor_slots,
                'max_stack': max_stack,
                'total_capacity': len(floor_slots) * max_stack,
            })

        # Store unsorted — strategy-specific sorting happens per panel
        orientation_cache[cache_key] = valid

    # ── Step 2: Sort panel list per strategy ──
    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        sorted_panels = sorted(panel_list, key=lambda p: (-p['spec'].weight, p['label']))
    elif strategy == LoadingStrategy.WALL_FIRST:
        sorted_panels = sorted(panel_list, key=lambda p: (
            0 if p['spec'].panel_type == PanelType.WALL else 1, p['label']))
    elif strategy == LoadingStrategy.ZONE_BASED:
        # Interleave by pod
        sorted_panels = sorted(panel_list, key=lambda p: (
            p['label'].split('_')[-1], p['label'].split('_')[0]))
    else:  # STRESS_OPTIMIZED
        sorted_panels = sorted(panel_list, key=lambda p: (
            -p['spec'].weight, -p['spec'].length * p['spec'].height))

    # ── Step 3: Main placement loop ──
    column_tracker = {}  # (x, y) -> {count, z_top, total_weight}

    for pid, panel_info in enumerate(sorted_panels):
        spec = panel_info["spec"]
        cache_key = (spec.panel_type.value, spec.length, spec.height, spec.thickness, spec.weight)
        raw_orientations = orientation_cache.get(cache_key, [])
        orientations = sort_orientations_for_strategy(strategy, raw_orientations)

        if not orientations:
            rejected.append({
                "id": pid, "label": panel_info["label"],
                "panel_type": panel_info["panel_type"],
                "reason": "No valid orientation fits in trailer"
            })
            continue

        placed_this = False

        for orient in orientations:
            name = orient['name']
            size = orient['size']
            axis_map = orient['axis_map']
            dx, dy, dz = size
            floor_slots = orient['floor_slots']
            max_stack = orient['max_stack']

            if not floor_slots:
                continue

            # Get strategy-ordered slots (recalculated each panel for CG-aware strategies)
            ordered_slots = apply_strategy_slot_order(
                strategy, floor_slots, placed, trailer, size
            )

            # ── Layer-first sorting ──
            # Sort candidates by: (current_layer ASC, strategy_order)
            # This ensures ALL floor positions fill before any stacking occurs
            def slot_sort_key(idx_slot):
                idx, (sx, sy) = idx_slot
                col_key = (round(sx, 2), round(sy, 2))
                state = column_tracker.get(col_key)
                layer = state['count'] if state else 0
                return (layer, idx)

            indexed_slots = list(enumerate(ordered_slots))
            indexed_slots.sort(key=slot_sort_key)

            for _, (sx, sy) in indexed_slots:
                col_key = (round(sx, 2), round(sy, 2))
                state = column_tracker.get(col_key)

                if state is None:
                    z = 0.0
                else:
                    if state['count'] >= max_stack:
                        continue  # column full
                    z = state['z_top']

                test_pos = (sx, sy, z)

                if not in_bounds(test_pos, size, trailer):
                    continue

                # Collision check (safety net for mixed orientations)
                if any(aabb_intersect(test_pos, size, p["pos"], p["size"]) for p in placed):
                    continue

                # Standard stress check
                stress_valid, msg = stress_ok(test_pos, size, spec.weight, placed, stress_config)
                if not stress_valid:
                    continue

                # Retroactive column shear check
                if z > Z_TOL:
                    col_ok, col_msg = column_stress_ok(
                        col_key, spec.weight, column_tracker, placed, stress_config
                    )
                    if not col_ok:
                        continue

                # ── Place the panel ──
                placed.append({
                    "id": pid,
                    "label": panel_info["label"],
                    "panel_type": panel_info["panel_type"],
                    "pos": test_pos,
                    "size": size,
                    "orientation": name,
                    "axis_map": axis_map,
                    "weight": spec.weight,
                    "is_trapezoid": spec.is_trapezoid,
                    "short_edge": spec.short_edge,
                    "spec_length": spec.length,
                    "spec_height": spec.height,
                    "spec_thickness": spec.thickness,
                })

                if state is None:
                    column_tracker[col_key] = {
                        'count': 1, 'z_top': round(z + dz, 4),
                        'total_weight': spec.weight,
                    }
                else:
                    state['count'] += 1
                    state['z_top'] = round(z + dz, 4)
                    state['total_weight'] += spec.weight

                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            rejected.append({
                "id": pid, "label": panel_info["label"],
                "panel_type": panel_info["panel_type"],
                "reason": "No valid position found (stress/collision/bounds/capacity)"
            })

    # ── Step 4: Post-placement stress analysis ──
    for i, panel in enumerate(placed):
        sa, sp = calculate_support_area(panel["pos"], panel["size"], placed[:i])
        comp = calculate_compression_stress(panel["weight"], sa)
        bend = calculate_bending_stress(panel["weight"], panel["pos"], panel["size"], sp)
        wa = get_weight_above(i, placed)
        shear = calculate_shear_stress(wa + panel["weight"], panel["size"])
        defl = calculate_deflection(panel["weight"], panel["pos"], panel["size"],
                                    stress_config.panel_youngs_modulus_psi, sp)
        panel["stress_analysis"] = {
            "compression_psi": round(comp, 2),
            "bending_stress": round(bend, 2),
            "shear_psi": round(shear, 2),
            "deflection_in": round(defl, 4),
            "weight_above_lb": round(wa, 2),
            "support_area_sqin": round(sa, 2)
        }

    # ── Step 5: Summary metrics ──
    wall_count = sum(1 for p in placed if p["panel_type"] == PanelType.WALL.value)
    floor_count = sum(1 for p in placed if p["panel_type"] == PanelType.FLOOR.value)
    cg = compute_cg(placed)
    weight_dist = compute_weight_distribution(placed, trailer_L)

    max_layer = 0
    for state in column_tracker.values():
        max_layer = max(max_layer, state['count'])

    return {
        "inputs": {
            "trailer": {"L": trailer_L, "W": trailer_W, "H": trailer_H},
            "stress_limits": {
                "max_compression_psi": stress_config.max_compression_psi,
                "max_bending_lbf_in": stress_config.max_bending_moment_lbf_in,
                "max_shear_psi": stress_config.max_shear_psi,
                "safety_factor": stress_config.safety_factor
            }
        },
        "settings": {
            "requested_panels": len(panel_list),
            "placed_panels": len(placed),
            "placed_walls": wall_count,
            "placed_floors": floor_count,
            "rejected_panels": len(rejected),
            "grid_step_in": float(step),
            "seed": int(seed),
            "min_support_frac": 0.5,
            "packing_strategy": strategy.value,
            "max_layers_used": max_layer,
        },
        "weight_distribution": {
            "center_of_gravity": cg,
            "by_thirds": weight_dist,
        },
        "placements": [
            {
                "id": p["id"],
                "label": p["label"],
                "panel_type": p["panel_type"],
                "position_vector": [round(p["pos"][0], 3), round(p["pos"][1], 3), round(p["pos"][2], 3)],
                "size": [round(p["size"][0], 3), round(p["size"][1], 3), round(p["size"][2], 3)],
                "orientation": p["orientation"],
                "axis_map": list(p["axis_map"]),
                "weight": p["weight"],
                "is_trapezoid": p["is_trapezoid"],
                "layer": int(round(p["pos"][2] / max(p["size"][2], 0.1))),
                "loading_order": i,
                "unloading_order": len(placed) - i - 1,
                "stress_analysis": p["stress_analysis"]
            }
            for i, p in enumerate(placed)
        ],
        "rejections": rejected
    }


# ─── 3D Visualization ───────────────────────────────────────────────────────

def box_edges(x, y, z, dx, dy, dz):
    p = [
        (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
        (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)
    ]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in e:
        xs += [p[a][0], p[b][0], None]
        ys += [p[a][1], p[b][1], None]
        zs += [p[a][2], p[b][2], None]
    return xs, ys, zs


def make_half_hex_vertices(x, y, z, dx, dy, dz, axis_map, short_edge, long_edge):
    offset = (long_edge - short_edge) / 2.0
    x_axis, y_axis, z_axis = axis_map
    if x_axis == "L" and y_axis == "H":
        b0, b1 = (x, y, z), (x + dx, y, z)
        b2, b3 = (x + offset + short_edge, y + dy, z), (x + offset, y + dy, z)
    elif x_axis == "H" and y_axis == "L":
        b0, b1 = (x, y, z), (x, y + dy, z)
        b2, b3 = (x + dx, y + offset + short_edge, z), (x + dx, y + offset, z)
    elif x_axis == "L" and z_axis == "H":
        b0, b1 = (x, y, z), (x + dx, y, z)
        b2, b3 = (x + offset + short_edge, y, z + dz), (x + offset, y, z + dz)
        t0 = (b0[0], y + dy, b0[2])
        t1 = (b1[0], y + dy, b1[2])
        t2 = (b2[0], y + dy, b2[2])
        t3 = (b3[0], y + dy, b3[2])
        return [b0, b1, b2, b3, t0, t1, t2, t3]
    elif x_axis == "H" and z_axis == "L":
        b0, b1 = (x, y, z), (x, y, z + dz)
        b2, b3 = (x + dx, y, z + offset + short_edge), (x + dx, y, z + offset)
        t0 = (b0[0], y + dy, b0[2])
        t1 = (b1[0], y + dy, b1[2])
        t2 = (b2[0], y + dy, b2[2])
        t3 = (b3[0], y + dy, b3[2])
        return [b0, b1, b2, b3, t0, t1, t2, t3]
    else:
        b0, b1 = (x, y, z), (x + dx, y, z)
        b2, b3 = (x + dx, y + dy, z), (x, y + dy, z)
    t0 = (b0[0], b0[1], z + dz)
    t1 = (b1[0], b1[1], z + dz)
    t2 = (b2[0], b2[1], z + dz)
    t3 = (b3[0], b3[1], z + dz)
    return [b0, b1, b2, b3, t0, t1, t2, t3]


def add_wire(fig, x, y, z, dx, dy, dz, color, name, opacity=1.0, width=4):
    xs, ys, zs = box_edges(x, y, z, dx, dy, dz)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                line=dict(color=color, width=width),
                                opacity=opacity, name=name, showlegend=(name != "")))


def add_solid(fig, x, y, z, dx, dy, dz, color, opacity=0.35):
    vx = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    vy = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    vz = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz,
                             i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                             color=color, opacity=opacity, showlegend=False))


def add_half_hex_solid(fig, verts, color, opacity=0.45):
    vx = [v[0] for v in verts]
    vy = [v[1] for v in verts]
    vz = [v[2] for v in verts]
    faces = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(3,2,6),(3,6,7),(0,3,7),(0,7,4),(1,2,6),(1,6,5)]
    fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz,
                             i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
                             color=color, opacity=opacity, showlegend=False))


def add_half_hex_wire(fig, verts, color, name="", opacity=1.0, width=4):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [verts[a][0], verts[b][0], None]
        ys += [verts[a][1], verts[b][1], None]
        zs += [verts[a][2], verts[b][2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                line=dict(color=color, width=width),
                                opacity=opacity, name=name, showlegend=(name != "")))


def get_stress_color(val, maxv):
    if maxv < 1e-6:
        return "#2ecc71"
    r = min(val / maxv, 1.0)
    return "#2ecc71" if r < 0.5 else "#f1c40f" if r < 0.75 else "#e74c3c"


def visualize(out, stress_config, color_by_stress, floor_spec):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]
    fig = go.Figure()

    # Trailer
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.2)
    add_wire(fig, 0, 0, 0, L, W, H, "#ecf0f1", "", opacity=0.3, width=2)
    for i in range(0, int(L), 50):
        add_wire(fig, i, 0, 0, 0, W, 0, "#34495e", "", opacity=0.15, width=1)
    for j in range(0, int(W), 50):
        add_wire(fig, 0, j, 0, L, 0, 0, "#34495e", "", opacity=0.15, width=1)

    wall_pal = ["#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#95a5a6"]
    floor_pal = ["#e67e22", "#e74c3c", "#f1c40f", "#d35400"]
    wi, fi = 0, 0
    tc = {}

    for p in out["placements"]:
        pt = p["panel_type"]
        o = p["orientation"]
        key = f"{pt}|{o}"
        if key not in tc:
            if pt == PanelType.FLOOR.value:
                tc[key] = floor_pal[fi % len(floor_pal)]
                fi += 1
            else:
                tc[key] = wall_pal[wi % len(wall_pal)]
                wi += 1

        c = get_stress_color(p["stress_analysis"]["compression_psi"], stress_config.max_compression_psi) if color_by_stress else tc[key]
        x, y, z = p["position_vector"]
        dx, dy, dz = p["size"]

        if p.get("is_trapezoid", False):
            am = tuple(p.get("axis_map", ["L", "H", "T"]))
            verts = make_half_hex_vertices(x, y, z, dx, dy, dz, am, floor_spec.short_edge, floor_spec.length)
            add_half_hex_solid(fig, verts, c, opacity=0.45)
            add_half_hex_wire(fig, verts, c, "", opacity=0.9, width=2)
        else:
            add_solid(fig, x, y, z, dx, dy, dz, c, opacity=0.4)
            add_wire(fig, x, y, z, dx, dy, dz, c, "", opacity=0.9, width=2)

    # CG marker
    if "weight_distribution" in out and out["placements"]:
        cg = out["weight_distribution"]["center_of_gravity"]
        fig.add_trace(go.Scatter3d(
            x=[cg["x"]], y=[cg["y"]], z=[cg["z"]],
            mode="markers+text",
            marker=dict(size=8, color="red", symbol="diamond"),
            text=["CG"], textposition="top center",
            textfont=dict(color="red", size=12),
            name="Center of Gravity"
        ))

    # Legend
    if not color_by_stress:
        added = set()
        for key, c in tc.items():
            count = sum(1 for p in out["placements"] if f"{p['panel_type']}|{p['orientation']}" == key)
            pt, o = key.split("|", 1)
            short = "Floor" if "Floor" in pt else "Wall"
            lbl = f"{short} - {o} ({count})"
            if lbl not in added:
                fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                            line=dict(color=c, width=6), name=lbl))
                added.add(lbl)

    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="Length (in)", yaxis_title="Width (in)", zaxis_title="Height (in)",
                   aspectmode="data", camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=11))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    st.subheader("Structural Analysis Summary")
    if out["placements"]:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("Panels", f"{out['settings']['placed_panels']} / {out['settings']['requested_panels']}")
        with c2:
            st.metric("W + F", f"{out['settings']['placed_walls']}W + {out['settings']['placed_floors']}F")
        with c3:
            tw = sum(p["weight"] for p in out["placements"])
            st.metric("Weight", f"{tw:,.0f} lb")
        with c4:
            mc = max((p["stress_analysis"]["compression_psi"] for p in out["placements"]), default=0)
            st.metric("Max Comp.", f"{mc:.1f} psi")
        with c5:
            st.metric("Layers", out['settings']['max_layers_used'])
        with c6:
            cg = out["weight_distribution"]["center_of_gravity"]
            st.metric("CG (X)", f"{cg['x']:.0f}\" / {tr['L']:.0f}\"")

        # Weight distribution bar
        wd = out["weight_distribution"]["by_thirds"]
        st.subheader("Weight Distribution (Trailer Thirds)")
        wd_c1, wd_c2, wd_c3 = st.columns(3)
        with wd_c1:
            st.metric("Front", f"{wd['front']:,.0f} lb")
        with wd_c2:
            st.metric("Middle", f"{wd['middle']:,.0f} lb")
        with wd_c3:
            st.metric("Rear", f"{wd['rear']:,.0f} lb")
    else:
        st.warning("No panels placed. Check rejections below.")


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="MCLOS - Modular Cargo Loading Optimization", layout="wide")
st.title("MCLOS - Modular Cargo Loading Optimization Software")
st.caption("HexHomes Panel Packing Optimizer  |  Alpha V1.4")

if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 99999)
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Loading Strategy")
    strategy_name = st.selectbox("Strategy", [s.value for s in LoadingStrategy], index=0,
                                  help="Gravity Layered: flat panels, low CG. Wall First: upright side-by-side. "
                                       "Zone Based: balanced across thirds. Stress Optimized: minimizes bottom stress.")
    strategy = [s for s in LoadingStrategy if s.value == strategy_name][0]

    st.markdown("---")
    st.header("Display")
    color_by_stress = st.checkbox("Color by Stress Level", value=False)

    st.markdown("---")
    with st.expander("Advanced Settings"):
        st.subheader("Structural Limits")
        adv_comp = st.number_input("Max Compression (psi)", value=50.0, min_value=1.0, key="ac")
        adv_bend = st.number_input("Max Bending (lbf-in)", value=10000.0, min_value=1.0, key="ab")
        adv_shear = st.number_input("Max Shear (psi)", value=30.0, min_value=1.0, key="as")
        adv_safety = st.number_input("Safety Factor", value=2.0, min_value=1.0, key="asf")
        adv_youngs = st.number_input("Young's Modulus (psi)", value=1800000.0, min_value=1000.0, key="ay")
        st.markdown("---")
        st.subheader("Handling")
        maxH = st.number_input("Max Horizontal (in)", value=230.0, min_value=1.0, key="amh")
        maxV = st.number_input("Max Vertical (in)", value=114.0, min_value=1.0, key="amv")
        st.markdown("---")
        st.subheader("Seed")
        st.caption(f"Current: `{st.session_state.seed}` (auto-randomized each run)")
        use_manual_seed = st.checkbox("Enable manual seed override", value=False, key="use_manual")
        if use_manual_seed:
            manual_seed = st.number_input("Manual Seed", min_value=0, value=st.session_state.seed, step=1, key="ams")
            st.session_state.seed = manual_seed
        grid_step = st.number_input("Grid Step (in)", min_value=0.5, value=5.5, step=0.5, key="ags",
                                    help="Used for fallback positioning only in V1.4")

stress_config = StressConfig(
    max_compression_psi=adv_comp, max_bending_moment_lbf_in=adv_bend,
    max_shear_psi=adv_shear, safety_factor=adv_safety, panel_youngs_modulus_psi=adv_youngs
)

# ─── Trailer ────────────────────────────────────────────────────────────────
st.subheader("Trailer Selection")
tc1, tc2 = st.columns([1, 2])
with tc1:
    tp_name = st.selectbox("Trailer Type", [t.value for t in TrailerPreset], index=0)
    tp = [t for t in TrailerPreset if t.value == tp_name][0]
with tc2:
    if tp != TrailerPreset.CUSTOM:
        d = TRAILER_DIMS[tp]
        st.info(f"**{tp.value}**: {d['L']:.0f}\" x {d['W']:.0f}\" x {d['H']:.0f}\"  \n{d['desc']}")
        tL, tW, tH = d["L"], d["W"], d["H"]
    else:
        st.caption("Enter custom dimensions.")
if tp == TrailerPreset.CUSTOM:
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        tL = st.number_input("Length (in)", value=636.0, min_value=1.0, key="tl")
    with cc2:
        tW = st.number_input("Width (in)", value=102.0, min_value=1.0, key="tw")
    with cc3:
        tH = st.number_input("Height (in)", value=110.0, min_value=1.0, key="th")

# ─── Panels ─────────────────────────────────────────────────────────────────
st.subheader("Panel Configuration")
input_mode = st.radio("Input method", ["By Pod Count (quick)", "Manual Panel Count"], horizontal=True)

if input_mode == "By Pod Count (quick)":
    cp1, cp2 = st.columns([1, 2])
    with cp1:
        num_pods = st.number_input("Pods", min_value=1, max_value=6, value=3, step=1,
                                   help="6 walls + 2 floors per pod")
    with cp2:
        tw = num_pods * 6
        tf = num_pods * 2
        st.markdown(f"**{num_pods} Pod{'s' if num_pods > 1 else ''}** = **{tw + tf} panels** ({tw}W + {tf}F)")
    use_pods = True
else:
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        manual_walls = st.number_input("Walls", min_value=0, value=18, step=1, key="mw")
    with mc2:
        manual_floors = st.number_input("Floors", min_value=0, value=6, step=1, key="mf")
    with mc3:
        st.metric("Total", manual_walls + manual_floors)
    use_pods = False

with st.expander("Panel Dimensions (edit if needed)"):
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Wall Panel**")
        wL = st.number_input("Length (in)", value=WALL_PANEL_DEFAULT.length, min_value=1.0, key="wl")
        wH = st.number_input("Height (in)", value=WALL_PANEL_DEFAULT.height, min_value=1.0, key="wh")
        wT = st.number_input("Thickness (in)", value=WALL_PANEL_DEFAULT.thickness, min_value=0.1, key="wt")
        wW = st.number_input("Weight (lb)", value=WALL_PANEL_DEFAULT.weight, min_value=0.1, key="ww")
    with pc2:
        st.markdown("**Floor Panel (Half-Hex)**")
        fL = st.number_input("Long Edge (in)", value=FLOOR_PANEL_DEFAULT.length, min_value=1.0, key="fl")
        fH = st.number_input("Depth (in)", value=FLOOR_PANEL_DEFAULT.height, min_value=1.0, key="fh")
        fT = st.number_input("Thickness (in)", value=FLOOR_PANEL_DEFAULT.thickness, min_value=0.1, key="ft")
        fW = st.number_input("Weight (lb)", value=FLOOR_PANEL_DEFAULT.weight, min_value=0.1, key="fw")
        fS = st.number_input("Short Edge (in)", value=FLOOR_PANEL_DEFAULT.short_edge, min_value=1.0, key="fs")

wall_spec = PanelSpec(panel_type=PanelType.WALL, length=wL, height=wH, thickness=wT, weight=wW)
floor_spec = PanelSpec(panel_type=PanelType.FLOOR, length=fL, height=fH, thickness=fT, weight=fW, short_edge=fS)

# ─── Run ────────────────────────────────────────────────────────────────────
st.markdown("---")
rc1, rc2 = st.columns([3, 1])
with rc1:
    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)
with rc2:
    st.markdown(f"**Seed:** `{st.session_state.seed}`")
    st.caption(f"Run #{st.session_state.run_count}")

if run_btn:
    if use_pods:
        panel_list = build_panel_list_from_pods(num_pods, wall_spec, floor_spec)
    else:
        panel_list = build_panel_list_manual(manual_walls, manual_floors, wall_spec, floor_spec)

    if not panel_list:
        st.error("No panels to optimize.")
    else:
        run_seed = st.session_state.seed
        t0 = time.time()
        with st.spinner(f"Optimizing {len(panel_list)} panels ({strategy.value})..."):
            out = pack_panels_v14(panel_list, tL, tW, tH, maxH, maxV,
                                  grid_step, run_seed, strategy, stress_config)
        elapsed = time.time() - t0

        # Only auto-randomize if manual override is not enabled
        if not st.session_state.get("use_manual", False):
            st.session_state.seed = random.randint(0, 99999)
        st.session_state.run_count += 1

        if "error" not in out:
            pc = out["settings"]["placed_panels"]
            rc = out["settings"]["requested_panels"]
            sr = (pc / rc * 100) if rc > 0 else 0

            s1, s2, s3 = st.columns(3)
            with s1:
                (st.success if sr == 100 else st.warning if pc > 0 else st.error)(
                    f"{'All ' if sr == 100 else ''}{pc} / {rc} panels loaded ({sr:.0f}%)")
            with s2:
                st.info(f"Time: **{elapsed:.2f}s**")
            with s3:
                st.info(f"Seed: **{run_seed}**")

            st.subheader("3D Trailer View")
            visualize(out, stress_config, color_by_stress, floor_spec)

            if out["rejections"]:
                with st.expander(f"Rejected Panels ({len(out['rejections'])})", expanded=True):
                    for r in out["rejections"]:
                        st.error(f"**{r['label']}** ({r['panel_type']}): {r['reason']}")

            if out["placements"]:
                with st.expander("Detailed Stress Analysis"):
                    sd = [{
                        "Panel": p["label"], "Type": p["panel_type"], "Orient": p["orientation"],
                        "Layer": p["layer"], "Comp (psi)": p["stress_analysis"]["compression_psi"],
                        "Bend": p["stress_analysis"]["bending_stress"],
                        "Shear (psi)": p["stress_analysis"]["shear_psi"],
                        "Defl (in)": p["stress_analysis"]["deflection_in"],
                        "Wt Above (lb)": p["stress_analysis"]["weight_above_lb"],
                    } for p in out["placements"]]
                    st.dataframe(sd, use_container_width=True)

            with st.expander("View JSON"):
                st.json(out)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download JSON", json.dumps(out, indent=2),
                                   f"mclos_v14_seed{run_seed}.json", "application/json", use_container_width=True)
            with d2:
                hdr = "id,label,type,x,y,z,dx,dy,dz,orient,weight,layer,load_order,comp,bend,shear,defl\n"
                rows = "\n".join([
                    f"{p['id']},{p['label']},{p['panel_type']},"
                    f"{p['position_vector'][0]},{p['position_vector'][1]},{p['position_vector'][2]},"
                    f"{p['size'][0]},{p['size'][1]},{p['size'][2]},{p['orientation']},{p['weight']},"
                    f"{p['layer']},{p['loading_order']},"
                    f"{p['stress_analysis']['compression_psi']},{p['stress_analysis']['bending_stress']},"
                    f"{p['stress_analysis']['shear_psi']},{p['stress_analysis']['deflection_in']}"
                    for p in out.get("placements", [])
                ])
                st.download_button("Download CSV", hdr + rows,
                                   f"mclos_v14_seed{run_seed}.csv", "text/csv", use_container_width=True)
        else:
            st.error(out["error"])
