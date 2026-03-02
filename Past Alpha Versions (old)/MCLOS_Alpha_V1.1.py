import random
import json
import time
import math
import streamlit as st
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =============================================================================
# MCLOS Alpha V1.1 - Modular Cargo Loading Optimization Software
# =============================================================================
# Changelog from Alpha:
#   - Added floor panel type (half-hexagon trapezoid) with proper geometry
#   - Pod-based input: select # of pods, auto-calculates wall + floor panels
#   - Trapezoid 3D rendering for floor panels (visual shape differentiation)
#   - UI overhaul: clean main interface, Advanced Settings in expander
#   - Auto-randomize seed after each optimization run
#   - Run timer showing optimization duration
#   - BUG FIX: Safety factor now correctly applied (multiply stress, not divide limit)
#   - BUG FIX: Shear stress now uses actual weight above from placed panels
#   - BUG FIX: Rejection reasons now tracked and displayed
#   - BUG FIX: Typo "Optimizaiton" -> "Optimization"
#   - Cleaned up unused imports
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


@dataclass
class StressConfig:
    max_compression_psi: float = 50.0
    max_bending_moment_lbf_in: float = 10000.0
    max_shear_psi: float = 30.0
    safety_factor: float = 2.0
    panel_youngs_modulus_psi: float = 1800000.0


@dataclass
class PanelSpec:
    """Defines a panel type with its dimensions and properties."""
    panel_type: PanelType
    length: float       # inches
    height: float       # inches
    thickness: float    # inches
    weight: float       # lbs
    # For trapezoid floor panels: the shorter parallel edge
    short_edge: float = 0.0  # 0 means rectangular (wall panel)

    @property
    def is_trapezoid(self):
        return self.short_edge > 0

    @property
    def bounding_box(self):
        """Returns (L, H, T) bounding box for collision detection."""
        return (self.length, self.height, self.thickness)


# Default panel specs
WALL_PANEL = PanelSpec(
    panel_type=PanelType.WALL,
    length=112.0,
    height=97.25,
    thickness=5.5,
    weight=220.0,
    short_edge=0.0
)

FLOOR_PANEL = PanelSpec(
    panel_type=PanelType.FLOOR,
    length=224.0,       # long parallel edge (inner side)
    height=111.87,      # the angled side dimension (depth of trapezoid)
    thickness=6.5,      # panel thickness
    weight=585.15,
    short_edge=112.0    # short parallel edge (outer side)
)


# ─── Geometry / Collision ────────────────────────────────────────────────────

def aabb_intersect(p1, s1, p2, s2):
    """Axis-aligned bounding box intersection test."""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return not (
        x1 + s1[0] <= x2 or x2 + s2[0] <= x1 or
        y1 + s1[1] <= y2 or y2 + s2[1] <= y1 or
        z1 + s1[2] <= z2 or z2 + s2[2] <= z1
    )


def in_bounds(pos, size, trailer):
    """Check if a panel at pos with given size fits within trailer bounds."""
    x, y, z = pos
    return (
        x >= 0 and y >= 0 and z >= 0 and
        x + size[0] <= trailer[0] and
        y + size[1] <= trailer[1] and
        z + size[2] <= trailer[2]
    )


def handling_ok(size, max_horizontal, max_vertical):
    """Check if panel can be physically handled given constraints."""
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical


# ─── Stress Calculations ────────────────────────────────────────────────────

def calculate_support_area(pos, size, placed, z_tol=1e-6):
    """Calculate how much of this panel's bottom is supported by panels below."""
    x, y, z = pos
    dx, dy, dz = size

    # On the trailer floor = fully supported
    if z <= z_tol:
        return dx * dy, []

    total_area = 0.0
    supporting_panels = []

    for p in placed:
        px, py, pz = p["pos"]
        pdx, pdy, pdz = p["size"]
        top = pz + pdz

        if abs(top - z) > z_tol:
            continue

        ox = max(0.0, min(x + dx, px + pdx) - max(x, px))
        oy = max(0.0, min(y + dy, py + pdy) - max(y, py))
        overlap_area = ox * oy

        if overlap_area > 1e-6:
            total_area += overlap_area
            supporting_panels.append({
                'panel': p,
                'overlap_area': overlap_area,
                'centroid': (px + pdx / 2, py + pdy / 2, top)
            })

    return total_area, supporting_panels


def calculate_compression_stress(weight, support_area):
    """Simple compression: weight / area."""
    if support_area < 1e-6:
        return float('inf')
    return weight / support_area


def calculate_bending_stress(panel_weight, size, supporting_panels):
    """Estimate bending stress from overhang relative to supports."""
    if not supporting_panels:
        return 0.0

    dx, dy, dz = size
    moment_of_inertia = (dy * dz ** 3) / 12.0

    max_moment = 0.0
    for support in supporting_panels:
        overhang_x = abs(support['centroid'][0] - dx / 2)
        overhang_y = abs(support['centroid'][1] - dy / 2)
        max_overhang = max(overhang_x, overhang_y)
        moment = panel_weight * max_overhang
        max_moment = max(max_moment, moment)

    if moment_of_inertia < 1e-6:
        return float('inf')

    return (max_moment * (dz / 2.0)) / moment_of_inertia


def calculate_shear_stress(total_weight_above, size):
    """Shear stress on the panel's cross section."""
    dx, dy, dz = size
    shear_area = min(dx, dy) * dz

    if shear_area < 1e-6:
        return float('inf')

    return total_weight_above / shear_area


def calculate_deflection(panel_weight, size, youngs_modulus, supporting_panels):
    """Cantilever deflection estimate for overhang."""
    if not supporting_panels:
        return 0.0

    dx, dy, dz = size
    moment_of_inertia = (dy * dz ** 3) / 12.0

    if moment_of_inertia < 1e-6:
        return float('inf')

    max_overhang = 0.0
    for support in supporting_panels:
        overhang = abs(support['centroid'][0] - dx / 2)
        max_overhang = max(max_overhang, overhang)

    if max_overhang < 1e-6:
        return 0.0

    deflection = (panel_weight * max_overhang ** 3) / (3.0 * youngs_modulus * moment_of_inertia)
    return deflection


def get_weight_above(panel_id, placed):
    """Calculate total weight of panels resting on top of this panel."""
    total_weight = 0.0
    panel = placed[panel_id]
    px, py, pz = panel["pos"]
    pdx, pdy, pdz = panel["size"]
    top = pz + pdz

    for other in placed[panel_id + 1:]:
        ox, oy, oz = other["pos"]
        odx, ody, odz = other["size"]

        if oz < top - 1e-6:
            continue

        x_overlap = max(0.0, min(px + pdx, ox + odx) - max(px, ox))
        y_overlap = max(0.0, min(py + pdy, oy + ody) - max(py, oy))

        if x_overlap > 1e-6 and y_overlap > 1e-6:
            total_weight += other["weight"]

    return total_weight


def stress_ok(pos, size, weight, placed, stress_config, z_tol=1e-6, min_support_frac=0.7):
    """
    Master stress validation.
    BUG FIX V1.1: Safety factor is now multiplied against the actual stress
    (not dividing the limit). Also shear now accounts for weight above.
    """
    x, y, z = pos
    dx, dy, dz = size
    panel_area = dx * dy

    support_area, supporting_panels = calculate_support_area(pos, size, placed, z_tol)

    if support_area < panel_area * min_support_frac:
        return False, "Insufficient support area"

    # Compression check (FIX: multiply stress by SF, compare to limit)
    compression_stress = calculate_compression_stress(weight, support_area)
    if compression_stress * stress_config.safety_factor > stress_config.max_compression_psi:
        return False, f"Compression stress too high: {compression_stress:.1f} psi (x{stress_config.safety_factor} SF = {compression_stress * stress_config.safety_factor:.1f})"

    # Bending check
    bending_stress = calculate_bending_stress(weight, size, supporting_panels)
    if bending_stress * stress_config.safety_factor > stress_config.max_bending_moment_lbf_in:
        return False, f"Bending stress too high: {bending_stress:.1f}"

    # Shear check (FIX: use actual weight above from already-placed panels)
    weight_above = sum(
        p["weight"] for p in placed
        if p["pos"][2] >= z + dz - 1e-6
        and max(0.0, min(x + dx, p["pos"][0] + p["size"][0]) - max(x, p["pos"][0])) > 1e-6
        and max(0.0, min(y + dy, p["pos"][1] + p["size"][1]) - max(y, p["pos"][1])) > 1e-6
    )
    total_shear_weight = weight + weight_above
    shear_stress = calculate_shear_stress(total_shear_weight, size)
    if shear_stress * stress_config.safety_factor > stress_config.max_shear_psi:
        return False, f"Shear stress too high: {shear_stress:.1f} psi"

    # Deflection check
    deflection = calculate_deflection(weight, size, stress_config.panel_youngs_modulus_psi, supporting_panels)
    max_deflection = min(dx, dy) / 360.0
    if deflection > max_deflection:
        return False, f"Deflection too large: {deflection:.3f} in"

    return True, "OK"


# ─── Orientation Generation ─────────────────────────────────────────────────

def generate_orientations(L, H, T):
    """Generate all 6 axis-aligned orientations for a rectangular panel."""
    return [
        ("flat_LxH", (L, H, T)),
        ("flat_HxL", (H, L, T)),
        ("stand_LxT", (L, T, H)),
        ("stand_TxL", (T, L, H)),
        ("stand_HxT", (H, T, L)),
        ("stand_TxH", (T, H, L)),
    ]


# ─── Position Search ────────────────────────────────────────────────────────

def find_lowest_position(x, y, size, weight, placed, trailer, stress_config, z_tol=1e-6, step=1.0):
    """Find the lowest valid z position at (x, y) for a panel."""
    dx, dy, dz = size
    max_z_steps = int(trailer[2] / step) + 1

    for z_idx in range(max_z_steps):
        z = z_idx * step
        test_pos = (x, y, z)

        if not in_bounds(test_pos, size, trailer):
            continue

        stress_valid, msg = stress_ok(test_pos, size, weight, placed, stress_config, z_tol)
        if not stress_valid:
            continue

        if any(aabb_intersect(test_pos, size, p["pos"], p["size"]) for p in placed):
            continue

        return test_pos

    return None


# ─── Position Generation Strategies ─────────────────────────────────────────

def generate_positions_gravity_layered(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    positions = [(x, y) for x in xs for y in ys]
    positions.sort(key=lambda p: (p[0], p[1]))
    return positions


def generate_positions_wall_first(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    positions = []
    for x in xs:
        for y in ys:
            dist_to_wall = min(x, y, trailer[0] - x, trailer[1] - y)
            positions.append((x, y, dist_to_wall))
    positions.sort(key=lambda p: (p[2], p[0] + p[1]))
    return [(x, y) for x, y, _ in positions]


def generate_positions_zone_based(trailer, step, num_zones=3):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    zone_width = trailer[0] / num_zones
    positions = []
    for zone in range(num_zones):
        zone_start = zone * zone_width
        zone_end = (zone + 1) * zone_width
        zone_positions = [(x, y) for x in xs if zone_start <= x < zone_end for y in ys]
        zone_positions.sort(key=lambda p: (p[0], p[1]))
        positions.extend(zone_positions)
    return positions


def generate_positions_stress_optimized(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    center_x = trailer[0] / 2
    center_y = trailer[1] / 2
    positions = [(x, y, ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5) for x in xs for y in ys]
    positions.sort(key=lambda p: p[2])
    return [(x, y) for x, y, _ in positions]


# ─── Main Packing Engine ────────────────────────────────────────────────────

def build_panel_list(num_pods, wall_spec, floor_spec):
    """
    Build an ordered list of panels to pack based on pod count.
    Per pod: 6 wall panels + 2 floor panels (half-hex).
    Floor panels are heavier so they go first (bottom of trailer).
    """
    panels = []

    # Floors first (heavier, go on bottom)
    for pod in range(num_pods):
        for half in range(2):
            panels.append({
                "spec": floor_spec,
                "label": f"Pod{pod + 1}_Floor_{half + 1}",
                "panel_type": PanelType.FLOOR.value,
            })

    # Then walls
    for pod in range(num_pods):
        for wall in range(6):
            panels.append({
                "spec": wall_spec,
                "label": f"Pod{pod + 1}_Wall_{wall + 1}",
                "panel_type": PanelType.WALL.value,
            })

    return panels


def pack_panels_multi(panel_list, trailer_L, trailer_W, trailer_H,
                      max_horizontal, max_vertical,
                      step, seed, strategy, stress_config):
    """
    V1.1 packing engine: supports multiple panel types.
    Returns results dict with placements and rejection info.
    """
    random.seed(seed)
    trailer = (trailer_L, trailer_W, trailer_H)

    # Build position candidates based on strategy
    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        xy_positions = generate_positions_gravity_layered(trailer, step)
    elif strategy == LoadingStrategy.WALL_FIRST:
        xy_positions = generate_positions_wall_first(trailer, step)
    elif strategy == LoadingStrategy.ZONE_BASED:
        xy_positions = generate_positions_zone_based(trailer, step)
    else:
        xy_positions = generate_positions_stress_optimized(trailer, step)

    placed = []
    rejected = []

    for pid, panel_info in enumerate(panel_list):
        spec = panel_info["spec"]
        bbox = spec.bounding_box  # (L, H, T)

        # Generate valid orientations for this panel's bounding box
        orientations = []
        for name, size in generate_orientations(bbox[0], bbox[1], bbox[2]):
            if in_bounds((0, 0, 0), size, trailer):
                orientations.append((name, size))

        if not orientations:
            rejected.append({
                "id": pid,
                "label": panel_info["label"],
                "panel_type": panel_info["panel_type"],
                "reason": "No valid orientation fits in trailer"
            })
            continue

        random.shuffle(orientations)
        placed_this = False

        for name, size in orientations:
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            for x, y in xy_positions:
                final_pos = find_lowest_position(
                    x, y, size, spec.weight, placed, trailer, stress_config, step=step
                )

                if final_pos is None:
                    continue

                placed.append({
                    "id": pid,
                    "label": panel_info["label"],
                    "panel_type": panel_info["panel_type"],
                    "pos": final_pos,
                    "size": size,
                    "orientation": name,
                    "weight": spec.weight,
                    "is_trapezoid": spec.is_trapezoid,
                    "short_edge": spec.short_edge,
                    "spec_dims": {"L": spec.length, "H": spec.height, "T": spec.thickness},
                })
                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            rejected.append({
                "id": pid,
                "label": panel_info["label"],
                "panel_type": panel_info["panel_type"],
                "reason": "No valid position found (all positions failed stress/collision checks)"
            })

    # Post-placement stress analysis
    for i, panel in enumerate(placed):
        support_area, supporting_panels = calculate_support_area(panel["pos"], panel["size"], placed[:i])
        compression = calculate_compression_stress(panel["weight"], support_area)
        bending = calculate_bending_stress(panel["weight"], panel["size"], supporting_panels)
        weight_above = get_weight_above(i, placed)
        shear = calculate_shear_stress(weight_above + panel["weight"], panel["size"])
        deflection = calculate_deflection(
            panel["weight"], panel["size"],
            stress_config.panel_youngs_modulus_psi, supporting_panels
        )

        panel["stress_analysis"] = {
            "compression_psi": round(compression, 2),
            "bending_stress": round(bending, 2),
            "shear_psi": round(shear, 2),
            "deflection_in": round(deflection, 4),
            "weight_above_lb": round(weight_above, 2),
            "support_area_sqin": round(support_area, 2)
        }

    # Count by type
    wall_count = sum(1 for p in placed if p["panel_type"] == PanelType.WALL.value)
    floor_count = sum(1 for p in placed if p["panel_type"] == PanelType.FLOOR.value)

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
            "min_support_frac": 0.7,
            "packing_strategy": strategy.value
        },
        "placements": [
            {
                "id": p["id"],
                "label": p["label"],
                "panel_type": p["panel_type"],
                "position_vector": [round(p["pos"][0], 3), round(p["pos"][1], 3), round(p["pos"][2], 3)],
                "size": [round(p["size"][0], 3), round(p["size"][1], 3), round(p["size"][2], 3)],
                "orientation": p["orientation"],
                "weight": p["weight"],
                "is_trapezoid": p["is_trapezoid"],
                "layer": int(round(p["pos"][2] / step)) if step > 0 else 0,
                "stress_analysis": p["stress_analysis"]
            }
            for p in placed
        ],
        "rejections": rejected
    }


# ─── 3D Visualization ───────────────────────────────────────────────────────

def box_edges(x, y, z, dx, dy, dz):
    """Generate wireframe edge coordinates for a rectangular box."""
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


def trapezoid_vertices(x, y, z, dx, dy, dz, short_edge_ratio):
    """
    Generate vertices for a trapezoid-shaped panel (half hexagon floor).
    The trapezoid has:
      - Bottom face: long edge = dx, short edge = dx * short_edge_ratio, depth = dy
      - Height (thickness) = dz
    Short edge is centered along the x-axis.
    """
    offset = (dx - dx * short_edge_ratio) / 2.0
    se = dx * short_edge_ratio

    # Bottom face (z)
    b0 = (x, y, z)              # long edge start
    b1 = (x + dx, y, z)         # long edge end
    b2 = (x + offset + se, y + dy, z)   # short edge end
    b3 = (x + offset, y + dy, z)        # short edge start

    # Top face (z + dz)
    t0 = (x, y, z + dz)
    t1 = (x + dx, y, z + dz)
    t2 = (x + offset + se, y + dy, z + dz)
    t3 = (x + offset, y + dy, z + dz)

    return [b0, b1, b2, b3, t0, t1, t2, t3]


def add_wire(fig, x, y, z, dx, dy, dz, color, name, opacity=1.0, width=4):
    """Add wireframe box to figure."""
    xs, ys, zs = box_edges(x, y, z, dx, dy, dz)
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        name=name,
        showlegend=(name != "")
    ))


def add_solid(fig, x, y, z, dx, dy, dz, color, opacity=0.35):
    """Add solid rectangular box to figure."""
    vx = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    vy = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    vz = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)
    ]
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color=color, opacity=opacity,
        showlegend=False
    ))


def add_trapezoid_solid(fig, x, y, z, dx, dy, dz, short_edge_ratio, color, opacity=0.35):
    """Add a solid trapezoid (half-hexagon floor panel) to the figure."""
    verts = trapezoid_vertices(x, y, z, dx, dy, dz, short_edge_ratio)
    vx = [v[0] for v in verts]
    vy = [v[1] for v in verts]
    vz = [v[2] for v in verts]

    # 8 vertices: 0-3 bottom, 4-7 top
    # Triangulate each face
    faces = [
        # Bottom
        (0, 1, 2), (0, 2, 3),
        # Top
        (4, 5, 6), (4, 6, 7),
        # Long edge side (front)
        (0, 1, 5), (0, 5, 4),
        # Short edge side (back)
        (3, 2, 6), (3, 6, 7),
        # Left angled side
        (0, 3, 7), (0, 7, 4),
        # Right angled side
        (1, 2, 6), (1, 6, 5),
    ]
    fig.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color=color, opacity=opacity,
        showlegend=False
    ))


def add_trapezoid_wire(fig, x, y, z, dx, dy, dz, short_edge_ratio, color, name="", opacity=1.0, width=4):
    """Add wireframe trapezoid to figure."""
    verts = trapezoid_vertices(x, y, z, dx, dy, dz, short_edge_ratio)
    # Edges: bottom face, top face, verticals
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [verts[a][0], verts[b][0], None]
        ys += [verts[a][1], verts[b][1], None]
        zs += [verts[a][2], verts[b][2], None]

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        name=name,
        showlegend=(name != "")
    ))


def get_stress_color(stress_val, max_val):
    """Green/Yellow/Red based on stress ratio."""
    if max_val < 1e-6:
        return "#2ecc71"
    ratio = min(stress_val / max_val, 1.0)
    if ratio < 0.5:
        return "#2ecc71"
    elif ratio < 0.75:
        return "#f1c40f"
    else:
        return "#e74c3c"


def visualize(out, stress_config, color_by_stress=False):
    """Build and display the 3D Plotly visualization."""
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]

    fig = go.Figure()

    # Trailer floor
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.2)
    # Trailer wireframe
    add_wire(fig, 0, 0, 0, L, W, H, "#ecf0f1", "", opacity=0.3, width=2)

    # Floor grid lines
    step_viz = 50
    for i in range(0, int(L), step_viz):
        add_wire(fig, i, 0, 0, 0, W, 0, "#34495e", "", opacity=0.15, width=1)
    for j in range(0, int(W), step_viz):
        add_wire(fig, 0, j, 0, L, 0, 0, "#34495e", "", opacity=0.15, width=1)

    # Color palettes - different for walls vs floors
    wall_palette = ["#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#95a5a6"]
    floor_palette = ["#e67e22", "#e74c3c", "#f1c40f", "#d35400"]

    wall_color_idx = 0
    floor_color_idx = 0
    type_colors = {}

    for p in out["placements"]:
        ptype = p["panel_type"]
        orient = p["orientation"]
        key = f"{ptype}_{orient}"

        if key not in type_colors:
            if ptype == PanelType.FLOOR.value:
                type_colors[key] = floor_palette[floor_color_idx % len(floor_palette)]
                floor_color_idx += 1
            else:
                type_colors[key] = wall_palette[wall_color_idx % len(wall_palette)]
                wall_color_idx += 1

        if color_by_stress:
            stress = p["stress_analysis"]["compression_psi"]
            c = get_stress_color(stress, stress_config.max_compression_psi)
        else:
            c = type_colors[key]

        x, y, z = p["position_vector"]
        dx, dy, dz = p["size"]

        if p.get("is_trapezoid", False):
            # Calculate short_edge_ratio from the original spec
            # short_edge / long_edge
            short_edge_ratio = FLOOR_PANEL.short_edge / FLOOR_PANEL.length
            add_trapezoid_solid(fig, x, y, z, dx, dy, dz, short_edge_ratio, c, opacity=0.45)
            add_trapezoid_wire(fig, x, y, z, dx, dy, dz, short_edge_ratio, c, "", opacity=0.9, width=2)
        else:
            add_solid(fig, x, y, z, dx, dy, dz, c, opacity=0.4)
            add_wire(fig, x, y, z, dx, dy, dz, c, "", opacity=0.9, width=2)

    # Legend entries
    if not color_by_stress:
        legend_added = set()
        for key, c in type_colors.items():
            ptype, orient = key.split("_", 1)
            count = sum(1 for p in out["placements"] if p["panel_type"] == ptype and p["orientation"] == orient)
            short_type = "Wall" if "Wall" in ptype else "Floor"
            legend_label = f"{short_type} - {orient} ({count})"
            if legend_label not in legend_added:
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="lines",
                    line=dict(color=c, width=6),
                    name=legend_label
                ))
                legend_added.add(legend_label)

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="Length (in)",
            yaxis_title="Width (in)",
            zaxis_title="Height (in)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=11)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    st.subheader("Structural Analysis Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Panels Loaded", f"{out['settings']['placed_panels']} / {out['settings']['requested_panels']}")
    with col2:
        st.metric("Walls / Floors", f"{out['settings']['placed_walls']}W + {out['settings']['placed_floors']}F")
    with col3:
        total_weight = sum(p["weight"] for p in out["placements"])
        st.metric("Total Weight", f"{total_weight:,.0f} lb")
    with col4:
        max_compression = max((p["stress_analysis"]["compression_psi"] for p in out["placements"]), default=0)
        st.metric("Max Compression", f"{max_compression:.1f} psi")
    with col5:
        max_deflection = max((p["stress_analysis"]["deflection_in"] for p in out["placements"]), default=0)
        st.metric("Max Deflection", f"{max_deflection:.4f} in")


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="MCLOS - Modular Cargo Loading Optimization", layout="wide")

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("MCLOS - Modular Cargo Loading Optimization Software")
st.caption("HexHomes Panel Packing Optimizer  |  Alpha V1.1")

# ─── Initialize session state ────────────────────────────────────────────────
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 99999)
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

# ─── Sidebar: Strategy & Display ────────────────────────────────────────────
with st.sidebar:
    st.header("Loading Strategy")
    strategy_name = st.selectbox(
        "Strategy",
        [s.value for s in LoadingStrategy],
        index=3,
        help="Controls how the optimizer searches for panel positions on the trailer."
    )
    strategy = [s for s in LoadingStrategy if s.value == strategy_name][0]

    st.markdown("---")
    st.header("Display Options")
    color_by_stress = st.checkbox("Color by Stress Level", value=False,
                                  help="Green = low stress, Yellow = moderate, Red = high")

    st.markdown("---")

    # Advanced Settings (collapsed by default)
    with st.expander("Advanced Settings"):
        st.subheader("Structural Limits")
        max_comp = st.number_input("Max Compression (psi)", value=50.0, min_value=1.0, key="adv_comp")
        max_bend = st.number_input("Max Bending (lbf-in)", value=10000.0, min_value=1.0, key="adv_bend")
        max_shear = st.number_input("Max Shear (psi)", value=30.0, min_value=1.0, key="adv_shear")
        safety = st.number_input("Safety Factor", value=2.0, min_value=1.0, key="adv_safety")
        youngs = st.number_input("Young's Modulus (psi)", value=1800000.0, min_value=1000.0, key="adv_youngs")

        st.markdown("---")
        st.subheader("Handling Limits")
        maxH = st.number_input("Max Horizontal Span (in)", value=230.0, min_value=1.0, key="adv_maxh",
                               help="Maximum horizontal dimension a worker can handle")
        maxV = st.number_input("Max Vertical Span (in)", value=114.0, min_value=1.0, key="adv_maxv",
                               help="Maximum vertical dimension a worker can handle")

        st.markdown("---")
        st.subheader("Grid & Seed")
        step = st.number_input("Grid Step (in)", min_value=0.5, value=2.0, step=0.5, key="adv_step",
                               help="Smaller = more precise but slower")
        manual_seed = st.number_input("Manual Seed Override", min_value=0, value=st.session_state.seed,
                                      step=1, key="adv_seed",
                                      help="Set a specific seed for reproducibility")
        if manual_seed != st.session_state.seed:
            st.session_state.seed = manual_seed

stress_config = StressConfig(
    max_compression_psi=max_comp,
    max_bending_moment_lbf_in=max_bend,
    max_shear_psi=max_shear,
    safety_factor=safety,
    panel_youngs_modulus_psi=youngs
)

# ─── Main Content ────────────────────────────────────────────────────────────

# Pod selection
st.subheader("Pod Configuration")
col_pod, col_summary = st.columns([1, 2])

with col_pod:
    num_pods = st.number_input("Number of Pods", min_value=1, max_value=6, value=6, step=1,
                               help="Each pod = 6 wall panels + 2 half-hex floor panels")

total_walls = num_pods * 6
total_floors = num_pods * 2
total_panels = total_walls + total_floors

with col_summary:
    st.markdown(f"""
    **{num_pods} Pod{'s' if num_pods > 1 else ''}** = **{total_panels} panels total**
    - {total_walls} wall panels (112" x 97.25" x 5.5", 220 lb each)
    - {total_floors} floor panels (224" x 111.87" x 6.5", 585.15 lb each)
    """)

# Trailer dimensions
st.subheader("Trailer Dimensions")
tc1, tc2, tc3 = st.columns(3)
with tc1:
    tL = st.number_input("Length (in)", value=636.0, min_value=1.0, key="trailer_l",
                         help="53-foot trailer = 636 inches")
with tc2:
    tW = st.number_input("Width (in)", value=102.0, min_value=1.0, key="trailer_w")
with tc3:
    tH = st.number_input("Height (in)", value=110.0, min_value=1.0, key="trailer_h",
                         help="Usable interior height")

# ─── Panel Dimension Overrides (expandable) ──────────────────────────────────
with st.expander("Panel Dimensions (edit if needed)"):
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("**Wall Panel**")
        wL = st.number_input("Wall Length (in)", value=WALL_PANEL.length, min_value=1.0, key="wl")
        wH = st.number_input("Wall Height (in)", value=WALL_PANEL.height, min_value=1.0, key="wh")
        wT = st.number_input("Wall Thickness (in)", value=WALL_PANEL.thickness, min_value=0.1, key="wt")
        wW = st.number_input("Wall Weight (lb)", value=WALL_PANEL.weight, min_value=0.1, key="ww")

    with pc2:
        st.markdown("**Floor Panel (Half-Hex)**")
        fL = st.number_input("Floor Long Edge (in)", value=FLOOR_PANEL.length, min_value=1.0, key="fl",
                             help="Inner/long parallel edge of trapezoid")
        fH = st.number_input("Floor Depth (in)", value=FLOOR_PANEL.height, min_value=1.0, key="fh",
                             help="Angled side dimension")
        fT = st.number_input("Floor Thickness (in)", value=FLOOR_PANEL.thickness, min_value=0.1, key="ft")
        fW = st.number_input("Floor Weight (lb)", value=FLOOR_PANEL.weight, min_value=0.1, key="fw")
        fS = st.number_input("Floor Short Edge (in)", value=FLOOR_PANEL.short_edge, min_value=1.0, key="fs",
                             help="Outer/short parallel edge of trapezoid")

# Build specs from UI values
wall_spec = PanelSpec(
    panel_type=PanelType.WALL,
    length=wL, height=wH, thickness=wT, weight=wW
)
floor_spec = PanelSpec(
    panel_type=PanelType.FLOOR,
    length=fL, height=fH, thickness=fT, weight=fW,
    short_edge=fS
)

# ─── Run Button ──────────────────────────────────────────────────────────────

st.markdown("---")

run_col1, run_col2 = st.columns([3, 1])
with run_col1:
    run_button = st.button("Run Optimization", type="primary", use_container_width=True)
with run_col2:
    st.markdown(f"**Seed:** `{st.session_state.seed}`")
    st.caption(f"Run #{st.session_state.run_count}")

if run_button:
    # Build panel list
    panel_list = build_panel_list(num_pods, wall_spec, floor_spec)

    # Run with timer
    start_time = time.time()
    with st.spinner(f"Optimizing {len(panel_list)} panels across {num_pods} pod(s)..."):
        out = pack_panels_multi(
            panel_list, tL, tW, tH,
            maxH, maxV,
            step, st.session_state.seed, strategy, stress_config
        )
    elapsed = time.time() - start_time

    # Auto-randomize seed for next run
    st.session_state.seed = random.randint(0, 99999)
    st.session_state.run_count += 1

    # ─── Results ─────────────────────────────────────────────────────────
    if "error" not in out:
        success_rate = (out["settings"]["placed_panels"] / out["settings"]["requested_panels"]) * 100

        # Status bar
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            if success_rate == 100:
                st.success(f"All {out['settings']['placed_panels']} panels loaded!")
            else:
                st.warning(f"Loaded {out['settings']['placed_panels']} / {out['settings']['requested_panels']} ({success_rate:.1f}%)")
        with status_col2:
            st.info(f"Optimization time: **{elapsed:.2f}s**")
        with status_col3:
            st.info(f"Seed used: **{out['settings']['seed']}**")

        # 3D Visualization
        st.subheader("3D Trailer View")
        visualize(out, stress_config, color_by_stress)

        # Rejection report
        if out["rejections"]:
            with st.expander(f"Rejected Panels ({len(out['rejections'])})", expanded=True):
                for r in out["rejections"]:
                    st.error(f"**{r['label']}** ({r['panel_type']}): {r['reason']}")

        # Detailed stress table
        with st.expander("Detailed Stress Analysis"):
            stress_data = []
            for p in out.get("placements", []):
                sa = p["stress_analysis"]
                stress_data.append({
                    "Panel": p["label"],
                    "Type": p["panel_type"],
                    "Orientation": p["orientation"],
                    "Compression (psi)": sa["compression_psi"],
                    "Bending Stress": sa["bending_stress"],
                    "Shear (psi)": sa["shear_psi"],
                    "Deflection (in)": sa["deflection_in"],
                    "Weight Above (lb)": sa["weight_above_lb"],
                    "Support Area (in^2)": sa["support_area_sqin"]
                })
            st.dataframe(stress_data, use_container_width=True)

        # JSON output
        with st.expander("View JSON Output"):
            st.json(out)

        # Downloads
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download JSON",
                data=json.dumps(out, indent=2),
                file_name=f"mclos_result_seed{out['settings']['seed']}.json",
                mime="application/json",
                use_container_width=True
            )
        with dl2:
            csv_header = "id,label,type,x,y,z,length,width,height,orientation,weight,compression_psi,bending,shear_psi,deflection_in\n"
            csv_rows = "\n".join([
                f"{p['id']},{p['label']},{p['panel_type']},"
                f"{p['position_vector'][0]},{p['position_vector'][1]},{p['position_vector'][2]},"
                f"{p['size'][0]},{p['size'][1]},{p['size'][2]},{p['orientation']},{p['weight']},"
                f"{p['stress_analysis']['compression_psi']},{p['stress_analysis']['bending_stress']},"
                f"{p['stress_analysis']['shear_psi']},{p['stress_analysis']['deflection_in']}"
                for p in out.get("placements", [])
            ])
            st.download_button(
                "Download CSV",
                data=csv_header + csv_rows,
                file_name=f"mclos_stress_seed{out['settings']['seed']}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.error(out["error"])
