import json
import time
import math
import base64
import os
import streamlit as st
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass

#V1.1 — Added floor panels with half-hexagon geometry and trapezoid rendering (Imperfect).
#V1.2 — Trailer presets (53ft/42ft enclosed & flatbed), manual panel count input.
#V1.3 — Orientation tracking (axis_map), improved stacking logic.
#V1.4 — Full packing engine rewrite. Layer-first filling, column tracking, 4 distinct loading strategies with different orientation preferences, CG & weight distribution, CSV export.
#V1.5 — Effective compression stress metric, worst case panel with 3D marker.
#V1.6 — Axle load analysis with moment-balance physics, DOT weight limits, drive/trailer tandem visualization, axle data in exports.
#V1.7 — Added user welcome page, and 3D visualization for axles.
#V1.8 — HexHomes honey/gold UI theme, logo integration, version string fixes, download filename corrections.
#V1.8.1 — Flatbed height corrected to 168" (14ft CA legal max without permit).
#V1.9 — Remove randomness, eliminate the implicit ~1.5" spacer, add trailer tolerance, balance axle loads, and automatically pick the lowest-load configuration.
#V2.0 — 1.5" default spacer, sidebar layout fix, upgraded stress calculations (biaxial bending, parabolic shear, proper deflection with Young's modulus), utilization-based worst-panel metric, removed effective compression.
#V3.0 — Flatbed overhang support (floor panels extend past deck width), true half-hex polygon collision (SAT), tessellated nesting of half-hex panels, flipped panel tracking. Floor panels now load on flatbed trailers.

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


@dataclass
class AxleConfig:
    """Axle positions measured from trailer front (x=0 = kingpin/cargo start).
    Standard 5-axle tractor-trailer: steer, drive tandem (2 axles), trailer tandem (2 axles).
    Drive tandem is on the tractor, positioned behind the kingpin relative to trailer cargo.
    Negative x = ahead of cargo area (on the tractor).
    """
    # Drive tandem (tractor rear axles) — center position from trailer front
    drive_tandem_x: float          # inches from trailer front (typically negative = on tractor)
    drive_tandem_spacing: float    # inches between the two drive axles
    # Trailer tandem — center position from trailer front
    trailer_tandem_x: float        # inches from trailer front
    trailer_tandem_spacing: float  # inches between the two trailer axles
    # Weight limits (federal DOT)
    steer_limit_lb: float = 12000.0     # Federal practical limit (bridge formula)
    drive_tandem_limit_lb: float = 34000.0   # Federal tandem limit
    trailer_tandem_limit_lb: float = 34000.0  # Federal tandem limit
    gross_limit_lb: float = 80000.0     # Federal GVW limit
    # Tractor weight (empty, distributed across steer + drive)
    tractor_weight_lb: float = 17000.0  # Typical day cab tractor weight
    # Tractor wheelbase: steer axle to drive tandem center
    tractor_wheelbase: float = 240.0    # ~20 ft typical


# Axle configs for standard trailer types
# 53-ft: Kingpin at x=0, trailer tandem center ~480" (40 ft) from kingpin
# 42-ft: Kingpin at x=0, trailer tandem center ~420" (35 ft) from kingpin
# Drive tandem: ~36" behind kingpin = x=-36 relative to cargo start
# Tandem spacing: ~49" (4'1") standard

AXLE_CONFIGS = {
    TrailerPreset.FT53_ENCLOSED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=480.0, trailer_tandem_spacing=49.0,
    ),
    TrailerPreset.FT53_FLATBED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=480.0, trailer_tandem_spacing=49.0,
    ),
    TrailerPreset.FT42_ENCLOSED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=420.0, trailer_tandem_spacing=49.0,
    ),
    TrailerPreset.FT42_FLATBED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=420.0, trailer_tandem_spacing=49.0,
    ),
}


TRAILER_DIMS = {
    TrailerPreset.FT53_ENCLOSED: {"L": 636.0, "W": 102.0, "H": 110.0,
                                   "desc": "53-ft enclosed trailer (standard dry van)"},
    TrailerPreset.FT53_FLATBED:  {"L": 636.0, "W": 102.0, "H": 168.0,
                                   "desc": "53-ft flatbed (open, legal height limit: 14ft / 168\" in CA without permit)"},
    TrailerPreset.FT42_ENCLOSED: {"L": 504.0, "W": 102.0, "H": 110.0,
                                   "desc": "42-ft enclosed trailer (standard)"},
    TrailerPreset.FT42_FLATBED:  {"L": 504.0, "W": 102.0, "H": 168.0,
                                   "desc": "42-ft flatbed (current HexHomes trailer, legal height limit: 14ft / 168\" in CA without permit)"},
}


@dataclass
class StressConfig:
    max_compression_psi: float = 50.0
    max_bending_stress_psi: float = 500.0
    max_shear_psi: float = 45.0
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


def in_bounds_with_overhang(pos, size, trailer, is_flatbed=False,
                             overhang_max=0.0, panel_is_floor_flat=False):
    """Bounds check allowing Y-overhang for floor panels on flatbed trailers.

    Overhang rules:
    - Only Y-axis (width) can overhang — no physical side walls on flatbed
    - Only for floor panels in flat orientation on flatbed trailers
    - X and Z bounds remain strict
    - Panel center-of-mass Y must remain within deck width
    """
    x, y, z = pos
    dx, dy, dz = size
    tL, tW, tH = trailer

    # X and Z: always strict
    if x < -0.01 or x + dx > tL + 0.01:
        return False
    if z < -0.01 or z + dz > tH + 0.01:
        return False

    # Y: allow overhang if conditions met
    if is_flatbed and panel_is_floor_flat and overhang_max > 0:
        if y < -(overhang_max + 0.01):
            return False
        if y + dy > tW + overhang_max + 0.01:
            return False
        # Safety: panel center Y must remain within deck
        panel_center_y = y + dy / 2.0
        if panel_center_y < 0 or panel_center_y > tW:
            return False
        return True
    else:
        return y >= -0.01 and y + dy <= tW + 0.01


def handling_ok(size, max_horizontal, max_vertical):
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical


# ─── Half-Hex Polygon Collision (V3.0) ───────────────────────────────────────

def get_trapezoid_footprint_2d(pos, size, axis_map, short_edge, long_edge, flipped=False):
    """Return the 2D XY polygon (list of (x,y) tuples) for a flat trapezoid panel.

    Normal (flipped=False):
        Wide edge at Y=y, narrow edge at Y=y+dy
    Flipped (flipped=True):
        Narrow edge at Y=y, wide edge at Y=y+dy
    """
    x, y, _ = pos
    dx, dy, _ = size
    offset = (long_edge - short_edge) / 2.0
    x_axis, y_axis = axis_map[0], axis_map[1]

    if x_axis == "L" and y_axis == "H":
        if not flipped:
            return [(x, y), (x + dx, y),
                    (x + offset + short_edge, y + dy), (x + offset, y + dy)]
        else:
            return [(x + offset, y), (x + offset + short_edge, y),
                    (x + dx, y + dy), (x, y + dy)]
    elif x_axis == "H" and y_axis == "L":
        if not flipped:
            return [(x, y), (x + dx, y + offset),
                    (x + dx, y + offset + short_edge), (x, y + dy)]
        else:
            return [(x, y + offset), (x + dx, y),
                    (x + dx, y + dy), (x, y + offset + short_edge)]
    # Non-flat orientations: rectangular footprint
    return [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]


def polygons_overlap_2d(poly_a, poly_b):
    """SAT (Separating Axis Theorem) for two convex 2D polygons.
    Returns True if polygons overlap."""
    def get_axes(polygon):
        axes = []
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            edge = (x2 - x1, y2 - y1)
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if length > 1e-9:
                axes.append((normal[0] / length, normal[1] / length))
        return axes

    def project(polygon, axis):
        dots = [p[0] * axis[0] + p[1] * axis[1] for p in polygon]
        return min(dots), max(dots)

    for axis in get_axes(poly_a) + get_axes(poly_b):
        min_a, max_a = project(poly_a, axis)
        min_b, max_b = project(poly_b, axis)
        # Tolerance matches existing AABB tolerance (0.01")
        if max_a <= min_b + 0.01 or max_b <= min_a + 0.01:
            return False  # Separating axis found — no overlap
    return True  # No separating axis — overlap


def panels_collide(p1_pos, p1_size, p1_info, p2_pos, p2_size, p2_info):
    """Check collision between two panels.
    Uses polygon collision for flat trapezoids, AABB for everything else.

    p1_info / p2_info: dicts with keys is_trapezoid, axis_map, short_edge,
    spec_length, flipped.
    """
    # Quick AABB pre-check
    if not aabb_intersect(p1_pos, p1_size, p2_pos, p2_size):
        return False

    # Determine if either panel is a flat trapezoid
    p1_flat_trap = (p1_info.get('is_trapezoid', False) and
                    p1_info.get('axis_map', ('', '', ''))[2] == 'T')
    p2_flat_trap = (p2_info.get('is_trapezoid', False) and
                    p2_info.get('axis_map', ('', '', ''))[2] == 'T')

    if p1_flat_trap or p2_flat_trap:
        # Z-layer check first
        z1, z2 = p1_pos[2], p2_pos[2]
        dz1, dz2 = p1_size[2], p2_size[2]
        if z1 + dz1 <= z2 + 0.01 or z2 + dz2 <= z1 + 0.01:
            return False  # Different layers

        # Get 2D footprints
        if p1_flat_trap:
            poly1 = get_trapezoid_footprint_2d(
                p1_pos, p1_size, p1_info['axis_map'],
                p1_info['short_edge'], p1_info['spec_length'],
                p1_info.get('flipped', False))
        else:
            x, y = p1_pos[0], p1_pos[1]
            dx, dy = p1_size[0], p1_size[1]
            poly1 = [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]

        if p2_flat_trap:
            poly2 = get_trapezoid_footprint_2d(
                p2_pos, p2_size, p2_info['axis_map'],
                p2_info['short_edge'], p2_info['spec_length'],
                p2_info.get('flipped', False))
        else:
            x, y = p2_pos[0], p2_pos[1]
            dx, dy = p2_size[0], p2_size[1]
            poly2 = [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]

        return polygons_overlap_2d(poly1, poly2)

    # Both non-trapezoid: AABB already confirmed collision
    return True


# ─── Stress Calculations (unchanged from V1.3) ──────────────────────────────

def calculate_support_area(pos, size, placed, gap=0.0):
    """Calculate overlap area between this panel and panels directly below.
    gap: spacer gap between stacked panels (inches). The Z tolerance is
    increased by the gap so that panels separated by a spacer are still
    recognized as supporting."""
    x, y, z = pos
    dx, dy, dz = size
    z_tolerance = Z_TOL + gap
    if z <= z_tolerance:
        return dx * dy, []
    total_area = 0.0
    supporting_panels = []
    for p in placed:
        px, py, pz = p["pos"]
        pdx, pdy, pdz = p["size"]
        top = pz + pdz
        if abs(top + gap - z) > Z_TOL:
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
    """
    Bending stress using beam theory (biaxial — checks X and Y spans).
    sigma_b = M * c / I
    Simply-supported (>=2 supports): M = w*L^2/8  (UDL)
    Cantilever (1 support):          M = w*L^2/2  (UDL)
    c = dz/2 (half-thickness), I = b*h^3/12
    Returns the worse (higher) stress of the two axes, in PSI.
    """
    if not supporting_panels:
        return 0.0  # on floor, no bending

    x, y, z = pos
    dx, dy, dz = size
    c = dz / 2.0  # distance from neutral axis to extreme fiber

    sx_positions = [s['centroid'][0] for s in supporting_panels]
    sy_positions = [s['centroid'][1] for s in supporting_panels]

    max_bending = 0.0

    # --- X-direction bending ---
    if len(sx_positions) >= 2:
        span_x = max(sx_positions) - min(sx_positions)
        if span_x > 1e-3:
            w_x = panel_weight / dx if dx > 1e-3 else 0.0
            M_x = w_x * span_x ** 2 / 8.0
            I_x = (dy * dz ** 3) / 12.0
            if I_x > 1e-6:
                max_bending = max(max_bending, M_x * c / I_x)
    elif len(sx_positions) == 1:
        overhang_x = max(abs(x - sx_positions[0]), abs(x + dx - sx_positions[0]))
        if overhang_x > 1e-3:
            w_x = panel_weight / dx if dx > 1e-3 else 0.0
            M_x = w_x * overhang_x ** 2 / 2.0
            I_x = (dy * dz ** 3) / 12.0
            if I_x > 1e-6:
                max_bending = max(max_bending, M_x * c / I_x)

    # --- Y-direction bending ---
    if len(sy_positions) >= 2:
        span_y = max(sy_positions) - min(sy_positions)
        if span_y > 1e-3:
            w_y = panel_weight / dy if dy > 1e-3 else 0.0
            M_y = w_y * span_y ** 2 / 8.0
            I_y = (dx * dz ** 3) / 12.0
            if I_y > 1e-6:
                max_bending = max(max_bending, M_y * c / I_y)
    elif len(sy_positions) == 1:
        overhang_y = max(abs(y - sy_positions[0]), abs(y + dy - sy_positions[0]))
        if overhang_y > 1e-3:
            w_y = panel_weight / dy if dy > 1e-3 else 0.0
            M_y = w_y * overhang_y ** 2 / 2.0
            I_y = (dx * dz ** 3) / 12.0
            if I_y > 1e-6:
                max_bending = max(max_bending, M_y * c / I_y)

    return max_bending


def calculate_shear_stress(total_weight, size):
    """
    Maximum shear stress for rectangular cross-section.
    tau_max = (3/2) * V / A   (parabolic distribution)
    V = total vertical shear force, A = min(dx,dy) * dz.
    """
    dx, dy, dz = size
    A_shear = min(dx, dy) * dz
    if A_shear < 0.01:
        return float('inf')
    return 1.5 * total_weight / A_shear


def calculate_deflection(panel_weight, pos, size, youngs_modulus, supporting_panels):
    """
    Maximum deflection using beam theory (biaxial).
    Simply-supported (>=2 supports): delta = 5*w*L^4 / (384*E*I)
    Cantilever (1 support):          delta = w*L^4 / (8*E*I)
    Young's modulus (E) is properly used here.
    Returns the worse (larger) deflection of the two axes, in inches.
    """
    if not supporting_panels:
        return 0.0

    x, y, z = pos
    dx, dy, dz = size

    sx_positions = [s['centroid'][0] for s in supporting_panels]
    sy_positions = [s['centroid'][1] for s in supporting_panels]

    max_defl = 0.0

    # --- X-direction deflection ---
    if len(sx_positions) >= 2:
        span_x = max(sx_positions) - min(sx_positions)
        if span_x > 1e-3:
            w_x = panel_weight / dx if dx > 1e-3 else 0.0
            I_x = (dy * dz ** 3) / 12.0
            if I_x > 1e-6 and youngs_modulus > 1e-3:
                max_defl = max(max_defl, (5.0 * w_x * span_x ** 4) / (384.0 * youngs_modulus * I_x))
    elif len(sx_positions) == 1:
        overhang_x = max(abs(x - sx_positions[0]), abs(x + dx - sx_positions[0]))
        if overhang_x > 1e-3:
            w_x = panel_weight / dx if dx > 1e-3 else 0.0
            I_x = (dy * dz ** 3) / 12.0
            if I_x > 1e-6 and youngs_modulus > 1e-3:
                max_defl = max(max_defl, (w_x * overhang_x ** 4) / (8.0 * youngs_modulus * I_x))

    # --- Y-direction deflection ---
    if len(sy_positions) >= 2:
        span_y = max(sy_positions) - min(sy_positions)
        if span_y > 1e-3:
            w_y = panel_weight / dy if dy > 1e-3 else 0.0
            I_y = (dx * dz ** 3) / 12.0
            if I_y > 1e-6 and youngs_modulus > 1e-3:
                max_defl = max(max_defl, (5.0 * w_y * span_y ** 4) / (384.0 * youngs_modulus * I_y))
    elif len(sy_positions) == 1:
        overhang_y = max(abs(y - sy_positions[0]), abs(y + dy - sy_positions[0]))
        if overhang_y > 1e-3:
            w_y = panel_weight / dy if dy > 1e-3 else 0.0
            I_y = (dx * dz ** 3) / 12.0
            if I_y > 1e-6 and youngs_modulus > 1e-3:
                max_defl = max(max_defl, (w_y * overhang_y ** 4) / (8.0 * youngs_modulus * I_y))

    return max_defl


def get_weight_above(panel_id, placed, gap=0.0):
    total = 0.0
    p = placed[panel_id]
    px, py, pz = p["pos"]
    pdx, pdy, pdz = p["size"]
    top = pz + pdz
    for other in placed[panel_id + 1:]:
        ox, oy, oz = other["pos"]
        odx, ody, odz = other["size"]
        if oz < top - Z_TOL - gap:
            continue
        xo = max(0.0, min(px + pdx, ox + odx) - max(px, ox))
        yo = max(0.0, min(py + pdy, oy + ody) - max(py, oy))
        if xo > 0.01 and yo > 0.01:
            total += other["weight"]
    return total


def stress_ok(pos, size, weight, placed, stress_config, min_support_frac=0.5, gap=0.0):
    x, y, z = pos
    dx, dy, dz = size
    panel_area = dx * dy
    support_area, supporting_panels = calculate_support_area(pos, size, placed, gap=gap)
    if support_area < panel_area * min_support_frac:
        return False, "Insufficient support area"
    comp = calculate_compression_stress(weight, support_area)
    if comp * stress_config.safety_factor > stress_config.max_compression_psi:
        return False, f"Compression too high: {comp:.1f} psi"
    bend = calculate_bending_stress(weight, pos, size, supporting_panels)
    if bend * stress_config.safety_factor > stress_config.max_bending_stress_psi:
        return False, f"Bending too high: {bend:.1f} psi"
    wa = sum(
        p["weight"] for p in placed
        if p["pos"][2] >= z + dz - Z_TOL - gap
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

def compute_max_stack_height(panel_weight, panel_size, trailer_H, stress_config, gap=0.0):
    """Pre-compute how many panels can stack before bottom exceeds shear.
    Uses tau = 1.5*V/A  =>  V_max = tau_allow * A / 1.5
    gap: spacer thickness between stacked panels (inches)."""
    dz = panel_size[2]
    shear_area = min(panel_size[0], panel_size[1]) * dz
    if shear_area < 0.01:
        return 1, dz
    max_shear = stress_config.max_shear_psi / stress_config.safety_factor
    max_column_weight = max_shear * shear_area / 1.5
    max_panels_shear = int(max_column_weight / panel_weight) if panel_weight > 0 else 999
    # Height limit: n panels + (n-1) gaps must fit in trailer_H
    # n * dz + (n-1) * gap <= trailer_H  =>  n <= (trailer_H + gap) / (dz + gap)
    stride_z = dz + gap
    max_panels_height = int((trailer_H + gap) / stride_z) if stride_z > 0 else 1
    max_panels = max(1, min(max_panels_shear, max_panels_height))
    return max_panels, max_panels * dz + max(0, max_panels - 1) * gap


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


def compute_axle_loads(placed, axle_config):
    """
    Compute load on each axle group using moment balance (lever arm physics).

    The trailer is modeled as a beam supported at two points:
      - Drive tandem (on tractor, near kingpin)
      - Trailer tandem (rear of trailer)

    For each panel, its weight creates a moment about each support point.
    Sum of moments = 0 gives the reaction forces at each support.

    The tractor's own weight is distributed between steer and drive axles
    based on the tractor wheelbase.

    Returns dict with per-axle loads, limits, and utilization.
    """
    ac = axle_config
    d_x = ac.drive_tandem_x      # drive tandem position (from trailer front)
    t_x = ac.trailer_tandem_x    # trailer tandem position
    span = t_x - d_x             # distance between support points

    if span < 1.0:
        return None  # invalid config

    # ── Cargo loads ──
    # For each panel, compute reaction forces at drive and trailer tandems
    cargo_on_drive = 0.0
    cargo_on_trailer = 0.0
    total_cargo = 0.0

    for p in placed:
        w = p["weight"]
        cx = p["pos"][0] + p["size"][0] / 2  # panel center X
        total_cargo += w

        # Moment about drive tandem: w * (cx - d_x) = R_trailer * span
        # R_trailer = w * (cx - d_x) / span
        # R_drive = w - R_trailer
        r_trailer = w * (cx - d_x) / span
        r_drive = w - r_trailer

        cargo_on_drive += r_drive
        cargo_on_trailer += r_trailer

    # ── Tractor weight ──
    # Tractor weight splits between steer and drive based on wheelbase
    # Steer axle is tractor_wheelbase ahead of drive tandem
    # Typical split: ~40% steer, ~60% drive for empty tractor
    tractor_w = ac.tractor_weight_lb
    steer_frac = 0.40
    tractor_on_steer = tractor_w * steer_frac
    tractor_on_drive = tractor_w * (1.0 - steer_frac)

    # ── Cargo also shifts some weight off drive onto steer (negative moment) ──
    # If cargo is behind the drive tandem, it lifts the steer axle slightly
    # The fifth wheel transfers: drive reaction includes tractor + cargo contribution
    # Steer axle load = tractor steer portion - (cargo moment effect on steer)
    # For simplicity and standard DOT calculations:
    # Steer = tractor_on_steer (cargo doesn't add to steer for trailer loads)
    steer_load = tractor_on_steer

    # Drive tandem total = tractor drive portion + cargo drive reaction
    drive_load = tractor_on_drive + cargo_on_drive

    # Trailer tandem total = cargo trailer reaction (no tractor weight here)
    trailer_load = cargo_on_trailer

    # Gross = steer + drive + trailer
    gross = steer_load + drive_load + trailer_load

    # Compute individual axle loads (tandem = split evenly between 2 axles)
    drive_per_axle = drive_load / 2.0
    trailer_per_axle = trailer_load / 2.0

    return {
        "steer_axle_lb": round(steer_load, 1),
        "drive_tandem_lb": round(drive_load, 1),
        "drive_per_axle_lb": round(drive_per_axle, 1),
        "trailer_tandem_lb": round(trailer_load, 1),
        "trailer_per_axle_lb": round(trailer_per_axle, 1),
        "gross_vehicle_weight_lb": round(gross, 1),
        "cargo_weight_lb": round(total_cargo, 1),
        "tractor_weight_lb": round(tractor_w, 1),
        "limits": {
            "steer_limit_lb": ac.steer_limit_lb,
            "drive_tandem_limit_lb": ac.drive_tandem_limit_lb,
            "trailer_tandem_limit_lb": ac.trailer_tandem_limit_lb,
            "gross_limit_lb": ac.gross_limit_lb,
        },
        "utilization_pct": {
            "steer": round(steer_load / ac.steer_limit_lb * 100, 1) if ac.steer_limit_lb > 0 else 0,
            "drive_tandem": round(drive_load / ac.drive_tandem_limit_lb * 100, 1) if ac.drive_tandem_limit_lb > 0 else 0,
            "trailer_tandem": round(trailer_load / ac.trailer_tandem_limit_lb * 100, 1) if ac.trailer_tandem_limit_lb > 0 else 0,
            "gross": round(gross / ac.gross_limit_lb * 100, 1) if ac.gross_limit_lb > 0 else 0,
        },
        "axle_positions": {
            "steer_axle_x": ac.drive_tandem_x - ac.tractor_wheelbase,
            "drive_tandem_x": ac.drive_tandem_x,
            "drive_tandem_spacing": ac.drive_tandem_spacing,
            "trailer_tandem_x": ac.trailer_tandem_x,
            "trailer_tandem_spacing": ac.trailer_tandem_spacing,
            "tractor_wheelbase": ac.tractor_wheelbase,
            },
    }




# ─── Axle Objective + Shift Optimizer ─────────────────────────────────────────

def axle_objective(axle_loads):
    """
    Single-number score for comparing axle-load "quality".
    Lower is better.

    Current objective:
      - Primary: minimize worst utilization among drive tandem, trailer tandem, and gross
      - Secondary: minimize drive-vs-trailer utilization imbalance
    """
    if not axle_loads:
        return float("inf")
    util = axle_loads.get("utilization_pct", {})
    max_u = max(
        util.get("drive_tandem", 0.0) / 100.0,
        util.get("trailer_tandem", 0.0) / 100.0,
        util.get("gross", 0.0) / 100.0,
    )
    imbalance = abs(util.get("drive_tandem", 0.0) - util.get("trailer_tandem", 0.0)) / 100.0
    return max_u + 0.25 * imbalance


def optimize_axle_shift(placed, axle_config, trailer_L, step_in=1.0):
    """
    If there is unused longitudinal slack, shift the entire load block forward/backward
    (within bounds) and pick the shift that minimizes axle_objective().

    Returns (best_shift_in, best_axle_loads_dict).
    """
    if not placed or axle_config is None:
        return 0.0, compute_axle_loads(placed, axle_config) if axle_config else None

    min_x = min(p["pos"][0] for p in placed)
    max_x = max(p["pos"][0] + p["size"][0] for p in placed)
    low = -min_x
    high = trailer_L - max_x
    if high < low:
        axle_now = compute_axle_loads(placed, axle_config)
        return 0.0, axle_now

    step = max(0.25, float(step_in))
    n = int((high - low) / step) + 1
    if n > 601:  # cap work
        step = (high - low) / 600.0 if (high - low) > 0 else 1.0
        n = 601

    best_shift = 0.0
    best_axle = None
    best_obj = float("inf")

    for i in range(n):
        s = low + i * step
        shifted = []
        for p in placed:
            pp = dict(p)
            x, y, z = pp["pos"]
            pp["pos"] = (x + s, y, z)
            shifted.append(pp)

        axle = compute_axle_loads(shifted, axle_config)
        obj = axle_objective(axle)
        if obj < best_obj:
            best_obj = obj
            best_shift = s
            best_axle = axle

    return round(best_shift, 3), best_axle
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

def generate_floor_slots(dx, dy, trailer_L, trailer_W, gap=0.0,
                         align_x="front", align_y="front", overhang_max=0.0):
    """
    Generate tight-packed, non-overlapping positions on the trailer floor.

    - gap: extra spacing between adjacent panels (in). Set 0.0 for no spacer.
    - align_x / align_y: where to place the packed "block" inside any leftover slack
      ("front"/"center"/"rear").
    - overhang_max: max per-side Y overhang (flatbed only). Effective width = trailer_W + 2*overhang_max.
    """
    try:
        gap = max(0.0, float(gap))
    except Exception:
        gap = 0.0

    if dx <= 0 or dy <= 0:
        return []

    eff_W = trailer_W + 2.0 * max(0.0, overhang_max)

    # If a single panel can't fit, no slots
    if trailer_L + 0.01 < dx or eff_W + 0.01 < dy:
        return []

    stride_x = dx + gap
    stride_y = dy + gap

    # Max counts along each axis (guard against FP noise)
    nx = int(math.floor((trailer_L - dx) / stride_x + 1.0 + 1e-9))
    ny = int(math.floor((eff_W - dy) / stride_y + 1.0 + 1e-9))
    nx = max(nx, 0)
    ny = max(ny, 0)

    used_L = nx * dx + max(0, nx - 1) * gap
    used_W = ny * dy + max(0, ny - 1) * gap
    slack_L = max(0.0, trailer_L - used_L)
    slack_W = max(0.0, eff_W - used_W)

    def start_from_align(align, slack):
        a = (align or "front").lower()
        if a.startswith("c"):
            return slack / 2.0
        if a.startswith("r") or a.startswith("b"):
            return slack
        return 0.0  # front/left

    x0 = start_from_align(align_x, slack_L)
    # Y-start: center within effective width, offset by -overhang_max
    # so positions are relative to the actual trailer deck (Y=0 = left edge of deck)
    y0_eff = start_from_align(align_y, slack_W)
    y0 = y0_eff - max(0.0, overhang_max)  # shift into trailer coords (may be negative)

    slots = []
    for i in range(nx):
        x = x0 + i * stride_x
        for j in range(ny):
            y = y0 + j * stride_y
            if x + dx <= trailer_L + 0.01:
                slots.append((round(x, 4), round(y, 4)))
    return slots


def generate_floor_slots_tessellated(dx, dy, trailer_L, trailer_W,
                                      short_edge, long_edge,
                                      gap=0.0, overhang_max=0.0,
                                      align_x="front"):
    """Generate tessellated floor slots for half-hex panels.

    Alternates normal and flipped trapezoids along X so sloped edges interlock.

    Normal:   wide (long_edge) at Y=y_base, narrow (short_edge) at Y=y_base+dy
    Flipped:  narrow at Y=y_base, wide at Y=y_base+dy

    Tessellated stride = long_edge - offset + gap  (e.g., 224 - 56 + 1.5 = 169.5")
    vs non-tessellated stride = long_edge + gap     (e.g., 224 + 1.5 = 225.5")
    Savings: offset per panel (56" for standard dims).

    Returns: list of (x, y, flipped) tuples.
    """
    if dx <= 0 or dy <= 0:
        return []

    offset = (long_edge - short_edge) / 2.0
    eff_W = trailer_W + 2.0 * max(0.0, overhang_max)

    if eff_W + 0.01 < dy:
        return []  # Can't fit even with overhang

    # Y position: center the panel within effective width, offset into trailer coords
    y_start = (trailer_W - dy) / 2.0  # may be negative if dy > trailer_W

    # Tessellated stride: distance between adjacent panel X-starts
    # Math: right edge of normal panel's slope meets left edge of flipped panel's slope
    # The -56t terms in the slope equations cancel, giving stride = long_edge - offset + gap
    tessellated_stride = long_edge - offset + gap

    if tessellated_stride <= 0:
        return []

    # Count how many panels fit
    # Panel i starts at x_start + i * tessellated_stride, ends at that + long_edge
    # Last panel: x_start + (n-1)*tessellated_stride + long_edge <= trailer_L
    # n <= (trailer_L - long_edge) / tessellated_stride + 1
    max_count = int(math.floor((trailer_L - long_edge) / tessellated_stride + 1.0 + 1e-9))
    max_count = max(max_count, 0)

    if max_count == 0:
        if long_edge <= trailer_L + 0.01:
            max_count = 1
        else:
            return []

    # Alignment
    used_L = (max_count - 1) * tessellated_stride + long_edge if max_count > 0 else 0
    slack_L = max(0.0, trailer_L - used_L)

    a = (align_x or "front").lower()
    if a.startswith("c"):
        x_offset = slack_L / 2.0
    elif a.startswith("r") or a.startswith("b"):
        x_offset = slack_L
    else:
        x_offset = 0.0

    slots = []
    for i in range(max_count):
        x = x_offset + i * tessellated_stride
        flipped = (i % 2 == 1)
        if x + long_edge <= trailer_L + 0.01:
            slots.append((round(x, 4), round(y_start, 4), flipped))

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
                    strategy, stress_config, axle_config=None,
                    panel_gap_in=0.0, trailer_tol_in=0.0,
                    align_x="front", align_y="front",
                    axle_optimize_shift=False, axle_shift_step=1.0,
                    is_flatbed=False, overhang_max_in=12.0):
    """
    V1.4 packing engine: layer-first, stress-aware, strategy-differentiated.
    V3.0: Added flatbed overhang, polygon collision, tessellated floor slots.

    Key improvements over V1.3:
    1. Pre-computes tight-packed floor slots (no grid search)
    2. Fills ALL floor slots before stacking (layer-first)
    3. Column tracker enforces max stack height
    4. Retroactive shear check on bottom panels
    5. Each strategy produces genuinely different slot ordering
    6. Tracks CG and weight distribution
    """
    # --- Fit / tolerance handling ---
    tol = max(0.0, float(trailer_tol_in))
    L_eff = trailer_L - tol
    W_eff = trailer_W - tol
    H_eff = trailer_H - tol
    if L_eff <= 0 or W_eff <= 0 or H_eff <= 0:
        return {"error": f"Trailer tolerance {tol:.2f}\" is too large for the selected trailer dimensions."}
    base_offset = tol / 2.0  # balances clearance on both sides
    trailer = (L_eff, W_eff, H_eff)

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
            # Determine if this panel qualifies for flatbed overhang
            panel_is_floor_flat = (spec.is_trapezoid and axis_map[2] == 'T')
            if not in_bounds_with_overhang((0, 0, 0), size, trailer,
                                           is_flatbed, overhang_max_in, panel_is_floor_flat):
                continue
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            dx, dy, dz = size

            # Use tessellated slot generator for flat floor panels on flatbed
            if spec.is_trapezoid and name.startswith("flat_") and is_flatbed:
                tess_slots = generate_floor_slots_tessellated(
                    dx, dy, trailer[0], trailer[1],
                    spec.short_edge, spec.length,
                    gap=panel_gap_in,
                    overhang_max=overhang_max_in,
                    align_x=align_x
                )
                floor_slots = [(s[0], s[1]) for s in tess_slots]
                flip_map = {(round(s[0], 4), round(s[1], 4)): s[2] for s in tess_slots}
            else:
                ovh = overhang_max_in if (is_flatbed and panel_is_floor_flat) else 0.0
                floor_slots = generate_floor_slots(dx, dy, trailer[0], trailer[1],
                                                    gap=panel_gap_in, align_x=align_x,
                                                    align_y=align_y, overhang_max=ovh)
                flip_map = {}

            max_stack, max_z = compute_max_stack_height(spec.weight, size, trailer_H, stress_config, gap=panel_gap_in)

            valid.append({
                'name': name,
                'size': size,
                'axis_map': axis_map,
                'floor_slots': floor_slots,
                'flip_map': flip_map,
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
            flip_map = orient.get('flip_map', {})
            max_stack = orient['max_stack']

            # Determine if this orientation qualifies for overhang
            panel_is_floor_flat = (spec.is_trapezoid and axis_map[2] == 'T')

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
                current_flipped = flip_map.get((round(sx, 4), round(sy, 4)), False)
                state = column_tracker.get(col_key)

                if state is None:
                    z = 0.0
                else:
                    if state['count'] >= max_stack:
                        continue  # column full
                    z = state['z_top']  # already includes spacer gap from previous placement

                test_pos = (sx, sy, z)

                # Bounds check — overhang-aware for flat floor panels on flatbed
                if not in_bounds_with_overhang(test_pos, size, trailer,
                                               is_flatbed, overhang_max_in, panel_is_floor_flat):
                    continue

                # Collision check — polygon-aware for trapezoid panels
                panel_info_new = {
                    'is_trapezoid': spec.is_trapezoid,
                    'axis_map': axis_map,
                    'short_edge': spec.short_edge,
                    'spec_length': spec.length,
                    'flipped': current_flipped,
                }
                if any(panels_collide(test_pos, size, panel_info_new,
                                       p["pos"], p["size"], p) for p in placed):
                    continue

                # Standard stress check
                stress_valid, msg = stress_ok(test_pos, size, spec.weight, placed, stress_config, gap=panel_gap_in)
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
                    "flipped": current_flipped,
                })

                if state is None:
                    column_tracker[col_key] = {
                        'count': 1, 'z_top': round(z + dz + panel_gap_in, 4),
                        'total_weight': spec.weight,
                    }
                else:
                    state['count'] += 1
                    state['z_top'] = round(z + dz + panel_gap_in, 4)
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

    # ── Step 3.5: Apply trailer tolerance offset + optional axle-load shift ──
    # Shift everything by base_offset so the "reserved tolerance" is split on both sides.
    if abs(base_offset) > 1e-9:
        for p in placed:
            x, y, z = p["pos"]
            p["pos"] = (x + base_offset, y + base_offset, z)

    axle_shift_in = 0.0
    axle_loads = compute_axle_loads(placed, axle_config) if axle_config else None

    # If enabled, use available slack to shift the entire load block to minimize axle utilization
    if axle_optimize_shift and axle_config and placed:
        axle_shift_in, axle_loads = optimize_axle_shift(
            placed, axle_config, trailer_L, step_in=axle_shift_step
        )
        if abs(axle_shift_in) > 1e-9:
            for p in placed:
                x, y, z = p["pos"]
                p["pos"] = (x + axle_shift_in, y, z)

    # ── Step 4: Post-placement stress analysis ──
    for i, panel in enumerate(placed):
        sa, sp = calculate_support_area(panel["pos"], panel["size"], placed[:i], gap=panel_gap_in)
        comp = calculate_compression_stress(panel["weight"], sa)
        bend = calculate_bending_stress(panel["weight"], panel["pos"], panel["size"], sp)
        wa = get_weight_above(i, placed, gap=panel_gap_in)
        shear = calculate_shear_stress(wa + panel["weight"], panel["size"])
        defl = calculate_deflection(panel["weight"], panel["pos"], panel["size"],
                                    stress_config.panel_youngs_modulus_psi, sp)

        dx, dy, dz = panel["size"]
        footprint = dx * dy
        total_load = panel["weight"] + wa

        # Utilization ratios (fraction of allowable, accounting for safety factor)
        comp_util = (comp * stress_config.safety_factor / stress_config.max_compression_psi
                     if stress_config.max_compression_psi > 0 else 0.0)
        bend_util = (bend * stress_config.safety_factor / stress_config.max_bending_stress_psi
                     if stress_config.max_bending_stress_psi > 0 else 0.0)
        shear_util = (shear * stress_config.safety_factor / stress_config.max_shear_psi
                      if stress_config.max_shear_psi > 0 else 0.0)
        max_defl_allow = min(dx, dy) / 360.0
        defl_util = (defl / max_defl_allow) if max_defl_allow > 1e-6 else 0.0

        utilization = max(comp_util, bend_util, shear_util, defl_util)
        limiting = ("compression" if comp_util == utilization else
                    "bending" if bend_util == utilization else
                    "shear" if shear_util == utilization else "deflection")

        panel["stress_analysis"] = {
            "compression_psi": round(comp, 2),
            "bending_stress_psi": round(bend, 2),
            "shear_psi": round(shear, 2),
            "deflection_in": round(defl, 4),
            "weight_above_lb": round(wa, 2),
            "total_load_lb": round(total_load, 2),
            "support_area_sqin": round(sa, 2),
            "footprint_sqin": round(footprint, 2),
            "utilization_ratio": round(utilization, 4),
            "limiting_factor": limiting,
        }

    # ── Step 5: Summary metrics ──
    wall_count = sum(1 for p in placed if p["panel_type"] == PanelType.WALL.value)
    floor_count = sum(1 for p in placed if p["panel_type"] == PanelType.FLOOR.value)
    cg = compute_cg(placed)
    weight_dist = compute_weight_distribution(placed, trailer_L)

    max_layer = 0
    for state in column_tracker.values():
        max_layer = max(max_layer, state['count'])

    # ── Step 6: Identify worst-case stress panel (by utilization ratio) ──
    worst_panel = None
    worst_util = 0.0
    for i, panel in enumerate(placed):
        util = panel["stress_analysis"]["utilization_ratio"]
        if util > worst_util:
            worst_util = util
            sa_data = panel["stress_analysis"]
            worst_panel = {
                "index": i,
                "label": panel["label"],
                "panel_type": panel["panel_type"],
                "position": list(panel["pos"]),
                "size": list(panel["size"]),
                "orientation": panel["orientation"],
                "compression_psi": sa_data["compression_psi"],
                "bending_stress_psi": sa_data["bending_stress_psi"],
                "shear_psi": sa_data["shear_psi"],
                "deflection_in": sa_data["deflection_in"],
                "total_load_lb": sa_data["total_load_lb"],
                "weight_above_lb": sa_data["weight_above_lb"],
                "own_weight_lb": panel["weight"],
                "footprint_sqin": sa_data["footprint_sqin"],
                "utilization_ratio": sa_data["utilization_ratio"],
                "limiting_factor": sa_data["limiting_factor"],
                "compression_limit_psi": stress_config.max_compression_psi,
                "bending_limit_psi": stress_config.max_bending_stress_psi,
                "shear_limit_psi": stress_config.max_shear_psi,
                "safety_factor": stress_config.safety_factor,
            }

    return {
        "inputs": {
            "trailer": {"L": trailer_L, "W": trailer_W, "H": trailer_H},
            "stress_limits": {
                "max_compression_psi": stress_config.max_compression_psi,
                "max_bending_stress_psi": stress_config.max_bending_stress_psi,
                "max_shear_psi": stress_config.max_shear_psi,
                "safety_factor": stress_config.safety_factor
            }
        },
        "settings": {
            "requested_panels": len(panel_list),
            "placed_panels": len(placed),
            "placed_walls": wall_count,
            "placed_floors": floor_count,            "rejected_panels": len(rejected),
            "min_support_frac": 0.5,
            "panel_gap_in": float(panel_gap_in),
            "trailer_tol_in": float(trailer_tol_in),
            "align_x": str(align_x),
            "align_y": str(align_y),
            "axle_shift_in": float(axle_shift_in),
            "axle_objective": round(axle_objective(axle_loads), 6) if axle_loads else None,
            "packing_strategy": strategy.value,
            "max_layers_used": max_layer,
            "is_flatbed": is_flatbed,
            "overhang_max_in": float(overhang_max_in),
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
                "flipped": p.get("flipped", False),
                "layer": int(round(p["pos"][2] / max(p["size"][2], 0.1))),
                "loading_order": i,
                "unloading_order": len(placed) - i - 1,
                "stress_analysis": p["stress_analysis"]
            }
            for i, p in enumerate(placed)
        ],
        "rejections": rejected,
        "worst_stress_panel": worst_panel,
        "axle_loads": axle_loads,
    }


# ─── 3D Visualization ───────────────────────────────────────────────────────



def optimize_best_configuration(panel_list, trailer_L, trailer_W, trailer_H,
                                max_horizontal, max_vertical,
                                stress_config, axle_config,
                                panel_gap_in=0.0, trailer_tol_in=0.0,
                                align_x="front", align_y="front",
                                axle_optimize_shift=False, axle_shift_step=1.0,
                                is_flatbed=False, overhang_max_in=12.0):
    """
    Runs all strategies and returns the best output by:
      1) maximize placed panels
      2) minimize axle_objective (worst utilization + imbalance)
    """
    best_out = None
    best_key = None

    for s in LoadingStrategy:
        out = pack_panels_v14(
            panel_list, trailer_L, trailer_W, trailer_H,
            max_horizontal, max_vertical,
            strategy=s, stress_config=stress_config, axle_config=axle_config,
            panel_gap_in=panel_gap_in, trailer_tol_in=trailer_tol_in,
            align_x=align_x, align_y=align_y,
            axle_optimize_shift=axle_optimize_shift, axle_shift_step=axle_shift_step,
            is_flatbed=is_flatbed, overhang_max_in=overhang_max_in
        )
        if "error" in out:
            continue

        placed = out.get("settings", {}).get("placed_panels", 0)
        axle_obj = out.get("settings", {}).get("axle_objective", float("inf"))
        key = (-placed, axle_obj)

        # Ensure we never compare a tuple with None; prefer new key if no best_key yet
        if best_key is None or key < best_key:
            best_out = out
            best_key = key
            best_out.setdefault("settings", {})["selected_strategy"] = s.value

    if best_out is None:
        return {"error": "No feasible configuration found under any strategy."}

    best_out["settings"]["auto_selected"] = True
    return best_out
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


def make_half_hex_vertices(x, y, z, dx, dy, dz, axis_map, short_edge, long_edge, flipped=False):
    offset = (long_edge - short_edge) / 2.0
    x_axis, y_axis, z_axis = axis_map
    if x_axis == "L" and y_axis == "H":
        if not flipped:
            # Normal: wide edge at Y=y, narrow at Y=y+dy
            b0, b1 = (x, y, z), (x + dx, y, z)
            b2, b3 = (x + offset + short_edge, y + dy, z), (x + offset, y + dy, z)
        else:
            # Flipped: narrow at Y=y, wide at Y=y+dy
            b0, b1 = (x + offset, y, z), (x + offset + short_edge, y, z)
            b2, b3 = (x + dx, y + dy, z), (x, y + dy, z)
    elif x_axis == "H" and y_axis == "L":
        if not flipped:
            b0, b1 = (x, y, z), (x, y + dy, z)
            b2, b3 = (x + dx, y + offset + short_edge, z), (x + dx, y + offset, z)
        else:
            b0, b1 = (x, y + offset, z), (x, y + offset + short_edge, z)
            b2, b3 = (x + dx, y + dy, z), (x + dx, y, z)
    elif x_axis == "L" and z_axis == "H":
        # Standing orientations: no flip (flipped only applies to flat)
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


def add_cylinder_y(fig, x_center, y0, y1, z_center, radius, color, opacity=0.65, segments=28,
                   name="", showlegend=False):
    """Add a solid cylinder whose axis runs along +Y (used for axle visualization)."""
    xs, ys, zs = [], [], []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cx = x_center + radius * math.cos(theta)
        cz = z_center + radius * math.sin(theta)
        xs.append(cx); ys.append(y0); zs.append(cz)
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        cx = x_center + radius * math.cos(theta)
        cz = z_center + radius * math.sin(theta)
        xs.append(cx); ys.append(y1); zs.append(cz)

    I, J, K = [], [], []
    for i in range(segments):
        j = (i + 1) % segments
        I += [i, j]
        J += [j, j + segments]
        K += [i + segments, i + segments]

    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=I, j=J, k=K,
        color=color, opacity=opacity,
        name=name, showlegend=showlegend
    ))

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


def visualize(out, stress_config, color_by_stress, show_stress_markers, show_axles, floor_spec):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]
    fig = go.Figure()

    # Trailer — flatbed shows open sides (no enclosure walls)
    is_fb = out.get("settings", {}).get("is_flatbed", False)
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.2)
    if is_fb:
        # Flatbed: only deck floor + corner height-limit posts (no side walls)
        for cx in [0, L]:
            for cy in [0, W]:
                add_wire(fig, cx, cy, 0, 0, 0, H, "#ecf0f1", "", opacity=0.15, width=1)
    else:
        add_wire(fig, 0, 0, 0, L, W, H, "#ecf0f1", "", opacity=0.3, width=2)
    for i in range(0, int(L), 50):
        add_wire(fig, i, 0, 0, 0, W, 0, "#34495e", "", opacity=0.15, width=1)
    for j in range(0, int(W), 50):
        add_wire(fig, 0, j, 0, L, 0, 0, "#34495e", "", opacity=0.15, width=1)

    wall_pal = ["#3498db", "#2ecc71", "#9b59b6", "#1abc9c", "#95a5a6"]
    floor_pal = ["#e67e22", "#e74c3c", "#f1c40f", "#d35400"]
    wi, fi = 0, 0
    tc = {}

    # Identify worst panel index for highlighting
    wsp = out.get("worst_stress_panel")
    worst_idx = wsp["index"] if wsp else -1

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

        # Choose color
        if color_by_stress:
            c = get_stress_color(
                p["stress_analysis"]["utilization_ratio"],
                1.0  # utilization_ratio of 1.0 = at limit
            )
        elif show_stress_markers and p["loading_order"] == worst_idx:
            c = "#ffff00"  # bright yellow for worst-case panel
        else:
            c = tc[key]

        x, y, z = p["position_vector"]
        dx, dy, dz = p["size"]

        # Use thicker/brighter wireframe for worst-case panel
        wire_width = 6 if (show_stress_markers and p["loading_order"] == worst_idx) else 2
        wire_opacity = 1.0 if (show_stress_markers and p["loading_order"] == worst_idx) else 0.9

        if p.get("is_trapezoid", False):
            am = tuple(p.get("axis_map", ["L", "H", "T"]))
            fl = p.get("flipped", False)
            verts = make_half_hex_vertices(x, y, z, dx, dy, dz, am,
                                            floor_spec.short_edge, floor_spec.length, flipped=fl)
            add_half_hex_solid(fig, verts, c, opacity=0.45)
            add_half_hex_wire(fig, verts, c, "", opacity=wire_opacity, width=wire_width)
        else:
            add_solid(fig, x, y, z, dx, dy, dz, c, opacity=0.4)
            add_wire(fig, x, y, z, dx, dy, dz, c, "", opacity=wire_opacity, width=wire_width)

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

    # Worst-stress marker
    if show_stress_markers and wsp:
        wx = wsp["position"][0] + wsp["size"][0] / 2
        wy = wsp["position"][1] + wsp["size"][1] / 2
        wz = wsp["position"][2] + wsp["size"][2] / 2
        fig.add_trace(go.Scatter3d(
            x=[wx], y=[wy], z=[wz],
            mode="markers+text",
            marker=dict(size=10, color="#ffff00", symbol="x", line=dict(width=2, color="black")),
            text=[f"MAX STRESS\n{wsp['label']}"],
            textposition="top center",
            textfont=dict(color="#ffff00", size=11),
            name=f"Max Stress: {wsp['label']}"
        ))

    # Axle visualization (3D) — axles + tires/wheels
    axle_data = out.get("axle_loads")
    if show_axles and axle_data:
        ap = axle_data["axle_positions"]

        # Visual geometry (inches, approximate)
        tire_radius = 12.0          # outer tire radius
        tire_width = 8.0            # tire thickness along Y
        rim_radius = 7.0            # rim radius
        rim_width = 5.0             # rim thickness along Y
        wheel_outset = 6.0          # how far outside trailer width to draw tires
        axle_bar_radius = 1.6       # axle bar thickness

        # Keep the axle bar passing through the wheel centers, but place axles/wheels UNDER the trailer


        # Trailer floor in this viz is at z=0; use a small drop so tire top sits just below the floor.


        trailer_bottom_z = 0.0


        suspension_drop = 4.0   # inches of clearance between trailer bottom and tire top


        wheel_center_z = trailer_bottom_z - (tire_radius + suspension_drop)


        axle_z = wheel_center_z
        def add_tire_pair(ax_x, color_axle, label_color, axle_name="", show_leg=False,
                          load_text=None, load_x=None, dual=False):
            """Draw one axle bar (across trailer width) plus tires and rims at each side."""
            # Axle bar (spans trailer width)
            add_cylinder_y(fig, ax_x, 0, W, axle_z, axle_bar_radius, color_axle,
                           opacity=0.80, name=axle_name, showlegend=show_leg)

            # Tire placement helper
            def draw_one_side(y_start, y_end):
                # Tire (black)
                add_cylinder_y(fig, ax_x, y_start, y_end, axle_z, tire_radius, "#111111",
                               opacity=0.96, name="", showlegend=False)
                # Rim (lighter gray, slightly inset)
                rim_y0 = y_start + (tire_width - rim_width) / 2.0
                rim_y1 = rim_y0 + rim_width
                add_cylinder_y(fig, ax_x, rim_y0, rim_y1, axle_z, rim_radius, "#bdc3c7",
                               opacity=0.95, name="", showlegend=False)

            if dual:
                # Dual tires: two tires per side (very simple spacing)
                gap = 1.0
                # Left side (outside trailer at negative Y)
                draw_one_side(-wheel_outset - tire_width, -wheel_outset)
                draw_one_side(-wheel_outset - 2 * tire_width - gap, -wheel_outset - tire_width - gap)
                # Right side (outside trailer at Y > W)
                draw_one_side(W + wheel_outset, W + wheel_outset + tire_width)
                draw_one_side(W + wheel_outset + tire_width + gap, W + wheel_outset + 2 * tire_width + gap)
            else:
                # Single tires per side
                draw_one_side(-wheel_outset - tire_width, -wheel_outset)
                draw_one_side(W + wheel_outset, W + wheel_outset + tire_width)

            # Load label (optional)
            if load_text is not None and load_x is not None:
                fig.add_trace(go.Scatter3d(
                    x=[load_x], y=[W / 2], z=[axle_z + tire_radius + 2.0],
                    mode="markers+text",
                    marker=dict(size=6, color=label_color, symbol="circle"),
                    text=[load_text],
                    textposition="top center",
                    textfont=dict(color=label_color, size=10),
                    name="", showlegend=False
                ))

        # Steer axle (green) — single tires
        steer_x = ap.get("steer_axle_x", None)
        if steer_x is not None:
            add_tire_pair(
                steer_x,
                color_axle="#2ecc71",
                label_color="#2ecc71",
                axle_name="Steer Axle",
                show_leg=True,
                load_text=f"Steer: {axle_data['steer_axle_lb']:,.0f} lb",
                load_x=steer_x,
                dual=False
            )

        # Drive tandem (red) — dual tires
        d_center = ap["drive_tandem_x"]
        d_half = ap["drive_tandem_spacing"] / 2.0
        for ax_x, show_leg in [
            (d_center - d_half, True),
            (d_center + d_half, False),
        ]:
            add_tire_pair(
                ax_x,
                color_axle="#e74c3c",
                label_color="#e74c3c",
                axle_name="Drive Tandem" if show_leg else "",
                show_leg=show_leg,
                load_text=None,
                load_x=None,
                dual=True
            )

        # Tandem load label at the center
        fig.add_trace(go.Scatter3d(
            x=[d_center], y=[W / 2], z=[axle_z + tire_radius + 2.0],
            mode="markers+text",
            marker=dict(size=6, color="#e74c3c", symbol="circle"),
            text=[f"Drive: {axle_data['drive_tandem_lb']:,.0f} lb"],
            textposition="top center",
            textfont=dict(color="#e74c3c", size=10),
            name="", showlegend=False
        ))

        # Trailer tandem (blue) — dual tires
        t_center = ap["trailer_tandem_x"]
        t_half = ap["trailer_tandem_spacing"] / 2.0
        for ax_x, show_leg in [
            (t_center - t_half, True),
            (t_center + t_half, False),
        ]:
            add_tire_pair(
                ax_x,
                color_axle="#3498db",
                label_color="#3498db",
                axle_name="Trailer Tandem" if show_leg else "",
                show_leg=show_leg,
                load_text=None,
                load_x=None,
                dual=True
            )

        # Tandem load label at the center
        fig.add_trace(go.Scatter3d(
            x=[t_center], y=[W / 2], z=[axle_z + tire_radius + 2.0],
            mode="markers+text",
            marker=dict(size=6, color="#3498db", symbol="circle"),
            text=[f"Trailer: {axle_data['trailer_tandem_lb']:,.0f} lb"],
            textposition="top center",
            textfont=dict(color="#3498db", size=10),
            name="", showlegend=False
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
        paper_bgcolor="#161208",
        scene=dict(
            xaxis_title="Length (in)", yaxis_title="Width (in)", zaxis_title="Height (in)",
            aspectmode="data", camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
            bgcolor="#1A160E",
            xaxis=dict(backgroundcolor="#1A160E", gridcolor="#3E3420", color="#D4A832"),
            yaxis=dict(backgroundcolor="#1A160E", gridcolor="#3E3420", color="#D4A832"),
            zaxis=dict(backgroundcolor="#1A160E", gridcolor="#3E3420", color="#D4A832"),
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(22,18,8,0.9)", font=dict(color="#F5F0E8", size=11))
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
            if wsp:
                util_pct = wsp['utilization_ratio'] * 100
                st.metric("Worst Util.", f"{util_pct:.1f}%",
                          help=f"Worst-case panel utilization (limiting: {wsp['limiting_factor']})")
            else:
                st.metric("Worst Util.", "N/A")
        with c5:
            st.metric("Layers", out['settings']['max_layers_used'])
        with c6:
            cg = out["weight_distribution"]["center_of_gravity"]
            st.metric("CG (X)", f"{cg['x']:.0f}\" / {tr['L']:.0f}\"")

        # Weight distribution
        wd = out["weight_distribution"]["by_thirds"]
        st.subheader("Weight Distribution (Trailer Thirds)")
        wd_c1, wd_c2, wd_c3 = st.columns(3)
        with wd_c1:
            st.metric("Front", f"{wd['front']:,.0f} lb")
        with wd_c2:
            st.metric("Middle", f"{wd['middle']:,.0f} lb")
        with wd_c3:
            st.metric("Rear", f"{wd['rear']:,.0f} lb")

        # Worst-case stress detail card
        if show_stress_markers and wsp:
            st.subheader("Worst-Case Stress Panel")
            ws1, ws2, ws3, ws4 = st.columns(4)
            with ws1:
                st.metric("Panel", wsp["label"])
                st.caption(f"{wsp['panel_type']} | {wsp['orientation']}")
            with ws2:
                st.metric("Total Load", f"{wsp['total_load_lb']:,.1f} lb")
                st.caption(f"Self: {wsp['own_weight_lb']:.0f} lb + Above: {wsp['weight_above_lb']:,.0f} lb")
            with ws3:
                util_pct = wsp["utilization_ratio"] * 100
                color_fn = st.success if util_pct < 50 else st.warning if util_pct < 80 else st.error
                st.metric("Utilization", f"{util_pct:.1f}%")
                color_fn(f"Limiting: {wsp['limiting_factor']}")
            with ws4:
                st.metric("Compression", f"{wsp['compression_psi']:.2f} psi")
                st.metric("Shear", f"{wsp['shear_psi']:.2f} psi")
            st.caption(
                f"Position: ({wsp['position'][0]:.0f}\", {wsp['position'][1]:.0f}\", {wsp['position'][2]:.0f}\") "
                f"| Bending: {wsp['bending_stress_psi']:.2f} psi | Defl: {wsp['deflection_in']:.4f}\""
            )
        # Axle load results card
        axle_data = out.get("axle_loads")
        if show_axles and axle_data:
            st.subheader("Axle Load Analysis")
            ax1, ax2, ax3, ax4 = st.columns(4)
            with ax1:
                st.metric("Steer Axle", f"{axle_data['steer_axle_lb']:,.0f} lb")
                u = axle_data["utilization_pct"]["steer"]
                lim = axle_data["limits"]["steer_limit_lb"]
                cfn = st.success if u < 50 else st.warning if u < 80 else st.error
                cfn(f"{u:.1f}% of {lim:,.0f} lb limit")
            with ax2:
                st.metric("Drive Tandem", f"{axle_data['drive_tandem_lb']:,.0f} lb")
                u = axle_data["utilization_pct"]["drive_tandem"]
                lim = axle_data["limits"]["drive_tandem_limit_lb"]
                cfn = st.success if u < 50 else st.warning if u < 80 else st.error
                cfn(f"{u:.1f}% of {lim:,.0f} lb limit")
                st.caption(f"Per axle: {axle_data['drive_per_axle_lb']:,.0f} lb")
            with ax3:
                st.metric("Trailer Tandem", f"{axle_data['trailer_tandem_lb']:,.0f} lb")
                u = axle_data["utilization_pct"]["trailer_tandem"]
                lim = axle_data["limits"]["trailer_tandem_limit_lb"]
                cfn = st.success if u < 50 else st.warning if u < 80 else st.error
                cfn(f"{u:.1f}% of {lim:,.0f} lb limit")
                st.caption(f"Per axle: {axle_data['trailer_per_axle_lb']:,.0f} lb")
            with ax4:
                st.metric("Gross Vehicle Wt", f"{axle_data['gross_vehicle_weight_lb']:,.0f} lb")
                u = axle_data["utilization_pct"]["gross"]
                lim = axle_data["limits"]["gross_limit_lb"]
                cfn = st.success if u < 50 else st.warning if u < 80 else st.error
                cfn(f"{u:.1f}% of {lim:,.0f} lb limit")
                st.caption(f"Cargo: {axle_data['cargo_weight_lb']:,.0f} lb + Tractor: {axle_data['tractor_weight_lb']:,.0f} lb")

    else:
        st.warning("No panels placed. Check rejections below.")


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="MCLOS - Modular Cargo Loading Optimization Software", layout="wide",)

# ─── Logo helper ──────────────────────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HexHomesLogo.png")

def get_logo_b64():
    """Return base64 encoded logo for CSS embedding. Returns None if file not found."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

_LOGO_B64 = get_logo_b64()

# ─── HexHomes Honey/Gold Theme ───────────────────────────────────────────
HONEY_CSS = """
<style>
/* ── Color palette ──
   Background:     #242012  (dark tan/brown, slightly darker)
   Sidebar:        #1C1810  (deeper dark brown)
   Primary gold:   #E8A817  (HexHomes logo yellow)
   Light gold:     #D4A832  (text accents)
   Pale gold:      #F0D68A  (secondary text)
   White:          #F5F0E8  (primary text, warm white)
   3D scene:       #1A160E  (darker than bg for visualizer)
   3D paper:       #161208  (darkest, chart wrapper)
*/

/* Main background — dark tan */
.stApp {
    background-color: #242012;
    color: #F5F0E8;
}

/* All general text — warm white */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
label, .stCaption, span, p, div {
    color: #F5F0E8 !important;
}

/* Sidebar — deeper dark brown */
section[data-testid="stSidebar"] {
    background-color: #1C1810 !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] span {
    color: #E8DABE !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #E8A817 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #3E3420 !important;
}

/* Primary buttons — honey gold */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #E8A817 !important;
    color: #1A1408 !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #D49A10 !important;
    color: #1A1408 !important;
}

/* Secondary buttons */
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
    border-color: #D4A832 !important;
    color: #F0D68A !important;
    background-color: transparent !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
    background-color: #302818 !important;
    border-color: #E8A817 !important;
}

/* Headers — light gold */
h1, h2, h3 {
    color: #E8A817 !important;
}

/* Subheaders slightly softer */
h2, h3 {
    color: #D4A832 !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #C0A870 !important;
}

/* Metric values — bright gold */
[data-testid="stMetricValue"] {
    color: #F0D68A !important;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #302818 !important;
    color: #F0D68A !important;
    border: 1px solid #D4A832 !important;
}
.stDownloadButton > button:hover {
    background-color: #3E3420 !important;
}

/* Expanders — themed background + border */
details[data-testid="stExpander"] {
    background-color: #1E1A12 !important;
    border: 1px solid #3E3420 !important;
    border-radius: 8px !important;
    padding: 2px 8px !important;
}
details[data-testid="stExpander"] summary {
    color: #D4A832 !important;
}
details[data-testid="stExpander"] summary:hover {
    color: #E8A817 !important;
}
/* Content inside expanders */
details[data-testid="stExpander"] > div {
    background-color: #1E1A12 !important;
}

/* JSON viewer inside expanders */
[data-testid="stJson"],
.stJson, pre {
    background-color: #1A160E !important;
    color: #F0D68A !important;
    border: 1px solid #3E3420 !important;
    border-radius: 6px !important;
}
/* JSON keys/values styling */
[data-testid="stJson"] span {
    color: #F0D68A !important;
}

/* Info boxes — dark warm tint */
div[data-testid="stAlert"] {
    background-color: #2A2418 !important;
    color: #F5F0E8 !important;
}

/* All input fields — selectbox, number, text, etc */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stMultiSelect > div > div,
.stDateInput > div > div > input,
.stTimeInput > div > div > input,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
input, select, textarea {
    background-color: #302818 !important;
    color: #F5F0E8 !important;
    border-color: #4A3D28 !important;
}
/* Selectbox dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
ul[data-baseweb="menu"] {
    background-color: #302818 !important;
}
[data-baseweb="menu"] li,
[role="option"] {
    background-color: #302818 !important;
    color: #F5F0E8 !important;
}
[data-baseweb="menu"] li:hover,
[role="option"]:hover {
    background-color: #3E3420 !important;
}
/* Focus states */
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus,
input:focus, select:focus, textarea:focus {
    border-color: #E8A817 !important;
    box-shadow: 0 0 0 1px #E8A817 !important;
}
/* Number input +/- buttons */
.stNumberInput button {
    background-color: #3E3420 !important;
    color: #F0D68A !important;
    border-color: #4A3D28 !important;
}
.stNumberInput button:hover {
    background-color: #4A3D28 !important;
}

/* Radio buttons */
.stRadio > div > label > div:first-child {
    color: #E8A817 !important;
}
.stRadio > div > label {
    color: #F5F0E8 !important;
}

/* Checkbox accent */
.stCheckbox > label > span > span {
    border-color: #D4A832 !important;
}

/* Plotly chart container — darker than page bg */
[data-testid="stPlotlyChart"] {
    border: 1px solid #3E3420;
    border-radius: 8px;
    padding: 4px;
    background-color: #161208 !important;
}
/* Plotly modebar (toolbar) */
.modebar {
    background-color: transparent !important;
}
.modebar-btn path {
    fill: #D4A832 !important;
}

/* Dataframe — themed */
.stDataFrame {
    background-color: #1E1A12 !important;
    border: 1px solid #3E3420 !important;
    border-radius: 6px !important;
}
.stDataFrame thead th {
    background-color: #302818 !important;
    color: #F0D68A !important;
}
.stDataFrame tbody td {
    background-color: #1E1A12 !important;
    color: #F5F0E8 !important;
}
.stDataFrame tbody tr:hover td {
    background-color: #2A2418 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #E8A817 !important;
}

/* Dividers */
hr {
    border-color: #3E3420 !important;
}

/* Success/warning/error — functional colors on dark bg */
div[data-testid="stAlert"] p {
    color: #F5F0E8 !important;
}

/* Caption text — muted gold */
.stCaption, [data-testid="stCaption"] {
    color: #C0A870 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #D4A832 !important;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: #E8A817 !important;
}

/* Tooltip / help icons */
[data-testid="stTooltipIcon"] {
    color: #D4A832 !important;
}

/* Scrollbar — warm tones */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1E1A12;
}
::-webkit-scrollbar-thumb {
    background: #4A3D28;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #5A4A30;
}
</style>
"""

st.markdown(HONEY_CSS, unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "started" not in st.session_state:
    st.session_state.started = False

# ─── Welcome page ──────────────────────────────────────────────────────────
def render_welcome():
    # Logo + title row
    if _LOGO_B64:
        logo_col, title_col = st.columns([1, 3])
        with logo_col:
            st.image(LOGO_PATH, width=180)
        with title_col:
            st.title("Modular Cargo Loading Optimization Software")
            st.caption("HexHomes Panel Packing Optimizer  |  Alpha V3.0")
    else:
        st.title("Modular Cargo Loading Optimization Software")
        st.caption("HexHomes Panel Packing Optimizer  |  Alpha V3.0")

    st.markdown("---")
    st.markdown("""
### Welcome

This app packs **HexHomes wall + floor panels** into common trailer presets (or custom dimensions), then reports:

- **3D load visualization** (panels, CG marker, and **3D axles**)
- **Structural checks** (support area, compression, bending, shear, deflection)
- **Loading strategies** (Gravity Layered, Wall First, Zone Based, Stress Optimized)
- **Axle-load analysis** (steer / drive tandem / trailer tandem vs DOT limits)

When you're ready, hit **Start** to open the optimizer.
""")


    c1 = st.columns(3)
    with c1[0]:
        if st.button("Start", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()



if not st.session_state.started:
    with st.sidebar:
        if _LOGO_B64:
            st.image(LOGO_PATH, width=120)
        st.header("MCLOS")
        st.write("Click start to enter the optimizer UI.")
        if st.button("Start MCLOS", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    render_welcome()
    st.stop()

# ─── Main app ──────────────────────────────────────────────────────────────
if _LOGO_B64:
    _hdr_logo, _hdr_title = st.columns([1, 5])
    with _hdr_logo:
        st.image(LOGO_PATH, width=100)
    with _hdr_title:
        st.title("MCLOS")
        st.caption("HexHomes Panel Packing Optimizer  |  Alpha V3.0")
else:
    st.title("MCLOS - Modular Cargo Loading Optimization Software")
    st.caption("HexHomes Panel Packing Optimizer  |  Alpha V3.0")

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    if _LOGO_B64:
        st.image(LOGO_PATH, width=100)
    # Navigation
    if st.button("Back to Welcome", use_container_width=True):
        st.session_state.started = False
        st.rerun()

    st.header("Loading Strategy")
    strategy_name = st.selectbox("Strategy", [s.value for s in LoadingStrategy], index=0,
                                  help="Gravity Layered: flat panels, low CG. Wall First: upright side-by-side. "
                                       "Zone Based: balanced across thirds. Stress Optimized: minimizes bottom stress.")
    strategy = [s for s in LoadingStrategy if s.value == strategy_name][0]

    st.markdown("---")
    st.header("Display")
    color_by_stress = st.checkbox("Color by Stress Level", value=False,
                                   help="Colors all panels green/yellow/red by worst utilization ratio")
    show_stress_markers = st.checkbox("Show Stress Markers", value=True,
                                      help="Highlights worst-case panel + shows stress detail card")
    show_axles = st.checkbox("Show Axle Positions", value=True,
                              help="Shows drive & trailer tandem axle lines + load analysis card")

    st.markdown("---")
    with st.expander("Advanced Settings"):
        st.subheader("Structural Limits")
        adv_comp = st.number_input("Max Compression (psi)", value=50.0, min_value=1.0, key="ac")
        adv_bend = st.number_input("Max Bending Stress (psi)", value=500.0, min_value=1.0, key="ab")
        adv_shear = st.number_input("Max Shear (psi)", value=45.0, min_value=1.0, key="as")
        adv_safety = st.number_input("Safety Factor", value=2.0, min_value=1.0, key="asf")
        adv_youngs = st.number_input("Young's Modulus (psi)", value=1800000.0, min_value=1000.0, key="ay")
        st.markdown("---")
        st.subheader("Handling")
        maxH = st.number_input("Max Horizontal (in)", value=230.0, min_value=1.0, key="amh")
        maxV = st.number_input("Max Vertical (in)", value=114.0, min_value=1.0, key="amv")

    st.markdown("---")
    st.subheader("Fit / Spacing")
    trailer_tol_in = st.number_input(
        "Trailer dimension tolerance (in)",
        value=0.0, min_value=0.0, step=0.25, key="ttol",
        help="Total clearance reserved across each dimension; applied as tol/2 on both sides."
    )
    panel_gap_in = st.number_input(
        "Panel-to-panel gap (in)",
        value=1.5, min_value=0.0, step=0.25, key="pgap",
        help="Gap between adjacent panels (wood spacer blocks). HexHomes standard: 1.5 in."
    )
    align_x = st.selectbox(
        "Longitudinal alignment",
        ["front", "center", "rear"], index=0, key="axalign",
        help="Where to place the packed block within leftover length slack."
    )
    align_y = st.selectbox(
        "Lateral alignment",
        ["front", "center", "rear"], index=1, key="ayalign",
        help="Where to place the packed block within leftover width slack."
    )
    st.markdown("---")
    st.subheader("Axle Load Optimization")
    auto_best = st.checkbox(
        "Auto-pick lowest axle-load configuration",
        value=True, key="autobest",
        help="Runs all strategies and keeps the configuration with the lowest axle objective (ties: more panels)."
    )
    axle_opt_shift = st.checkbox(
        "Fine-tune by shifting load block",
        value=True, key="axshift",
        help="Uses unused length to shift the whole load forward/back to reduce axle utilization."
    )
    axle_shift_step = st.number_input(
        "Shift search step (in)",
        value=1.0, min_value=0.25, step=0.25, key="axstep"
    )

stress_config = StressConfig(
    max_compression_psi=adv_comp, max_bending_stress_psi=adv_bend,
    max_shear_psi=adv_shear, safety_factor=adv_safety, panel_youngs_modulus_psi=adv_youngs
)

# ─── Trailer ────────────────────────────────────────────────────────────────
st.subheader("Trailer Selection")
# Initialize trailer dims with safe defaults so static analyzers can't flag them as unbound.
tL = TRAILER_DIMS[TrailerPreset.FT53_ENCLOSED]["L"]
tW = TRAILER_DIMS[TrailerPreset.FT53_ENCLOSED]["W"]
tH = TRAILER_DIMS[TrailerPreset.FT53_ENCLOSED]["H"]
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

# Flatbed detection — determines if overhang and tessellation are available
if tp == TrailerPreset.CUSTOM:
    is_flatbed = st.checkbox("Flatbed (open sides, allows floor panel overhang)", value=False, key="custom_fb")
else:
    is_flatbed = tp in (TrailerPreset.FT53_FLATBED, TrailerPreset.FT42_FLATBED)

# Flatbed overhang control (only relevant for flatbed trailers)
if is_flatbed:
    overhang_max_in = st.number_input(
        "Max flatbed overhang per side (in)",
        value=12.0, min_value=0.0, max_value=24.0, step=1.0, key="overhang",
        help="Floor panels can extend past deck width on flatbed trailers (no walls). "
             "Max per-side overhang. 0 = no overhang allowed."
    )
else:
    overhang_max_in = 0.0

# Axle config: use preset or build default for custom trailers
if tp in AXLE_CONFIGS:
    axle_config = AXLE_CONFIGS[tp]
else:
    # Custom trailer: estimate trailer tandem at ~75% of length (common position)
    # Use tL if defined by the UI, otherwise fall back to known TRAILER_DIMS or a reasonable default.
    trailer_length = None
    maybe_tL = locals().get("tL")
    if isinstance(maybe_tL, (int, float)):
        trailer_length = maybe_tL
    elif tp in TRAILER_DIMS and "L" in TRAILER_DIMS[tp]:
        trailer_length = TRAILER_DIMS[tp]["L"]
    else:
        trailer_length = 504.0  # conservative default (42-ft trailer length in inches)

    axle_config = AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=round(trailer_length * 0.75, 1), trailer_tandem_spacing=49.0,
    )

# ─── Panels ─────────────────────────────────────────────────────────────────
st.subheader("Panel Configuration")
# Define num_pods with a safe default so static analysis can't report it as unbound;
# the Streamlit UI will overwrite this when "By Pod Count (quick)" is selected.
num_pods = 0
# Predefine manual counts so static analyzers won't flag them as possibly unbound
manual_walls = 0
manual_floors = 0
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
    st.caption(f"Run #{st.session_state.run_count}")

if run_btn:
    if use_pods:
        panel_list = build_panel_list_from_pods(num_pods, wall_spec, floor_spec)
    else:
        panel_list = build_panel_list_manual(manual_walls, manual_floors, wall_spec, floor_spec)

    if not panel_list:
        st.error("No panels to optimize.")
    else:
        t0 = time.time()
        with st.spinner(f"Optimizing {len(panel_list)} panels..."):
            if auto_best:
                out = optimize_best_configuration(
                    panel_list, tL, tW, tH,
                    maxH, maxV,
                    stress_config=stress_config, axle_config=axle_config,
                    panel_gap_in=panel_gap_in, trailer_tol_in=trailer_tol_in,
                    align_x=align_x, align_y=align_y,
                    axle_optimize_shift=axle_opt_shift, axle_shift_step=axle_shift_step,
                    is_flatbed=is_flatbed, overhang_max_in=overhang_max_in
                )
            else:
                out = pack_panels_v14(
                    panel_list, tL, tW, tH,
                    maxH, maxV,
                    strategy=strategy, stress_config=stress_config, axle_config=axle_config,
                    panel_gap_in=panel_gap_in, trailer_tol_in=trailer_tol_in,
                    align_x=align_x, align_y=align_y,
                    axle_optimize_shift=axle_opt_shift, axle_shift_step=axle_shift_step,
                    is_flatbed=is_flatbed, overhang_max_in=overhang_max_in
                )
        elapsed = time.time() - t0
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
                sel = out.get('settings', {}).get('selected_strategy', strategy.value)
                axobj = out.get('settings', {}).get('axle_objective', None)
                shift = out.get('settings', {}).get('axle_shift_in', 0.0)
                label = f"Strategy: **{sel}**"
                if axobj is not None:
                    label += f" | Axle score: **{axobj:.3f}**"
                if abs(shift) > 1e-6:
                    label += f" | Shift: **{shift:.1f} in**"
                st.info(label)
            st.subheader("3D Trailer View")
            visualize(out, stress_config, color_by_stress, show_stress_markers, show_axles, floor_spec)

            if out["rejections"]:
                with st.expander(f"Rejected Panels ({len(out['rejections'])})", expanded=True):
                    for r in out["rejections"]:
                        st.error(f"**{r['label']}** ({r['panel_type']}): {r['reason']}")

            if out["placements"]:
                with st.expander("Detailed Stress Analysis"):
                    sd = [{
                        "Panel": p["label"], "Type": p["panel_type"], "Orient": p["orientation"],
                        "Layer": p["layer"],
                        "Comp (psi)": p["stress_analysis"]["compression_psi"],
                        "Bend (psi)": p["stress_analysis"]["bending_stress_psi"],
                        "Shear (psi)": p["stress_analysis"]["shear_psi"],
                        "Defl (in)": p["stress_analysis"]["deflection_in"],
                        "Util Ratio": p["stress_analysis"]["utilization_ratio"],
                        "Limiting": p["stress_analysis"]["limiting_factor"],
                        "Total Load (lb)": p["stress_analysis"]["total_load_lb"],
                        "Wt Above (lb)": p["stress_analysis"]["weight_above_lb"],
                        "Footprint (sq.in)": p["stress_analysis"]["footprint_sqin"],
                    } for p in out["placements"]]
                    st.dataframe(sd, use_container_width=True)

            with st.expander("View JSON"):
                st.json(out)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("Download JSON", json.dumps(out, indent=2),
                                   f"mclos_v30_run{st.session_state.run_count}.json", "application/json", use_container_width=True)
            with d2:
                hdr = "id,label,type,x,y,z,dx,dy,dz,orient,weight,layer,load_order,flipped,comp_psi,bend_psi,shear_psi,defl_in,util_ratio,limiting,total_load,wt_above\n"
                rows = "\n".join([
                    f"{p['id']},{p['label']},{p['panel_type']},"
                    f"{p['position_vector'][0]},{p['position_vector'][1]},{p['position_vector'][2]},"
                    f"{p['size'][0]},{p['size'][1]},{p['size'][2]},{p['orientation']},{p['weight']},"
                    f"{p['layer']},{p['loading_order']},{p.get('flipped', False)},"
                    f"{p['stress_analysis']['compression_psi']},"
                    f"{p['stress_analysis']['bending_stress_psi']},"
                    f"{p['stress_analysis']['shear_psi']},"
                    f"{p['stress_analysis']['deflection_in']},"
                    f"{p['stress_analysis']['utilization_ratio']},"
                    f"{p['stress_analysis']['limiting_factor']},"
                    f"{p['stress_analysis']['total_load_lb']},{p['stress_analysis']['weight_above_lb']}"
                    for p in out.get("placements", [])
                ])
                # Append axle load summary section to CSV
                axle_csv = out.get("axle_loads")
                if axle_csv:
                    rows += "\n\n# Axle Load Analysis\n"
                    rows += "axle_group,load_lb,limit_lb,utilization_pct,per_axle_lb\n"
                    rows += f"Steer,{axle_csv['steer_axle_lb']},{axle_csv['limits']['steer_limit_lb']},{axle_csv['utilization_pct']['steer']},N/A\n"
                    rows += f"Drive Tandem,{axle_csv['drive_tandem_lb']},{axle_csv['limits']['drive_tandem_limit_lb']},{axle_csv['utilization_pct']['drive_tandem']},{axle_csv['drive_per_axle_lb']}\n"
                    rows += f"Trailer Tandem,{axle_csv['trailer_tandem_lb']},{axle_csv['limits']['trailer_tandem_limit_lb']},{axle_csv['utilization_pct']['trailer_tandem']},{axle_csv['trailer_per_axle_lb']}\n"
                    rows += f"Gross Vehicle,{axle_csv['gross_vehicle_weight_lb']},{axle_csv['limits']['gross_limit_lb']},{axle_csv['utilization_pct']['gross']},N/A\n"
                    rows += f"\n# Cargo Weight: {axle_csv['cargo_weight_lb']} lb | Tractor Weight: {axle_csv['tractor_weight_lb']} lb\n"
                st.download_button("Download CSV", hdr + rows,
                                   f"mclos_v30_run{st.session_state.run_count}.csv", "text/csv", use_container_width=True)
        else:
            st.error(out["error"])
