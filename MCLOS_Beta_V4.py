import json
import math
import base64
import os
import streamlit as st
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass


class LoadingStrategy(Enum):
    GRAVITY_LAYERED = "Gravity Layered"
    WALL_FIRST = "Wall First"
    ZONE_BASED = "Zone Based"
    STRESS_OPTIMIZED = "Stress Optimized"


class PanelType(Enum):
    WALL = "Wall Panel"
    FLOOR = "Floor Panel (Half-Hex)"
    ROOF = "Roof Panel (Half-Hex)"


class TrailerPreset(Enum):
    FT53_FLATBED = "53-ft Flatbed"
    FT42_FLATBED = "42-ft Flatbed"
    CUSTOM = "Custom"


class AxleConfig:
    def __init__(self, drive_tandem_x, drive_tandem_spacing, trailer_tandem_x, trailer_tandem_spacing,
                 steer_limit_lb=12000.0, drive_tandem_limit_lb=34000.0,
                 trailer_tandem_limit_lb=34000.0, gross_limit_lb=80000.0,
                 tractor_weight_lb=17000.0, tractor_wheelbase=240.0):
        self.drive_tandem_x = drive_tandem_x
        self.drive_tandem_spacing = drive_tandem_spacing
        self.trailer_tandem_x = trailer_tandem_x
        self.trailer_tandem_spacing = trailer_tandem_spacing
        self.steer_limit_lb = steer_limit_lb
        self.drive_tandem_limit_lb = drive_tandem_limit_lb
        self.trailer_tandem_limit_lb = trailer_tandem_limit_lb
        self.gross_limit_lb = gross_limit_lb
        self.tractor_weight_lb = tractor_weight_lb
        self.tractor_wheelbase = tractor_wheelbase


AXLE_CONFIGS = {
    TrailerPreset.FT53_FLATBED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=480.0, trailer_tandem_spacing=49.0,
    ),
    TrailerPreset.FT42_FLATBED: AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=378.0, trailer_tandem_spacing=49.0,
    ),
}


TRAILER_DIMS = {
    TrailerPreset.FT53_FLATBED:  {"L": 636.0, "W": 102.0, "H": 168.0,
                                   "desc": "53-ft flatbed (open, legal height limit: 14ft / 168\" in CA without permit)"},
    TrailerPreset.FT42_FLATBED:  {"L": 504.0, "W": 102.0, "H": 168.0,
                                   "desc": "42-ft flatbed (open, legal height limit: 14ft / 168\" in CA without permit)"},
}


class StressConfig:
    def __init__(self, max_compression_psi=50.0, max_bending_stress_psi=500.0,
                 max_shear_psi=45.0, safety_factor=2.0, panel_youngs_modulus_psi=1800000.0):
        self.max_compression_psi = max_compression_psi
        self.max_bending_stress_psi = max_bending_stress_psi
        self.max_shear_psi = max_shear_psi
        self.safety_factor = safety_factor
        self.panel_youngs_modulus_psi = panel_youngs_modulus_psi


class PanelSpec:
    def __init__(self, panel_type, length, height, thickness, weight, short_edge=0.0):
        self.panel_type = panel_type
        self.length = length
        self.height = height
        self.thickness = thickness
        self.weight = weight
        self.short_edge = short_edge

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
ROOF_PANEL_DEFAULT = PanelSpec(
    panel_type=PanelType.ROOF, length=225.96, height=98.95, thickness=25.0,
    weight=1002.0, short_edge=112.0
)


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
    x, y, z = pos
    dx, dy, dz = size
    tL, tW, tH = trailer

    if x < -0.01 or x + dx > tL + 0.01:
        return False
    if z < -0.01 or z + dz > tH + 0.01:
        return False

    if is_flatbed and panel_is_floor_flat and overhang_max > 0:
        if y < -(overhang_max + 0.01):
            return False
        if y + dy > tW + overhang_max + 0.01:
            return False
        panel_center_y = y + dy / 2.0
        if panel_center_y < 0 or panel_center_y > tW:
            return False
        return True
    else:
        return y >= -0.01 and y + dy <= tW + 0.01


def handling_ok(size, max_horizontal, max_vertical):
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical


def get_trapezoid_footprint_2d(pos, size, axis_map, short_edge, long_edge, flipped=False):
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
    return [(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]


def polygons_overlap_2d(poly_a, poly_b):
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
        if max_a <= min_b + 0.01 or max_b <= min_a + 0.01:
            return False
    return True


def panels_collide(p1_pos, p1_size, p1_info, p2_pos, p2_size, p2_info):
    if not aabb_intersect(p1_pos, p1_size, p2_pos, p2_size):
        return False

    p1_flat_trap = (p1_info.get('is_trapezoid', False) and
                    p1_info.get('axis_map', ('', '', ''))[2] == 'T')
    p2_flat_trap = (p2_info.get('is_trapezoid', False) and
                    p2_info.get('axis_map', ('', '', ''))[2] == 'T')

    if p1_flat_trap or p2_flat_trap:
        z1, z2 = p1_pos[2], p2_pos[2]
        dz1, dz2 = p1_size[2], p2_size[2]
        if z1 + dz1 <= z2 + 0.01 or z2 + dz2 <= z1 + 0.01:
            return False

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

    return True


def calculate_support_area(pos, size, placed, gap=0.0):
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
    if not supporting_panels:
        return 0.0

    x, y, z = pos
    dx, dy, dz = size
    c = dz / 2.0

    sx_positions = [s['centroid'][0] for s in supporting_panels]
    sy_positions = [s['centroid'][1] for s in supporting_panels]

    max_bending = 0.0

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
    dx, dy, dz = size
    A_shear = min(dx, dy) * dz
    if A_shear < 0.01:
        return float('inf')
    return 1.5 * total_weight / A_shear


def calculate_deflection(panel_weight, pos, size, youngs_modulus, supporting_panels):
    if not supporting_panels:
        return 0.0

    x, y, z = pos
    dx, dy, dz = size

    sx_positions = [s['centroid'][0] for s in supporting_panels]
    sy_positions = [s['centroid'][1] for s in supporting_panels]

    max_defl = 0.0

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
    for i, other in enumerate(placed):
        if i == panel_id:
            continue
        ox, oy, oz = other["pos"]
        odx, ody, odz = other["size"]
        if oz < top - Z_TOL - gap:
            continue
        xo = max(0.0, min(px + pdx, ox + odx) - max(px, ox))
        yo = max(0.0, min(py + pdy, oy + ody) - max(py, oy))
        if xo > 0.01 and yo > 0.01:
            overlap_area = xo * yo
            other_area = max(odx * ody, 0.01)
            frac = min(overlap_area / other_area, 1.0)
            total += other["weight"] * frac
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


def compute_max_stack_height(panel_weight, panel_size, trailer_H, stress_config, gap=0.0):
    dz = panel_size[2]
    shear_area = min(panel_size[0], panel_size[1]) * dz
    if shear_area < 0.01:
        return 1, dz
    max_shear = stress_config.max_shear_psi / stress_config.safety_factor
    max_column_weight = max_shear * shear_area / 1.5
    max_panels_shear = int(max_column_weight / panel_weight) if panel_weight > 0 else 999
    stride_z = dz + gap
    max_panels_height = int((trailer_H + gap) / stride_z) if stride_z > 0 else 1
    max_panels = max(1, min(max_panels_shear, max_panels_height))
    return max_panels, max_panels * dz + max(0, max_panels - 1) * gap


def column_stress_ok(col_key, new_weight, column_tracker, placed, stress_config):
    state = column_tracker.get(col_key)
    if state is None:
        return True, "OK"

    new_total = state['total_weight'] + new_weight

    bottom_panel = None
    for p in placed:
        px, py = round(p["pos"][0], 2), round(p["pos"][1], 2)
        if (px, py) == col_key and (bottom_panel is None or p["pos"][2] < bottom_panel["pos"][2]):
            bottom_panel = p

    if bottom_panel is None:
        return True, "OK"

    shear = calculate_shear_stress(new_total, bottom_panel["size"])
    if shear * stress_config.safety_factor > stress_config.max_shear_psi:
        return False, f"Column shear on bottom panel: {shear:.1f} psi"

    return True, "OK"


def compute_cg(placed):
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
    ac = axle_config
    d_x = ac.drive_tandem_x
    t_x = ac.trailer_tandem_x
    span = t_x - d_x

    if span < 1.0:
        return None

    cargo_on_drive = 0.0
    cargo_on_trailer = 0.0
    total_cargo = 0.0

    for p in placed:
        w = p["weight"]
        cx = p["pos"][0] + p["size"][0] / 2
        total_cargo += w

        r_trailer = w * (cx - d_x) / span
        r_drive = w - r_trailer

        cargo_on_drive += r_drive
        cargo_on_trailer += r_trailer

    tractor_w = ac.tractor_weight_lb
    steer_frac = 0.40
    tractor_on_steer = tractor_w * steer_frac
    tractor_on_drive = tractor_w * (1.0 - steer_frac)

    steer_load = tractor_on_steer
    drive_load = tractor_on_drive + cargo_on_drive
    trailer_load = cargo_on_trailer

    gross = steer_load + drive_load + trailer_load

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


def axle_objective(axle_loads):
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
    if n > 601:
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


def generate_floor_slots(dx, dy, trailer_L, trailer_W, gap=0.0,
                         align_x="front", align_y="front", overhang_max=0.0):
    try:
        gap = max(0.0, float(gap))
    except Exception:
        gap = 0.0

    if dx <= 0 or dy <= 0:
        return []

    eff_W = trailer_W + 2.0 * max(0.0, overhang_max)

    if trailer_L + 0.01 < dx or eff_W + 0.01 < dy:
        return []

    stride_x = dx + gap
    stride_y = dy + gap

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
        return 0.0

    x0 = start_from_align(align_x, slack_L)
    y0_eff = start_from_align(align_y, slack_W)
    y0 = y0_eff - max(0.0, overhang_max)

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
    if dx <= 0 or dy <= 0:
        return []

    offset = (long_edge - short_edge) / 2.0
    eff_W = trailer_W + 2.0 * max(0.0, overhang_max)

    if eff_W + 0.01 < dy:
        return []

    y_start = (trailer_W - dy) / 2.0

    tessellated_stride = long_edge - offset + gap

    if tessellated_stride <= 0:
        return []

    max_count = int(math.floor((trailer_L - long_edge) / tessellated_stride + 1.0 + 1e-9))
    max_count = max(max_count, 0)

    if max_count == 0:
        if long_edge <= trailer_L + 0.01:
            max_count = 1
        else:
            return []

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
    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        return sorted(orientations, key=lambda o: (
            0 if o['axis_map'][2] == 'T' else 1,
            -o['total_capacity'],
        ))

    elif strategy == LoadingStrategy.WALL_FIRST:
        return sorted(orientations, key=lambda o: (
            1 if o['axis_map'][2] == 'T' else 0,
            -o['size'][2],
            -len(o['floor_slots']),
        ))

    elif strategy == LoadingStrategy.ZONE_BASED:
        return sorted(orientations, key=lambda o: (
            0 if o['axis_map'][2] == 'T' else 1,
            -o['total_capacity'],
        ))

    elif strategy == LoadingStrategy.STRESS_OPTIMIZED:
        return sorted(orientations, key=lambda o: (
            -(o['size'][0] * o['size'][1]),
            0 if o['axis_map'][2] == 'T' else 1,
        ))

    return orientations


def apply_strategy_slot_order(strategy, floor_slots, placed, trailer, panel_size, is_flatbed=False):
    dx, dy, dz = panel_size
    trailer_L, trailer_W, trailer_H = trailer

    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        if is_flatbed:
            center_y = trailer_W / 2.0
            return sorted(floor_slots, key=lambda s: (abs(s[1] + dy / 2.0 - center_y), s[0]))
        return sorted(floor_slots, key=lambda s: (s[0], s[1]))

    elif strategy == LoadingStrategy.WALL_FIRST:
        def wall_priority(slot):
            x, y = slot
            dist_to_side = min(y, max(0, trailer_W - (y + dy)))
            return (round(dist_to_side, 2), x, y)
        return sorted(floor_slots, key=wall_priority)

    elif strategy == LoadingStrategy.ZONE_BASED:
        zone_width = trailer_L / 3.0
        zones = [[], [], []]
        for slot in floor_slots:
            zi = min(int(slot[0] / zone_width), 2)
            zones[zi].append(slot)
        if is_flatbed:
            center_y = trailer_W / 2.0
            for z in zones:
                z.sort(key=lambda s: (abs(s[1] + dy / 2.0 - center_y), s[0]))
        else:
            for z in zones:
                z.sort(key=lambda s: (s[0], s[1]))
        result = []
        max_len = max((len(z) for z in zones), default=0)
        for i in range(max_len):
            for z in zones:
                if i < len(z):
                    result.append(z[i])
        return result

    elif strategy == LoadingStrategy.STRESS_OPTIMIZED:
        center_x = trailer_L / 2.0
        if placed:
            tw = sum(p["weight"] for p in placed)
            cg_x = sum(p["weight"] * (p["pos"][0] + p["size"][0] / 2) for p in placed) / tw if tw > 0 else center_x
        else:
            cg_x = center_x

        def cg_score(slot):
            slot_cx = slot[0] + dx / 2
            dist = abs(slot_cx - center_x)
            imbalance = (slot_cx - center_x) * (cg_x - center_x)
            return dist + imbalance * 0.3
        return sorted(floor_slots, key=cg_score)

    return floor_slots


def build_panel_list_from_pods(num_pods, wall_spec, floor_spec, roof_spec):
    panels = []
    for pod in range(num_pods):
        for half in range(2):
            panels.append({
                "spec": floor_spec,
                "label": f"Pod{pod + 1}_Floor_{half + 1}",
                "panel_type": PanelType.FLOOR.value,
            })
    for pod in range(num_pods):
        panels.append({
            "spec": roof_spec,
            "label": f"Pod{pod + 1}_Roof",
            "panel_type": PanelType.ROOF.value,
        })
    for pod in range(num_pods):
        for wall in range(6):
            panels.append({
                "spec": wall_spec,
                "label": f"Pod{pod + 1}_Wall_{wall + 1}",
                "panel_type": PanelType.WALL.value,
            })
    return panels


def build_panel_list_manual(num_walls, num_floors, num_roofs, wall_spec, floor_spec, roof_spec):
    panels = []
    for i in range(num_floors):
        panels.append({
            "spec": floor_spec,
            "label": f"Floor_{i + 1}",
            "panel_type": PanelType.FLOOR.value,
        })
    for i in range(num_roofs):
        panels.append({
            "spec": roof_spec,
            "label": f"Roof_{i + 1}",
            "panel_type": PanelType.ROOF.value,
        })
    for i in range(num_walls):
        panels.append({
            "spec": wall_spec,
            "label": f"Wall_{i + 1}",
            "panel_type": PanelType.WALL.value,
        })
    return panels


def pack_panels_v14(panel_list, trailer_L, trailer_W, trailer_H,
                    max_horizontal, max_vertical,
                    strategy, stress_config, axle_config=None,
                    panel_gap_in=0.0, trailer_tol_in=0.0,
                    align_x="front", align_y="front",
                    axle_optimize_shift=False, axle_shift_step=1.0,
                    is_flatbed=False, overhang_max_in=12.0,
                    allow_vertical=False):
    tol = max(0.0, float(trailer_tol_in))
    L_eff = trailer_L - tol
    W_eff = trailer_W - tol
    H_eff = trailer_H - tol
    if L_eff <= 0 or W_eff <= 0 or H_eff <= 0:
        return {"error": f"Trailer tolerance {tol:.2f}\" is too large for the selected trailer dimensions."}
    base_offset = tol / 2.0
    trailer = (L_eff, W_eff, H_eff)

    # Z-probe only needed for high panel counts (5+ pods) where
    # roof-floor contention on tessellated positions requires stacking recovery.
    # For low counts (1-4 pods), skipping z-probe keeps arrangements compact.
    use_z_probe = len(panel_list) > 36

    placed = []
    rejected = []

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
            panel_is_floor_flat = (spec.is_trapezoid and axis_map[2] == 'T')
            if not in_bounds_with_overhang((0, 0, 0), size, trailer,
                                           is_flatbed, overhang_max_in, panel_is_floor_flat):
                continue
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            if is_flatbed and axis_map[2] != 'T':
                continue

            dx, dy, dz = size

            if spec.is_trapezoid and name.startswith("flat_") and is_flatbed and trailer_L > 510:
                # Tessellated packing with alternating flip for longer trailers
                # (53-ft / 636"). Interlocking trapezoids maximize space.
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

        orientation_cache[cache_key] = valid

    if is_flatbed and len(panel_list) > 36:
        # High panel counts (5+ pods): floors must claim ground slots before
        # roofs to avoid roof-floor contention on tessellated positions.
        sorted_panels = sorted(panel_list, key=lambda p: (
            {PanelType.FLOOR: 0, PanelType.ROOF: 1}.get(p['spec'].panel_type, 2),
            -p['spec'].weight, p['label']
        ))
    elif is_flatbed:
        # Low panel counts (1-4 pods): weight-based sort gives compact,
        # balanced arrangements since there is ample room.
        sorted_panels = sorted(panel_list, key=lambda p: (
            0 if p['spec'].panel_type in (PanelType.FLOOR, PanelType.ROOF) else 1,
            -p['spec'].weight, p['label']
        ))
    elif strategy == LoadingStrategy.GRAVITY_LAYERED:
        sorted_panels = sorted(panel_list, key=lambda p: (-p['spec'].weight, p['label']))
    elif strategy == LoadingStrategy.WALL_FIRST:
        sorted_panels = sorted(panel_list, key=lambda p: (
            0 if p['spec'].panel_type == PanelType.WALL else 1, p['label']))
    elif strategy == LoadingStrategy.ZONE_BASED:
        sorted_panels = sorted(panel_list, key=lambda p: (
            p['label'].split('_')[-1], p['label'].split('_')[0]))
    else:
        sorted_panels = sorted(panel_list, key=lambda p: (
            -p['spec'].weight, -p['spec'].length * p['spec'].height))

    column_tracker = {}

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

            panel_is_floor_flat = (spec.is_trapezoid and axis_map[2] == 'T')

            if not floor_slots:
                continue

            ordered_slots = apply_strategy_slot_order(
                strategy, floor_slots, placed, trailer, size, is_flatbed=is_flatbed
            )

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
                        continue
                    z = state['z_top']

                test_pos = (sx, sy, z)

                if not in_bounds_with_overhang(test_pos, size, trailer,
                                               is_flatbed, overhang_max_in, panel_is_floor_flat):
                    continue

                panel_info_new = {
                    'is_trapezoid': spec.is_trapezoid,
                    'axis_map': axis_map,
                    'short_edge': spec.short_edge,
                    'spec_length': spec.length,
                    'flipped': current_flipped,
                }
                if any(panels_collide(test_pos, size, panel_info_new,
                                       p["pos"], p["size"], p) for p in placed):
                    if use_z_probe:
                        # Z-probe: find highest z_top from XY-overlapping panels and retry
                        probe_z = z
                        for p in placed:
                            px, py, pz = p["pos"]
                            pdx, pdy, pdz = p["size"]
                            if sx < px + pdx and sx + dx > px and sy < py + pdy and sy + dy > py:
                                p_top = round(pz + pdz + panel_gap_in, 4)
                                if p_top > probe_z:
                                    probe_z = p_top
                        if probe_z > z:
                            z = probe_z
                            test_pos = (sx, sy, z)
                            if not in_bounds_with_overhang(test_pos, size, trailer,
                                                           is_flatbed, overhang_max_in, panel_is_floor_flat):
                                continue
                            if any(panels_collide(test_pos, size, panel_info_new,
                                                   p["pos"], p["size"], p) for p in placed):
                                continue
                        else:
                            continue
                    else:
                        continue

                stress_valid, msg = stress_ok(test_pos, size, spec.weight, placed, stress_config, gap=panel_gap_in)
                if not stress_valid:
                    continue

                if z > Z_TOL:
                    col_ok, col_msg = column_stress_ok(
                        col_key, spec.weight, column_tracker, placed, stress_config
                    )
                    if not col_ok:
                        continue

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

        if not placed_this and placed:
            for orient in orientations:
                name = orient['name']
                size = orient['size']
                axis_map = orient['axis_map']
                dx, dy, dz = size
                panel_is_floor_flat = (spec.is_trapezoid and axis_map[2] == 'T')

                for base in placed:
                    bx, by, bz = base["pos"]
                    bdx, bdy, bdz = base["size"]

                    test_z = round(bz + bdz + panel_gap_in, 4)

                    candidates = set()
                    candidates.add((bx, by))
                    clamp_x = max(0.0, min(bx, trailer_L - dx))
                    clamp_y = max(0.0, min(by, trailer_W - dy))
                    candidates.add((round(clamp_x, 4), round(clamp_y, 4)))
                    center_x_on_base = bx + (bdx - dx) / 2.0
                    center_y_on_base = by + (bdy - dy) / 2.0
                    cx = max(0.0, min(center_x_on_base, trailer_L - dx))
                    cy = max(0.0, min(center_y_on_base, trailer_W - dy))
                    candidates.add((round(cx, 4), round(cy, 4)))

                    for (tx, ty) in candidates:
                        test_pos = (tx, ty, test_z)

                        if not in_bounds_with_overhang(test_pos, size, trailer,
                                                       is_flatbed, overhang_max_in, panel_is_floor_flat):
                            continue

                        panel_info_new = {
                            'is_trapezoid': spec.is_trapezoid,
                            'axis_map': axis_map,
                            'short_edge': spec.short_edge,
                            'spec_length': spec.length,
                            'flipped': False,
                        }
                        if any(panels_collide(test_pos, size, panel_info_new,
                                               p["pos"], p["size"], p) for p in placed):
                            continue

                        stress_valid, msg = stress_ok(test_pos, size, spec.weight, placed, stress_config, gap=panel_gap_in)
                        if not stress_valid:
                            continue

                        col_key_ms = (round(tx, 2), round(ty, 2))
                        if test_z > Z_TOL:
                            col_ok, col_msg = column_stress_ok(
                                col_key_ms, spec.weight, column_tracker, placed, stress_config
                            )
                            if not col_ok:
                                continue

                        placed.append({
                            "id": pid,
                            "label": panel_info["label"],
                            "panel_type": panel_info["panel_type"],
                            "pos": test_pos,
                            "size": size,
                            "orientation": name + " (mixed-stack)",
                            "axis_map": axis_map,
                            "weight": spec.weight,
                            "is_trapezoid": spec.is_trapezoid,
                            "short_edge": spec.short_edge,
                            "spec_length": spec.length,
                            "spec_height": spec.height,
                            "spec_thickness": spec.thickness,
                            "flipped": False,
                        })

                        state_ms = column_tracker.get(col_key_ms)
                        if state_ms is None:
                            column_tracker[col_key_ms] = {
                                'count': 1, 'z_top': round(test_z + dz + panel_gap_in, 4),
                                'total_weight': spec.weight,
                            }
                        else:
                            state_ms['count'] += 1
                            state_ms['z_top'] = round(test_z + dz + panel_gap_in, 4)
                            state_ms['total_weight'] += spec.weight

                        placed_this = True
                        break
                    if placed_this:
                        break
                if placed_this:
                    break

        if not placed_this:
            rejected.append({
                "id": pid, "label": panel_info["label"],
                "panel_type": panel_info["panel_type"],
                "reason": "No valid position found (stress/collision/bounds/capacity)"
            })

    # ── Vertical Wall Fallback (second pass) ──────────────────────────
    # After the main placement loop, attempt to fill remaining gaps with
    # standing wall panels. Ground-level only (z=0) to avoid random panels
    # on top.  Only runs on flatbed when allow_vertical is enabled.
    vertical_fill_count = 0
    if allow_vertical and is_flatbed and rejected:
        wall_rejected = [r for r in rejected
                         if r["panel_type"] == PanelType.WALL.value]

        still_rejected = []
        for rj in wall_rejected:
            rid = rj["id"]
            rj_spec = sorted_panels[rid]["spec"]
            wall_L, wall_H, wall_T = rj_spec.length, rj_spec.height, rj_spec.thickness

            standing_orients = []
            for oname, osize, oaxis in generate_orientations_wall(wall_L, wall_H, wall_T):
                if oaxis[2] == 'T':
                    continue  # skip flat orientations
                odx, ody, odz = osize
                if odx > trailer_L or ody > trailer[1] or odz > trailer_H:
                    continue
                if not handling_ok(osize, max_horizontal, max_vertical):
                    continue
                standing_orients.append((oname, osize, oaxis))

            # Prefer thin-dx orientations first so the panel's flat face
            # points inward toward the load rather than sideways.
            standing_orients.sort(key=lambda o: o[1][0])

            placed_vert = False
            for oname, osize, oaxis in standing_orients:
                odx, ody, odz = osize
                x_step = max(odx, 1.0)
                y_step = max(ody + panel_gap_in, 1.0)

                sx = 0.0
                while sx + odx <= trailer_L + 0.01:
                    sy = 0.0
                    while sy + ody <= trailer[1] + 0.01:
                        test_pos = (round(sx, 4), round(sy, 4), 0.0)
                        panel_info_vert = {
                            "is_trapezoid": False,
                            "flipped": False,
                            "short_edge": 0,
                            "spec_length": rj_spec.length,
                        }

                        if not in_bounds_with_overhang(test_pos, osize, trailer,
                                                       is_flatbed, 0.0, False):
                            sy += y_step
                            continue

                        if any(panels_collide(test_pos, osize, panel_info_vert,
                                               p["pos"], p["size"], p) for p in placed):
                            sy += y_step
                            continue

                        stress_valid, _ = stress_ok(test_pos, osize, rj_spec.weight,
                                                    placed, stress_config, gap=panel_gap_in)
                        if not stress_valid:
                            sy += y_step
                            continue

                        placed.append({
                            "id": rid,
                            "label": rj["label"],
                            "panel_type": rj["panel_type"],
                            "pos": test_pos,
                            "size": osize,
                            "orientation": f"{oname} (vertical-fill)",
                            "axis_map": oaxis,
                            "weight": rj_spec.weight,
                            "is_trapezoid": False,
                            "short_edge": 0,
                            "spec_length": rj_spec.length,
                            "spec_height": rj_spec.height,
                            "spec_thickness": rj_spec.thickness,
                            "flipped": False,
                        })

                        col_key = (round(sx / max(odx, 1), 0), round(sy / max(ody, 1), 0))
                        if col_key not in column_tracker:
                            column_tracker[col_key] = {
                                'count': 1,
                                'z_top': round(odz + panel_gap_in, 4),
                                'total_weight': rj_spec.weight,
                            }
                        else:
                            column_tracker[col_key]['count'] += 1
                            column_tracker[col_key]['z_top'] = round(odz + panel_gap_in, 4)
                            column_tracker[col_key]['total_weight'] += rj_spec.weight

                        vertical_fill_count += 1
                        placed_vert = True
                        break
                    if placed_vert:
                        break
                    sx += x_step
                if placed_vert:
                    break

            if not placed_vert:
                still_rejected.append(rj)

        non_wall_rejected = [r for r in rejected
                             if r["panel_type"] != PanelType.WALL.value]
        rejected = non_wall_rejected + still_rejected

    if abs(base_offset) > 1e-9:
        for p in placed:
            x, y, z = p["pos"]
            p["pos"] = (x + base_offset, y + base_offset, z)

    axle_shift_in = 0.0
    axle_loads = compute_axle_loads(placed, axle_config) if axle_config else None

    if axle_optimize_shift and axle_config and placed:
        axle_shift_in, axle_loads = optimize_axle_shift(
            placed, axle_config, trailer_L, step_in=axle_shift_step
        )
        if abs(axle_shift_in) > 1e-9:
            for p in placed:
                x, y, z = p["pos"]
                p["pos"] = (x + axle_shift_in, y, z)

    for i, panel in enumerate(placed):
        sa, sp = calculate_support_area(panel["pos"], panel["size"], placed, gap=panel_gap_in)
        comp = calculate_compression_stress(panel["weight"], sa)
        bend = calculate_bending_stress(panel["weight"], panel["pos"], panel["size"], sp)
        wa = get_weight_above(i, placed, gap=panel_gap_in)
        shear = calculate_shear_stress(wa + panel["weight"], panel["size"])
        defl = calculate_deflection(panel["weight"], panel["pos"], panel["size"],
                                    stress_config.panel_youngs_modulus_psi, sp)

        dx, dy, dz = panel["size"]
        footprint = dx * dy
        total_load = panel["weight"] + wa

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

    wall_count = sum(1 for p in placed if p["panel_type"] == PanelType.WALL.value)
    floor_count = sum(1 for p in placed if p["panel_type"] == PanelType.FLOOR.value)
    roof_count = sum(1 for p in placed if p["panel_type"] == PanelType.ROOF.value)
    cg = compute_cg(placed)
    weight_dist = compute_weight_distribution(placed, trailer_L)

    max_layer = 0
    for state in column_tracker.values():
        max_layer = max(max_layer, state['count'])

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
            "placed_floors": floor_count,
            "placed_roofs": roof_count,
            "rejected_panels": len(rejected),
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
            "vertical_fill_count": vertical_fill_count,
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
                "short_edge": p.get("short_edge", 0),
                "spec_length": p.get("spec_length", 0),
                "layer": int(round(p["pos"][2] / max(p["size"][2] + panel_gap_in, 0.1))),
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


def optimize_best_configuration(panel_list, trailer_L, trailer_W, trailer_H,
                                max_horizontal, max_vertical,
                                stress_config, axle_config,
                                panel_gap_in=0.0, trailer_tol_in=0.0,
                                align_x="front", align_y="front",
                                axle_optimize_shift=False, axle_shift_step=1.0,
                                is_flatbed=False, overhang_max_in=12.0,
                                allow_vertical=False):
    best_out = None
    best_key = None

    for s in LoadingStrategy:
        if is_flatbed and s == LoadingStrategy.WALL_FIRST:
            continue
        out = pack_panels_v14(
            panel_list, trailer_L, trailer_W, trailer_H,
            max_horizontal, max_vertical,
            strategy=s, stress_config=stress_config, axle_config=axle_config,
            panel_gap_in=panel_gap_in, trailer_tol_in=trailer_tol_in,
            align_x=align_x, align_y=align_y,
            axle_optimize_shift=axle_optimize_shift, axle_shift_step=axle_shift_step,
            is_flatbed=is_flatbed, overhang_max_in=overhang_max_in,
            allow_vertical=allow_vertical
        )
        if "error" in out:
            continue

        placed = out.get("settings", {}).get("placed_panels", 0)
        axle_obj = out.get("settings", {}).get("axle_objective", float("inf"))

        if is_flatbed and len(panel_list) <= 36:
            # Low pod counts: prefer compact X-spread for balanced appearance
            placements = out.get("placements", [])
            if placements:
                xs = [p["position_vector"][0] for p in placements]
                x_spread = max(xs) - min(xs)
            else:
                x_spread = float("inf")
            key = (-placed, x_spread, axle_obj)
        else:
            key = (-placed, axle_obj)

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
            b0, b1 = (x, y, z), (x + dx, y, z)
            b2, b3 = (x + offset + short_edge, y + dy, z), (x + offset, y + dy, z)
        else:
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


def visualize(out, stress_config, color_by_stress, show_stress_markers, show_axles, floor_spec,
              show_tractor=False):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]
    fig = go.Figure()

    is_fb = out.get("settings", {}).get("is_flatbed", False)
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.2)
    if is_fb:
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
    roof_pal = ["#8e44ad", "#6c3483", "#a569bd", "#7d3c98"]
    wi, fi, ri = 0, 0, 0
    tc = {}

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
            elif pt == PanelType.ROOF.value:
                tc[key] = roof_pal[ri % len(roof_pal)]
                ri += 1
            else:
                tc[key] = wall_pal[wi % len(wall_pal)]
                wi += 1

        if color_by_stress:
            c = get_stress_color(
                p["stress_analysis"]["utilization_ratio"],
                1.0
            )
        elif show_stress_markers and p["loading_order"] == worst_idx:
            c = "#ffff00"
        else:
            c = tc[key]

        x, y, z = p["position_vector"]
        dx, dy, dz = p["size"]

        wire_width = 6 if (show_stress_markers and p["loading_order"] == worst_idx) else 2
        wire_opacity = 1.0 if (show_stress_markers and p["loading_order"] == worst_idx) else 0.9

        if p.get("is_trapezoid", False):
            am = tuple(p.get("axis_map", ["L", "H", "T"]))
            fl = p.get("flipped", False)
            p_short = p.get("short_edge", floor_spec.short_edge) or floor_spec.short_edge
            p_spec_len = p.get("spec_length", floor_spec.length) or floor_spec.length
            verts = make_half_hex_vertices(x, y, z, dx, dy, dz, am,
                                            p_short, p_spec_len, flipped=fl)
            add_half_hex_solid(fig, verts, c, opacity=0.45)
            add_half_hex_wire(fig, verts, c, "", opacity=wire_opacity, width=wire_width)
        else:
            add_solid(fig, x, y, z, dx, dy, dz, c, opacity=0.4)
            add_wire(fig, x, y, z, dx, dy, dz, c, "", opacity=wire_opacity, width=wire_width)

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

    axle_data = out.get("axle_loads")
    if show_axles and axle_data:
        ap = axle_data["axle_positions"]

        tire_radius = 12.0
        tire_width = 8.0
        rim_radius = 7.0
        rim_width = 5.0
        wheel_outset = 6.0
        axle_bar_radius = 1.6

        trailer_bottom_z = 0.0
        suspension_drop = 4.0
        wheel_center_z = trailer_bottom_z - (tire_radius + suspension_drop)
        axle_z = wheel_center_z

        def add_tire_pair(ax_x, color_axle, label_color, axle_name="", show_leg=False,
                          load_text=None, load_x=None, dual=False):
            add_cylinder_y(fig, ax_x, 0, W, axle_z, axle_bar_radius, color_axle,
                           opacity=0.80, name=axle_name, showlegend=show_leg)

            def draw_one_side(y_start, y_end):
                add_cylinder_y(fig, ax_x, y_start, y_end, axle_z, tire_radius, "#111111",
                               opacity=0.96, name="", showlegend=False)
                rim_y0 = y_start + (tire_width - rim_width) / 2.0
                rim_y1 = rim_y0 + rim_width
                add_cylinder_y(fig, ax_x, rim_y0, rim_y1, axle_z, rim_radius, "#bdc3c7",
                               opacity=0.95, name="", showlegend=False)

            if dual:
                gap = 1.0
                draw_one_side(-wheel_outset - tire_width, -wheel_outset)
                draw_one_side(-wheel_outset - 2 * tire_width - gap, -wheel_outset - tire_width - gap)
                draw_one_side(W + wheel_outset, W + wheel_outset + tire_width)
                draw_one_side(W + wheel_outset + tire_width + gap, W + wheel_outset + 2 * tire_width + gap)
            else:
                draw_one_side(-wheel_outset - tire_width, -wheel_outset)
                draw_one_side(W + wheel_outset, W + wheel_outset + tire_width)

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

        fig.add_trace(go.Scatter3d(
            x=[d_center], y=[W / 2], z=[axle_z + tire_radius + 2.0],
            mode="markers+text",
            marker=dict(size=6, color="#e74c3c", symbol="circle"),
            text=[f"Drive: {axle_data['drive_tandem_lb']:,.0f} lb"],
            textposition="top center",
            textfont=dict(color="#e74c3c", size=10),
            name="", showlegend=False
        ))

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

        fig.add_trace(go.Scatter3d(
            x=[t_center], y=[W / 2], z=[axle_z + tire_radius + 2.0],
            mode="markers+text",
            marker=dict(size=6, color="#3498db", symbol="circle"),
            text=[f"Trailer: {axle_data['trailer_tandem_lb']:,.0f} lb"],
            textposition="top center",
            textfont=dict(color="#3498db", size=10),
            name="", showlegend=False
        ))

    axle_data_for_tractor = out.get("axle_loads")
    if show_tractor and axle_data_for_tractor:
        ap_t = axle_data_for_tractor["axle_positions"]
        drive_cx = ap_t["drive_tandem_x"]
        wheelbase = ap_t.get("tractor_wheelbase", 240.0)
        steer_ax = drive_cx - wheelbase

        cab_width = 96.0
        hood_width = 80.0
        frame_height = 6.0
        hood_height = 50.0
        cab_height = 100.0
        roof_thickness = 8.0
        bumper_overhang = 24.0

        bumper_x = steer_ax - bumper_overhang
        windshield_x = steer_ax + 80.0
        cab_back_x = drive_cx + 30.0
        fifth_wheel_x = -24.0

        cab_y0 = (W - cab_width) / 2.0
        hood_y0 = (W - hood_width) / 2.0

        tractor_color = "#3d4f5f"
        hood_color = "#4a6274"
        frame_color = "#2c3e50"
        roof_color = "#344a5e"
        fifth_wh_color = "#555555"

        rail_width = 4.0
        rail_y_left = (W - 36.0) / 2.0
        rail_y_right = rail_y_left + 36.0
        frame_z0 = -frame_height
        frame_len = abs(bumper_x)
        add_solid(fig, bumper_x, rail_y_left, frame_z0, frame_len, rail_width, frame_height,
                  frame_color, opacity=0.2)
        add_solid(fig, bumper_x, rail_y_right - rail_width, frame_z0, frame_len, rail_width, frame_height,
                  frame_color, opacity=0.2)

        add_solid(fig, bumper_x, hood_y0, 0, windshield_x - bumper_x, hood_width, hood_height,
                  hood_color, opacity=0.18)
        add_wire(fig, bumper_x, hood_y0, 0, windshield_x - bumper_x, hood_width, hood_height,
                 hood_color, "", opacity=0.15, width=1)

        cab_len = cab_back_x - windshield_x
        add_solid(fig, windshield_x, cab_y0, 0, cab_len, cab_width, cab_height,
                  tractor_color, opacity=0.2)
        add_wire(fig, windshield_x, cab_y0, 0, cab_len, cab_width, cab_height,
                 tractor_color, "Tractor Cab", opacity=0.2, width=1)

        add_solid(fig, windshield_x, cab_y0, cab_height, cab_len, cab_width, roof_thickness,
                  roof_color, opacity=0.18)

        plate_w = 40.0
        plate_d = 24.0
        plate_y0 = (W - plate_w) / 2.0
        add_solid(fig, fifth_wheel_x, plate_y0, -2.0, plate_d, plate_w, 2.0,
                  fifth_wh_color, opacity=0.25)

        bumper_bar_h = 10.0
        add_solid(fig, bumper_x - 3.0, hood_y0 - 4.0, 8.0, 3.0, hood_width + 8.0, bumper_bar_h,
                  "#444444", opacity=0.22)

        stack_radius = 3.0
        stack_height = 40.0
        stack_x = cab_back_x - 6.0
        for sy in [cab_y0 - 6.0, cab_y0 + cab_width + 6.0 - stack_radius * 2]:
            add_solid(fig, stack_x, sy, cab_height, stack_radius * 2, stack_radius * 2, stack_height,
                      "#666666", opacity=0.2)

    if not color_by_stress:
        added = set()
        for key, c in tc.items():
            count = sum(1 for p in out["placements"] if f"{p['panel_type']}|{p['orientation']}" == key)
            pt, o = key.split("|", 1)
            short = "Floor" if "Floor" in pt else "Roof" if "Roof" in pt else "Wall"
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

    st.subheader("Structural Analysis Summary")
    if out["placements"]:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Panels", f"{out['settings']['placed_panels']} / {out['settings']['requested_panels']}")
        with c2:
            st.metric("W + F + R", f"{out['settings']['placed_walls']}W + {out['settings']['placed_floors']}F + {out['settings'].get('placed_roofs', 0)}R")
        with c3:
            tw = sum(p["weight"] for p in out["placements"])
            st.metric("Weight", f"{tw:,.0f} lb")
        with c4:
            st.metric("Layers", out['settings']['max_layers_used'])
        with c5:
            cg = out["weight_distribution"]["center_of_gravity"]
            st.metric("CG (X)", f"{cg['x']:.0f}\" / {tr['L']:.0f}\"")

        wd = out["weight_distribution"]["by_thirds"]
        st.subheader("Weight Distribution (Trailer Thirds)")
        wd_c1, wd_c2, wd_c3 = st.columns(3)
        with wd_c1:
            st.metric("Front", f"{wd['front']:,.0f} lb")
        with wd_c2:
            st.metric("Middle", f"{wd['middle']:,.0f} lb")
        with wd_c3:
            st.metric("Rear", f"{wd['rear']:,.0f} lb")

        if show_stress_markers and wsp:
            st.subheader("Worst-Case Stress Panel")
            ws1, ws2, ws3 = st.columns(3)
            with ws1:
                st.metric("Panel", wsp["label"])
                st.caption(f"{wsp['panel_type']} | {wsp['orientation']}")
            with ws2:
                st.metric("Total Load", f"{wsp['total_load_lb']:,.1f} lb")
                st.caption(f"Self: {wsp['own_weight_lb']:.0f} lb + Above: {wsp['weight_above_lb']:,.0f} lb")
            with ws3:
                st.metric("Compression", f"{wsp['compression_psi']:.2f} psi")
                st.metric("Shear", f"{wsp['shear_psi']:.2f} psi")
            st.caption(
                f"Position: ({wsp['position'][0]:.0f}\", {wsp['position'][1]:.0f}\", {wsp['position'][2]:.0f}\") "
                f"| Bending: {wsp['bending_stress_psi']:.2f} psi | Defl: {wsp['deflection_in']:.4f}\""
            )

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


st.set_page_config(page_title="MCLOS - Modular Cargo Loading Optimization Software", layout="wide",)

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HexHomesLogo.png")

def get_logo_b64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

_LOGO_B64 = get_logo_b64()

HONEY_CSS = """
<style>
.stApp {
    background-color: #242012;
    color: #F5F0E8;
}

.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
label, .stCaption, span, p, div {
    color: #F5F0E8 !important;
}

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

.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
    border-color: #D4A832 !important;
    color: #F0D68A !important;
    background-color: transparent !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
    background-color: #302818 !important;
    border-color: #E8A817 !important;
}

h1, h2, h3 {
    color: #E8A817 !important;
}

h2, h3 {
    color: #D4A832 !important;
}

[data-testid="stMetricLabel"] {
    color: #C0A870 !important;
}

[data-testid="stMetricValue"] {
    color: #F0D68A !important;
}

.stDownloadButton > button {
    background-color: #302818 !important;
    color: #F0D68A !important;
    border: 1px solid #D4A832 !important;
}
.stDownloadButton > button:hover {
    background-color: #3E3420 !important;
}

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
details[data-testid="stExpander"] > div {
    background-color: #1E1A12 !important;
}

[data-testid="stJson"],
.stJson, pre {
    background-color: #1A160E !important;
    color: #F0D68A !important;
    border: 1px solid #3E3420 !important;
    border-radius: 6px !important;
}
[data-testid="stJson"] span {
    color: #F0D68A !important;
}

div[data-testid="stAlert"] {
    background-color: #2A2418 !important;
    color: #F5F0E8 !important;
}

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
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus,
input:focus, select:focus, textarea:focus {
    border-color: #E8A817 !important;
    box-shadow: 0 0 0 1px #E8A817 !important;
}
.stNumberInput button {
    background-color: #3E3420 !important;
    color: #F0D68A !important;
    border-color: #4A3D28 !important;
}
.stNumberInput button:hover {
    background-color: #4A3D28 !important;
}

.stRadio > div > label > div:first-child {
    color: #E8A817 !important;
}
.stRadio > div > label {
    color: #F5F0E8 !important;
}

.stCheckbox > label > span > span {
    border-color: #D4A832 !important;
}

[data-testid="stPlotlyChart"] {
    border: 1px solid #3E3420;
    border-radius: 8px;
    padding: 4px;
    background-color: #161208 !important;
}
.modebar {
    background-color: transparent !important;
}
.modebar-btn path {
    fill: #D4A832 !important;
}

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

.stSpinner > div {
    border-top-color: #E8A817 !important;
}

hr {
    border-color: #3E3420 !important;
}

div[data-testid="stAlert"] p {
    color: #F5F0E8 !important;
}

.stCaption, [data-testid="stCaption"] {
    color: #C0A870 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #D4A832 !important;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: #E8A817 !important;
}

[data-testid="stTooltipIcon"] {
    color: #D4A832 !important;
}

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

if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if "started" not in st.session_state:
    st.session_state.started = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def render_welcome():
    if _LOGO_B64:
        logo_col, title_col = st.columns([1, 3])
        with logo_col:
            st.image(LOGO_PATH, width=180)
        with title_col:
            st.title("Modular Cargo Loading Optimization Software")
            st.caption("HexHomes Panel Packing Optimizer  |  Beta V4")
    else:
        st.title("Modular Cargo Loading Optimization Software")
        st.caption("HexHomes Panel Packing Optimizer  |  Beta V4")

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
            st.image(LOGO_PATH, width=200)
        st.header("MCLOS")
        st.write("Click start to enter the optimizer UI.")
        if st.button("Start MCLOS", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    render_welcome()
    st.stop()

st.title("MCLOS - Modular Cargo Loading Optimization Software")
st.caption("HexHomes Panel Packing Optimizer  |  Beta V4")

with st.sidebar:
    if _LOGO_B64:
        st.image(LOGO_PATH, width=200)
    if st.button("Back to Welcome", use_container_width=True):
        st.session_state.started = False
        st.rerun()

    st.header("Display")
    color_by_stress = st.checkbox("Color by Stress Level", value=False,
                                   help="Colors all panels green/yellow/red by worst utilization ratio")
    show_stress_markers = st.checkbox("Show Stress Markers", value=True,
                                      help="Highlights worst-case panel + shows stress detail card")
    show_axles = st.checkbox("Show Axle Positions", value=True,
                              help="Shows drive & trailer tandem axle lines + load analysis card")
    show_tractor = st.checkbox("Show Tractor Cab", value=True,
                                help="Shows a simple 3D model of the tractor/cab at the front of the trailer")

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

st.subheader("Trailer Selection")
tL = TRAILER_DIMS[TrailerPreset.FT53_FLATBED]["L"]
tW = TRAILER_DIMS[TrailerPreset.FT53_FLATBED]["W"]
tH = TRAILER_DIMS[TrailerPreset.FT53_FLATBED]["H"]
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

if tp == TrailerPreset.CUSTOM:
    is_flatbed = st.checkbox("Flatbed (open sides, allows floor panel overhang)", value=False, key="custom_fb")
else:
    is_flatbed = tp in (TrailerPreset.FT53_FLATBED, TrailerPreset.FT42_FLATBED)

if is_flatbed:
    overhang_max_in = st.number_input(
        "Max flatbed overhang per side (in)",
        value=12.0, min_value=0.0, max_value=24.0, step=1.0, key="overhang",
        help="Floor panels can extend past deck width on flatbed trailers (no walls). "
             "Max per-side overhang. 0 = no overhang allowed."
    )
    allow_vertical = st.checkbox(
        "Allow Vertical Panels (gap fill)",
        value=True,
        key="allow_vert",
        help="When enabled, wall panels may be placed vertically as a fallback"
             "to fill gaps that cannot fit flat panels. Currently only applies to the 42-ft flatbed trailer."
    )
else:
    overhang_max_in = 0.0
    allow_vertical = False

if tp in AXLE_CONFIGS:
    axle_config = AXLE_CONFIGS[tp]
else:
    trailer_length = tL

    axle_config = AxleConfig(
        drive_tandem_x=-36.0, drive_tandem_spacing=49.0,
        trailer_tandem_x=round(trailer_length * 0.75, 1), trailer_tandem_spacing=49.0,
    )

st.subheader("Panel Configuration")
num_pods = 0
manual_walls = 0
manual_floors = 0
manual_roofs = 0
input_mode = st.radio("Input method", ["By Pod Count (quick)", "Manual Panel Count"], horizontal=True)

if input_mode == "By Pod Count (quick)":
    cp1, cp2 = st.columns([1, 2])
    with cp1:
        num_pods = st.number_input("Pods", min_value=1, max_value=6, value=3, step=1,
                                   help="6 walls + 2 floors per pod")
    with cp2:
        tw = num_pods * 6
        tf = num_pods * 2
        tr_ = num_pods * 1
        st.markdown(f"**{num_pods} Pod{'s' if num_pods > 1 else ''}** = **{tw + tf + tr_} panels** ({tw}W + {tf}F + {tr_}R)")
    use_pods = True
else:
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        manual_walls = st.number_input("Walls", min_value=0, value=18, step=1, key="mw")
    with mc2:
        manual_floors = st.number_input("Floors", min_value=0, value=6, step=1, key="mf")
    with mc3:
        manual_roofs = st.number_input("Roofs", min_value=0, value=3, step=1, key="mr")
    with mc4:
        st.metric("Total", manual_walls + manual_floors + manual_roofs)
    use_pods = False

with st.expander("Panel Dimensions (edit if needed)"):
    pc1, pc2, pc3 = st.columns(3)
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
    with pc3:
        st.markdown("**Roof Panel (Half-Hex, folded)**")
        rL = st.number_input("Long Edge (in)", value=ROOF_PANEL_DEFAULT.length, min_value=1.0, key="rl")
        rH = st.number_input("Depth (in)", value=ROOF_PANEL_DEFAULT.height, min_value=1.0, key="rh")
        rT = st.number_input("Thickness (in)", value=ROOF_PANEL_DEFAULT.thickness, min_value=0.1, key="rt")
        rW = st.number_input("Weight (lb)", value=ROOF_PANEL_DEFAULT.weight, min_value=0.1, key="rw")
        rS = st.number_input("Short Edge (in)", value=ROOF_PANEL_DEFAULT.short_edge, min_value=1.0, key="rs")

wall_spec = PanelSpec(panel_type=PanelType.WALL, length=wL, height=wH, thickness=wT, weight=wW)
floor_spec = PanelSpec(panel_type=PanelType.FLOOR, length=fL, height=fH, thickness=fT, weight=fW, short_edge=fS)
roof_spec = PanelSpec(panel_type=PanelType.ROOF, length=rL, height=rH, thickness=rT, weight=rW, short_edge=rS)

if fS >= fL:
    st.warning("Floor short edge must be less than long edge. Check floor panel dimensions.")
if rS >= rL:
    st.warning("Roof short edge must be less than long edge. Check roof panel dimensions.")

st.markdown("---")
rc1, rc2 = st.columns([3, 1])
with rc1:
    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)
with rc2:
    st.caption(f"Run #{st.session_state.run_count}")

if run_btn:
    if use_pods:
        panel_list = build_panel_list_from_pods(num_pods, wall_spec, floor_spec, roof_spec)
    else:
        panel_list = build_panel_list_manual(manual_walls, manual_floors, manual_roofs, wall_spec, floor_spec, roof_spec)

    if not panel_list:
        st.error("No panels to optimize.")
    else:
        with st.spinner(f"Optimizing {len(panel_list)} panels..."):
            out = optimize_best_configuration(
                panel_list, tL, tW, tH,
                maxH, maxV,
                stress_config=stress_config, axle_config=axle_config,
                panel_gap_in=panel_gap_in, trailer_tol_in=trailer_tol_in,
                align_x=align_x, align_y=align_y,
                axle_optimize_shift=axle_opt_shift, axle_shift_step=axle_shift_step,
                is_flatbed=is_flatbed, overhang_max_in=overhang_max_in,
                allow_vertical=allow_vertical
            )
        st.session_state.run_count += 1
        st.session_state.last_result = out

out = st.session_state.last_result
if out is not None:
    if "error" not in out:
        pc = out["settings"]["placed_panels"]
        rc = out["settings"]["requested_panels"]
        sr = (pc / rc * 100) if rc > 0 else 0

        (st.success if sr == 100 else st.warning if pc > 0 else st.error)(
            f"{'All ' if sr == 100 else ''}{pc} / {rc} panels loaded ({sr:.0f}%)")
        st.subheader("3D Trailer View")
        visualize(out, stress_config, color_by_stress, show_stress_markers, show_axles, floor_spec,
                  show_tractor=show_tractor)

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
                               f"mclos_beta_v4_run{st.session_state.run_count}.json", "application/json", use_container_width=True)
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
                               f"mclos_beta_v4_run{st.session_state.run_count}.csv", "text/csv", use_container_width=True)
    else:
        st.error(out["error"])