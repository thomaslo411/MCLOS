import random
import json
import streamlit as st
import plotly.graph_objects as go
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

class LoadingStrategy(Enum):
    GRAVITY_LAYERED = "Gravity Layered"
    WALL_FIRST = "Wall First"
    ZONE_BASED = "Zone Based"
    STRESS_OPTIMIZED = "Stress Optimized"

@dataclass
class StressConfig:
    max_compression_psi: float = 50.0
    max_bending_moment_lbf_in: float = 10000.0
    max_shear_psi: float = 30.0
    safety_factor: float = 2.0
    panel_youngs_modulus_psi: float = 1800000.0

def aabb_intersect(p1, s1, p2, s2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return not (
        x1 + s1[0] <= x2 or x2 + s2[0] <= x1 or
        y1 + s1[1] <= y2 or y2 + s2[1] <= y1 or
        z1 + s1[2] <= z2 or z2 + s2[2] <= z1
    )

def in_bounds(pos, size, trailer):
    x, y, z = pos
    return (
        x >= 0 and y >= 0 and z >= 0 and
        x + size[0] <= trailer[0] and
        y + size[1] <= trailer[1] and
        z + size[2] <= trailer[2]
    )

def handling_ok(size, max_horizontal, max_vertical):
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical

def calculate_support_area(pos, size, placed, z_tol=1e-6):
    x, y, z = pos
    dx, dy, dz = size
    
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
                'centroid': (px + pdx/2, py + pdy/2, top)
            })
    
    return total_area, supporting_panels

def calculate_compression_stress(weight, support_area):
    if support_area < 1e-6:
        return float('inf')
    return weight / support_area

def calculate_bending_stress(panel_weight, size, supporting_panels):
    if not supporting_panels:
        return 0.0
    
    dx, dy, dz = size
    moment_of_inertia = (dy * dz**3) / 12.0
    
    max_moment = 0.0
    for support in supporting_panels:
        overhang_x = abs(support['centroid'][0] - dx/2)
        overhang_y = abs(support['centroid'][1] - dy/2)
        max_overhang = max(overhang_x, overhang_y)
        moment = panel_weight * max_overhang
        max_moment = max(max_moment, moment)
    
    if moment_of_inertia < 1e-6:
        return float('inf')
    
    return (max_moment * (dz / 2.0)) / moment_of_inertia

def calculate_shear_stress(total_weight_above, size):
    dx, dy, dz = size
    shear_area = min(dx, dy) * dz
    
    if shear_area < 1e-6:
        return float('inf')
    
    return total_weight_above / shear_area

def calculate_deflection(panel_weight, size, youngs_modulus, supporting_panels):
    if not supporting_panels:
        return 0.0
    
    dx, dy, dz = size
    moment_of_inertia = (dy * dz**3) / 12.0
    
    if moment_of_inertia < 1e-6:
        return float('inf')
    
    max_overhang = 0.0
    for support in supporting_panels:
        overhang = abs(support['centroid'][0] - dx/2)
        max_overhang = max(max_overhang, overhang)
    
    if max_overhang < 1e-6:
        return 0.0
    
    deflection = (panel_weight * max_overhang**3) / (3.0 * youngs_modulus * moment_of_inertia)
    return deflection

def get_weight_above(panel_id, placed):
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
    x, y, z = pos
    dx, dy, dz = size
    panel_area = dx * dy
    
    support_area, supporting_panels = calculate_support_area(pos, size, placed, z_tol)
    
    if support_area < panel_area * min_support_frac:
        return False, "Insufficient support area"
    
    compression_stress = calculate_compression_stress(weight, support_area)
    if compression_stress > stress_config.max_compression_psi / stress_config.safety_factor:
        return False, f"Compression stress too high: {compression_stress:.1f} psi"
    
    bending_stress = calculate_bending_stress(weight, size, supporting_panels)
    max_bending = stress_config.max_bending_moment_lbf_in / stress_config.safety_factor
    if bending_stress > max_bending:
        return False, f"Bending stress too high: {bending_stress:.1f}"
    
    total_weight_above = weight
    shear_stress = calculate_shear_stress(total_weight_above, size)
    if shear_stress > stress_config.max_shear_psi / stress_config.safety_factor:
        return False, f"Shear stress too high: {shear_stress:.1f} psi"
    
    deflection = calculate_deflection(weight, size, stress_config.panel_youngs_modulus_psi, supporting_panels)
    max_deflection = min(dx, dy) / 360.0
    if deflection > max_deflection:
        return False, f"Deflection too large: {deflection:.3f} in"
    
    return True, "OK"

def generate_orientations(L, H, T):
    return [
        ("flat_LxH", (L, H, T)),
        ("flat_HxL", (H, L, T)),
        ("stand_LxT", (L, T, H)),
        ("stand_TxL", (T, L, H)),
        ("stand_HxT", (H, T, L)),
        ("stand_TxH", (T, H, L)),
    ]

def find_lowest_position(x, y, size, weight, placed, trailer, stress_config, z_tol=1e-6, step=1.0):
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

def generate_positions_gravity_layered(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    
    positions = []
    for x in xs:
        for y in ys:
            positions.append((x, y))
    
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
        
        zone_positions = []
        for x in xs:
            if zone_start <= x < zone_end:
                for y in ys:
                    zone_positions.append((x, y))
        
        zone_positions.sort(key=lambda p: (p[0], p[1]))
        positions.extend(zone_positions)
    
    return positions

def generate_positions_stress_optimized(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    
    positions = []
    center_x = trailer[0] / 2
    center_y = trailer[1] / 2
    
    for x in xs:
        for y in ys:
            dist_to_center = ((x - center_x)**2 + (y - center_y)**2)**0.5
            positions.append((x, y, dist_to_center))
    
    positions.sort(key=lambda p: p[2])
    return [(x, y) for x, y, _ in positions]

def pack_panels(panel_L, panel_H, panel_T, panel_W,
                trailer_L, trailer_W, trailer_H,
                max_horizontal, max_vertical,
                num_panels, step, seed, strategy, stress_config):

    random.seed(seed)
    trailer = (trailer_L, trailer_W, trailer_H)

    orientations = []
    for name, size in generate_orientations(panel_L, panel_H, panel_T):
        if in_bounds((0,0,0), size, trailer):
            orientations.append((name, size))

    if not orientations:
        return {"error": "No orientations fit inside trailer."}

    placed = []
    rejection_reasons = {}
    
    if strategy == LoadingStrategy.GRAVITY_LAYERED:
        xy_positions = generate_positions_gravity_layered(trailer, step)
    elif strategy == LoadingStrategy.WALL_FIRST:
        xy_positions = generate_positions_wall_first(trailer, step)
    elif strategy == LoadingStrategy.ZONE_BASED:
        xy_positions = generate_positions_zone_based(trailer, step)
    else:
        xy_positions = generate_positions_stress_optimized(trailer, step)

    for pid in range(num_panels):
        placed_this = False
        random.shuffle(orientations)

        for name, size in orientations:
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            for x, y in xy_positions:
                final_pos = find_lowest_position(x, y, size, panel_W, placed, trailer, stress_config, step=step)
                
                if final_pos is None:
                    continue

                placed.append({
                    "id": pid,
                    "pos": final_pos,
                    "size": size,
                    "orientation": name,
                    "weight": panel_W
                })
                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            break

    for i, panel in enumerate(placed):
        support_area, supporting_panels = calculate_support_area(panel["pos"], panel["size"], placed[:i])
        compression = calculate_compression_stress(panel["weight"], support_area)
        bending = calculate_bending_stress(panel["weight"], panel["size"], supporting_panels)
        weight_above = get_weight_above(i, placed)
        shear = calculate_shear_stress(weight_above + panel["weight"], panel["size"])
        deflection = calculate_deflection(panel["weight"], panel["size"], stress_config.panel_youngs_modulus_psi, supporting_panels)
        
        panel["stress_analysis"] = {
            "compression_psi": round(compression, 2),
            "bending_stress": round(bending, 2),
            "shear_psi": round(shear, 2),
            "deflection_in": round(deflection, 4),
            "weight_above_lb": round(weight_above, 2),
            "support_area_sqin": round(support_area, 2)
        }

    return {
        "inputs": {
            "panel": {"L": panel_L, "H": panel_H, "T": panel_T, "W": panel_W},
            "trailer": {"L": trailer_L, "W": trailer_W, "H": trailer_H},
            "stress_limits": {
                "max_compression_psi": stress_config.max_compression_psi,
                "max_bending_lbf_in": stress_config.max_bending_moment_lbf_in,
                "max_shear_psi": stress_config.max_shear_psi,
                "safety_factor": stress_config.safety_factor
            }
        },
        "settings": {
            "requested_panels": int(num_panels),
            "placed_panels": len(placed),
            "grid_step_in": float(step),
            "seed": int(seed),
            "min_support_frac": 0.7,
            "packing_strategy": strategy.value
        },
        "placements": [
            {
                "id": p["id"],
                "position_vector": [round(p["pos"][0], 3), round(p["pos"][1], 3), round(p["pos"][2], 3)],
                "size": [round(p["size"][0], 3), round(p["size"][1], 3), round(p["size"][2], 3)],
                "orientation": p["orientation"],
                "weight": p["weight"],
                "layer": int(round(p["pos"][2] / step)),
                "stress_analysis": p["stress_analysis"]
            }
            for p in placed
        ]
    }

def box_edges(x, y, z, dx, dy, dz):
    p = [
        (x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z),
        (x,y,z+dz),(x+dx,y,z+dz),(x+dx,y+dy,z+dz),(x,y+dy,z+dz)
    ]
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a,b in e:
        xs += [p[a][0], p[b][0], None]
        ys += [p[a][1], p[b][1], None]
        zs += [p[a][2], p[b][2], None]
    return xs, ys, zs

def add_wire(fig, x,y,z,dx,dy,dz,color,name,opacity=1.0,width=4):
    xs,ys,zs = box_edges(x,y,z,dx,dy,dz)
    fig.add_trace(go.Scatter3d(
        x=xs,y=ys,z=zs,
        mode="lines",
        line=dict(color=color,width=width),
        opacity=opacity,
        name=name,
        showlegend=(name != "")
    ))

def add_solid(fig, x,y,z,dx,dy,dz,color,opacity=0.35):
    vx=[x,x+dx,x+dx,x,x,x+dx,x+dx,x]
    vy=[y,y,y+dy,y+dy,y,y,y+dy,y+dy]
    vz=[z,z,z,z,z+dz,z+dz,z+dz,z+dz]
    faces=[(0,1,2),(0,2,3),(4,5,6),(4,6,7),
           (0,1,5),(0,5,4),(1,2,6),(1,6,5),
           (2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    fig.add_trace(go.Mesh3d(
        x=vx,y=vy,z=vz,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        color=color,opacity=opacity,
        showlegend=False
    ))

def get_stress_color(stress_val, max_val):
    ratio = min(stress_val / max_val, 1.0)
    if ratio < 0.5:
        return "#2ecc71"
    elif ratio < 0.75:
        return "#f1c40f"
    else:
        return "#e74c3c"

def visualize(out, stress_config, color_by_stress=False):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]

    fig = go.Figure()
    
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.2)
    add_wire(fig, 0, 0, 0, L, W, H, "#ecf0f1", "", opacity=0.3, width=2)
    
    step_viz = 50
    for i in range(0, int(L), step_viz):
        add_wire(fig, i, 0, 0, 0, W, 0, "#34495e", "", opacity=0.15, width=1)
    for j in range(0, int(W), step_viz):
        add_wire(fig, 0, j, 0, L, 0, 0, "#34495e", "", opacity=0.15, width=1)

    palette = ["#2ecc71","#3498db","#e74c3c","#f1c40f","#9b59b6","#1abc9c","#e67e22","#95a5a6"]
    cmap = {}

    for p in out["placements"]:
        o = p["orientation"]
        if o not in cmap:
            cmap[o] = palette[len(cmap) % len(palette)]
        
        if color_by_stress:
            stress = p["stress_analysis"]["compression_psi"]
            c = get_stress_color(stress, stress_config.max_compression_psi)
        else:
            c = cmap[o]

        x,y,z = p["position_vector"]
        dx,dy,dz = p["size"]

        add_solid(fig, x,y,z,dx,dy,dz,c,opacity=0.4)
        add_wire(fig, x,y,z,dx,dy,dz,c,"",opacity=0.9,width=2)

    if not color_by_stress:
        for o, c in cmap.items():
            count = sum(1 for p in out["placements"] if p["orientation"] == o)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="lines",
                line=dict(color=c, width=6),
                name=f"{o} ({count})"
            ))

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="Length (in)",
            yaxis_title="Width (in)",
            zaxis_title="Height (in)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0))
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Structural Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Panels Loaded", out["settings"]["placed_panels"])
    with col2:
        total_weight = sum(p["weight"] for p in out["placements"])
        st.metric("Total Weight", f"{total_weight:,.0f} lb")
    with col3:
        max_compression = max((p["stress_analysis"]["compression_psi"] for p in out["placements"]), default=0)
        st.metric("Max Compression", f"{max_compression:.1f} psi")
    with col4:
        max_deflection = max((p["stress_analysis"]["deflection_in"] for p in out["placements"]), default=0)
        st.metric("Max Deflection", f"{max_deflection:.4f} in")

st.set_page_config(page_title="Modular Cargo Loading Optimizaiton Software", layout="wide")
st.title("Modular Cargo Loading Optimizaiton Software")

with st.sidebar:
    st.header("Loading Strategy")
    strategy_name = st.selectbox(
        "Strategy",
        [s.value for s in LoadingStrategy],
        index=3
    )
    strategy = LoadingStrategy([s for s in LoadingStrategy if s.value == strategy_name][0])
    
    st.markdown("---")
    st.header("Structural Limits")
    max_comp = st.number_input("Max Compression (psi)", value=50.0, min_value=1.0)
    max_bend = st.number_input("Max Bending (lbf·in)", value=10000.0, min_value=1.0)
    max_shear = st.number_input("Max Shear (psi)", value=30.0, min_value=1.0)
    safety = st.number_input("Safety Factor", value=2.0, min_value=1.0)
    youngs = st.number_input("Young's Modulus (psi)", value=1800000.0, min_value=1000.0)
    
    color_by_stress = st.checkbox("Color by Stress Level", value=False)

stress_config = StressConfig(
    max_compression_psi=max_comp,
    max_bending_moment_lbf_in=max_bend,
    max_shear_psi=max_shear,
    safety_factor=safety,
    panel_youngs_modulus_psi=youngs
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Panel Dimensions")
    pL = st.number_input("Length (in)", value=112.0, min_value=1.0)
    pH = st.number_input("Height (in)", value=97.25, min_value=1.0)
    pT = st.number_input("Thickness (in)", value=5.5, min_value=0.1)
    pW = st.number_input("Weight (lb)", value=220.0, min_value=0.1)

with c2:
    st.subheader("Trailer Dimensions")
    tL = st.number_input("Trailer Length (in)", value=630.0, min_value=1.0)
    tW = st.number_input("Trailer Width (in)", value=102.0, min_value=1.0)
    tH = st.number_input("Trailer Height (in)", value=162.0, min_value=1.0)

st.subheader("Optimization Settings")
col1, col2, col3 = st.columns(3)

with col1:
    maxH = st.number_input("Max Horizontal (in)", value=145.0, min_value=1.0)
    maxV = st.number_input("Max Vertical (in)", value=114.0, min_value=1.0)

with col2:
    num = st.number_input("Target Panels", min_value=1, value=24, step=1)
    step = st.number_input("Grid Step (in)", min_value=0.5, value=2.0, step=0.5)

with col3:
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

if st.button("Run Optimization", type="primary", use_container_width=True):
    with st.spinner(f"Optimizing with stress analysis..."):
        out = pack_panels(pL, pH, pT, pW, tL, tW, tH, maxH, maxV, num, step, seed, strategy, stress_config)
    
    if "error" not in out:
        success_rate = (out["settings"]["placed_panels"] / out["settings"]["requested_panels"]) * 100
        if success_rate == 100:
            st.success(f"Successfully loaded all {out['settings']['placed_panels']} panels!")
        else:
            st.warning(f"Loaded {out['settings']['placed_panels']} of {out['settings']['requested_panels']} panels ({success_rate:.1f}%)")
    
    st.subheader("3D Visualization")
    visualize(out, stress_config, color_by_stress)

    with st.expander("Detailed Stress Analysis"):
        stress_data = []
        for p in out.get("placements", []):
            sa = p["stress_analysis"]
            stress_data.append({
                "Panel": p["id"],
                "Compression (psi)": sa["compression_psi"],
                "Bending Stress": sa["bending_stress"],
                "Shear (psi)": sa["shear_psi"],
                "Deflection (in)": sa["deflection_in"],
                "Weight Above (lb)": sa["weight_above_lb"],
                "Support Area (in²)": sa["support_area_sqin"]
            })
        st.dataframe(stress_data, use_container_width=True)

    with st.expander("View JSON Output"):
        st.json(out)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download JSON",
            data=json.dumps(out, indent=2),
            file_name="stress_optimized_packing.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        csv_header = "id,x,y,z,length,width,height,orientation,weight,compression_psi,bending,shear_psi,deflection_in\n"
        csv_rows = "\n".join([
            f"{p['id']},{p['position_vector'][0]},{p['position_vector'][1]},{p['position_vector'][2]},"
            f"{p['size'][0]},{p['size'][1]},{p['size'][2]},{p['orientation']},{p['weight']},"
            f"{p['stress_analysis']['compression_psi']},{p['stress_analysis']['bending_stress']},"
            f"{p['stress_analysis']['shear_psi']},{p['stress_analysis']['deflection_in']}"
            for p in out.get("placements", [])
        ])
        st.download_button(
            "Download CSV",
            data=csv_header + csv_rows,
            file_name="stress_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
