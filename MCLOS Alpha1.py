import json
import random
from dataclasses import dataclass
from enum import Enum

import plotly.graph_objects as go
import streamlit as st



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
    panel_youngs_modulus_psi: float = 1_800_000.0


def aabb_intersect(p1, s1, p2, s2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return not (
        x1 + s1[0] <= x2
        or x2 + s2[0] <= x1
        or y1 + s1[1] <= y2
        or y2 + s2[1] <= y1
        or z1 + s1[2] <= z2
        or z2 + s2[2] <= z1
    )


def in_bounds(pos, size, trailer):
    x, y, z = pos
    dx, dy, dz = size
    L, W, H = trailer
    return x >= 0 and y >= 0 and z >= 0 and x + dx <= L and y + dy <= W and z + dz <= H


def handling_ok(size, max_horizontal, max_vertical):
    return max(size[0], size[1]) <= max_horizontal and size[2] <= max_vertical


def generate_orientations(L, H, T):
    return [
        ("flat_LxH", (L, H, T)),
        ("flat_HxL", (H, L, T)),
        ("stand_LxT", (L, T, H)),
        ("stand_TxL", (T, L, H)),
        ("stand_HxT", (H, T, L)),
        ("stand_TxH", (T, H, L)),
    ]




def _zkey(z: float) -> float:
    return round(float(z), 6)


def calculate_support_area(pos, size, by_top, z_tol=1e-6):
    """Support area for a panel at pos/size.

    by_top: dict[z_top -> list[panel_record]]
    returns: (support_area, supporting_panels)
    supporting_panels entries match original Alpha structure.
    """

    x, y, z = pos
    dx, dy, _ = size

    if z <= z_tol:
        return dx * dy, []

    total_area = 0.0
    supporting = []
    for p in by_top.get(_zkey(z), []):
        px, py, _pz = p["pos"]
        pdx, pdy, pdz = p["size"]
        top = _zkey(_pz + pdz)
        if abs(top - _zkey(z)) > z_tol:
            continue

        ox = max(0.0, min(x + dx, px + pdx) - max(x, px))
        oy = max(0.0, min(y + dy, py + pdy) - max(y, py))
        overlap_area = ox * oy
        if overlap_area > 1e-6:
            total_area += overlap_area
            supporting.append(
                {
                    "panel": p,
                    "overlap_area": overlap_area,
                    "centroid": (px + pdx / 2, py + pdy / 2, float(top)),
                }
            )

    return total_area, supporting


def calculate_compression_stress(weight, support_area):
    return float("inf") if support_area < 1e-6 else float(weight) / float(support_area)


def calculate_bending_stress(panel_weight, size, supporting_panels):
    if not supporting_panels:
        return 0.0

    dx, dy, dz = size
    I = (dy * dz**3) / 12.0
    if I < 1e-6:
        return float("inf")

    max_moment = 0.0
    for s in supporting_panels:
        overhang_x = abs(s["centroid"][0] - dx / 2)
        overhang_y = abs(s["centroid"][1] - dy / 2)
        max_moment = max(max_moment, panel_weight * max(overhang_x, overhang_y))

    return (max_moment * (dz / 2.0)) / I


def calculate_shear_stress(total_weight_above, size):
    dx, dy, dz = size
    shear_area = min(dx, dy) * dz
    return float("inf") if shear_area < 1e-6 else float(total_weight_above) / float(shear_area)


def calculate_deflection(panel_weight, size, youngs_modulus, supporting_panels):
    if not supporting_panels:
        return 0.0

    dx, dy, dz = size
    I = (dy * dz**3) / 12.0
    if I < 1e-6:
        return float("inf")

    max_overhang = 0.0
    for s in supporting_panels:
        max_overhang = max(max_overhang, abs(s["centroid"][0] - dx / 2))

    if max_overhang < 1e-6:
        return 0.0

    return (panel_weight * max_overhang**3) / (3.0 * youngs_modulus * I)


def stress_ok(pos, size, weight, by_top, stress_config, z_tol=1e-6, min_support_frac=0.7):
    dx, dy, _dz = size
    panel_area = dx * dy

    support_area, supporting_panels = calculate_support_area(pos, size, by_top, z_tol)
    if support_area < panel_area * min_support_frac:
        return False, "Insufficient support area"

    comp = calculate_compression_stress(weight, support_area)
    if comp > stress_config.max_compression_psi / stress_config.safety_factor:
        return False, f"Compression stress too high: {comp:.1f} psi"

    bend = calculate_bending_stress(weight, size, supporting_panels)
    max_bend = stress_config.max_bending_moment_lbf_in / stress_config.safety_factor
    if bend > max_bend:
        return False, f"Bending stress too high: {bend:.1f}"

    shear = calculate_shear_stress(weight, size)
    if shear > stress_config.max_shear_psi / stress_config.safety_factor:
        return False, f"Shear stress too high: {shear:.1f} psi"

    defl = calculate_deflection(weight, size, stress_config.panel_youngs_modulus_psi, supporting_panels)
    max_defl = min(dx, dy) / 360.0
    if defl > max_defl:
        return False, f"Deflection too large: {defl:.3f} in"

    return True, "OK"


def weight_above(panel, placed, z_tol=1e-6):
    px, py, pz = panel["pos"]
    pdx, pdy, pdz = panel["size"]
    top = pz + pdz

    total = 0.0
    for other in placed:
        if other is panel:
            continue
        ox, oy, oz = other["pos"]
        odx, ody, _odz = other["size"]
        if oz < top - z_tol:
            continue
        x_overlap = max(0.0, min(px + pdx, ox + odx) - max(px, ox))
        y_overlap = max(0.0, min(py + pdy, oy + ody) - max(py, oy))
        if x_overlap > 1e-6 and y_overlap > 1e-6:
            total += other["weight"]
    return total



def _ptkey(p, q):
    x, y, z = p
    q = max(float(q), 1e-6)
    return (round(x / q), round(y / q), round(z / q))


def pack_panels(
    panel_L,
    panel_H,
    panel_T,
    panel_W,
    trailer_L,
    trailer_W,
    trailer_H,
    max_horizontal,
    max_vertical,
    num_panels,
    step,
    seed,
    strategy,
    stress_config,
    *,
    min_support_frac=0.7,
):
    """Greedy packer using Extreme-Points candidate set.

    step is used as *candidate quantization* (dedup key) and for the reported "layer".
    """

    random.seed(int(seed))
    trailer = (float(trailer_L), float(trailer_W), float(trailer_H))

    opts = [
        (name, size)
        for name, size in generate_orientations(float(panel_L), float(panel_H), float(panel_T))
        if in_bounds((0.0, 0.0, 0.0), size, trailer) and handling_ok(size, max_horizontal, max_vertical)
    ]
    if not opts:
        return {"error": "No orientations fit inside trailer (bounds/handling)."}

    placed = []
    by_top = {}  


    cand = {(0.0, 0.0, 0.0)}
    cand_keys = {_ptkey((0.0, 0.0, 0.0), step)}

    def add_cand(p):
        if not (0.0 <= p[0] <= trailer[0] and 0.0 <= p[1] <= trailer[1] and 0.0 <= p[2] <= trailer[2]):
            return
        k = _ptkey(p, step)
        if k in cand_keys:
            return
        cand_keys.add(k)
        cand.add(p)

    def prune_cand():
        dead = []
        eps = 1e-9
        for p in cand:
            x, y, z = p
            if x < 0 or y < 0 or z < 0 or x > trailer[0] or y > trailer[1] or z > trailer[2]:
                dead.append(p)
                continue
            for b in placed:
                bx, by, bz = b["pos"]
                bdx, bdy, bdz = b["size"]
                inside = (
                    (bx + eps) < x < (bx + bdx - eps)
                    and (by + eps) < y < (by + bdy - eps)
                    and (bz + eps) < z < (bz + bdz - eps)
                )
                if inside:
                    dead.append(p)
                    break
        for p in dead:
            cand.discard(p)
            cand_keys.discard(_ptkey(p, step))

    def cand_key(p):
        x, y, z = p
        L, W, _H = trailer
        if strategy == LoadingStrategy.GRAVITY_LAYERED:
            return (z, x, y)
        if strategy == LoadingStrategy.WALL_FIRST:
            dist_wall = min(x, y, max(0.0, L - x), max(0.0, W - y))
            return (z, dist_wall, x + y)
        if strategy == LoadingStrategy.ZONE_BASED:
            zones = 3
            zone_w = L / zones if L > 1e-6 else 1.0
            zone = min(zones - 1, max(0, int(x / zone_w)))
            return (zone, z, x, y)
        cx, cy = L / 2.0, W / 2.0
        dist_center = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        return (z, dist_center, x, y)

    def fits_at(pos, size):
        if not in_bounds(pos, size, trailer):
            return False
        ok, _msg = stress_ok(pos, size, panel_W, by_top, stress_config, min_support_frac=min_support_frac)
        if not ok:
            return False
        return not any(aabb_intersect(pos, size, p["pos"], p["size"]) for p in placed)

    for pid in range(int(num_panels)):
        random.shuffle(opts)
        placed_this = False

        for pos in sorted(cand, key=cand_key):
            for oname, size in opts:
                if not fits_at(pos, size):
                    continue

                x, y, z = pos
                dx, dy, dz = size

                rec = {
                    "id": pid,
                    "pos": (float(x), float(y), float(z)),
                    "size": (float(dx), float(dy), float(dz)),
                    "orientation": oname,
                    "weight": float(panel_W),
                }
                placed.append(rec)
                by_top.setdefault(_zkey(z + dz), []).append(rec)

        
                add_cand((x + dx, y, z))
                add_cand((x, y + dy, z))
                add_cand((x, y, z + dz))
                add_cand((x + dx, y + dy, z))
                add_cand((x + dx, y, z + dz))
                add_cand((x, y + dy, z + dz))

                placed_this = True
                break
            if placed_this:
                break

        if not placed_this:
            break
        prune_cand()


    by_top_all = {}
    for p in placed:
        x, y, z = p["pos"]
        dx, dy, dz = p["size"]
        by_top_all.setdefault(_zkey(z + dz), []).append(p)

    for p in placed:
        support_area, supports = calculate_support_area(p["pos"], p["size"], by_top_all)
        comp = calculate_compression_stress(p["weight"], support_area)
        bend = calculate_bending_stress(p["weight"], p["size"], supports)
        wab = weight_above(p, placed)
        shear = calculate_shear_stress(wab + p["weight"], p["size"])
        defl = calculate_deflection(p["weight"], p["size"], stress_config.panel_youngs_modulus_psi, supports)

        p["stress_analysis"] = {
            "compression_psi": round(comp, 2),
            "bending_stress": round(bend, 2),
            "shear_psi": round(shear, 2),
            "deflection_in": round(defl, 4),
            "weight_above_lb": round(wab, 2),
            "support_area_sqin": round(support_area, 2),
        }

    return {
        "inputs": {
            "panel": {"L": panel_L, "H": panel_H, "T": panel_T, "W": panel_W},
            "trailer": {"L": trailer_L, "W": trailer_W, "H": trailer_H},
            "stress_limits": {
                "max_compression_psi": stress_config.max_compression_psi,
                "max_bending_lbf_in": stress_config.max_bending_moment_lbf_in,
                "max_shear_psi": stress_config.max_shear_psi,
                "safety_factor": stress_config.safety_factor,
            },
        },
        "settings": {
            "requested_panels": int(num_panels),
            "placed_panels": len(placed),
            "grid_step_in": float(step),
            "seed": int(seed),
            "min_support_frac": float(min_support_frac),
            "packing_strategy": strategy.value,
        },
        "placements": [
            {
                "id": p["id"],
                "position_vector": [round(p["pos"][0], 3), round(p["pos"][1], 3), round(p["pos"][2], 3)],
                "size": [round(p["size"][0], 3), round(p["size"][1], 3), round(p["size"][2], 3)],
                "orientation": p["orientation"],
                "weight": p["weight"],
                "layer": int(round(p["pos"][2] / float(step))) if float(step) > 1e-9 else 0,
                "stress_analysis": p["stress_analysis"],
            }
            for p in placed
        ],
    }




def box_edges(x, y, z, dx, dy, dz):
    p = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in e:
        xs += [p[a][0], p[b][0], None]
        ys += [p[a][1], p[b][1], None]
        zs += [p[a][2], p[b][2], None]
    return xs, ys, zs


def add_wire(fig, x, y, z, dx, dy, dz, color, name="", opacity=1.0, width=3):
    xs, ys, zs = box_edges(x, y, z, dx, dy, dz)
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=name,
            showlegend=bool(name),
        )
    )


def add_solid(fig, x, y, z, dx, dy, dz, color, opacity=0.35):
    vx = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
    vy = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
    vz = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    fig.add_trace(
        go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color=color,
            opacity=opacity,
            showlegend=False,
        )
    )


def get_stress_color(stress_val, max_val):
    if max_val <= 1e-9:
        return "#95a5a6"
    ratio = min(float(stress_val) / float(max_val), 1.0)
    if ratio < 0.5:
        return "#2ecc71"
    if ratio < 0.75:
        return "#f1c40f"
    return "#e74c3c"


def visualize(out, stress_config, color_by_stress=False):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]

    fig = go.Figure()
    add_solid(fig, 0, 0, 0, L, W, 0.5, "#2c3e50", opacity=0.18)
    add_wire(fig, 0, 0, 0, L, W, H, "#ecf0f1", opacity=0.35, width=2)

    palette = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
    cmap = {}

    for p in out.get("placements", []):
        o = p["orientation"]
        cmap.setdefault(o, palette[len(cmap) % len(palette)])
        c = (
            get_stress_color(p["stress_analysis"]["compression_psi"], stress_config.max_compression_psi)
            if color_by_stress
            else cmap[o]
        )

        x, y, z = p["position_vector"]
        dx, dy, dz = p["size"]
        add_solid(fig, x, y, z, dx, dy, dz, c, opacity=0.4)
        add_wire(fig, x, y, z, dx, dy, dz, c, opacity=0.9, width=2)

    if not color_by_stress:
        for o, c in cmap.items():
            count = sum(1 for p in out.get("placements", []) if p["orientation"] == o)
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="lines",
                    line=dict(color=c, width=6),
                    name=f"{o} ({count})",
                )
            )

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="Length (in)",
            yaxis_title="Width (in)",
            zaxis_title="Height (in)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Structural Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Panels Loaded", out["settings"]["placed_panels"])
    with c2:
        total_w = sum(p["weight"] for p in out.get("placements", []))
        st.metric("Total Weight", f"{total_w:,.0f} lb")
    with c3:
        max_comp = max((p["stress_analysis"]["compression_psi"] for p in out.get("placements", [])), default=0)
        st.metric("Max Compression", f"{max_comp:.1f} psi")
    with c4:
        max_defl = max((p["stress_analysis"]["deflection_in"] for p in out.get("placements", [])), default=0)
        st.metric("Max Deflection", f"{max_defl:.4f} in")



st.set_page_config(page_title="Modular Cargo Loading Optimization Software", layout="wide")
st.title("Modular Cargo Loading Optimization Software")

with st.sidebar:
    st.header("Loading Strategy")
    strategy_name = st.selectbox("Strategy", [s.value for s in LoadingStrategy], index=3)
    strategy = LoadingStrategy([s for s in LoadingStrategy if s.value == strategy_name][0])

    st.markdown("---")
    st.header("Structural Limits")
    max_comp = st.number_input("Max Compression (psi)", value=50.0, min_value=1.0)
    max_bend = st.number_input("Max Bending (lbf·in)", value=10000.0, min_value=1.0)
    max_shear = st.number_input("Max Shear (psi)", value=30.0, min_value=1.0)
    safety = st.number_input("Safety Factor", value=2.0, min_value=1.0)
    youngs = st.number_input("Young's Modulus (psi)", value=1_800_000.0, min_value=1000.0)

    color_by_stress = st.checkbox("Color by Stress Level", value=False)

stress_config = StressConfig(
    max_compression_psi=float(max_comp),
    max_bending_moment_lbf_in=float(max_bend),
    max_shear_psi=float(max_shear),
    safety_factor=float(safety),
    panel_youngs_modulus_psi=float(youngs),
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
    st.caption("Optimization note: Grid Step is used as candidate-point quantization in this optimized build.")

with col3:
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

if st.button("Run Optimization", type="primary", use_container_width=True):
    with st.spinner("Optimizing with stress analysis (fast EP search)..."):
        out = pack_panels(pL, pH, pT, pW, tL, tW, tH, maxH, maxV, num, step, seed, strategy, stress_config)

    if "error" not in out:
        placed_n = out["settings"]["placed_panels"]
        req_n = out["settings"]["requested_panels"]
        rate = (placed_n / req_n) * 100 if req_n else 0
        if rate >= 99.999:
            st.success(f"Successfully loaded all {placed_n} panels!")
        else:
            st.warning(f"Loaded {placed_n} of {req_n} panels ({rate:.1f}%)")

    st.subheader("3D Visualization")
    visualize(out, stress_config, color_by_stress)

    with st.expander("Detailed Stress Analysis"):
        rows = []
        for p in out.get("placements", []):
            sa = p["stress_analysis"]
            rows.append(
                {
                    "Panel": p["id"],
                    "Compression (psi)": sa["compression_psi"],
                    "Bending Stress": sa["bending_stress"],
                    "Shear (psi)": sa["shear_psi"],
                    "Deflection (in)": sa["deflection_in"],
                    "Weight Above (lb)": sa["weight_above_lb"],
                    "Support Area (in²)": sa["support_area_sqin"],
                }
            )
        st.dataframe(rows, use_container_width=True)

    with st.expander("View JSON Output"):
        st.json(out)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            "Download JSON",
            data=json.dumps(out, indent=2),
            file_name="stress_optimized_packing.json",
            mime="application/json",
            use_container_width=True,
        )

    with c_dl2:
        csv_header = "id,x,y,z,length,width,height,orientation,weight,compression_psi,bending,shear_psi,deflection_in\n"
        csv_rows = "\n".join(
            [
                f"{p['id']},{p['position_vector'][0]},{p['position_vector'][1]},{p['position_vector'][2]},"
                f"{p['size'][0]},{p['size'][1]},{p['size'][2]},{p['orientation']},{p['weight']},"
                f"{p['stress_analysis']['compression_psi']},{p['stress_analysis']['bending_stress']},"
                f"{p['stress_analysis']['shear_psi']},{p['stress_analysis']['deflection_in']}"
                for p in out.get("placements", [])
            ]
        )
        st.download_button(
            "Download CSV",
            data=csv_header + csv_rows,
            file_name="stress_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
