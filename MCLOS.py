import random
import json
import streamlit as st
import plotly.graph_objects as go

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

def support_ok(pos, size, placed, z_tol=1e-6, min_support_frac=0.7):
    x, y, z = pos
    dx, dy, dz = size
    if z <= z_tol:
        return True

    need = dx * dy * min_support_frac
    supported = 0.0

    for p in placed:
        px, py, pz = p["pos"]
        pdx, pdy, pdz = p["size"]
        top = pz + pdz

        if abs(top - z) > z_tol:
            continue

        ox = max(0.0, min(x + dx, px + pdx) - max(x, px))
        oy = max(0.0, min(y + dy, py + pdy) - max(y, py))
        supported += ox * oy

        if supported >= need:
            return True

    return False

def generate_orientations(L, H, T):
    return [
        ("flat_LxH", (L, H, T)),
        ("flat_HxL", (H, L, T)),
        ("stand_LxT", (L, T, H)),
        ("stand_TxL", (T, L, H)),
        ("stand_HxT", (H, T, L)),
        ("stand_TxH", (T, H, L)),
    ]

def candidate_positions(trailer, step):
    xs = [i * step for i in range(int(trailer[0] // step) + 1)]
    ys = [i * step for i in range(int(trailer[1] // step) + 1)]
    zs = [i * step for i in range(int(trailer[2] // step) + 1)]
    pts = [(x, y, z) for x in xs for y in ys for z in zs]
    random.shuffle(pts)
    return pts

def pack_panels(panel_L, panel_H, panel_T, panel_W,
                trailer_L, trailer_W, trailer_H,
                max_horizontal, max_vertical,
                num_panels, step, seed):

    random.seed(seed)
    trailer = (trailer_L, trailer_W, trailer_H)

    orientations = []
    for name, size in generate_orientations(panel_L, panel_H, panel_T):
        if in_bounds((0,0,0), size, trailer):
            orientations.append((name, size))

    if not orientations:
        return {"error": "No orientations fit inside trailer."}

    pts = candidate_positions(trailer, step)
    placed = []

    for pid in range(num_panels):
        placed_this = False
        random.shuffle(orientations)

        for name, size in orientations:
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            for pos in pts:
                if not in_bounds(pos, size, trailer):
                    continue

                if not support_ok(pos, size, placed):
                    continue

                if any(aabb_intersect(pos, size, p["pos"], p["size"]) for p in placed):
                    continue

                placed.append({
                    "id": pid,
                    "pos": pos,
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

    return {
        "inputs": {
            "panel": {"L": panel_L, "H": panel_H, "T": panel_T, "W": panel_W},
            "trailer": {"L": trailer_L, "W": trailer_W, "H": trailer_H}
        },
        "settings": {
            "requested_panels": int(num_panels),
            "placed_panels": len(placed),
            "grid_step_in": float(step),
            "seed": int(seed),
            "min_support_frac": 0.7
        },
        "placements": [
            {
                "id": p["id"],
                "position_vector": [round(p["pos"][0], 3), round(p["pos"][1], 3), round(p["pos"][2], 3)],
                "size": [round(p["size"][0], 3), round(p["size"][1], 3), round(p["size"][2], 3)],
                "orientation": p["orientation"],
                "weight": p["weight"]
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
        name=name
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

def visualize(out):
    if "error" in out:
        st.error(out["error"])
        return

    tr = out["inputs"]["trailer"]
    L, W, H = tr["L"], tr["W"], tr["H"]

    fig = go.Figure()
    add_wire(fig, 0,0,0, L,W,H, "white", "Trailer Outline", opacity=0.9, width=6)

    palette = ["#2ecc71","#3498db","#e74c3c","#f1c40f","#9b59b6","#1abc9c","#e67e22","#95a5a6"]
    cmap = {}

    for p in out["placements"]:
        o = p["orientation"]
        if o not in cmap:
            cmap[o] = palette[len(cmap) % len(palette)]
        c = cmap[o]

        x,y,z = p["position_vector"]
        dx,dy,dz = p["size"]

        add_solid(fig, x,y,z,dx,dy,dz,c,opacity=0.28)
        add_wire(fig, x,y,z,dx,dy,dz,c,o,opacity=0.95,width=4)

    for o, c in cmap.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color=c, width=6),
            name=o
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="Length (in)",
            yaxis_title="Width (in)",
            zaxis_title="Height (in)",
            aspectmode="data"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title="Modular Cargo Loading Optimization", layout="wide")
st.title("Modular Cargo Loading Optimization System")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Panel Inputs (in, lb)")
    pL = st.number_input("Panel Length", value=112.0)
    pH = st.number_input("Panel Height", value=97.25)
    pT = st.number_input("Panel Thickness", value=5.5)
    pW = st.number_input("Panel Weight", value=220.0)

with c2:
    st.subheader("Trailer Inputs (in)")
    tL = st.number_input("Trailer Length", value=630.0)
    tW = st.number_input("Trailer Width", value=102.0)
    tH = st.number_input("Trailer Height", value=162.0)

    st.subheader("Handling + Solver Settings")
    maxH = st.number_input("Max Horizontal", value=145.0)
    maxV = st.number_input("Max Vertical", value=114.0)

    num = st.number_input("Requested Panels", min_value=0, value=24, step=1)
    step = st.number_input("Grid Step (in)", min_value=0.1, value=1.0, step=0.5)
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

if st.button("Run Optimization", type="primary"):
    out = pack_panels(pL,pH,pT,pW, tL,tW,tH, maxH,maxV, num,step,seed)
    st.subheader("3D Arrangement Visualization")
    visualize(out)

    st.subheader("Output JSON")
    st.json(out)

    st.download_button(
        "Download JSON",
        data=json.dumps(out, indent=2),
        file_name="packing_output.json",
        mime="application/json"
    )
