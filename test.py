import random
import json
import streamlit as st

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
    horiz = max(size[0], size[1])
    vert = size[2]
    return (horiz <= max_horizontal) and (vert <= max_vertical)

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

def pack_panels_3d(panel_L, panel_H, panel_T, panel_W,
                   trailer_L, trailer_W, trailer_H,
                   max_horizontal, max_vertical,
                   num_panels=6, step=1.0, seed=42):

    random.seed(seed)

    trailer = (trailer_L, trailer_W, trailer_H)

    orientations = []
    for name, size in generate_orientations(panel_L, panel_H, panel_T):
        if in_bounds((0, 0, 0), size, trailer):
            orientations.append((name, size))

    if not orientations:
        return {"error": "No orientations fit inside trailer."}

    pts = candidate_positions(trailer, step)
    placed = []

    for pid in range(num_panels):
        placed_this = False
        random.shuffle(orientations)

        for orient_name, size in orientations:
            if not handling_ok(size, max_horizontal, max_vertical):
                continue

            for pos in pts:
                if not in_bounds(pos, size, trailer):
                    continue

                collision = False
                for p in placed:
                    if aabb_intersect(pos, size, p["pos"], p["size"]):
                        collision = True
                        break
                if collision:
                    continue

                placed.append({
                    "id": pid,
                    "pos": pos,
                    "size": size,
                    "orientation": orient_name,
                    "weight": panel_W
                })
                placed_this = True
                break

            if placed_this:
                break

        if not placed_this:
            break

    result = {
        "units": "in_lb",
        "inputs": {
            "panel": {
                "length": panel_L,
                "height": panel_H,
                "thickness": panel_T,
                "weight": panel_W
            },
            "trailer": {
                "length": trailer_L,
                "width": trailer_W,
                "height": trailer_H
            },
            "limits": {
                "max_horizontal": max_horizontal,
                "max_vertical": max_vertical
            }
        },
        "settings": {
            "requested_panels": num_panels,
            "placed_panels": len(placed),
            "grid_step_in": step,
            "seed": seed
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

    return result


st.set_page_config(page_title="Modular Cargo Loading Optimization System", layout="wide")
st.title("Modular Cargo Loading Optimization System")
st.caption("Alpha Prototype")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Panel Inputs")
    panel_L = st.number_input("Panel Length (in)", value=112.0)
    panel_H = st.number_input("Panel Height (in)", value=97.25)
    panel_T = st.number_input("Panel Thickness (in)", value=5.5)
    panel_W = st.number_input("Panel Weight (lb)", value=220.0)

    st.subheader("Trailer Inputs")
    trailer_L = st.number_input("Trailer Length (in)", value=630.0)
    trailer_W = st.number_input("Trailer Width (in)", value=102.0)
    trailer_H = st.number_input("Trailer Height (in)", value=162.0)

with col2:
    st.subheader("Handling + Solver Settings")
    max_horizontal = st.number_input("Max Horizontal", value=145.0)
    max_vertical = st.number_input("Max Vertical", value=114.0)

    num_panels = st.number_input("Requested Panels", min_value=0, value=6, step=1)
    step = st.number_input("Grid Step (in)", min_value=0.1, value=1.0, step=0.5)
    seed = st.number_input("Random Seed", min_value=0, value=42, step=1)

run = st.button("Run Optimization", type="primary")

if run:
    out = pack_panels_3d(
        panel_L, panel_H, panel_T, panel_W,
        trailer_L, trailer_W, trailer_H,
        max_horizontal, max_vertical,
        num_panels=int(num_panels),
        step=float(step),
        seed=int(seed)
    )

    st.subheader("Output JSON")
    st.json(out)

    st.download_button(
        label="Download JSON",
        data=json.dumps(out, indent=2),
        file_name="packing_output.json",
        mime="application/json"
    )
