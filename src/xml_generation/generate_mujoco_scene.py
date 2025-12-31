# -*- coding: utf-8 -*-
"""
Flexible MuJoCo scene generator that supports arbitrary robots & objects.

It:
- Accepts a robot include XML path (any robot).
- Parses the robot XML to discover body names automatically.
- Generates sensible contact exclusions (e.g., floor/table vs end-effector parts).
- Keeps the original YCB-object include behavior.
- Pretty-prints XML (no external service dependency).
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Iterable, List, Tuple, Optional

# -------- pretty print helper (no external dependency) ----------
def _indent(elem, level=0):
    # ET.indent is 3.9+, but we make a tiny compatible helper
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _indent(e, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# -------- robot XML parsing utilities ----------
def _collect_body_names_from_robot_xml(robot_xml_path: str) -> List[str]:
    """
    Recursively collects all <body name="..."> from a robot XML include file.
    """
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()
    body_names = []

    # typical structure is <mujoco><worldbody>...</worldbody></mujoco>
    for body in root.findall(".//worldbody//body"):
        name = body.get("name")
        if name:
            body_names.append(name)
    # also include top-level bodies directly under <mujoco> if any
    for body in root.findall("./body"):
        name = body.get("name")
        if name:
            body_names.append(name)

    # deduplicate preserving order
    seen = set()
    uniq = []
    for n in body_names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq

def _heuristic_end_effector_bodies(body_names: Iterable[str]) -> List[str]:
    """
    Heuristically select end-effector-related bodies for exclusion rules.
    Falls back to 'leaf-like' names if common keywords aren't found.
    """
    patterns = [
        r"hand", r"finger", r"gripper", r"fingertip", r"tcp", r"\bee\b", r"tool",
        r"flange", r"wrist", r"palm", r"end[_-]?effector", r"link(6|7|8|9|10)\b"
    ]
    regex = re.compile("|".join(patterns), re.IGNORECASE)

    # first pass: keyword matches
    candidates = [n for n in body_names if regex.search(n)]

    # fallback: if empty, prefer the "deepest" names (often leaves)
    if not candidates and body_names:
        # assume leaf-like if name appears only once as a prefix of other names
        # simple heuristic: longest names tend to be leaves
        sorted_by_len = sorted(body_names, key=lambda s: (-len(s), s))
        candidates = sorted_by_len[:3]  # pick a few likely leaves

    # always dedup & keep order as in body_names
    seen = set()
    ordered = []
    for n in body_names:
        if n in candidates and n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered

# -------- core generator ----------
def generate_mujoco_scene(
    output_file: str,
    # object_name: str,
    # object_base_path: str,
    obj_path: str,
    robot_include_path: str,
    robot_name: Optional[str] = None,
    static_bodies: Tuple[str, ...] = ("floor", "table"),
    also_exclude_all_robot_vs_floor_table: bool = False,
    table_size : Tuple[float, ...] = (0.4, 0.4, 0.2),
) -> None:
    """
    Generate a MuJoCo scene XML that includes:
      - the given robot (include file),
      - the given YCB object include,
      - floor/table,
      - contact exclusions computed automatically from the robot file.

    Args:
        output_file: where to write the scene XML.
        object_name: YCB object dir name (e.g., "065-j_cups").
        object_base_path: YCB base dir containing <object_name>/tsdf/textured/textured.xml
        robot_include_path: path to robot XML include.
        robot_name: label in <mujoco model="...">; default = basename(robot_include_path).
        static_bodies: which 'environment' bodies to create & consider for exclusions.
        also_exclude_all_robot_vs_floor_table: if True, add exclusions from (floor,table) to ALL robot bodies
                                              in addition to the heuristic EEF exclusions.
    """
    if robot_name is None:
        robot_name = os.path.splitext(os.path.basename(robot_include_path))[0]

    mujoco = ET.Element("mujoco", {"model": f"{robot_name} scene"})

    # includes
    ET.SubElement(mujoco, "include", {"file": robot_include_path})

    # obj_path = f"{object_base_path}/{object_name}/tsdf/textured/textured.xml"
    # obj_path = f"{object_base_path}/{object_name}/tsdf/textured/textured.xml"
    obj_path

    ET.SubElement(mujoco, "include", {"file": obj_path})

    # camera & stats
    ET.SubElement(mujoco, "statistic", {"center": "0.3 0 0.4", "extent": "1"})
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20", "offwidth": "1920", "offheight": "1080"})

    # assets
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"
    })
    ET.SubElement(asset, "texture", {
        "type": "2d", "name": "groundplane", "builtin": "checker", "mark": "edge",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8",
        "width": "300", "height": "300"
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane", "texture": "groundplane", "texuniform": "true",
        "texrepeat": "5 5", "reflectance": "0.2"
    })

    # world
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    floor = ET.SubElement(worldbody, "body", {"name": static_bodies[0]})
    ET.SubElement(floor, "geom", {"name": static_bodies[0], "size": "0 0 0.05", "type": "plane", "material": "groundplane"})

    # ---------------------------------- Table ----------------------------------
    table = ET.SubElement(worldbody, "body", {"name": static_bodies[1], "pos": f"0 0 {table_size[2]}"})
    ET.SubElement(table,
        "geom", {
            "name"    : static_bodies[1],
            # "size"    : f"{table_size} {table_size} {table_size}",
            "size"    : f"{table_size[0]} {table_size[1]} {table_size[2]}",
            "type"    : "box",
            "rgba"    : "1 1 1 0.3",
            # "friction": "0.001 0.005 0.0001"
             "friction": "10 0.005 0.0001",
    })

    # import ipdb; ipdb.set_trace()
    # ----------------- Canonical Fingertip Origin Site in World Frame -----------------
    ET.SubElement(worldbody,
        "site", {
            "name" : "canonical_fingertip_origin",
            "pos"  : f"0 0 {table_size[2]*2}",
            "rgba" : "1 0 1 0.3",
            "quat" : "0 1 0 0",
        })

    # ------------------------------- contact exclusions --------------------------------
    contact = ET.SubElement(mujoco, "contact")
    # always exclude floor vs table
    ET.SubElement(contact, "exclude", {"body1": static_bodies[0], "body2": static_bodies[1]})

    # analyze robot file
    try:
        robot_bodies = _collect_body_names_from_robot_xml(robot_include_path)
    except Exception as e:
        robot_bodies = []
        print(f"[WARN] Failed to read robot XML '{robot_include_path}': {e}")

    eef_like = _heuristic_end_effector_bodies(robot_bodies)

    def add_pairs(b1s: Iterable[str], b2s: Iterable[str]):
        for a in b1s:
            for b in b2s:
                if a == b:
                    continue
                ET.SubElement(contact, "exclude", {"body1": a, "body2": b})

    # Exclude floor/table with end-effector-like parts
    add_pairs([static_bodies[0], static_bodies[1]], eef_like)

    # Optionally, exclude all robot bodies with floor/table
    if also_exclude_all_robot_vs_floor_table and robot_bodies:
        add_pairs([static_bodies[0], static_bodies[1]], robot_bodies)

    # keyframe
    # keyframe = ET.SubElement(mujoco, "keyframe")
    # ET.SubElement(keyframe, "key", {
    #     "name": "home",
    #     "qpos": "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04       0 0 0.196   1 0 0 0",
    #     "ctrl": "0 0 0 -1.57079 0 1.57079 -0.7853 0.04"
    # })

    _indent(mujoco)

    tree = ET.ElementTree(mujoco)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote scene: {output_file}")
    print(f"  - Robot bodies found: {len(robot_bodies)}")
    if robot_bodies:
        print(f"  - Example bodies: {robot_bodies[:8]}")
    print(f"  - EEF-like bodies for exclusions: {eef_like}")


if __name__ == '__main__':

    # ---- demo using your uploaded Panda & a sample YCB object path ----
    # (Writes into /mnt/data for you to download)
    demo_out = "/mnt/data/generated_scene_flexible.xml"
    robot_xml = "/mnt/data/panda_arm_with_hand.xml"  # uploaded by you
    # You can change this to your actual YCB root once you download and run locally.
    dummy_ycb_root = "/home/you/ycb-tools/models/ycb"
    dummy_object = "065-j_cups"

    generate_mujoco_scene(
        output_file=demo_out,
        object_name=dummy_object,
        object_base_path=dummy_ycb_root,
        robot_include_path=robot_xml,
        robot_name="panda",
        also_exclude_all_robot_vs_floor_table=False,  # set True if you want everything excluded vs table/floor
    )

    print(f"\nDownload the generated XML here: /mnt/data/generated_scene_flexible.xml")
