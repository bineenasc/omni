"""Helper script — run once to generate soccer_1v1.wbt."""
import os

base = os.path.dirname(os.path.abspath(__file__))
src  = os.path.join(base, "soccer.wbt")
dst  = os.path.join(base, "soccer_1v1.wbt")

with open(src, encoding="utf-8") as f:
    lines = f.readlines()

# Lines 1-6: header/EXTERNPROTO — keep as-is (IMPORTABLE EXTERNPROTO works for
# both dynamic insertion and static declarations in Webots R2025a).
# Lines 7-260: WorldInfo + lights + arena geometry — copy unchanged.
header = "".join(lines[0:260])

nodes = (
    "\n"
    "DEF BOLA Solid {\n"
    "  translation 0 0.025 0\n"
    "  children [\n"
    "    DEF BALL_SHAPE Shape {\n"
    "      appearance PBRAppearance {\n"
    "        baseColor 1 0.54 0.08\n"
    "        roughness 0.3\n"
    "        metalness 0\n"
    "      }\n"
    "      geometry Sphere {\n"
    "        radius 0.025\n"
    "        subdivision 4\n"
    "      }\n"
    "    }\n"
    "  ]\n"
    '  name "ball"\n'
    '  model "ball"\n'
    "  boundingObject USE BALL_SHAPE\n"
    "  physics Physics {\n"
    "    density -1\n"
    "    mass 0.055\n"
    "    centerOfMass [\n"
    "      0 0 0\n"
    "    ]\n"
    "    damping Damping {\n"
    "      linear 0.17\n"
    "      angular 0.33\n"
    "    }\n"
    "  }\n"
    "}\n"
    "DEF VIPER Viper {\n"
    "  translation 0 0 -1.0\n"
    "  rotation 1 0 0 -1.5707953071795862\n"
    '  name "viper"\n'
    '  controller "robot_controller"\n'
    "}\n"
    "DEF TITAN Titan {\n"
    "  translation 0 0 1.0\n"
    "  rotation 1 0 0 -1.5707953071795862\n"
    '  name "titan"\n'
    '  controller "robot_controller"\n'
    "  action_channel 2\n"
    "  sensor_channel 3\n"
    "}\n"
    "Robot {\n"
    "  children [\n"
    "    Emitter {\n"
    '      name "supervisor_emitter"\n'
    "    }\n"
    "    Receiver {\n"
    '      name "supervisor_receiver"\n'
    "      channel 1\n"
    "    }\n"
    "    Emitter {\n"
    '      name "supervisor_emitter_titan"\n'
    "      channel 2\n"
    "    }\n"
    "    Receiver {\n"
    '      name "supervisor_receiver_titan"\n'
    "      channel 3\n"
    "    }\n"
    "  ]\n"
    '  controller "soccer_supervisor_1v1"\n'
    "  supervisor TRUE\n"
    "}\n"
)

with open(dst, "w", encoding="utf-8") as f:
    f.write(header)
    f.write(nodes)

print(f"Written: {dst}  ({os.path.getsize(dst):,} bytes)")
