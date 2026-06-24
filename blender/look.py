"""M0 最小 look：只为看清网格在动。M2 会替换内脏，但保持 setup() 签名。"""
import bpy
import mathutils


def setup(scene, engine: str = "BLENDER_EEVEE", samples: int = 16) -> None:
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.device = "CPU"
    elif hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples

    # 相机：放在 -Y、抬高，瞄准原点
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    loc = mathutils.Vector((0.0, -4.0, 2.2))
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector((0, 0, 0)) - loc).to_track_quat("-Z", "Y").to_euler()

    # 太阳光
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (0.6, 0.2, 0.3)

    # 世界背景
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)

    # 简单材质贴到所有导入的 mesh
    mat = bpy.data.materials.new("Cloth")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.7
    for ob in scene.objects:
        if ob.type == "MESH":
            ob.data.materials.clear()
            ob.data.materials.append(mat)
