import sys
import os
import asyncio

# Try to import SimulationApp. checking both new and old paths
try:
    from isaacsim import SimulationApp
except ImportError:
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError:
        print("Error: Could not import SimulationApp. Make sure Isaac Sim is installed.")
        sys.exit(1)

# Configuration for the simulation app
# We need the asset converter extension
config = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
    "extensions": [
        "omni.kit.asset_converter"
    ]
}

# Start the application
print("[Isaac Converter] Starting SimulationApp...")
simulation_app = SimulationApp(config)

# Import omni modules after app startup
import omni.kit.asset_converter
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Kind, Gf

def find_texture(name_part, texture_dir):
    """Find a texture file containing name_part in texture_dir (case insensitive)"""
    if not os.path.exists(texture_dir):
        return None
    
    files = os.listdir(texture_dir)
    name_part = name_part.lower()
    
    # Exact match preference or "contains"
    candidates = []
    for f in files:
        if name_part in f.lower() and (f.lower().endswith('.jpg') or f.lower().endswith('.png')):
            candidates.append(f)
            
    if not candidates:
        return None
        
    # Sort by length to find most specific or just take first
    candidates.sort(key=len)
    return os.path.join(texture_dir, candidates[0])

def bind_materials(stage, texture_dir):
    print(f"[Isaac Converter] Starting Material/Texture binding...")
    
    # Common texture mappings based on keywords
    # (Keyword in Material Name) -> { 'diffuse': 'filename_part', 'normal': 'filename_part', ... }
    rules = [
        {
            "keywords": ["leaf", "leaves"],
            "textures": {
                "diffuse_texture": "Orange-leaves-kat.jpg",
                "normal_texture": "Orange-leaves-Normal.jpg",
                "opacity_texture": "Orange-leaves-Transl.jpg" # Often mapped to opacity or transmission
            }
        },
        {
            "keywords": ["orange", "fruit"], # Orange fruit, exclude if matched by leaf above
            "textures": {
                "diffuse_texture": "Orange_Base.jpg",
                "normal_texture": "Orange_Normal.jpg",
                "bump_texture": "Orange_Bump.jpg"
            }
        },
        {
            "keywords": ["bark", "trunk", "wood", "branch"],
            "textures": {
                "diffuse_texture": "tree_bark.jpg"
            }
        },
        {
            "keywords": ["pot", "vase", "planter"],
            "textures": {
                "diffuse_texture": "pot-gryazi-7.jpg",
                "normal_texture": "Pot_normal.jpg"
            }
        },
        {
            "keywords": ["gravel", "soil", "dirt", "ground"],
            "textures": {
                "diffuse_texture": "gravel texture-seamless_hr_DIFFUSE.jpg",
                "normal_texture": "gravel texture-seamless_hr_NORM.jpg",
                "displacement_texture": "gravel texture-seamless_hr_DISPL.jpg"
            }
        },
        {
            "keywords": ["peduncle", "stem"],
            "textures": {
                "diffuse_texture": "Orange_peduncle .jpg"
            }
        }
    ]

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
            
        mat_name = prim.GetName().lower()
        print(f"[Isaac Converter] Processing Material: {prim.GetName()}")
        
        # Find matching rule
        matched_textures = {}
        for rule in rules:
            if any(k in mat_name for k in rule["keywords"]):
                matched_textures = rule["textures"]
                break
        
        if not matched_textures:
            continue
            
        print(f"  -> Matched keywords. Applying textures: {matched_textures}")
        
        # Get or Create Shader
        # Usually Asset Converter creates a UsdShade.Shader child. We want to modify it.
        material = UsdShade.Material(prim)
        
        # Iterate children to find the OmniPBR or PBR shader
        shader = None
        for child in prim.GetChildren():
            if child.IsA(UsdShade.Shader):
                shader = UsdShade.Shader(child)
                break
        
        # If no shader found (unlikely if converted), or we want to force OmniPBR
        if not shader:
            # Create a new OmniPBR shader
            shader = UsdShade.Shader.Define(stage, prim.GetPath().AppendChild("Shader"))
            shader.CreateIdAttr("OmniPBR.mdl")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            # Bind output
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")

        # Helper to set texture input
        def set_texture(input_name, filename):
            full_path = os.path.join(texture_dir, filename)
            if os.path.exists(full_path):
                # Use relative path for portability if possible, or absolute
                # Here using absolute for safety in this environment
                clean_path = full_path.replace("\\", "/")
                shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset).Set(clean_path)
                # Enable texture influence if needed (OmniPBR specific flags)
                # e.g. albedo_add (defaults usually work if texture is present)

        # Apply textures
        if "diffuse_texture" in matched_textures:
            set_texture("diffuse_texture", matched_textures["diffuse_texture"])
            
        if "normal_texture" in matched_textures:
            set_texture("normal_texture", matched_textures["normal_texture"])
            
        if "bump_texture" in matched_textures:
            # OmniPBR usually uses normal_texture, but can handle bump via height or specific inputs
            # We'll map bump to 'bump_map_texture' or similar if OmniPBR supports it, 
            # or just rely on normal.
            pass 

        if "opacity_texture" in matched_textures:
            set_texture("opacity_texture", matched_textures["opacity_texture"])
            # Ensure opacity mode is enabled
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)

async def convert(input_path, output_path):
    print(f"[Isaac Converter] Converting: {input_path}")
    print(f"[Isaac Converter] To: {output_path}")
    
    converter = omni.kit.asset_converter.get_instance()
    context = omni.kit.asset_converter.AssetConverterContext()
    # Enable materials and PBR
    context.ignore_materials = False
    context.export_preview_surface = False # Use OmniPBR/MDL for better quality in Isaac Sim
    context.use_meter_as_world_unit = True
    context.merge_all_meshes = False
    
    # Ensure textures are searched in the 'textures' subdirectory if needed
    # The converter usually looks relative to input, but we can try to help it
    # context.search_paths = [os.path.dirname(input_path), os.path.join(os.path.dirname(input_path), "textures")]
    
    task = converter.create_converter_task(input_path, output_path, None, context)
    
    # Wait for the task to finish
    success = await task.wait_until_finished()
    
    if not success:
        print(f"[Isaac Converter] ERROR: Failed to convert {input_path}")
        detailed_error = task.get_error_message()
        if detailed_error:
            print(f"[Isaac Converter] Details: {detailed_error}")
        return False
    
    print(f"[Isaac Converter] Conversion phase complete. Starting Post-Processing...")
    
    # Post-processing: Add properties, Kind, and Default Prim
    try:
        stage = Usd.Stage.Open(output_path)
        if not stage:
            print(f"[Isaac Converter] Error: Could not open converted stage.")
            return False

        # 0. Bind Textures (New Step)
        texture_dir = os.path.join(os.path.dirname(input_path), "textures")
        bind_materials(stage, texture_dir)

        # 1. Find and Set Default Prim
        # Note: using IsAbstract() instead of IsClass() for compatibility
        root_prims = [p for p in stage.GetPseudoRoot().GetChildren() if not p.IsAbstract() and not p.GetName().startswith("Render")]
        valid_root = None
        
        if root_prims:
            valid_root = root_prims[0]
            stage.SetDefaultPrim(valid_root)
            print(f"[Isaac Converter] Set Default Prim: {valid_root.GetName()}")
            
            # 2. Set Kind (Component) - Important for Isaac Sim selection/assembly
            if Usd.ModelAPI(valid_root):
                Usd.ModelAPI(valid_root).SetKind(Kind.Tokens.component)
                print(f"[Isaac Converter] Set Kind: component")

        # 3. Enable Physics (Optional - simple collision)
        # Note: For complex trees, exact collision is heavy. We won't auto-add collision 
        # to avoid crashing sim, but we ensure Xformable.
        
        # 4. Check Materials
        material_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material_count += 1
        
        print(f"[Isaac Converter] Found {material_count} materials.")
        
        # 5. Save changes
        stage.GetRootLayer().Save()
        print(f"[Isaac Converter] SUCCESS: Post-processing completed.")
        
    except Exception as e:
        print(f"[Isaac Converter] Warning during post-processing: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the whole process if post-processing has minor issues, 
        # but the file was created.
    
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_with_isaacsim.py <input_fbx> <output_usd>")
        return

    input_fbx = os.path.abspath(sys.argv[1])
    output_usd = os.path.abspath(sys.argv[2])

    if not os.path.exists(input_fbx):
        print(f"Error: Input file not found: {input_fbx}")
        return

    # Run the async conversion loop
    async def run_loop():
        await convert(input_fbx, output_usd)

    # Execute the async task while updating the app
    task = asyncio.ensure_future(run_loop())
    
    while not task.done():
        simulation_app.update()
    
    simulation_app.close()

if __name__ == "__main__":
    main()
