# check_config.py
import yaml

print("--- Running YAML Configuration Diagnostic ---")

try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("SUCCESS: config.yaml was successfully parsed.")
except Exception as e:
    print(f"FATAL ERROR: Could not parse config.yaml. Please fix the syntax.")
    print(f"--> Error details: {e}")
    exit()

print("\nChecking top-level keys...")
expected_top_keys = ['components', 'pipeline', 'data_path', 'vector_store_path']
for key in expected_top_keys:
    if key in config:
        print(f"- Found top-level key: '{key}' (Correct)")
    else:
        print(f"- MISSING top-level key: '{key}' (ERROR!)")
        print("  (This is likely an indentation problem. This key should have ZERO spaces before it.)")

print("\nChecking keys inside 'components'...")
if 'components' in config:
    expected_component_keys = ['loaders', 'splitters', 'embedders', 'llms', 'retrievers', 'rerankers']
    component_keys = list(config['components'].keys())
    print(f"Found: {component_keys}")
    
    for key in expected_component_keys:
        if key in component_keys:
            print(f"- Found component key: '{key}' (Correct)")
        else:
            print(f"- MISSING component key: '{key}' (ERROR!)")
            print("  (This is an indentation problem. This key should be indented under 'components'.)")
else:
    print("Cannot check 'components' because it was not found at the top level.")

print("\n--- Diagnostic Complete ---")
