"""
Feature Extraction Tool
Extracts and formats features from circuit-tracer JSON output.
"""

import json
import sys

def extract_features(input_path, output_path=None):
    """Extract features from circuit-tracer JSON output."""
    
    important_layers = ['10', '11', '12', '13', '14', '15', '16', '17']
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    results = []
    
    for node in nodes:
        layer = node.get('layer')
        if layer in important_layers:
            feature = node.get('feature')
            description = node.get('clerp', '')
            results.append(f"Layer: {layer}, Feature: {feature}, Description: {description}")
    
    if output_path:
        with open(output_path, 'w') as out_file:
            out_file.write('\n'.join(results))
    else:
        print('\n'.join(results))
    
    return len(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tracer_output.py <input.json> [output.txt]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    count = extract_features(input_file, output_file)
    print(f"Extracted {count} features from layers 10-17")