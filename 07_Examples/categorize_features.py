"""
Feature Categorization Tool
Categorizes discovered features from circuit analysis by semantic domain.
"""

import json
import sys

def categorize_features(file_path):
    """Categorize features by semantic domain."""
    
    categories = {
        "science_discovery": ["science", "scientific", "discovery", "finding", "research", "scientists"],
        "mythical_creatures": ["unicorn", "mythical", "fantastical", "fantasy"],
        "geography_location": ["valley", "mountain", "andes", "remote", "unexplored", "location", "geography"],
        "surprise_shock": ["shocking", "surprise", "unexpected"],
    }
    
    category_counts = {str(layer): {cat: 0 for cat in categories} 
                      for layer in range(10, 18)}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) < 3:
                continue
            layer = parts[0].split(": ")[1]
            description = parts[2].split(": ")[1].lower()
            
            for category, keywords in categories.items():
                if any(keyword in description for keyword in keywords):
                    category_counts[layer][category] += 1
    
    for layer, counts in category_counts.items():
        print(f"Layer {layer}:")
        for category, count in counts.items():
            if count > 0:
                print(f"  {category}: {count}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'analysis_results.txt'
    categorize_features(file_path)
