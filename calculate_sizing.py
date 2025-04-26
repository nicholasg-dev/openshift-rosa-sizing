#!/usr/bin/env python3
# ============================================================================
# OpenShift ROSA Sizing Calculator
# ============================================================================
#
# This script analyzes metrics collected by collect_metrics.py and provides
# sizing recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.
#
# The script reads the metrics JSON file and calculates recommended instance
# types, node counts, and other configuration parameters based on usage patterns.
#
# Usage:
#   ./calculate_sizing.py --input metrics.json [options]
#
# Author: OpenShift Sizing Team
# ============================================================================

import json
import argparse
import sys
from datetime import datetime

def load_instance_types(file_path):
    """Load instance types from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        instance_types = {}
        for instance in data['InstanceTypes']:
            instance_type = instance['InstanceType']
            instance_types[instance_type] = {
                'vcpu': instance['VCpuInfo']['DefaultVCpus'],
                'memory_gb': instance['MemoryInfo']['SizeInMiB'] / 1024,
                'bare_metal': instance.get('BareMetal', False)
            }
        return instance_types
    except Exception as e:
        print(f"Error loading instance types: {e}")
        sys.exit(1)

INSTANCE_TYPES = load_instance_types('instance_types.json')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate ROSA cluster sizing recommendations based on metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input metrics JSON file (from collect_metrics.py)'
    )
    
    parser.add_argument(
        '--output',
        default='rosa_sizing.json',
        help='Output file for sizing recommendations'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format (json or text)'
    )
    
    parser.add_argument(
        '--redundancy',
        type=float,
        default=1.3,
        help='Redundancy factor for capacity planning (e.g., 1.3 = 30% extra)'
    )
    
    return parser.parse_args()

def load_metrics(input_file):
    """Load metrics from the JSON file."""
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        sys.exit(1)

def calculate_worker_nodes(metrics, redundancy):
    """Calculate recommended worker node configuration with detailed analysis."""
    cpu_peak = metrics['cpu_usage']['peak'] * redundancy
    memory_peak = metrics['memory_usage']['peak'] * redundancy
    
    recommendations = []
    for instance_type, specs in INSTANCE_TYPES.items():
        # Calculate nodes needed based on CPU and memory
        nodes_cpu = max(2, round(cpu_peak / specs['vcpu']))  # ROSA minimum 2 nodes
        login_nodes = max(2, round(memory_peak / specs['memory_gb']))
        recommended_nodes = max(nodes_cpu, login_nodes)
        
        cpu_util = (cpu_peak / recommended_nodes) / specs['vcpu']
        memory_util = (memory_peak / recommended_nodes) / specs['memory_gb']
        
        # Check if this configuration meets ROSA requirements
        if recommended_nodes >= 2:
            recommendations.append({
                'instance_type': instance_type,
                'node_count': recommended_nodes,
                'specs': specs,
                'utilization': {
                    'cpu': cpu_util,
                    'memory': memory_util
                },
                'rationale': f"CPU peak: {cpu_peak:.2f} cores, Memory peak: {memory_peak:.2f} GB"
            })
    
    # Sort by CPU utilization efficiency
    recommendations.sort(key=lambda x: abs(x['utilization']['cpu'] - 1))
    return recommendations[:3]

def generate_recommendations(metrics, redundancy):
    """Generate comprehensive sizing recommendations with detailed analysis."""
    worker_recommendations = calculate_worker_nodes(metrics, redundancy)
    storage_recommendations = calculate_storage(metrics, redundancy)
    
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_collection_time': metrics['metadata']['collection_time'],
            'redundancy_factor': redundancy
        },
        'summary': {
            'current_metrics': {
                'cpu_cores_peak': metrics['cpu_usage']['peak'],
                'memory_gb_peak': metrics['memory_usage']['peak'],
                'pod_count_peak': metrics['pod_count']['peak'],
                'storage_gb_peak': metrics['pvc_storage']['peak']
            }
        },
        'worker_nodes': {
            'recommendations': worker_recommendations,
            'notes': [
                'ROSA minimum 2 worker nodes recommended',
                'Multiple AZs recommended for HA'
            ]
        },
        'storage': storage_recommendations
    }

def calculate_storage(metrics, redundancy):
    """Calculate storage recommendations."""
    storage_peak = metrics['pvc_storage']['peak'] * redundancy
    
    return {
        'total_storage_gb': round(storage_peak),
        'recommendations': {
            'gp3': {
                'type': 'gp3',
                'description': 'General Purpose SSD',
                'recommended_for': 'Most workloads',
                'min_size_gb': round(storage_peak)
            },
            'io2': {
                'type': 'io2',
                'description': 'Provisioned IOPS SSD',
                'recommended_for': 'I/O-intensive workloads',
                'min_size_gb': round(storage_peak)
            }
        }
    }

def generate_recommendations(metrics, redundancy):
    """Generate comprehensive sizing recommendations."""
    worker_recommendations = calculate_worker_nodes(metrics, redundancy)
    storage_recommendations = calculate_storage(metrics, redundancy)
    
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_collection_time': metrics['metadata']['collection_time'],
            'redundancy_factor': redundancy
        },
        'summary': {
            'current_metrics': {
                'cpu_cores_peak': metrics['cpu_usage']['peak'],
                'memory_gb_peak': metrics['memory_usage']['peak'],
                'pod_count_peak': metrics['pod_count']['peak'],
                'storage_gb_peak': metrics['pvc_storage']['peak']
            }
        },
        'worker_nodes': {
            'recommendations': worker_recommendations,
            'notes': [
                'Minimum 3 nodes recommended for high availability',
                'Node counts include redundancy factor',
                'Consider using multiple availability zones',
                'Actual requirements may vary based on workload patterns'
            ]
        },
        'storage': storage_recommendations
    }

def format_text_output(recommendations):
    """Format recommendations as human-readable text."""
    text = []
    text.append("=" * 80)
    text.append("ROSA Cluster Sizing Recommendations")
    text.append("=" * 80)
    
    text.append("\nGenerated at: " + recommendations['metadata']['generated_at'])
    text.append("Based on metrics from: " + recommendations['metadata']['metrics_collection_time'])
    text.append(f"Redundancy factor: {recommendations['metadata']['redundancy_factor']}")
    
    text.append("\nCurrent Usage Peaks:")
    text.append(f"- CPU Cores: {recommendations['summary']['current_metrics']['cpu_cores_peak']:.2f}")
    text.append(f"- Memory (GB): {recommendations['summary']['current_metrics']['memory_gb_peak']:.2f}")
    text.append(f"- Pod Count: {recommendations['summary']['current_metrics']['pod_count_peak']:.0f}")
    text.append(f"- Storage (GB): {recommendations['summary']['current_metrics']['storage_gb_peak']:.2f}")
    
    text.append("\nWorker Node Recommendations:")
    for i, rec in enumerate(recommendations['worker_nodes']['recommendations'], 1):
        text.append(f"\n{i}. {rec['instance_type']} Configuration:")
        text.append(f"   - Number of nodes: {rec['node_count']}")
        text.append(f"   - vCPUs per node: {rec['specs']['vcpu']}")
        text.append(f"   - Memory per node: {rec['specs']['memory_gb']} GB")
        text.append(f"   - Use case: {rec['specs']['use_case']}")
        text.append(f"   - CPU utilization: {rec['utilization']['cpu']*100:.1f}%")
        text.append(f"   - Memory utilization: {rec['utilization']['memory']*100:.1f}%")
    
    text.append("\nStorage Recommendations:")
    text.append(f"Total storage required: {recommendations['storage']['total_storage_gb']} GB")
    for storage_type, details in recommendations['storage']['recommendations'].items():
        text.append(f"\n{storage_type.upper()}:")
        text.append(f"- Type: {details['type']}")
        text.append(f"- Description: {details['description']}")
        text.append(f"- Recommended for: {details['recommended_for']}")
        text.append(f"- Minimum size: {details['min_size_gb']} GB")
    
    text.append("\nNotes:")
    for note in recommendations['worker_nodes']['notes']:
        text.append(f"- {note}")
    
    return "\n".join(text)

def main():
    """Main function to run the sizing calculations."""
    args = parse_arguments()
    
    # Load metrics
    metrics = load_metrics(args.input)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics, args.redundancy)
    
    # Save recommendations
    try:
        if args.format == 'json':
            with open(args.output, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"\nRecommendations saved to {args.output}")
            
            # Also create a text summary
            summary_file = args.output.rsplit('.', 1)[0] + '_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(format_text_output(recommendations))
            print(f"Summary saved to {summary_file}")
        else:
            # Save text format
            with open(args.output, 'w') as f:
                f.write(format_text_output(recommendations))
            print(f"\nRecommendations saved to {args.output}")
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
