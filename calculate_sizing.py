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
import os
from datetime import datetime

def load_instance_types(file_path):
    """Load instance types from JSON file."""
    try:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        instance_types = {}
        for instance in data['InstanceTypes']:
            instance_type = instance['InstanceType']
            instance_types[instance_type] = {
                'vcpu': instance['VCpuInfo']['DefaultVCpus'],
                'memory_gb': instance['MemoryInfo']['SizeInMiB'] / 1024,
                'bare_metal': instance.get('BareMetal', False),
                'hourly_cost': instance.get('Pricing', {}).get('us-east-1', 0),
                'family': instance['InstanceType'].split('.')[0]
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

import math

def bin_packing_simulation(metrics, instance_type, redundancy, max_pods_per_node=250):
    """Simulate bin packing to estimate required nodes using requests, usage peaks, and pod density."""
    vcpu_per_node = INSTANCE_TYPES[instance_type]['vcpu']
    memory_per_node = INSTANCE_TYPES[instance_type]['memory_gb']

    # Baseline: average resource requests (guaranteed schedulable)
    cpu_req = metrics['cpu_requests']['average']
    mem_req = metrics['memory_requests']['average']
    nodes_cpu_req = max(2, math.ceil(cpu_req / vcpu_per_node))
    nodes_mem_req = max(2, math.ceil(mem_req / memory_per_node))
    nodes_for_requests = max(nodes_cpu_req, nodes_mem_req)

    # Buffer: peak usage with redundancy
    cpu_peak = metrics['cpu_usage']['peak'] * redundancy
    mem_peak = metrics['memory_usage']['peak'] * redundancy
    nodes_cpu_peak = max(2, math.ceil(cpu_peak / vcpu_per_node))
    nodes_mem_peak = max(2, math.ceil(mem_peak / memory_per_node))
    nodes_for_peak = max(nodes_cpu_peak, nodes_mem_peak)

    # Pod density constraint
    required_pods = metrics['pod_count']['peak'] * redundancy
    nodes_for_pods = max(2, math.ceil(required_pods / max_pods_per_node))

    # Final node count: must satisfy all three constraints
    recommended_nodes = max(nodes_for_requests, nodes_for_peak, nodes_for_pods)

    return recommended_nodes, nodes_for_requests, nodes_for_peak, nodes_for_pods, max_pods_per_node

def calculate_worker_nodes(metrics, redundancy):
    """Calculate recommended worker node configuration using requests for baseline, peaks for buffer, and pod density."""
    recommendations = []
    for instance_type, specs in INSTANCE_TYPES.items():
        recommended_nodes, nodes_for_requests, nodes_for_peak, nodes_for_pods, max_pods_per_node = bin_packing_simulation(metrics, instance_type, redundancy)
        # Utilization relative to total node capacity
        cpu_util = (metrics['cpu_requests']['average']) / (recommended_nodes * specs['vcpu'])
        memory_util = (metrics['memory_requests']['average']) / (recommended_nodes * specs['memory_gb'])
        rationale = (
            f"Node count is the maximum of:\n"
            f"- Requests baseline: {nodes_for_requests} nodes (CPU req avg: {metrics['cpu_requests']['average']:.2f}, Mem req avg: {metrics['memory_requests']['average']:.2f} GB)\n"
            f"- Peak usage w/ redundancy: {nodes_for_peak} nodes (CPU peak: {metrics['cpu_usage']['peak']:.2f} cores, Mem peak: {metrics['memory_usage']['peak']:.2f} GB, redundancy: {redundancy})\n"
            f"- Pod density: {nodes_for_pods} nodes (Pod peak: {metrics['pod_count']['peak']:.0f}, max pods per node: {max_pods_per_node})"
        )
        recommendations.append({
            'instance_type': instance_type,
            'node_count': recommended_nodes,
            'specs': specs,
            'utilization': {
                'cpu': cpu_util,
                'memory': memory_util
            },
            'rationale': rationale,
            'estimated_cost': {
                'hourly': specs['hourly_cost'] * recommended_nodes,
                'monthly': specs['hourly_cost'] * recommended_nodes * 730,
                'instance_family': specs['family']
            }
        })
    # Sort recommendations by CPU utilization efficiency (requests-based)
    recommendations.sort(key=lambda x: abs(x['utilization']['cpu'] - 1))
    return recommendations[:3]

def generate_recommendations(metrics, redundancy):
    """Generate comprehensive sizing recommendations with detailed analysis."""
    worker_recommendations = calculate_worker_nodes(metrics, redundancy)
    storage_recommendations = calculate_storage(metrics, redundancy)
    
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_collection_window': {
                'start': metrics['metadata'].get('window_start', metrics['metadata'].get('collection_time')),
                'end': metrics['metadata'].get('collection_time'),
                'duration_hours': metrics['metadata'].get('duration_hours', 24)
            },
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
    """Calculate storage recommendations with IOPS analysis."""
    storage_peak = metrics['pvc_storage']['peak'] * redundancy
    iops_peak = metrics.get('pvc_iops', {}).get('peak', 3000) * redundancy
    throughput_peak = metrics.get('pvc_throughput', {}).get('peak', 125) * redundancy
    
    # Determine storage profile based on IOPS needs
    storage_profile = 'balanced'
    if iops_peak > 16000:
        storage_profile = 'high-iops'
    elif iops_peak < 3000:
        storage_profile = 'standard'
    
    recommendations = {
        'total_storage_gb': round(storage_peak),
        'performance_requirements': {
            'iops_peak': round(iops_peak),
            'throughput_peak_mbps': round(throughput_peak),
            'storage_profile': storage_profile
        },
        'recommendations': {
            'gp3': {
                'type': 'gp3',
                'description': 'General Purpose SSD',
                'recommended_for': 'Most workloads',
                'min_size_gb': round(storage_peak),
                'min_iops': max(3000, round(iops_peak * 0.8)),
                'min_throughput': max(125, round(throughput_peak * 0.8))
            },
            'io2': {
                'type': 'io2',
                'description': 'Provisioned IOPS SSD',
                'recommended_for': 'I/O-intensive workloads',
                'min_size_gb': round(storage_peak),
                'min_iops': max(100, round(iops_peak * 0.8)),
                'min_throughput': max(125, round(throughput_peak * 0.8))
            }
        }
    }
    
    return recommendations

def detect_workload_type(metrics):
    """Determine if workload is CPU-bound, memory-bound, or balanced."""
    cpu_peak = metrics['cpu_usage']['peak']
    memory_peak = metrics['memory_usage']['peak']
    
    # Calculate ratio of CPU to memory usage
    ratio = cpu_peak / (memory_peak + 0.0001)  # Avoid division by zero
    
    if ratio > 1.5:
        return 'CPU-bound'
    elif ratio < 0.67:
        return 'Memory-bound'
    else:
        return 'Balanced'

def generate_recommendations(metrics, redundancy):
    """Generate comprehensive sizing recommendations."""
    workload_type = detect_workload_type(metrics)
    worker_recommendations = calculate_worker_nodes(metrics, redundancy)
    storage_recommendations = calculate_storage(metrics, redundancy)
    
    # Add ROSA-specific context
    rosa_context = {
        'rosa_specifics': {
            'control_plane': 'Managed by ROSA (3 m5.xlarge nodes)',
            'infra_nodes': 'Managed by ROSA (3 m5.xlarge nodes)',
            'minimum_workers': 2,
            'sla_requirements': '3+ nodes across multiple AZs recommended for production'
        }
    }
    
    return {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_collection_window': {
                'start': metrics['metadata'].get('window_start'),
                'end': metrics['metadata'].get('collection_time'),
                'duration_hours': metrics['metadata'].get('duration_hours', 24)
            },
            'redundancy_factor': redundancy,
            'rosa_context': rosa_context,
            'analysis_note': 'Recommendations based on peak usage during collection window'
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
    window = recommendations['metadata']['metrics_collection_window']
    text.append(f"Metrics collection window: {window['start']} to {window['end']} ({window['duration_hours']} hours)")
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
        text.append(f"   - Memory per node: {rec['specs']['memory_gb']:.1f} GB")
        text.append(f"   - Bare metal: {'Yes' if rec['specs'].get('bare_metal', False) else 'No'}")
        text.append(f"   - CPU utilization: {rec['utilization']['cpu']*100:.1f}%")
        text.append(f"   - Memory utilization: {rec['utilization']['memory']*100:.1f}%")
        text.append(f"   - Rationale: {rec.get('rationale', 'N/A')}")
        text.append(f"   - Estimated Cost:")
        text.append(f"     - Hourly: ${rec['estimated_cost']['hourly']:.2f}")
        text.append(f"     - Monthly: ${rec['estimated_cost']['monthly']:.2f}")
        text.append(f"     - Instance Family: {rec['estimated_cost']['instance_family']}")
    
    text.append("\nStorage Performance Requirements:")
    perf = recommendations['storage']['performance_requirements']
    text.append(f"- Peak IOPS: {perf['iops_peak']}")
    text.append(f"- Peak Throughput: {perf['throughput_peak_mbps']} MB/s")
    text.append(f"- Profile: {perf['storage_profile']}")
    
    text.append("\nStorage Recommendations:")
    text.append(f"Total storage required: {recommendations['storage']['total_storage_gb']} GB")
    for storage_type, details in recommendations['storage']['recommendations'].items():
        text.append(f"\n{storage_type.upper()}:")
        text.append(f"- Type: {details['type']}")
        text.append(f"- Description: {details['description']}")
        text.append(f"- Recommended for: {details['recommended_for']}")
        text.append(f"- Minimum size: {details['min_size_gb']} GB")
        text.append(f"- Minimum IOPS: {details['min_iops']}")
        text.append(f"- Minimum Throughput: {details['min_throughput']} MB/s")
    
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
