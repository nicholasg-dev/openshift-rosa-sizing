#!/usr/bin/env python3
"""
OpenShift ROSA Sizing Tool - Sizing Calculator Script
Version: 2.0

This script analyzes collected metrics to provide ROSA sizing recommendations
including optimal AWS instance types, node counts, and storage requirements.
"""

import argparse
import json
import os
import sys
import math
from datetime import datetime

# Default configuration
DEFAULT_INPUT = "cluster_metrics.json"
DEFAULT_OUTPUT = "rosa_sizing.json"
DEFAULT_FORMAT = "json"
DEFAULT_REDUNDANCY = 1.3

# AWS instance type definitions for ROSA worker nodes
# Format: name: {"vcpus": int, "memory": float, "family": str, "generation": int,
#                "network": str, "storage": str, "description": str}
AWS_INSTANCE_TYPES = {
    # General Purpose - Latest Gen (M7i)
    "m7i.xlarge": {
        "vcpus": 4,
        "memory": 16,
        "family": "general",
        "generation": 7,
        "network": "up to 12.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.2xlarge": {
        "vcpus": 8,
        "memory": 32,
        "family": "general",
        "generation": 7,
        "network": "up to 15 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.4xlarge": {
        "vcpus": 16,
        "memory": 64,
        "family": "general",
        "generation": 7,
        "network": "up to 25 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.8xlarge": {
        "vcpus": 32,
        "memory": 128,
        "family": "general",
        "generation": 7,
        "network": "25 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.12xlarge": {
        "vcpus": 48,
        "memory": 192,
        "family": "general",
        "generation": 7,
        "network": "37.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.16xlarge": {
        "vcpus": 64,
        "memory": 256,
        "family": "general",
        "generation": 7,
        "network": "50 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.24xlarge": {
        "vcpus": 96,
        "memory": 384,
        "family": "general",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Intel Sapphire Rapids"
    },
    "m7i.metal-24xl": {
        "vcpus": 96,
        "memory": 384,
        "family": "general",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen general purpose - Bare metal"
    },

    # General Purpose - Previous Gen (M6i)
    "m6i.xlarge": {
        "vcpus": 4,
        "memory": 16,
        "family": "general",
        "generation": 6,
        "network": "up to 12.5 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.2xlarge": {
        "vcpus": 8,
        "memory": 32,
        "family": "general",
        "generation": 6,
        "network": "up to 15 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.4xlarge": {
        "vcpus": 16,
        "memory": 64,
        "family": "general",
        "generation": 6,
        "network": "up to 25 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.8xlarge": {
        "vcpus": 32,
        "memory": 128,
        "family": "general",
        "generation": 6,
        "network": "25 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.12xlarge": {
        "vcpus": 48,
        "memory": 192,
        "family": "general",
        "generation": 6,
        "network": "37.5 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.16xlarge": {
        "vcpus": 64,
        "memory": 256,
        "family": "general",
        "generation": 6,
        "network": "50 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.24xlarge": {
        "vcpus": 96,
        "memory": 384,
        "family": "general",
        "generation": 6,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.32xlarge": {
        "vcpus": 128,
        "memory": 512,
        "family": "general",
        "generation": 6,
        "network": "100 Gbps",
        "storage": "ebs",
        "description": "General purpose - Intel Ice Lake"
    },
    "m6i.metal": {
        "vcpus": 128,
        "memory": 512,
        "family": "general",
        "generation": 6,
        "network": "100 Gbps",
        "storage": "ebs",
        "description": "General purpose - Bare metal"
    },

    # Compute Optimized - Latest Gen (C7i)
    "c7i.xlarge": {
        "vcpus": 4,
        "memory": 8,
        "family": "compute",
        "generation": 7,
        "network": "up to 12.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.2xlarge": {
        "vcpus": 8,
        "memory": 16,
        "family": "compute",
        "generation": 7,
        "network": "up to 15 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.4xlarge": {
        "vcpus": 16,
        "memory": 32,
        "family": "compute",
        "generation": 7,
        "network": "up to 25 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.8xlarge": {
        "vcpus": 32,
        "memory": 64,
        "family": "compute",
        "generation": 7,
        "network": "25 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.12xlarge": {
        "vcpus": 48,
        "memory": 96,
        "family": "compute",
        "generation": 7,
        "network": "37.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.16xlarge": {
        "vcpus": 64,
        "memory": 128,
        "family": "compute",
        "generation": 7,
        "network": "50 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.24xlarge": {
        "vcpus": 96,
        "memory": 192,
        "family": "compute",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Intel Sapphire Rapids"
    },
    "c7i.metal-24xl": {
        "vcpus": 96,
        "memory": 192,
        "family": "compute",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen compute optimized - Bare metal"
    },

    # Memory Optimized - Latest Gen (R7i)
    "r7i.xlarge": {
        "vcpus": 4,
        "memory": 32,
        "family": "memory",
        "generation": 7,
        "network": "up to 12.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.2xlarge": {
        "vcpus": 8,
        "memory": 64,
        "family": "memory",
        "generation": 7,
        "network": "up to 15 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.4xlarge": {
        "vcpus": 16,
        "memory": 128,
        "family": "memory",
        "generation": 7,
        "network": "up to 25 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.8xlarge": {
        "vcpus": 32,
        "memory": 256,
        "family": "memory",
        "generation": 7,
        "network": "25 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.12xlarge": {
        "vcpus": 48,
        "memory": 384,
        "family": "memory",
        "generation": 7,
        "network": "37.5 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.16xlarge": {
        "vcpus": 64,
        "memory": 512,
        "family": "memory",
        "generation": 7,
        "network": "50 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.24xlarge": {
        "vcpus": 96,
        "memory": 768,
        "family": "memory",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Intel Sapphire Rapids"
    },
    "r7i.metal-24xl": {
        "vcpus": 96,
        "memory": 768,
        "family": "memory",
        "generation": 7,
        "network": "75 Gbps",
        "storage": "ebs",
        "description": "Latest gen memory optimized - Bare metal"
    }
}

def load_metrics(input_file):
    """
    Load metrics from input file

    Args:
        input_file (str): Path to input file

    Returns:
        dict: Loaded metrics
    """
    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metrics file: {e}")
        sys.exit(1)

def analyze_workload_profile(metrics):
    """
    Analyze workload profile to determine if it's CPU or memory bound

    Args:
        metrics (dict): Metrics data

    Returns:
        str: Workload profile ("cpu-bound", "memory-bound", or "balanced")
    """
    try:
        cpu_peak = metrics["metrics"]["cpu"]["peak"]
        memory_peak = metrics["metrics"]["memory"]["peak"]

        # Calculate CPU to memory ratio (cores per GB)
        ratio = cpu_peak / memory_peak if memory_peak > 0 else 0

        # Determine workload profile based on ratio
        if ratio > 0.25:  # More than 1 core per 4 GB
            return "cpu-bound"
        elif ratio < 0.1:  # Less than 1 core per 10 GB
            return "memory-bound"
        else:
            return "balanced"
    except (KeyError, TypeError):
        return "balanced"  # Default to balanced if metrics are missing

def calculate_node_requirements(metrics, redundancy_factor):
    """
    Calculate node requirements based on metrics

    Args:
        metrics (dict): Metrics data
        redundancy_factor (float): Redundancy factor

    Returns:
        dict: Node requirements
    """
    # Extract peak values
    cpu_peak = metrics["metrics"]["cpu"]["peak"]
    memory_peak = metrics["metrics"]["memory"]["peak"]
    pod_count_peak = metrics["metrics"]["pods"]["peak"]

    # Extract resource requests/limits if available
    cpu_requests = metrics["metrics"]["cpu"].get("requests_peak", 0)
    memory_requests = metrics["metrics"]["memory"].get("requests_peak", 0)
    cpu_limits = metrics["metrics"]["cpu"].get("limits_peak", 0)
    memory_limits = metrics["metrics"]["memory"].get("limits_peak", 0)

    # Use the maximum of actual usage and requests for sizing
    # This ensures we account for both actual usage and scheduler constraints
    cpu_sizing_baseline = max(cpu_peak, cpu_requests)
    memory_sizing_baseline = max(memory_peak, memory_requests)

    # Apply redundancy factor
    cpu_required = cpu_sizing_baseline * redundancy_factor
    memory_required = memory_sizing_baseline * redundancy_factor
    pod_count_required = pod_count_peak * redundancy_factor

    # Analyze workload profile
    workload_profile = analyze_workload_profile(metrics)

    # Calculate minimum nodes needed for high availability
    min_nodes_ha = 3

    # We won't make assumptions about vCPU/memory per node here
    # This will be calculated properly in the recommend_instance_types function
    # based on actual instance specifications

    return {
        "cpu_required": cpu_required,
        "memory_required": memory_required,
        "pod_count_required": pod_count_required,
        "cpu_peak": cpu_peak,
        "memory_peak": memory_peak,
        "cpu_requests": cpu_requests,
        "memory_requests": memory_requests,
        "cpu_limits": cpu_limits,
        "memory_limits": memory_limits,
        "workload_profile": workload_profile,
        "min_nodes_ha": min_nodes_ha
    }

def recommend_instance_types(requirements):
    """
    Recommend instance types based on requirements

    Args:
        requirements (dict): Node requirements

    Returns:
        list: Recommended instance types with detailed analysis
    """
    cpu_required = requirements["cpu_required"]
    memory_required = requirements["memory_required"]
    pod_count_required = requirements["pod_count_required"]
    workload_profile = requirements["workload_profile"]
    min_nodes_ha = requirements["min_nodes_ha"]

    # Get the resource usage vs request information for reporting
    cpu_peak = requirements["cpu_peak"]
    memory_peak = requirements["memory_peak"]
    cpu_requests = requirements["cpu_requests"]
    memory_requests = requirements["memory_requests"]
    cpu_limits = requirements["cpu_limits"]
    memory_limits = requirements["memory_limits"]

    # Filter instance types based on workload profile and generation
    filtered_instances = {}

    # First pass: filter by workload profile and get latest generations
    generations = {}
    for name, specs in AWS_INSTANCE_TYPES.items():
        family = specs["family"]
        gen = specs["generation"]

        # Track highest generation for each family
        if family not in generations or gen > generations[family]:
            generations[family] = gen

    # Second pass: filter instances based on workload profile and latest/near-latest gen
    if workload_profile == "cpu-bound":
        # For CPU-bound, prioritize compute instances, then general purpose
        for name, specs in AWS_INSTANCE_TYPES.items():
            family = specs["family"]
            gen = specs["generation"]

            # Include if it's compute family and current or previous generation
            if family == "compute" and gen >= generations[family] - 1:
                filtered_instances[name] = specs
            # Add general purpose as fallback if current generation
            elif family == "general" and gen == generations[family]:
                filtered_instances[name] = specs

    elif workload_profile == "memory-bound":
        # For memory-bound, prioritize memory optimized instances
        for name, specs in AWS_INSTANCE_TYPES.items():
            family = specs["family"]
            gen = specs["generation"]

            # Include if it's memory family and current or previous generation
            if family == "memory" and gen >= generations[family] - 1:
                filtered_instances[name] = specs

    else:  # balanced
        # For balanced workloads, prioritize general purpose instances
        for name, specs in AWS_INSTANCE_TYPES.items():
            family = specs["family"]
            gen = specs["generation"]

            # Include if it's general purpose and current or previous generation
            if family == "general" and gen >= generations[family] - 1:
                filtered_instances[name] = specs

    # If filtered list is too small, add instances from one generation older
    if len(filtered_instances) < 5:
        for name, specs in AWS_INSTANCE_TYPES.items():
            family = specs["family"]
            gen = specs["generation"]

            if family in generations and gen == generations[family] - 1:
                filtered_instances[name] = specs

    # Calculate nodes needed for each instance type
    recommendations = []
    for name, specs in AWS_INSTANCE_TYPES.items():
        # Skip if instance type doesn't match workload profile
        if workload_profile == "cpu-bound" and specs["family"] not in ["compute", "general"]:
            continue
        if workload_profile == "memory-bound" and specs["family"] not in ["memory", "general"]:
            continue

        # Calculate nodes needed based on different factors
        # 1. Based on CPU requirements
        cpu_nodes = math.ceil(cpu_required / specs["vcpus"])

        # 2. Based on Memory requirements
        memory_nodes = math.ceil(memory_required / specs["memory"])

        # 3. Based on Pod count (assuming 110 pods per node as a conservative default)
        # This can be adjusted based on AWS instance type limits
        max_pods_per_node = min(110, 110 * (specs["vcpus"] / 4))  # Scale with vCPUs, capped at 110
        pod_nodes = math.ceil(pod_count_required / max_pods_per_node)

        # Take the maximum of all calculations and ensure minimum HA requirement
        nodes_needed = max(cpu_nodes, memory_nodes, pod_nodes, min_nodes_ha)

        # Calculate utilization percentages
        cpu_utilization = (cpu_required / nodes_needed / specs["vcpus"]) * 100
        memory_utilization = (memory_required / nodes_needed / specs["memory"]) * 100
        pod_utilization = (pod_count_required / nodes_needed / max_pods_per_node) * 100

        # Determine limiting factor
        limiting_factors = []
        if nodes_needed == min_nodes_ha and max(cpu_nodes, memory_nodes, pod_nodes) < min_nodes_ha:
            # If the minimum HA requirement is the limiting factor
            limiting_factors.append("HA Minimum")
        else:
            # Normal resource-based limiting factors
            if cpu_nodes >= memory_nodes and cpu_nodes >= pod_nodes:
                limiting_factors.append("CPU")
            if memory_nodes >= cpu_nodes and memory_nodes >= pod_nodes:
                limiting_factors.append("Memory")
            if pod_nodes >= cpu_nodes and pod_nodes >= memory_nodes:
                limiting_factors.append("Pod Count")

        # Skip if utilization is too low or too high
        # More lenient with low utilization when enforcing min_nodes_ha
        if nodes_needed == min_nodes_ha:
            # For minimum HA deployments, be very lenient with lower bound
            # Only filter out configurations with too high utilization
            if cpu_utilization > 80 or memory_utilization > 80:
                continue
            # No lower bound for minimum HA deployments to accommodate small workloads
        else:
            # For larger deployments, aim for better efficiency
            if cpu_utilization < 30 or cpu_utilization > 80:
                continue
            if memory_utilization < 30 or memory_utilization > 80:
                continue

        # Calculate efficiency score
        # Higher score = better fit
        # Factors:
        # 1. Generation (newer = better)
        # 2. Resource utilization (closer to target = better)
        # 3. Network capability (higher = better)
        # 4. Node count (closer to minimum = better)
        generation_score = specs["generation"] * 10  # Newer generations get higher score

        # Target utilization is 60-70% for normal deployments
        # For minimum HA deployments with small workloads, we're more lenient
        if nodes_needed == min_nodes_ha:
            # For minimum HA deployments, adjust target utilization based on workload size
            # For very small workloads, even low utilization is acceptable
            cpu_target = max(20, min(65, cpu_utilization * 1.2))  # Adjust target closer to actual utilization
            memory_target = max(20, min(65, memory_utilization * 1.2))

            cpu_util_score = 100 - min(abs(cpu_target - cpu_utilization), 50)  # Cap penalty at 50 points
            memory_util_score = 100 - min(abs(memory_target - memory_utilization), 50)
        else:
            # Standard scoring for larger deployments
            cpu_util_score = 100 - abs(65 - cpu_utilization)
            memory_util_score = 100 - abs(65 - memory_utilization)

        # Node count score - prefer configurations closer to min_nodes_ha
        node_count_score = 100 - ((nodes_needed - min_nodes_ha) * 10)  # -10 points per additional node
        node_count_score = max(0, node_count_score)  # Don't go below 0

        # Network score based on capability
        network_score = 0
        if "Gbps" in specs["network"]:
            try:
                network_speed = float(specs["network"].split()[0].split("-")[0])
                network_score = min(network_speed * 2, 100)  # Cap at 100
            except (ValueError, IndexError):
                pass

        # Combine scores with weights
        efficiency_score = (
            generation_score * 0.25 +      # 25% weight for generation
            cpu_util_score * 0.2 +         # 20% weight for CPU utilization
            memory_util_score * 0.2 +      # 20% weight for memory utilization
            network_score * 0.2 +          # 20% weight for network capability
            node_count_score * 0.15        # 15% weight for node count
        )

        # Add to recommendations
        recommendations.append({
            "instance_type": name,
            "vcpus": specs["vcpus"],
            "memory_gb": specs["memory"],
            "description": specs["description"],
            "network": specs["network"],
            "generation": specs["generation"],
            "nodes_needed": nodes_needed,
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "pod_utilization": pod_utilization,
            "limiting_factors": limiting_factors,
            "cpu_nodes": cpu_nodes,
            "memory_nodes": memory_nodes,
            "pod_nodes": pod_nodes,
            "total_vcpus": specs["vcpus"] * nodes_needed,
            "total_memory_gb": specs["memory"] * nodes_needed,
            "max_pods_per_node": max_pods_per_node,
            "total_pods_capacity": max_pods_per_node * nodes_needed,
            "efficiency_score": efficiency_score,
            "resource_analysis": {
                "cpu": {
                    "peak_usage": cpu_peak,
                    "requests": cpu_requests,
                    "limits": cpu_limits,
                    "sizing_basis": "requests" if cpu_requests > cpu_peak else "usage"
                },
                "memory": {
                    "peak_usage_gb": memory_peak,
                    "requests_gb": memory_requests,
                    "limits_gb": memory_limits,
                    "sizing_basis": "requests" if memory_requests > memory_peak else "usage"
                }
            }
        })

    # Sort recommendations by efficiency score (higher is better)
    # This considers: node count, resource utilization, generation, and network capability
    recommendations.sort(key=lambda x: (-x["efficiency_score"], x["nodes_needed"]))

    # Return top 5 recommendations
    return recommendations[:5]

def calculate_storage_requirements(metrics, redundancy_factor):
    """
    Calculate storage requirements based on metrics

    Args:
        metrics (dict): Metrics data
        redundancy_factor (float): Redundancy factor

    Returns:
        dict: Storage requirements
    """
    try:
        # Extract storage values
        storage_peak = metrics["metrics"]["storage"]["peak"]
        storage_avg = metrics["metrics"]["storage"]["average"]

        # Apply redundancy factor
        storage_required = storage_peak * redundancy_factor

        # Round up to nearest 10 GB
        storage_required = round(storage_required / 10) * 10 + 10

        return {
            "storage_required_gb": storage_required,
            "storage_avg_gb": storage_avg
        }
    except (KeyError, TypeError) as e:
        print(f"Error calculating storage requirements: {e}")
        return {"storage_required_gb": 100, "storage_avg_gb": 50}  # Default values

def generate_recommendations(metrics, redundancy_factor):
    """
    Generate sizing recommendations

    Args:
        metrics (dict): Metrics data
        redundancy_factor (float): Redundancy factor

    Returns:
        dict: Sizing recommendations
    """
    # Calculate requirements
    node_requirements = calculate_node_requirements(metrics, redundancy_factor)
    storage_requirements = calculate_storage_requirements(metrics, redundancy_factor)

    # Get instance recommendations
    instance_recommendations = recommend_instance_types(node_requirements)

    # Get the minimum nodes required from the first recommendation
    min_nodes_required = instance_recommendations[0]["nodes_needed"] if instance_recommendations else node_requirements["min_nodes_ha"]

    # Extract resource requests/limits if available
    cpu_requests = metrics["metrics"]["cpu"].get("requests_peak", 0)
    memory_requests = metrics["metrics"]["memory"].get("requests_peak", 0)
    cpu_limits = metrics["metrics"]["cpu"].get("limits_peak", 0)
    memory_limits = metrics["metrics"]["memory"].get("limits_peak", 0)

    # Generate recommendations
    recommendations = {
        "metadata": {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_collection_date": metrics["metadata"]["collection_date"],
            "metrics_period_days": metrics["metadata"]["days"],
            "redundancy_factor": redundancy_factor
        },
        "cluster_metrics": {
            "cpu_peak_cores": metrics["metrics"]["cpu"]["peak"],
            "cpu_avg_cores": metrics["metrics"]["cpu"]["average"],
            "memory_peak_gb": metrics["metrics"]["memory"]["peak"],
            "memory_avg_gb": metrics["metrics"]["memory"]["average"],
            "pod_count_peak": metrics["metrics"]["pods"]["peak"],
            "pod_count_avg": metrics["metrics"]["pods"]["average"],
            "storage_peak_gb": metrics["metrics"]["storage"]["peak"],
            "storage_avg_gb": metrics["metrics"]["storage"]["average"],
            "cpu_requests_peak": cpu_requests,
            "memory_requests_peak_gb": memory_requests,
            "cpu_limits_peak": cpu_limits,
            "memory_limits_peak_gb": memory_limits
        },
        "requirements": {
            "cpu_required_cores": node_requirements["cpu_required"],
            "memory_required_gb": node_requirements["memory_required"],
            "pod_count_required": node_requirements["pod_count_required"],
            "storage_required_gb": storage_requirements["storage_required_gb"],
            "workload_profile": node_requirements["workload_profile"],
            "sizing_basis": {
                "cpu": "requests" if cpu_requests > metrics["metrics"]["cpu"]["peak"] else "usage",
                "memory": "requests" if memory_requests > metrics["metrics"]["memory"]["peak"] else "usage"
            }
        },
        "instance_recommendations": instance_recommendations,
        "general_recommendations": {
            "minimum_worker_nodes": node_requirements["min_nodes_ha"],  # Always recommend at least 3 nodes for HA
            "recommended_worker_nodes": min_nodes_required,
            "storage_recommendation": f"{storage_requirements['storage_required_gb']:.0f} GB minimum"
        }
    }

    return recommendations

def format_text_report(recommendations):
    """
    Format recommendations as text report

    Args:
        recommendations (dict): Sizing recommendations

    Returns:
        str: Formatted text report
    """
    report = []

    # Add header
    report.append("=" * 80)
    report.append("ROSA SIZING RECOMMENDATIONS REPORT")
    report.append("=" * 80)

    # Add metadata
    report.append("\nREPORT METADATA:")
    report.append(f"  Generation Date: {recommendations['metadata']['generation_date']}")
    report.append(f"  Metrics Collection Date: {recommendations['metadata']['metrics_collection_date']}")
    report.append(f"  Metrics Period: {recommendations['metadata']['metrics_period_days']} days")
    report.append(f"  Redundancy Factor: {recommendations['metadata']['redundancy_factor']}")

    # Add cluster metrics
    report.append("\nCLUSTER METRICS:")
    report.append(f"  CPU Usage: {recommendations['cluster_metrics']['cpu_peak_cores']:.2f} cores peak, {recommendations['cluster_metrics']['cpu_avg_cores']:.2f} cores average")
    report.append(f"  Memory Usage: {recommendations['cluster_metrics']['memory_peak_gb']:.2f} GB peak, {recommendations['cluster_metrics']['memory_avg_gb']:.2f} GB average")
    report.append(f"  Pod Count: {int(recommendations['cluster_metrics']['pod_count_peak'])} peak, {int(recommendations['cluster_metrics']['pod_count_avg'])} average")
    report.append(f"  Storage Usage: {recommendations['cluster_metrics']['storage_peak_gb']:.2f} GB peak, {recommendations['cluster_metrics']['storage_avg_gb']:.2f} GB average")

    # Add requirements
    report.append("\nCLUSTER REQUIREMENTS (with redundancy):")
    report.append(f"  CPU Required: {recommendations['requirements']['cpu_required_cores']:.2f} cores")
    report.append(f"  Memory Required: {recommendations['requirements']['memory_required_gb']:.2f} GB")
    report.append(f"  Pod Count Required: {int(recommendations['requirements']['pod_count_required'])}")
    report.append(f"  Storage Required: {recommendations['requirements']['storage_required_gb']} GB")
    report.append(f"  Workload Profile: {recommendations['requirements']['workload_profile']}")

    # Add instance recommendations
    report.append("\nRECOMMENDED INSTANCE TYPES:")

    for i, rec in enumerate(recommendations["instance_recommendations"], 1):
        report.append(f"\n  OPTION {i}: {rec['instance_type']} ({rec['description']})")
        report.append(f"    vCPUs: {rec['vcpus']} per node, Memory: {rec['memory_gb']} GB per node")
        # Get the limiting factor as a string
        limiting_factor = ", ".join(rec['limiting_factors']) if 'limiting_factors' in rec else "Unknown"
        report.append(f"    Nodes Required: {rec['nodes_needed']} (limited by {limiting_factor})")
        report.append(f"    Utilization: CPU {rec['cpu_utilization']:.1f}%, Memory {rec['memory_utilization']:.1f}%")
        report.append(f"    Total Capacity: {rec['total_vcpus']} vCPUs, {rec['total_memory_gb']} GB Memory")

    # Add general recommendations
    report.append("\nGENERAL RECOMMENDATIONS:")
    report.append(f"  Minimum Worker Nodes: {recommendations['general_recommendations']['minimum_worker_nodes']}")
    report.append(f"  Recommended Worker Nodes: {recommendations['general_recommendations']['recommended_worker_nodes']} (for high availability)")
    report.append(f"  Storage Recommendation: {recommendations['general_recommendations']['storage_recommendation']}")

    # Add footer with notes
    report.append("\nNOTES:")
    report.append("  - ROSA includes managed control plane and infra nodes (typically on m5.xlarge instances)")
    report.append("  - Recommendations focus on worker nodes which handle application workloads")
    report.append("  - 3+ worker nodes recommended for high availability")
    report.append("  - Multiple availability zones recommended for production environments")
    report.append("  - For small workloads, utilization may be low due to the 3-node minimum requirement")
    report.append("  - For larger deployments, utilization targets are 30-80% for both CPU and memory")
    report.append("  - Use AWS Pricing Calculator to estimate costs based on these recommendations")

    return "\n".join(report)

def main():
    """Main function to calculate sizing"""
    parser = argparse.ArgumentParser(description="OpenShift ROSA Sizing Tool - Sizing Calculator")

    # Required arguments
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input metrics file (default: {DEFAULT_INPUT})")

    # Optional arguments
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--format", choices=["json", "text"], default=DEFAULT_FORMAT, help=f"Output format (default: {DEFAULT_FORMAT})")
    parser.add_argument("--redundancy", type=float, default=DEFAULT_REDUNDANCY, help=f"Redundancy factor (default: {DEFAULT_REDUNDANCY})")

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from {args.input}...")
    metrics = load_metrics(args.input)

    # Generate recommendations
    print("Generating sizing recommendations...")
    recommendations = generate_recommendations(metrics, args.redundancy)

    # Create output
    if args.format == "json":
        output_content = json.dumps(recommendations, indent=2)
    else:  # text
        output_content = format_text_report(recommendations)

    # Create backup of existing output file if it exists
    if os.path.exists(args.output):
        backup_file = f"{args.output}.bak"
        print(f"Creating backup of existing output file: {backup_file}")
        os.rename(args.output, backup_file)

    # Write recommendations to output file
    with open(args.output, "w") as f:
        f.write(output_content)

    print(f"Recommendations saved to {args.output}")

    # Print summary
    print("\nRecommendation Summary:")
    print(f"Workload Profile: {recommendations['requirements']['workload_profile']}")
    print(f"Top Instance Recommendation: {recommendations['instance_recommendations'][0]['instance_type']} with {recommendations['instance_recommendations'][0]['nodes_needed']} nodes")
    print(f"Minimum Worker Nodes: {recommendations['general_recommendations']['minimum_worker_nodes']}")
    print(f"Recommended Worker Nodes: {recommendations['general_recommendations']['recommended_worker_nodes']} (for high availability)")

if __name__ == "__main__":
    main()
