#!/usr/bin/env python3
"""
OpenShift ROSA Sizing Tool - Sizing Calculator Script
Version: 2.3

This script analyzes collected metrics from an existing OpenShift cluster
to provide ROSA sizing recommendations including optimal AWS instance types,
node counts, and storage requirements. It contrasts the observed workload
metrics with the calculated requirements and recommended ROSA configuration.

Note: The input metrics file should contain peak/average usage and request data.
      It typically does NOT include the total capacity of the source cluster.
      This tool sizes the *target* ROSA cluster based on the *observed workload demand*.
"""

import argparse
import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, Any, List # Added type hints

# Default configuration
DEFAULT_INPUT: str = "cluster_metrics.json"
DEFAULT_OUTPUT: str = "rosa_sizing.json"
DEFAULT_FORMAT: str = "json"
DEFAULT_REDUNDANCY: float = 1.3
MIN_HA_WORKER_NODES: int = 3 # Minimum nodes for High Availability
DEFAULT_MAX_PODS_PER_NODE: int = 150 # Default Kubernetes/OpenShift limit per node
MIN_STORAGE_GB: int = 100 # Minimum recommended storage capacity for PVs
TARGET_UTILIZATION_PERCENT: int = 75 # Target utilization for scoring

# Weights for efficiency scoring (total should sum to 1.0)
SCORE_WEIGHTS: Dict[str, float] = {
    "generation": 0.25,
    "cpu_util": 0.20,
    "memory_util": 0.20,
    "network": 0.20,
    "node_count": 0.15,
}

# AWS instance type definitions for ROSA worker nodes
# Format: name: {"vcpus": int, "memory": float, "family": str, "generation": int,
#                "network": str, "storage": str, "description": str}
# Added Graviton (ARM) instances and ensured bare metal types are included
AWS_INSTANCE_TYPES: Dict[str, Dict[str, Any]] = {
    # General Purpose - Latest Gen (M7i)
    "m7i.xlarge": { "vcpus": 4, "memory": 16, "family": "general", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.2xlarge": { "vcpus": 8, "memory": 32, "family": "general", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.4xlarge": { "vcpus": 16, "memory": 64, "family": "general", "generation": 7, "network": "up to 25 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.8xlarge": { "vcpus": 32, "memory": 128, "family": "general", "generation": 7, "network": "25 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.12xlarge": { "vcpus": 48, "memory": 192, "family": "general", "generation": 7, "network": "37.5 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.16xlarge": { "vcpus": 64, "memory": 256, "family": "general", "generation": 7, "network": "50 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.24xlarge": { "vcpus": 96, "memory": 384, "family": "general", "generation": 7, "network": "75 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Intel Sapphire Rapids" },
    "m7i.metal-24xl": { "vcpus": 96, "memory": 384, "family": "general", "generation": 7, "network": "75 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Bare metal" },
    # General Purpose - Previous Gen (M6i)
    "m6i.xlarge": { "vcpus": 4, "memory": 16, "family": "general", "generation": 6, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "General purpose - Intel Ice Lake" },
    "m6i.2xlarge": { "vcpus": 8, "memory": 32, "family": "general", "generation": 6, "network": "up to 15 Gbps", "storage": "ebs", "description": "General purpose - Intel Ice Lake" },
    "m6i.4xlarge": { "vcpus": 16, "memory": 64, "family": "general", "generation": 6, "network": "up to 25 Gbps", "storage": "ebs", "description": "General purpose - Intel Ice Lake" },
    "m6i.8xlarge": { "vcpus": 32, "memory": 128, "family": "general", "generation": 6, "network": "25 Gbps", "storage": "ebs", "description": "General purpose - Intel Ice Lake" },
    "m6i.metal": { "vcpus": 128, "memory": 512, "family": "general", "generation": 6, "network": "100 Gbps", "storage": "ebs", "description": "General purpose - Bare metal" },
    # General Purpose - Latest Gen (M7g) - Graviton 3
    "m7g.xlarge": { "vcpus": 4, "memory": 16, "family": "general", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Graviton 3" },
    "m7g.2xlarge": { "vcpus": 8, "memory": 32, "family": "general", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Graviton 3" },
    "m7g.4xlarge": { "vcpus": 16, "memory": 64, "family": "general", "generation": 7, "network": "up to 25 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Graviton 3" },
    "m7g.8xlarge": { "vcpus": 32, "memory": 128, "family": "general", "generation": 7, "network": "25 Gbps", "storage": "ebs", "description": "Latest gen general purpose - Graviton 3" },

    # Compute Optimized - Latest Gen (C7i)
    "c7i.xlarge": { "vcpus": 4, "memory": 8, "family": "compute", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Intel Sapphire Rapids" },
    "c7i.2xlarge": { "vcpus": 8, "memory": 16, "family": "compute", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Intel Sapphire Rapids" },
    "c7i.4xlarge": { "vcpus": 16, "memory": 32, "family": "compute", "generation": 7, "network": "up to 25 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Intel Sapphire Rapids" },
    "c7i.metal-24xl": { "vcpus": 96, "memory": 192, "family": "compute", "generation": 7, "network": "75 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Bare metal" },
    # Compute Optimized - Latest Gen (C7g) - Graviton 3
    "c7g.xlarge": { "vcpus": 4, "memory": 8, "family": "compute", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Graviton 3" },
    "c7g.2xlarge": { "vcpus": 8, "memory": 16, "family": "compute", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen compute optimized - Graviton 3" },

    # Memory Optimized - Latest Gen (R7i)
    "r7i.xlarge": { "vcpus": 4, "memory": 32, "family": "memory", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Intel Sapphire Rapids" },
    "r7i.2xlarge": { "vcpus": 8, "memory": 64, "family": "memory", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Intel Sapphire Rapids" },
     "r7i.4xlarge": { "vcpus": 16, "memory": 128, "family": "memory", "generation": 7, "network": "up to 25 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Intel Sapphire Rapids" },
    "r7i.metal-24xl": { "vcpus": 96, "memory": 768, "family": "memory", "generation": 7, "network": "75 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Bare metal" },
    # Memory Optimized - Latest Gen (R7g) - Graviton 3
    "r7g.xlarge": { "vcpus": 4, "memory": 32, "family": "memory", "generation": 7, "network": "up to 12.5 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Graviton 3" },
    "r7g.2xlarge": { "vcpus": 8, "memory": 64, "family": "memory", "generation": 7, "network": "up to 15 Gbps", "storage": "ebs", "description": "Latest gen memory optimized - Graviton 3" },

    # ... (you can add more instance types as needed) ...
}


def load_metrics(input_file: str) -> Dict[str, Any]:
    """
    Load metrics from input file

    Args:
        input_file (str): Path to input file

    Returns:
        dict: Loaded metrics
    """
    try:
        with open(input_file, "r") as f:
            metrics_data = json.load(f)

        # Basic validation for required top-level keys
        required_top_keys = ["metadata", "metrics"]
        for key in required_top_keys:
            if key not in metrics_data:
                 raise ValueError(f"Missing required key in metrics file: top-level '{key}'")

        # Basic validation for required metric types and peak/average values
        required_metric_types = ["cpu", "memory", "pods", "storage"]
        if "metrics" in metrics_data:
             for metric_type in required_metric_types:
                  if metric_type not in metrics_data["metrics"]:
                       print(f"Warning: Missing metrics for '{metric_type}'. Calculations may be affected.", file=sys.stderr)
                       # Add dummy data to prevent errors later, but warn
                       metrics_data["metrics"][metric_type] = {"peak": 0, "average": 0}
                  else:
                    if "peak" not in metrics_data["metrics"][metric_type] or "average" not in metrics_data["metrics"][metric_type]:
                        print(f"Warning: Missing peak or average data for '{metric_type}'. Using 0.", file=sys.stderr)
                        metrics_data["metrics"][metric_type]["peak"] = metrics_data["metrics"][metric_type].get("peak", 0)
                        metrics_data["metrics"][metric_type]["average"] = metrics_data["metrics"][metric_type].get("average", 0)
                    if "requests_peak" not in metrics_data["metrics"][metric_type]:
                         metrics_data["metrics"][metric_type]["requests_peak"] = 0
                    if "limits_peak" not in metrics_data["metrics"][metric_type]:
                         metrics_data["metrics"][metric_type]["limits_peak"] = 0


        return metrics_data

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing metrics file {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid metrics file format: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading metrics: {e}", file=sys.stderr)
         sys.exit(1)


def analyze_workload_profile(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze workload profile to determine if it's CPU or memory bound,
    using peaks and requests from the observed metrics.

    Args:
        metrics (dict): Metrics data

    Returns:
        dict: Workload profile analysis including calculated ratio and profile type.
    """
    try:
        # Get metrics safely using .get() chain
        cpu_metrics = metrics.get("metrics", {}).get("cpu", {})
        memory_metrics = metrics.get("metrics", {}).get("memory", {})

        cpu_peak = cpu_metrics.get("peak", 0)
        memory_peak = memory_metrics.get("peak", 0)
        cpu_requests = cpu_metrics.get("requests_peak", 0)
        memory_requests = memory_metrics.get("requests_peak", 0)

        # Use the maximum of actual usage and requests for analysis baseline
        cpu_sizing_baseline = max(cpu_peak, cpu_requests)
        memory_sizing_baseline = max(memory_peak, memory_requests)

        # Calculate CPU to memory ratio (cores per GB) using sizing baseline
        # Avoid division by zero: if memory is 0, handle based on CPU
        ratio = 0.0
        if memory_sizing_baseline > 0:
            ratio = cpu_sizing_baseline / memory_sizing_baseline
        elif cpu_sizing_baseline > 0:
            # If CPU is positive but memory is zero, treat as highly CPU bound
            ratio = float('inf')

        # Determine workload profile based on ratio
        # Thresholds (cores per GB):
        # - ratio > 0.25 (approx 1 core per 4 GB): Tends towards CPU-bound
        # - ratio < 0.10 (approx 1 core per 10 GB): Tends towards Memory-bound
        # - Values between indicate balanced
        profile_type: str
        if ratio > 0.25:
            profile_type = "cpu-bound"
        elif ratio > 0 and ratio < 0.10: # Ensure ratio is positive if memory is > 0
            profile_type = "memory-bound"
        else: # Includes ratio == 0 and ratio == inf case with cpu > 0 and mem == 0
            # Refine: if CPU > 0 and Memory == 0, it's CPU bound
            if cpu_sizing_baseline > 0 and memory_sizing_baseline == 0:
                 profile_type = "cpu-bound" # Effectively infinite ratio
            elif cpu_sizing_baseline == 0 and memory_sizing_baseline > 0:
                 profile_type = "memory-bound" # Effectively zero ratio
            else: # Both are zero or very small
                 profile_type = "balanced" # Minimal or no workload observed


        return {
            "ratio_cores_per_gb": ratio if ratio != float('inf') else "Infinity", # Represent Inf clearly
            "profile_type": profile_type,
            "sizing_basis_from_metrics": {
                "cpu": "requests" if cpu_requests > cpu_peak else ("usage" if cpu_peak > 0 else "zero"),
                "memory": "requests" if memory_requests > memory_peak else ("usage" if memory_peak > 0 else "zero")
            }
        }
    except Exception as e:
        # Handle any unexpected errors during profile analysis
        print(f"Warning: Could not fully analyze workload profile: {e}. Defaulting to balanced.", file=sys.stderr)
        return {
             "ratio_cores_per_gb": "Unknown",
             "profile_type": "balanced",
             "sizing_basis_from_metrics": { "cpu": "error", "memory": "error"}
        }


def calculate_sizing_requirements(metrics: Dict[str, Any], redundancy_factor: float) -> Dict[str, Any]:
    """
    Calculate required resources in the target ROSA cluster based on observed
    metrics (peak/requests) and redundancy factor.

    Args:
        metrics (dict): Metrics data from the source cluster.
        redundancy_factor (float): Redundancy factor to apply.

    Returns:
        dict: Calculated resource requirements for the target ROSA cluster.
    """
    try:
        # Get metrics safely using .get() chain
        cpu_metrics = metrics.get("metrics", {}).get("cpu", {})
        memory_metrics = metrics.get("metrics", {}).get("memory", {})
        pods_metrics = metrics.get("metrics", {}).get("pods", {})
        storage_metrics = metrics.get("metrics", {}).get("storage", {})

        # Default to 0 if keys are missing
        cpu_peak = cpu_metrics.get("peak", 0)
        memory_peak = memory_metrics.get("peak", 0)
        pod_count_peak = pods_metrics.get("peak", 0)
        storage_peak = storage_metrics.get("peak", 0)

        cpu_requests = cpu_metrics.get("requests_peak", 0)
        memory_requests = memory_metrics.get("requests_peak", 0)
        cpu_limits = cpu_metrics.get("limits_peak", 0) # Keep limits for info
        memory_limits = memory_metrics.get("limits_peak", 0) # Keep limits for info

        # Use the maximum of actual usage peak and requests peak for sizing basis
        # This ensures we account for both actual historical high watermarks
        # and resources potentially reserved by applications.
        cpu_sizing_baseline = max(cpu_peak, cpu_requests)
        memory_sizing_baseline = max(memory_peak, memory_requests)

        # Apply redundancy factor to sizing baseline
        cpu_required = cpu_sizing_baseline * redundancy_factor
        memory_required = memory_sizing_baseline * redundancy_factor
        pod_count_required = pod_count_peak * redundancy_factor
        storage_required_raw = storage_peak * redundancy_factor

        # Round up storage to nearest 10 GB and ensure minimum
        storage_required_gb = math.ceil(storage_required_raw / 10) * 10
        storage_required_gb = max(storage_required_gb, MIN_STORAGE_GB)

        workload_profile_analysis = analyze_workload_profile(metrics)

        return {
            "cpu_required_cores": cpu_required,
            "memory_required_gb": memory_required,
            "pod_count_required": pod_count_required,
            "storage_required_gb": storage_required_gb,
            "workload_profile_analysis": workload_profile_analysis,
            "redundancy_factor": redundancy_factor,
            # Include raw peaks/requests/limits here for easy access in formatting/comparison
            "raw_metrics_peak": {
                 "cpu_usage": cpu_peak, "memory_usage": memory_peak, "pods": pod_count_peak, "storage_usage": storage_peak,
                 "cpu_requests": cpu_requests, "memory_requests": memory_requests,
                 "cpu_limits": cpu_limits, "memory_limits": memory_limits,
            },
            "sizing_baseline": { # The values *before* redundancy
                "cpu": cpu_sizing_baseline, "memory": memory_sizing_baseline, "pods": pod_count_peak # Pods baseline is just peak
            }
         }

    except Exception as e: # Catch potential errors during calculation
         print(f"Error calculating sizing requirements: {e}", file=sys.stderr)
         # Return default/zero requirements if critical metrics are missing
         return {
            "cpu_required_cores": 0, "memory_required_gb": 0, "pod_count_required": 0,
            "storage_required_gb": MIN_STORAGE_GB,
            "workload_profile_analysis": {"ratio_cores_per_gb": "Unknown", "profile_type": "unknown", "sizing_basis_from_metrics": {"cpu": "error", "memory": "error"}},
            "redundancy_factor": redundancy_factor,
            "raw_metrics_peak": {"cpu_usage": 0, "memory_usage": 0, "pods": 0, "storage_usage": 0, "cpu_requests": 0, "memory_requests": 0, "cpu_limits": 0, "memory_limits": 0},
            "sizing_baseline": {"cpu": 0, "memory": 0, "pods": 0}
         }


def calculate_instance_score(instance_specs: Dict[str, Any], requirements: Dict[str, Any], nodes_needed: int) -> float:
    """
    Calculates an efficiency score for a given instance type and node count
    based on how well it meets the requirements.

    Args:
        instance_specs (dict): Specifications of the instance type.
        requirements (dict): Calculated requirements (CPU, Memory, Pods) including redundancy.
        nodes_needed (int): The calculated number of nodes for this instance type.

    Returns:
        float: The calculated efficiency score.
    """
    cpu_required = requirements["cpu_required_cores"]
    memory_required = requirements["memory_required_gb"]
    pod_count_required = requirements["pod_count_required"]
    min_nodes_ha = MIN_HA_WORKER_NODES

    # Calculate total capacity provided by this instance type * nodes_needed
    total_instance_vcpus = instance_specs.get("vcpus", 0) * nodes_needed
    total_instance_memory_gb = instance_specs.get("memory", 0) * nodes_needed

    # Note: Pod capacity per node might vary slightly based on specific AWS instance limits
    # and OpenShift configuration, but 110 is a standard and safe default limit.
    max_pods_per_node = min(DEFAULT_MAX_PODS_PER_NODE, DEFAULT_MAX_PODS_PER_NODE * (instance_specs.get("vcpus", 0) / 4)) # Simple scaling cap based on vCPU
    total_instance_pod_capacity = max_pods_per_node * nodes_needed


    # Calculate utilization percentages based on REQUIRED resources vs. PROVIDED capacity
    # Ensure division by zero is handled
    cpu_utilization = (cpu_required / total_instance_vcpus) * 100 if total_instance_vcpus > 0 else 0
    memory_utilization = (memory_required / total_instance_memory_gb) * 100 if total_instance_memory_gb > 0 else 0
    pod_utilization = (pod_count_required / total_instance_pod_capacity) * 100 if total_instance_pod_capacity > 0 else 0


    # --- Scoring Components ---

    # 1. Generation score: Newer generations score higher (e.g., gen 7 = 70 points)
    generation_score = instance_specs.get("generation", 0) * 10

    # 2. Utilization scores: Closer to target utilization scores higher
    # Penalizes deviation from TARGET_UTILIZATION_PERCENT.
    # Penalty capped to avoid extreme scores for very low utilization in small clusters
    max_penalty = 50 # Cap deviation penalty at 50 points
    cpu_util_score = 100 - min(abs(TARGET_UTILIZATION_PERCENT - cpu_utilization), max_penalty)
    memory_util_score = 100 - min(abs(TARGET_UTILIZATION_PERCENT - memory_utilization), max_penalty)

    # 3. Node count score: Fewer nodes (closer to min_ha) score higher
    # Penalize extra nodes beyond the HA minimum
    node_count_score = 100 - ((nodes_needed - min_nodes_ha) * 10)
    node_count_score = max(0, node_count_score) # Score cannot be negative

    # 4. Network score: Higher network capability scores higher
    # Extract numerical speed (e.g., "up to 12.5 Gbps" -> 12.5, "100 Gbps" -> 100)
    # Scale network speed to a score out of 100
    network_score = 0
    network_str = instance_specs.get("network", "0 Gbps").lower()
    try:
        network_speed_str = network_str.replace("up to ", "").split()[0].split("-")[0]
        network_speed = float(network_speed_str)
        # Simple scaling: 1 Gbps -> 1 point, 100 Gbps -> 100 points. Cap at 100.
        network_score = min(network_speed, 100)
    except (ValueError, IndexError):
        pass # If parsing fails, network score remains 0


    # Combine scores using weights
    total_weight_sum = sum(SCORE_WEIGHTS.values())
    if total_weight_sum == 0: total_weight_sum = 1 # Avoid division by zero

    efficiency_score = (
        generation_score * SCORE_WEIGHTS.get("generation", 0) +
        cpu_util_score * SCORE_WEIGHTS.get("cpu_util", 0) +
        memory_util_score * SCORE_WEIGHTS.get("memory_util", 0) +
        network_score * SCORE_WEIGHTS.get("network", 0) +
        node_count_score * SCORE_WEIGHTS.get("node_count", 0)
    ) / total_weight_sum # Normalize by total weight


    return efficiency_score


def recommend_worker_node_options(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recommend worker node instance types and counts based on calculated requirements.

    Evaluates all relevant instance types and ranks them by an efficiency score.

    Args:
        requirements (dict): Calculated resource requirements (including redundancy).

    Returns:
        list: Ranked list of recommended instance configurations.
    """
    cpu_required = requirements["cpu_required_cores"]
    memory_required = requirements["memory_required_gb"]
    pod_count_required = requirements["pod_count_required"]
    min_nodes_ha = MIN_HA_WORKER_NODES

    all_potential_recommendations: List[Dict[str, Any]] = []

    # Iterate through all available instance types
    # No filtering by workload family here, let the score determine suitability
    for name, specs in AWS_INSTANCE_TYPES.items():

        vcpus_per_node = specs.get("vcpus", 0)
        memory_gb_per_node = specs.get("memory", 0)

        # Calculate nodes needed based purely on resource requirements (before HA minimum)
        # Handle cases where required resources or instance capacity might be zero
        cpu_nodes_basis = math.ceil(cpu_required / vcpus_per_node) if vcpus_per_node > 0 and cpu_required > 0 else (1 if cpu_required > 0 else 0) # Need at least 1 node if CPU > 0 and instance has 0 CPU
        memory_nodes_basis = math.ceil(memory_required / memory_gb_per_node) if memory_gb_per_node > 0 and memory_required > 0 else (1 if memory_required > 0 else 0) # Need at least 1 node if Memory > 0 and instance has 0 Memory
        # Pods calculation uses the capped max_pods_per_node
        max_pods_per_node = min(DEFAULT_MAX_PODS_PER_NODE, DEFAULT_MAX_PODS_PER_NODE * (vcpus_per_node / 4)) # Simple scaling cap
        pod_nodes_basis = math.ceil(pod_count_required / max_pods_per_node) if max_pods_per_node > 0 and pod_count_required > 0 else (1 if pod_count_required > 0 else 0) # Need at least 1 node if Pods > 0 and capacity is 0

        # Determine the *theoretical* nodes needed based on the maximum resource/pod need
        max_basis_nodes = max(cpu_nodes_basis, memory_nodes_basis, pod_nodes_basis)

        # The actual nodes needed is the maximum of the theoretical need and the HA minimum
        nodes_needed = max(max_basis_nodes, min_nodes_ha)

        # Skip instances that theoretically cannot meet the requirements (e.g., 0 vCPU instance for a CPU req > 0)
        # Or if the required resources are > 0 but the calculated nodes needed is still 0 (should be caught by max_basis_nodes logic)
        if (cpu_required > 0 and vcpus_per_node == 0) or \
           (memory_required > 0 and memory_gb_per_node == 0) or \
           (pod_count_required > 0 and max_pods_per_node == 0):
             continue


        # Calculate utilization percentages for the calculated node count
        total_instance_vcpus = vcpus_per_node * nodes_needed
        total_instance_memory_gb = memory_gb_per_node * nodes_needed
        total_instance_pod_capacity = max_pods_per_node * nodes_needed # Total capacity for pod count check

        # Ensure division by zero is handled for utilization calculation
        cpu_utilization = (cpu_required / total_instance_vcpus) * 100 if total_instance_vcpus > 0 else 0
        memory_utilization = (memory_required / total_instance_memory_gb) * 100 if total_instance_memory_gb > 0 else 0
        pod_utilization = (pod_count_required / total_instance_pod_capacity) * 100 if total_instance_pod_capacity > 0 else 0


        # Skip configurations with extremely high utilization (>95%) as they lack headroom and might indicate undersizing
        if cpu_utilization > 95.01 or memory_utilization > 95.01: # Use slight tolerance for float comparison
            # Optionally print a warning here about extreme utilization
            # print(f"Warning: Skipping {name} x{nodes_needed} due to extremely high utilization (CPU: {cpu_utilization:.1f}%, Mem: {memory_utilization:.1f}%)", file=sys.stderr)
            continue

        # Determine limiting factor(s) - This explains *why* 'max_basis_nodes' was calculated
        limiting_factors = []
        # Only list resource factors if requirements are > 0
        if cpu_required > 0 and cpu_nodes_basis >= max_basis_nodes and cpu_nodes_basis > 0:
             limiting_factors.append("CPU")
        if memory_required > 0 and memory_nodes_basis >= max_basis_nodes and memory_nodes_basis > 0:
             limiting_factors.append("Memory")
        if pod_count_required > 0 and pod_nodes_basis >= max_basis_nodes and pod_nodes_basis > 0:
             limiting_factors.append("Pod Count")

        # If the HA minimum forced the node count higher, note that too
        if nodes_needed > max_basis_nodes:
             limiting_factors.append("HA Minimum Enforced")
        # If max_basis_nodes was 0 (no requirements) and nodes_needed is 3 (min_ha)
        elif max_basis_nodes == 0 and nodes_needed == min_nodes_ha:
             limiting_factors.append("HA Minimum")
        elif not limiting_factors: # Fallback if logic above misses something
             limiting_factors.append("Unknown") # Should ideally not happen


        # Calculate efficiency score for this instance type and node count
        efficiency_score = calculate_instance_score(specs, requirements, nodes_needed)

        # Add this potential recommendation
        all_potential_recommendations.append({
            "instance_type": name,
            "vcpus_per_node": vcpus_per_node,
            "memory_gb_per_node": memory_gb_per_node,
            "description": specs.get("description", "N/A"),
            "network": specs.get("network", "N/A"),
            "generation": specs.get("generation", 0),
            "nodes_needed": nodes_needed,
            "cpu_utilization_at_nodes": cpu_utilization,
            "memory_utilization_at_nodes": memory_utilization,
            "pod_utilization_at_nodes": pod_utilization,
            "limiting_factors": limiting_factors,
            "total_vcpus_capacity": total_instance_vcpus,
            "total_memory_gb_capacity": total_instance_memory_gb,
            "total_pods_capacity": total_instance_pod_capacity,
            "efficiency_score": efficiency_score,
            # Show the basis nodes before HA for clarity in explanation
            "basis_nodes_cpu": cpu_nodes_basis,
            "basis_nodes_memory": memory_nodes_basis,
            "basis_nodes_pods": pod_nodes_basis,
        })

    # Sort recommendations by efficiency score (higher is better)
    # As a tie-breaker, prefer fewer nodes if scores are very close or identical
    all_potential_recommendations.sort(key=lambda x: (-x["efficiency_score"], x["nodes_needed"]))

    # Return the top N recommendations (e.g., top 10)
    # We will present the top one as the primary recommendation
    return all_potential_recommendations[:10]


def generate_recommendations(metrics: Dict[str, Any], redundancy_factor: float) -> Dict[str, Any]:
    """
    Generate sizing recommendations including observed metrics, calculated
    requirements, recommended configuration, and other options.

    Args:
        metrics (dict): Input metrics data (observed workload demand).
        redundancy_factor (float): Redundancy factor.

    Returns:
        dict: Comprehensive sizing recommendations.
    """
    # 1. Calculate Requirements for ROSA (based on observed workload + redundancy)
    calculated_requirements = calculate_sizing_requirements(metrics, redundancy_factor)
    workload_profile_analysis = calculated_requirements["workload_profile_analysis"]

    # 2. Recommend Instance Types and Counts to meet requirements
    worker_node_options = recommend_worker_node_options(calculated_requirements)

    # Default recommended config if no options found (shouldn't happen with instances listed and MIN_HA_WORKER_NODES > 0)
    recommended_config_summary: Dict[str, Any] = {
         "instance_type": "N/A",
         "worker_node_count": MIN_HA_WORKER_NODES, # Default to HA minimum
         "total_cpu_cores": 0,
         "total_memory_gb": 0,
         "total_pods": 0,
         "total_storage_gb": calculated_requirements.get("storage_required_gb", MIN_STORAGE_GB), # Still recommend storage
         "notes": "No suitable instance types found matching criteria or metrics were insufficient."
    }

    # If recommendations exist, use the top one for the summary
    if worker_node_options:
        top_option = worker_node_options[0]
        recommended_config_summary = {
            "instance_type": top_option["instance_type"],
            "worker_node_count": top_option["nodes_needed"],
            "total_cpu_cores": top_option["total_vcpus_capacity"],
            "total_memory_gb": top_option["total_memory_gb_capacity"],
            "total_pods": top_option["total_pods_capacity"],
            "total_storage_gb": calculated_requirements.get("storage_required_gb", MIN_STORAGE_GB),
            "notes": "This is the top-ranked option based on efficiency score."
        }


    # 3. Structure the Output Data
    recommendations: Dict[str, Any] = {
        "report_metadata": {
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics_collection_date": metrics.get("metadata", {}).get("collection_date", "Unknown"),
            "metrics_period_days": metrics.get("metadata", {}).get("days", "Unknown"),
            "tool_version": "2.3",
            "tool_notes": "Sizing is based on observed workload metrics (demand) from the source cluster, not its total available capacity (supply)."
        },
        "observed_workload_metrics": { # Renamed for clarity
            "description": "Peak and average usage and requests observed in the source OpenShift cluster during the collection period. This represents the workload DEMAND.",
            # Use .get() chain for safety as validation is less strict for presentation
            "cpu_cores": {"peak_usage": metrics.get("metrics", {}).get("cpu", {}).get("peak", 0),
                          "average_usage": metrics.get("metrics", {}).get("cpu", {}).get("average", 0),
                          "peak_requests": metrics.get("metrics", {}).get("cpu", {}).get("requests_peak", 0),
                          "peak_limits": metrics.get("metrics", {}).get("cpu", {}).get("limits_peak", 0),
                          },
            "memory_gb": {"peak_usage": metrics.get("metrics", {}).get("memory", {}).get("peak", 0),
                          "average_usage": metrics.get("metrics", {}).get("memory", {}).get("average", 0),
                          "peak_requests": metrics.get("metrics", {}).get("memory", {}).get("requests_peak", 0),
                          "peak_limits": metrics.get("metrics", {}).get("memory", {}).get("limits_peak", 0),
                         },
            "pods": {"peak": metrics.get("metrics", {}).get("pods", {}).get("peak", 0),
                     "average": metrics.get("metrics", {}).get("pods", {}).get("average", 0)},
            "storage_gb": {"peak_usage": metrics.get("metrics", {}).get("storage", {}).get("peak", 0),
                           "average_usage": metrics.get("metrics", {}).get("storage", {}).get("average", 0)},
        },
        "calculated_rosa_requirements": { # Renamed for clarity
            "description": "The minimum resources/pods/storage required in the target ROSA cluster, calculated from observed metrics (demand) plus the redundancy factor.",
            "basis_from_observed_metrics": { # Added baseline before redundancy for context
                 "cpu_cores": calculated_requirements.get("sizing_baseline",{}).get("cpu", 0),
                 "memory_gb": calculated_requirements.get("sizing_baseline",{}).get("memory", 0),
                 "pod_count": calculated_requirements.get("sizing_baseline",{}).get("pods", 0),
                 "storage_gb": calculated_requirements.get("raw_metrics_peak",{}).get("storage_usage", 0), # Storage baseline is peak usage
            },
            "redundancy_factor_applied": calculated_requirements.get("redundancy_factor", 0.0),
            "final_required_resources": { # The values *after* redundancy
                 "cpu_cores": calculated_requirements.get("cpu_required_cores", 0.0),
                 "memory_gb": calculated_requirements.get("memory_required_gb", 0.0),
                 "pod_count": calculated_requirements.get("pod_count_required", 0.0),
                 "storage_gb": calculated_requirements.get("storage_required_gb", MIN_STORAGE_GB),
            },
            "workload_profile": workload_profile_analysis.get("profile_type", "Unknown"),
            "workload_cpu_memory_ratio_basis": workload_profile_analysis.get("ratio_cores_per_gb", "Unknown"), # Ratio calculated from sizing baseline
             "sizing_basis_from_metrics_details": workload_profile_analysis.get("sizing_basis_from_metrics", {}), # Indicates if usage or requests drove baseline
        },
        "recommended_rosa_configuration_summary": recommended_config_summary, # Renamed for clarity
        "worker_node_options": worker_node_options,
        "general_notes": [
            f"These recommendations apply specifically to ROSA worker nodes, which are where your application workloads run. Note that a ROSA cluster also includes Red Hat managed control plane (typically 3 nodes) and infrastructure nodes (typically 3 nodes) which run on smaller, managed instances (e.g., m5.xlarge).",
            f"A minimum of {MIN_HA_WORKER_NODES} worker nodes distributed across availability zones is strongly recommended for production clusters to ensure high availability, resilience, and sufficient capacity for core OpenShift components.",
            "The storage recommendation is for Persistent Volumes (PVs) used by stateful applications to store data. It does not include ephemeral storage or the root volumes used by the worker nodes themselves.",
            "Selecting the right EBS volume type (e.g., gp3, io1) and configuring sufficient IOPS/throughput is often as critical as capacity for storage performance. Consider your application's I/O needs.",
            f"The maximum number of pods scheduled per worker node is typically limited by Kubernetes/OpenShift configuration (default {DEFAULT_MAX_PODS_PER_NODE} pods) and potentially AWS networking limits (IP addresses per ENI). This tool uses {DEFAULT_MAX_PODS_PER_NODE} as a safe default baseline for pod sizing.",
            "For complex workloads or applications with significantly different resource profiles, consider using multiple worker node pools within your ROSA cluster (e.g., one pool for general purpose workloads, another for high-CPU jobs on c-series, etc.). This tool provides ranked options across instance families but the primary recommendation assumes a single worker pool configuration.",
            f"Utilization percentages shown are based on the *calculated required resources* (after applying the redundancy factor) divided by the *total available capacity* of the recommended node configuration. For very small workloads where the {MIN_HA_WORKER_NODES}-node minimum is the limiting factor, utilization may appear very low; this is expected.",
            f"The efficiency score ranks worker node configurations based on a balance of resource utilization (aiming near {TARGET_UTILIZATION_PERCENT}%), instance generation (newer is better), network capability, and node count (fewer nodes is generally more cost-effective and simpler to manage, especially closer to the HA minimum).",
            "Cost is a significant factor in cloud sizing. Use the AWS Pricing Calculator to estimate the cost of the recommended configurations based on your region and desired purchasing options (On-Demand, Savings Plans, Reserved Instances).",
            "These recommendations are estimates based on historical workload demand and a configurable redundancy factor. It is essential to monitor your ROSA cluster after migration and be prepared to adjust scaling (vertically by changing instance types, or horizontally by adding/removing nodes) based on observed performance, future growth, or changes in application workload patterns."
        ]
    }

    return recommendations

def format_text_report(recommendations: Dict[str, Any]) -> str:
    """
    Format recommendations as text report

    Args:
        recommendations (dict): Sizing recommendations

    Returns:
        str: Formatted text report
    """
    report: List[str] = []

    # Add header
    report.append("=" * 80)
    report.append("ROSA SIZING RECOMMENDATIONS REPORT")
    report.append("=" * 80)

    # Add summary at the top
    summary_config = recommendations.get("recommended_rosa_configuration_summary", {}) # Use .get() for safety
    calculated_reqs = recommendations.get("calculated_rosa_requirements", {})
    workload_profile = calculated_reqs.get('workload_profile', 'Unknown')

    report.append(f"\nSUMMARY:")
    report.append(f"  Workload Profile: {workload_profile.upper()}")
    report.append(f"  Recommended ROSA Configuration:")
    report.append(f"    Worker Instance Type: {summary_config.get('instance_type', 'N/A')}")
    report.append(f"    Worker Node Count: {summary_config.get('worker_node_count', 'N/A')}")
    report.append(f"    Total Worker Capacity Provided: {summary_config.get('total_cpu_cores', 0.0):.1f} vCPUs, {summary_config.get('total_memory_gb', 0.0):.1f} GB Memory")
    report.append(f"    Recommended Storage (Persistent Volumes): {summary_config.get('total_storage_gb', 0):.0f} GB minimum")


    # Add metadata
    report.append("\nREPORT METADATA:")
    report.append(f"  Generation Date: {recommendations.get('report_metadata', {}).get('generation_date', 'Unknown')}")
    report.append(f"  Metrics Collection Date: {recommendations.get('report_metadata', {}).get('metrics_collection_date', 'Unknown')}")
    report.append(f"  Metrics Period: {recommendations.get('report_metadata', {}).get('metrics_period_days', 'Unknown')} days")
    report.append(f"  Tool Version: {recommendations.get('report_metadata', {}).get('tool_version', 'Unknown')}")
    report.append(f"  Tool Notes: {recommendations.get('report_metadata', {}).get('tool_notes', 'N/A')}")


    # Add observed current cluster state (Workload DEMAND)
    observed = recommendations.get("observed_workload_metrics", {})
    report.append("\nOBSERVED WORKLOAD METRICS (from source cluster):")
    report.append("  These reflect the peak usage and requests observed during the collection period.")
    report.append("  (Note: These represent the workload DEMAND, not the total capacity of the source cluster)")
    report.append(f"  CPU Usage: {observed.get('cpu_cores', {}).get('peak_usage', 0.0):.2f} cores peak, {observed.get('cpu_cores', {}).get('average_usage', 0.0):.2f} cores average")
    report.append(f"  Memory Usage: {observed.get('memory_gb', {}).get('peak_usage', 0.0):.2f} GB peak, {observed.get('memory_gb', {}).get('average_usage', 0.0):.2f} GB average")
    report.append(f"  Pod Count: {int(observed.get('pods', {}).get('peak', 0)):d} peak, {int(observed.get('pods', {}).get('average', 0)):d} average")
    report.append(f"  Storage Usage: {observed.get('storage_gb', {}).get('peak_usage', 0.0):.2f} GB peak, {observed.get('storage_gb', {}).get('average_usage', 0.0):.2f} GB average")
    report.append(f"  CPU Requests: {observed.get('cpu_cores', {}).get('peak_requests', 0.0):.2f} cores peak")
    report.append(f"  Memory Requests: {observed.get('memory_gb', {}).get('peak_requests', 0.0):.2f} GB peak")
    report.append(f"  CPU Limits: {observed.get('cpu_cores', {}).get('peak_limits', 0.0):.2f} cores peak")
    report.append(f"  Memory Limits: {observed.get('memory_gb', {}).get('peak_limits', 0.0):.2f} GB peak")


    # Add calculated requirements (Basis for ROSA sizing)
    calculated_reqs = recommendations.get("calculated_rosa_requirements", {})
    raw_baseline = calculated_reqs.get('basis_from_observed_metrics', {})
    final_reqs = calculated_reqs.get('final_required_resources', {})
    sizing_basis_details = calculated_reqs.get('sizing_basis_from_metrics_details', {})


    report.append("\nCALCULATED ROSA REQUIREMENTS (Sizing Basis):")
    report.append(f"  Calculated from Observed Metrics (using {'requests' if raw_baseline.get('cpu',0) > observed.get('cpu_cores',{}).get('peak_usage',0) else 'usage'} for CPU, {'requests' if raw_baseline.get('memory',0) > observed.get('memory_gb',{}).get('peak_usage',0) else 'usage'} for Memory) plus {calculated_reqs.get('redundancy_factor_applied', 0.0):.1f}x redundancy.")
    report.append(f"  Sizing Baseline (max of peak usage/requests): {raw_baseline.get('cpu',0.0):.2f} cores CPU, {raw_baseline.get('memory',0.0):.2f} GB Memory, {int(raw_baseline.get('pod_count',0)):d} Pods, {raw_baseline.get('storage_gb',0.0):.2f} GB Storage")
    report.append(f"  Required Resources (after redundancy):")
    report.append(f"    CPU: {final_reqs.get('cpu_cores', 0.0):.2f} cores")
    report.append(f"    Memory: {final_reqs.get('memory_gb', 0.0):.2f} GB")
    report.append(f"    Pod Count: {int(final_reqs.get('pod_count', 0)):d}")
    report.append(f"    Storage: {final_reqs.get('storage_gb', 0):.0f} GB minimum")
    report.append(f"  Workload Profile: {calculated_reqs.get('workload_profile', 'Unknown')} (Ratio: {calculated_reqs.get('workload_cpu_memory_ratio_basis', 'Unknown')} cores/GB)")

    # Add the comparison section
    report.append("\nSIZING COMPARISON:")
    report.append(f"{'Metric':<15}{'Peak Observed':>15}{'Req\'d (w/Redundancy)':>22}{'Recommended ROSA Capacity':>28}")
    report.append("-" * 80)
    report.append(f"{'CPU (cores)':<15}{observed.get('cpu_cores', {}).get('peak_usage', 0.0):>15.2f}{final_reqs.get('cpu_cores', 0.0):>22.2f}{summary_config.get('total_cpu_cores', 0.0):>28.1f}")
    report.append(f"{'Memory (GB)':<15}{observed.get('memory_gb', {}).get('peak_usage', 0.0):>15.2f}{final_reqs.get('memory_gb', 0.0):>22.2f}{summary_config.get('total_memory_gb', 0.0):>28.1f}")
    report.append(f"{'Pods':<15}{int(observed.get('pods', {}).get('peak', 0)):>15d}{int(final_reqs.get('pod_count', 0)):>22d}{int(summary_config.get('total_pods', 0)):>28d}")
    report.append(f"{'Storage (GB)':<15}{observed.get('storage_gb', {}).get('peak_usage', 0.0):>15.2f}{final_reqs.get('storage_gb', 0):>22.0f}{summary_config.get('total_storage_gb', 0):>28.0f}") # Storage capacity isn't explicitly defined per instance type, so compare required vs recommended PV size


    # Add instance options
    report.append("\nWORKER NODE OPTIONS (Ranked by Efficiency Score):")

    worker_options = recommendations.get("worker_node_options", [])
    if not worker_options:
        report.append("  No suitable worker node configurations found.")
    else:
        for i, rec in enumerate(worker_options, 1):
            report.append(f"\n  OPTION {i}: {rec.get('instance_type', 'N/A')} ({rec.get('description', 'N/A')})")
            report.append(f"    vCPUs per node: {rec.get('vcpus_per_node', 'N/A')}, Memory per node: {rec.get('memory_gb_per_node', 'N/A')} GB")
            report.append(f"    Calculated Nodes Needed: {rec.get('nodes_needed', 'N/A')}")

            # Explain how nodes_needed was determined
            limiting_factors_explanation = []
            limiting_factors = rec.get('limiting_factors', ["Unknown"])
            basis_nodes_cpu = rec.get('basis_nodes_cpu', 0)
            basis_nodes_memory = rec.get('basis_nodes_memory', 0)
            basis_nodes_pods = rec.get('basis_nodes_pods', 0)
            reqs_final = recommendations.get('calculated_rosa_requirements', {}).get('final_required_resources', {}) # Get final requirements for context

            for factor in limiting_factors:
                 if factor == "CPU":
                     limiting_factors_explanation.append(f"CPU requirements (initially needed {basis_nodes_cpu} nodes for {reqs_final.get('cpu_cores', 0.0):.1f} cores)")
                 elif factor == "Memory":
                      limiting_factors_explanation.append(f"Memory requirements (initially needed {basis_nodes_memory} nodes for {reqs_final.get('memory_gb', 0.0):.1f} GB)")
                 elif factor == "Pod Count":
                      limiting_factors_explanation.append(f"Pod count (initially needed {basis_nodes_pods} nodes for {int(reqs_final.get('pod_count', 0)):d} pods)")
                 elif factor == "HA Minimum Enforced":
                      limiting_factors_explanation.append(f"High Availability minimum of {MIN_HA_WORKER_NODES} nodes enforced")
                 elif factor == "HA Minimum":
                      limiting_factors_explanation.append(f"High Availability minimum of {MIN_HA_WORKER_NODES} nodes")
                 else:
                      limiting_factors_explanation.append(factor)

            report.append(f"    (Node count determined by: {', '.join(limiting_factors_explanation)})")

            # Use .get() for utilization values too
            cpu_util = rec.get('cpu_utilization_at_nodes', 0.0)
            mem_util = rec.get('memory_utilization_at_nodes', 0.0)
            pod_util = rec.get('pod_utilization_at_nodes', 0.0)
            report.append(f"    Utilization at {rec.get('nodes_needed', 'N/A')} nodes: CPU {cpu_util:.1f}%, Memory {mem_util:.1f}%, Pod {pod_util:.1f}%")

            # Use .get() for total capacity values too
            total_cpu = rec.get('total_vcpus_capacity', 0.0)
            total_mem = rec.get('total_memory_gb_capacity', 0.0)
            total_pods = rec.get('total_pods_capacity', 0)
            report.append(f"    Total Capacity at {rec.get('nodes_needed', 'N/A')} nodes: {total_cpu:.1f} vCPUs, {total_mem:.1f} GB Memory, {int(total_pods):d} Pods")
            report.append(f"    Efficiency Score: {rec.get('efficiency_score', 0.0):.2f}")


    # Add general notes
    report.append("\nGENERAL NOTES:")
    general_notes = recommendations.get("general_notes", [])
    for note in general_notes:
        report.append(f"  - {note}")

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
        try:
            backup_file = f"{args.output}.bak"
            print(f"Creating backup of existing output file: {backup_file}")
            os.rename(args.output, backup_file)
        except OSError as e:
            print(f"Warning: Could not create backup of {args.output}: {e}", file=sys.stderr)


    # Write recommendations to output file
    try:
        with open(args.output, "w") as f:
            f.write(output_content)
        print(f"Recommendations saved to {args.output}")
    except IOError as e:
        print(f"Error writing output file {args.output}: {e}", file=sys.stderr)
        sys.exit(1)


    # Print summary to console
    summary_config = recommendations.get("recommended_rosa_configuration_summary", {})
    calculated_reqs = recommendations.get("calculated_rosa_requirements", {})
    workload_profile = calculated_reqs.get('workload_profile', 'Unknown')

    print("\n--- Recommendation Summary ---")
    print(f"Workload Profile: {workload_profile.upper()}")
    print(f"Recommended ROSA Worker Configuration:")
    print(f"  Instance Type: {summary_config.get('instance_type', 'N/A')}")
    print(f"  Worker Node Count: {summary_config.get('worker_node_count', 'N/A')}")
    print(f"  Total Worker Capacity: {summary_config.get('total_cpu_cores', 0.0):.1f} vCPUs, {summary_config.get('total_memory_gb', 0.0):.1f} GB Memory")
    print(f"  Storage (PVs): {summary_config.get('total_storage_gb', 0):.0f} GB minimum")
    print("----------------------------")
    print(f"Full details saved to {args.output} ({args.format} format).")


if __name__ == "__main__":
    main()