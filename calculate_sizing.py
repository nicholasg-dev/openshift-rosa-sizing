#!/usr/bin/env python3
"""
OpenShift ROSA Sizing Tool - Sizing Calculator Script
Version: 2.5  # Increment version

This script analyzes collected data (metrics and original cluster sizing)
from an existing OpenShift cluster to provide ROSA sizing recommendations.
It determines sizing based on observed workload demand and presents this
alongside the original cluster's allocated capacity for context and comparison.
"""

import argparse
import json
import os
import sys
import math
import re # Import re for parsing resource strings
from datetime import datetime, timezone # Import timezone for UTC
from typing import Dict, Any, List # Added type hints

# Default configuration
DEFAULT_INPUT: str = "cluster_data.json" # Updated default input file name
DEFAULT_OUTPUT: str = "rosa_sizing.json"
DEFAULT_FORMAT: str = "json"
DEFAULT_REDUNDANCY: float = 1.2 # Increased redundancy for small workloads
MIN_HA_WORKER_NODES: int = 3 # Minimum nodes for High Availability
DEFAULT_MAX_PODS_PER_NODE: int = 150 # Default Kubernetes/OpenShift limit per node (consider actual limits if known)
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

# AWS instance type definitions for ROSA worker nodes (kept the same as provided)
# Format: name: {"vcpus": int, "memory": float, "family": str, "generation": int,
#                "network": str, "storage": str, "description": str}
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


def load_cluster_data(input_file: str) -> Dict[str, Any]:
    """
    Load cluster data (metrics and sizing) from input file.

    Args:
        input_file (str): Path to input file

    Returns:
        dict: Loaded cluster data
    """
    try:
        with open(input_file, "r") as f:
            cluster_data = json.load(f)

        # Basic validation for required top-level keys
        required_top_keys = ["metadata", "metrics", "cluster_sizing"] # Added cluster_sizing
        for key in required_top_keys:
            if key not in cluster_data:
                 # cluster_sizing might be missing if collection failed partially
                 if key == "cluster_sizing":
                      print(f"Warning: Missing expected top-level key in input file: '{key}'. Sizing comparison may be incomplete.", file=sys.stderr)
                      cluster_data["cluster_sizing"] = {} # Add empty dict to avoid errors later
                 elif key == "metrics":
                      print(f"Warning: Missing expected top-level key in input file: '{key}'. Metric-based sizing will be impossible.", file=sys.stderr)
                      cluster_data["metrics"] = {} # Add empty dict
                 else:
                    raise ValueError(f"Missing required top-level key in input file: '{key}'")


        # Basic validation for required metric types and peak/average values
        # Updated keys to match collector script output
        required_metric_types = ["cpu_usage", "memory_usage", "pod_count", "storage_usage_pvc"]
        metrics_data = cluster_data.get("metrics", {}) # Get metrics safely

        for metric_type in required_metric_types:
             if metric_type not in metrics_data:
                  print(f"Warning: Missing metrics for '{metric_type}'. Calculations may be affected.", file=sys.stderr)
                  # Add dummy data to prevent errors later, but warn
                  metrics_data[metric_type] = {"peak": 0, "average": 0}
             else:
               # Updated keys for requests/limits access
               if "peak" not in metrics_data[metric_type]: # Average is less critical for peak sizing
                   print(f"Warning: Missing peak data for '{metric_type}'. Using 0.", file=sys.stderr)
                   metrics_data[metric_type]["peak"] = 0
               if "requests_peak" not in metrics_data[metric_type]:
                    # If requests_peak is missing, default to 0
                    metrics_data[metric_type]["requests_peak"] = 0
                    if metrics_data[metric_type]["peak"] > 0: # Only warn if there is actual usage
                         print(f"Warning: Missing 'requests_peak' for '{metric_type}'. Sizing baseline will rely only on peak usage ({metrics_data[metric_type]['peak']:.2f}).", file=sys.stderr)

               # Limits are less critical for sizing baseline calculation itself, but good for info
               if "limits_peak" not in metrics_data[metric_type]:
                    metrics_data[metric_type]["limits_peak"] = 0


        return cluster_data

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading or parsing input file {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid input file format: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading data: {e}", file=sys.stderr)
         sys.exit(1)


def analyze_workload_profile(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze workload profile to determine if it's CPU or memory bound,
    using peaks and requests from the observed metrics.

    Args:
        metrics_data (dict): Metrics data extracted from the cluster data.

    Returns:
        dict: Workload profile analysis including calculated ratio and profile type.
    """
    try:
        # Access metrics safely using .get() chain and updated keys
        cpu_metrics = metrics_data.get("cpu_usage", {}) # Use cpu_usage key
        memory_metrics = metrics_data.get("memory_usage", {}) # Use memory_usage key

        cpu_peak = cpu_metrics.get("peak", 0.0)
        memory_peak = memory_metrics.get("peak", 0.0)
        cpu_requests = cpu_metrics.get("requests_peak", 0.0) # Use requests_peak key
        memory_requests = memory_metrics.get("requests_peak", 0.0) # Use requests_peak key

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
        # If both are 0, ratio is effectively 0.0, which is handled below

        # Determine workload profile based on ratio
        # Thresholds (cores per GB):
        # - ratio > 0.25 (approx 1 core per 4 GB): Tends towards CPU-bound (e.g. c-series)
        # - ratio < 0.10 (approx 1 core per 10 GB): Tends towards Memory-bound (e.g. r-series)
        # - Values between indicate balanced (e.g. m-series)
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


def calculate_sizing_requirements(cluster_data: Dict[str, Any], redundancy_factor: float) -> Dict[str, Any]:
    """
    Calculate required resources in the target ROSA cluster based on observed
    metrics (peak/requests) and redundancy factor.

    Args:
        cluster_data (dict): Full cluster data from the source cluster.
        redundancy_factor (float): Redundancy factor to apply.

    Returns:
        dict: Calculated resource requirements for the target ROSA cluster.
    """
    try:
        # Get metrics safely using .get() chain and **updated keys**
        metrics_data = cluster_data.get("metrics", {})
        cpu_metrics = metrics_data.get("cpu_usage", {})
        memory_metrics = metrics_data.get("memory_usage", {})
        pods_metrics = metrics_data.get("pod_count", {})
        storage_metrics = metrics_data.get("storage_usage_pvc", {}) # Key for PV/PVC usage

        # Default to 0 if keys are missing (handled in load_cluster_data validation, but good practice)
        cpu_peak = cpu_metrics.get("peak", 0.0)
        memory_peak = memory_metrics.get("peak", 0.0)
        pod_count_peak = pods_metrics.get("peak", 0.0)
        storage_peak = storage_metrics.get("peak", 0.0) # This is the peak PV/PVC usage

        cpu_requests = cpu_metrics.get("requests_peak", 0.0)
        memory_requests = memory_metrics.get("requests_peak", 0.0)
        cpu_limits = cpu_metrics.get("limits_peak", 0.0) # Keep limits for info
        memory_limits = memory_metrics.get("limits_peak", 0.0) # Keep limits for info


        # Use the maximum of actual usage peak and requests peak for sizing basis
        # This ensures we account for both actual historical high watermarks
        # and resources potentially reserved by applications.
        cpu_sizing_baseline = max(cpu_peak, cpu_requests)
        memory_sizing_baseline = max(memory_peak, memory_requests)

        # Apply redundancy factor to sizing baseline
        # Ensure required resources are at least 0 if metrics are somehow negative
        cpu_required = max(0.0, cpu_sizing_baseline * redundancy_factor)
        memory_required = max(0.0, memory_sizing_baseline * redundancy_factor)
        pod_count_required = max(0.0, pod_count_peak * redundancy_factor) # Pods can be float for calculation, rounded later
        storage_required_raw = max(0.0, storage_peak * redundancy_factor) # Required storage is based on PV usage

        # Round up storage to nearest 10 GB and ensure minimum
        storage_required_gb = math.ceil(storage_required_raw / 10) * 10
        storage_required_gb = max(storage_required_gb, MIN_STORAGE_GB)

        workload_profile_analysis = analyze_workload_profile(metrics_data) # Pass just the metrics data

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
            "cpu_required_cores": 0.0, "memory_required_gb": 0.0, "pod_count_required": 0.0,
            "storage_required_gb": MIN_STORAGE_GB,
            "workload_profile_analysis": {"ratio_cores_per_gb": "Unknown", "profile_type": "unknown", "sizing_basis_from_metrics": {"cpu": "error", "memory": "error"}},
            "redundancy_factor": redundancy_factor,
            "raw_metrics_peak": {"cpu_usage": 0.0, "memory_usage": 0.0, "pods": 0.0, "storage_usage": 0.0, "cpu_requests": 0.0, "memory_requests": 0.0, "cpu_limits": 0.0, "memory_limits": 0.0},
            "sizing_baseline": {"cpu": 0.0, "memory": 0.0, "pods": 0.0}
         }


def calculate_instance_score(instance_specs: Dict[str, Any], requirements: Dict[str, Any], nodes_needed: int, max_basis_nodes: int) -> float:
    """
    Calculates an efficiency score for a given instance type and node count
    based on how well it meets the requirements.
    (This function remains largely unchanged as it operates on calculated requirements)

    Args:
        instance_specs (dict): Specifications of the instance type.
        requirements (dict): Calculated requirements (CPU, Memory, Pods) including redundancy.
        nodes_needed (int): The final calculated number of nodes (includes HA minimum).
        max_basis_nodes (int): The theoretical nodes needed based purely on resource/pod requirements (before HA minimum).


    Returns:
        float: The calculated efficiency score.
    """
    cpu_required = requirements.get("cpu_required_cores", 0.0)
    memory_required = requirements.get("memory_required_gb", 0.0)
    pod_count_required = requirements.get("pod_count_required", 0.0) # Keep as float for calculation
    min_nodes_ha = MIN_HA_WORKER_NODES

    # Calculate total capacity provided by this instance type * nodes_needed
    total_instance_vcpus = instance_specs.get("vcpus", 0) * nodes_needed
    total_instance_memory_gb = instance_specs.get("memory", 0) * nodes_needed

    # Note: Pod capacity per node might vary slightly based on specific AWS instance limits
    # and OpenShift configuration, but 110 is a standard and safe default limit per kubelet.
    # A simple scaling cap based on vCPU per node can be used as a heuristic.
    # Ensure we don't divide by zero if vcpus_per_node is 0
    vcpus_per_node = instance_specs.get("vcpus", 0)
    pod_capacity_heuristic_per_node = min(DEFAULT_MAX_PODS_PER_NODE, DEFAULT_MAX_PODS_PER_NODE * (vcpus_per_node / 4)) if vcpus_per_node > 0 else DEFAULT_MAX_PODS_PER_NODE # Assume base is 4 vCPU node
    if vcpus_per_node == 0: pod_capacity_heuristic_per_node = 0 # An instance with 0 vCPU likely can't run pods
    total_instance_pod_capacity = pod_capacity_heuristic_per_node * nodes_needed


    # Calculate utilization percentages based on REQUIRED resources vs. PROVIDED capacity
    # Ensure division by zero is handled
    cpu_utilization = (cpu_required / total_instance_vcpus) * 100 if total_instance_vcpus > 0 else 0.0
    memory_utilization = (memory_required / total_instance_memory_gb) * 100 if total_instance_memory_gb > 0 else 0.0
    # Pod utilization based on the rounded up required pod count vs total capacity
    pod_utilization = (math.ceil(pod_count_required) / total_instance_pod_capacity) * 100 if total_instance_pod_capacity > 0 else 0.0


    # --- Scoring Components ---

    # 1. Generation score: Newer generations score higher (e.g., gen 7 = 70 points)
    generation_score = instance_specs.get("generation", 0) * 10.0

    # 2. Utilization scores: Closer to target utilization scores higher
    # Penalizes deviation from TARGET_UTILIZATION_PERCENT.

    # Determine max penalty based on whether minimum nodes were enforced
    # If nodes_needed == MIN_HA_WORKER_NODES AND max_basis_nodes < MIN_HA_WORKER_NODES,
    # it means the workload is small and the node count is driven by the HA minimum.
    # In this scenario, low utilization is expected and should be penalized less severely.
    is_min_nodes_enforced = (nodes_needed == MIN_HA_WORKER_NODES and max_basis_nodes < MIN_HA_WORKER_NODES)

    # Cap deviation penalty: higher for normal scaling, lower when min nodes enforced
    max_util_penalty = 50.0 # Max penalty points for large deviations (normal case)
    min_node_max_util_penalty = 20.0 # Reduced max penalty when min nodes enforced

    penalty_cap = min_node_max_util_penalty if is_min_nodes_enforced else max_util_penalty

    cpu_util_score = 100.0 - min(abs(TARGET_UTILIZATION_PERCENT - cpu_utilization), penalty_cap)
    memory_util_score = 100.0 - min(abs(TARGET_UTILIZATION_PERCENT - memory_utilization), penalty_cap)

    # 3. Node count score: Fewer nodes (closer to min_ha) score higher
    # Penalize extra nodes beyond the HA minimum
    node_count_score = 100.0 - ((nodes_needed - min_nodes_ha) * 10.0)
    node_count_score = max(0.0, node_count_score) # Score cannot be negative

    # 4. Network score: Higher network capability scores higher
    # Extract numerical speed (e.g., "up to 12.5 Gbps" -> 12.5, "100 Gbps" -> 100)
    # Scale network speed to a score out of 100
    network_score = 0.0
    network_str = instance_specs.get("network", "0 Gbps").lower()
    try:
        # Extract numbers from strings like "up to 12.5 gbps", "25 gbps", "75 gbps"
        speed_match = re.search(r'(\d+(\.\d+)?)\s*gbps', network_str)
        if speed_match:
            network_speed = float(speed_match.group(1))
            # Simple scaling: 1 Gbps -> ~1 point, 100 Gbps -> 100 points. Cap at 100.
            network_score = min(network_speed, 100.0)
    except (ValueError, IndexError):
        pass # If parsing fails, network score remains 0.0


    # Combine scores using weights
    total_weight_sum = sum(SCORE_WEIGHTS.values())
    if total_weight_sum == 0: total_weight_sum = 1 # Avoid division by zero

    efficiency_score = (
        generation_score * SCORE_WEIGHTS.get("generation", 0.0) +
        cpu_util_score * SCORE_WEIGHTS.get("cpu_util", 0.0) +
        memory_util_score * SCORE_WEIGHTS.get("memory_util", 0.0) +
        network_score * SCORE_WEIGHTS.get("network", 0.0) +
        node_count_score * SCORE_WEIGHTS.get("node_count", 0.0)
    ) / total_weight_sum # Normalize by total weight


    return efficiency_score


def recommend_worker_node_options(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recommend worker node instance types and counts based on calculated requirements.

    Evaluates all relevant instance types and ranks them by an efficiency score.
    (This function remains unchanged as it operates on calculated requirements)

    Args:
        requirements (dict): Calculated resource requirements (including redundancy).

    Returns:
        list: Ranked list of recommended instance configurations.
    """

    cpu_required = requirements.get("cpu_required_cores", 0.0)
    memory_required = requirements.get("memory_required_gb", 0.0)
    pod_count_required = requirements.get("pod_count_required", 0.0) # Keep as float for calculation
    min_nodes_ha = MIN_HA_WORKER_NODES

    all_potential_recommendations: List[Dict[str, Any]] = []

    # Iterate through all available instance types
    # No filtering by workload family here, let the score determine suitability
    for name, specs in AWS_INSTANCE_TYPES.items():

        vcpus_per_node = specs.get("vcpus", 0)
        memory_gb_per_node = specs.get("memory", 0)

        # Skip instances with zero capacity if requirements are non-zero
        if (cpu_required > 0 and vcpus_per_node == 0) or \
           (memory_required > 0 and memory_gb_per_node == 0):
             continue

        # Calculate nodes needed based purely on resource requirements (before HA minimum)
        # Handle cases where required resources or instance capacity might be zero
        cpu_nodes_basis = math.ceil(cpu_required / vcpus_per_node) if vcpus_per_node > 0 and cpu_required > 0 else (1 if cpu_required > 0 else 0)
        memory_nodes_basis = math.ceil(memory_required / memory_gb_per_node) if memory_gb_per_node > 0 and memory_required > 0 else (1 if memory_required > 0 else 0)

        # Pods calculation uses the capped max_pods_per_node heuristic
        # Ensure we don't divide by zero if vcpus_per_node is 0
        pod_capacity_heuristic_per_node = min(DEFAULT_MAX_PODS_PER_NODE, DEFAULT_MAX_PODS_PER_NODE * (vcpus_per_node / 4)) if vcpus_per_node > 0 else DEFAULT_MAX_PODS_PER_NODE
        if vcpus_per_node == 0: pod_capacity_heuristic_per_node = 0 # An instance with 0 vCPU likely can't run pods

        pod_nodes_basis = math.ceil(pod_count_required / pod_capacity_heuristic_per_node) if pod_capacity_heuristic_per_node > 0 and pod_count_required > 0 else (1 if pod_count_required > 0 else 0)

        # Determine the *theoretical* nodes needed based on the maximum resource/pod need
        max_basis_nodes = max(cpu_nodes_basis, memory_nodes_basis, pod_nodes_basis)

        # The actual nodes needed is the maximum of the theoretical need and the HA minimum
        nodes_needed = max(max_basis_nodes, min_nodes_ha)

         # Basic sanity check: if requirements > 0 but nodes_needed is 0, something is wrong or instance is unsuitable
        if (cpu_required > 0 or memory_required > 0 or pod_count_required > 0) and nodes_needed == 0:
            continue # Skip this instance type

        # Calculate utilization percentages for the calculated node count
        total_instance_vcpus = vcpus_per_node * nodes_needed
        total_instance_memory_gb = memory_gb_per_node * nodes_needed
        # Pod utilization based on the rounded up required pod count vs total capacity
        total_instance_pod_capacity = pod_capacity_heuristic_per_node * nodes_needed # Total capacity for pod count check

        # Ensure division by zero is handled for utilization calculation
        cpu_utilization = (cpu_required / total_instance_vcpus) * 100 if total_instance_vcpus > 0 else 0.0
        memory_utilization = (memory_required / total_instance_memory_gb) * 100 if total_instance_memory_gb > 0 else 0.0
        pod_utilization = (math.ceil(pod_count_required) / total_instance_pod_capacity) * 100 if total_instance_pod_capacity > 0 else 0.0


        # Skip configurations with extremely high utilization (>95%) as they lack headroom and might indicate undersizing
        if cpu_utilization > 95.01 or memory_utilization > 95.01: # Use slight tolerance for float comparison
            # Optionally print a warning here about extreme utilization
            # print(f"Warning: Skipping {name} x{nodes_needed} due to extremely high utilization (CPU: {cpu_utilization:.1f}%, Mem: {memory_utilization:.1f}%)", file=sys.stderr)
            continue

        # Determine limiting factor(s) - This explains *why* 'max_basis_nodes' was calculated
        limiting_factors = []
        # Only list resource factors if requirements are > 0 AND they were the basis for max_basis_nodes
        # Check against a small epsilon for float comparison if needed, but direct comparison is usually ok for integers from ceil()
        if cpu_required > 0 and cpu_nodes_basis >= max_basis_nodes and cpu_nodes_basis > 0:
             limiting_factors.append("CPU")
        if memory_required > 0 and memory_nodes_basis >= max_basis_nodes and memory_nodes_basis > 0:
             limiting_factors.append("Memory")
        # Use ceil for comparison as pod_count_required is float
        if pod_count_required > 0 and pod_nodes_basis >= max_basis_nodes and pod_nodes_basis > 0:
             limiting_factors.append("Pod Count")


        # If the HA minimum forced the node count higher, note that too
        if nodes_needed > max_basis_nodes and max_basis_nodes > 0:
             limiting_factors.append("HA Minimum Enforced")
        # If max_basis_nodes was 0 (no requirements) and nodes_needed is 3 (min_ha)
        elif max_basis_nodes == 0 and nodes_needed == min_nodes_ha:
             limiting_factors.append("HA Minimum")
        elif not limiting_factors and nodes_needed > 0: # Fallback if logic above misses something but nodes were required
             limiting_factors.append("Resource Needs") # General placeholder
        elif not limiting_factors and nodes_needed == 0: # If no requirements and no nodes needed (should be caught earlier)
             limiting_factors.append("No Workload")


        # Calculate efficiency score for this instance type and node count
        efficiency_score = calculate_instance_score(specs, requirements, nodes_needed, max_basis_nodes) # Pass max_basis_nodes

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
            "total_pods_capacity": total_instance_pod_capacity, # Use heuristic capacity here
            "efficiency_score": efficiency_score,
            # Show the basis nodes before HA for clarity in explanation
            "basis_nodes_cpu": cpu_nodes_basis,
            "basis_nodes_memory": memory_nodes_basis,
            "basis_nodes_pods": pod_nodes_basis,
            "max_basis_nodes_before_ha": max_basis_nodes # Add max before HA for clarity
        })

    # Sort recommendations by efficiency score (higher is better)
    # As a tie-breaker, prefer fewer nodes if scores are very close or identical
    all_potential_recommendations.sort(key=lambda x: (-x["efficiency_score"], x["nodes_needed"]))

    # Return the top N recommendations (e.g., top 10)
    # We will present the top one as the primary recommendation
    return all_potential_recommendations[:10]


def generate_recommendations(cluster_data: Dict[str, Any], redundancy_factor: float) -> Dict[str, Any]:
    """
    Generate sizing recommendations including observed metrics, calculated
    requirements, recommended configuration, and other options.

    Args:
        cluster_data (dict): Full cluster data (metrics and sizing).
        redundancy_factor (float): Redundancy factor.

    Returns:
        dict: Comprehensive sizing recommendations.
    """
    # 1. Calculate Requirements for ROSA (based on observed workload + redundancy)
    # Pass the full cluster_data to calculate_sizing_requirements so it can access metrics
    calculated_requirements = calculate_sizing_requirements(cluster_data, redundancy_factor)
    workload_profile_analysis = calculated_requirements.get("workload_profile_analysis", {})

    # 2. Recommend Instance Types and Counts to meet requirements
    # This function only needs the calculated requirements
    worker_node_options = recommend_worker_node_options(calculated_requirements)

    # Default recommended config if no options found
    # (shouldn't happen with instances listed and MIN_HA_WORKER_NODES > 0 unless reqs are negative/invalid)
    recommended_config_summary: Dict[str, Any] = {
         "instance_type": "N/A",
         "worker_node_count": MIN_HA_WORKER_NODES, # Default to HA minimum
         "total_cpu_cores": 0.0,
         "total_memory_gb": 0.0,
         "total_pods": 0,
         "total_storage_gb": calculated_requirements.get("storage_required_gb", MIN_STORAGE_GB), # Still recommend storage
         "notes": "No suitable instance types found matching criteria or metrics were insufficient to calculate requirements."
    }

    # If recommendations exist, use the top one for the summary
    if worker_node_options:
        top_option = worker_node_options[0]
        recommended_config_summary = {
            "instance_type": top_option["instance_type"],
            "worker_node_count": top_option["nodes_needed"],
            "total_cpu_cores": top_option["total_vcpus_capacity"],
            "total_memory_gb": top_option["total_memory_gb_capacity"],
            "total_pods": int(top_option["total_pods_capacity"]), # Round down for summary
            "total_storage_gb": calculated_requirements.get("storage_required_gb", MIN_STORAGE_GB),
            "notes": "This is the top-ranked option based on efficiency score."
        }
    elif calculated_requirements.get("cpu_required_cores", 0) > 0 or calculated_requirements.get("memory_required_gb", 0) > 0 or calculated_requirements.get("pod_count_required", 0) > 0:
         # If requirements were calculated but no options found, provide a default based on minimal requirements
         # This might happen if available instance types are too large/small or filtering logic is too strict
         recommended_config_summary["notes"] = "Could not find ranked instance options. Defaulting to minimum HA nodes."


    # 3. Structure the Output Data
    # Now include the original cluster sizing data in the output
    recommendations: Dict[str, Any] = {
        "report_metadata": {
            "generation_date": datetime.now(timezone.utc).isoformat(), # Use UTC and ISO format
            # Use .get() chain to safely access metadata from loaded data
            "metrics_collection_date": cluster_data.get("metadata", {}).get("collection_date_utc", cluster_data.get("metadata", {}).get("collection_date", "Unknown")), # Check for new and old key
            "metrics_period_days": cluster_data.get("metadata", {}).get("prometheus_data_range", {}).get("days", cluster_data.get("metadata", {}).get("days", "Unknown")), # Check for new and old key
            "metrics_period_step": cluster_data.get("metadata", {}).get("prometheus_data_range", {}).get("step", "Unknown"), # Add step
            "tool_version": "2.5", # Updated version
            "tool_notes": "Sizing is based primarily on observed workload metrics (demand) from the source cluster plus redundancy. The original cluster's allocated capacity is provided for context and comparison."
        },
        "observed_original_cluster_sizing": cluster_data.get("cluster_sizing", {}), # Include the original sizing data
        "observed_workload_metrics": { # Renamed for clarity
            "description": "Peak and average usage and requests observed in the source OpenShift cluster during the collection period. This represents the workload DEMAND.",
            # Use .get() chain for safety and **updated keys**
            "cpu_cores": {"peak_usage": cluster_data.get("metrics", {}).get("cpu_usage", {}).get("peak", 0.0),
                          "average_usage": cluster_data.get("metrics", {}).get("cpu_usage", {}).get("average", 0.0),
                          "peak_requests": cluster_data.get("metrics", {}).get("cpu_usage", {}).get("requests_peak", 0.0),
                          "peak_limits": cluster_data.get("metrics", {}).get("cpu_usage", {}).get("limits_peak", 0.0),
                          },
            "memory_gb": {"peak_usage": cluster_data.get("metrics", {}).get("memory_usage", {}).get("peak", 0.0),
                          "average_usage": cluster_data.get("metrics", {}).get("memory_usage", {}).get("average", 0.0),
                          "peak_requests": cluster_data.get("metrics", {}).get("memory_usage", {}).get("requests_peak", 0.0),
                          "peak_limits": cluster_data.get("metrics", {}).get("memory_usage", {}).get("limits_peak", 0.0),
                         },
            "pods": {"peak": cluster_data.get("metrics", {}).get("pod_count", {}).get("peak", 0.0),
                     "average": cluster_data.get("metrics", {}).get("pod_count", {}).get("average", 0.0)},
            "storage_gb": {"peak_usage": cluster_data.get("metrics", {}).get("storage_usage_pvc", {}).get("peak", 0.0), # Use storage_usage_pvc
                           "average_usage": cluster_data.get("metrics", {}).get("storage_usage_pvc", {}).get("average", 0.0)}, # Use storage_usage_pvc
        },
        "calculated_rosa_requirements": { # Renamed for clarity
            "description": "The minimum resources/pods/storage required in the target ROSA cluster, calculated from observed metrics (demand) plus the redundancy factor.",
            "basis_from_observed_metrics": { # Added baseline before redundancy for context
                 "cpu_cores": calculated_requirements.get("sizing_baseline",{}).get("cpu", 0.0),
                 "memory_gb": calculated_requirements.get("sizing_baseline",{}).get("memory", 0.0),
                 "pod_count": calculated_requirements.get("sizing_baseline",{}).get("pods", 0.0),
                 "storage_gb": calculated_requirements.get("raw_metrics_peak",{}).get("storage_usage", 0.0), # Storage baseline is peak usage
            },
            "redundancy_factor_applied": calculated_requirements.get("redundancy_factor", 0.0),
            "final_required_resources": { # The values *after* redundancy
                 "cpu_cores": calculated_requirements.get("cpu_required_cores", 0.0),
                 "memory_gb": calculated_requirements.get("memory_required_gb", 0.0),
                 "pod_count": math.ceil(calculated_requirements.get("pod_count_required", 0.0)), # Round up required pods
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
            f"The maximum number of pods scheduled per worker node is typically limited by Kubernetes/OpenShift configuration ({DEFAULT_MAX_PODS_PER_NODE} pods by default) and potentially AWS networking limits (IP addresses per ENI). This tool uses a heuristic of {DEFAULT_MAX_PODS_PER_NODE} pods per node for sizing calculations.",
            "For complex workloads or applications with significantly different resource profiles, consider using multiple worker node pools within your ROSA cluster (e.g., one pool for general purpose workloads, another for high-CPU jobs on c-series, etc.). This tool provides ranked options across instance families but the primary recommendation assumes a single worker pool configuration.",
            f"Utilization percentages shown are based on the *calculated required resources* (after applying the redundancy factor) divided by the *total available capacity* of the recommended node configuration. For very small workloads where the {MIN_HA_WORKER_NODES}-node minimum is the limiting factor, utilization may appear very low; this is expected.",
            f"The efficiency score ranks worker node configurations based on a balance of resource utilization (aiming near {TARGET_UTILIZATION_PERCENT}%), instance generation (newer is better), network capability, and node count (fewer nodes is generally more cost-effective and simpler to manage, especially closer to the HA minimum). Note: For very small workloads where the node count is fixed at {MIN_HA_WORKER_NODES} due to the HA requirement, the penalty for low utilization is reduced in the scoring.",
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
    report_metadata = recommendations.get('report_metadata', {})
    report.append(f"  Generation Date: {report_metadata.get('generation_date', 'Unknown')}")
    report.append(f"  Metrics Collection Date: {report_metadata.get('metrics_collection_date', 'Unknown')}")
    report.append(f"  Metrics Period: {report_metadata.get('metrics_period_days', 'Unknown')} days (step: {report_metadata.get('metrics_period_step', 'Unknown')})") # Added step
    report.append(f"  Tool Version: {report_metadata.get('tool_version', 'Unknown')}")
    report.append(f"  Tool Notes: {report_metadata.get('tool_notes', 'N/A')}")


    # --- Add Observed Original Cluster Sizing ---
    original_sizing = recommendations.get("observed_original_cluster_sizing", {})
    report.append("\nOBSERVED ORIGINAL CLUSTER SIZING:")
    if original_sizing and original_sizing.get('total_nodes') is not None: # Check if sizing data was successfully collected
        report.append(f"  Total Nodes (All Roles): {original_sizing.get('total_nodes', 'N/A')}")
        report.append(f"  Worker Nodes: {original_sizing.get('worker_node_count', 'N/A')}")
        report.append("  Nodes by Role:")
        # Use .get() for roles dictionary and items() for safe iteration
        for role, count in original_sizing.get('nodes_by_role', {}).items():
             report.append(f"    {role}: {count}")
        report.append("  Nodes by Instance Type:")
        # Use .get() for instance types dictionary and items() for safe iteration
        for instance, count in original_sizing.get('nodes_by_instance_type', {}).items():
             report.append(f"    {instance}: {count}")
        # Use .get() for nested dictionaries and default values (showing worker capacity)
        report.append(f"  Total Worker Capacity: CPU: {original_sizing.get('total_capacity', {}).get('cpu_cores', 0.0):.2f} cores, Memory: {original_sizing.get('total_capacity', {}).get('memory_gb', 0.0):.2f} GB")
        report.append(f"  Total Worker Allocatable: CPU: {original_sizing.get('total_allocatable', {}).get('cpu_cores', 0.0):.2f} cores, Memory: {original_sizing.get('total_allocatable', {}).get('memory_gb', 0.0):.2f} GB")
        # Add a note about the source of this data
        report.append("  (Data collected via 'oc get nodes')")
    else:
        report.append("  Original cluster sizing data not available in input file.")


    # Add observed current cluster state (Workload DEMAND)
    # Use .get() chain for safety and **updated keys**
    observed = recommendations.get("observed_workload_metrics", {})
    report.append("\nOBSERVED WORKLOAD METRICS (from source cluster):")
    if observed and observed.get("cpu_cores"): # Basic check if metrics were collected
        report.append("  These reflect the peak usage and requests observed during the collection period.")
        report.append("  (Note: These represent the workload DEMAND, not the total capacity of the source cluster)")
        report.append(f"  CPU Usage: {observed.get('cpu_cores', {}).get('peak_usage', 0.0):.2f} cores peak, {observed.get('cpu_cores', {}).get('average_usage', 0.0):.2f} cores average")
        report.append(f"  Memory Usage: {observed.get('memory_gb', {}).get('peak_usage', 0.0):.2f} GB peak, {observed.get('memory_gb', {}).get('average_usage', 0.0):.2f} GB average")
        report.append(f"  Pod Count: {int(observed.get('pods', {}).get('peak', 0.0)):d} peak, {int(observed.get('pods', {}).get('average', 0.0)):d} average") # Cast float to int for display
        report.append(f"  Storage Usage (PV/PVC): {observed.get('storage_gb', {}).get('peak_usage', 0.0):.2f} GB peak, {observed.get('storage_gb', {}).get('average_usage', 0.0):.2f} GB average")
        report.append(f"  CPU Requests: {observed.get('cpu_cores', {}).get('peak_requests', 0.0):.2f} cores peak")
        report.append(f"  Memory Requests: {observed.get('memory_gb', {}).get('peak_requests', 0.0):.2f} GB peak")
        report.append(f"  CPU Limits: {observed.get('cpu_cores', {}).get('limits_peak', 0.0):.2f} cores peak") # Fixed typo here
        report.append(f"  Memory Limits: {observed.get('memory_gb', {}).get('limits_peak', 0.0):.2f} GB peak") # Fixed typo here
        report.append("  (Data collected via Prometheus)")
    else:
         report.append("  Prometheus metrics not available in input file.")


    # Add calculated requirements (Basis for ROSA sizing)
    calculated_reqs = recommendations.get("calculated_rosa_requirements", {})
    raw_baseline = calculated_reqs.get('basis_from_observed_metrics', {})
    final_reqs = calculated_reqs.get('final_required_resources', {})
    sizing_basis_details = calculated_reqs.get('sizing_basis_from_metrics_details', {})

    report.append("\nCALCULATED ROSA REQUIREMENTS (Sizing Basis):")
    if calculated_reqs and calculated_reqs.get('final_required_resources'): # Check if calculation was successful
        # Check if sizing basis details are valid before formatting
        cpu_basis = sizing_basis_details.get('cpu', 'error')
        memory_basis = sizing_basis_details.get('memory', 'error')
        report.append(f"  Calculated from Observed Metrics (using {cpu_basis} for CPU, {memory_basis} for Memory) plus {calculated_reqs.get('redundancy_factor_applied', 0.0):.1f}x redundancy.")
        report.append(f"  Sizing Baseline (max of peak usage/requests): {raw_baseline.get('cpu',0.0):.2f} cores CPU, {raw_baseline.get('memory',0.0):.2f} GB Memory, {int(raw_baseline.get('pod_count',0.0)):d} Pods, {raw_baseline.get('storage_gb',0.0):.2f} GB Storage")
        report.append(f"  Required Resources (after redundancy):")
        report.append(f"    CPU: {final_reqs.get('cpu_cores', 0.0):.2f} cores")
        report.append(f"    Memory: {final_reqs.get('memory_gb', 0.0):.2f} GB")
        report.append(f"    Pod Count: {int(final_reqs.get('pod_count', 0)):d}") # Cast to int for required count
        report.append(f"    Storage: {final_reqs.get('storage_gb', 0):.0f} GB minimum")
        report.append(f"  Workload Profile: {calculated_reqs.get('workload_profile', 'Unknown')} (Ratio: {calculated_reqs.get('workload_cpu_memory_ratio_basis', 'Unknown')} cores/GB)")
    else:
        report.append("  Sizing requirements could not be calculated (metrics missing or calculation error).")


    # Add the comparison section - Now includes original sizing vs. ROSA recommendation vs. ROSA requirements
    report.append("\nSIZING COMPARISON:")
    # Adjust column widths for the new format
    report.append(f"{'Metric':<15}{'Peak Usage':>15}{'Req w/Redundancy':>19}{'Original Capacity':>19}{'Recommended Capacity':>23}")
    report.append("-" * 95) # Adjust line length

    # Get data safely
    observed_usage = observed.get('cpu_cores', {}).get('peak_usage', 0.0)
    req_cpu = final_reqs.get('cpu_cores', 0.0)
    orig_cap_cpu = original_sizing.get('total_capacity', {}).get('cpu_cores', 0.0)
    rec_cap_cpu = summary_config.get('total_cpu_cores', 0.0)
    report.append(f"{'CPU (cores)':<15}{observed_usage:>15.2f}{req_cpu:>19.2f}{orig_cap_cpu:>19.2f}{rec_cap_cpu:>23.1f}")

    observed_usage = observed.get('memory_gb', {}).get('peak_usage', 0.0)
    req_mem = final_reqs.get('memory_gb', 0.0)
    orig_cap_mem = original_sizing.get('total_capacity', {}).get('memory_gb', 0.0)
    rec_cap_mem = summary_config.get('total_memory_gb', 0.0)
    report.append(f"{'Memory (GB)':<15}{observed_usage:>15.2f}{req_mem:>19.2f}{orig_cap_mem:>19.2f}{rec_cap_mem:>23.1f}")

    observed_peak_pods = int(observed.get('pods', {}).get('peak', 0.0)) # Cast to int
    req_pods = int(final_reqs.get('pod_count', 0)) # Cast to int
    orig_cap_pods = "N/A" # Pod capacity not in 'oc get nodes' capacity field directly
    rec_cap_pods = int(summary_config.get('total_pods', 0)) # Cast to int
    report.append(f"{'Pods':<15}{observed_peak_pods:>15d}{req_pods:>19d}{orig_cap_pods:>19}{rec_cap_pods:>23d}")

    observed_usage_storage = observed.get('storage_gb', {}).get('peak_usage', 0.0)
    req_storage = final_reqs.get('storage_gb', 0)
    orig_cap_storage = "N/A" # Original node disk capacity not collected, storage is PV/PVC
    rec_cap_storage = summary_config.get('total_storage_gb', 0)
    report.append(f"{'Storage (GB)':<15}{observed_usage_storage:>15.2f}{req_storage:>19.0f}{orig_cap_storage:>19}{rec_cap_storage:>23.0f}") # Storage capacity isn't explicitly defined per instance type, so compare required vs recommended PV size


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
            max_basis_nodes = rec.get('max_basis_nodes_before_ha', 0) # Get max before HA
            reqs_final = recommendations.get('calculated_rosa_requirements', {}).get('final_required_resources', {}) # Get final requirements for context


            # Dynamically build the explanation based on listed factors and basis nodes
            # Use the calculated max_basis_nodes for context in the explanation
            basis_parts = []
            if basis_nodes_cpu > 0: basis_parts.append(f"CPU ({basis_nodes_cpu} nodes for {reqs_final.get('cpu_cores', 0.0):.1f} cores)")
            if basis_nodes_memory > 0: basis_parts.append(f"Memory ({basis_nodes_memory} nodes for {reqs_final.get('memory_gb', 0.0):.1f} GB)")
            if basis_nodes_pods > 0: basis_parts.append(f"Pod Count ({basis_nodes_pods} nodes for {int(reqs_final.get('pod_count', 0.0)):d} pods)") # Cast to int here

            if max_basis_nodes > 0:
                 basis_explanation = f"max requirement ({max_basis_nodes} nodes based on {', '.join(basis_parts)})"
            elif basis_parts: # Should be covered by max_basis_nodes > 0, but as a fallback
                 basis_explanation = f"resource requirements ({', '.join(basis_parts)})"
            else:
                 basis_explanation = "no significant workload"


            node_determination_note = f"    (Node count determined by: {'High Availability minimum' if 'HA Minimum' in limiting_factors else basis_explanation}{', HA Minimum enforced' if 'HA Minimum Enforced' in limiting_factors else ''})"
            report.append(node_determination_note)


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
    # Updated help text to reflect combined data
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input cluster data file (JSON format, generated by collect_data.py) (default: {DEFAULT_INPUT})")

    # Optional arguments
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--format", choices=["json", "text"], default=DEFAULT_FORMAT, help=f"Output format (default: {DEFAULT_FORMAT})")
    # Corrected the typo here: DEFAULT_REDUNDancy -> DEFAULT_REDUNDANCY
    parser.add_argument("--redundancy", type=float, default=DEFAULT_REDUNDANCY, help=f"Redundancy factor (default: {DEFAULT_REDUNDANCY})")

    args = parser.parse_args()

    # Load cluster data (metrics and sizing)
    print(f"Loading cluster data from {args.input}...")
    # Use the updated function name
    cluster_data = load_cluster_data(args.input)

    # Generate recommendations - pass the full data structure
    print("Generating sizing recommendations based on observed workload demand...")
    recommendations = generate_recommendations(cluster_data, args.redundancy)

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
    original_sizing = recommendations.get("observed_original_cluster_sizing", {})

    print("\n--- Recommendation Summary ---")
    print(f"Workload Profile: {workload_profile.upper()}")
    print(f"Recommended ROSA Worker Configuration:")
    print(f"  Instance Type: {summary_config.get('instance_type', 'N/A')}")
    print(f"  Worker Node Count: {summary_config.get('worker_node_count', 'N/A')}")
    print(f"  Total Worker Capacity: {summary_config.get('total_cpu_cores', 0.0):.1f} vCPUs, {summary_config.get('total_memory_gb', 0.0):.1f} GB Memory")
    print(f"  Storage (PVs): {summary_config.get('total_storage_gb', 0):.0f} GB minimum")
    print("----------------------------")
    if original_sizing and original_sizing.get('total_nodes') is not None:
        print("Observed Original Cluster Sizing:")
        print(f"  Total Nodes (All Roles): {original_sizing.get('total_nodes', 'N/A')}")
        print(f"  Worker Nodes: {original_sizing.get('worker_node_count', 'N/A')}")
        original_capacity = original_sizing.get('total_capacity', {})
        print(f"  Total Worker Capacity: CPU: {original_capacity.get('cpu_cores', 0.0):.2f} cores, Memory: {original_capacity.get('memory_gb', 0.0):.2f} GB")
        print("----------------------------")
    else:
         print("Original cluster sizing data not available in input.")
         print("----------------------------")

    print(f"Full details saved to {args.output} ({args.format} format).")


if __name__ == "__main__":
    main()