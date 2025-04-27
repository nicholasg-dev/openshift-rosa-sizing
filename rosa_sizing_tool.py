#!/usr/bin/env python3
# =============================================================================
# OpenShift ROSA Sizing Tool (Unified Script)
# =============================================================================
#
# This script provides both metric collection (from Prometheus) and sizing
# recommendations for Red Hat OpenShift Service on AWS (ROSA) clusters.
#
# Usage:
#   python rosa_sizing_tool.py collect --prometheus-url ... --token ... [options]
#   python rosa_sizing_tool.py size --input metrics.json [options]
#
# Author: OpenShift Sizing Team
# =============================================================================

import argparse
import sys
import os
import json
import math
import time
import subprocess
import shutil
import glob
from datetime import datetime, timedelta
import requests
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===============================
# Constants for Sizing Calculations
# ===============================
DEFAULT_REDUNDANCY_FACTOR = 1.3
DEFAULT_MAX_PODS_PER_NODE = 250
MIN_WORKER_NODES = 2
CPU_BOUND_THRESHOLD_RATIO = 1.5
MEMORY_BOUND_THRESHOLD_RATIO = 0.67
DEFAULT_STORAGE_IOPS = 3000
HIGH_IOPS_THRESHOLD = 16000
DEFAULT_STORAGE_THROUGHPUT = 125  # MB/s
THROUGHPUT_IOPS_RATIO = 0.8
HOURS_PER_MONTH_AVG = 730  # Approximate
PERCENTILE_KEY = '95th_percentile'
PEAK_KEY = 'peak'
DIVISION_BY_ZERO_AVOIDANCE = 0.0001

INSTANCE_TYPES_FILE = 'instance_types.json'

# ===============================
# Utility Functions
# ===============================
def check_dependencies():
    # Requests is imported at top. Placeholder for future checks.
    return True

def check_openshift_login():
    try:
        result = subprocess.run(['oc', 'whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            print(f"Logged into OpenShift as: {result.stdout.strip()}")
            return True
        else:
            print("Warning: Not logged into OpenShift. Some features may not work correctly.")
            print("You can still proceed with the Prometheus URL and token.")
            print("To log in, run: oc login <cluster-url>")
            return False
    except Exception as e:
        print(f"Warning: Error checking OpenShift login status: {e}")
        print("You can still proceed with the Prometheus URL and token.")
        return False

def validate_positive_integer(value):
    try:
        val = int(value)
        if val <= 0:
            raise ValueError("Value must be a positive integer.")
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Please enter a positive integer.")

# ===============================
# Metric Collection Logic
# ===============================
def query_prometheus(args, query, start_time, end_time):
    headers = {"Authorization": f"Bearer {args.token}"}
    params = {
        "query": query,
        "start": int(start_time),
        "end": int(end_time),
        "step": args.step
    }
    try:
        resp = requests.get(
            f"{args.prometheus_url}/api/v1/query_range",
            headers=headers,
            params=params,
            verify=args.verify_ssl
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return None

def process_metric_data(data):
    if not data or 'data' not in data or 'result' not in data['data']:
        return {"average": 0, "peak": 0, "min": 0, "samples": 0}
    values = []
    for result in data['data']['result']:
        for v in result['values']:
            try:
                values.append(float(v[1]))
            except Exception:
                continue
    if not values:
        return {"average": 0, "peak": 0, "min": 0, "samples": 0}
    return {
        "average": sum(values) / len(values),
        "peak": max(values),
        "min": min(values),
        "samples": len(values)
    }

def collect_metrics(args):
    import shlex
    # 1. Check OpenShift login
    whoami = subprocess.run(['oc', 'whoami'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if whoami.returncode != 0:
        print('You are not logged into OpenShift. Please login:')
        cluster_url = input('OpenShift API URL (e.g. https://api.<cluster>.<domain>:6443): ').strip()
        username = input('OpenShift Username: ').strip()
        import getpass
        password = getpass.getpass('OpenShift Password: ')
        login_cmd = ['oc', 'login', cluster_url, '-u', username, '-p', password, '--insecure-skip-tls-verify']
        login = subprocess.run(login_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if login.returncode != 0:
            print('Login failed! Exiting.')
            print(login.stderr)
            sys.exit(1)
        print('Login successful.')
    else:
        print(f"Logged into OpenShift as: {whoami.stdout.strip()}")

    # 2. Create service account if missing
    sa_name = 'prometheus-client'
    sa_ns = 'openshift-monitoring'
    sa_check = subprocess.run(['oc', 'get', 'sa', sa_name, '-n', sa_ns], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if sa_check.returncode != 0:
        print('Creating service account for Prometheus access...')
        subprocess.run(['oc', 'create', 'sa', sa_name, '-n', sa_ns], check=True)
    else:
        print('Service account already exists.')

    # 3. Bind cluster-monitoring-view role
    rb_check = subprocess.run(['oc', 'adm', 'policy', 'who-can', 'get', 'pods', '-n', sa_ns], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Always (re)bind for idempotency
    subprocess.run(['oc', 'adm', 'policy', 'add-cluster-role-to-user', 'cluster-monitoring-view', f'-z', sa_name, '-n', sa_ns], check=True)
    print('Ensured service account has cluster-monitoring-view role.')

    # 4. Generate token
    print('Generating token for service account...')
    token_proc = subprocess.run(['oc', 'create', 'token', sa_name, '-n', sa_ns, '--duration=8760h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if token_proc.returncode != 0 or not token_proc.stdout.strip():
        print('Failed to generate token!')
        print(token_proc.stderr)
        sys.exit(1)
    token = token_proc.stdout.strip()
    args.token = token
    print('Token generated.')

    # 5. Get Prometheus route
    print('Discovering Prometheus route...')
    route_proc = subprocess.run(['oc', 'get', 'route', 'prometheus-k8s', '-n', sa_ns, '-o', "jsonpath={.spec.host}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if route_proc.returncode != 0 or not route_proc.stdout.strip():
        print('Failed to get Prometheus route!')
        print(route_proc.stderr)
        sys.exit(1)
    prometheus_host = route_proc.stdout.strip()
    args.prometheus_url = f'https://{prometheus_host}'
    print(f'Prometheus URL: {args.prometheus_url}')

    print(f"Collecting metrics for the past {args.days} days...")
    end_time = int(time.time())
    start_time = end_time - args.days * 86400
    metrics = {}
    queries = {
        "cpu_usage": "sum(node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate)",
        "memory_usage": "sum(container_memory_working_set_bytes{container!=\"\"})/1024/1024/1024",
        "pod_count": "count(kube_pod_info)",
        "pvc_storage": "sum(kubelet_volume_stats_used_bytes)/1024/1024/1024",
        "node_count": "count(kube_node_info)",
        "cpu_requests": "sum(kube_pod_container_resource_requests{resource=\"cpu\"})",
        "memory_requests": "sum(kube_pod_container_resource_requests{resource=\"memory\"})/1024/1024/1024",
        "namespace_count": "count(kube_namespace_created)"
    }
    for key, query in queries.items():
        data = query_prometheus(args, query, start_time, end_time)
        metrics[key] = process_metric_data(data)
        if metrics[key]["samples"] == 0:
            print(f"Warning: No data returned for {key}")
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {args.output}\n")
    print("Metrics Summary:")
    for k, v in metrics.items():
        print(f"{k}:\n  Average: {v['average']:.2f}\n  Peak: {v['peak']:.2f}\n  Min: {v['min']:.2f}\n  Samples: {v['samples']}")
    print("\nThese metrics will be used for ROSA cluster sizing calculations.")
    print("Run the 'size' subcommand next to generate sizing recommendations.")

# ===============================
# Sizing Calculation Logic
# ===============================
def load_instance_types(file_path):
    try:
        with open(file_path, 'r') as f:
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

INSTANCE_TYPES = load_instance_types(INSTANCE_TYPES_FILE)

def load_metrics(input_file):
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        sys.exit(1)

def bin_packing_simulation(metrics, instance_type, redundancy, max_pods_per_node=DEFAULT_MAX_PODS_PER_NODE):
    vcpu_per_node = INSTANCE_TYPES[instance_type]['vcpu']
    memory_per_node = INSTANCE_TYPES[instance_type]['memory_gb']
    cpu_95p = metrics['cpu_usage'].get(PERCENTILE_KEY, metrics['cpu_usage'][PEAK_KEY])
    mem_95p = metrics['memory_usage'].get(PERCENTILE_KEY, metrics['memory_usage'][PEAK_KEY])
    pod_95p = metrics['pod_count'].get(PERCENTILE_KEY, metrics['pod_count'][PEAK_KEY])
    required_cpu = cpu_95p * redundancy
    required_memory = mem_95p * redundancy
    required_pods = pod_95p * redundancy
    nodes_cpu = max(MIN_WORKER_NODES, math.ceil(required_cpu / vcpu_per_node))
    nodes_memory = max(MIN_WORKER_NODES, math.ceil(required_memory / memory_per_node))
    nodes_pods = max(MIN_WORKER_NODES, math.ceil(required_pods / max_pods_per_node))
    recommended_nodes = max(nodes_cpu, nodes_memory, nodes_pods)
    return recommended_nodes, nodes_cpu, nodes_memory, nodes_pods, max_pods_per_node, required_cpu, required_memory, required_pods

def calculate_worker_nodes(metrics, redundancy, profile='balanced', workload_type=None, exclude_bare_metal=True):
    recommendations = []
    for instance_type, specs in INSTANCE_TYPES.items():
        if exclude_bare_metal and specs.get('bare_metal', False):
            continue
        if workload_type == 'CPU-bound' and not instance_type.startswith('c'):
            continue
        if workload_type == 'Memory-bound' and not instance_type.startswith('r'):
            continue
        if workload_type == 'Balanced' and not instance_type.startswith('m'):
            continue
        recommended_nodes, nodes_cpu, nodes_memory, nodes_pods, max_pods_per_node, required_cpu, required_memory, required_pods = bin_packing_simulation(metrics, instance_type, redundancy)
        cpu_util = required_cpu / (recommended_nodes * specs['vcpu'])
        memory_util = required_memory / (recommended_nodes * specs['memory_gb'])
        rationale = (
            f"Node count is the maximum of (using 95th percentile):\n"
            f"- CPU: {nodes_cpu} nodes (CPU 95p: {required_cpu:.2f})\n"
            f"- Memory: {nodes_memory} nodes (Mem 95p: {required_memory:.2f} GB)\n"
            f"- Pod density: {nodes_pods} nodes (Pod 95p: {required_pods:.0f}, max pods per node: {max_pods_per_node})"
        )
        recommendations.append({
            'instance_type': instance_type,
            'node_count': recommended_nodes,
            'specs': specs,
            'utilization': {'cpu': cpu_util, 'memory': memory_util},
            'rationale': rationale,
            'estimated_cost': {
                'hourly': specs['hourly_cost'] * recommended_nodes,
                'monthly': specs['hourly_cost'] * recommended_nodes * HOURS_PER_MONTH_AVG,
                'instance_family': specs['family']
            }
        })
    recommendations.sort(key=lambda x: (x['estimated_cost']['hourly'], x['node_count']))
    return recommendations

def generate_recommendations(metrics, redundancy):
    profiles = ['cost', 'balanced', 'performance']
    all_recommendations = {}
    for profile in profiles:
        if profile == 'cost':
            recs = calculate_worker_nodes(metrics, redundancy, profile='balanced', workload_type=None)
            all_recommendations[profile] = recs[:1]  # Cheapest
        elif profile == 'balanced':
            recs = calculate_worker_nodes(metrics, redundancy, profile='balanced', workload_type=None)
            all_recommendations[profile] = recs[:1]  # Balanced
        elif profile == 'performance':
            recs = calculate_worker_nodes(metrics, redundancy, profile='balanced', workload_type=None)
            all_recommendations[profile] = recs[:1]  # Highest perf
    return all_recommendations

def format_text_output(recommendations):
    output = []
    for profile, recs in recommendations.items():
        output.append(f"Profile: {profile}")
        for rec in recs:
            output.append(f"  Instance Type: {rec['instance_type']}")
            output.append(f"  Node Count: {rec['node_count']}")
            output.append(f"  Specs: vCPU={rec['specs']['vcpu']}, Mem={rec['specs']['memory_gb']} GB, Bare Metal={rec['specs']['bare_metal']}")
            output.append(f"  Utilization: CPU={rec['utilization']['cpu']:.2%}, Mem={rec['utilization']['memory']:.2%}")
            output.append(f"  Estimated Cost: Hourly=${rec['estimated_cost']['hourly']:.2f}, Monthly=${rec['estimated_cost']['monthly']:.2f}")
            output.append(f"  Rationale: {rec['rationale']}")
            output.append("")
    return '\n'.join(output)

def size_main(args):
    metrics = load_metrics(args.input)
    recommendations = generate_recommendations(metrics, args.redundancy)
    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Recommendations saved to {args.output}")
    else:
        output = format_text_output(recommendations)
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Recommendations saved to {args.output}")

# ===============================
# Main CLI
# ===============================
def main():
    parser = argparse.ArgumentParser(description='OpenShift ROSA Sizing Tool (Unified)')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Collect subcommand
    collect_parser = subparsers.add_parser('collect', help='Collect metrics from Prometheus')
    collect_parser.add_argument('--prometheus-url', required=False, help='Prometheus base URL')
    collect_parser.add_argument('--token', required=False, help='Bearer token for Prometheus')
    collect_parser.add_argument('--output', default='cluster_metrics.json', help='Output file for metrics')
    collect_parser.add_argument('--days', type=validate_positive_integer, default=7, help='Days of data to collect')
    collect_parser.add_argument('--step', default='1h', help='Prometheus query step (e.g. 1h, 30m)')
    collect_parser.add_argument('--verify-ssl', action='store_true', help='Verify SSL certs (default: False)')
    collect_parser.set_defaults(func=collect_metrics)

    # Size subcommand
    size_parser = subparsers.add_parser('size', help='Generate sizing recommendations from metrics')
    size_parser.add_argument('--input', required=True, help='Input metrics JSON file')
    size_parser.add_argument('--output', default='rosa_sizing.json', help='Output file for sizing recommendations')
    size_parser.add_argument('--format', choices=['json', 'text'], default='json', help='Output format')
    size_parser.add_argument('--redundancy', type=float, default=DEFAULT_REDUNDANCY_FACTOR, help='Redundancy factor (e.g. 1.3)')
    size_parser.set_defaults(func=size_main)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
